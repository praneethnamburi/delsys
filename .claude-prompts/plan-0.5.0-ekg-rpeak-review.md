# Plan 0.5.0 — EKG R-peak review + persist/reload (curate-and-reload loop)

Design doc. Goal: let a human **review auto-detected R-peaks, correct them
(add/remove peaks, mark noisy segments, flip polarity, tag), and persist the
result to a sidecar** so that any future load — including loading the delsys
`.h5` Log file — reproduces the **curated final peaks** with no re-review and no
pickle.

Pain point (Praneeth, PE course, 2026-07-13): **the absence of a packaged
editor.** Detection itself works (`EKG.find_rpeaks_pn`); what never graduated out
of `pn-projects/projects/gibson/gib01.py` is the interactive edit-and-persist
loop. This plan graduates it.

Pulled into **0.5.0** (per Praneeth). Design-first; implement after sign-off.

## Reference implementation (already exists, project-local)

`gib01.EKGBrowser` (`gib01.py:849`, a `dnav.PlotBrowser` subclass) is the model:

- **3 panels** — raw EKG + peak markers (`*` default / `+` added / `x` removed)
  + noisy-segment ticks; IBI-over-time; IBI histogram.
- **Edit keys** — `a` add peak (snap to peak/valley/exact via an `m`-cycled edit
  mode, within a search window), `d` remove nearest, `n` mark noisy segment (two
  clicks), `f` flip polarity, `1/2/3` tag reviewed/representative/interesting,
  `t` tag-reviewed-and-advance, `g` reset review window, `s` save.
- **Model** — every edit writes into `sig.meta` using the *exact keys*
  `delsys.EKG._initialize_meta_for_rpeaks` already defines: `rpeaks_idx_added`,
  `rpeaks_idx_removed`, `noisy_segments_idx`, `is_flipped`, `tags`. **delsys
  already owns the data model.** What's missing is (a) the editor and (b) a clean
  persistence format. gib01's persistence is a `dill.dump` of whole `EKG`
  objects (the pickle fragility the `.h5` work exists to kill) and is
  **index-based** (breaks on resample/slice/reload).

## What we build on (shipped in 0.5.0a)

- `annotate.py` `_MarkingMixin` + `SignalAnnotator`/`SensorAnnotator` — datanav
  `SignalBrowser`/`PlotBrowser` subclasses that mark into `.delsys-events`.
  Datanavigator is a **lazy, optional import** (inside the factories). This
  already resolved the spec's held-open "EKG review UI — datanav dep vs
  from-scratch" decision in favour of *datanav-as-lazy-dep*.
- `clean_review.py` `Log.clean()` — the structural template for an
  **instance-attached** interactive reviewer that writes a **decision** sidecar.
- `_events.py` — the `.delsys-events` doc: `noise` track (per-signal
  `{windows, dead}`) + typed `marker` tracks, keyed by structural **address**
  `"<sensor>.<modality>[.<coord>] | <label>"`, with `read_events`/`write_events`,
  relabel-on-save, and `resolve_key` (modality-**general** — already addresses
  EKG: `SUBCHANNEL_MAP['EKG'] == ('A',)`, so `"5.EKG.A | Chest"` or coord-less
  `"5.EKG | Chest"`).
- `_clean.py` — `.delsys-artifact` **decision** sidecar
  (`{schema, cleaning: {config, accept, ...}}`); the `output = f(raw, decision)`
  reproducibility contract, incl. an `accept` review flag.

## Decision 1 — persistence format (the one Praneeth flagged)

**An R-peak review is a *derived decision*, not a raw annotation.** Its output is
`final_peaks = f(raw_ekg, detector_config, added, removed, noise, flip)` — the
same reproducibility contract as `.delsys-artifact`, not the "loose human marks"
contract of `.delsys-events`. Decompose what must persist, per EKG channel:

| Piece | Nature | `.delsys-events` today |
|---|---|---|
| noisy segments | time windows | **native** — this *is* the `noise` track |
| added peaks | point times | marker-track shaped |
| removed peaks | diff vs detector output | awkward — markers are "hand-placed, not algorithm-seeded" |
| `is_flipped` | per-channel bool | **no slot** |
| tags | per-channel labels | no clean slot (event `tags` are per-event) |
| detector config (highpass, hr_max) | scalars | **no slot** — needed to make `removed` reproducible |

Events cleanly holds 2/6, awkwardly 1/6, and has **no home** for 3/6.

**Recommendation: one file, split by nature — fold into `.delsys-events`, but as
a new first-class `rpeaks` *section* (a peer of `noise`, NOT a marker track),
with noisy-segments reusing the shared `noise` track.** Bump events schema 1→2.

```json
{
  "schema": 2,
  "events": {
    "noise":  {"kind": "noise",
               "signals": {"5.EKG.A | Chest": {"windows": [[12.1, 12.4]]}}},
    "rpeaks": {"kind": "rpeaks",
               "signals": {"5.EKG.A | Chest": {
                   "detector": {"name": "pn", "highpass": 5.0, "hr_max": 200.0},
                   "added":   [1.402, 2.101],
                   "removed": [3.550],
                   "flipped": false,
                   "tags":    ["reviewed"]
               }}}
  }
}
```

Why this over the two rejected alternatives:

- **vs. a dedicated `.delsys-rpeaks` file** — cleaner in isolation, but a
  noisy-segment is a *wall-clock quality span*; it belongs in the shared `noise`
  track so both the EMG cleaner (`delsys.clean`) and the EKG reader see it. A
  separate file forces either duplication or a coordination loss (EKG noise
  invisible to EMG cleaning). Because `.delsys-events` noise is **per-address**,
  an EKG-scoped noise window (`5.EKG.A`) masks only the EKG channel unless the
  user marks it whole-sensor — so sharing the file does **not** force
  cross-modality masking; the address controls scope. That kills the only real
  reason to separate. It would also add a 4th sidecar type
  (`.delsys-noise` legacy / `.delsys-events` / `.delsys-artifact` / `.delsys-rpeaks`).
- **vs. abusing marker tracks** (`rpeak_added` / `rpeak_removed` markers) — zero
  schema change, but "removed" has no meaning in the additive marker model, and
  flip/tags/detector still need a non-marker home. Once you need that home,
  putting *all* rpeak state in it is cleaner than scattering it. So a dedicated
  `rpeaks` section beats two marker tracks + an orphan scalar blob.

Cost: a modest `_events.py` extension — `read_events`/`write_events` learn a
third `kind` (`rpeaks`) alongside `noise`/`marker`; typed accessors
`read_rpeaks_signals(path)` / fold into `write_events`. Reuses all the
address/relabel/`resolve_key` machinery unchanged.

**Times, not indices.** `added`/`removed` and noise windows are **seconds on the
Log clock**. This is mandatory: the `.h5` reloads at *native* rate (a different
sample grid than the reviewed one) and slicing changes the grid too. Storing
times and re-resolving to indices against the current grid on load also fixes
the standing `EKG.__getitem__` "slicing breaks the rpeak cache" TODO — same root
cause, one mechanism. `EKG.meta`'s `rpeaks_idx_*` stay as the in-memory cache;
the sidecar is the durable, grid-independent source of truth.

*(Open sub-question for sign-off: keep the legacy per-signal `.delsys-noise`
migration bridge working for the EKG noise too — yes, via the existing
`noise_signals_for` fallback; no new bridge needed.)*

## Decision 2 — reviewer attached to the EKG instance (per Praneeth)

Entry point on the **`EKG` class**, not `Log`:

```python
ekg = lf.ekg
peaks = ekg.review()          # opens the editor; on close, curated peaks are live
```

- **Graduate `EKGBrowser` into `delsys`** (e.g. `delsys.rpeak_review` module,
  lazy-datanav like `annotate`/`clean_review`). Port the 3-panel layout +
  keybindings + edit-mode state var almost verbatim; the edit model is already
  `EKG.meta`. `EKG.review(path=None, **detector_kwargs)` is a thin launcher.
- **Provenance stamp.** An `EKG` today knows its `meta['sensor']` but not its
  source file. `lf.ekg` (and `Sensor.ekg`) must stamp the bundle with its
  **source path** (`lf.fname`) and structural **address**, so `review()` and the
  auto-loader know the sidecar path (`_events.events_path_for(source)`) and which
  `signals` key to read/write — no args needed. `path=` overrides. (gib01 did
  this ad-hoc via `meta['sav_name']`/`meta['id']`; we formalize it.)
- **Save** writes noise → `noise` track and the rpeak decision → `rpeaks` section
  of the **same** `.delsys-events` at the source stem, relabelled on the way out
  (reuse `_MarkingMixin.save`'s relabel pattern).
- **Multi-channel** (`lf.ekg` with ≥2 EKG channels): the reviewer steps channels
  via a dropdown (PlotBrowser-style), each channel its own address/section entry.
  Single channel is the common case and opens directly.
- **Flip polarity = a button, and it re-detects.** The reviewer exposes an
  explicit **Flip** button (keep the `f` key too). Pressing it calls the existing
  `EKG.flip_signal()`, which toggles `is_flipped` **and re-runs `find_rpeaks`** on
  the negated signal, then redraws. The `flipped` state is written into the
  `rpeaks` section and reproduced on auto-load (a flipped-and-curated trial
  reloads flipped, with its added/removed diff applied on top of the re-detected,
  flipped baseline). This is the one edit that changes the *baseline* rather than
  just diffing it, so it must re-detect, not merely negate the display.
- **One detector, no detector box.** The reviewer runs the single `pn` detector
  (`find_rpeaks_pn`), exactly like gib01. There is **no UI widget to pick a
  detector**; to review with different params, pass them to
  `ekg.review(highpass=..., hr_max=...)`. The stored `detector` block is
  **provenance only** (see Decision 1 clarification below) — it does not drive
  any UI.

### Explicitly NOT in this feature (per Praneeth, 2026-07-13)

- **No detector-selection UI.** No dropdown/box to switch detectors in the
  reviewer.
- **No multi-detector "layers".** We are **not** building "show me the peaks from
  `pn`, now overlay the peaks from `iw`" comparison/layering. There is one
  baseline (`default`), the human curates it, done. The `detector.name` field
  merely leaves the *door* open for alternate back-ends later; the format *could*
  later hold more than one detector entry per address if layering is ever wanted,
  but that is out of scope here and no code assumes it.

## Decision 3 — auto-load on Log load (per Praneeth)

"Future loads (including loading the delsys Log file) grab the sidecar if it
exists." New behavior — nothing auto-applies sidecars on load today
(`delsys.clean` consumes them explicitly).

- **Lazy, at EKG materialization** — when `lf.ekg` builds the aggregate (or on
  first `find_rpeaks()`/`rpeak_times()`), if a sibling `.delsys-events` has an
  `rpeaks`/`noise` entry for the channel's address: re-run the detector with the
  **stored `detector` config** to reproduce `default`, apply `added`/`removed`
  (by nearest-time within tolerance), set `is_flipped`/`tags`, resolve noise
  windows → `noisy_segments_idx`. Result: `lf.ekg.rpeak_times()` returns curated
  final peaks with zero extra calls.
- **Must stay lazy** — do NOT run detection at `Log(".h5")` construction; that
  would defeat the header-only cheap-load property (`{id: Log(h5)}` dicts). Apply
  only when the EKG signals are actually materialized.
- **Escape hatch** — `apply_rpeaks=False` (or a `.ekg_raw` accessor) to get the
  un-curated auto-detection, for re-review or debugging.

Reproducibility parallel to `.delsys-artifact`: storing `detector` freezes the
decision. If delsys later changes the default `highpass`/`hr_max`, a curated
trial still reproduces its exact `default` set (and therefore which peaks
`removed` refers to), instead of silently drifting. **The `detector` block is
provenance/reproducibility metadata only** — not a UI control and not a layer
system (see "Explicitly NOT in this feature"). For 0.5.0 the only value of
`detector.name` is `"pn"`; on reload we re-run `find_rpeaks_pn` with the stored
params to reproduce `default`. An unrecognized `name` (a future back-end not
present in the reader) is a clear error, not a silent fallback.

## 0.5.0 scope

**In:** `rpeaks` section in `_events.py` (schema→2) + typed accessors;
`delsys.rpeak_review` editor (graduated `EKGBrowser`, lazy-datanav); EKG
provenance stamp; time-based sidecar (fixes the slicing/reload TODO);
lazy auto-load on EKG materialization + `apply_rpeaks=False`; a
`tutorials/rpeaks_review.md` walkthrough (detect → `ekg.review()` → save →
reload); tests (round-trip, native-reload grid-independence, slice
correctness, auto-load).

**Out / follow-up:** batch `delsys.extract_hr(source)` (mirror `process`/`clean`,
per-trial HR/IBI sidecar + report) — natural 0.5.1; **multi-detector back-ends,
a detector-selection UI, and multi-detector layer/comparison** (deferred per
Praneeth — the `detector.name` field is the only forward hook we take now);
quality-confidence flag; graduating `HRVMetrics` from gib01 into delsys.

**Coordination with the already-planned 0.5.0 items** (drop `delsys.cleaning`
aliases; migrate `_aggregate_bundles` → `pysampled.Data.merge_along_signal_name`
— the latter still gated on pysampled 1.3.0). This feature is independent of
both; sequence the release cut after it lands.

## Resolved (Praneeth, 2026-07-13)

1. **Format** — `rpeaks` as a new events `kind` inside `.delsys-events` (peer of
   `noise`); noisy segments reuse the shared `noise` track. **Not** a separate
   `.delsys-rpeaks` file. ✔
2. **Multi-channel `review()`** — step channels in one browser (dropdown). ✔
3. **Auto-load trigger** — at `lf.ekg` build, with an `apply_rpeaks=False`
   escape hatch; stays lazy (never at `Log(".h5")` construction). ✔
4. **Times, always** — `added`/`removed`/noise stored as seconds on the Log
   clock; re-resolved to indices per current grid on load. ✔
5. **Detector block = provenance only** — store `{"name": "pn", "highpass",
   "hr_max"}` for reproducibility. **No detector-selection UI, no multi-detector
   layering/comparison** (deferred). Params for a review come from `review()`
   kwargs, not a widget. ✔

No open questions remain — ready to implement on `0.5.0a2` on sign-off.

## Test plan

- Round-trip: review edits → save → `read_rpeaks_signals` → identical decision.
- **Grid independence**: review on a CSV-loaded EKG, save, reload the same trial
  as native-rate `.h5`, assert curated `rpeak_times()` match within one sample.
- Slice correctness: `ekg[a:b].rpeak_times()` returns peaks in the window at
  correct positions (closes the `__getitem__` TODO).
- Auto-load: `Log(h5)` with a sibling `.delsys-events` yields curated peaks;
  `apply_rpeaks=False` yields raw auto-detection.
- Noise sharing: an EKG-addressed noise window is visible to both
  `ekg.rpeak_times()` (drops peaks inside) and `delsys.clean` (masks the EKG
  channel only, not EMG).
- Detector-freeze: changing the package default `highpass` does not change a
  curated trial's reproduced `default` set.
