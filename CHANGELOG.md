# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-14

### Added

- **`EKG.review()` — interactive R-peak reviewer + persist/reload
  (`delsys.rpeak_review`).** The curate-and-reload loop that graduates the bespoke
  `EKGBrowser` from `pn-projects/projects/gibson/gib01.py` into the package. Opens
  a `datanavigator` `PlotBrowser` (imported lazily) over the EKG's channel(s) —
  raw trace + default/added/removed peak markers + noisy-segment shading, an
  inter-beat-interval trace, and an IBI histogram — with:
  - **a** add a peak at the cursor (snapped by an *edit mode* — `peak` / `valley`
    / `exact`, cycled by **m**), **d** remove the nearest, **n** mark a noisy
    segment (two presses), **1**/**2**/**3** tag `reviewed` / `representative` /
    `interesting`. Add and remove are **inverses**: adding at a peak you removed
    *restores* it (drops it from the removed set) instead of stacking a duplicate,
    and removing a peak you added *undoes* the addition.
  - a **Flip** button (and **f**) that flips polarity **and re-detects** — the one
    edit that changes the detector baseline, so do it first, before add/remove.
  - a **Save** button (and **s**) that writes each channel's decision to the
    sibling `<stem>.delsys-events` sidecar; the raw/IBI zoom persists across edits.
  - a **Help** button (and **ctrl+k**) opens datanavigator's grouped key-binding
    cheatsheet, and a compact shortcut legend is drawn on the figure.
  - A multi-channel (aggregate) EKG is split and stepped via the channel dropdown.
- **Persisted, grid-independent R-peak curation (`rpeaks` type in
  `<stem>.delsys-events`).** R-peak review is a reproducible *decision*, not a raw
  annotation, so it lands as a new reserved type in the unified sidecar (peer of
  `noise`; noisy segments reuse the shared `noise` track). Per channel it records
  `{detector, added, removed, flipped, tags}` — peak **times, never indices**, so
  the decision reproduces on any sample grid (native-rate `.h5` reload, a slice).
  `final_peaks = f(raw_ekg, decision)`.
  - `EKG.rpeaks_decision()` serializes the current curation; `apply_rpeaks_decision()`
    reproduces it on this grid by re-running the recorded detector (honouring
    `flipped`) then applying the human diff (`added` → nearest sample, `removed`
    → nearest default peak). Only *human* removals are persisted (a new
    `rpeaks_idx_autopruned` meta key separates the detector's own double-peak
    prune) — an auto-pruned double sits ~1 beat-width from a real peak, so
    persisting it would kill that peak on a grid where the double never appears.
  - `EKG.load_rpeaks()` / `save_rpeaks()` (path defaults to the sibling of
    `meta["source"]`), `delsys._events.read_rpeaks_signals(path)`, and the
    addressing + file glue in `delsys._rpeaks`.
- **`Log.ekg` auto-applies a saved decision; `Log.ekg_raw` bypasses it.** On a
  single-channel EKG, if a sibling `.delsys-events` carries an `rpeaks` decision
  for the channel, `lf.ekg.rpeak_times()` returns the *curated final* peaks with
  no extra step (a malformed sidecar warns rather than breaking signal access; no
  sidecar is a fast no-op). `lf.ekg_raw` gives the un-curated bundle — the
  cleaner's ECG reference uses it, so cleaning never pays for auto-load. The bundle
  carries its source in `meta["source"]`; keyed by stem, a CSV and its native `.h5`
  share one sidecar.
- **Public metadata + dispatch helpers.** `delsys.duration(fname)` reads a
  trial's length in seconds straight from the CSV header (no `Log` construction),
  raising a clear `ValueError` for EMGworks exports, which don't record duration.
  `delsys.mod_to_attr` / `delsys.modset_to_strlist` expose the modality-dispatch
  helpers used to map a modality tag to its `Sensor` attribute. These were
  previously reached as private `immersionlab.delsys._parse_hdr` /
  `_mod_to_attr` / `_modset_to_strlist`; the package-extraction shim
  (`from delsys import *`) only re-exported `__all__`, so those cross-package
  callers broke — they now have stable public names.
- **HDF5 native checkpoint.** `delsys.to_native_h5(csv, out_h5)` converts a
  Trigno Discover CSV into a self-contained, native-rate HDF5 checkpoint, and
  `Log(path, target_sr=..., clock_mul=..., t0=...)` reloads it (the constructor
  now dispatches on a `.h5`/`.hdf5` suffix). `Log.to_hdf5(path)` writes the
  current `Log` directly. Design:
  - **Native storage, resample-on-load.** Trigno-Base modalities are stored at
    the acquisition rate (`target_sr=None`) and re-resampled when read; async
    link devices (VO2/HR/SmO2/Thb) are written as terminal snapshots. `clock_mul`
    and `t0` are *not* baked in — alignment to another clock is a load-time
    argument, so one checkpoint serves any target rate / clock without re-export.
  - **Lazy load.** `Log(".h5")` reads only the header (root attrs, channelmap,
    per-sensor metadata) at construction — cheap enough that `{id: Log(...)}`
    dicts over a whole study cost almost nothing. The signal datasets are read +
    resampled on first access to `sensors` / `signals` / `sr_orig` (and therefore
    the bundle properties / `find` / `__getitem__`), via `Log.__getattr__`.
  - **`float32` + `lzf`**, lossless vs the CSV's ~7 significant figures; the
    reader upcasts to `float64` before any resampling, so no downstream numeric
    work (filters, ICA) runs in single precision. Round-trips bitwise (within
    float32) against `Log(csv)` across all Discover fixtures incl. link devices;
    on real recordings the `.h5` is ~8-9x smaller than the pickle and the
    self-contained checkpoint makes the source CSV disposable.
  - The embedded channelmap + canonical sensor order keep multi-sensor bundle
    columns in their `Log(csv)` order.
  - Adds an `h5py` dependency.
- **`delsys.process(source)`** — batch CSV → `.h5` converter (mirrors
  `telemed.process`): walks a path/folder/iterable, converts each CSV
  idempotently (`skip_existing` / `overwrite`), and returns a `{path: status}`
  dict (`"built"` / `"hit"` / `"skipped: ..."` / `"error: ..."`), writing a
  per-folder `delsys_process_report.txt`. **Smart channelmap resolution**:
  candidates are found in the CSV folder *and* parent folders
  (`channelmap_search_parents`, default 1); the filename hint
  (`delsys_channelmap_Trial_A_B.txt` for a trial range, else the default) is
  cross-checked against the CSV's actual channel-number set, and under the
  default `channelmap_policy="strict"` a name-pick that doesn't match is flagged
  and skipped (vs `"lenient"` = pick the content-matching map + warn, or
  `"name_only"`). `progress=True` (default) prints a triage line, a per-file
  `tqdm` progress bar, and a `built/hit/skipped/error` summary, surfacing
  skips/errors as they occur. Adds a `tqdm` dependency.
- **`delsys.read_channelmap(path)`** → `(sensor_map, sensor_name_replace)`.
  Channelmaps may now carry an optional `[sensor_name_replace]` section
  (`old = new` lines) holding acquisition-typo corrections alongside the map, so
  a single sidecar is self-describing; older readers ignore the section.
  - **EMGworks** is supported alongside Discover. Its time window depends on the
    requested `target_sr` (Discover's is `(0, duration)`), so the checkpoint stores
    the widest (`min_sr=1`) window plus the raw extent and *trims* on reload to
    reproduce `Log(csv, target_sr)` bitwise for any target. Reload must use
    `clock_mul=1` (the native interpolation grid is fixed at export); a
    clock-shifted EMGworks reload raises `NotImplementedError`. Discover has no such
    restriction.
- **`delsys.clean(source)`** — batch EMG/EKG-artifact cleaning of native `.h5`
  checkpoints (mirrors `delsys.process`): walks a path/folder/iterable for raw
  `Trial_*.h5`, runs `Log.clean_emg_ekg_artifact` on each, and writes a
  terminal-snapshot `<stem>_cleaned.h5` plus a `<stem>_cleaning_report.pdf` next
  to the checkpoint. Idempotent (`skip_existing` / `overwrite`), `tqdm` progress,
  per-folder `delsys_cleaning_report.txt`, returns a `{raw_h5: status}` dict
  (`"cleaned"` / `"hit"` / `"skipped: ..."` / `"error: ..."`). The raw checkpoint
  is immutable; its own `*_cleaned.h5` outputs are skipped on re-walk. A fresh
  `Log` is loaded per trial, so the in-place cleaning never touches a Log a caller
  holds.
  - **Per-log decision sidecar** (`<stem>.delsys-artifact`, sibling to the
    checkpoint — the same per-log, file-centric model as `<stem>.delsys-noise`)
    makes the cleaning reproducible: `cleaned.h5 = f(raw.h5, <stem>.delsys-artifact)`.
    Each sidecar records the ICA components removed, the spliced variant
    (`splice_source`), the motion pairing, an optional `noise_event_ref`, an
    `accept` review flag, and the rest of the `CleaningConfig` knobs. Marking a
    trial's `accept` `false` (after eyeballing its PDF) blocks regeneration even
    under `overwrite=True` until the decision is fixed and the flag flipped back.
    A trial with no sidecar is cleaned with auto-detection and its resolved
    decision is *frozen* into a fresh sidecar; a later pass replays it
    (auto-detection off, recorded components applied verbatim). Because the FastICA
    fit is seeded, a re-run reproduces the cleaned checkpoint bit-for-bit — verified
    on the pia02 sandbox (a forced replay of a ~5.9M-sample cleaned trial is
    byte-identical). The `config` argument is only the base for trials *without* a
    sidecar; it never overrides an existing decision. Edit the sidecar (swap the
    auto-chosen IC, change `splice_source`, attach a noise event), or use the
    interactive `Log.clean()`, and re-run with `overwrite=True` to
    regenerate just the touched trials. `delsys.clean(..., record_decisions=False)`
    cleans with the call's defaults and reads/writes no sidecars; a per-folder
    `delsys_cleaning_report.txt` summarizes each run (an overview, not the source
    of truth).
- **Noise-window consumption** (`delsys._noise`) — reads human-authored noise
  Events (marked in `datanavigator`'s `SignalBrowser`) as **plain JSON**, with no
  `datanavigator` dependency. `read_noise_intervals(path, trial_id)` parses the
  `[metadata, data]` Event file (effective windows = `default + added` minus
  `removed`; a tuple `trial_id` is stringified to match the on-disk
  `"(2, 14, 17)"` key). `apply_noise_mask(lf, intervals)` blanks each window to
  `NaN` and interpolates it back (`policy="nan_interp"`), modality-agnostically
  (a noise window is a wall-clock span — EMG, ACC, … all get the same treatment),
  rebuilding the per-sensor bundles so the change is visible to the cleaner.
  `delsys.clean` wires this in: a manifest entry's `noise_event_ref` (`{path,
  key}`) is applied before the artifact cleaner runs, and the per-folder report
  notes `noise_masked=N`. The v1 surface covers the batch hook; per-modality
  window scoping and alternative fill policies are follow-ups (see `TODO.md`).
- **Per-signal noise sidecar** (`<stem>.delsys-noise`, in `delsys._noise`) — a
  delsys-file-centric noise record that travels next to one `Trial_N.h5`, keyed
  by a structural *signal address* (vs the trial-id-keyed, flat-interval
  datanavigator Event path above). Composite suffix (not `.json`) so portfolio
  `*.json` tooling skips it.
  - **Key grammar** `"<sensor>.<modality>[.<coord>] | <label>"` —
    `parse_key` / `format_key` / `format_signal_key` / `resolve_key`. The address
    left of `" | "` is authoritative (`<modality>` ∈ EMGS/EMGD/EMGQ/ACC/GYRO/FSR;
    `<coord>` from each modality's `SUBCHANNEL_MAP`); omitting `<coord>` fans out
    to every sub-channel of that sensor+modality. The label is informational and
    ignored on resolve, reusing the sensor/modality/sub-channel matching of
    `Log._splice_emg_back`.
  - **Per-signal masking** — `apply_noise_mask` is refactored onto a shared
    `_mask_signals` core (its flat-interval, modality-agnostic behavior is
    unchanged). New `apply_noise_sidecar(lf, path)` resolves each key to columns
    and applies, per signal, two span kinds: `windows` (transient noise → NaN +
    interpolate) and `dead` (no usable signal → **zero-fill**, applied after
    interpolation so it wins overlaps). Spans accept `null` endpoints — `[T, null]`
    = dead from `T` onward, `[null, null]` (or `"dead": true`) = the whole extent
    — so a sensor that dies mid-recording and one dead from the start use the same
    field. JSON body `{"schema", "signals": {key: {"windows", "dead"}}}` via
    `read_noise_sidecar` / `write_noise_sidecar` (a bare list value is windows-only
    shorthand).
  - `delsys.clean` defaults a trial's `noise_event_ref` to a sibling annotation
    sidecar when present — the unified `<stem>.delsys-events` (its `noise` type)
    in preference to a legacy `<stem>.delsys-noise` — consumed before the cleaner;
    the resolved basename is frozen into the decision sidecar as provenance.
    Consumption dispatches by suffix (`_apply_noise_ref`), so existing
    `<stem>.delsys-noise` and datanavigator-Event refs still work. The
    `_apply_noise_signal_map` core is shared by the legacy and unified paths.
- **Unified annotation sidecar** (`<stem>.delsys-events`, in `delsys._events`) —
  one per-log file holding **every** human annotation, split by *event type*: the
  built-in `noise` quality track (windows + dead, the `_noise` grammar above) plus
  typed **marker** tracks (`"1"`, `"2"`, …) — point (`size=1`) or window
  (`size=2`) events authored **per signal** (the address is provenance) and read
  back as **trial-level** markers via `collapse_markers` (keep-all + provenance by
  default; optional proximity dedupe). JSON `{"schema", "events": {type: …}}` via
  `read_events` / `write_events`; `noise_signals_for` + `migrate_noise_sidecar`
  bridge a legacy `<stem>.delsys-noise` (read when no unified file is present;
  folded in on next save). `apply_events_noise(lf, path)` masks off the `noise`
  type. datanavigator's per-`Event` save is bypassed in favour of this one
  unified write — it stays only the in-memory marking engine.
  - **Per-event note/tags (format-ready).** A marker event serializes as a bare
    `[t, ...]` or, when annotated, the object form `{"seq": [...], "note": ...,
    "tags": [...]}`; `read_marker_records` / `collapse_markers` surface them and the
    annotator round-trips them. (Authoring UI is a follow-up.)
- **Per-project config + named event types** (`delsys_project.toml`, in
  `delsys._project` / `delsys._event_types`) — a project's delsys settings live in a
  TOML committed in its repo; today it holds the **event-type vocabulary**, later
  things like `target_sr`. Resolution: `DELSYS_PROJECT_CONFIG` env → walk up from
  the trial folder for a `delsys_project.toml` → built-in defaults. Types are
  **slug-keyed** (`slug` written into `.delsys-events`; `label` is what a rename
  edits, so rename needs no file migration) with a bound `key`, `size`, and
  `color` (`EventType`). `Log.view()` loads its marker tracks from the resolved
  config (an explicit `events=` still overrides); `_project.scaffold(path)` writes a
  starter config. Reads/writes via `tomlkit`, so an interactive event-type edit is
  **surgical** — it preserves comments and the rest of the document. Adds a
  `tomlkit` dependency.
- **`Log.view(kind=..., events=...)`** — interactive annotator (`delsys.annotate`;
  `datanavigator` imported lazily so the delsys core stays dnav-free). Marks two
  kinds of annotation into one `<stem>.delsys-events` sidecar (shared logic via
  `_MarkingMixin`), with a single **Save** button (auto-seeded from an existing
  unified file, or a legacy `<stem>.delsys-noise`, on open):
  - **noise** — hover and press **`n`** (two presses fix a window), **`alt+n`** to
    remove the nearest, **`d`** to toggle the address dead. A per-signal quality
    mask consumed by `delsys.clean`.
  - **typed markers** — point / window events of arbitrary type added with the
    **digit keys** (`1` → a `"1"`-event, `2` → a `"2"`-event, …), removed with
    `alt+<digit>`; the tracks are configurable via `events=` (default: a point
    `"1"` + a window `"2"`). Authored per signal, collapsed to trial-level on read.

  Marks are shaded/lined on the trace; annotations index by structural **address**
  (label-free), so a sidecar from a previous session renders even if its keys carry
  a different/older `| <label>` (saving re-attaches the current label).
  `Log.annotate_noise(view=...)` is kept as a **deprecated alias** for `view()`.
  - **`kind="signal"`** (default) — `SignalBrowser` subclass: flip through every
    channel via the dropdown; a **Mod scope** toggle records *noise* against the
    channel (coord-ful) or whole sensor+modality (coord-less); markers always record
    the coord-ful channel; **Toggle dead** / **Undo window** buttons.
  - **`kind="sensor"`** — `PlotBrowser` subclass: one sensor's modalities as
    stacked, time-aligned subplots, picked via the dropdown. **EMGQ / FSR / Analog
    get one subplot per sub-channel** (so each Quattro channel, FSR pad, or Sync
    line is individually markable — and a Sync carrying a single line shows one
    panel), while EMGS/EKG (single trace) and ACC/GYRO (X/Y/Z overlaid) stay as one
    whole-modality subplot; a sensor mixing EMGQ with ACC/GYRO shows them all.
    Marking targets the hovered subplot's address; a **Sensor scope** toggle instead
    fans a *noise* mark across every modality of the sensor (a wall-clock burst hits
    them all). Built for blips shared across a sensor's channels.
- **`Log.clean()`** — interactive ECG/ICA cleaning tool (`delsys.clean_review`;
  `datanavigator` imported lazily). The single-log interactive counterpart to the
  batch `delsys.clean()`, and the unified successor to the old read-only
  `CleaningResult.review` / `review_components` viewers. One window, two regions: an
  **all-components bar** of each IC's EKG correlation (**click a bar — or `j`/`k` —
  to inspect that IC**; `1` / the button toggles its removal, red = removed) with the
  inspected IC's detail below it, and a **channel reviewer** (raw vs the chosen
  cleaned variant + PSD; step channels with the arrow keys / `channel` dropdown). The
  three time-domain panels share one x-axis whose zoom persists across redraws
  (compare ICs/channels at a fixed window; **Auto limits** resets), and each panel's
  y rescales to the data in the visible x-window. A single
  **Motion** auto/off toggle and a **splice** selector drive the rest. **Save
  decision** writes
  the explicit decision (removal set + splice + motion) to the sibling
  `<stem>.delsys-artifact` and clears the stale `*_cleaned.h5`, so the next
  `delsys.clean()` reproduces exactly what was previewed. (For headless/programmatic
  single-log cleaning use `clean_emg_ekg_artifact`.)
- **`delsys.CleaningSession`** — the headless fit-once core behind `Log.clean()`
  (`CleaningSession.from_log(lf, config=...)`). Fits FastICA once, then
  `.recompute(components_to_remove, motion=..., motiononly=...)` re-derives a full
  `CleaningResult` for any removal set / motion pairing *without refitting*
  (component removal is a reconstruction; the EKG + ACC regressions are cheap linear
  solves) — cheap enough to drive the live picker. The resampled ACC predictors are
  cached per pairing (a component toggle never re-resamples), and `motiononly=False`
  skips the second motion pass when the preview isn't showing that variant.
  `.auto_components()` gives the auto-detected default set.
- **`delsys._clean.upsert_decision(checkpoint, …)`** — write/replace a checkpoint's
  `<stem>.delsys-artifact` decision sidecar (the reviewer's Save); optionally clears
  the stale `*_cleaned.h5`. Read/write the sidecar directly via
  `delsys._clean.read_decision` / `write_decision`.
- `tutorials/workflow.md` — end-to-end walkthrough (`process` → `.h5` →
  `clean` → `*_cleaned.h5` → analysis), covering the decision-sidecar edit/re-run
  loop; section 4 now leads with `lf.view()` + the unified `.delsys-events` sidecar
  (legacy `.delsys-noise` + the datanavigator-Event path kept as fallbacks).

### Changed

- **Default `TARGET_SR` is now native (preserve acquisition rate) for
  Trigno-base modalities** (EMGS / EMGD / EMGQ / ACC / GYRO / FSR / EKG / Analog).
  A bare `delsys.Log(csv)` (or `Log(h5)`) no longer resamples these — it returns
  each modality at its native acquisition rate, with no anti-alias filter
  (previously it resampled to fixed rates, e.g. Analog→2400 Hz, EMG→1920 Hz).
  **This is a behavior change**: downstream code that relied on the old uniform
  rates should pass an explicit `target_sr=` to restore per-modality resampling.
  Motivation: native rate preserves event timing that resample-on-load smears
  (e.g. an analog gate's rising edge). Link devices (VO2 / HR / SmO2 / Thb) are
  asynchronous and stay resampled. A consequence is that `lf.emg` can now span
  multiple native rates across sensors (e.g. Trigno-base at 1259 Hz + Quattro at
  2222 Hz); `lf.<modality>` downsamples to the lowest present rate with a
  `UserWarning`, and `Log.clean_emg_ekg_artifact` harmonizes to one rate and
  commits the cleaned EMG at that rate.
- **Bundle `signal_names` keep the full body-location name.** `_trim_location`
  now keeps everything *before* the `(...)` parenthetical (dropping only the
  parenthetical — the FSR/Quattro position map or an EMG alt-name) instead of
  only the first whitespace token. Single-word camelCase locations (e.g.
  `LBrachialis`) are unchanged; multi-word/spaced placements (plausible in
  bilateral layouts) are now preserved (`"L Big Toe"` → `LBigToe`, previously
  `L`). The leading `L`/`R`/`C` side marker was always part of the location and
  is still kept. `delsys._noise.format_signal_key` builds its sidecar `<label>`
  from the same per-channel bundle name (`Sensor._make_bundle_labels`), so an
  annotator key reads e.g. `9.FSR.C | LFoot_Ball`.
- `_parse_dataframe_emgworks` now accepts per-modality `target_sr=None`
  (preserve-native), needed by the HDF5 checkpoint's native export. The
  channel-grid window still snaps to the coarsest *requested* rate exactly as
  before (existing `Log(csv, target_sr)` output is unchanged, verified bitwise);
  `None` entries are simply excluded from that snap instead of crashing it.

### Removed

- **`CleaningResult.review()` and `CleaningResult.review_components()`** (and their
  internal viewers). The two read-only matplotlib viewers are replaced by the single
  interactive `Log.clean()` window above, which folds in their per-channel
  raw-vs-cleaned and per-IC inspection panels and adds the act-on-it loop (toggle →
  re-clean → save). The batch `CleaningResult.generate_report()` PDF is unchanged.

## [0.4.1] - 2026-05-10

### Fixed

- `Log.modalities`, `Log.locations`, and `Log.modality_sensors` no
  longer crash on `Log` objects loaded from very-old pickles where
  per-:class:`Signal` ``meta`` is empty. All three now derive from
  :attr:`Log.sensors` — repaired by :meth:`Sensor.__setstate__` — so
  they resolve on the same pickles that the 0.4.0
  `clean_emg_ekg_artifact()` fix already handled. As a side effect
  `Log.locations` (previously raising `AttributeError` because
  ``Signal`` has no ``.location`` property — broken on every load,
  not just stale pickles) now returns the set of
  ``sensor.location`` values, and `Log.modality_sensors` returns
  ``{modality: {sensor.name, ...}}`` instead of the documentation-
  contradicting ``{modality: {modality_as_attr_name}}`` it produced
  before. No internal callers depended on the old behavior.

## [0.4.0] - 2026-05-10

Headline feature: a `Log`-integrated EMG/EKG artifact cleaning
pipeline. Ports the multi-stage cleaner from
`pn-projects/projects/emg_ica_cleaning.py` (preprocess → ICA-based
ECG suppression → ACC-guided motion regression with safety gates)
into the package, with a clean splice-back into `lf.signals` /
`lf.sensors[*].emg`.

### Added

- `Log.clean_emg_ekg_artifact(*, config, motion, in_place, generate_report, splice_source)` —
  end-to-end pipeline running on every EMG channel in the Log. By
  default mutates `lf.signals` in place, rebuilds the affected
  `Sensor.emg` bundles, and writes a multi-page PDF report next to
  the source CSV. Pass `in_place=False` to inspect diagnostics
  without mutating, or `generate_report=False` to skip the PDF step.
  `splice_source` chooses which cleaned variant gets spliced back —
  `"combined"` (default), `"ekgonly"`, or `"motiononly"`.
- `delsys.cleaning` module — building blocks for users who want to
  drive the pipeline manually (`fit_ica`,
  `score_components_against_ekg`, `auto_select_ekg_components`,
  `reconstruct_without_components`, `regress_out_ekg_from_emg`,
  `regress_out_motion_from_emg`, `harmonize_multirate_inputs`,
  `run_pipeline`).
- `CleaningConfig` / `CleaningResult` dataclasses (re-exported from
  `delsys`) — the configuration and result containers for
  `Log.clean_emg_ekg_artifact`. `CleaningResult` carries
  `cleaned_emg_ekgonly` (preprocess+ECG only) and
  `cleaned_emg_motiononly` (preprocess+motion only) variants
  alongside the combined `cleaned_emg`, plus `feature_names` and
  `fname` so the report and review helpers can label channels and
  default the output path.
- `CleaningResult.generate_report(path=None)` — writes a single
  multi-page PDF (page 1: ranked summary table; subsequent pages:
  one per EMG channel with raw vs each cleaning variant + PSD).
  Defaults to `<source_csv_stem>_cleaning_report.pdf` next to the
  input CSV when `result.fname` is stamped.
- `CleaningResult.review(channels=None)` — interactive matplotlib
  viewer with three stacked time-domain panels (raw vs ekg-only, raw
  vs motion-only, raw vs cleaned), arrow-key channel navigation, and
  per-overlay toggles (`e` / `m` / `c` / `o`). The three panels share
  both x and y axes so amplitude comparisons across stages line up
  without zoom-juggling.
- `CleaningResult.review_components(components=None)` — stacked
  4-panel viewer over the ICA components: top panel is the IC time
  course, the next three are the input signals it most contributes to
  (ranked by `|A[i, c]|`). Arrow-key cycling, `home` / `end` jumps,
  `q` to close. Use to decide whether to manually add or drop a
  component from the auto-detected set.
- `CleaningResult.ica` and `CleaningResult.ica_input_feature_names`
  fields — full `ICAResult` (model, sources, mixing, feature names)
  from the ECG stage plus the per-input-row labels (EMG names with
  `"EKG"` appended). Both are `None` when the ECG stage didn't run.
  Powers `review_components` and is exposed for power-user
  introspection.
- PDF report layout — page 1 is the new ECG diagnostics page (bar
  plot of per-IC correlation against the EKG reference, threshold
  line, and a text block listing the components removed); page 2 is
  the ranked summary table, now with a numeric `channel` column, a
  per-channel `location` label (from `lf.emg.signal_names`), and a
  `motion dB` column isolating the motion stage's contribution;
  pages 3..N are the per-channel pages (now with both x- and y-axis
  sharing across the three time-domain panels). The cleaner shifts
  the EMG baseline up front via `pysampled.Data.shift_baseline` so
  the dB metrics are not biased by a constant DC offset.
- `tutorials/cleaning_emg_ekg_artifact.md` — end-to-end walkthrough
  covering load → dry-run → PDF report → interactive review →
  in-place mutation → power-user knobs. Also covers
  `review_components`, `splice_source`, and the new tutorial sample.
- `scripts/make_tutorial_sample.py` and the bundled
  `tutorials/data/taichi_trial5_6s.csv` (6 s, every sensor kept)
  + matching reference report PDF — sample data the tutorial points
  at, big enough for ICA to converge on a real recording.

### Internal

- ECG component selection defaults to lagged-correlation
  auto-detection. Manual override via
  `CleaningConfig.ecg_components_to_remove`.
- Motion regression default ACC source is sensor-paired auto-discovery
  (Trigno Avanti sensors that carry both EMG and ACC). Custom
  pairings via `motion={emg_num: acc_num_or_location}`.
- Pipeline runs offline only in v1. The realtime / overlap-add
  variant from the source is intentionally not ported — restore if
  a real streaming use case appears.
- `run_pipeline` runs one extra `regress_out_motion_from_emg` pass
  on the preprocessed signal (skipping the ECG step) to populate
  `cleaned_emg_motiononly`. Cheap compared to the ICA fit, which is
  not duplicated.
- Reporting / review helpers (`_rank_channels_by_attenuation`,
  `_draw_channel_panels`, `_motion_outcome_for_channel`, etc.) live
  in `src/delsys/cleaning.py` alongside the dataclass. Matplotlib /
  scipy.signal.welch are imported lazily inside the helpers so
  `run_pipeline`-only callers don't pay the import cost.

### Fixed

- `clean_emg_ekg_artifact()` no longer crashes on `Log` objects
  loaded from very-old pickles where the per-:class:`Signal` ``meta``
  dict is empty. `_normalize_signal_lengths` reads ``meta.get("modality")``
  defensively; the splice-back updates each affected sensor's
  ``emg`` bundle via :meth:`pysampled.Data._clone` instead of
  rebuilding the whole :class:`Sensor` from ``lf.signals`` — so the
  cleaning lands on ``lf.emg`` even when per-:class:`Signal` access
  paths can't be repaired.
- Auto-report path is checked for write access *before* the cleaning
  pipeline runs. A locked PDF (file open in another viewer) now
  raises a clear :class:`PermissionError` with a "close it and
  re-run, or pass ``generate_report=False``" hint up front — no more
  wasted ICA work plus a half-applied in-place splice with no fresh
  report to match it.
- `_band_power` (Welch integral used by the report's `ecg-band dB`
  column) is NumPy 2.0-compatible. The previous
  `getattr(np, "trapezoid", np.trapz)` fallback evaluated the
  default eagerly and tripped the expired-attribute error on NumPy
  2.0; the new form uses `hasattr(np, "trapezoid")` so the legacy
  `np.trapz` is only accessed on NumPy < 1.26.

## [0.3.0] - 2026-05-10

Bundled cleanups: deprecates the legacy `Log.__getitem__` lookup,
introduces a registry-driven dispatch for both modality bundles
(internal) and link devices, and removes two public constants whose
sentinel-int identity was never the right abstraction.

The breaking piece is the constant removal — small enough that the
maintainer confirmed no caller depends on the old names. The originally
0.3.0-targeted `clean_emg_ekg_artifact` port and the
`_aggregate_bundles` → `pysampled.merge_along_signal_name` migration
shift to a later release.

### Removed (BREAKING)

- `delsys.VO2_SENSOR_NUM` and `delsys.HR_SENSOR_NUM` are no longer
  exported from the top-level package (or from `delsys._constants`).
  The sentinel values (`900` / `901`) survive as the integers stored
  in `delsys.LINK_DEVICE_REGISTRY` so existing pickles still resolve.

  **Migration:**

  | Before (0.2.x)                       | After (0.3.0)                                            |
  |--------------------------------------|----------------------------------------------------------|
  | `s.number == delsys.VO2_SENSOR_NUM`  | `s.is_link` (any link device) or `"VO2" in s.modalities` |
  | `delsys.HR_SENSOR_NUM`               | `delsys.LINK_DEVICE_REGISTRY["HR Strap"][1]`             |

### Deprecated

- `Log.__getitem__` — docstring-only deprecation (no runtime warning).
  `Log.find(...)` is the public-facing replacement; it takes named
  filters and always returns a list, instead of overloading five key
  types and collapsing single-match results. The legacy method is
  retained indefinitely for backward compatibility — there is no
  removal plan.

### Added

- `Sensor.is_link` — boolean property identifying link devices (VO2
  Master / HR Strap, plus any future entry in
  `LINK_DEVICE_REGISTRY`). Use this in preference to comparing
  `sensor.number` against magic ints.
- `LINK_DEVICE_REGISTRY` (in `delsys._constants`, re-exported from the
  top-level package) — `{sensor_name_substring: (modality, sensor_number)}`
  driving link-device detection in `_parse_sig_name_discover`. Adding a
  new link device is now a registry edit + corresponding
  `SUBCHANNEL_MAP` / `TARGET_SR` / `MODALITY_REGISTRY` entries — no new
  branches in the parser.

### Fixed

- `EMG.get_features` no longer trips pysampled's label/data validation.
  The previous implementation computed its time vector by feeding a
  `lambda x, ax: x.squeeze()` callable into `apply_running_win`, whose
  output collapsed the channel axis and ended up with `n_signals !=
  len(signal_names) * len(signal_coords)`. The time vector now comes
  directly from `make_running_win.center_idx` applied to the source
  signal's time grid, which sidesteps the round trip entirely.

### Internal

- `MODALITY_REGISTRY` in `delsys.sensor` replaces the if/elif modality
  dispatch in `Sensor.__init__`. New modalities (e.g. the
  `SmO2`/`Thb` keys already in `TARGET_SR`) become a one-line
  registry edit. Behavior is unchanged for every modality the parser
  currently emits.
- `_parse_sig_name_discover` link-device branch now iterates
  `LINK_DEVICE_REGISTRY` instead of hard-coding `"VO2 Master"` /
  `"HR Strap"` substring checks.
- `Log.export_to_csv` switched from `self[modality]` to
  `self.find(modality=modality, as_="modality")` so the only remaining
  `__getitem__` callers are external.

## [0.2.0] - 2026-05-10

> **Note:** 0.2.0 was tagged but not published — see [0.3.0](#030---2026-05-10).
> The changes described below shipped to PyPI as part of 0.3.0.

Breaking change to the typed `Log.<modality>` accessors plus a
parse-time fix for same-rate sample-count drift. Bumped to **0.2.0**
under the "breaking change → minor on 0.x" semver-on-0.x convention.

### Changed (BREAKING)

- `Log.emg` / `Log.ekg` / `Log.acc` / `Log.gyro` / `Log.fsr` /
  `Log.analog` / `Log.vo2master` / `Log.hrstrap` now return a single
  aggregated `pysampled.Data` per modality (channels stacked across
  every sensor that carries the modality), or `None` if no sensor
  does. Previously each returned a `List[Bundle]` (one entry per
  sensor). Use `bundle.split_by_signal_name()` to recover the
  per-sensor list.
- `EMG.process(amp_kind="nk")` and `EKG.find_rpeaks_pn` now raise
  `NotImplementedError` on multi-channel input with a hint to use
  `split_by_signal_name`. Previously both silently flattened
  column-major and produced nonsense.
- `FSR.a` / `.b` / `.c` / `.d` raise `NotImplementedError` on
  aggregate FSR (≠ 4 channels) since "the 4th channel" is meaningless
  across heterogeneous sensors. The per-Sensor 4-channel view is
  unchanged.

### Added

- `_normalize_signal_lengths` — same-rate length-drift normalization
  in `_util.py`, called from `Log.__init__` between parser dispatch
  and the per-Sensor stack. Tail-trims each `(modality, sr)` group to
  its shortest length so floating-point drift from the per-format
  resample step no longer trips `Sensor`'s same-modality length
  assert. Drift exceeding `_DRIFT_TOLERANCE` (4 samples) emits a
  `UserWarning`.
- `_aggregate_bundles` helper — stacks per-Sensor bundles along the
  signal axis, validates `signal_coords`/`axis`/`t0` agreement, and
  downsamples higher-rate parts to the lowest sampling rate (with
  `UserWarning`) when the input is multi-rate within one modality.
- `bundle.sensors` (plural) property on every modality bundle
  (`Signal`, `IMU`, `FSR`, `VO2Master`, `EMG`, `EKG`). Returns the
  aggregate's `meta["sensors"]` list when present, else
  `[meta["sensor"]]` for the per-Sensor case (length 1).
- `meta["sensors"]` convention on aggregate bundles, aligned with
  `signal_names` (so `len(bundle.meta["sensors"]) == len(bundle.signal_names)`).

### Fixed

- `IMU.x` / `.y` / `.z` use coord-lookup (`s["x"]`) instead of
  positional column slicing, so the same accessor works on per-Sensor
  IMU `(n, 3)` and aggregate IMU `(n, 3*N)`. The previous positional
  slice silently returned only the first sensor's column on multi-
  sensor input.

### Internal

- Moved `_SUBCHANNEL_KEYS` (FSR / Quattro channelmap parenthetical
  key format) from `_util.py` to `_constants.py` alongside
  `SUBCHANNEL_MAP`. No public API change.

### Migration

| Before (0.1.x)                | After (0.2.0)                                       |
|-------------------------------|-----------------------------------------------------|
| `for emg in lf.emg:`          | `for emg in lf.emg.split_by_signal_name():`         |
| `lf.acc[0].magnitude()`       | `lf.acc.split_by_signal_name()[0].magnitude()`      |
| `len(lf.emg)`                 | `len(lf.emg.signal_names) if lf.emg else 0`         |
| `if lf.acc:`                  | `if lf.acc is not None:`                            |
| `lf.acc[0]`                   | `lf.acc.split_by_signal_name()[0]`                  |
| `lf.fsr('')[1]`               | `lf.fsr.split_by_signal_name()[1]` (or by name)     |

Downstream consumers that iterate `Log.<modality>` lists need updating;
consumers that use `Sensor.<modality>` (per-Sensor view) are unchanged.

### Provenance

The accessor reshape and the drift fix are coupled: one bundle per
modality only makes sense if every sensor's channels for that modality
end up at identical post-resample lengths. The smoking-gun pickle is
at `C:/dev/immersionToolbox/_data/_resampling numbers/`, where two
ACC channels nominally at the same rate ended up 1 sample apart
because of a Trigno frame quantization boundary. Centralizing the fix
post-parse keeps the per-format parsers' quirks intact.

## [0.1.1] - 2026-05-09

Non-breaking metadata enrichment that also restores correct
`Data.magnitude()` semantics for downstream callers depending on
[pysampled](https://github.com/praneethnamburi/pysampled) ≥ 1.2.0
(per-`signal_name` L2). Existing pickles still unpickle fine —
`pysampled.Data.__setstate__` rebuilds defaults when the new fields are
missing.

### Added

- `delsys._util._trim_location` / `_canonical_label` /
  `_parse_fsr_quattro_positions` — small helpers used by
  `Sensor.__init__` to derive bundle labels from the channelmap.
- `Sensor._make_bundle_labels` — single source of truth for the
  per-modality `(signal_names, signal_coords)` convention:

  | Modality        | `signal_names`                                  | `signal_coords` |
  |-----------------|-------------------------------------------------|-----------------|
  | ACC / GYRO      | `[trimmed_location]`                            | `["x","y","z"]` |
  | EMGS            | `[loc]`                                         | `["emg"]`       |
  | EMGD            | `[loc_A, loc_B]`                                | `["emg"]`       |
  | EMGQ            | parsed positions, else `[loc_A..D]`             | `["emg"]`       |
  | FSR             | parsed positions, else `[loc_A..D]`             | `["fsr"]`       |
  | EKG             | `[loc]`                                         | `["ekg"]`       |
  | Analog          | `[loc]` (1ch) or `[loc_A..D]` (multi-ch sync)   | `["analog"]`    |
  | VO2             | 8 fixed names                                   | `["value"]`     |
  | HR              | `["heart_rate"]`                                | `["bpm"]`       |

- FSR / Quattro position-aware naming via the channelmap parenthetical
  (e.g. `LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)` →
  `["LFoot_Heel", "LFoot_OuterEdge", "LFoot_Ball", "LFoot_Toe"]`).
- `chN` fallback for sensors without a channelmap entry — keeps every
  bundle uniquely labelled.
- New `tests/test_util.py` plus extensions to `tests/test_signals.py`
  and `tests/test_log.py` covering the new label conventions, the
  inheritance change, and end-to-end propagation through `Log()`.

### Changed

- `IMU.x/y/z`, `FSR.a..d`, `VO2Master.*` no longer hardcode their
  `signal_names`; they inherit the parent's labels and only override the
  field that actually differs (single coord for `IMU.x/y/z`,
  parent-indexed name for `FSR` / `VO2Master`). Effect:
  `imu.x.signal_names == imu.signal_names`, and a single-axis `IMU`
  carries its sensor identity through downstream chains.
- All bundles in `Sensor.__init__` are now constructed with `axis=0`
  explicit so very short fixtures (e.g. a 1-row `(1, 8)` VO2 array)
  don't trip pysampled's `argmax`-based axis inference.
- `Sensor.__setstate__` now auto-relabels every attached bundle on
  unpickle, using the Sensor's own attributes (`number`, `location`,
  `modalities`) as the source of truth. Pickled `Log`s saved with
  delsys < 0.1.1 (or the legacy `immersionToolbox/immersionlab/delsys.py`
  shim) come out with the new convention with no caller changes — no
  `lf.relabel()` to remember. The relabel is idempotent on fresh 0.1.1+
  pickles. Per-`Signal` `meta` (`modality` / `subchannel` / `sensor`)
  is *not* recoverable from those very-old pickles; bundle-level
  access (`lf.acc[i]`, `lf.emg[i]`, `bundle["LFoot_Heel"]`, etc.) is
  what's repaired.
- **Bug fix:** `Analog` and `HRStrap` bundles now carry
  `meta=sensor_meta`. Previously the `pysampled.Data(...)`
  constructions for these two modalities dropped the parent
  `SensorInfo`, contradicting the `lf.analog[i].sensor.location`
  contract documented elsewhere in the API.
- **Dependency floor:** `pysampled` is now pinned to `>=1.2.0` (was
  `>=1.1.1`). The whole point of 0.1.1 is to make per-`signal_name`
  `magnitude()` (a 1.2.0 change) produce the right answer for
  Delsys data; tests assert `mag.signal_coords == ['mag']` which
  is also a 1.2.0 standardization. Older pysampled versions are no
  longer supported.

### Provenance

This release is the metadata complement to pysampled 1.2.0's
per-`signal_name` `magnitude()` change: with default labelling, every
downstream `acc.magnitude()` call in `pn-projects/wobble` would have
returned three independent per-axis abs values instead of the global
L2. Bundle-level labels make the original semantics work again — no
code change required at the call sites.

## [0.1.0] - 2026-05-09

First public release. The package is a standalone extraction and polish of the
Delsys CSV loader that previously lived in
`immersionToolbox/immersionlab/delsys.py` for several years. The
`immersionlab.delsys` module is now a thin shim that re-exports `delsys`, so
existing callers continue to work without code changes.

### Added

- Modular package layout under `src/delsys/`: `log`, `sensor`, `signals`,
  `emg`, `ekg`, plus internal `_parse`, `_metadata`, `_constants`, `_util`.
- `Log.find()` — typed query API for sensors, modality bundles, or raw
  signals, with filters by modality / side / location / sensor number / name.
- Direct typed accessors on `Log`: `lf.emg`, `lf.ekg`, `lf.acc`, `lf.gyro`,
  `lf.fsr`, `lf.analog`, `lf.vo2master`, `lf.hrstrap`. Side accessors
  `lf.left` / `lf.right` / `lf.center` return lists of `Sensor`.
- `EMG.rms()` — clean RMS amplitude pipeline (shift_baseline → highpass →
  lowpass → notch → running RMS) producing an envelope at the requested
  output sampling rate.
- `Log.add_sensor_group()` for user-defined groupings.
- `_metadata.py` module holding the three foundational namedtuples
  (`SensorLog`, `SensorInfo`, `SigInfoDelsys`) — extracted to break the
  `signals.py` ↔ `sensor.py` import cycle.
- Sensor metadata is now stored in `pysampled.Data.meta['sensor']` and
  propagates automatically through filtering, slicing, and resampling.
- Type hints (PEP 484) on every public function/method signature.
- Google-style docstrings throughout, ready for Sphinx + napoleon.
- Test suite: 94 tests across `test_parse.py`, `test_log.py`,
  `test_signals.py`, `test_emg.py`, `test_ekg.py` with fixtures for all
  five header formats (EMGworks, Discover 1.4.2 / 1.5.0 / 1.6.4 / 1.7.0).
- `scripts/make_fixture.py` for trimming real CSVs into committable fixtures.
- Standalone PyPI-publishable package with `pyproject.toml`.

### Changed

- **Minimum Python is now 3.10** (was 3.9). Bumped because a required
  dependency (`neurokit2`) ships PEP 604 union syntax (``X | None``) at
  module-import time, which fails on 3.9. Python 3.9 itself reached
  end-of-life in October 2025.
- **Bug fix:** `VO2Master` column ordering — all 8 channel properties used
  to map to column 1; now correctly map to columns 0..7 per
  `SUBCHANNEL_MAP['VO2']`.
- **Bug fix:** `_parse_sig_name` raises `ValueError` for an unknown modality
  instead of silently `print`ing and then raising a less-helpful `KeyError`
  downstream.
- **Bug fix:** `_detect_parser` raises `ValueError` instead of bare
  `Exception` when link data is exported without time-series columns.
- **Bug fix:** Frame-count invariant in the Discover parsers uses explicit
  `if not ...: raise ValueError(...)` (was `assert`, which is disabled by
  `python -O`).
- **Bug fix:** VO2 / HR detection in `_parse_sig_name_discover` uses
  `if`/`elif`/`else` (was two `if`s, with the second checking `'HR'`
  substring rather than `'HR Strap'` — could mis-trigger).
- **Bug fix:** `EMG.process()` preserves sensor metadata on the result
  (was lost via direct `self.__class__(...)` construction).
- **Bug fix:** `EMG.process()` no longer mutates `self._history` in place;
  history is built fresh on the new instance.
- **Bug fix:** `EMG.tkeo()` history entry is a single tuple (was a
  list-containing-tuple).
- O(N×M) sensor lookup inside the per-channel parser loops replaced with a
  pre-built `{sensor_number: SensorInfo}` dict (O(1) per channel).
- All regexes hoisted to module-level constants in `_parse.py`.
- Build backend: standalone `pyproject.toml` (no longer dependent on
  `immersionToolbox`'s build setup).
- Dependencies trimmed: only `pysampled`, `numpy`, `scipy`, `pandas`,
  `matplotlib`, `scikit-learn`, `heartpy`, `neurokit2`. No
  `immersionlab.utils`, no `pntools.sampled`.

### Removed

- `DataMod` base class — modality classes (`Signal`, `IMU`, `FSR`,
  `VO2Master`, `EMG`, `EKG`) now inherit from `pysampled.Data` directly.
  `envelope2` was upstreamed to pysampled in the same release window.
- `ica.py` (the accelerometer-impact ICA cleaning utility) — its scope
  didn't match the package's primary purpose (EMG-from-EKG cleaning).
  A `Log`-integrated EMG/EKG cleaning method is planned for a follow-up
  release; see `TODO.md` for the design questions.
- `decreturn` decorator and the `to=` parameter from `EMG.process_nk`,
  `EMG.get_features_nk`, `EMG.get_features`, `EKG.process_nk`,
  `EKG.get_features_hp`. These methods now return a plain `dict`; callers
  wrap with `pd.DataFrame(...)` if they want a tabular result.
- `EKG.find_rpeaks_hp`, `EKG.find_rpeaks_nk`, `EKG.clean_rpeaks`,
  `EKG.hrv`, `EKG.find_rr`, `EKG.find_rr_nk`, `EKG._get_sav_name` —
  alternative R-peak back-ends, the manual JSON-annotation workflow, and
  related downstream methods. `EKG.find_rpeaks_pn` (alias `find_rpeaks`)
  remains as the canonical detector.
- `if __name__ == '__main__'` smoke-test block with hard-coded UNC paths
  from the original monolithic file.

### Provenance

Extracted from `C:/dev/immersionToolbox/immersionlab/delsys.py` (1316 lines
monolithic). Built on top of [pysampled](https://github.com/praneethnamburi/pysampled).
