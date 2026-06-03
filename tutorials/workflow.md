# End-to-end workflow: CSV → checkpoint → cleaned checkpoint

This walkthrough ties the batch tools together into the workflow a study
actually runs: take a folder of raw Delsys CSV exports, convert them to
self-contained HDF5 checkpoints, clean ECG and motion artifact out of every
trial, and end up with analysis-ready `*_cleaned.h5` files plus a reviewable,
*re-runnable* record of every decision.

```
Trial_5.csv ──process()──▶ Trial_5.h5 ──clean()──▶ Trial_5_cleaned.h5
                           (raw, native rate)        (cleaned snapshot)
                                                     + Trial_5_cleaning_report.pdf
                                                     + Trial_5.delsys-artifact  (decision)
```

Two kinds of cleaning show up here, and they have **separate owners**:

* **Algorithmic** ECG suppression + motion regression — `delsys`'s
  [`Log.clean_emg_ekg_artifact`](cleaning_emg_ekg_artifact.md), driven in batch
  by `delsys.clean`.
* **Human** annotation — noise windows (a cable bump, a dropped-sample burst) and
  typed event markers — authored in `lf.view()` (a `datanavigator` `SignalBrowser`
  subclass) and written to one `<stem>.delsys-events` sidecar; the noise track is
  *consumed* here.

## 1. CSV → native checkpoint (`process`)

`delsys.process` walks a folder, resolves each CSV's channelmap, and writes a
native-rate `.h5` next to it. The checkpoint is self-contained (channelmap +
header embedded) and target-independent — pick the analysis rate later, on load.

```python
import delsys

delsys.process("S:/2210000787 - TaiChi/data/005/delsys")
# delsys.process: 7 CSV(s) under '.../delsys'.
# delsys.process: built 7 | hit 0 | skipped 0 | error 0
```

The CSV is now disposable; everything downstream reads the `.h5`. See the HDF5
notes in `CHANGELOG.md` for why the checkpoint stores native rate and resamples
on load (one checkpoint serves any target rate / clock).

## 2. Checkpoint → cleaned checkpoint (`clean`)

`delsys.clean` mirrors `process`: it walks the folder for raw `Trial_*.h5`,
runs the cleaner on each, and writes a terminal-snapshot `*_cleaned.h5`. Its
own outputs are skipped on re-walk, so it never tries to clean a cleaned file.

```python
delsys.clean("S:/2210000787 - TaiChi/data/005/delsys")
# delsys.clean: 7 checkpoint(s) under '.../delsys'.
# delsys.clean: cleaned 7 | hit 0 | skipped 0 | error 0
```

Each trial produces three things next to its checkpoint:

* `Trial_5_cleaned.h5` — the cleaned signals as a terminal snapshot (loaded the
  same way as any checkpoint: `delsys.Log("Trial_5_cleaned.h5")`).
* `Trial_5_cleaning_report.pdf` — the per-trial report (IC ↔ EKG diagnostics,
  ranked attenuation table, per-channel before/after). The report path anchors
  on the checkpoint, so it lands right beside it.
* a `Trial_5.delsys-artifact` — the **decision sidecar** (next section).

`clean` is idempotent: a trial whose `*_cleaned.h5` already exists is a `hit`.
Pass `overwrite=True` to force a re-run (you'll do this after editing the
sidecar).

## 3. The decision sidecar — `cleaned.h5 = f(raw.h5, .delsys-artifact)`

The raw checkpoint is immutable. Everything that turns it into a *particular*
cleaned output — the ICA components removed, which cleaned variant was spliced,
the motion pairing, any noise reference, and the rest of the
[`CleaningConfig`](api.md) knobs — lives in a sibling `<stem>.delsys-artifact`
sidecar that travels with the checkpoint (the same per-log model as
`<stem>.delsys-events`):

```javascript
{
  "schema": 1,
  "cleaning": {
    // ICA component indices to zero out; [] removes none.
    "ecg_components_to_remove": [3],
    // Which cleaned variant to splice back: "combined" | "ekgonly" | "motiononly".
    "splice_source": "combined",
    // ACC pairing: "auto" | null (skip motion stage) | {emg_sensor: acc_sensor_or_location}.
    "motion": "auto",
    // Noise windows: null | "<stem>.delsys-events" | "<stem>.delsys-noise" | {"path": "<event>.json", "key": "(2, 14, 17)"}.
    "noise_event_ref": null,
    // Review status: null (unreviewed) | true (approved) | false (blocks regen).
    "accept": null,
    // Any CleaningConfig knob; omitted ones fall back to the dataclass default.
    "config": {
      "preprocess_highpass_hz": 20.0,  // float Hz | null (skip the high-pass)
      "use_ecg_stage": true,           // true | false (skip ICA + EKG regression)
      "use_motion_stage": true,        // true | false (skip ACC regression)
      "min_variance_ratio": 0.1,       // 0..1 safety gate (higher = reject more motion cleaning)
      "...": "..."                     // see the CleaningConfig docstring for the full set
    }
  }
}
```

The full set of `config` knobs (and their defaults) is the
[`CleaningConfig`](api.md) dataclass. A per-folder `delsys_cleaning_report.txt`
summarizes a run (an overview, not the source of truth).

On the **first** pass, a trial with no sidecar is cleaned with
auto-detection on, and its *resolved* decision is frozen into a fresh sidecar. On a
**later** pass, the frozen decision is replayed (auto-detection off, the recorded
components applied verbatim). Because the FastICA fit is seeded, replaying the
auto-chosen components reconstructs the same signal — a re-run reproduces the
cleaned checkpoint bit-for-bit. That is the reproducibility contract:

```python
delsys.clean(folder)                  # first pass: auto + freeze the sidecars
delsys.clean(folder, overwrite=True)  # replay: byte-identical cleaned.h5
```

### The edit / re-run loop

Open `Trial_5_cleaning_report.pdf`. If the auto-detected IC looks wrong (page 1
ranks each IC by its correlation with the EKG), the easiest fix is the
interactive reviewer — it picks components, previews the result, and writes the
decision sidecar for you:

```python
delsys.Log("Trial_5.h5").clean()      # click ICs, pick splice, Save
delsys.clean(folder, overwrite=True)  # regenerate from the saved decision
```

`lf.clean()`'s **Save** writes the decision into `Trial_5.delsys-artifact`
and clears the stale `Trial_5_cleaned.h5`, so the re-run regenerates from it (see
the [cleaning tutorial](cleaning_emg_ekg_artifact.md)). You can also hand-edit the
sidecar directly — change the component or the spliced variant — and re-run:

```javascript
// Trial_5.delsys-artifact — change the component, or the spliced variant
{
  "schema": 1,
  "cleaning": {
    "ecg_components_to_remove": [5],     // was [3]
    "splice_source": "ekgonly",          // skip an over-aggressive motion stage
    "...": "..."
  }
}
```

```python
delsys.clean(folder, overwrite=True)   # regenerates Trial_5 from the edited sidecar
```

Only the trials you touched change; an unedited sidecar replays unchanged. The
`config` you pass to `clean()` is the *base* for trials that have **no** sidecar
yet — it never overrides an existing decision (that would break reproducibility).

`accept` tracks review status (`null` = not yet reviewed). If a trial's cleaning
is no good and you don't want it regenerated or trusted, set `accept` to `false`:
`clean()` then skips it (`skipped: rejected`) even under `overwrite=True`, until
you fix the decision and flip the flag back.

## 4. Annotating a trial: `lf.view()` → `.delsys-events`

Algorithmic cleaning handles ECG and motion artifact. Everything a human marks by
eye — gross **noise** (a cable yank, a reseat, a dropped-sample burst, a dead
electrode) *and* **events** (note onsets, phrase boundaries, task phases) — goes
through `Log.view()`, which opens an interactive browser (a `datanavigator`
`SignalBrowser` subclass) over the Log's signals and writes one unified
`<stem>.delsys-events` sidecar. `datanavigator` is imported lazily there, so the
delsys core stays `datanavigator`-free until you call it:

```python
lf = delsys.Log("Trial_5.h5")
lf.view()                       # noise track + a "1" point + a "2" window track
lf.view(events={"onset": 1, "phrase": 2})   # custom marker tracks
```

* The sidebar **dropdown** lists every signal by its structural key
  `"<sensor>.<modality>.<coord> | <location>"`; pick one (or arrow through them).
* **Noise:** hover and press **`n`** to mark a window — two presses fix its start
  and end; **`alt+n`** removes the nearest; **`d`** (Toggle dead) marks the current
  scope dead for the whole recording.
* **Markers:** press a **digit** to drop a typed event at the cursor — **`1`** adds
  a `"1"`-event (a point), **`2`** a `"2"`-event (a window, two presses); **`alt+1`**
  / **`alt+2`** remove the nearest. Each mark records the **signal it was placed
  from** (provenance), and is read back as a trial-level marker.
* The **Mod scope** toggle records a *noise* window against the whole
  sensor+modality (coord-less key) instead of the single channel — for a bump that
  hits every axis; markers always record the coord-ful channel. **Undo window**
  drops the last noise window; **Save** writes the sidecar.

For noise shared across a sensor's channels (a mechanical/cable artifact that
hits EMG *and* ACC/GYRO together), the **sensor-centric** view stacks one
sensor's modalities as time-aligned subplots:

```python
lf.view("sensor")   # dropdown picks the sensor; n / alt+n / d / digits
```

Here marking targets the **hovered subplot's** address; a **Sensor scope** toggle
fans a noise mark across every modality of the sensor. Both views write the same
`<stem>.delsys-events`.

Marks land in a sibling `<stem>.delsys-events` file (JSON; composite suffix so
portfolio `*.json` tooling skips it), split by event type, each keyed by signal
address:

```javascript
// Trial_5.delsys-events
{
  "schema": 1,
  "events": {
    "noise": { "kind": "noise", "signals": {
      "3.EMGS.A | Tricep_L":  { "windows": [[1.0, 2.0]] },
      "4.ACC | Bicep_R":      { "windows": [[5.2, 5.4]] },   // whole-modality
      "9.FSR.C | LFoot_Ball": { "dead": [[12.4, null]] }      // dies at 12.4 s
    }},
    "1": { "kind": "marker", "size": 1, "signals": {
      "3.EMGS.A | Tricep_L": [[1.40], [2.10]]                 // two point events
    }}
  }
}
```

Noise `windows` are blanked to `NaN` and interpolated back; `dead` spans are
zero-filled (a `null` endpoint is open — `[T, null]` = dead from `T` on,
`[null, null]` = the whole recording). `delsys.clean` **auto-consumes** a sibling
`.delsys-events` (its `noise` type; no manifest edit needed) and records it in the
trial's `noise_event_ref` as provenance:

```python
delsys.clean(folder, overwrite=True)
# delsys_cleaning_report.txt:  Trial_5.h5 - cleaned (ecg=[3], splice=combined, noise_masked=4)
```

Marker tracks are read back collapsed across signals into trial-level markers
(every mark kept, with its originating address; pass `dedupe=<seconds>` to merge
near-coincident marks):

```python
from delsys import _events

recs = _events.collapse_markers("Trial_5.delsys-events", "1")
# [{"seq": [1.40], "address": "3.EMGS.A", "label": "Tricep_L"}, ...]
```

To drive noise masking directly (outside the batch):

```python
from delsys import _events

lf = delsys.Log("Trial_5.h5")
_events.apply_events_noise(lf, "Trial_5.delsys-events")   # per-signal addresses
```

A legacy `<stem>.delsys-noise` (the pre-unification per-signal sidecar) is still
read when no `.delsys-events` is present, and folded into the unified file on the
next `lf.view()` save.

### Legacy: datanavigator noise Events (trial-keyed, flat intervals)

The older path consumes a `datanavigator` noise **Event** file — plain JSON (a
`[metadata, data]` list whose `data` maps each trial-id to `added` intervals in
seconds), read directly with **no** `datanavigator` dependency, and applied
modality-agnostically to every signal. It still works: set a manifest
`noise_event_ref` to `{path, key}` (a `.json` path; `key` is the trial-id tuple)
and `clean` dispatches by suffix.

```python
from delsys import _noise

intervals = _noise.read_noise_intervals("event_noise_acc_bicep.json", (2, 14, 17))
_noise.apply_noise_mask(lf, intervals)                       # all modalities
_noise.apply_noise_mask(lf, intervals, modalities=["EMGS"])  # or restrict
```

## 5. Analysis on the cleaned checkpoints

A `*_cleaned.h5` loads like any checkpoint, so existing analysis code needs no
changes — point it at the cleaned files:

```python
import glob

for h5 in glob.glob("S:/.../005/delsys/Trial_*_cleaned.h5"):
    lf = delsys.Log(h5)
    for emg in lf.emg.split_by_signal_name():
        envelope = emg.process(amp_kind="envelope2")
        ...
```

Holding a whole study in memory is cheap: `Log(".h5")` is lazy — only the header
is read at construction, and the signal datasets load on first access. So
`{trial_id: delsys.Log(h5) for h5 in ...}` over hundreds of trials costs almost
nothing until you touch each one.
