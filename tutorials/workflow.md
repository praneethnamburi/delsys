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
                                                     + delsys_cleaning.json  (decisions)
```

Two kinds of cleaning show up here, and they have **separate owners**:

* **Algorithmic** ECG suppression + motion regression — `delsys`'s
  [`Log.clean_emg_ekg_artifact`](cleaning_emg_ekg_artifact.md), driven in batch
  by `delsys.clean`.
* **Human** noise-window marking (a cable bump, a dropped-sample burst) —
  authored in `datanavigator`'s `SignalBrowser` and merely *consumed* here.

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
* an entry in `delsys_cleaning.json` — the **decisions manifest** (next section).

`clean` is idempotent: a trial whose `*_cleaned.h5` already exists is a `hit`.
Pass `overwrite=True` to force a re-run (you'll do this after editing the
manifest).

## 3. The decisions manifest — `cleaned.h5 = f(raw.h5, manifest)`

The raw checkpoint is immutable. Everything that turns it into a *particular*
cleaned output — the ICA components removed, which cleaned variant was spliced,
the motion pairing, any noise-event reference, and the rest of the
[`CleaningConfig`](api.md) knobs — lives in a per-folder `delsys_cleaning.json`
keyed by trial id (the checkpoint stem):

```javascript
{
  "schema": 1,
  "trials": {
    "Trial_5": {
      // ICA component indices to zero out; [] removes none.
      "ecg_components_to_remove": [3],
      // Which cleaned variant to splice back: "combined" | "ekgonly" | "motiononly".
      "splice_source": "combined",
      // ACC pairing: "auto" | null (skip motion stage) | {emg_sensor: acc_sensor_or_location}.
      "motion": "auto",
      // Noise windows: null | {"path": "<event>.json", "key": "(2, 14, 17)"}.
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
}
```

The full set of `config` knobs (and their defaults) is the
[`CleaningConfig`](api.md) dataclass.

On the **first** pass, a trial with no manifest entry is cleaned with
auto-detection on, and its *resolved* decision is frozen into the manifest. On a
**later** pass, the frozen decision is replayed (auto-detection off, the recorded
components applied verbatim). Because the FastICA fit is seeded, replaying the
auto-chosen components reconstructs the same signal — a re-run reproduces the
cleaned checkpoint bit-for-bit. That is the reproducibility contract:

```python
delsys.clean(folder)                  # first pass: auto + freeze manifest
delsys.clean(folder, overwrite=True)  # replay: byte-identical cleaned.h5
```

### The edit / re-run loop

Open `Trial_5_cleaning_report.pdf`. If the auto-detected IC looks wrong (page 1
ranks each IC by its correlation with the EKG), pick a better one with
`review_components` — see the
[cleaning tutorial](cleaning_emg_ekg_artifact.md) — then edit the manifest and
re-run:

```javascript
// delsys_cleaning.json — change the component, or the spliced variant
"Trial_5": {
  "ecg_components_to_remove": [5],        // was [3]
  "splice_source": "ekgonly",            // skip an over-aggressive motion stage
  ...
}
```

```python
delsys.clean(folder, overwrite=True)   # regenerates Trial_5 from the edited entry
```

Only the trials you touched change; an unedited entry replays unchanged. The
`config` you pass to `clean()` is the *base* for trials that have **no** entry
yet — it never overrides an existing decision (that would break reproducibility).

`accept` tracks review status (`null` = not yet reviewed). If a trial's cleaning
is no good and you don't want it regenerated or trusted, set `accept` to `false`:
`clean()` then skips it (`skipped: rejected`) even under `overwrite=True`, until
you fix the decision and flip the flag back.

## 4. Authoring noise masks in datanavigator, consuming them here

Algorithmic cleaning handles ECG and motion artifact. Gross human-visible
noise — a cable yank, a sensor reseat, a dropped-sample burst — is better marked
by eye. Do that in `datanavigator`'s `SignalBrowser`: scrub the trial, mark the
bad spans as a noise Event, and save it. The Event is plain JSON (a
`[metadata, data]` list whose `data` maps each trial-id to `added` intervals in
seconds) — `delsys` reads it directly, with **no** `datanavigator` dependency.

Point a trial at its noise Event by setting `noise_event_ref` in the manifest
(`key` is the datanavigator trial-id tuple; a relative `path` resolves against
the checkpoint's folder):

```javascript
"Trial_5": {
  "ecg_components_to_remove": [3],
  "noise_event_ref": {
    "path": "event_noise_acc_bicep.json",
    "key": "(2, 14, 17)"
  },
  ...
}
```

```python
delsys.clean(folder, overwrite=True)
# delsys_cleaning_report.txt:  Trial_5.h5 - cleaned (ecg=[3], splice=combined, noise_masked=16)
```

The default policy is **NaN + interpolate**, applied modality-agnostically: a
noise window is a wall-clock span, so every modality (EMG, ACC, …) gets the same
treatment, and the masked-and-filled signals feed the artifact cleaner. To drive
this directly (outside the batch):

```python
from delsys import _noise

lf = delsys.Log("Trial_5.h5")
intervals = _noise.read_noise_intervals("event_noise_acc_bicep.json", (2, 14, 17))
_noise.apply_noise_mask(lf, intervals)            # all modalities
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
