# Cleaning ECG and motion artifact from EMG

Walkthrough of `Log.clean_emg_ekg_artifact` end-to-end, from loading
a CSV to inspecting the results in a PDF report and an interactive
viewer. Snippets run verbatim against `tests/fixtures/discover170.csv`
in the repo.

## 1. Loading

`delsys.Log` reads a CSV exported from EMGworks or Trigno Discover and
groups its channels into per-sensor modality bundles. The cleaner needs
two things on the resulting `Log`:

* an EMG bundle (`lf.emg`) — the matrix of channels to clean.
* an EKG bundle (`lf.ekg`) — the cardiac reference used by the
  ECG-suppression stage.

```python
import delsys

lf = delsys.Log("tests/fixtures/discover170.csv")
assert lf.emg is not None, "no EMG channels — nothing to clean"
assert lf.ekg is not None, "no EKG reference — ECG stage will be skipped"
```

### Provenance of the bundled tutorial sample

For a longer real recording, use the bundled tutorial sample:

```python
lf = delsys.Log("tutorials/data/taichi_trial5_6s.csv")
```

The committed sample is a 6-second slice of
`S:/2210000787 - TaiChi/data/005/delsys/Trial_5.csv` — picked because
it shows clean ICA-based ECG suppression on a real recording.
Re-generate via `python scripts/make_tutorial_sample.py`.

## 2. Inspecting layout

Per-EMG-channel labels live on the aggregate bundle's `signal_names`.
Sensors that carry both EMG and ACC (typical Trigno Avanti) will be
auto-paired by the motion stage; EMGQ-only Quattro sensors won't.

```python
print(lf.emg.signal_names)
# ['ch1', 'ch2', 'ch3', 'ch4', ...]

emg_with_acc = [s for s in lf.sensors if hasattr(s, "emg") and hasattr(s, "acc")]
emg_only = [s for s in lf.sensors if hasattr(s, "emg") and not hasattr(s, "acc")]
print(f"EMG+ACC sensors: {len(emg_with_acc)}, EMG-only sensors: {len(emg_only)}")
```

## 3. Dry run

`in_place=False` returns the diagnostics without mutating `lf.signals`.
`generate_report=False` skips the PDF auto-write while you iterate.

```python
result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)

print("ECG components removed:", result.diagnostics["ecg"]["components_removed"])
print("ECG corr scores:", result.diagnostics["ecg"]["ic_ekg_corr_scores"])

motion = result.diagnostics["motion"]
if motion.get("used") is False:
    print("motion stage skipped")
else:
    for c in motion["per_channel"]:
        print(c["channel"], c["reason"])
```

`result.cleaned_emg_ekgonly` is the preprocess+ECG variant;
`result.cleaned_emg_motiononly` is the preprocess+motion variant.
Either is `None` when its stage didn't run, so handle that.

## 4. Generating a PDF report

`result.generate_report()` writes a single multi-page PDF. Page 1 is
the ranked summary table (most-attenuated channel first); subsequent
pages plot raw vs each cleaning variant for one channel apiece, plus
a PSD subplot.

```python
report_path = result.generate_report()
print("wrote", report_path)
```

When called with no `path=`, the PDF lands next to the source CSV at
`<stem>_cleaning_report.pdf`. `Log.clean_emg_ekg_artifact` does this
for you by default — `generate_report=False` opts out.

```python
# Equivalent to: lf.clean_emg_ekg_artifact() and then a PDF appears.
result = lf.clean_emg_ekg_artifact()
```

## 5. Interactive review

`result.review()` opens a 3-panel matplotlib window — raw vs ekg-only,
raw vs motion-only, raw vs combined-cleaned — and cycles through every
EMG channel in ranked-by-attenuation order.

Key bindings:

| key             | action                              |
|-----------------|-------------------------------------|
| `→` / `n`       | next channel (wrap)                 |
| `←` / `p`       | previous channel (wrap)             |
| `home` / `end`  | first / last channel                |
| `e`             | toggle ekg-only overlay             |
| `m`             | toggle motion-only overlay          |
| `c`             | toggle combined-cleaned overlay     |
| `o`             | toggle all overlays at once         |
| `q`             | close                               |

Restrict to specific channels with `channels=[...]` (column indices
into the EMG matrix):

```python
# Inspect just the top 5 most-attenuated channels.
ranked = sorted(
    range(result.cleaned_emg.shape[1]),
    key=lambda i: result.cleaned_emg[:, i].var() / result.stages["raw"][:, i].var(),
)
result.review(channels=ranked[:5])
```

## 5b. Reviewing ICA components

`result.review_components()` opens a 4-panel viewer showing one
component at a time — the IC time course on top, then the three input
signals it most contributes to (ranked by `|A[i, c]|`, the absolute
mixing-matrix coefficient). Use this when the auto-detected component
looks wrong: cycle through every IC, decide which to keep / drop, then
re-run with a manual override via
`CleaningConfig.ecg_components_to_remove`.

Key bindings:

| key             | action                              |
|-----------------|-------------------------------------|
| `→` / `n`       | next component (wrap)               |
| `←` / `p`       | previous component (wrap)           |
| `home` / `end`  | first / last component              |
| `q`             | close                               |

```python
result.review_components()              # cycle through every IC
result.review_components(components=[0, 4, 7])  # restrict to a subset
```

The viewer needs `result.ica` to be populated, which only happens when
the ECG stage ran. `result.ica_input_feature_names` lists the EMG
channel names with `"EKG"` appended — those are the labels rendered on
the contributor panels.

## 6. Mutating in place

The default `in_place=True` rewrites the EMG sample arrays inside
`lf.signals` and rebuilds the affected `Sensor.emg` bundles, so
`lf.emg()` and `lf.sensors[*].emg()` reflect the cleaned data on
next access. The returned `CleaningResult` still carries the cleaned
matrix and per-stage snapshots for diagnostics.

```python
raw_before = lf.emg().copy()
lf.clean_emg_ekg_artifact()  # mutates lf.signals, writes the PDF
import numpy as np

assert not np.array_equal(lf.emg(), raw_before)
```

After the call, downstream EMG envelope / feature pipelines that read
`lf.emg` see the cleaned signal — no separate `processed_emg` argument
to thread through.

### Picking which variant to splice back

The default splices `result.cleaned_emg` (preprocess + ECG + motion)
back into `lf.signals`. When the motion stage is too aggressive on a
trial — the safety gates pass but the residual still over-cleans — you
can splice the ECG-only variant instead:

```python
# Use the ECG-only cleaning, skip the motion regression.
lf.clean_emg_ekg_artifact(splice_source="ekgonly")
```

Or, when the motion stage is doing the heavy lifting and the ECG
suppression is shaving useful EMG off:

```python
lf.clean_emg_ekg_artifact(splice_source="motiononly")
```

`splice_source` is ignored when `in_place=False`. The auto-report runs
*after* the splice-back, so the per-channel pages reflect what
`lf.emg` will look like.

## 7. Power-user knobs

`CleaningConfig` exposes every stage knob. See its docstring for the
full list; common overrides:

```python
from delsys import CleaningConfig

# Manual ECG component override (skip auto-detection).
cfg = CleaningConfig(
    ecg_auto_remove_components=False,
    ecg_components_to_remove=[2, 5],
)

# Skip the motion stage entirely.
cfg = CleaningConfig(use_motion_stage=False)

# Tighten the safety gates so aggressive motion cleaning gets rejected.
cfg = CleaningConfig(min_variance_ratio=0.4, min_power_ratio=0.4)

result = lf.clean_emg_ekg_artifact(config=cfg, in_place=False)
```

Custom EMG↔ACC pairing lets you point an EMG sensor at a different
sensor's ACC bundle (e.g., a co-located Trigno IM sensor):

```python
# EMG sensor 4 borrows ACC from sensor 11; sensor 14 stays auto.
result = lf.clean_emg_ekg_artifact(motion={4: 11})
```

For one-off IC inspection without running the full pipeline, drop
into the building blocks directly:

```python
import numpy as np
from delsys.cleaning import fit_ica, score_components_against_ekg

emg_2d = np.asarray(lf.emg())
ekg_1d = np.asarray(lf.ekg()).reshape(-1)
ica = fit_ica(np.column_stack([emg_2d, ekg_1d]))
scores, lags = score_components_against_ekg(ica.sources, ekg_1d)
print("IC scores against EKG:", scores)
```

## 8. Reference report

The PDF generated from the bundled tutorial sample lives at
`tutorials/data/taichi_trial5_6s_cleaning_report.pdf` — open it
alongside this walkthrough to see what each page looks like end-to-end:

* **Page 1.** ECG diagnostics — bar plot of per-IC correlation against
  the EKG reference (highlighted bars are the components that were
  removed), plus the threshold line and a text block listing the
  components removed.
* **Page 2.** Summary table — one row per EMG channel, ranked by
  total-power dB attenuation, with `total dB`, `ecg-band dB`,
  `motion dB`, and the motion stage's per-channel outcome.
* **Pages 3..N.** One per channel — three time-domain panels (raw vs
  ekg-only, raw vs motion-only, raw vs cleaned) sharing both axes, plus
  a PSD subplot.
