# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-05-10

Headline feature: a `Log`-integrated EMG/EKG artifact cleaning
pipeline. Ports the multi-stage cleaner from
`pn-projects/projects/emg_ica_cleaning.py` (preprocess → ICA-based
ECG suppression → ACC-guided motion regression with safety gates)
into the package, with a clean splice-back into `lf.signals` /
`lf.sensors[*].emg`.

### Added

- `Log.clean_emg_ekg_artifact(*, config, motion, in_place, generate_report)` —
  end-to-end pipeline running on every EMG channel in the Log. By
  default mutates `lf.signals` in place, rebuilds the affected
  `Sensor.emg` bundles, and writes a multi-page PDF report next to
  the source CSV. Pass `in_place=False` to inspect diagnostics
  without mutating, or `generate_report=False` to skip the PDF step.
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
  per-overlay toggles (`e` / `m` / `c` / `o`).
- `tutorials/cleaning_emg_ekg_artifact.md` — end-to-end walkthrough
  covering load → dry-run → PDF report → interactive review →
  in-place mutation → power-user knobs.

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
- `delsys.cleaning` re-exports the original symbol names from
  `pn-projects/projects/emg_ica_cleaning.py`
  (`EMGPipelineConfig` → `CleaningConfig`,
  `EMGPipelineResult` → `CleaningResult`,
  `fit_ica_emg` → `fit_ica`,
  `score_ica_components_against_ekg` → `score_components_against_ekg`,
  `run_emg_pipeline` → `run_pipeline`) as one-release-window aliases
  so downstream callers can switch their import path with no other
  changes.

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
