# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
