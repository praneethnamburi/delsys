# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
