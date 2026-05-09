# TODO

Items deferred during the initial standalone-package extraction.

## Post-0.1.0 roadmap

- **EMG/EKG artifact cleaning at the `Log` level.** Port `C:\dev\pn-projects\projects\emg_ica_cleaning.py` (multi-stage pipeline: harmonization → preprocess → ICA-based ECG suppression with auto-component-detection by lagged correlation → ACC-guided motion regression with safety gates) into `delsys` as a `Log.clean_emg_ekg_artifact(...)` method. The integration work is to (a) gather all EMG `Signal`s + the EKG `Signal` (and optional per-EMG ACC predictors) from `lf`, (b) run the pipeline, (c) splice cleaned samples back into `lf.signals` per channel and rebuild affected `EMG` bundles in `lf.sensors[*].emg`, since `EMG._sig` is constructed by stacking signals at `Sensor.__init__` time. The previous ACC-only `ica.py` was removed in 0.1.0 because it didn't serve this stated primary purpose.

## Decorator behavior

- **`decreturn` ignores wrapped-function defaults.** Methods decorated with `decreturn` (e.g. `EMG.get_features`, `EKG.get_features_hp`) declare a default `to=dict` in their signature, but the decorator reads `kwargs['to']` before calling the function and raises `KeyError` when the caller omits `to=`. Should fall back to the function's default. Caught while writing F4 tests on 2026-05-08.

## Restructure / API

- **`Sensor.__init__` modality dispatch.** It uses an if/elif chain on modality strings to choose which class to instantiate. A small `MODALITY_REGISTRY` mapping `{'EMG': EMG, 'EKG': EKG, ...}` would be cleaner and would make adding modalities (e.g. SmO2/Thb that already appear in `TARGET_SR`) one-line changes.

- **`Log.__getitem__` overloading.** Mixes sensor-lookup and signal-lookup based on key type (int, single-letter, modality string, location, name). Works but is undiscoverable. The new `lf.find(...)` is the public-facing replacement; `__getitem__` could be deprecated when there's a transition window.

- **VO2 / HR identity.** `VO2_SENSOR_NUM` and `HR_SENSOR_NUM` are placeholder integers chosen to not collide with Trigno-base sensor numbers. Brittle if Delsys ever ships a third link device. A real fix identifies link sensors by type rather than number — would also clean up the special-cased branches in `_parse_sig_name_discover` and `Sensor.__init__`.

## Pre-existing in-code TODOs

- **EKG**: "Slicing will not work well with the cached rpeak indices. TODO: modify the `__getitem__` method."
- **`_parse_dataframe_discover_with_link`**: "Make the exception explicit. The 13.5 ms sampling-rate calculation only applies for trigno base, not link devices."
- **VO2 link**: "VO2 can start delayed (fill these data points with zeros), VO2 can finish before the other system."
- **`discover_basic` parser path**: "Check if this works for a file with timestamps exported." Needs a fixture with timestamps but no link sensors.

## Domain / units

- VO2Master `VO2_absolute` returns raw CSV values (~37000 for moderate exercise). Likely a units issue, not a column-mapping one — verify against a known-correct VO2 reading and add unit conversion if needed.

## Stage 4 status (added 2026-05-08)

All six features have docs + tests in `tests/`:

1. ✅ CSV header parsing — `tests/test_parse.py` (22 tests).
2. ✅ End-to-end `Log` loading — `tests/test_log.py` (20 tests, parametrized over all 7 fixtures).
3. ✅ Signal classes — `tests/test_signals.py` (30 tests).
4. ✅ `EMG.process` envelope pipeline — `tests/test_emg.py` (14 tests).
5. ✅ `EKG` R-peak detection — `tests/test_ekg.py` (NeuroKit's `ecg_simulate`).
6. ✅ ICA cleaning — `tests/test_ica.py` (using a `Log`-shaped mock).

Fixtures live under `tests/fixtures/` and are generated from `_data/` via
`scripts/make_fixture.py`.
