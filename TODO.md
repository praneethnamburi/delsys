# TODO

Items deferred during the initial standalone-package extraction.

## Coverage snapshot (2026-05-09)

Recorded after Phase E. Overall **81% line / 74% branch** across 766 statements.
Treat as informational — the user's direction was *measure-and-report only* for
0.1.0; no new tests were written to chase numbers.

| Module | Line cov | Branch cov | Note |
|---|---|---|---|
| `__init__.py` | 100% | n/a | Re-exports + module docstring. |
| `_constants.py` | 100% | n/a | Constants only. |
| `_metadata.py` | 100% | n/a | Three namedtuples. |
| `signals.py` | 96% | 98% | Sensor / shape / column properties exercised by F3. |
| `_parse.py` | 95% | 93% | Three uncovered branches: `_fix_corrupted_sensor_names` early-return for empty replace dict, an EMGworks edge case, two link-parser error paths. |
| `sensor.py` | 90% | 90% | Uncovered: `Sensor.get_signal()` priority-sequence walk (lines 122–126). |
| `ekg.py` | 81% | 73% | Uncovered: `process_nk` (no integration test), `get_features_hp` (no integration test), `_get_t_noisy_segments` non-empty branch, `flip_signal` re-detection on already-flipped signal. |
| `emg.py` | 80% | 80% | Uncovered: `_freq_funcs` lambdas (the freq-domain feature dict only verified by key, not by per-feature value), `process_nk`, `get_features_nk`. |
| `_util.py` | 78% | 60% | Uncovered: the `EMG`-prefix early-return path of `_mod_to_attr` is exercised only by lower-case input; uppercase input not directly tested. |
| **`log.py`** | **57%** | **47%** | Biggest gap. Uncovered: `__getitem__` / `_getitem_onekey` legacy lookup (lines 437–473), `export_to_csv` (603–633), several `find()` branches (501–530), the `_combine_signal_sensor_info` error path with the channelmap-mismatch traceback (256–260). |

Follow-ups (deferred to 0.1.1+):

- **`log.py` coverage push.** The legacy `__getitem__` and `export_to_csv`
  paths are public surface — a few targeted tests would close the biggest
  branch-coverage gap. The legacy bracket lookup is a candidate for
  deprecation, so test value depends on whether we plan to keep or remove
  it.
- **`emg.py` / `ekg.py` NeuroKit-backed features.** `process_nk`,
  `get_features_nk`, `process_nk` (EKG), `get_features_hp` are uncovered.
  These wrap upstream libraries; tests would mostly verify the dict-shape
  contract.

To regenerate this snapshot::

    pytest --cov

(Source filter, branch tracking, and exclude rules live in
`[tool.coverage.run]` / `[tool.coverage.report]` in `pyproject.toml`.)

## Shipped in 0.1.1 (2026-05-09)

- ✅ Bundle metadata enrichment in `Sensor.__init__` — every modality
  bundle now carries meaningful `signal_names` / `signal_coords` derived
  from the channelmap or sensor number, replacing the
  `["s0","s1",...]` / `["x"]` defaults that pysampled would otherwise
  fall through to. Restores correct `acc.magnitude()` semantics under
  pysampled ≥ 1.2.0.
- ✅ FSR / Quattro position-aware naming via the channelmap
  parenthetical (e.g. `LFoot (1-Heel, ...)` →
  `["LFoot_Heel", "LFoot_OuterEdge", "LFoot_Ball", "LFoot_Toe"]`).
- ✅ `chN` fallback for sensors without a channelmap entry.
- ✅ `IMU.x/y/z`, `FSR.a..d`, `VO2Master.*` inherit parent labels rather
  than hardcoding their own — a single-axis IMU keeps its sensor name.
- ✅ `Analog` and `HRStrap` bundles now carry `meta=sensor_meta`
  (previously dropped — minor pre-existing bug).
- ✅ `Sensor.__setstate__` auto-relabels bundles on unpickle, so old
  pickles produced before 0.1.1 (or by the legacy
  `immersionToolbox/immersionlab/delsys.py` shim) come out with the new
  `signal_names` / `signal_coords` convention with no caller changes.

### Known limitations carrying forward

- **Per-`Signal` `meta` on very-old pickles is unrecoverable.** Pickles
  produced before sensor metadata moved into `pysampled.Data.meta`
  (i.e. ones where every `Signal` has `meta == {}`) lose
  `signal.modality` / `signal.subchannel` / `signal.sensor` access.
  Bundle-level access (`lf.acc[i]`, `lf.emg[i]`,
  `bundle["LFoot_Heel"]`) is fully repaired by `Sensor.__setstate__`,
  so this only bites code that walks `lf.signals` directly. Reloading
  the original CSV is the workaround.

## Post-0.1.0 roadmap

- **EMG/EKG artifact cleaning at the `Log` level.** Port `C:\dev\pn-projects\projects\emg_ica_cleaning.py` (multi-stage pipeline: harmonization → preprocess → ICA-based ECG suppression with auto-component-detection by lagged correlation → ACC-guided motion regression with safety gates) into `delsys` as a `Log.clean_emg_ekg_artifact(...)` method. The integration work is to (a) gather all EMG `Signal`s + the EKG `Signal` (and optional per-EMG ACC predictors) from `lf`, (b) run the pipeline, (c) splice cleaned samples back into `lf.signals` per channel and rebuild affected `EMG` bundles in `lf.sensors[*].emg`, since `EMG._sig` is constructed by stacking signals at `Sensor.__init__` time. The previous ACC-only `ica.py` was removed in 0.1.0 because it didn't serve this stated primary purpose.

## Resampling defaults (post-0.1.0)

- **Reconsider the `TARGET_SR` defaults.** Currently every modality has a non-`None` default rate, so loading a CSV always resamples. A "preserve native rate" mode is conceptually attractive but not a five-minute change. Open design questions:

    1. EMGworks parser (`_parse_dataframe_emgworks`) doesn't handle a `None` target rate; needs a "skip resample" branch like the Discover-basic parser.
    2. Link devices (VO2 Master, HR Strap) are *asynchronous* — they have no native sampling rate, so `pysampled.uniform_resample` always needs a target. The "all-None" idea has no clean answer here. Possible fallback: pick rate from the median inter-sample interval, or require an explicit value just for link devices.
    3. API surface: a `mode='native'`/`'analysis'` knob vs. just changing default `TARGET_SR` values? The latter is a breaking change for downstream code that depends on current uniform rates.

  Per-modality `None` is already supported via explicit `target_sr={'EMGS': None, ...}`, so power users have an escape hatch today.

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
