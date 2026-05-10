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
  branch-coverage gap. As of 0.3.0 `__getitem__` is docs-only deprecated
  with no removal plan, so dedicated coverage of its branches is
  worth-doing rather than throw-away work.
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

## Shipped in 0.4.0 (2026-05-10)

- ✅ `Log.clean_emg_ekg_artifact(*, config, motion, in_place, generate_report, splice_source)` —
  end-to-end EMG/EKG artifact cleaner ported from
  `pn-projects/projects/emg_ica_cleaning.py`. By default mutates
  `lf.signals` in place, rebuilds the affected `Sensor.emg`
  bundles, and writes a multi-page PDF next to the source CSV;
  pass `in_place=False` / `generate_report=False` to opt out.
  `splice_source="ekgonly"` / `"motiononly"` swap which variant
  gets spliced back when a stage is doing more harm than good.
- ✅ `delsys.cleaning` module — building blocks for power users
  (`fit_ica`, `score_components_against_ekg`,
  `auto_select_ekg_components`, `reconstruct_without_components`,
  `regress_out_ekg_from_emg`, `regress_out_motion_from_emg`,
  `harmonize_multirate_inputs`, `run_pipeline`) plus
  `CleaningConfig` / `CleaningResult` dataclasses.
- ✅ `CleaningResult.cleaned_emg_ekgonly` / `cleaned_emg_motiononly`
  stage-isolated variants, plus `feature_names` and `fname` so the
  reporting helpers can label channels and default the output path.
- ✅ `CleaningResult.generate_report(path=None)` — single multi-page
  PDF (ranked summary on page 1, per-channel pages thereafter).
- ✅ `CleaningResult.review(channels=None)` — interactive matplotlib
  viewer with arrow-key channel cycling and `e` / `m` / `c` / `o`
  overlay toggles. Three time-domain panels share both x and y axes.
- ✅ `CleaningResult.review_components(components=None)` — 4-panel
  viewer over ICA components (IC time course + top three input
  contributors ranked by `|A[i, c]|`). Pairs with the new
  `result.ica` / `result.ica_input_feature_names` fields, populated
  whenever the ECG stage runs.
- ✅ PDF gains a new ECG diagnostics page first (bar plot of per-IC
  correlation against the EKG reference + components-removed text);
  summary table grows a `motion dB` column isolating the motion
  stage's contribution; per-channel pages share y-axes across the
  three time-domain panels.
- ✅ `tutorials/cleaning_emg_ekg_artifact.md` — end-to-end walkthrough,
  extended with `review_components` + `splice_source` sections and a
  pointer to the bundled reference report.
- ✅ `scripts/make_tutorial_sample.py` + committed
  `tutorials/data/taichi_trial5_6s.csv` and matching reference
  PDF — 6 s slice of a real TaiChi recording (every sensor kept), big
  enough for ICA to converge cleanly.
- ✅ Original symbol names re-exported as one-release-window aliases
  for source-compatibility with `pn-projects/projects/emg_ica_cleaning.py`.

## Shipped in 0.3.0 (2026-05-10)

- ✅ `Log.__getitem__` deprecated (docstring-only — no removal
  planned). `Log.find(...)` is the public replacement.
- ✅ `MODALITY_REGISTRY` in `sensor.py` replaces the if/elif modality
  dispatch in `Sensor.__init__`.
- ✅ `LINK_DEVICE_REGISTRY` in `_constants.py` consolidates link-device
  detection in `_parse_sig_name_discover`.
- ✅ `Sensor.is_link` property added.
- ✅ **BREAKING:** `VO2_SENSOR_NUM` / `HR_SENSOR_NUM` removed from
  public exports. Synthetic numbers survive inside
  `LINK_DEVICE_REGISTRY` so existing pickles still resolve.
- ✅ `Log.export_to_csv` migrated from `self[modality]` to
  `self.find(modality=modality, as_="modality")`.

## Shipped in 0.2.0 (2026-05-10)

- ✅ `Log.<modality>` accessors return a single aggregated bundle per
  modality (channels stacked across all sensors that have the
  modality) instead of `List[Bundle]`. Migration via
  `bundle.split_by_signal_name()`.
- ✅ Same-rate sample-count drift normalized at parse time
  (`_normalize_signal_lengths` in `_util.py`), so per-Sensor stacking
  no longer trips on 1-sample drift between channels of one modality.
- ✅ `_aggregate_bundles` helper for multi-Sensor stacking, with
  lowest-SR resample + `UserWarning` for multi-rate input.
- ✅ `bundle.sensors` (plural) property unifies per-Sensor and
  aggregate views; `meta["sensors"]` convention aligned with
  `signal_names` on aggregates.
- ✅ `IMU.x/y/z` use coord-lookup so the same accessor works on
  per-Sensor and aggregate shapes.
- ✅ `FSR.a/b/c/d` guarded to the 4-channel per-Sensor view; aggregate
  FSR raises with a hint pointing at `split_by_signal_name()`.
- ✅ `EMG.process(amp_kind='nk')` and `EKG.find_rpeaks_pn` raise
  `NotImplementedError` on multi-channel input (was silent
  column-major flatten).

### Followups (target 0.5.0)

- **Migrate `_aggregate_bundles` to `pysampled.Data.merge_along_signal_name`**
  once pysampled ships those classmethods (deferred from pysampled
  1.2.0). Status check: pysampled 1.2.0 explicitly pulled them
  before release; revisit when pysampled 1.3.0 plans firm up.
- **Plotting helpers for the cleaner.** Port
  `plot_ica_components` / `plot_signals_before_after` from
  `pn-projects/projects/emg_ica_cleaning.py` if a downstream caller
  asks. Likely a `delsys.plotting` module.
- **Realtime / overlap-add cleaning variant.** Port the chunked
  offline `_run_pipeline_realtime` from the source if a real
  streaming workflow shows up. Offline-only was the deliberate v1
  scope.
- **Drop the back-compat aliases in `delsys.cleaning`**
  (`EMGPipelineConfig` / `EMGPipelineResult` / `fit_ica_emg` /
  `score_ica_components_against_ekg` / `run_emg_pipeline`) once
  downstream callers have migrated to the canonical names.

### Known limitations carrying forward

- **Per-`Signal` `meta` on very-old pickles is unrecoverable.** Pickles
  produced before sensor metadata moved into `pysampled.Data.meta`
  (i.e. ones where every `Signal` has `meta == {}`) lose
  `signal.modality` / `signal.subchannel` / `signal.sensor` access.
  Bundle-level access (`lf.acc[i]`, `lf.emg[i]`,
  `bundle["LFoot_Heel"]`) is fully repaired by `Sensor.__setstate__`,
  so this only bites code that walks `lf.signals` directly. Reloading
  the original CSV is the workaround.

## Resampling defaults (post-0.1.0)

- **Reconsider the `TARGET_SR` defaults.** Currently every modality has a non-`None` default rate, so loading a CSV always resamples. A "preserve native rate" mode is conceptually attractive but not a five-minute change. Open design questions:

    1. EMGworks parser (`_parse_dataframe_emgworks`) doesn't handle a `None` target rate; needs a "skip resample" branch like the Discover-basic parser.
    2. Link devices (VO2 Master, HR Strap) are *asynchronous* — they have no native sampling rate, so `pysampled.uniform_resample` always needs a target. The "all-None" idea has no clean answer here. Possible fallback: pick rate from the median inter-sample interval, or require an explicit value just for link devices.
    3. API surface: a `mode='native'`/`'analysis'` knob vs. just changing default `TARGET_SR` values? The latter is a breaking change for downstream code that depends on current uniform rates.

  Per-modality `None` is already supported via explicit `target_sr={'EMGS': None, ...}`, so power users have an escape hatch today.

## Restructure / API

- **`Log.__getitem__` overloading.** Docstring-deprecated as of 0.3.0,
  retained indefinitely. Removal is not planned — kept here for
  visibility in case the call shape causes problems for new users.

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
