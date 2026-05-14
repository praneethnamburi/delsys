# TODO

Open design questions and deferred work for the next release cycle.
Release narrative lives in [`CHANGELOG.md`](CHANGELOG.md).

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

## Followups (target 0.5.0)

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

## Known limitations carrying forward

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
