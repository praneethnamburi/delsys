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

- **HDF5 checkpoint follow-ups** (native checkpoint for Discover + EMGworks
  shipped in `_hdf5.py` — see CHANGELOG `[Unreleased]`):
  - **EMGworks `clock_mul != 1` reload.** Currently rejected: the native interp
    grid is fixed at export (`clock_mul=1`), so a clock-shifted reload can't be
    reproduced by reinterpretation. Supporting it would mean storing the raw
    per-channel (timestamp, value) pairs and re-interpolating at load — defeats the
    uniform-grid storage. Revisit only if a sync workflow needs to reload EMGworks
    checkpoints directly at a non-unit clock (gaitmusic resamples post-load today).
  - **Lazy / partial reads + byte-budget LRU.** `read_into` currently hydrates
    the whole file eagerly. For multi-trial / cross-project use (datanest data
    fields), read only the accessed modalities/time-slices and cap resident
    memory with a byte-budget LRU shared across open checkpoints.
  - **Embed dropped-samples + richer header** in the checkpoint (parser writes
    a `_dropped_samples.txt` sidecar today; fold it in for true self-containment).
  - **datanest integration.** Loader pulls per-trial `clock_mul`/`t0` from the
    trial-DB row so `db.add_data_field('delsys', loader, 'trial_id')` holds lazy
    handles that hydrate + align on access.
- **Migrate `_aggregate_bundles` to `pysampled.Data.merge_along_signal_name`**
  once pysampled ships those classmethods (deferred from pysampled
  1.2.0). Status check: pysampled 1.2.0 explicitly pulled them
  before release; revisit when pysampled 1.3.0 plans firm up.
- **In-log noise marking + per-log sidecar + SignalBrowser (target 0.5.0).**
  Designed — see [`.claude-prompts/plan-0.5.0-noise-marking-signalbrowser.md`](.claude-prompts/plan-0.5.0-noise-marking-signalbrowser.md).
  `lf.annotate_noise()` opens a delsys subclass of datanavigator's `SignalBrowser`
  (with a new signal dropdown) to mark noise per signal and drive interactive
  cleaning; windows persist to a per-log `<stem>.delsys-noise` sidecar keyed by
  `"<sensor>.<modality>[.<coord>] | <label>"`; the per-folder
  `delsys_cleaning.json` `noise_event_ref` points at it. Phased: (A) dnav sidebar
  dropdown, (B) delsys sidecar data layer, (C) the subclass + `annotate_noise`.
- **Batch cleaning (`delsys.clean`) follow-ups** (the batch/manifest/docs layer
  shipped in `_clean.py` + `_noise.py` — see CHANGELOG `[Unreleased]`):
  - **Noise-mask consumption — full implementation.** v1 wires the hook: a
    manifest `noise_event_ref` is read as plain JSON and applied with the
    NaN+interpolate policy across *all* modalities before cleaning. Fast-follow:
    (a) per-modality / per-sensor window scoping (the bicep-ACC Event shouldn't
    necessarily blank the EKG); (b) alternative fill policies beyond
    `nan_interp` (hold-last, zero, exclude-from-fit); (c) decide whether the
    masked windows should be *recorded* in the cleaned `.h5` (a `noise_mask`
    dataset) so downstream code can re-exclude them rather than trust the
    interpolation.
  - **Trial-id ↔ checkpoint mapping.** The manifest is keyed by checkpoint stem
    (`Trial_5`), while datanavigator noise Events are keyed by the external
    trial-id tuple (`(2, 14, 17)`), carried in `noise_event_ref.key`. When the
    datanest trial-DB lands, let `clean()` accept a `trial_id` resolver so the
    tuple can be derived from the row instead of hand-written into each entry.
  - **Manifest-driven `target_sr` / `clock_mul`.** `clean()` cleans at one
    `target_sr` (default `TARGET_SR`) and `clock_mul=1`; per-trial alignment
    (from the trial-DB row) is a downstream concern today. Revisit once the
    datanest integration pulls `clock_mul`/`t0` per trial.
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
