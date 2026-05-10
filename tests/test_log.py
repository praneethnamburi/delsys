"""End-to-end tests for ``delsys.Log``: loading every fixture format and
verifying counts, modalities, and side-effects (dropped-samples report,
pickle round-trip)."""

import pickle
import shutil

import pytest

from delsys import EKG, EMG, FSR, IMU, Log, VO2Master

# ---------------------------------------------------------------------------
# Per-fixture round-trip — every CSV format yields a Log with expected counts
# ---------------------------------------------------------------------------


def test_log_loads_each_fixture(sample_csv, tmp_path):
    """Every committed fixture round-trips through Log() with the expected
    sensor and signal counts."""
    src, n_sensors, n_signals = sample_csv
    # Copy to tmp so the dropped_samples side-effect file doesn't pollute
    # the fixtures directory.
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    lf = Log(str(dst))
    assert len(lf.sensors) == n_sensors
    assert len(lf.signals) == n_signals
    assert lf.dur > 0
    assert lf.sampling_rates  # non-empty
    assert lf.name == src.stem


# ---------------------------------------------------------------------------
# Modality content per format
# ---------------------------------------------------------------------------


def _load(fixtures_dir, name, tmp_path):
    """Load a fixture into a tmp dir to keep side-effect files isolated."""
    src = fixtures_dir / name
    dst = tmp_path / name
    shutil.copy(src, dst)
    return Log(str(dst))


def test_emgworks_has_emg_acc_gyro(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    mods = lf.modalities
    assert "EMGS" in mods
    assert "ACC" in mods
    assert "GYRO" in mods


def test_discover142_has_emg_and_emg_class_instances(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover142.csv", tmp_path)
    assert "EMGS" in lf.modalities
    assert isinstance(lf.emg, EMG)


def test_discover164_link_has_vo2_and_hr(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_link.csv", tmp_path)
    assert "VO2" in lf.modalities
    assert "HR" in lf.modalities
    assert lf.vo2master is not None
    assert isinstance(lf.vo2master, VO2Master)


def test_discover164_basic_has_no_link_modalities(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_basic.csv", tmp_path)
    assert "VO2" not in lf.modalities
    assert "HR" not in lf.modalities


def test_discover164_mvc_has_emg_and_analog(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    assert "EMGS" in lf.modalities
    assert "Analog" in lf.modalities


# ---------------------------------------------------------------------------
# Side-effect: dropped-samples report
# ---------------------------------------------------------------------------


def test_dropped_samples_report_written_for_discover(fixtures_dir, tmp_path):
    """Discover parsers write a per-channel dropped-sample report next to the CSV."""
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    report = tmp_path / "discover164_mvc_dropped_samples.txt"
    assert report.exists()
    contents = report.read_text()
    # One line per signal.
    assert contents.strip().count("\n") + 1 >= len(lf.signals)
    # Each line has the count-fraction-percent format.
    assert "%" in contents


def test_no_dropped_samples_report_for_emgworks(fixtures_dir, tmp_path):
    """EMGworks parser does not produce a dropped-samples report."""
    _load(fixtures_dir, "emgworks.csv", tmp_path)
    report = tmp_path / "emgworks_dropped_samples.txt"
    assert not report.exists()


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


def test_log_pickle_roundtrip(fixtures_dir, tmp_path):
    """A loaded Log can be pickled and unpickled without losing structure."""
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    blob = pickle.dumps(lf)
    lf2 = pickle.loads(blob)
    assert len(lf2.sensors) == len(lf.sensors)
    assert len(lf2.signals) == len(lf.signals)
    assert lf2.modalities == lf.modalities
    # Class identity is preserved through pickle — bundles still work.
    if lf2.emg is not None:
        assert isinstance(lf2.emg, EMG)


def test_pickle_relabels_legacy_default_labels(fixtures_dir, tmp_path):
    """Pickles produced before delsys 0.1.1 (or by the legacy
    immersionToolbox shim) carry default ``['s0', ...] / ['x']`` labels
    and an empty bundle ``meta``. ``Sensor.__setstate__`` repairs both
    on unpickle, using the Sensor's own attributes as the source of truth.
    """
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)

    # Simulate stale state: wipe every bundle's labels and meta so it
    # looks the way pre-0.1.1 pickles do on disk.
    expected_after = {}  # (sensor_idx, attr) -> (names, coords)
    for s_idx, sensor in enumerate(lf.sensors):
        for attr in ("emg", "ekg", "acc", "gyro", "fsr", "analog", "vo2master", "hrstrap"):
            bundle = getattr(sensor, attr, None)
            if bundle is None:
                continue
            expected_after[(s_idx, attr)] = (
                list(bundle.signal_names),
                list(bundle.signal_coords),
            )
            n = bundle._sig.shape[1] if bundle._sig.ndim == 2 else 1
            bundle.signal_names = [f"s{i}" for i in range(n)]
            bundle.signal_coords = ["x"]
            bundle.meta = {}

    blob = pickle.dumps(lf)
    lf2 = pickle.loads(blob)

    # Every bundle is back to the 0.1.1 convention, and meta['sensor'] is
    # populated so downstream filter / clone calls keep the sensor identity.
    for (s_idx, attr), (names, coords) in expected_after.items():
        bundle = getattr(lf2.sensors[s_idx], attr)
        assert bundle.signal_names == names, f"sensor[{s_idx}].{attr} signal_names not restored"
        assert bundle.signal_coords == coords, f"sensor[{s_idx}].{attr} signal_coords not restored"
        assert bundle.meta.get("sensor") is not None, (
            f"sensor[{s_idx}].{attr} meta['sensor'] not restored"
        )
        assert bundle.meta["sensor"].number == lf2.sensors[s_idx].number


def test_pickle_relabel_preserves_existing_meta_keys(fixtures_dir, tmp_path):
    """Relabel uses ``meta.setdefault('sensor', ...)`` so it must not stomp
    pre-existing meta keys (e.g. EKG's cached rpeak indices)."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.ekg is not None, "discover170 fixture should have an EKG bundle"
    # Reach the per-Sensor EKG view via the source Sensor; the
    # rpeak-meta cache lives on the per-Sensor bundle, and the aggregate
    # rebuilds itself from the Sensors at access time.
    sensor = next(s for s in lf.sensors if hasattr(s, "ekg"))
    sensor.ekg.meta["rpeaks_idx_default"] = [10, 20, 30]
    # Wipe labels to mimic stale-pickle state but keep meta keys.
    sensor.ekg.signal_names = ["s0"]
    sensor.ekg.signal_coords = ["x"]

    lf2 = pickle.loads(pickle.dumps(lf))
    sensor2 = next(s for s in lf2.sensors if hasattr(s, "ekg"))
    ekg2 = sensor2.ekg
    assert ekg2.meta["rpeaks_idx_default"] == [10, 20, 30]
    # Labels were rebuilt from defaults (the exact value depends on the
    # fixture's channelmap; the point is they no longer look defaulted).
    assert ekg2.signal_names != ["s0"]
    assert ekg2.signal_coords == ["ekg"]


# ---------------------------------------------------------------------------
# sensor_name_replace (Discover) — full integration via Log()
# ---------------------------------------------------------------------------


def test_sensor_name_replace_via_log(fixtures_dir, tmp_path):
    """``sensor_name_replace`` rewrites the prefix of combined ``"sensor: signal"``
    column headers, which propagates into ``signal.sensor_name``.

    Real use case: typo fixes during acquisition (e.g. ``'EMG O1' -> 'EMG 01'``).
    The replacement must preserve the parser-expected ``"<modality> <num> ..."``
    layout, so we just swap the trailing token.
    """
    src = fixtures_dir / "discover164_mvc.csv"
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    # Load once to find a real sensor name. Pick one that has the standard
    # multi-token layout so the rewritten version still parses.
    baseline = Log(str(dst))
    candidate = next(n for n in baseline.sensor_names if len(n.split(" ")) >= 2)
    new_name = candidate + "_renamed"  # keeps multi-token structure intact

    lf = Log(str(dst), sensor_name_replace={candidate: new_name})
    assert new_name in lf.sensor_names
    assert candidate not in lf.sensor_names


# ---------------------------------------------------------------------------
# is_resampled / is_shifted / is_adjusted flags
# ---------------------------------------------------------------------------


def test_default_is_not_adjusted(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    assert lf.is_resampled() is False
    assert lf.is_shifted() is False
    assert lf.is_adjusted() is False


def test_clock_mul_marks_resampled(fixtures_dir, tmp_path):
    src = fixtures_dir / "discover164_mvc.csv"
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    lf = Log(str(dst), clock_mul=1.001)
    assert lf.is_resampled()
    assert lf.is_adjusted()


def test_t0_marks_shifted(fixtures_dir, tmp_path):
    src = fixtures_dir / "discover164_mvc.csv"
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    lf = Log(str(dst), t0=10.0)
    assert lf.is_shifted()
    assert lf.is_adjusted()


# ---------------------------------------------------------------------------
# add_sensor_group rejects unknown sensor numbers
# ---------------------------------------------------------------------------


def test_log_load_with_synthetic_drift_normalizes(fixtures_dir, tmp_path, monkeypatch):
    """Inject 1-sample drift on one ACC sub-channel inside the parser; the
    Log() constructor must call ``_normalize_signal_lengths`` before
    grouping signals into Sensors so the same-modality length assert
    survives drift."""
    from delsys import _parse, log

    real_parser = _parse._parse_dataframe_emgworks

    def wrapped(*args, **kwargs):
        t_min, t_max, sr_orig, signals = real_parser(*args, **kwargs)
        # Find one ACC channel on some sensor and shorten it by 1 sample;
        # within-Sensor stacking would otherwise trip the same-len assert
        # because ACC is a 3-channel modality (X / Y / Z).
        for i, sig in enumerate(signals):
            if sig.modality == "ACC":
                signals[i] = sig._clone(sig()[:-1])
                break
        return t_min, t_max, sr_orig, signals

    monkeypatch.setattr(log, "_parse_dataframe_emgworks", wrapped)

    src = fixtures_dir / "emgworks.csv"
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    # Without normalization, this would trip Sensor.__init__'s
    # `len(np.unique([len(s) for s in this_signals])) == 1` assert.
    lf = Log(str(dst))
    acc_lens = {len(s) for s in lf.signals if s.modality == "ACC"}
    assert len(acc_lens) == 1, f"ACC lengths still drifting: {acc_lens}"


def test_add_sensor_group_validates_membership(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    valid_num = lf.sensor_numbers[0]
    lf.add_sensor_group("group1", (valid_num,))
    assert lf.sensor_groups["group1"] == (valid_num,)

    with pytest.raises(AssertionError):
        lf.add_sensor_group("bad", (99999,))


# ---------------------------------------------------------------------------
# 0.1.1: bundle metadata propagation through Log()
# ---------------------------------------------------------------------------


def test_log_acc_bundle_has_meaningful_signal_names(fixtures_dir, tmp_path):
    """Loaded ACC bundles carry a non-default signal_name (one entry) and
    the canonical x/y/z signal_coords."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    assert lf.acc is not None, "emgworks fixture should have ACC bundles"
    bundle = lf.acc.split_by_signal_name()[0]
    assert bundle.signal_coords == ["x", "y", "z"]
    assert len(bundle.signal_names) == 1
    # No channelmap was supplied — fall back to ``ch<N>`` per the plan.
    assert bundle.signal_names[0].startswith("ch")


def test_log_acc_indexable_by_signal_name(fixtures_dir, tmp_path):
    """ACC bundle supports pysampled label-based indexing via its signal_name."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    bundle = lf.acc.split_by_signal_name()[0]
    name = bundle.signal_names[0]
    sub = bundle[name]
    assert sub().shape == bundle().shape


def test_log_acc_magnitude_returns_global_l2(fixtures_dir, tmp_path):
    """Post-0.1.1: acc.magnitude() collapses x/y/z to a single L2 column,
    not three independent per-axis abs values."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    bundle = lf.acc.split_by_signal_name()[0]
    mag = bundle.magnitude()
    assert mag().ndim == 2
    assert mag().shape[1] == 1
    assert mag.signal_coords == ["mag"]


def test_log_emg_bundle_has_meaningful_signal_names(fixtures_dir, tmp_path):
    """EMG bundle carries a non-default signal_name (one entry per channel)
    and the modality coord ['emg']."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    assert lf.emg is not None
    bundle = lf.emg.split_by_signal_name()[0]
    assert bundle.signal_coords == ["emg"]
    # Single-channel EMGS gets one entry; multi-channel bundles get one per channel.
    assert len(bundle.signal_names) == bundle.shape[1]
    # Default 's0' / 's1' fallback would be a regression — make sure no entry
    # starts with the bare ``'s'`` + digit pattern from pysampled defaults.
    for name in bundle.signal_names:
        assert not (name.startswith("s") and name[1:].isdigit())


# ---------------------------------------------------------------------------
# 0.2.0: Log.<modality> accessors return a single aggregated bundle (or None)
# ---------------------------------------------------------------------------


def test_log_emg_returns_single_aggregate_bundle(fixtures_dir, tmp_path):
    """``lf.emg`` is a single :class:`EMG` instance, not a list."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    assert isinstance(lf.emg, EMG)
    n_emg_sensors = sum(1 for s in lf.sensors if hasattr(s, "emg"))
    # n_signals == sum of per-sensor channels; for single-channel EMGS in
    # the emgworks fixture this equals the number of EMG sensors.
    assert lf.emg().shape[1] == n_emg_sensors


def test_log_emg_signal_names_concatenated_in_sensor_order(fixtures_dir, tmp_path):
    """Aggregate signal_names are concatenated in lf.sensors order."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    expected = []
    for s in lf.sensors:
        if hasattr(s, "emg"):
            expected.extend(s.emg.signal_names)
    assert lf.emg.signal_names == expected


def test_log_emg_split_by_signal_name_recovers_per_sensor_count(fixtures_dir, tmp_path):
    """``split_by_signal_name`` recovers the old list view."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    parts = lf.emg.split_by_signal_name()
    n_emg_sensors = sum(1 for s in lf.sensors if hasattr(s, "emg"))
    # emgworks fixture is single-channel EMGS, so per-name equals per-sensor.
    assert len(parts) == n_emg_sensors


def test_log_acc_aggregate_signal_coords(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    assert lf.acc.signal_coords == ["x", "y", "z"]


def test_log_acc_index_by_location_returns_one_sensor(fixtures_dir, tmp_path):
    """Indexing the aggregate ACC by signal_name picks one sensor's data."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.acc is not None
    name = lf.acc.signal_names[0]
    sub = lf.acc[name]
    # Single-name slice has 3 columns (x/y/z) for one sensor.
    assert sub().shape[1] == 3
    assert sub.signal_names == [name]


def test_log_returns_none_for_absent_modality(fixtures_dir, tmp_path):
    """An aggregate accessor returns ``None`` (not ``[]``) when no sensor
    has that modality. Use ``vo2master`` on a basic Discover fixture —
    no link devices means no VO2."""
    lf = _load(fixtures_dir, "discover164_basic.csv", tmp_path)
    assert "VO2" not in lf.modalities
    assert lf.vo2master is None
    assert lf.hrstrap is None


def test_log_aggregate_meta_sensors_length_matches_signal_names(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    assert len(lf.emg.meta["sensors"]) == len(lf.emg.signal_names)
    assert len(lf.acc.meta["sensors"]) == len(lf.acc.signal_names)


def test_log_pickle_roundtrip_with_aggregate(fixtures_dir, tmp_path):
    """Aggregate accessor still works after pickle round-trip."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    pre_names = list(lf.emg.signal_names)
    blob = pickle.dumps(lf)
    lf2 = pickle.loads(blob)
    assert isinstance(lf2.emg, EMG)
    assert lf2.emg.signal_names == pre_names


def test_log_aggregate_magnitude_per_sensor(fixtures_dir, tmp_path):
    """``lf.acc.magnitude()`` returns one mag column per ACC sensor."""
    lf = _load(fixtures_dir, "emgworks.csv", tmp_path)
    n_acc_sensors = sum(1 for s in lf.sensors if hasattr(s, "acc"))
    mag = lf.acc.magnitude()
    assert mag.signal_coords == ["mag"]
    assert mag().shape[1] == n_acc_sensors
