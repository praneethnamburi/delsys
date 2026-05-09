"""End-to-end tests for ``delsys.Log``: loading every fixture format and
verifying counts, modalities, and side-effects (dropped-samples report,
pickle round-trip)."""
import pickle
import shutil

import pytest

from delsys import EMG, EKG, FSR, IMU, Log, VO2Master


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
    assert all(isinstance(b, EMG) for b in lf.emg)


def test_discover164_link_has_vo2_and_hr(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_link.csv", tmp_path)
    assert "VO2" in lf.modalities
    assert "HR" in lf.modalities
    assert len(lf.vo2master) == 1
    assert isinstance(lf.vo2master[0], VO2Master)


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
    if lf2.emg:
        assert isinstance(lf2.emg[0], EMG)


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
    candidate = next(n for n in baseline.sensor_names if len(n.split(' ')) >= 2)
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

def test_add_sensor_group_validates_membership(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover164_mvc.csv", tmp_path)
    valid_num = lf.sensor_numbers[0]
    lf.add_sensor_group("group1", (valid_num,))
    assert lf.sensor_groups["group1"] == (valid_num,)

    with pytest.raises(AssertionError):
        lf.add_sensor_group("bad", (99999,))
