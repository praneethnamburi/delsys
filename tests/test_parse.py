"""Tests for ``delsys._parse``: CSV header parsing, version detection, signal-name parsing."""
import pytest

from delsys._constants import HR_SENSOR_NUM, VO2_SENSOR_NUM
from delsys._parse import (
    _detect_parser,
    _fix_corrupted_sensor_names,
    _parse_hdr,
    _parse_sig_name,
)


# ---------------------------------------------------------------------------
# _parse_hdr — application & version detection across all 7 fixture formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("emgworks.csv", {"application": "EMGworks", "skiprows": 0}),
        ("discover142.csv", {"application": "Trigno Discover", "application_version": "1.4.2", "skiprows": 6}),
        ("discover150.csv", {"application": "Trigno Discover", "application_version": "1.5.0", "skiprows": 7}),
        ("discover164_link.csv", {"application": "Trigno Discover", "application_version": "1.6.4", "skiprows": 7}),
        ("discover164_basic.csv", {"application": "Trigno Discover", "application_version": "1.6.4", "skiprows": 7}),
        ("discover164_mvc.csv", {"application": "Trigno Discover", "application_version": "1.6.4", "skiprows": 7}),
        ("discover170.csv", {"application": "Trigno Discover", "application_version": "1.7.0", "skiprows": 7}),
    ],
)
def test_parse_hdr_application_and_version(fixtures_dir, fixture_name, expected):
    hdr = _parse_hdr(str(fixtures_dir / fixture_name))
    for key, value in expected.items():
        assert hdr[key] == value, f"{fixture_name}: hdr[{key!r}] = {hdr.get(key)!r}, expected {value!r}"


def test_parse_hdr_emgworks_minimal_keys(fixtures_dir):
    """EMGworks files don't carry version/duration in the header."""
    hdr = _parse_hdr(str(fixtures_dir / "emgworks.csv"))
    assert hdr["application"] == "EMGworks"
    assert "duration_s" not in hdr
    assert "application_version" not in hdr


def test_parse_hdr_discover_carries_metadata(fixtures_dir):
    """Discover files carry datetime, duration, sensor_signal_names, and sensor_name_mode."""
    hdr = _parse_hdr(str(fixtures_dir / "discover164_mvc.csv"))
    assert hdr["application"] == "Trigno Discover"
    assert isinstance(hdr["datetime"], str) and hdr["datetime"]
    assert isinstance(hdr["duration_s"], float) and hdr["duration_s"] > 0
    assert isinstance(hdr["sensor_signal_names"], list) and len(hdr["sensor_signal_names"]) > 0
    assert isinstance(hdr["sensor_name_mode"], dict)


# ---------------------------------------------------------------------------
# _parse_sig_name — column-header → SigInfoDelsys
# ---------------------------------------------------------------------------

@pytest.fixture
def trigno_target_sr():
    """A target_sr dict that covers every modality the parser may produce."""
    return {
        "EMGS": 2000.0, "EMGD": 2000.0, "EMGQ": 2000.0,
        "ACC": 200.0, "GYRO": 200.0,
        "FSR": 200.0, "Analog": None, "EKG": 1259.0,
        "VO2": None, "HR": None,
    }


@pytest.mark.parametrize(
    "header, expected_modality, expected_subchannel",
    [
        ("EMG 01 04498 (54717): EMG 1 (mV)", "EMGS", "A"),
        ("Analog 13 00125 (62202): Analog 1 (V)", "Analog", "A"),
    ],
)
def test_parse_sig_name_discover_emg_and_analog(header, expected_modality, expected_subchannel, trigno_target_sr):
    info = _parse_sig_name(header, "Trigno Discover", trigno_target_sr)
    assert info.modality == expected_modality
    assert info.subchannel == expected_subchannel
    assert info.sensor_number == 1 if expected_modality == "EMGS" else info.sensor_number == 13


def test_parse_sig_name_vo2_link_uses_synthetic_sensor_number(trigno_target_sr):
    info = _parse_sig_name(
        "VO2 Master  (0): Resp. Rate (BrPM)",
        "Trigno Discover", trigno_target_sr,
    )
    assert info.modality == "VO2"
    assert info.sensor_number == VO2_SENSOR_NUM


def test_parse_sig_name_hr_link_uses_synthetic_sensor_number(trigno_target_sr):
    info = _parse_sig_name(
        "HR Strap  (0): HR (BPM)",
        "Trigno Discover", trigno_target_sr,
    )
    assert info.modality == "HR"
    assert info.sensor_number == HR_SENSOR_NUM


def test_parse_sig_name_emg_quattro_tag(trigno_target_sr):
    info = _parse_sig_name(
        "Quattro 02 12345 (67890): EMG 1 (mV)",
        "Trigno Discover", trigno_target_sr,
    )
    assert info.modality == "EMGQ"


def test_parse_sig_name_emg_duo_tag(trigno_target_sr):
    info = _parse_sig_name(
        "Duo 03 12345 (67890): EMG 1 (mV)",
        "Trigno Discover", trigno_target_sr,
    )
    assert info.modality == "EMGD"


# ---------------------------------------------------------------------------
# _detect_parser — picks the right per-format dataframe parser
# ---------------------------------------------------------------------------

def test_detect_parser_emgworks(fixtures_dir):
    hdr = _parse_hdr(str(fixtures_dir / "emgworks.csv"))
    assert _detect_parser(hdr, time_names=[]) == "emgworks"


def test_detect_parser_discover_basic(fixtures_dir):
    hdr = _parse_hdr(str(fixtures_dir / "discover164_basic.csv"))
    # No VO2/HR columns and no time-series columns -> basic parser.
    assert _detect_parser(hdr, time_names=[]) == "discover_basic"


def test_detect_parser_discover_link_with_timestamps(fixtures_dir):
    hdr = _parse_hdr(str(fixtures_dir / "discover164_link.csv"))
    # discover164_link.csv has VO2/HR + Time Series columns.
    time_names = [c for c in hdr["sensor_signal_names"] if "Time Series" in c]
    assert len(time_names) > 0
    assert _detect_parser(hdr, time_names=time_names) == "discover_link"


def test_detect_parser_link_without_timestamps_raises(fixtures_dir):
    """When VO2/HR columns are present but Time Series is not, parser raises
    a clear error (link data needs timestamps for resampling)."""
    hdr = _parse_hdr(str(fixtures_dir / "discover164_link.csv"))
    with pytest.raises(Exception, match="Time Series"):
        _detect_parser(hdr, time_names=[])


# ---------------------------------------------------------------------------
# _fix_corrupted_sensor_names — repairs misspellings via prefix substitution
# ---------------------------------------------------------------------------

def test_fix_corrupted_sensor_names_prefix_swap():
    sig_names = ["EMG 01 corrupt: EMG 1", "EMG 02 ok: EMG 1"]
    out = _fix_corrupted_sensor_names(sig_names, {"EMG 01 corrupt": "EMG 01 fixed"})
    assert out == ["EMG 01 fixed: EMG 1", "EMG 02 ok: EMG 1"]


def test_fix_corrupted_sensor_names_no_match_passthrough():
    sig_names = ["EMG 01: EMG 1", "EMG 02: EMG 1"]
    out = _fix_corrupted_sensor_names(sig_names, {"DOES NOT MATCH": "x"})
    assert out == sig_names


def test_fix_corrupted_sensor_names_empty_replace_dict_passthrough():
    sig_names = ["EMG 01: EMG 1", "EMG 02: EMG 1"]
    assert _fix_corrupted_sensor_names(sig_names, {}) == sig_names
