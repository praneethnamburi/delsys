"""Synthetic-array tests for signal classes: ``Signal``, ``IMU``, ``FSR``,
``VO2Master``. No fixture files required."""

import numpy as np
import pytest

from delsys import FSR, IMU, SensorInfo, Signal, VO2Master

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sensor_info(modality_set, number=1, name="test", lrc="C", location="Hip"):
    """Build a synthetic SensorInfo for testing."""
    return SensorInfo(
        name=name,
        modalities=modality_set,
        number=number,
        type_sensorlog=None,
        lrc=lrc,
        location=location,
    )


def _const_columns(n_samples, n_channels):
    """Stack columns where column ``i`` is filled with the value ``i + 1``."""
    return np.column_stack([np.full(n_samples, i + 1.0) for i in range(n_channels)])


# ---------------------------------------------------------------------------
# IMU — three-axis access (X, Y, Z map to columns 0, 1, 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def imu():
    si = _sensor_info({"ACC"})
    sig = _const_columns(1000, 3)  # column 0=1.0, 1=2.0, 2=3.0
    return IMU(sig, sr=200.0, t0=0.0, meta={"sensor": si})


def test_imu_shape(imu):
    assert imu.shape == (1000, 3)


@pytest.mark.parametrize("axis_name, expected_value", [("x", 1.0), ("y", 2.0), ("z", 3.0)])
def test_imu_axis_maps_to_correct_column(imu, axis_name, expected_value):
    axis = getattr(imu, axis_name)
    assert np.allclose(axis(), expected_value)


def test_imu_axis_returns_imu_instance(imu):
    """Axis access goes through ``_clone`` so the result is still an IMU
    (carrying ``.sensor`` metadata)."""
    assert isinstance(imu.x, IMU)
    assert imu.x.sensor is imu.sensor


# ---------------------------------------------------------------------------
# FSR — four-channel access (A, B, C, D map to columns 0, 1, 2, 3)
# ---------------------------------------------------------------------------


@pytest.fixture
def fsr():
    si = _sensor_info({"FSR"})
    sig = _const_columns(500, 4)  # column 0=1.0, ..., 3=4.0
    return FSR(sig, sr=148.0, t0=0.0, meta={"sensor": si})


def test_fsr_shape(fsr):
    assert fsr.shape == (500, 4)


@pytest.mark.parametrize(
    "channel, expected",
    [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)],
)
def test_fsr_channel_maps_to_correct_column(fsr, channel, expected):
    ch = getattr(fsr, channel)
    assert np.allclose(ch(), expected)


def test_fsr_channel_returns_fsr_instance(fsr):
    assert isinstance(fsr.a, FSR)
    assert fsr.a.sensor is fsr.sensor


# ---------------------------------------------------------------------------
# VO2Master — eight-column ordering (regression for the original copy-paste bug)
# ---------------------------------------------------------------------------


@pytest.fixture
def vo2():
    si = _sensor_info({"VO2"}, number=900, location=None)
    sig = _const_columns(50, 8)  # column i has value i+1
    return VO2Master(sig, sr=1.0, t0=0.0, meta={"sensor": si})


def test_vo2master_shape(vo2):
    assert vo2.shape == (50, 8)


@pytest.mark.parametrize(
    "attr, expected_column_value",
    [
        ("rr", 1.0),  # column 0: respiration rate
        ("td", 2.0),  # column 1: tidal volume
        ("vent", 3.0),  # column 2: ventilation
        ("Feo2", 4.0),  # column 3: FeO2
        ("vo2", 5.0),  # column 4: VO2 absolute
        ("ap", 6.0),  # column 5: ambient pressure
        ("fl", 7.0),  # column 6: flow sensor
        ("o2_hum", 8.0),  # column 7: O2 sensor humidity
    ],
)
def test_vo2master_columns_in_order(vo2, attr, expected_column_value):
    """Regression: VO2Master used to map every property to column 1; this
    locks in the correct 0..7 mapping per SUBCHANNEL_MAP['VO2']."""
    val = getattr(vo2, attr)
    assert np.allclose(
        val(), expected_column_value
    ), f"VO2Master.{attr} should be {expected_column_value}, got {val().mean()}"


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("respiration_rate", "rr"),
        ("tidal_vol", "td"),
        ("ventilation", "vent"),
        ("VO2_absolute", "vo2"),
        ("ambient_pressure", "ap"),
        ("flow_sensor", "fl"),
        ("oxygen_sensor_humidity", "o2_hum"),
    ],
)
def test_vo2master_aliases_match_canonical(vo2, alias, canonical):
    """Each VO2 channel has a short and a verbose name pointing at the same column."""
    assert np.allclose(getattr(vo2, alias)(), getattr(vo2, canonical)())


# ---------------------------------------------------------------------------
# Signal — sensor metadata round-trip + clone
# ---------------------------------------------------------------------------


def test_signal_records_sensor_modality_subchannel():
    si = _sensor_info({"EMGS"})
    sig = np.linspace(0, 1, 500)
    s = Signal(sig, sr=2000.0, t0=0.5, meta={"sensor": si, "modality": "EMGS", "subchannel": "A"})
    assert s.sensor is si
    assert s.modality == "EMGS"
    assert s.subchannel == "A"
    # `sensor_name` historically stores the modality-as-attr, not the sensor's name.
    assert s.sensor_name == "emg"


def test_signal_clone_preserves_metadata():
    """Cloning a Signal (e.g. via filtering) keeps sensor/modality/subchannel."""
    si = _sensor_info({"EMGS"})
    s = Signal(
        np.linspace(0, 1, 500), 2000.0, meta={"sensor": si, "modality": "EMGS", "subchannel": "A"}
    )
    s2 = s.lowpass(50)  # filtering goes through _clone
    assert isinstance(s2, Signal)
    assert s2.sensor is si
    assert s2.modality == "EMGS"
    assert s2.subchannel == "A"
