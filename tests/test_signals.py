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
    return IMU(
        sig,
        sr=200.0,
        t0=0.0,
        meta={"sensor": si},
        signal_names=["test"],
        signal_coords=["x", "y", "z"],
    )


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
    return FSR(
        sig,
        sr=148.0,
        t0=0.0,
        meta={"sensor": si},
        signal_names=["a", "b", "c", "d"],
        signal_coords=["fsr"],
    )


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
    return VO2Master(
        sig,
        sr=1.0,
        t0=0.0,
        meta={"sensor": si},
        signal_names=[
            "resp_rate",
            "tidal_vol",
            "ventilation",
            "feo2",
            "vo2_absolute",
            "ambient_pressure",
            "flow_sensor",
            "oxygen_sensor_humidity",
        ],
        signal_coords=["value"],
    )


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


# ---------------------------------------------------------------------------
# Sensor.__init__ — per-modality bundle labels (signal_names / signal_coords)
# ---------------------------------------------------------------------------

from delsys import EKG, EMG, Sensor  # noqa: E402  (kept near the helpers it serves)
import pysampled  # noqa: E402


def _build_sensor(modality, location, subchannels=None, n_samples=20, number=1, sensor_type=None):
    """Construct a Sensor with synthetic Signals for one modality."""
    from delsys._constants import SUBCHANNEL_MAP

    subs = subchannels if subchannels is not None else SUBCHANNEL_MAP[modality]
    si = SensorInfo(
        name="testsensor",
        modalities={modality},
        number=number,
        type_sensorlog=sensor_type,
        lrc="L",
        location=location,
    )
    sigs = []
    for sub in subs:
        s = Signal(
            np.zeros(n_samples),
            sr=200.0,
            t0=0.0,
            meta={"sensor": si, "modality": modality, "subchannel": sub},
        )
        sigs.append(s)
    return Sensor(si, sigs)


def test_sensor_acc_labels_use_trimmed_location():
    sensor = _build_sensor("ACC", "LBicep")
    assert sensor.acc.signal_names == ["LBicep"]
    assert sensor.acc.signal_coords == ["x", "y", "z"]


def test_sensor_gyro_labels_use_trimmed_location():
    sensor = _build_sensor("GYRO", "RForearm")
    assert sensor.gyro.signal_names == ["RForearm"]
    assert sensor.gyro.signal_coords == ["x", "y", "z"]


def test_sensor_acc_no_channelmap_falls_back_to_ch_number():
    sensor = _build_sensor("ACC", None, number=7)
    assert sensor.acc.signal_names == ["ch7"]
    assert sensor.acc.signal_coords == ["x", "y", "z"]


def test_sensor_acc_strips_parenthetical_in_location():
    sensor = _build_sensor("ACC", "LPinkyReach (LPalmaris Longus)")
    assert sensor.acc.signal_names == ["LPinkyReach"]


def test_sensor_emgs_label():
    sensor = _build_sensor("EMGS", "LBrachialis")
    assert sensor.emg.signal_names == ["LBrachialis"]
    assert sensor.emg.signal_coords == ["emg"]


def test_sensor_emgd_two_channel_letter_fallback():
    sensor = _build_sensor("EMGD", "LBicep")
    assert sensor.emg.signal_names == ["LBicep_A", "LBicep_B"]
    assert sensor.emg.signal_coords == ["emg"]


def test_sensor_emgq_parsed_letters():
    sensor = _build_sensor(
        "EMGQ", "LForearmExtensors (A-Index, B-Middle, C-Ring, D-Little)"
    )
    assert sensor.emg.signal_names == [
        "LForearmExtensors_Index",
        "LForearmExtensors_Middle",
        "LForearmExtensors_Ring",
        "LForearmExtensors_Little",
    ]
    assert sensor.emg.signal_coords == ["emg"]


def test_sensor_emgq_fallback_when_no_parenthetical():
    sensor = _build_sensor("EMGQ", "LForearm")
    assert sensor.emg.signal_names == [
        "LForearm_A",
        "LForearm_B",
        "LForearm_C",
        "LForearm_D",
    ]


def test_sensor_fsr_parsed_positions():
    sensor = _build_sensor("FSR", "LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)")
    assert sensor.fsr.signal_names == [
        "LFoot_Heel",
        "LFoot_OuterEdge",
        "LFoot_Ball",
        "LFoot_Toe",
    ]
    assert sensor.fsr.signal_coords == ["fsr"]


def test_sensor_fsr_fallback_letters_when_unparsed():
    sensor = _build_sensor("FSR", "LFoot")
    assert sensor.fsr.signal_names == ["LFoot_A", "LFoot_B", "LFoot_C", "LFoot_D"]
    assert sensor.fsr.signal_coords == ["fsr"]


def test_sensor_ekg_labels():
    sensor = _build_sensor("EKG", "Chest")
    assert sensor.ekg.signal_names == ["Chest"]
    assert sensor.ekg.signal_coords == ["ekg"]


def test_sensor_analog_labels_and_carries_meta():
    # Use a custom 1-channel Sensor; Analog SUBCHANNEL_MAP is ('A',).
    sensor = _build_sensor("Analog", "Sync")
    assert isinstance(sensor.analog, pysampled.Data)
    assert sensor.analog.signal_names == ["Sync"]
    assert sensor.analog.signal_coords == ["analog"]
    # Plan calls out the pre-existing meta-loss bug — Analog should now carry
    # the parent SensorInfo through `meta['sensor']`.
    assert sensor.analog.meta is not None
    assert sensor.analog.meta.get("sensor") is sensor.analog.meta["sensor"]
    assert sensor.analog.meta["sensor"].location == "Sync"


def test_sensor_vo2master_labels():
    # Production drops the BreathingCycle column at parse time (see
    # _parse.py:605–606), so the bundle has 8 channels, not 9.
    vo2_subs = (
        "Resp.Rate",
        "TidalVol.",
        "Ventilation(L/min)",
        "FeO2(%)",
        "VO2Absolute",
        "AmbientPressure",
        "FlowSensor",
        "OxygenSensor",
    )
    sensor = _build_sensor("VO2", None, subchannels=vo2_subs, number=900)
    assert sensor.vo2master.signal_names == [
        "resp_rate",
        "tidal_vol",
        "ventilation",
        "feo2",
        "vo2_absolute",
        "ambient_pressure",
        "flow_sensor",
        "oxygen_sensor_humidity",
    ]
    assert sensor.vo2master.signal_coords == ["value"]


def test_sensor_hrstrap_labels_and_carries_meta():
    sensor = _build_sensor("HR", None, number=901)
    assert sensor.hrstrap.signal_names == ["heart_rate"]
    assert sensor.hrstrap.signal_coords == ["bpm"]
    # HR also gains meta=sensor_meta in 0.1.1 (paired with Analog fix).
    assert sensor.hrstrap.meta is not None
    assert sensor.hrstrap.meta["sensor"].number == 901


def test_sensor_emg_indexable_by_signal_name():
    """Bundles with meaningful signal_names support pysampled label indexing."""
    sensor = _build_sensor("EMGS", "LBrachialis")
    emg = sensor.emg
    # `emg["LBrachialis"]` is the user-facing label lookup pysampled 1.1.0+ provides.
    sub = emg["LBrachialis"]
    assert isinstance(sub, EMG)
    assert sub().shape == emg().shape


def test_sensor_acc_indexable_by_location():
    sensor = _build_sensor("ACC", "LBicep")
    acc = sensor.acc
    sub = acc[acc.signal_names[0]]
    assert sub().shape == acc().shape


# ---------------------------------------------------------------------------
# Sub-channel views — IMU.x/y/z, FSR.a..d, VO2Master.* inherit parent labels
# ---------------------------------------------------------------------------


def test_imu_axis_inherits_parent_signal_name():
    """IMU.x/y/z share the parent's signal_names and pin signal_coords to one axis."""
    sensor = _build_sensor("ACC", "LBicep")
    acc = sensor.acc
    assert acc.x.signal_names == ["LBicep"]
    assert acc.x.signal_coords == ["x"]
    assert acc.y.signal_coords == ["y"]
    assert acc.z.signal_coords == ["z"]


def test_imu_magnitude_collapses_to_single_signal():
    """After labels propagate, IMU.magnitude() returns shape (n, 1) not (n, 3)."""
    sensor = _build_sensor("ACC", "LBicep", n_samples=100)
    mag = sensor.acc.magnitude()
    assert mag().ndim == 2
    assert mag().shape[1] == 1
    # pysampled 1.2.0 sets signal_coords=['mag'] on magnitude output.
    assert mag.signal_coords == ["mag"]


def test_fsr_channel_inherits_parent_coords():
    """FSR.a..d carry one signal_name (the parent's i-th) and the parent's coords."""
    sensor = _build_sensor("FSR", "LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)")
    fsr = sensor.fsr
    assert fsr.a.signal_names == ["LFoot_Heel"]
    assert fsr.b.signal_names == ["LFoot_OuterEdge"]
    assert fsr.c.signal_names == ["LFoot_Ball"]
    assert fsr.d.signal_names == ["LFoot_Toe"]
    assert fsr.a.signal_coords == fsr.signal_coords  # ['fsr']


def test_fsr_channel_inherits_fallback_letters():
    sensor = _build_sensor("FSR", "LFoot")
    fsr = sensor.fsr
    assert fsr.a.signal_names == ["LFoot_A"]
    assert fsr.d.signal_names == ["LFoot_D"]


# ---------------------------------------------------------------------------
# 0.2.0: ``.sensors`` (plural) property unifies per-Sensor and aggregate views
# ---------------------------------------------------------------------------


def test_fsr_a_raises_on_multi_sensor_aggregate():
    """FSR.a..d are positional and only meaningful on a 4-channel per-Sensor
    view. On a multi-sensor aggregate (8 / 12 / ... channels) they raise
    with a hint pointing at split_by_signal_name() / name lookup."""
    si_l = _sensor_info({"FSR"}, number=1, location="LFoot")
    si_r = _sensor_info({"FSR"}, number=2, location="RFoot")
    sig = np.zeros((100, 8))
    agg = FSR(
        sig,
        sr=120.0,
        axis=0,
        t0=0.0,
        meta={"sensors": [si_l] * 4 + [si_r] * 4},
        signal_names=[f"LFoot_{k}" for k in "ABCD"] + [f"RFoot_{k}" for k in "ABCD"],
        signal_coords=["fsr"],
    )
    with pytest.raises(NotImplementedError, match="per-Sensor"):
        _ = agg.a


def test_fsr_a_works_on_per_sensor_view():
    """Pin 0.1.1 behavior: on a 4-channel per-Sensor FSR, .a/.b/.c/.d
    pick one column each, with the parent's signal_coords inherited."""
    sensor = _build_sensor("FSR", "LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)")
    fsr = sensor.fsr
    assert fsr.a.signal_names == ["LFoot_Heel"]
    assert fsr.a.signal_coords == ["fsr"]


def test_imu_axis_works_on_aggregate_shape():
    """``imu.x`` on a multi-sensor aggregate IMU returns one column per
    sensor (all of them on the X axis), with signal_coords=['x'] and
    signal_names preserved across sensors."""
    si_l = _sensor_info({"ACC"}, number=1, location="L")
    si_r = _sensor_info({"ACC"}, number=2, location="R")
    sig = np.column_stack(
        [
            np.full(100, 1.0),  # L_x
            np.full(100, 2.0),  # L_y
            np.full(100, 3.0),  # L_z
            np.full(100, 11.0),  # R_x
            np.full(100, 12.0),  # R_y
            np.full(100, 13.0),  # R_z
        ]
    )
    agg = IMU(
        sig,
        sr=120.0,
        axis=0,
        t0=0.0,
        meta={"sensors": [si_l, si_r]},
        signal_names=["L", "R"],
        signal_coords=["x", "y", "z"],
    )
    x = agg.x
    assert x().shape == (100, 2)
    assert x.signal_names == ["L", "R"]
    assert x.signal_coords == ["x"]
    # Columns are L_x then R_x — taken from the agg's name-major layout.
    assert np.allclose(x()[:, 0], 1.0)
    assert np.allclose(x()[:, 1], 11.0)


def test_imu_axis_per_sensor_unchanged():
    """Pin the 0.1.1 per-Sensor behavior: imu.x on a single-sensor IMU
    returns a single-column IMU with signal_coords=['x'] and the
    per-Sensor signal_name."""
    sensor = _build_sensor("ACC", "LBicep")
    acc = sensor.acc
    x = acc.x
    assert x.signal_coords == ["x"]
    assert x.signal_names == ["LBicep"]


def test_bundle_sensors_property_per_sensor():
    """Per-Sensor bundles carry meta['sensor'] (singular). The new ``.sensors``
    plural property returns a length-1 list with that record so user code can
    treat both views uniformly."""
    sensor = _build_sensor("ACC", "LBicep")
    acc = sensor.acc
    assert acc.sensor is not None
    assert acc.sensors == [acc.sensor]


def test_bundle_sensors_property_aggregate():
    """Aggregate bundles carry meta['sensors'] (plural). ``.sensors`` returns
    the list aligned with signal_names."""
    si_l = _sensor_info({"ACC"}, number=1, location="L")
    si_r = _sensor_info({"ACC"}, number=2, location="R")
    sig = np.zeros((100, 6))
    agg = IMU(
        sig,
        sr=120.0,
        axis=0,
        t0=0.0,
        meta={"sensors": [si_l, si_r]},
        signal_names=["L", "R"],
        signal_coords=["x", "y", "z"],
    )
    assert agg.sensors == [si_l, si_r]
    # ``sensor`` (singular) returns None on aggregate views since there's no
    # one-to-one mapping.
    assert agg.sensor is None


def test_bundle_sensors_property_on_emg():
    """``.sensors`` lives on every modality bundle, not just IMU."""
    sensor = _build_sensor("EMGS", "LBrachialis")
    emg = sensor.emg
    assert emg.sensors == [emg.sensor]


def test_bundle_sensors_property_on_ekg():
    sensor = _build_sensor("EKG", "Chest")
    ekg = sensor.ekg
    assert ekg.sensors == [ekg.sensor]


def test_bundle_sensors_property_on_fsr():
    sensor = _build_sensor("FSR", "LFoot")
    fsr = sensor.fsr
    assert fsr.sensors == [fsr.sensor]


def test_bundle_sensors_property_on_vo2master():
    vo2_subs = (
        "Resp.Rate",
        "TidalVol.",
        "Ventilation(L/min)",
        "FeO2(%)",
        "VO2Absolute",
        "AmbientPressure",
        "FlowSensor",
        "OxygenSensor",
    )
    sensor = _build_sensor("VO2", None, subchannels=vo2_subs, number=900)
    vo2 = sensor.vo2master
    assert vo2.sensors == [vo2.sensor]


def test_vo2master_channel_inherits_parent_coords():
    """VO2Master.* carry the parent's i-th signal name and the parent's coords."""
    vo2_subs = (
        "Resp.Rate",
        "TidalVol.",
        "Ventilation(L/min)",
        "FeO2(%)",
        "VO2Absolute",
        "AmbientPressure",
        "FlowSensor",
        "OxygenSensor",
    )
    sensor = _build_sensor("VO2", None, subchannels=vo2_subs, number=900)
    vo2 = sensor.vo2master
    assert vo2.rr.signal_names == ["resp_rate"]
    assert vo2.td.signal_names == ["tidal_vol"]
    assert vo2.vent.signal_names == ["ventilation"]
    assert vo2.Feo2.signal_names == ["feo2"]
    assert vo2.vo2.signal_names == ["vo2_absolute"]
    assert vo2.ap.signal_names == ["ambient_pressure"]
    assert vo2.fl.signal_names == ["flow_sensor"]
    assert vo2.o2_hum.signal_names == ["oxygen_sensor_humidity"]
    # All keep the parent's coords (['value']).
    assert vo2.rr.signal_coords == vo2.signal_coords
    assert vo2.o2_hum.signal_coords == vo2.signal_coords
