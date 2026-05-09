"""Module-level constants for the delsys package.

The metadata namedtuples (``SensorLog``, ``SensorInfo``, ``SigInfoDelsys``)
that used to live here moved to :mod:`delsys._metadata` to avoid a circular
import between :mod:`delsys.signals` and :mod:`delsys.sensor`.
"""
from typing import Dict, Optional, Tuple


#: Default per-modality target sampling rate (Hz) used to normalize raw signals
#: during loading. Override by passing ``target_sr=`` to :class:`delsys.Log`.
#:
#: Keys are normalized modality tags (see ``SUBCHANNEL_MAP``). A value of
#: ``None`` for a modality means "skip resampling" — currently honored by the
#: Discover-basic and Discover-link Trigno-base parsers; not yet by the
#: EMGworks parser or by link devices (see ``TODO.md``).
TARGET_SR: Dict[str, Optional[float]] = {
    "EMGS": 1920, "EMGD": 1920, "EMGQ": 1920,
    "ACC": 120, "GYRO": 120,
    "FSR": 120, "EKG": 120,
    "Analog": 2400,
    "SmO2": 5, "Thb": 5,
    "VO2": 1, "HR": 1,
}

#: Canonical sub-channel ordering per modality. Used by :class:`delsys.Sensor`
#: to stack per-channel :class:`delsys.Signal` objects into multi-channel
#: modality bundles (:class:`delsys.IMU`, :class:`delsys.FSR`,
#: :class:`delsys.VO2Master`) in a stable order.
SUBCHANNEL_MAP: Dict[str, Tuple[str, ...]] = {
    "EMGS": ("A",),
    "EMGD": ("A", "B"),
    "EMGQ": ("A", "B", "C", "D"),
    "ACC": ("X", "Y", "Z"),
    "GYRO": ("X", "Y", "Z"),
    "FSR": ("A", "B", "C", "D"),
    "EKG": ("A",),
    "Analog": ("A",),
    "VO2": (
        "BreathingCycle", "Resp.Rate", "TidalVol.", "Ventilation(L/min)",
        "FeO2(%)", "VO2Absolute", "AmbientPressure", "FlowSensor", "OxygenSensor",
    ),
    "HR": ("HeartRate",),
}

#: Recognized acquisition applications. The header parser uses this to gate
#: which per-format reader runs.
APPLICATIONS: Tuple[str, ...] = ("EMGworks", "Trigno Discover")

#: Placeholder sensor numbers for Delsys link devices (VO2 Master, HR Strap).
#: Chosen to sit far from Trigno-Base sensor numbers (typically 1–16) so
#: ``sensor_number`` lookups remain unambiguous.
VO2_SENSOR_NUM: int = 900
HR_SENSOR_NUM: int = 901
