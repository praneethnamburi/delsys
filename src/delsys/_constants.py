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
#:
#: The default is now **native** (``None``) for the uniformly-sampled Trigno-base
#: modalities, so a bare ``delsys.Log(h5)`` preserves their native rates and never
#: resamples / anti-alias-filters them on load. The **link devices** below
#: (``SmO2``, ``Thb``, ``VO2``, ``HR``) keep their target rates: they arrive
#: irregularly (breath-by-breath / per-beat), so they must be resampled onto a
#: uniform grid and ``None`` is not valid for them. To also resample the base
#: modalities on load, pass ``target_sr=`` explicitly or restore the reference
#: dict below — the rates these modalities were previously normalized to:
#:
#: TARGET_SR: Dict[str, Optional[float]] = {
#:     "EMGS": 1920,
#:     "EMGD": 1920,
#:     "EMGQ": 1920,
#:     "ACC": 120,
#:     "GYRO": 120,
#:     "FSR": 120,
#:     "EKG": 120,
#:     "Analog": 2400,
#:     "SmO2": 5,
#:     "Thb": 5,
#:     "VO2": 1,
#:     "HR": 1,
#: }
TARGET_SR: Dict[str, Optional[float]] = {
    "EMGS": None,
    "EMGD": None,
    "EMGQ": None,
    "ACC": None,
    "GYRO": None,
    "FSR": None,
    "EKG": None,
    "Analog": None,
    # Link devices: irregularly sampled -> must be resampled onto a uniform grid
    # (``None`` is not honored for these). Keep the reference rates.
    "SmO2": 5,
    "Thb": 5,
    "VO2": 1,
    "HR": 1,
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
        "BreathingCycle",
        "Resp.Rate",
        "TidalVol.",
        "Ventilation(L/min)",
        "FeO2(%)",
        "VO2Absolute",
        "AmbientPressure",
        "FlowSensor",
        "OxygenSensor",
    ),
    "HR": ("HeartRate",),
}

#: Sub-channel keys expected inside the FSR / Quattro channelmap location
#: parenthetical (e.g. ``LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)``).
#: ``"FSR"`` entries use numeric keys (``1``/``2``/``3``/``4``); ``"EMGQ"``
#: (Quattro) uses letters (``A``/``B``/``C``/``D``). This is the textual
#: convention the parenthetical is parsed against — distinct from
#: :data:`SUBCHANNEL_MAP`, which is the canonical sub-channel ordering used
#: when stacking per-channel arrays into modality bundles.
_SUBCHANNEL_KEYS: Dict[str, Tuple[str, ...]] = {
    "FSR": ("1", "2", "3", "4"),
    "EMGQ": ("A", "B", "C", "D"),
}

#: Recognized acquisition applications. The header parser uses this to gate
#: which per-format reader runs.
APPLICATIONS: Tuple[str, ...] = ("EMGworks", "Trigno Discover")

#: Link-device identification. Maps a substring of the column-header
#: ``sensor_name`` to ``(modality, synthetic_sensor_number)``. The
#: per-format parser walks this dict in registration order; first
#: substring match wins. The synthetic numbers sit far from Trigno-Base
#: sensor numbers (typically 1–16) so ``sensor_number`` lookups remain
#: unambiguous.
#:
#: Adding a new link device requires:
#:
#: 1. one entry here,
#: 2. a :data:`SUBCHANNEL_MAP` entry for its modality,
#: 3. a :data:`TARGET_SR` entry,
#: 4. a ``MODALITY_REGISTRY`` entry in :mod:`delsys.sensor`.
LINK_DEVICE_REGISTRY: Dict[str, Tuple[str, int]] = {
    "VO2 Master": ("VO2", 900),
    "HR Strap": ("HR", 901),
}
