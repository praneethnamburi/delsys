"""``Sensor`` — one physical Delsys sensor with all its modality bundles.

A ``Sensor`` is built from a :class:`SensorInfo` metadata record and the
list of :class:`Signal` objects produced by the loader for that sensor
number. Its :meth:`__init__` groups signals by modality, stacks per-channel
arrays in canonical sub-channel order, and constructs the appropriate
modality bundle (``EMG``, ``EKG``, ``IMU``, ``FSR``, ``VO2Master``,
``Analog``, or ``HRStrap``) for each.
"""

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pysampled

from delsys._constants import LINK_DEVICE_REGISTRY, SUBCHANNEL_MAP
from delsys._metadata import SensorInfo
from delsys._util import _mod_to_attr, _parse_fsr_quattro_positions, _trim_location
from delsys.ekg import EKG
from delsys.emg import EMG
from delsys.signals import FSR, IMU, Signal, VO2Master

#: Fixed channel names for the VO2Master link device, in canonical column
#: order (matches the column ordering produced by the parser via
#: :data:`delsys._constants.SUBCHANNEL_MAP`'s ``'VO2'`` entry, minus the
#: dropped ``BreathingCycle`` column).
_VO2_SIGNAL_NAMES: Tuple[str, ...] = (
    "resp_rate",
    "tidal_vol",
    "ventilation",
    "feo2",
    "vo2_absolute",
    "ambient_pressure",
    "flow_sensor",
    "oxygen_sensor_humidity",
)

#: Letter suffixes used for multi-channel fallback names (Analog with >1
#: channel, EMGD/EMGQ/FSR fallbacks). Beyond the four letters, callers fall
#: back to numeric suffixes.
_LETTER_KEYS: Tuple[str, ...] = ("A", "B", "C", "D")

#: Modality tag → bundle class. Drives the dispatch in :class:`Sensor`'s
#: constructor: each parsed modality is wrapped in the corresponding
#: :class:`pysampled.Data` subclass (or plain :class:`pysampled.Data` for
#: ``Analog`` and ``HR``, which don't have a sensor-aware bundle class).
#: Adding a new modality is a one-line edit here plus a
#: :data:`delsys._constants.SUBCHANNEL_MAP` entry.
MODALITY_REGISTRY: Dict[str, Type[pysampled.Data]] = {
    "EMGS": EMG,
    "EMGD": EMG,
    "EMGQ": EMG,
    "EKG": EKG,
    "ACC": IMU,
    "GYRO": IMU,
    "FSR": FSR,
    "Analog": pysampled.Data,
    "VO2": VO2Master,
    "HR": pysampled.Data,
}


class Sensor:
    """All signals from one physical Delsys sensor, exposed as typed bundles.

    The constructor walks the ``signal_list``, groups signals by modality,
    and attaches each modality bundle as an instance attribute named per
    :func:`_mod_to_attr` (``emg``, ``ekg``, ``acc``, ``gyro``, ``fsr``,
    ``analog``, ``vo2master``, ``hrstrap``). The metadata fields from
    ``sensor_info`` (``name``, ``number``, ``type_sensorlog``, ``lrc``,
    ``location``, ``modalities``) are also copied onto the instance for
    direct attribute access.

    Args:
        sensor_info: Combined sensor metadata produced by the loader.
        signal_list: All :class:`Signal` objects belonging to this sensor.
            Every entry's ``signal.sensor`` must equal ``sensor_info``.

    Attributes:
        name (str): Human-readable sensor name (e.g. ``"EMG 01 04498"``).
        number (int): Delsys sensor number.
        type_sensorlog (Optional[str]): Sensor type from the channelmap
            (e.g. ``"EMG"``, ``"FSR"``). ``None`` if no channelmap was used.
        lrc (Optional[str]): Sensor side — ``'L'``, ``'R'``, ``'C'`` or ``None``.
        location (Optional[str]): Body-location label from the channelmap.
        modalities (set[str]): All modality tags carried by this sensor's signals.
        emg (EMG): Present iff this sensor has an EMG modality.
        ekg (EKG): Present iff this sensor has an EKG modality.
        acc (IMU): Present iff this sensor has ACC.
        gyro (IMU): Present iff this sensor has GYRO.
        fsr (FSR): Present iff this sensor has FSR.
        analog (pysampled.Data): Present iff this sensor has Analog.
            Plain :class:`pysampled.Data` (not a sensor-aware bundle).
        vo2master (VO2Master): Present iff this sensor is a VO2 Master link device.
        hrstrap (pysampled.Data): Present iff this sensor is an HR Strap link
            device. Plain :class:`pysampled.Data` (not a sensor-aware bundle).

    Raises:
        AssertionError: If ``sensor_info`` is not a :class:`SensorInfo`, or
            if any element of ``signal_list`` is not a :class:`Signal`, or
            if a signal's ``.sensor`` does not match ``sensor_info``, or if
            channels of one modality have inconsistent sample counts or
            ``t0``.

    Note:
        Construction via :class:`Log` runs every signal through
        :func:`delsys._util._normalize_signal_lengths` first, so the
        same-modality length assert is a safety net rather than a
        tripwire on that path. Direct callers building a :class:`Sensor`
        from un-normalized signals are responsible for length consistency.
    """

    def __init__(self, sensor_info: SensorInfo, signal_list: List[Signal]) -> None:
        assert isinstance(sensor_info, SensorInfo)
        for k, v in sensor_info._asdict().items():
            setattr(self, k, v)

        assert isinstance(signal_list, list)
        for s in signal_list:
            assert isinstance(s, Signal)
            assert sensor_info == s.sensor

        # Group signals by modality and attach the appropriate bundle class.
        for mod in self.modalities:
            this_signals = [s for s in signal_list if s.modality == mod]
            assert len(np.unique([len(s) for s in this_signals])) == 1  # same n_samples per channel
            assert len(set([x._t0 for x in this_signals])) == 1
            (t0,) = np.unique([s._t0 for s in this_signals])
            (sr,) = np.unique([s.sr for s in this_signals])
            mod_sig_list: List[Signal] = []
            for subchannel in SUBCHANNEL_MAP[mod]:
                mod_sig_list += [s for s in this_signals if s.subchannel == subchannel]
            sig = np.vstack([s() for s in mod_sig_list]).T

            # Modality-aware bundles carry the SensorInfo via ``meta['sensor']``
            # so user code can ask ``ekg.sensor.location`` / ``emg.sensor.lrc``
            # etc. without walking back through Log.sensors.
            sensor_meta = {"sensor": sensor_info}

            n_channels = sig.shape[1] if sig.ndim == 2 else 1
            signal_names, signal_coords = self._make_bundle_labels(
                mod, sensor_info, n_channels
            )
            # ``axis=0`` is what the loader produces (samples down rows,
            # channels across columns) — make it explicit so very short
            # fixtures don't trip pysampled's argmax-based axis inference
            # (e.g. a (1, 8) VO2 array would otherwise look like 1 channel
            # of 8 samples).
            bundle_kwargs = dict(
                sr=sr,
                axis=0,
                t0=t0,
                meta=sensor_meta,
                signal_names=signal_names,
                signal_coords=signal_coords,
            )

            cls = MODALITY_REGISTRY[mod]
            setattr(self, _mod_to_attr(mod), cls(sig, **bundle_kwargs))

    def __setstate__(self, state: dict) -> None:
        """Auto-relabel bundles on unpickle.

        Pickles produced before delsys 0.1.1 (or by the legacy
        ``immersionToolbox/immersionlab/delsys.py`` shim) carry pysampled's
        default ``signal_names=['s0', 's1', ...]`` / ``signal_coords=['x']``
        labelling — and often an empty ``meta`` dict. Once the pickle is
        loaded we have everything we need to rebuild the labels: the parent
        ``Sensor`` retains ``number``, ``location``, ``modalities``, etc.
        as plain attributes. Walk the attached bundles and re-stamp them
        with the 0.1.1 convention, plus seed ``meta['sensor']`` so future
        filter / clone operations propagate the sensor identity.

        Idempotent: running on a fresh 0.1.1+ pickle produces identical
        labels.

        Per-:class:`Signal` ``meta`` (``modality`` / ``subchannel`` /
        ``sensor``) is *not* recoverable from such old pickles — the bundle
        view is sufficient for the typical ``lf.acc[i]`` / ``lf.emg[i]``
        access pattern.
        """
        self.__dict__.update(state)
        self._relabel_bundles()

    def _relabel_bundles(self) -> None:
        """Rebuild ``signal_names`` / ``signal_coords`` (and ``meta['sensor']``)
        on every modality bundle attached to this :class:`Sensor`, using the
        Sensor's own attributes as the source of truth.

        Called from :meth:`__setstate__`; safe to call directly on a
        live :class:`Sensor` to refresh labels.
        """
        if not hasattr(self, "modalities"):
            return  # very-old / hand-built Sensor without modality info
        sensor_info = SensorInfo(
            name=getattr(self, "name", None),
            modalities=self.modalities,
            number=getattr(self, "number", 0),
            type_sensorlog=getattr(self, "type_sensorlog", None),
            lrc=getattr(self, "lrc", None),
            location=getattr(self, "location", None),
        )
        for mod in self.modalities:
            attr = _mod_to_attr(mod)
            bundle = getattr(self, attr, None)
            if bundle is None or not hasattr(bundle, "_sig"):
                continue
            n_channels = bundle._sig.shape[1] if bundle._sig.ndim == 2 else 1
            names, coords = self._make_bundle_labels(mod, sensor_info, n_channels)
            bundle.signal_names = names
            bundle.signal_coords = coords
            if not hasattr(bundle, "meta") or bundle.meta is None:
                bundle.meta = {}
            bundle.meta.setdefault("sensor", sensor_info)

    @staticmethod
    def _make_bundle_labels(
        mod: str, sensor_info: SensorInfo, n_channels: int
    ) -> Tuple[List[str], List[str]]:
        """Build ``(signal_names, signal_coords)`` for a modality bundle.

        Centralizes the label conventions documented in the 0.1.1 plan:

        * ACC / GYRO use a single signal name (the trimmed location) with
          three signal coordinates ``x``/``y``/``z``.
        * EMGS / EKG use a single signal name with a single modality coord.
        * EMGD / EMGQ / FSR are multi-name (one entry per channel) with a
          single coordinate. EMGQ and FSR try to parse the channelmap
          parenthetical for position names; on failure they fall back to
          A/B/C/D suffixes.
        * Analog is normally single-channel (``[loc] x ['analog']``) but
          some sync sensors come through with multiple channels; multi-
          channel Analog falls back to ``[loc_A, loc_B, ...]``.
        * VO2 has eight fixed signal names.
        * HR is always ``['heart_rate'] x ['bpm']``.

        Args:
            mod: Modality tag (one of the keys in
                :data:`delsys._constants.SUBCHANNEL_MAP`).
            sensor_info: The owning sensor's metadata; supplies ``location``
                and ``number`` (the latter is only used for the no-channelmap
                ``chN`` fallback).
            n_channels: Actual number of channels in the stacked array.
                Used for Analog (which is variable) and for the catch-all
                fallback path.

        Returns:
            ``(signal_names, signal_coords)`` lists ready to pass to a
            :class:`pysampled.Data` constructor.
        """
        loc = _trim_location(sensor_info.location, sensor_info.number)

        if mod in ("ACC", "GYRO"):
            return [loc], ["x", "y", "z"]
        if mod == "EKG":
            return [loc], ["ekg"]
        if mod == "Analog":
            if n_channels == 1:
                return [loc], ["analog"]
            return [
                f"{loc}_{_LETTER_KEYS[i]}" if i < len(_LETTER_KEYS) else f"{loc}_{i}"
                for i in range(n_channels)
            ], ["analog"]
        if mod == "VO2":
            return list(_VO2_SIGNAL_NAMES), ["value"]
        if mod == "HR":
            return ["heart_rate"], ["bpm"]
        if mod == "FSR":
            parsed = _parse_fsr_quattro_positions(sensor_info.location, "FSR")
            keys = parsed if parsed is not None else list(SUBCHANNEL_MAP["FSR"])
            return [f"{loc}_{k}" for k in keys], ["fsr"]
        if mod == "EMGS":
            return [loc], ["emg"]
        if mod == "EMGD":
            return [f"{loc}_{k}" for k in SUBCHANNEL_MAP["EMGD"]], ["emg"]
        if mod == "EMGQ":
            parsed = _parse_fsr_quattro_positions(sensor_info.location, "EMGQ")
            keys = parsed if parsed is not None else list(SUBCHANNEL_MAP["EMGQ"])
            return [f"{loc}_{k}" for k in keys], ["emg"]
        # Unknown modality — keep the bundle constructable by sizing
        # signal_names to the actual channel count. Should never hit in
        # practice (every modality the parser emits has a branch above).
        return [
            f"{loc}_{_LETTER_KEYS[i]}" if i < len(_LETTER_KEYS) else f"{loc}_{i}"
            for i in range(n_channels)
        ], ["value"]

    @property
    def is_link(self) -> bool:
        """``True`` if this sensor is a Delsys link device.

        A link device is any sensor whose modality appears in
        :data:`delsys._constants.LINK_DEVICE_REGISTRY` — currently the
        VO2 Master and HR Strap. Use this in preference to comparing
        ``Sensor.number`` against magic constants: the underlying
        synthetic numbers are an internal detail of how the parser
        labels link channels, not a stable identifier.
        """
        link_modalities = {mod for mod, _ in LINK_DEVICE_REGISTRY.values()}
        return bool(self.modalities & link_modalities)

    def get_signal(self) -> Optional[Union[EMG, EKG, pysampled.Data, FSR, VO2Master]]:
        """Return the first non-IMU bundle attached to this sensor.

        Useful when EMG/EKG and Analog/FSR get mis-typed and you don't know
        which attribute to reach for. Lookup priority is: ``emg``, ``ekg``,
        ``analog``, ``fsr``, ``vo2master``, ``hrstrap``. Returns ``None`` if
        the sensor only has IMU data (or no data at all).
        """
        priority_sequence = ["emg", "ekg", "analog", "fsr", "vo2master", "hrstrap"]
        for attr_name in priority_sequence:
            if hasattr(self, attr_name):
                return getattr(self, attr_name)
        return None
