"""``Sensor`` — one physical Delsys sensor with all its modality bundles.

A ``Sensor`` is built from a :class:`SensorInfo` metadata record and the
list of :class:`Signal` objects produced by the loader for that sensor
number. Its :meth:`__init__` groups signals by modality, stacks per-channel
arrays in canonical sub-channel order, and constructs the appropriate
modality bundle (``EMG``, ``EKG``, ``IMU``, ``FSR``, ``VO2Master``,
``Analog``, or ``HRStrap``) for each.
"""

from typing import List, Optional, Union

import numpy as np
import pysampled

from delsys._constants import SUBCHANNEL_MAP
from delsys._metadata import SensorInfo
from delsys._util import _mod_to_attr
from delsys.ekg import EKG
from delsys.emg import EMG
from delsys.signals import FSR, IMU, Signal, VO2Master


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

            if mod in ("ACC", "GYRO"):
                setattr(self, _mod_to_attr(mod), IMU(sig, sr, t0=t0, meta=sensor_meta))
            elif mod == "FSR":
                self.fsr = FSR(sig, sr, t0=t0, meta=sensor_meta)
            elif mod == "EKG":
                self.ekg = EKG(sig, sr, t0=t0, meta=sensor_meta)
            elif mod == "Analog":
                # Analog stays as a plain pysampled.Data for backward compatibility;
                # users wanting metadata can find it on the parent Sensor or via
                # ``lf.find(modality='Analog', as_='sensor')``.
                self.analog = pysampled.Data(sig, sr, t0=t0)
            elif mod == "VO2":
                self.vo2master = VO2Master(sig, sr, t0=t0, meta=sensor_meta)
            elif mod == "HR":
                # HR Strap also stays as plain pysampled.Data for now.
                self.hrstrap = pysampled.Data(sig, sr, t0=t0)
            else:
                assert mod.startswith("EMG")
                self.emg = EMG(sig, sr, t0=t0, meta=sensor_meta)

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
