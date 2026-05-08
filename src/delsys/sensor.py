"""``Sensor`` aggregates all signals from one physical sensor and exposes
modality-specific data through typed bundles (EMG, EKG, IMU, FSR, ...).
"""
import numpy as np
import pysampled

from delsys._constants import SUBCHANNEL_MAP
from delsys._util import _mod_to_attr
from delsys.signals import Signal, SensorInfo, IMU, FSR, VO2Master
from delsys.emg import EMG
from delsys.ekg import EKG


class Sensor:
    def __init__(self, sensor_info, signal_list):
        assert isinstance(sensor_info, SensorInfo)
        for k, v in sensor_info._asdict().items():
            setattr(self, k, v)

        assert isinstance(signal_list, list)
        for s in signal_list:
            assert isinstance(s, Signal)
            assert sensor_info == s.sensor

        # Group signal list by modality and attach the appropriate bundle class
        for mod in self.modalities:
            this_signals = [s for s in signal_list if s.modality == mod]
            assert len(np.unique([len(s) for s in this_signals])) == 1  # same n_samples per channel
            assert len(set([x._t0 for x in this_signals])) == 1
            t0, = np.unique([s._t0 for s in this_signals])
            sr, = np.unique([s.sr for s in this_signals])
            mod_sig_list = []
            for subchannel in SUBCHANNEL_MAP[mod]:
                mod_sig_list += [s for s in this_signals if s.subchannel == subchannel]
            sig = np.vstack([s() for s in mod_sig_list]).T

            # Modality-aware bundles carry the SensorInfo so user code can ask
            # ``ekg.sensor.location`` / ``emg.sensor.lrc`` etc. without walking
            # back through Log.sensors.
            if mod in ('ACC', 'GYRO'):
                setattr(self, _mod_to_attr(mod), IMU(sig, sr, t0=t0, sensor=sensor_info))
            elif mod == 'FSR':
                self.fsr = FSR(sig, sr, t0=t0, sensor=sensor_info)
            elif mod == 'EKG':
                self.ekg = EKG(sig, sr, t0=t0, sensor=sensor_info)
            elif mod == 'Analog':
                # Analog stays as a plain pysampled.Data for backward compatibility;
                # users wanting metadata can find it on the parent Sensor or via
                # ``lf.find(modality='Analog', as_='sensor')``.
                self.analog = pysampled.Data(sig, sr, t0=t0)
            elif mod == 'VO2':
                self.vo2master = VO2Master(sig, sr, t0=t0, sensor=sensor_info)
            elif mod == 'HR':
                # HR Strap also stays as plain pysampled.Data for now.
                self.hrstrap = pysampled.Data(sig, sr, t0=t0)
            else:
                assert mod.startswith('EMG')
                self.emg = EMG(sig, sr, t0=t0, sensor=sensor_info)

    def get_signal(self):
        """Return the non-IMU piece of the delsys sensor.
        Handy when EKG/EMG and Analog/FSR get mis-typed."""
        priority_sequence = ['emg', 'ekg', 'analog', 'fsr']
        for attr_name in priority_sequence:
            if hasattr(self, attr_name):
                return getattr(self, attr_name)
