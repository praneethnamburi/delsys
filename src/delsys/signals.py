"""Signal classes: per-channel ``Signal`` and modality-bundled ``IMU``,
``FSR``, ``VO2Master``. All inherit from ``DataMod`` and carry a ``sensor``
metadata attribute.

Also defines the ``SensorInfo`` and ``SensorLog`` namedtuples used as
foundational metadata records throughout the package.
"""
from collections import namedtuple

from delsys._datamod import DataMod
from delsys._util import _mod_to_attr


# Manually entered sensor map record (one row per sensor in delsys_channelmap.txt).
# lrc is L/R/C/None for left/right/center/unspecified.
SensorLog = namedtuple('SensorLog', 'number type_sensorlog lrc location')

# Sensor metadata combined from the channelmap and the CSV header.
SensorInfo = namedtuple('SensorInfo', 'name modalities number type_sensorlog lrc location')


class Signal(DataMod):
    """One-channel time series tagged with its source sensor + modality + sub-channel."""

    def __init__(self, sig, sr, sensor, modality, subchannel, axis=None, history=None, t0=0.):
        super().__init__(sig, sr, axis=axis, history=history, t0=t0, sensor=sensor)
        assert isinstance(sensor, SensorInfo), (
            f"Signal.sensor must be a SensorInfo namedtuple, got {type(sensor).__name__}"
        )
        self.modality = modality
        self.subchannel = subchannel
        # Historical name; this stores the modality-as-attribute-name (e.g. 'emg', 'vo2master'),
        # not the sensor's display name. Kept for backward compatibility.
        self.sensor_name = _mod_to_attr(modality)

    def _clone(self, proc_sig, his_append=None):
        if his_append is None:
            his = self._history  # only useful when cloning without manipulating the data
        else:
            his = self._history + [his_append]
        return self.__class__(
            proc_sig, self.sr, self.sensor, self.modality, self.subchannel,
            self.axis, his, self._t0,
        )


class IMU(DataMod):
    """Tri-axial accelerometer/gyroscope bundle (X, Y, Z columns)."""

    x = property(lambda s: s._clone(s()[:, 0], signal_names=["acc_x"], signal_coords=["x"]))
    y = property(lambda s: s._clone(s()[:, 1], signal_names=["acc_y"], signal_coords=["y"]))
    z = property(lambda s: s._clone(s()[:, 2], signal_names=["acc_z"], signal_coords=["z"]))


class FSR(DataMod):
    """Force-sensitive-resistor bundle (4 channels: A, B, C, D)."""

    a = property(lambda s: s._clone(s()[:, 0], signal_names=["fsr_a"], signal_coords=["a"]))
    b = property(lambda s: s._clone(s()[:, 1], signal_names=["fsr_b"], signal_coords=["b"]))
    c = property(lambda s: s._clone(s()[:, 2], signal_names=["fsr_c"], signal_coords=["c"]))
    d = property(lambda s: s._clone(s()[:, 3], signal_names=["fsr_d"], signal_coords=["d"]))


class VO2Master(DataMod):
    """VO2 Master link-device bundle (8 channels in fixed order; BreathingCycle
    is dropped at parse time, so it does not appear here)."""

    rr = respiration_rate           = property(lambda s: s._clone(s()[:, 0], signal_names=["resp_rate"], signal_coords=["value"]))
    td = tidal_vol                  = property(lambda s: s._clone(s()[:, 1], signal_names=["tidal_vol"], signal_coords=["value"]))
    vent = ventilation              = property(lambda s: s._clone(s()[:, 2], signal_names=["ventilation"], signal_coords=["value"]))
    Feo2                            = property(lambda s: s._clone(s()[:, 3], signal_names=["feo2"], signal_coords=["value"]))
    vo2 = VO2_absolute              = property(lambda s: s._clone(s()[:, 4], signal_names=["vo2_absolute"], signal_coords=["value"]))
    ap = ambient_pressure           = property(lambda s: s._clone(s()[:, 5], signal_names=["ambient_pressure"], signal_coords=["value"]))
    fl = flow_sensor                = property(lambda s: s._clone(s()[:, 6], signal_names=["flow_sensor"], signal_coords=["value"]))
    o2_hum = oxygen_sensor_humidity = property(lambda s: s._clone(s()[:, 7], signal_names=["oxygen_sensor_humidity"], signal_coords=["value"]))
