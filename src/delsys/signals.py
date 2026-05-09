"""Signal classes.

Defines the per-channel ``Signal`` class and the modality-bundled classes
``IMU`` (3-axis accelerometer/gyroscope), ``FSR`` (4-channel force-sensitive
resistor), and ``VO2Master`` (8-channel respiratory gas analyzer). Each is a
thin extension of :class:`pysampled.Data` that adds modality-specific
property accessors plus two tiny shared conveniences (``sensor`` and
``shape``).

Sensor metadata is carried through :attr:`pysampled.Data.meta` under the
key ``'sensor'`` (and, for :class:`Signal`, also ``'modality'`` and
``'subchannel'``), so it survives every clone, slice, filter, or resample
operation pysampled performs.

The metadata namedtuples (``SensorLog`` and ``SensorInfo``) live in
:mod:`delsys._metadata`.
"""

from typing import Optional

import pysampled

from delsys._metadata import SensorInfo
from delsys._util import _mod_to_attr


class Signal(pysampled.Data):
    """One per-channel time series tagged with sensor + modality + sub-channel.

    A ``Signal`` is the basic unit produced by the per-format dataframe
    parsers: one row in :attr:`Log.signals` corresponds to one column of
    sample data after resampling. Three identifying fields — ``sensor``,
    ``modality``, and ``subchannel`` — live in :attr:`pysampled.Data.meta`
    so they survive cloning operations (filtering, resampling, slicing)
    automatically.

    Construct with ``meta`` populated::

        Signal(sig, sr, t0=t0, meta={
            "sensor": sensor_info,
            "modality": "EMGS",
            "subchannel": "A",
        })

    Attributes:
        sensor (SensorInfo): The source sensor's metadata record.
        modality (str): Normalized modality tag.
        subchannel (str): Sub-channel label.
        sensor_name (str): The modality-as-attribute-name (e.g. ``'emg'``,
            ``'vo2master'``); kept for backward compatibility.
        shape (tuple): Shape of the underlying sample array.
    """

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The :class:`SensorInfo` record, or ``None`` if not set."""
        return self.meta.get("sensor") if self.meta else None

    @property
    def modality(self) -> str:
        """Normalized modality tag (one of ``'EMGS'``, ``'EMGD'``, ``'EMGQ'``,
        ``'EKG'``, ``'ACC'``, ``'GYRO'``, ``'FSR'``, ``'Analog'``, ``'VO2'``,
        ``'HR'``)."""
        return self.meta["modality"]

    @property
    def subchannel(self) -> str:
        """Sub-channel label — ``'A'`` for single-channel modalities,
        ``'X'``/``'Y'``/``'Z'`` for IMU axes, ``'A'``/``'B'``/``'C'``/``'D'``
        for multi-channel EMG and FSR."""
        return self.meta["subchannel"]

    @property
    def sensor_name(self) -> str:
        """The modality-as-attribute-name (e.g. ``'emg'``, ``'vo2master'``).

        Historical name; kept for backward compatibility. This is the
        modality translated into the attribute name used on the parent
        :class:`Sensor`, not the sensor's display name.
        """
        return _mod_to_attr(self.meta["modality"])

    @property
    def shape(self) -> tuple:
        """Shape of the underlying sample array (shortcut for ``self._sig.shape``)."""
        return self._sig.shape


class IMU(pysampled.Data):
    """Tri-axial accelerometer or gyroscope bundle.

    Holds the three axes of a Delsys IMU as a single 2-D array of shape
    ``(n_samples, 3)`` with columns ``X``, ``Y``, ``Z`` in that order. Each
    axis is exposed as a property returning a single-axis :class:`IMU`.

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            for sensor in lf.find(modality="ACC", as_="sensor"):
                acc = sensor.acc
                x_axis = acc.x         # IMU containing only the X column
                magnitude = (acc.x() ** 2 + acc.y() ** 2 + acc.z() ** 2) ** 0.5
    """

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The :class:`SensorInfo` record, or ``None`` if not set."""
        return self.meta.get("sensor") if self.meta else None

    @property
    def shape(self) -> tuple:
        """Shape of the underlying sample array."""
        return self._sig.shape

    x = property(lambda s: s._clone(s()[:, 0], signal_names=["acc_x"], signal_coords=["x"]))
    y = property(lambda s: s._clone(s()[:, 1], signal_names=["acc_y"], signal_coords=["y"]))
    z = property(lambda s: s._clone(s()[:, 2], signal_names=["acc_z"], signal_coords=["z"]))


class FSR(pysampled.Data):
    """Force-sensitive-resistor bundle (four channels).

    Stores the four FSR channels as a single 2-D array of shape
    ``(n_samples, 4)`` with columns ``A``, ``B``, ``C``, ``D`` in that order.
    Each channel is exposed as a property returning a single-channel
    :class:`FSR`.

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            for sensor in lf.find(modality="FSR", as_="sensor"):
                heel = sensor.fsr.a
                ball = sensor.fsr.b
    """

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The :class:`SensorInfo` record, or ``None`` if not set."""
        return self.meta.get("sensor") if self.meta else None

    @property
    def shape(self) -> tuple:
        """Shape of the underlying sample array."""
        return self._sig.shape

    a = property(lambda s: s._clone(s()[:, 0], signal_names=["fsr_a"], signal_coords=["a"]))
    b = property(lambda s: s._clone(s()[:, 1], signal_names=["fsr_b"], signal_coords=["b"]))
    c = property(lambda s: s._clone(s()[:, 2], signal_names=["fsr_c"], signal_coords=["c"]))
    d = property(lambda s: s._clone(s()[:, 3], signal_names=["fsr_d"], signal_coords=["d"]))


class VO2Master(pysampled.Data):
    """VO2 Master link-device bundle (eight channels in fixed order).

    The ``BreathingCycle`` channel is dropped at parse time, so it does not
    appear here. The remaining eight columns are exposed under both short
    and verbose property names so notebook code can use whichever reads
    better:

    +-----+-----------------------------+----------------------------+
    | Idx | Short name                  | Verbose alias              |
    +=====+=============================+============================+
    | 0   | :attr:`rr`                  | ``respiration_rate``       |
    +-----+-----------------------------+----------------------------+
    | 1   | :attr:`td`                  | ``tidal_vol``              |
    +-----+-----------------------------+----------------------------+
    | 2   | :attr:`vent`                | ``ventilation``            |
    +-----+-----------------------------+----------------------------+
    | 3   | :attr:`Feo2`                | (none)                     |
    +-----+-----------------------------+----------------------------+
    | 4   | :attr:`vo2`                 | ``VO2_absolute``           |
    +-----+-----------------------------+----------------------------+
    | 5   | :attr:`ap`                  | ``ambient_pressure``       |
    +-----+-----------------------------+----------------------------+
    | 6   | :attr:`fl`                  | ``flow_sensor``            |
    +-----+-----------------------------+----------------------------+
    | 7   | :attr:`o2_hum`              | ``oxygen_sensor_humidity`` |
    +-----+-----------------------------+----------------------------+

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            vo2master, = lf.vo2master
            mean_breathing_rate = vo2master.rr().mean()
            tidal_volume_ml = vo2master.tidal_vol() * 1000
    """

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The :class:`SensorInfo` record, or ``None`` if not set."""
        return self.meta.get("sensor") if self.meta else None

    @property
    def shape(self) -> tuple:
        """Shape of the underlying sample array."""
        return self._sig.shape

    rr = respiration_rate = property(
        lambda s: s._clone(s()[:, 0], signal_names=["resp_rate"], signal_coords=["value"])
    )
    td = tidal_vol = property(
        lambda s: s._clone(s()[:, 1], signal_names=["tidal_vol"], signal_coords=["value"])
    )
    vent = ventilation = property(
        lambda s: s._clone(s()[:, 2], signal_names=["ventilation"], signal_coords=["value"])
    )
    Feo2 = property(lambda s: s._clone(s()[:, 3], signal_names=["feo2"], signal_coords=["value"]))
    vo2 = VO2_absolute = property(
        lambda s: s._clone(s()[:, 4], signal_names=["vo2_absolute"], signal_coords=["value"])
    )
    ap = ambient_pressure = property(
        lambda s: s._clone(s()[:, 5], signal_names=["ambient_pressure"], signal_coords=["value"])
    )
    fl = flow_sensor = property(
        lambda s: s._clone(s()[:, 6], signal_names=["flow_sensor"], signal_coords=["value"])
    )
    o2_hum = oxygen_sensor_humidity = property(
        lambda s: s._clone(
            s()[:, 7], signal_names=["oxygen_sensor_humidity"], signal_coords=["value"]
        )
    )
