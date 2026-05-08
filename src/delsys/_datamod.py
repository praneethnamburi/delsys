"""Base class adding extras to ``pysampled.Data``: extrema-based envelope and
optional per-instance ``.sensor`` metadata that survives ``_clone`` calls.

All modality-aware classes (Signal, IMU, FSR, VO2Master, EMG, EKG) inherit
from ``DataMod`` so they all carry sensor metadata uniformly.
"""
from typing import Optional, Union

import numpy as np
import pysampled
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema


class DataMod(pysampled.Data):
    """``pysampled.Data`` extended with ``envelope2`` and optional ``sensor``.

    The ``sensor`` attribute holds a ``SensorInfo`` namedtuple identifying which
    physical sensor produced this signal. It defaults to ``None`` at the class
    level so that pickle files saved before this attribute existed unpickle
    cleanly (they get the class-level default).
    """

    sensor = None  # class-level default — applies to instances loaded from old pickles

    def __init__(self, *args, sensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        if sensor is not None:
            self.sensor = sensor

    def _clone(self, *args, **kwargs):
        """Preserve ``sensor`` through the standard pysampled ``_clone`` flow."""
        new = super()._clone(*args, **kwargs)
        new.sensor = self.sensor
        return new

    def envelope2(self, side: str = 'upper', lowpass: Optional[Union[float, int]] = None):
        """Maxima/minima-based envelope. Set ``side='lower'`` for the lower envelope."""
        if side not in ('upper', 'lower'):
            raise ValueError("Not supported side. It must be either upper or lower")

        comp = np.greater if side == 'upper' else np.less
        proc_signs = []
        idxs = 0
        for sign in self._sig.T:
            idxs = argrelextrema(sign, comp)[0]
            idxs = np.concatenate(([0], idxs, [-1]))
            proc_sig = interp1d(
                self.t[idxs], sign[idxs], kind='linear')(
                np.linspace(self.t[idxs][0], self.t[idxs][-1], len(self), endpoint=False)
            )
            proc_signs.append(proc_sig)
        proc_signs = np.array(proc_signs).T
        self._history += [('envelope2_' + side, None)]
        if lowpass:
            return self.__class__(proc_signs, self.sr, axis=self.axis, t0=self.t[idxs][0],
                                  history=self._history).lowpass(lowpass, order=2)
        return self.__class__(proc_signs, self.sr, axis=self.axis, t0=self.t[idxs][0], history=self._history)

    @property
    def shape(self):
        return self._sig.shape
