"""EMG-specific signal class and feature extraction."""
from typing import Dict, Optional, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
import pysampled
from scipy.fftpack import fft, fftfreq

from delsys._datamod import DataMod
from delsys._util import decreturn


class EMG(DataMod):
    """EMG bundle: extends ``DataMod`` with EMG-specific preprocessing and features."""

    @decreturn
    def process_nk(self, to=dict):
        """Process EMG with NeuroKit2.

        :param to: Return type — dict, pd.DataFrame, np.ndarray, or list.
        """
        signals, _ = nk.emg_process(self().flatten(), self.sr)
        return signals.to_dict()

    def tkeo(self):
        """Teager–Kaiser Energy Operator — improves onset detection."""
        tkeo = self().copy()
        tkeo[1:-1] = self._sig[1:-1] * self._sig[1:-1] - self._sig[:-2] * self._sig[2:]
        # Correct the data at the extremities
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
        return self._clone(tkeo, his_append=[("Teager-Kaiser", None)])

    def process(self, amp_kind: str = 'envelope2', lowpass: Optional[Union[float, int]] = None, **kwargs):
        """Preprocess: bandpass 20-500 Hz → rectify → amplitude → optional lowpass.

        :param amp_kind: 'rms', 'mean', 'envelope', 'envelope2', or 'nk'.
        :param lowpass: Optional final lowpass cutoff (Hz).
        :param kwargs: ``win_size``/``win_inc`` for rms/mean; ``order`` for filters.
        """
        if amp_kind not in ('rms', 'mean', 'envelope', 'envelope2', 'nk'):
            raise ValueError("Not supported kind. It must be: rms, mean, envelope, envelope2 or nk.")

        if 'order' in kwargs:
            order = kwargs['order']
            del kwargs['order']
        else:
            order = 4
        # Bandpass: <20 Hz is motion noise, >450 Hz is electrical noise
        proc_sig12 = self.shift_baseline().bandpass(20, 500, order=order).apply(np.abs)

        if amp_kind == 'rms':
            rms = lambda x, ax: np.sqrt(np.mean(x ** 2, axis=ax))
            proc_sig3 = proc_sig12.apply_running_win(rms, **kwargs).resample(self.sr)
        elif amp_kind == 'mean':
            proc_sig3 = proc_sig12.apply_running_win(np.mean, **kwargs).resample(self.sr)
        elif amp_kind == 'envelope':
            proc_sig3 = proc_sig12.envelope(lowpass=10)
        elif amp_kind == 'envelope2':
            proc_sig3 = proc_sig12.envelope2(lowpass=10)
        elif amp_kind == "nk":
            proc_sig3 = nk.emg_amplitude(proc_sig12().flatten())
        else:
            raise ValueError

        self._history += [('preprocess_emg', amp_kind)]

        if lowpass:
            return self.__class__(
                proc_sig3(), self.sr, axis=self.axis, t0=proc_sig3._t0, history=proc_sig3._history,
            ).lowpass(lowpass, order=order)
        return self.__class__(
            proc_sig3(), self.sr, axis=self.axis, t0=proc_sig3._t0, history=proc_sig3._history,
        )

    @staticmethod
    def _temp_funcs(signal: pysampled.Data, win_size: float) -> Dict:
        th = np.mean(signal(), axis=signal.axis) + 3 * np.std(signal(), axis=signal.axis)
        funcs = {
            'mean': np.mean,
            'med': np.median,
            'var': np.var,
            'rms': lambda x, ax: np.sqrt(np.mean(x ** 2, axis=ax)),
            'int': lambda x, ax: np.sum(np.abs(x), axis=ax),
            'mav': lambda x, ax: np.sum(np.abs(x), axis=ax) / win_size,
            'log': lambda x, ax: np.exp(np.sum(np.log10(np.abs(x))) / win_size),
            'wav': lambda x, ax: np.sum(np.abs(np.diff(x, axis=ax)), axis=ax),
            'dsd': lambda x, ax: np.sqrt(np.sum(np.diff(x, axis=ax) ** 2, axis=ax)) / (win_size - 1),
            'zcr': lambda x, ax: len(np.where(np.diff(np.sign(x), axis=ax))[0]),
            'wamp': lambda x, ax: np.sum((np.abs(np.diff(x, axis=ax))) >= th, axis=ax),
            'myop': lambda x, ax: np.sum(x >= th, axis=ax) / len(x),
        }
        return funcs

    @staticmethod
    def _freq_funcs(sr: float) -> Dict:
        def spect(x):
            freq = fftfreq(x.size, d=1 / sr)
            power = (np.abs(fft(x)) ** 2).flatten()
            pos = np.where((freq > 0) & (freq < sr / 2))[0]
            return freq[pos], power[pos]

        def freq_rat(freq, power, ax):
            ulc = np.sum(power[(freq >= 20) & (freq <= 200)], axis=ax)
            uhc = np.sum(power[(freq > 200) & (freq <= 450)], axis=ax)
            return ulc / uhc

        funcs = {
            'frr': lambda x, ax: freq_rat(*spect(x), ax),
            'mnp': lambda x, ax: np.mean(spect(x)[1], axis=ax),
            'vap': lambda x, ax: np.var(spect(x)[1], axis=ax),
            'mnf': lambda x, ax: np.mean(spect(x)[0], axis=ax),
            'vaf': lambda x, ax: np.var(spect(x)[0], axis=ax),
            'mwf': lambda x, ax: np.sum(spect(x)[0] * spect(x)[1], axis=ax) / np.sum(spect(x)[1], axis=ax),
            'mav': lambda x, ax: spect(x)[0][
                np.searchsorted(np.cumsum(spect(x)[1]), np.sum(spect(x)[1]) * 0.5)],
            'pkf': lambda x, ax: spect(x)[0][spect(x)[1].argmax()],
        }
        return funcs

    @decreturn
    def get_features_nk(self, method='interval', to=dict):
        """EMG features via NeuroKit2."""
        signals = self.process_nk(to=pd.DataFrame)
        signals = nk.emg_analyze(signals, self.sr, method=method)
        return signals.to_dict()

    @decreturn
    def get_features(self, kind: str = 'all', win_size: float = 0.25, win_inc: float = 0.1, to=dict):
        """Temporal and/or frequency features of a raw EMG signal.

        :param kind: 'all', 'temp', or 'freq'.
        """
        if kind not in ('all', 'temp', 'freq'):
            raise ValueError("Not supported kind. It must be: all, temp, or freq.")

        proc_sig = self.bandpass(20, 450, order=4)

        funcst = self._temp_funcs(self, win_size)
        funcsf = self._freq_funcs(self.sr)

        if kind == 'all':
            funcs = {**funcst, **funcsf}
        elif kind == 'temp':
            funcs = funcst
        elif kind == 'freq':
            funcs = funcsf
        else:
            raise ValueError

        features = {'time': proc_sig.apply_running_win(lambda x, ax: x.squeeze(), win_size, win_inc).t}
        for name, func in funcs.items():
            features[name] = proc_sig.apply_running_win(func, win_size, win_inc)().flatten()

        return features
