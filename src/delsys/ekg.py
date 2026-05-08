"""EKG-specific signal class with R-peak detection, HRV, and respiration rate."""
import bisect
import copy
import json
import os
from pathlib import Path
from typing import Tuple, Union
from warnings import warn

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from delsys._datamod import DataMod
from delsys._util import decreturn


class EKG(DataMod):
    """ECG bundle with R-peak detection, HRV, and respiration-rate extraction.

    CAUTION: Slicing will not work well with the cached r-peak indices.
    TODO: modify the ``__getitem__`` method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_meta_for_rpeaks()

    def _initialize_meta_for_rpeaks(self):
        if self.meta is None:
            self.meta = {}
        self.meta['rpeaks_idx_default'] = []
        self.meta['rpeaks_idx_removed'] = []
        self.meta['rpeaks_idx_added'] = []
        self.meta['noisy_segments_idx'] = []
        self.meta['is_flipped'] = False
        self.meta['tags'] = []  # e.g. reviewed, representative, interesting

    @property
    def hr(self) -> float:
        """Mean heart rate over the ECG signal."""
        return hp.process(self().flatten(), self.sr)[1]['bpm']

    @property
    def rr(self) -> float:
        """Mean respiration rate over the ECG signal."""
        return hp.process(self().flatten(), self.sr)[1]['breathingrate']

    @decreturn
    def process_nk(self, method='neurokit', to=dict):
        """Process ECG with NeuroKit2 and return key signals."""
        signals, _ = nk.ecg_process(self().flatten(), self.sr, method=method)
        return signals.to_dict()

    @decreturn
    def get_features_hp(self, win_size: float = 5., win_inc: float = .1, to=dict, **kwargs):
        """ECG features via HeartPy, segmentwise.

        :param win_size: Window width (seconds).
        :param win_inc: Window step (seconds).
        """
        win_over = float(np.clip(1 - win_inc / win_size, 0., 0.999))
        result = hp.process_segmentwise(
            self().flatten(), self.sr, segment_width=win_size, segment_overlap=win_over, **kwargs,
        )[1]
        result['time'] = [(round(self.t[idx[0]], 2), round(self.t[idx[1]], 2)) for idx in result['segment_indices']]
        return result

    def _get_sav_name(self):
        if hasattr(self, 'parent'):
            return os.path.join(Path(self.parent.fname).parent, Path(self.parent.fname).stem + '_ekginfo.json')
        if 'sav_name' in self.meta:
            return self.meta['sav_name']
        assert hasattr(self, 'sav_name')
        return self.sav_name

    # TODO: Modify path to be an attribute from Signal so there will be no need to include it as an argument.
    def find_rpeaks_hp(
            self, path=None, to: str = 'idx', cleaned: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Find R-peaks using HeartPy. Optionally apply manual-cleaning JSON.

        :param to: 'idx' (array of indices) or 'data' (times + values).
        :param cleaned: If True, apply manual add/remove from JSON file.
        """
        if path is None:
            path = self._get_sav_name()

        if to not in ('idx', 'data'):
            raise ValueError('Value of argument to not identified. Either idx or data.')

        if cleaned:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    modifier = json.load(f)
            else:
                warn(f'JSON File {path} does not exist. Not previously hand-cleaned.')
                modifier = {'add': [], 'remove': []}
                with open(path, 'w') as f:
                    json.dump(modifier, f)

        peaks = hp.process(self().flatten(), self.sr)[0]['peaklist']

        if cleaned:
            for peak_rem in modifier['remove']:
                try:
                    peaks.remove(peak_rem)
                except ValueError:
                    print('Manually removed peak does not exist in the automatic peak detection. '
                          'Check if correct JSON file is being used.')
            bisect.insort(peaks, modifier['add'])  # Insert in order
            peaks = list(set(peaks))  # Remove duplicates

        if to == 'idx':
            return np.array(peaks)
        else:
            return self.t[peaks], self().flatten()[peaks]

    def find_rpeaks_nk(self, to: str = 'idx', **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Find R-peaks using NeuroKit2."""
        ecg = nk.ecg_clean(self().flatten(), **kwargs)
        data, _ = nk.ecg_peaks(ecg, **kwargs)
        peaks = np.where(data == 1)[0]

        if to == 'idx':
            return np.array(peaks)
        else:
            return self.t[peaks], self().flatten()[peaks]

    def find_rpeaks_pn(self, highpass=5, hr_max=200):
        """Modified HeartPy R-peak detection: highpass first, then prune spurious double-peaks.

        :param highpass: Highpass cutoff (Hz) before HeartPy. Default 5.
        :param hr_max: Maximum plausible HR (bpm); peaks producing higher inferred HR
                       are pruned. Default 200.

        Note: edits ``self.meta`` in place; returns indices via ``_get_rpeaks_from_meta``.
        """
        sig = copy.deepcopy(self)
        if 'is_flipped' in self.meta and self.meta['is_flipped']:
            sig = sig.apply(lambda x: -x)
        if highpass is not None:
            sig = sig.highpass(highpass)
        peak_idx = hp.process(sig().flatten(), sig.sr)[0]['peaklist']
        self.meta['rpeaks_idx_default'] = peak_idx

        # Prune sudden jumps to high HR
        ihr_bpm = 60 / np.diff(self.t[peak_idx])
        double_pk_locs = np.where(ihr_bpm > hr_max)[0]
        peak_idx_to_remove = []
        for double_pk_loc in double_pk_locs:
            val1 = sig()[peak_idx[double_pk_loc]]
            val2 = sig()[peak_idx[double_pk_loc + 1]]
            if val1 > val2:
                peak_idx_to_remove.append(peak_idx[double_pk_loc + 1])
            else:
                peak_idx_to_remove.append(peak_idx[double_pk_loc])
        self.meta['rpeaks_idx_removed'] = peak_idx_to_remove
        return self._get_rpeaks_from_meta()

    find_rpeaks = find_rpeaks_pn  # alias

    def _get_rpeaks_from_meta(self):
        x = list(
            set(list(self.meta['rpeaks_idx_default']) + list(self.meta['rpeaks_idx_added']))
            - set(self.meta['rpeaks_idx_removed'])
        )
        x.sort()
        return x

    def _get_t_noisy_segments(self):
        assert 'noisy_segments_idx' in self.meta
        if len(self.meta['noisy_segments_idx']) > 0:
            return np.sort(np.array(self.t[np.array(self.meta['noisy_segments_idx']).flatten()]))
        return np.empty(0)

    def rpeak_times(self):
        if len(self.meta['rpeaks_idx_default']) == 0:
            self.find_rpeaks()
        x = np.array(self._get_rpeaks_from_meta())

        # Identify all r-peaks that fall inside noisy segments
        rpeak_in_ns = np.zeros(len(x), dtype=bool)
        if len(self.meta['noisy_segments_idx']) > 0:
            for xi, xval in enumerate(x):
                for ns_idx in self.meta['noisy_segments_idx']:
                    if (xval > ns_idx[0] and xval < ns_idx[1]):
                        rpeak_in_ns[xi] = True
                        break

        pk_idx = np.r_[:len(x)][~rpeak_in_ns]
        x = x[~rpeak_in_ns]
        return x, pk_idx

    def ihr(self):
        """Instantaneous heart rate. Returns ``(times, bpm_values)``; ``np.nan`` across noisy gaps."""
        x, pk_idx = self.rpeak_times()
        xdata = []
        ydata = []
        this_t = self.t[x]
        for cnt in range(len(x) - 1):
            xdata.append((this_t[cnt] + this_t[cnt + 1]) / 2)
            if pk_idx[cnt + 1] - pk_idx[cnt] == 1:  # not interrupted by a noisy segment
                ydata.append(60 / (this_t[cnt + 1] - this_t[cnt]))
            else:
                ydata.append(np.nan)

        return xdata, ydata

    def flip_signal(self):
        """Flip the signal and re-process."""
        if 'is_flipped' not in self.meta:
            self.meta['is_flipped'] = False
        self.meta['is_flipped'] = not self.meta['is_flipped']
        self.find_rpeaks()

    def clean_rpeaks(self, path=None, action='all'):
        """Manually add or remove R-peaks via an interactive plot."""
        if path is None:
            path = self._get_sav_name()

        exist_flag = os.path.exists(path)
        if exist_flag:
            with open(path, 'r') as f:
                modifier = json.load(f)
        else:
            modifier = {'add': [], 'remove': []}
            with open(path, 'w') as f:
                json.dump(modifier, f)

        r_idx = self.find_rpeaks_hp(path, to='idx', cleaned=exist_flag)

        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        ax.plot(self.t, self()), ax.plot(self.t[r_idx], self()[r_idx], '*')
        ax.set_title('ECG Signal'), ax.set_xlabel('Time [s]'), ax.set_ylabel('Voltage [V]')
        fig.show()

        def onpick(event):
            xdata = event.artist.get_xdata()
            ydata = event.artist.get_ydata()
            sel_idx = event.ind
            des_idx = sel_idx[np.argmax(ydata[sel_idx])]
            ax.plot(xdata[des_idx], ydata[des_idx], '*')
            if action == 'all':
                if des_idx in r_idx:
                    modifier['remove'].append(des_idx)
                else:
                    modifier['add'].append(des_idx)
            elif action == 'add' or action == 'remove':
                modifier[action].append(des_idx)
            else:
                raise ValueError(f'Action {action} does not exist. Options: all, add, remove')
        fig.canvas.mpl_connect('pick_event', onpick)

        with open(path, 'w') as f:
            json.dump(modifier, f)

    def hrv(self, path, cleaned: bool = False):
        """Heart rate variability series, returned as an EKG with the same sampling rate."""
        time, _ = self.find_rpeaks_hp(path, to='data', cleaned=cleaned)
        vals = 60 / np.diff(time)
        proc_sig = interp1d(time[1:], vals, kind='cubic', axis=self.axis)(np.arange(time[1], time[-1], 1 / self.sr))

        self._history += [('HRV computation', None)]
        return self.__class__(proc_sig, self.sr, axis=self.axis, t0=time[1], history=self._history)

    def find_rr_nk(self, path, cleaned: bool = False, method='vangent2019'):
        """Respiration rate from ECG via NeuroKit2.

        :param method: 'vangent2019', 'soni2019', 'charlton2016', or 'sarkar2015'.
        """
        hrv = self.hrv(path, cleaned=cleaned)
        rr = nk.ecg_rsp(hrv(), sampling_rate=hrv.sr, method=method)
        return self._clone(rr, ('respiration_rate', method))

    def find_rr(self, path, cleaned: bool = False):
        """Respiration rate from ECG using extrema interpolation."""
        time, vals = self.find_rpeaks_hp(path, to='data', cleaned=cleaned)
        rr_idx = argrelextrema(vals, np.greater)
        proc_sig = interp1d(
            time[rr_idx], vals[rr_idx], kind='cubic', axis=self.axis,
        )(np.arange(time[rr_idx][0], time[rr_idx][-1], 1 / self.sr))
        self._history += [('respiration_rate', 'Own')]
        return self.__class__(proc_sig, self.sr, axis=self.axis, t0=time[rr_idx][0], history=self._history)
