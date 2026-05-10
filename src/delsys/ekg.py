"""EKG-specific signal class and analysis utilities.

Defines :class:`EKG`, a :class:`pysampled.Data` extension that adds:

* R-peak detection (highpass + HeartPy + heuristic prune for spurious double
  peaks).
* Aggregate heart-rate and respiration-rate properties.
* Manual annotation slots in ``self.meta`` (added/removed/noisy peaks,
  flip flag, free-text tags).
* Per-segment feature extraction via HeartPy.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import heartpy as hp
import neurokit2 as nk
import numpy as np
import pysampled

from delsys._metadata import SensorInfo
from delsys.signals import _bundle_sensors


class EKG(pysampled.Data):
    """ECG bundle with R-peak detection, HRV-relevant metadata, and rate properties.

    R-peak indices and manual annotations live in :attr:`pysampled.Data.meta`
    under fixed keys initialized by :meth:`_initialize_meta_for_rpeaks`:

    +-------------------------+-----------------------------------------------+
    | meta key                | Contents                                      |
    +=========================+===============================================+
    | ``rpeaks_idx_default``  | Auto-detected R-peak sample indices.          |
    +-------------------------+-----------------------------------------------+
    | ``rpeaks_idx_added``    | User-added peaks merged into the result.      |
    +-------------------------+-----------------------------------------------+
    | ``rpeaks_idx_removed``  | User-removed peaks subtracted from the result.|
    +-------------------------+-----------------------------------------------+
    | ``noisy_segments_idx``  | List of ``(start_idx, end_idx)`` pairs marking|
    |                         | unusable spans; peaks inside them are dropped |
    |                         | by :meth:`rpeak_times`.                       |
    +-------------------------+-----------------------------------------------+
    | ``is_flipped``          | Whether to negate before peak detection.      |
    +-------------------------+-----------------------------------------------+
    | ``tags``                | Free-text tags (``'reviewed'``,               |
    |                         | ``'representative'``, ``'interesting'``).     |
    +-------------------------+-----------------------------------------------+

    .. note::

        Slicing currently does **not** reindex these cached R-peak entries;
        a sliced EKG still references peaks at the parent's sample positions.
        See :file:`TODO.md` for the planned ``__getitem__`` fix.

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            ekg = lf.ekg[0]
            peaks = ekg.find_rpeaks()       # alias for find_rpeaks_pn
            t_peaks = ekg.t[peaks]
            t_window, bpm = ekg.ihr()       # instantaneous HR over time
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_meta_for_rpeaks()

    def _initialize_meta_for_rpeaks(self) -> None:
        """Populate :attr:`meta` with the canonical R-peak annotation keys.

        Idempotent: replaces any existing values. Called from ``__init__``,
        so freshly constructed (or freshly cloned) EKGs always start with a
        clean annotation slate.
        """
        if self.meta is None:
            self.meta = {}
        self.meta["rpeaks_idx_default"] = []
        self.meta["rpeaks_idx_removed"] = []
        self.meta["rpeaks_idx_added"] = []
        self.meta["noisy_segments_idx"] = []
        self.meta["is_flipped"] = False
        self.meta["tags"] = []  # e.g. reviewed, representative, interesting

    @property
    def sensor(self) -> Optional[SensorInfo]:
        """The :class:`SensorInfo` record, or ``None`` if not set.

        ``None`` on aggregate views — use :attr:`sensors` (plural) to get
        the per-channel list.
        """
        return self.meta.get("sensor") if self.meta else None

    @property
    def sensors(self) -> List[SensorInfo]:
        """All :class:`SensorInfo` records this bundle carries.

        See :func:`delsys.signals._bundle_sensors`.
        """
        return _bundle_sensors(self)

    @property
    def shape(self) -> tuple:
        """Shape of the underlying sample array."""
        return self._sig.shape

    @property
    def hr(self) -> float:
        """Mean heart rate (bpm) over the full signal, via HeartPy."""
        return hp.process(self().flatten(), self.sr)[1]["bpm"]

    @property
    def rr(self) -> float:
        """Mean respiration rate (breaths per minute) over the full signal, via HeartPy."""
        return hp.process(self().flatten(), self.sr)[1]["breathingrate"]

    def process_nk(self, method: str = "neurokit") -> Dict[str, Any]:
        """Process the ECG with NeuroKit2's :func:`nk.ecg_process`.

        Args:
            method: Cleaning method passed to NeuroKit (``'neurokit'``,
                ``'biosppy'``, ``'pantompkins1985'``, ``'hamilton2002'``,
                ``'elgendi2010'``, ``'engzeemod2012'``).

        Returns:
            Dict of per-sample NeuroKit signal traces (``ECG_Clean``,
            ``ECG_R_Peaks``, ``ECG_Rate``, ...). Wrap in ``pd.DataFrame(...)``
            if you want a DataFrame.
        """
        signals, _ = nk.ecg_process(self().flatten(), self.sr, method=method)
        return signals.to_dict()

    def get_features_hp(
        self,
        win_size: float = 5.0,
        win_inc: float = 0.1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Segment-wise ECG features via HeartPy's :func:`hp.process_segmentwise`.

        Args:
            win_size: Segment width in seconds.
            win_inc: Segment step in seconds.
            **kwargs: Forwarded to :func:`hp.process_segmentwise`.

        Returns:
            Dict keyed by HeartPy metric name (e.g. ``'bpm'``, ``'sdnn'``,
            ``'rmssd'``). The added ``'time'`` entry holds
            ``(t_start, t_end)`` pairs for each segment. Wrap in
            ``pd.DataFrame(...)`` if you want a DataFrame.
        """
        win_over = float(np.clip(1 - win_inc / win_size, 0.0, 0.999))
        result = hp.process_segmentwise(
            self().flatten(),
            self.sr,
            segment_width=win_size,
            segment_overlap=win_over,
            **kwargs,
        )[1]
        result["time"] = [
            (round(self.t[idx[0]], 2), round(self.t[idx[1]], 2))
            for idx in result["segment_indices"]
        ]
        return result

    def find_rpeaks_pn(self, highpass: float = 5.0, hr_max: float = 200.0) -> List[int]:
        """Detect R-peaks: highpass-prefilter, then HeartPy, then prune double peaks.

        HeartPy occasionally reports two peaks for a single QRS complex
        when the signal is noisy. This method post-processes its output by
        comparing the inter-peak interval to ``hr_max`` and dropping the
        smaller-amplitude peak from each pair that implies an implausibly
        high heart rate.

        Args:
            highpass: Highpass cutoff (Hz) applied before HeartPy. ``None`` to
                skip the prefilter. Default 5.
            hr_max: Maximum plausible heart rate (bpm). Pairs whose inferred
                inter-peak rate exceeds this are pruned. Default 200.

        Returns:
            Sorted list of R-peak sample indices, with manual additions
            merged in and manual removals subtracted.

        Side effects:
            Edits :attr:`meta` in place — sets ``rpeaks_idx_default`` and
            ``rpeaks_idx_removed``. Manual ``rpeaks_idx_added`` entries are
            preserved.
        """
        sig = copy.deepcopy(self)
        if "is_flipped" in self.meta and self.meta["is_flipped"]:
            sig = sig.apply(lambda x: -x)
        if highpass is not None:
            sig = sig.highpass(highpass)
        peak_idx = hp.process(sig().flatten(), sig.sr)[0]["peaklist"]
        self.meta["rpeaks_idx_default"] = peak_idx

        # Prune sudden jumps to high HR (likely double-detected peaks).
        ihr_bpm = 60 / np.diff(self.t[peak_idx])
        double_pk_locs = np.where(ihr_bpm > hr_max)[0]
        peak_idx_to_remove: List[int] = []
        for double_pk_loc in double_pk_locs:
            val1 = sig()[peak_idx[double_pk_loc]]
            val2 = sig()[peak_idx[double_pk_loc + 1]]
            if val1 > val2:
                peak_idx_to_remove.append(peak_idx[double_pk_loc + 1])
            else:
                peak_idx_to_remove.append(peak_idx[double_pk_loc])
        self.meta["rpeaks_idx_removed"] = peak_idx_to_remove
        return self._get_rpeaks_from_meta()

    find_rpeaks = find_rpeaks_pn  #: Canonical alias for :meth:`find_rpeaks_pn`.

    def _get_rpeaks_from_meta(self) -> List[int]:
        """Resolve ``meta``'s default + added − removed sets into a sorted index list."""
        x = list(
            set(list(self.meta["rpeaks_idx_default"]) + list(self.meta["rpeaks_idx_added"]))
            - set(self.meta["rpeaks_idx_removed"])
        )
        x.sort()
        return x

    def _get_t_noisy_segments(self) -> np.ndarray:
        """Sample times of the start/end indices of every noisy segment."""
        assert "noisy_segments_idx" in self.meta
        if len(self.meta["noisy_segments_idx"]) > 0:
            return np.sort(np.array(self.t[np.array(self.meta["noisy_segments_idx"]).flatten()]))
        return np.empty(0)

    def rpeak_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return R-peak sample indices that fall **outside** noisy segments.

        Calls :meth:`find_rpeaks` first if the meta cache is empty.

        Returns:
            ``(idx, pk_idx)`` where ``idx`` is the array of clean R-peak
            sample indices, and ``pk_idx`` is the index of each surviving
            peak within the original (pre-filtered) ordered peak list. The
            second array is what :meth:`ihr` uses to detect peaks straddling
            a noisy segment.
        """
        if len(self.meta["rpeaks_idx_default"]) == 0:
            self.find_rpeaks()
        x = np.array(self._get_rpeaks_from_meta())

        # Identify all r-peaks that fall inside noisy segments
        rpeak_in_ns = np.zeros(len(x), dtype=bool)
        if len(self.meta["noisy_segments_idx"]) > 0:
            for xi, xval in enumerate(x):
                for ns_idx in self.meta["noisy_segments_idx"]:
                    if xval > ns_idx[0] and xval < ns_idx[1]:
                        rpeak_in_ns[xi] = True
                        break

        pk_idx = np.r_[: len(x)][~rpeak_in_ns]
        x = x[~rpeak_in_ns]
        return x, pk_idx

    def ihr(self) -> Tuple[List[float], List[float]]:
        """Instantaneous heart rate.

        Returns:
            ``(times, bpm)`` — the timestamps mid-way between consecutive
            clean R-peaks and the corresponding instantaneous HR. ``np.nan``
            is inserted across noisy-segment gaps so that consumers can
            distinguish missing data from continuous low values.
        """
        x, pk_idx = self.rpeak_times()
        xdata: List[float] = []
        ydata: List[float] = []
        this_t = self.t[x]
        for cnt in range(len(x) - 1):
            xdata.append((this_t[cnt] + this_t[cnt + 1]) / 2)
            if pk_idx[cnt + 1] - pk_idx[cnt] == 1:  # not interrupted by a noisy segment
                ydata.append(60 / (this_t[cnt + 1] - this_t[cnt]))
            else:
                ydata.append(np.nan)

        return xdata, ydata

    def flip_signal(self) -> None:
        """Toggle the ``is_flipped`` flag and re-detect R-peaks.

        Useful when an ECG channel was recorded with reversed polarity:
        :meth:`find_rpeaks_pn` will negate the signal before detection.
        """
        if "is_flipped" not in self.meta:
            self.meta["is_flipped"] = False
        self.meta["is_flipped"] = not self.meta["is_flipped"]
        self.find_rpeaks()
