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
        self.meta["rpeaks_idx_autopruned"] = []
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

        Raises:
            NotImplementedError: If the bundle holds more than one channel
                (i.e. an aggregate EKG). HeartPy expects a 1D series; use
                ``ekg.split_by_signal_name()[i]`` to pick one channel
                first.
        """
        if self.n_signals() > 1:
            raise NotImplementedError(
                "find_rpeaks_pn requires a single EKG channel. Use "
                "ekg['<sensor_location>'] or "
                "ekg.split_by_signal_name()[i] to select one."
            )
        sig = copy.deepcopy(self)
        if "is_flipped" in self.meta and self.meta["is_flipped"]:
            sig = sig.apply(lambda x: -x)
        if highpass is not None:
            sig = sig.highpass(highpass)
        peak_idx = hp.process(sig().flatten(), sig.sr)[0]["peaklist"]
        self.meta["rpeaks_idx_default"] = peak_idx
        # Record the detector provenance so a persisted decision can reproduce
        # this exact default set on reload (see :meth:`rpeaks_decision`).
        self.meta["_rpeak_detector"] = {"name": "pn", "highpass": highpass, "hr_max": hr_max}

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
        # Track the detector's own double-peak prune separately from human
        # removals, so a persisted decision stores only *human* intent and the
        # grid-specific auto-prune is regenerated by re-running the detector
        # (see :meth:`rpeaks_decision`). An auto-pruned double sits ~1 beat-width
        # from a real peak, so persisting it would snap to and kill that real peak
        # on a grid where the spurious double never appears.
        self.meta["rpeaks_idx_autopruned"] = list(peak_idx_to_remove)
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

    # ------------------------------------------------------------------
    # Grid-independent review decision (times, not indices) — the durable
    # form persisted to the ``rpeaks`` type of a ``<stem>.delsys-events``
    # sidecar. ``final_peaks = f(raw_ekg, decision)`` reproduces on any grid
    # (native-rate ``.h5`` reload, a slice). See
    # ``.claude-prompts/plan-0.5.0-ekg-rpeak-review.md``.
    # ------------------------------------------------------------------

    def rpeaks_decision(self) -> Dict[str, Any]:
        """Serialize the current curation as a **times-based** decision dict.

        The inverse of :meth:`apply_rpeaks_decision`. Peak sample indices in
        :attr:`meta` are converted to seconds on this Log's clock so the record
        survives resampling / reload on a different sample grid.

        Returns:
            ``{"detector", "added", "removed", "flipped", "tags"}`` where ``added``
            / ``removed`` are peak **times**. ``removed`` holds only *human*
            removals (``rpeaks_idx_removed`` minus the detector's own auto-prune) —
            the auto-prune is grid-specific and regenerated by re-running the
            detector on reload.

        Raises:
            NotImplementedError: On a multi-channel (aggregate) EKG.
        """
        self._require_single_channel("rpeaks_decision")
        det = self.meta.get("_rpeak_detector") or {"name": "pn", "highpass": 5.0, "hr_max": 200.0}
        t = self.t
        human_removed = sorted(
            set(self.meta.get("rpeaks_idx_removed", []))
            - set(self.meta.get("rpeaks_idx_autopruned", []))
        )
        return {
            "detector": dict(det),
            "added": [float(t[i]) for i in self.meta.get("rpeaks_idx_added", [])],
            "removed": [float(t[i]) for i in human_removed],
            "flipped": bool(self.meta.get("is_flipped", False)),
            "tags": list(self.meta.get("tags", [])),
        }

    def apply_rpeaks_decision(
        self,
        decision: Dict[str, Any],
        noise_windows: Optional[List[List[float]]] = None,
    ) -> List[int]:
        """Reproduce a persisted curation on **this** EKG's sample grid.

        Re-runs the recorded detector (honouring ``flipped``) to regenerate the
        default set + its double-peak prune, then applies the human diff:
        ``added`` peaks are placed at the nearest sample to each time; ``removed``
        peaks are matched to the nearest *default* peak and unioned with the fresh
        auto-prune; ``noise_windows`` become noisy segments. The result is the same
        curated peak set the reviewer saw, re-resolved to this grid.

        Args:
            decision: A dict from :meth:`rpeaks_decision` /
                :func:`delsys._events.read_rpeaks_signals`.
            noise_windows: ``[[t0, t1], ...]`` in seconds (from the sidecar's
                ``noise`` track for this channel), masked as noisy segments.

        Returns:
            The curated R-peak sample indices (``default ∪ added − removed``,
            outside noisy segments handled downstream by :meth:`rpeak_times`).

        Raises:
            NotImplementedError: On a multi-channel (aggregate) EKG.
            ValueError: On an unrecognized detector name.
        """
        self._require_single_channel("apply_rpeaks_decision")
        det = decision.get("detector") or {}
        name = det.get("name", "pn")
        if name != "pn":
            raise ValueError(f"unknown rpeak detector {name!r}; only 'pn' is supported.")
        self.meta["is_flipped"] = bool(decision.get("flipped", False))
        # Reproduce the default set + auto-prune deterministically (honours flip).
        self.find_rpeaks_pn(
            highpass=det.get("highpass", 5.0),
            hr_max=det.get("hr_max", 200.0),
        )
        self.meta["rpeaks_idx_added"] = self._times_to_sample_idx(decision.get("added", []))
        human_removed = self._times_to_default_peak_idx(decision.get("removed", []))
        self.meta["rpeaks_idx_removed"] = sorted(
            set(self.meta.get("rpeaks_idx_removed", [])) | set(human_removed)
        )
        if noise_windows:
            self.meta["noisy_segments_idx"] = self._windows_to_idx_pairs(noise_windows)
        self.meta["tags"] = list(decision.get("tags", []))
        return self._get_rpeaks_from_meta()

    def _require_single_channel(self, who: str) -> None:
        if self.n_signals() > 1:
            raise NotImplementedError(
                f"{who} requires a single EKG channel. Use "
                "ekg['<sensor_location>'] or ekg.split_by_signal_name()[i]."
            )

    def _times_to_sample_idx(self, times: List[float]) -> List[int]:
        """Nearest sample index for each time (for human-*added* peaks)."""
        t = self.t
        return sorted({int(np.argmin(np.abs(t - float(x)))) for x in (times or [])})

    def _times_to_default_peak_idx(self, times: List[float], tol: float = 0.1) -> List[int]:
        """Nearest *default-peak* index for each removed time, within ``tol`` s.

        A removed peak references a detected (default) peak, so it snaps to the
        nearest default peak rather than the nearest raw sample. A removed time
        with **no** default peak within ``tol`` is skipped — the peak it referenced
        isn't present on this grid (e.g. a boundary double-peak the detector prunes
        on one sample rate but never emits on another), so there is nothing to
        remove and it must not snap to and kill an unrelated real peak.
        """
        default = list(self.meta.get("rpeaks_idx_default", []))
        if not default or not times:
            return []
        dt = self.t[np.array(default)]
        out = []
        for x in times:
            j = int(np.argmin(np.abs(dt - float(x))))
            if abs(float(dt[j]) - float(x)) <= tol:
                out.append(int(default[j]))
        return sorted(set(out))

    def _windows_to_idx_pairs(self, windows: List[List[float]]) -> List[List[int]]:
        """Convert ``[[t0, t1], ...]`` noise windows to ``[[i0, i1], ...]`` pairs."""
        t = self.t
        out: List[List[int]] = []
        for w in windows or []:
            if w is None or len(w) < 2:
                continue
            a = int(np.argmin(np.abs(t - float(w[0]))))
            b = int(np.argmin(np.abs(t - float(w[1]))))
            if b < a:
                a, b = b, a
            out.append([a, b])
        return out
