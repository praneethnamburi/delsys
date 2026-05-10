"""EMG-specific signal class and feature extraction.

Defines :class:`EMG`, a :class:`pysampled.Data` extension that adds the
canonical EMG preprocessing pipeline (bandpass → rectify → amplitude →
optional lowpass), the Teager–Kaiser energy operator, and two feature
extractors (a NeuroKit2 wrapper and a hand-rolled temporal/frequency
feature dict).
"""

from typing import Any, Callable, Dict, List, Optional, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
import pysampled
from scipy.fftpack import fft, fftfreq

from delsys._metadata import SensorInfo
from delsys.signals import _bundle_sensors


class EMG(pysampled.Data):
    """EMG bundle: :class:`pysampled.Data` plus EMG preprocessing and features.

    Holds an EMG signal as either a 1-D ``(n_samples,)`` array (single
    channel) or a 2-D ``(n_samples, n_channels)`` array (Quattro / Duo).
    The :class:`SensorInfo` record for the source sensor lives in
    ``self.meta['sensor']`` and survives clone/filter/resample operations
    automatically; the convenience :attr:`sensor` property reads it back.

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            for emg in lf.emg:
                envelope = emg.process(amp_kind="envelope2")
                features = emg.get_features(kind="temp", win_size=0.25, win_inc=0.1)
    """

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

    def process_nk(self) -> Dict[str, Any]:
        """Process the EMG with NeuroKit2's :func:`nk.emg_process`.

        Returns:
            Dict of per-sample NeuroKit signal traces (clean signal,
            amplitude, activations, ...). Wrap in ``pd.DataFrame(...)`` if
            you want a DataFrame.
        """
        signals, _ = nk.emg_process(self().flatten(), self.sr)
        return signals.to_dict()

    def tkeo(self) -> "EMG":
        """Apply the Teager–Kaiser Energy Operator.

        TKEO improves onset detection by emphasizing transients in both
        amplitude and frequency. The output preserves shape, sampling rate,
        and ``meta`` (including ``sensor``) via :meth:`pysampled.Data._clone`.

        Returns:
            A new :class:`EMG` of the same shape and sampling rate.
        """
        tkeo = self().copy()
        tkeo[1:-1] = self._sig[1:-1] * self._sig[1:-1] - self._sig[:-2] * self._sig[2:]
        # Correct the data at the extremities
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
        return self._clone(tkeo, his_append=("Teager-Kaiser", None))

    def process(
        self,
        amp_kind: str = "envelope2",
        lowpass: Optional[Union[float, int]] = None,
        **kwargs: Any,
    ) -> "EMG":
        """Run the canonical EMG preprocessing pipeline.

        Steps: ``shift_baseline`` → bandpass 20–500 Hz → ``abs`` → amplitude
        extraction (per ``amp_kind``) → optional final lowpass.

        Args:
            amp_kind: Amplitude extractor — one of ``'rms'``, ``'mean'``,
                ``'envelope'``, ``'envelope2'``, or ``'nk'``.
            lowpass: Optional cutoff (Hz) for a final lowpass after the
                amplitude step. ``None`` skips it.
            **kwargs: Forwarded to the amplitude extractor. ``order`` (int)
                sets the bandpass/lowpass filter order (default 4).
                ``win_size`` and ``win_inc`` (seconds) configure the window
                for ``'rms'`` and ``'mean'``.

        Returns:
            A new :class:`EMG` holding the processed signal at the original
            sampling rate.

        Raises:
            ValueError: If ``amp_kind`` is not one of the supported values.
        """
        if amp_kind not in ("rms", "mean", "envelope", "envelope2", "nk"):
            raise ValueError(
                "Not supported kind. It must be: rms, mean, envelope, envelope2 or nk."
            )

        if "order" in kwargs:
            order = kwargs["order"]
            del kwargs["order"]
        else:
            order = 4
        # Bandpass: <20 Hz is motion noise, >450 Hz is electrical noise
        proc_sig12 = self.shift_baseline().bandpass(20, 500, order=order).apply(np.abs)

        if amp_kind == "rms":
            rms = lambda x, ax: np.sqrt(np.mean(x**2, axis=ax))  # noqa: E731
            proc_sig3 = proc_sig12.apply_running_win(rms, **kwargs).resample(self.sr)
        elif amp_kind == "mean":
            proc_sig3 = proc_sig12.apply_running_win(np.mean, **kwargs).resample(self.sr)
        elif amp_kind == "envelope":
            proc_sig3 = proc_sig12.envelope(lowpass=10)
        elif amp_kind == "envelope2":
            proc_sig3 = proc_sig12.envelope2(lowpass=10)
        elif amp_kind == "nk":
            proc_sig3 = nk.emg_amplitude(proc_sig12().flatten())
        else:
            raise ValueError

        # Carry the parent's meta (including ``sensor``) onto the new EMG so
        # downstream code can still ask ``processed.sensor.location`` etc.
        new_meta = dict(self.meta) if self.meta else {}

        if lowpass:
            return self.__class__(
                proc_sig3(),
                self.sr,
                axis=self.axis,
                t0=proc_sig3._t0,
                history=proc_sig3._history + [("preprocess_emg", amp_kind)],
                meta=new_meta,
            ).lowpass(lowpass, order=order)
        return self.__class__(
            proc_sig3(),
            self.sr,
            axis=self.axis,
            t0=proc_sig3._t0,
            history=proc_sig3._history + [("preprocess_emg", amp_kind)],
            meta=new_meta,
        )

    def rms(
        self,
        bandpass_low: float = 20.0,
        bandpass_high: float = 500.0,
        power_line_frequency: float = 60.0,
        win_size: float = 0.05,
        envelope_sr: float = 240.0,
    ) -> pysampled.Data:
        """RMS amplitude envelope on a clean filter chain.

        Pipeline: ``shift_baseline`` → highpass → lowpass → notch (power
        line) → running RMS over ``win_size``. The window step is set to
        ``1 / envelope_sr`` so the output sampling rate is exactly
        ``envelope_sr``.

        Notch-filtering the power line interference makes this preferable
        to ``process(amp_kind='rms')`` when working in line-frequency-noisy
        environments.

        Returns a plain :class:`pysampled.Data` rather than :class:`EMG`,
        because the amplitude envelope isn't an EMG signal anymore (different
        sampling rate, different units). The history of the filter chain and
        ``self.meta`` (including ``sensor``) are propagated onto the result.

        Args:
            bandpass_low: Highpass cutoff in Hz. Default 20.
            bandpass_high: Lowpass cutoff in Hz. Default 500 (drop to 450 if
                hardware bandwidth is the dominant constraint).
            power_line_frequency: Notch frequency in Hz. Default 60.
            win_size: RMS window length in seconds. Default 0.05.
            envelope_sr: Output sampling rate in Hz. Default 240.

        Returns:
            A :class:`pysampled.Data` holding the RMS amplitude envelope,
            sampled at ``envelope_sr``. ``meta`` carries the source sensor
            and ``_history`` reflects the full filter + RMS chain.

        Example:
            .. code-block:: python

                import delsys

                lf = delsys.Log("trial_01.csv")
                envelope = lf.emg[0].rms(envelope_sr=240)
                # Override defaults for hardware with narrower bandwidth:
                envelope = lf.emg[0].rms(bandpass_high=450)
        """
        _rms = lambda x, ax: np.sqrt(np.mean(x**2))  # noqa: E731
        filtered = (
            self.shift_baseline()
            .highpass(bandpass_low)
            .lowpass(bandpass_high)
            .notch(power_line_frequency)
        )
        envelope = filtered.apply_running_win(
            _rms,
            win_size=win_size,
            win_inc=1.0 / envelope_sr,
        )
        # ``apply_running_win`` in pysampled constructs a plain ``Data`` and
        # doesn't carry history / meta through (the sampling rate change is the
        # blocker for using ``_clone``). Patch them on after the fact so
        # downstream code can still ask ``envelope.sensor.location`` and read
        # the full processing chain.
        envelope._history = filtered._history + [
            ("rms", {"win_size": win_size, "envelope_sr": envelope_sr}),
        ]
        envelope.meta = dict(self.meta) if self.meta else {}
        return envelope

    @staticmethod
    def _temp_funcs(signal: pysampled.Data, win_size: float) -> Dict[str, Callable]:
        """Build a dict of temporal feature functions keyed by short name.

        Args:
            signal: Source signal — used to compute the activation threshold
                ``th = mean + 3 * std`` shared by ``wamp`` and ``myop``.
            win_size: Window length in seconds; baked into ``mav``, ``log``,
                and ``dsd``.

        Returns:
            Dict mapping feature name (e.g. ``'mean'``, ``'rms'``, ``'mav'``)
            to a callable ``f(x, ax) -> scalar`` suitable for
            :meth:`pysampled.Data.apply_running_win`.
        """
        th = np.mean(signal(), axis=signal.axis) + 3 * np.std(signal(), axis=signal.axis)
        funcs: Dict[str, Callable] = {
            "mean": np.mean,
            "med": np.median,
            "var": np.var,
            "rms": lambda x, ax: np.sqrt(np.mean(x**2, axis=ax)),
            "int": lambda x, ax: np.sum(np.abs(x), axis=ax),
            "mav": lambda x, ax: np.sum(np.abs(x), axis=ax) / win_size,
            "log": lambda x, ax: np.exp(np.sum(np.log10(np.abs(x))) / win_size),
            "wav": lambda x, ax: np.sum(np.abs(np.diff(x, axis=ax)), axis=ax),
            "dsd": lambda x, ax: np.sqrt(np.sum(np.diff(x, axis=ax) ** 2, axis=ax))
            / (win_size - 1),
            "zcr": lambda x, ax: len(np.where(np.diff(np.sign(x), axis=ax))[0]),
            "wamp": lambda x, ax: np.sum((np.abs(np.diff(x, axis=ax))) >= th, axis=ax),
            "myop": lambda x, ax: np.sum(x >= th, axis=ax) / len(x),
        }
        return funcs

    @staticmethod
    def _freq_funcs(sr: float) -> Dict[str, Callable]:
        """Build a dict of frequency-domain feature functions keyed by short name.

        Args:
            sr: Sampling rate in Hz; used by the per-window FFT.

        Returns:
            Dict mapping feature name (e.g. ``'mnf'``, ``'pkf'``, ``'frr'``)
            to a callable ``f(x, ax) -> scalar`` suitable for
            :meth:`pysampled.Data.apply_running_win`.
        """

        def spect(x):
            freq = fftfreq(x.size, d=1 / sr)
            power = (np.abs(fft(x)) ** 2).flatten()
            pos = np.where((freq > 0) & (freq < sr / 2))[0]
            return freq[pos], power[pos]

        def freq_rat(freq, power, ax):
            ulc = np.sum(power[(freq >= 20) & (freq <= 200)], axis=ax)
            uhc = np.sum(power[(freq > 200) & (freq <= 450)], axis=ax)
            return ulc / uhc

        funcs: Dict[str, Callable] = {
            "frr": lambda x, ax: freq_rat(*spect(x), ax),
            "mnp": lambda x, ax: np.mean(spect(x)[1], axis=ax),
            "vap": lambda x, ax: np.var(spect(x)[1], axis=ax),
            "mnf": lambda x, ax: np.mean(spect(x)[0], axis=ax),
            "vaf": lambda x, ax: np.var(spect(x)[0], axis=ax),
            "mwf": lambda x, ax: np.sum(spect(x)[0] * spect(x)[1], axis=ax)
            / np.sum(spect(x)[1], axis=ax),
            "mav": lambda x, ax: spect(x)[0][
                np.searchsorted(np.cumsum(spect(x)[1]), np.sum(spect(x)[1]) * 0.5)
            ],
            "pkf": lambda x, ax: spect(x)[0][spect(x)[1].argmax()],
        }
        return funcs

    def get_features_nk(self, method: str = "interval") -> Dict[str, Any]:
        """Compute EMG features via NeuroKit2's :func:`nk.emg_analyze`.

        Args:
            method: NeuroKit analysis method (``'interval'``, ``'event-related'``).

        Returns:
            Dict of per-feature values. Wrap in ``pd.DataFrame(...)`` if you
            want a DataFrame.
        """
        signals = pd.DataFrame(self.process_nk())
        signals = nk.emg_analyze(signals, self.sr, method=method)
        return signals.to_dict()

    def get_features(
        self,
        kind: str = "all",
        win_size: float = 0.25,
        win_inc: float = 0.1,
    ) -> Dict[str, Any]:
        """Hand-rolled temporal and/or frequency features over a sliding window.

        Args:
            kind: Which feature family to compute — ``'all'`` (temporal +
                frequency), ``'temp'``, or ``'freq'``.
            win_size: Window length in seconds.
            win_inc: Window step in seconds.

        Returns:
            Dict keyed by feature name. The ``'time'`` entry holds the
            per-window timestamps. Wrap in ``pd.DataFrame(...)`` if you want
            a DataFrame.

        Raises:
            ValueError: If ``kind`` is not one of ``'all'``, ``'temp'``,
                or ``'freq'``.
        """
        if kind not in ("all", "temp", "freq"):
            raise ValueError("Not supported kind. It must be: all, temp, or freq.")

        proc_sig = self.bandpass(20, 450, order=4)

        funcst = self._temp_funcs(self, win_size)
        funcsf = self._freq_funcs(self.sr)

        if kind == "all":
            funcs = {**funcst, **funcsf}
        elif kind == "temp":
            funcs = funcst
        elif kind == "freq":
            funcs = funcsf
        else:
            raise ValueError

        features: Dict[str, Any] = {
            "time": proc_sig.apply_running_win(lambda x, ax: x.squeeze(), win_size, win_inc).t,
        }
        for name, func in funcs.items():
            features[name] = proc_sig.apply_running_win(func, win_size, win_inc)().flatten()

        return features
