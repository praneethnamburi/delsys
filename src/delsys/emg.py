"""EMG-specific signal class and feature extraction.

Defines :class:`EMG`, a :class:`pysampled.Data` extension that adds the
canonical EMG preprocessing pipeline (bandpass → rectify → amplitude →
optional lowpass), the Teager–Kaiser energy operator, and two feature
extractors (a NeuroKit2 wrapper and a hand-rolled temporal/frequency
feature dict).
"""

import datetime as _dt
import json
import os
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
import pysampled
from scipy.fftpack import fft, fftfreq

from delsys._metadata import SensorInfo
from delsys.signals import _bundle_sensors

#: Default output rate for amplitude envelopes, in Hz. Named because two things have to agree on
#: it: :meth:`EMG.rms` requests it, and :func:`cocontraction` resamples a pair onto it.
ENVELOPE_SR = 240.0

#: Sentinel for "no reference argument was given, so inherit whatever the Log stamped on".
#: Needed because ``None`` is a meaningful *value* for :func:`cocontraction` -- it says "compute
#: the index unnormalised" -- and cannot also serve as the default.
INHERIT = type("_Inherit", (), {"__repr__": lambda self: "INHERIT"})()


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
            if self.n_signals() > 1:
                raise NotImplementedError(
                    "EMG.process(amp_kind='nk') requires single-channel "
                    "input — NeuroKit2's emg_amplitude is 1D-only. Use "
                    "bundle.split_by_signal_name() and process each "
                    "per-channel slice."
                )
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
        envelope_sr: float = ENVELOPE_SR,
        normalize: bool = False,
    ) -> pysampled.Data:
        """RMS amplitude envelope on a clean filter chain.

        Pipeline: ``shift_baseline`` → highpass → lowpass → notch (power
        line) → running RMS over ``win_size``.

        .. warning::
           ``envelope_sr`` is a **request, not a guarantee.** The window step has to be a whole
           number of input samples, so ``pysampled`` delivers ``sr / round(sr / envelope_sr)`` --
           e.g. a request for 240 Hz gives 251.9 Hz on a 1259 Hz single-differential channel and
           246.9 Hz on a 2222 Hz Quattro. ``t0`` also lands on the centre of the first window,
           which differs with the window length in samples. **Two channels of different native
           rate therefore never share a time grid**, and cannot be combined sample-by-sample
           without resampling first -- which is what :func:`cocontraction` does.

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
            envelope_sr: Requested output sampling rate in Hz. Default 240; see the warning
                above -- the achieved rate is the nearest one a whole-sample step allows.
            normalize: Divide each channel by its amplitude reference, giving a **fraction** of
                that reference rather than mV. The reference comes from the Log
                (``Log(..., reference=ref)`` or ``lf.reference = ref``); raises if none is
                attached. See :func:`normalize`.

        Returns:
            A :class:`pysampled.Data` holding the RMS amplitude envelope, sampled at
            approximately ``envelope_sr`` (read ``.sr`` for the achieved rate; the history
            records both). ``meta`` carries the source sensor
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
            ("rms", {"win_size": win_size, "envelope_sr": envelope_sr,
                      "envelope_sr_achieved": float(envelope.sr)}),
        ]
        envelope.meta = dict(self.meta) if self.meta else {}
        return globals()["normalize"](envelope) if normalize else envelope

    def cocontraction(self, other: "EMG", **kwargs) -> pysampled.Data:
        """Co-contraction index against ``other`` -- see :func:`cocontraction`.

        Method form of the module function; the index is symmetric, so ``a.cocontraction(b)`` and
        ``b.cocontraction(a)`` agree.
        """
        return cocontraction(self, other, **kwargs)

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

        # Time vector at window centers, sampled from proc_sig's own time
        # grid. Going through ``apply_running_win`` here would force a
        # squeeze callable whose output shape collapses the channel axis,
        # which then fails pysampled's label/data validation.
        rw = proc_sig.make_running_win(win_size, win_inc)
        features: Dict[str, Any] = {
            "time": proc_sig.t[np.array(rw.center_idx)],
        }
        for name, func in funcs.items():
            features[name] = proc_sig.apply_running_win(func, win_size, win_inc)().flatten()

        return features


# ---------------------------------------------------------------------------
# Amplitude normalisation + co-contraction
# ---------------------------------------------------------------------------
#: Default statistic for :func:`reference`: the maximum of the smoothed/RMS envelope.
#: Halaki & Ginn (2012, doi:10.5772/49957) state the accepted practice directly -- "the maximum
#: value obtained from the processed signals during all repetitions of the test is then used as
#: the reference value for normalizing the EMG signals, processed in the same way". Note the
#: *processed*: the peak of an envelope, never a raw sample.
REFERENCE_STAT = "max"
#: Envelopes below this fraction of the reference are treated as numerically zero when forming
#: the co-contraction ratio, to keep 0/0 out of the arithmetic. Not a physiological threshold --
#: see :func:`cocontraction` for why the index tends to zero there anyway.
COCONTRACTION_EPS = 1e-9


class Reference(Mapping):
    """Per-channel amplitude reference for normalising EMG, with its provenance.

    A bare ``{channel: value}`` dict would lose the two things that make a normalised number
    interpretable months later: *what* was measured and *how* it was reduced. So this carries
    ``stat`` and ``source`` alongside ``values`` and round-trips to JSON.

    Channels are keyed by **name** (the channelmap location, e.g. ``"RForearmFlexors"``), never by
    sensor number: a sensor swap mid-study changes the number while the muscle stays the same, and
    normalising the flexors by the deltoid's maximum is the kind of error that never announces
    itself. Reads it as a mapping (``ref["RBiceps"]``, ``in``, ``len``).

    Args:
        values: ``{channel name: reference amplitude}`` in the units of the source EMG (mV).
        stat: How each value was reduced from the envelope (see :func:`reference`).
        source: Path of the trial the values came from -- an MVC take, or a task trial when
            peak-of-task normalisation is being used instead.
        created: ISO timestamp; filled in automatically.
    """

    def __init__(self, values: Dict[str, float], stat: str = REFERENCE_STAT,
                 source: Optional[str] = None, created: Optional[str] = None,
                 rms_kw: Optional[dict] = None) -> None:
        self.values = {str(k): float(v) for k, v in dict(values).items()}
        self.stat = str(stat)
        self.source = source
        self.created = created or _dt.datetime.now().isoformat(timespec="seconds")
        #: The envelope settings the reference was measured with. Recorded because "process the
        #: MVC exactly as the task signal" is the actual standard (Halaki & Ginn 2012; ISEK 1999,
        #: which requires the averaging interval be *reported*) -- so a mismatch between the two
        #: has to be visible rather than silent.
        self.rms_kw = dict(rms_kw or {})

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return (f"Reference({len(self.values)} channels, stat={self.stat!r}, "
                f"source={os.path.basename(self.source) if self.source else None!r})")

    def save(self, path: str) -> str:
        """Write to JSON. The project decides where -- a reference is per *session*, and delsys
        has no session concept, so it is never auto-discovered the way ``.delsys-events`` is."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"values": self.values, "stat": self.stat, "source": self.source,
                       "created": self.created, "rms_kw": self.rms_kw},
                      f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    @classmethod
    def load(cls, path: str) -> "Reference":
        """Read back a saved reference."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["values"], d.get("stat", REFERENCE_STAT), d.get("source"), d.get("created"),
                   d.get("rms_kw"))


def _channels(obj) -> List["EMG"]:
    """Single-channel EMG bundles from an EMG bundle or a Log."""
    emg = getattr(obj, "emg", obj)
    if emg is None:
        raise ValueError("no EMG channels found")
    return emg.split_by_signal_name() if emg.n_signals() > 1 else [emg]


def _reduce(env: np.ndarray, stat) -> float:
    """Reduce an amplitude envelope to one reference number."""
    env = np.asarray(env, dtype=float).reshape(-1)
    env = env[np.isfinite(env)]
    if env.size == 0:
        return float("nan")
    if isinstance(stat, (int, float)) and not isinstance(stat, bool):
        return float(np.percentile(env, float(stat)))
    if stat == "max":
        return float(env.max())
    if isinstance(stat, str) and stat.startswith("p"):
        return float(np.percentile(env, float(stat[1:])))
    raise ValueError(f"unknown reference stat {stat!r}; use 'max', 'pNN', or a percentile number")


def _check_rms_kw(rms_kw, caller):
    """Validate ``**rms_kw`` against :meth:`EMG.rms` before it is forwarded.

    A ``**kwargs`` pass-through fails deep inside the callee with a message naming a function the
    caller never invoked, and silently swallows a typo. Two cases are worth naming explicitly:
    wrapping the settings in a dict (``rms_kw=dict(win_size=0.2)``) instead of passing them
    directly, and ``normalize=True`` -- which would reduce a reference from an already-normalised
    envelope, or normalise twice inside :func:`cocontraction`.
    """
    import inspect

    allowed = [n for n, prm in inspect.signature(EMG.rms).parameters.items()
               if n not in ("self", "normalize")]
    if "rms_kw" in rms_kw:
        raise TypeError(
            f"{caller}() takes envelope settings directly, not wrapped in a dict -- write "
            f"{caller}(..., win_size=0.2), not {caller}(..., rms_kw=dict(win_size=0.2))")
    if "normalize" in rms_kw:
        raise TypeError(
            "reference() measures the reference FROM an unnormalised envelope; normalising first "
            "would make it circular (every channel would come back at 1.0)"
            if caller == "reference" else
            "cocontraction() normalises internally; normalize= would apply it twice. Control it "
            "with reference= instead -- omit to inherit, or pass None to stay unnormalised.")
    unknown = [k for k in rms_kw if k not in allowed]
    if unknown:
        raise TypeError(f"{caller}() got unexpected envelope setting(s) {unknown}; "
                        f"EMG.rms accepts {allowed}")


def _rms_params(env) -> dict:
    """The envelope settings recorded on a processed signal, or ``{}`` if it carries none.

    ``envelope_sr_achieved`` is dropped: it legitimately differs between channels of different
    native rate (see :meth:`EMG.rms`), so comparing it would fire on every mixed-sensor pair.
    """
    for name, params in reversed(list(getattr(env, "_history", []) or [])):
        if name == "rms":
            return {k: v for k, v in dict(params).items() if k != "envelope_sr_achieved"}
    return {}


def reference(source, stat=REFERENCE_STAT, source_path: Optional[str] = None,
              **rms_kw) -> Reference:
    """Per-channel amplitude reference from a designated trial.

    Pass whatever trial the *project* designates as the reference -- a maximal-effort (MVC) take,
    or an ordinary task trial when no usable MVC exists and peak-of-task normalisation is the
    fallback. Only ``Reference.source`` distinguishes the two, so the choice stays visible in the
    record rather than becoming a separate code path.

    Each channel is reduced from its :meth:`EMG.rms` envelope, not from the raw signal -- which is
    the accepted practice rather than a choice made here: Halaki & Ginn (2012,
    :doi:`10.5772/49957`) put it as "the maximum value obtained from the processed signals during
    all repetitions of the test ... processed in the same way". Hence ``stat="max"`` by default:
    the peak of an envelope, never of a raw sample. A percentile (``"p99"``, or a bare number) is
    available as a robustness option when a take carries a known artifact, but it is a departure
    from the cited practice -- say so if you use it.

    **Process the reference exactly as the task signal.** That is the load-bearing half of the same
    recommendation, and ISEK's 1999 standards require the averaging interval be *reported*, which
    presumes it is fixed. ``rms_kw`` is therefore stored on the returned :class:`Reference`, and
    :func:`normalize` warns when it does not match the envelope being normalised.

    Args:
        source: A :class:`~delsys.log.Log` or an EMG bundle.
        stat: ``"max"``, ``"pNN"`` (e.g. ``"p99"``), or a percentile as a number.
        source_path: Recorded as the reference's provenance; taken from a Log's ``fname``
            when not given.
        **rms_kw: Envelope settings, passed straight through to :meth:`EMG.rms` --
            ``win_size``, ``envelope_sr``, filter cutoffs. Give them **directly**, e.g.
            ``reference(lf, win_size=0.2)``. Use the SAME settings here as in the analysis, or
            the normalisation is against a differently-smoothed quantity (:func:`normalize`
            warns if they disagree).

    Returns:
        A :class:`Reference` keyed by channel name.
    """
    _check_rms_kw(rms_kw, "reference")
    if source_path is None:
        source_path = getattr(source, "fname", None)
    values, effective = {}, {}
    for ch in _channels(source):
        name = (list(ch.signal_names or []) or [None])[0]
        if name is None:
            continue
        env = ch.rms(**rms_kw)
        values[name] = _reduce(np.asarray(env()), stat)
        effective = effective or _rms_params(env)
    return Reference(values, stat=stat, source=source_path, rms_kw=effective)


def _resolve_reference(obj, reference):
    """The reference to use: an explicit one, else whatever the Log stamped onto the bundle."""
    if reference is not None:
        return reference
    meta = getattr(obj, "meta", None) or {}
    return meta.get("reference")


def normalize(data, reference=None):
    """Divide each channel by its reference amplitude -> **fraction** of reference.

    ``data`` is an amplitude envelope (the output of :meth:`EMG.rms`) or an EMG bundle; the
    reference is taken from ``data.meta`` when not passed, which is how ``Log(..., reference=ref)``
    reaches here without being threaded through every call.

    Channels are matched by name. A channel with no entry in the reference raises rather than
    passing through unnormalised -- a silently mixed-units array is worse than a stop.
    """
    ref = _resolve_reference(data, reference)
    if ref is None:
        raise ValueError(
            "no reference: pass reference=..., or attach one with Log(..., reference=ref) / "
            "lf.reference = ref")
    names = list(getattr(data, "signal_names", []) or [])
    x = np.asarray(data()).copy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not names:
        raise ValueError("cannot normalize: the data carries no signal_names to match on")
    missing = [n for n in names if n not in ref]
    if missing:
        raise KeyError(f"no reference amplitude for {missing}; reference has {sorted(ref)}")
    want, got = dict(getattr(ref, "rms_kw", {}) or {}), _rms_params(data)
    differs = {k: (want[k], got[k]) for k in want if k in got and want[k] != got[k]}
    if differs:
        warnings.warn(
            "normalizing against a reference processed differently: "
            + ", ".join(f"{k} {w!r} (reference) vs {g!r} (data)" for k, (w, g) in differs.items())
            + ". The standard practice is to process both identically (Halaki & Ginn 2012); "
            "the normalised values are otherwise a ratio of two different quantities.",
            RuntimeWarning, stacklevel=2)
    for i, n in enumerate(names):
        v = float(ref[n])
        x[:, i] = x[:, i] / v if np.isfinite(v) and v > 0 else np.nan
    out = data._clone(x.reshape(np.asarray(data()).shape))
    out.meta = dict(getattr(data, "meta", {}) or {})
    out.meta["normalized"] = {"stat": getattr(ref, "stat", None),
                              "source": getattr(ref, "source", None)}
    return out


def _on_common_grid(envs, sr):
    """Resample envelopes onto one time grid at ``sr`` over the span they share.

    Needed because :meth:`EMG.rms` cannot deliver a requested rate exactly (see its warning), so
    two channels of different native rate arrive on different grids with different ``t0``. Linear
    interpolation is adequate here and nowhere near the limiting approximation: an RMS envelope has
    already been smoothed over ``win_size`` (50 ms by default), so it carries nothing near the
    ~4 ms grid spacing.
    """
    t0 = max(float(e.t[0]) for e in envs)
    t1 = min(float(e.t[-1]) for e in envs)
    if not t1 > t0:
        raise ValueError(
            f"the envelopes share no time span (latest start {t0:.3f} s, earliest end {t1:.3f} s); "
            "a pair from different trials cannot be compared")
    t = t0 + np.arange(int(np.floor((t1 - t0) * sr)) + 1) / float(sr)
    out = [np.interp(t, np.asarray(e.t, dtype=float), np.asarray(e()).reshape(-1)) for e in envs]
    return t, out


def cocontraction(a, b, reference=INHERIT, min_activation: float = 0.0, **rms_kw):
    """Co-contraction index of an agonist/antagonist pair (Rudolph et al.).

    Per sample, with ``lo`` and ``hi`` the smaller and larger of the two normalised amplitudes::

        CCI = (lo / hi) * (lo + hi)

    The ratio term is 1 when the pair is balanced and 0 when one muscle carries everything; the sum
    term scales that by how hard they are both working, so balanced-but-quiet does not score like
    balanced-and-loud.

    Rudolph et al. 2000, *Knee Surg Sports Traumatol Arthrosc* 8(5):262-269,
    :doi:`10.1007/s001670000130`. Note that the published renderings of the formula are not
    consistent across the secondary literature that cites it; this is the common form, but check it
    against the primary paper before it carries a claim.

    **At rest the index is zero, and that is the limit, not a convention.** Because ``lo <= hi`` by
    construction the ratio lies in [0, 1], so ``CCI <= lo + hi``; as activation falls to zero the
    sum term drags the index to zero whatever the ratio does. The only care needed is arithmetic:
    ``0/0`` is guarded so it yields 0 rather than NaN.

    Both channels are enveloped here rather than accepted pre-computed, because the pair must share
    a filter chain and an output grid to be comparable -- and these pairs routinely cross sampling
    rates (a 1259 Hz single-differential flexor against a 2222 Hz Quattro extensor). That crossing
    is exactly why the envelopes are then **resampled onto a common grid**: ``EMG.rms`` cannot hit a
    requested ``envelope_sr`` exactly (its step is a whole number of input samples), so those two
    channels come back at 251.9 and 246.9 Hz with ``t0`` 133 us apart. Multiplying them
    sample-by-sample would silently drift the pair out of time. The result is on the requested
    ``envelope_sr``, over the span the two channels share.

    Args:
        a, b: Single-channel :class:`EMG` bundles. Order is irrelevant -- the index is symmetric.
        reference: A :class:`Reference`. **Omit it** to inherit whatever the Log stamped on
            (``Log(..., reference=ref)`` / ``lf.reference = ref``) -- the normal path, since the
            reference is set once per file. Pass ``reference=None`` *explicitly* to compute the
            index unnormalised even when one is attached: legitimate within one pair on one
            participant, but the sum term is then in raw mV and is not comparable across muscles
            or people.
        min_activation: Force the index to 0 where the larger envelope is below this (in reference
            units, so a fraction when normalised). Default 0 -- the formula already tends to zero,
            so this is only for trimming a noisy baseline.
        **rms_kw: Envelope settings, passed straight through to :meth:`EMG.rms` for both
            channels, e.g. ``cocontraction(a, b, win_size=0.2)``.

    Returns:
        A :class:`pysampled.Data` at ``envelope_sr``, unitless, ``meta["cocontraction"]`` recording
        the two channel names, whether it was normalised, and both native rates.

    Two properties of the index to carry into any interpretation:

    * **It is not scale-invariant.** The sum term inherits whatever the normalisation set, so
      values are comparable only across signals reduced by the same ``stat`` from references
      collected the same way. Two studies' CCIs are not on one scale.
    * **A low value is ambiguous** -- it can mean both muscles are quiet, or that one dominates.
      Report the two activations alongside the index, not the index alone (Carey, De Groote &
      Sawers, *PLOS One* 2026, :doi:`10.1371/journal.pone.0343081`, which also confirms that
      amplitude-driven indices of this family go to zero when the antagonist is inactive).
    """
    _check_rms_kw(rms_kw, "cocontraction")
    for ch, nm in ((a, "a"), (b, "b")):
        if ch is None or ch.n_signals() != 1:
            raise ValueError(f"cocontraction needs single-channel EMG; {nm} has "
                             f"{0 if ch is None else ch.n_signals()}")
    ea, eb = a.rms(**rms_kw), b.rms(**rms_kw)
    if reference is INHERIT:
        ref = _resolve_reference(a, None) or _resolve_reference(b, None)
    else:
        ref = reference                     # including an explicit None: do not normalise
    if ref is not None:
        ea, eb = normalize(ea, ref), normalize(eb, ref)
    sr = float(rms_kw.get("envelope_sr", ENVELOPE_SR))
    t, (xa, xb) = _on_common_grid((ea, eb), sr)
    lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(hi > max(COCONTRACTION_EPS, float(min_activation)),
                       (lo / hi) * (lo + hi), 0.0)
    out = pysampled.Data(cci, sr=sr, t0=float(t[0]))
    names = [(list(getattr(ch, "signal_names", []) or [None]) or [None])[0] for ch in (a, b)]
    out.meta = {"cocontraction": {"channels": names, "normalized": ref is not None,
                                  "reference_stat": getattr(ref, "stat", None),
                                  "envelope_sr": sr,
                                  "native_sr": [float(a.sr), float(b.sr)]}}
    return out
