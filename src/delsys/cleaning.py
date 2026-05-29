"""Multi-stage EMG cleaning pipeline.

This module ports the staged cleaner from
``pn-projects/projects/emg_ica_cleaning.py`` into the package. Three
stages, all optional via :class:`CleaningConfig`:

1. **Preprocess** — :func:`pysampled.Data.interpnan` plus an optional
   high/low-pass.
2. **ECG suppression** — fit FastICA on the EMG matrix concatenated with
   the EKG reference, score components by lagged correlation against the
   EKG, drop the worst offenders, then ridge-regress the EKG residual
   out of every channel.
3. **Motion suppression** — per-channel ridge regression of lagged ACC
   predictors with safety gates that reject regressions which would
   shrink the signal below a variance / power threshold.

The high-level entry point is :class:`Log.clean_emg_ekg_artifact` (in
:mod:`delsys.log`); :func:`run_pipeline` is the lower-level numpy
runner the method wraps. Everything else is exposed as a building block
so power users can drive the pipeline component-by-component.

Realtime / overlap-add and matplotlib helpers from the source are
intentionally not ported. The realtime variant was a chunked offline
run rather than true streaming; ship it back if a real streaming use
case appears.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pysampled
from sklearn.decomposition import FastICA


@dataclass
class ICAResult:
    """Container for fitted ICA state and derived arrays.

    Attributes:
        model: The fitted :class:`sklearn.decomposition.FastICA` instance.
        sources: Source time courses, shape ``(n_samples, n_components)``.
        mixing: Mixing matrix, shape ``(n_signals, n_components)``.
        feature_names: Optional per-input-channel labels (length ``n_signals``).
    """

    model: FastICA
    sources: np.ndarray
    mixing: np.ndarray
    feature_names: Optional[List[str]] = None


@dataclass
class CleaningConfig:
    """All knobs for :class:`Log.clean_emg_ekg_artifact`.

    Defaults reproduce the source pipeline's behavior. Stage gates
    (``use_ecg_stage`` / ``use_motion_stage``) skip the entire stage
    when ``False``; per-stage knobs only apply when the stage runs.

    Preprocess:
        preprocess_highpass_hz: High-pass cutoff before ECG / motion
            stages. ``None`` to skip. Default 20 Hz.
        preprocess_lowpass_hz: Optional low-pass cutoff applied after
            the high-pass. ``None`` to skip.
        preprocess_order: Filter order; ``None`` lets pysampled pick.

    ECG (ICA) stage:
        use_ecg_stage: Master switch. If ``False``, skip ICA / EKG
            regression entirely.
        ecg_n_components: ``n_components`` passed to FastICA. ``None``
            uses ``n_signals`` (one per input channel including EKG).
        ecg_components_to_remove: Manual override — explicit list of IC
            indices to zero. When ``None``, components are picked by
            auto-detection (when ``ecg_auto_remove_components=True``).
        ecg_auto_remove_components: Pick ECG-like components by lagged
            correlation against the EKG reference.
        ecg_max_auto_components: Cap on how many ICs the auto-detector
            removes. Default 1 (the strongest EKG-correlated IC).
        ecg_corr_threshold: Minimum absolute lagged correlation for an
            IC to be flagged.
        ecg_corr_max_lag_samples: Lag window (± samples) searched when
            scoring IC-vs-EKG correlation.
        ecg_use_regression: Run a second pass that ridge-regresses the
            lagged EKG out of every channel after IC removal. Cleans up
            residual artifact the IC step misses.
        ecg_reg_max_lag_samples: Lag window for the regression pass.
        ecg_reg_ridge_alpha: Ridge regularization strength for the
            regression pass.
        ica_random_state: Seed passed to FastICA.

    Motion (ACC) stage:
        use_motion_stage: Master switch. If ``False``, skip the motion
            regression step.
        motion_max_lag_samples: Lag window for the per-channel ACC
            regression.
        motion_ridge_alpha: Ridge regularization strength.
        motion_include_magnitude: Append the L2-norm of multi-axis ACC
            to the predictor matrix.
        motion_include_derivative: Append the per-feature first
            difference to the predictor matrix.
        min_variance_ratio: Reject the regression on a channel when the
            variance of the cleaned residual drops below this fraction
            of the input variance. Guards against over-cleaning.
        min_power_ratio: Same idea, on mean-square power.
    """

    # Preprocess
    preprocess_highpass_hz: Optional[float] = 20.0
    preprocess_lowpass_hz: Optional[float] = None
    preprocess_order: Optional[int] = None

    # ECG (ICA) stage
    use_ecg_stage: bool = True
    ecg_n_components: Optional[int] = None
    ecg_components_to_remove: Optional[Union[str, List[int]]] = None
    ecg_auto_remove_components: bool = True
    ecg_max_auto_components: int = 1
    ecg_corr_threshold: float = 0.25
    ecg_corr_max_lag_samples: int = 10
    ecg_use_regression: bool = True
    ecg_reg_max_lag_samples: int = 10
    ecg_reg_ridge_alpha: float = 1e-6
    ica_random_state: int = 0

    # Motion (ACC) stage
    use_motion_stage: bool = True
    motion_max_lag_samples: int = 10
    motion_ridge_alpha: float = 1e-3
    motion_include_magnitude: bool = True
    motion_include_derivative: bool = False
    min_variance_ratio: float = 0.10
    min_power_ratio: float = 0.10


@dataclass
class CleaningResult:
    """Outputs and per-stage diagnostics from one pipeline run.

    Attributes:
        cleaned_emg: Final cleaned EMG, shape ``(n_samples, n_emg_channels)``.
        sr: Pipeline sampling rate (Hz).
        time: Time grid for ``cleaned_emg``.
        stages: Per-stage intermediate arrays — keys ``'raw'``,
            ``'preprocessed'``, ``'post_ecg'``, ``'cleaned'``.
        diagnostics: Per-stage diagnostic dicts — keys ``'ecg'``,
            ``'motion'``, plus ``'harmonization'`` when produced by
            :class:`Log.clean_emg_ekg_artifact`.
        coefficients: Fitted regression coefficients — keys
            ``'ecg_regression_beta'``, ``'motion_betas'``.
        cleaned_emg_ekgonly: Same as ``cleaned_emg`` but with the motion
            stage skipped (preprocess + ECG). ``None`` when the ECG
            stage didn't run.
        cleaned_emg_motiononly: Same as ``cleaned_emg`` but with the ECG
            stage skipped (preprocess + motion). ``None`` when the
            motion stage didn't run.
        feature_names: Per-EMG-channel labels (length
            ``n_emg_channels``). Used by :meth:`generate_report` and the
            :mod:`delsys.clean_review` picker to title each channel.
        fname: Source CSV path. Stamped by
            :class:`Log.clean_emg_ekg_artifact` so that
            :meth:`generate_report` can default to a sibling of the
            input file.
        ica: Full :class:`ICAResult` from the ECG stage (model, sources,
            mixing matrix, feature names). ``None`` when the ECG stage
            didn't run. Surfaced by :class:`CleaningSession` for the
            interactive component picker.
        ica_input_feature_names: Per-input-row labels for ``ica.mixing``
            — the EMG channel names with ``"EKG"`` appended as the last
            entry. ``None`` when the ECG stage didn't run. Distinct from
            :attr:`feature_names` (EMG only).
    """

    cleaned_emg: np.ndarray
    sr: float
    time: np.ndarray
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    coefficients: Dict[str, Any] = field(default_factory=dict)
    cleaned_emg_ekgonly: Optional[np.ndarray] = None
    cleaned_emg_motiononly: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    fname: Optional[str] = None
    ica: Optional["ICAResult"] = None
    ica_input_feature_names: Optional[List[str]] = None

    def generate_report(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write a multi-page PDF report summarizing the cleaning result.

        Page 1 is a ranked summary table (one row per EMG channel,
        sorted most-attenuated first). Subsequent pages plot raw vs
        each cleaning variant for one channel apiece, in the same
        ranked order.

        Args:
            path: Output PDF path. When ``None``, defaults to
                ``<dir(self.fname)>/<stem(self.fname)>_cleaning_report.pdf``;
                raises :class:`ValueError` if ``self.fname`` is also
                unset.

        Returns:
            The :class:`pathlib.Path` of the file that was written.
        """
        return _write_report_pdf(self, path)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def fit_ica(
    emg_2d: np.ndarray,
    *,
    n_components: Optional[int] = None,
    random_state: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    whiten: str = "unit-variance",
    feature_names: Optional[List[str]] = None,
) -> ICAResult:
    """Fit FastICA on an EMG matrix shaped ``(time_points, signals)``.

    Args:
        emg_2d: 2-D float array, samples down rows.
        n_components: Number of components to extract. Defaults to
            ``emg_2d.shape[1]``.
        random_state: Seed for reproducibility.
        max_iter: FastICA iteration cap.
        tol: FastICA convergence tolerance.
        whiten: FastICA ``whiten`` parameter.
        feature_names: Optional per-input-channel labels to attach to
            the returned :class:`ICAResult`.

    Returns:
        :class:`ICAResult` containing the fitted model and source/mix
        matrices.
    """
    x = np.asarray(emg_2d)
    if x.ndim != 2:
        raise ValueError("emg_2d must be 2D with shape (time_points, signals).")

    if n_components is None:
        n_components = x.shape[1]
    if n_components < 1 or n_components > x.shape[1]:
        raise ValueError("n_components must be in [1, n_signals].")

    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        whiten=whiten,
    )
    sources = ica.fit_transform(x)
    return ICAResult(
        model=ica, sources=sources, mixing=ica.mixing_, feature_names=feature_names
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation guarded against near-zero variance inputs."""
    a0 = np.asarray(a) - np.mean(a)
    b0 = np.asarray(b) - np.mean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def _shift_1d(x: np.ndarray, lag: int) -> np.ndarray:
    """Shift a 1-D signal by ``lag`` samples with zero padding."""
    out = np.zeros_like(x)
    if lag == 0:
        out[:] = x
    elif lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


def _build_lagged_design(
    sig_1d: np.ndarray, max_lag_samples: int = 10, add_intercept: bool = True
) -> np.ndarray:
    """Build a lagged design matrix ``[x(t-L), ..., x(t), ..., x(t+L)]``."""
    x = np.asarray(sig_1d).reshape(-1)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    cols = [_shift_1d(x, int(lag)) for lag in lags]
    design = np.column_stack(cols)
    if add_intercept:
        design = np.column_stack([design, np.ones(x.shape[0])])
    return design


def _build_multifeature_lagged_design(
    features_2d: np.ndarray, max_lag_samples: int = 10, add_intercept: bool = True
) -> np.ndarray:
    """Lagged design matrix for multi-feature predictors (one block per feature)."""
    feats = np.asarray(features_2d)
    if feats.ndim != 2:
        raise ValueError("features_2d must be 2D with shape (time, features).")

    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    cols = []
    for k in range(feats.shape[1]):
        vec = feats[:, k]
        cols.extend([_shift_1d(vec, int(lag)) for lag in lags])

    design = np.column_stack(cols)
    if add_intercept:
        design = np.column_stack([design, np.ones(feats.shape[0])])
    return design


def _parse_component_selection(selection_text: str, n_components: int) -> List[int]:
    """Parse text such as ``'0,1,4-6'`` into sorted component indices."""
    cleaned = (selection_text or "").strip().replace(" ", "")
    if cleaned == "":
        return []

    chosen = set()
    for token in cleaned.split(","):
        if token == "":
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            chosen.update(range(lo, hi + 1))
        else:
            chosen.add(int(token))

    chosen_sorted = sorted(chosen)
    bad = [i for i in chosen_sorted if i < 0 or i >= n_components]
    if bad:
        raise ValueError(
            f"Component indices out of range: {bad}. Valid range is 0..{n_components - 1}."
        )
    return chosen_sorted


def _normalize_component_selection(
    selection: Optional[Union[str, List[int]]], n_components: int
) -> List[int]:
    """Coerce a manual component selection into a validated index list.

    ``selection`` may be ``None`` (no manual selection), a comma/range
    string parsed by :func:`_parse_component_selection`, or any iterable
    of ints. Out-of-range indices raise :class:`ValueError`.
    """
    if selection is None:
        return []
    if isinstance(selection, str):
        return _parse_component_selection(selection, n_components)
    selection_idx = sorted(set(int(i) for i in selection))
    bad = [i for i in selection_idx if i < 0 or i >= n_components]
    if bad:
        raise ValueError(
            f"Component indices out of range: {bad}. Valid range is 0..{n_components - 1}."
        )
    return selection_idx


def score_components_against_ekg(
    sources: np.ndarray, ekg_1d: np.ndarray, *, max_lag_samples: int = 10
) -> tuple:
    """Score each IC by max absolute correlation against lagged EKG.

    Returns:
        ``(scores, best_lags)`` — both shape ``(n_components,)``. Scores
        are absolute correlations in ``[0, 1]``; lags are signed sample
        offsets at which the maximum was found.
    """
    s = np.asarray(sources)
    ekg = np.asarray(ekg_1d).reshape(-1)
    if s.ndim != 2:
        raise ValueError("sources must be 2D with shape (time_points, n_components).")
    if s.shape[0] != ekg.shape[0]:
        raise ValueError("sources and ekg_1d must have the same number of time points.")

    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    lagged_ekg = [_shift_1d(ekg, int(lag)) for lag in lags]

    scores = np.zeros(s.shape[1], dtype=float)
    best_lags = np.zeros(s.shape[1], dtype=int)
    for ic_idx in range(s.shape[1]):
        corrs = np.array([abs(_safe_corr(s[:, ic_idx], xlag)) for xlag in lagged_ekg])
        best_idx = int(np.argmax(corrs))
        scores[ic_idx] = float(corrs[best_idx])
        best_lags[ic_idx] = int(lags[best_idx])
    return scores, best_lags


def auto_select_ekg_components(
    corr_scores: np.ndarray,
    *,
    min_corr: float = 0.25,
    keep_at_least_one_when_strong: bool = True,
    max_components: Optional[int] = 1,
) -> List[int]:
    """Pick IC indices likely carrying ECG artifact.

    Args:
        corr_scores: Per-component scores from
            :func:`score_components_against_ekg`.
        min_corr: Threshold above which a component is flagged.
        keep_at_least_one_when_strong: If no IC clears ``min_corr`` but
            the strongest is at least 0.2, still keep that one.
        max_components: Cap on the number of components returned.
            ``None`` for no cap.

    Returns:
        Sorted-by-score (descending) list of IC indices to remove.
    """
    scores = np.asarray(corr_scores).reshape(-1)
    selected = np.where(scores >= float(min_corr))[0].tolist()
    selected = sorted(selected, key=lambda i: float(scores[i]), reverse=True)

    if (
        not selected
        and keep_at_least_one_when_strong
        and scores.size > 0
        and float(np.max(scores)) >= 0.2
    ):
        selected = [int(np.argmax(scores))]

    if max_components is not None and max_components >= 0:
        selected = selected[: int(max_components)]
    return selected


def reconstruct_without_components(
    ica_result: ICAResult, components_to_remove: Optional[Union[str, List[int]]]
) -> tuple:
    """Reconstruct the input from ICA sources after zeroing selected components.

    Returns:
        ``(cleaned, src_clean)`` — the inverse-transformed signal
        ``(n_samples, n_signals)`` and the modified source matrix
        ``(n_samples, n_components)`` with the selected components
        zeroed.
    """
    remove_idx = _normalize_component_selection(
        components_to_remove, ica_result.sources.shape[1]
    )

    src_clean = ica_result.sources.copy()
    if remove_idx:
        src_clean[:, remove_idx] = 0.0

    cleaned = ica_result.model.inverse_transform(src_clean)
    return cleaned, src_clean


def regress_out_ekg_from_emg(
    emg_2d: np.ndarray,
    ekg_1d: np.ndarray,
    *,
    max_lag_samples: int = 10,
    ridge_alpha: float = 1e-6,
) -> tuple:
    """Ridge-regress the lagged EKG basis from every EMG channel.

    Returns:
        ``(cleaned, beta)`` — the residual ``(n_samples, n_channels)``
        and the fitted coefficient matrix ``(n_lags+1, n_channels)``.
    """
    emg = np.asarray(emg_2d)
    ekg = np.asarray(ekg_1d).reshape(-1)
    if emg.ndim != 2:
        raise ValueError("emg_2d must be 2D with shape (time_points, emg_channels).")
    if emg.shape[0] != ekg.shape[0]:
        raise ValueError("emg_2d and ekg_1d must have the same number of samples.")

    x = _build_lagged_design(ekg, max_lag_samples=max_lag_samples, add_intercept=True)
    xtx = x.T @ x
    xtx_reg = xtx + float(ridge_alpha) * np.eye(xtx.shape[0])
    xt = x.T

    cleaned = np.zeros_like(emg, dtype=float)
    beta = np.zeros((x.shape[1], emg.shape[1]))
    for ch in range(emg.shape[1]):
        y = emg[:, ch]
        b = np.linalg.solve(xtx_reg, xt @ y)
        y_hat = x @ b
        cleaned[:, ch] = y - y_hat
        beta[:, ch] = b
    return cleaned, beta


def _prepare_motion_features(
    acc_sig: np.ndarray,
    *,
    include_magnitude: bool = True,
    include_derivative: bool = False,
) -> np.ndarray:
    """Build the predictor matrix for one EMG channel from an ACC stream."""
    x = np.asarray(acc_sig)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Each ACC stream must be 1D or 2D.")

    cols = [x]
    if include_magnitude and x.shape[1] > 1:
        cols.append(np.linalg.norm(x, axis=1, keepdims=True))

    feat = np.column_stack(cols)
    if include_derivative:
        dfeat = np.vstack([np.zeros((1, feat.shape[1])), np.diff(feat, axis=0)])
        feat = np.column_stack([feat, dfeat])
    return feat


def regress_out_motion_from_emg(
    emg_2d: np.ndarray,
    acc_by_emg: Optional[Dict[int, np.ndarray]],
    *,
    max_lag_samples: int = 10,
    ridge_alpha: float = 1e-3,
    include_magnitude: bool = True,
    include_derivative: bool = False,
    min_variance_ratio: float = 0.1,
    min_power_ratio: float = 0.1,
) -> tuple:
    """Per-channel ACC-guided ridge regression with safety gates.

    Channels without an ACC predictor pass through unchanged with a
    diagnostic entry ``reason='missing_acc_predictor'``. Channels where
    the regression would shrink the residual below ``min_variance_ratio``
    or ``min_power_ratio`` of the input keep the input
    (``reason='rejected_by_safety_gate'``).

    Returns:
        ``(cleaned, betas, diag)`` — cleaned EMG matrix, per-channel
        fitted coefficient vectors, and a ``{'per_channel': [...]}``
        diagnostics dict.
    """
    emg = np.asarray(emg_2d)
    if emg.ndim != 2:
        raise ValueError("emg_2d must be 2D with shape (time_points, emg_channels).")

    cleaned = np.asarray(emg, dtype=float).copy()
    betas: Dict[int, np.ndarray] = {}
    channel_diag: List[Dict[str, Any]] = []

    for ch in range(emg.shape[1]):
        y = emg[:, ch]
        if acc_by_emg is None or ch not in acc_by_emg:
            channel_diag.append(
                {
                    "channel": ch,
                    "used": False,
                    "reason": "missing_acc_predictor",
                    "variance_ratio": 1.0,
                    "power_ratio": 1.0,
                }
            )
            continue

        features = _prepare_motion_features(
            acc_by_emg[ch],
            include_magnitude=include_magnitude,
            include_derivative=include_derivative,
        )

        if features.shape[0] != y.shape[0]:
            raise ValueError(
                f"ACC predictor length mismatch for channel {ch}: "
                f"acc={features.shape[0]}, emg={y.shape[0]}"
            )

        x = _build_multifeature_lagged_design(
            features, max_lag_samples=max_lag_samples, add_intercept=True
        )
        xtx = x.T @ x
        xtx_reg = xtx + float(ridge_alpha) * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx_reg, x.T @ y)
        y_hat = x @ beta
        y_candidate = y - y_hat

        var_orig = float(np.var(y))
        var_new = float(np.var(y_candidate))
        variance_ratio = var_new / max(var_orig, 1e-12)

        p_orig = float(np.mean(y**2))
        p_new = float(np.mean(y_candidate**2))
        power_ratio = p_new / max(p_orig, 1e-12)

        accepted = (variance_ratio >= float(min_variance_ratio)) and (
            power_ratio >= float(min_power_ratio)
        )
        if accepted:
            cleaned[:, ch] = y_candidate
            reason = "accepted"
        else:
            reason = "rejected_by_safety_gate"

        betas[ch] = beta
        channel_diag.append(
            {
                "channel": ch,
                "used": True,
                "accepted": accepted,
                "reason": reason,
                "variance_ratio": variance_ratio,
                "power_ratio": power_ratio,
                "n_predictors": int(x.shape[1]),
            }
        )

    return cleaned, betas, {"per_channel": channel_diag}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _resample_with_pysampled(
    sig: np.ndarray, sr: float, target_sr: float
) -> np.ndarray:
    """Resample a numpy array via :class:`pysampled.Data`, no-op when rates match."""
    if abs(float(sr) - float(target_sr)) < 1e-9:
        return np.asarray(sig)
    data = (
        pysampled.Data(np.asarray(sig), sr=float(sr))
        .interpnan()
        .resample(float(target_sr))
    )
    return np.asarray(data())


def harmonize_multirate_inputs(
    emg_2d: np.ndarray,
    emg_sr: float,
    *,
    ekg_1d: Optional[np.ndarray] = None,
    ekg_sr: Optional[float] = None,
    acc_by_emg: Optional[Dict[int, np.ndarray]] = None,
    acc_sr: Optional[Union[float, Dict[int, float]]] = None,
    target_sr: Optional[float] = None,
) -> Dict[str, Any]:
    """Resample EMG / EKG / ACC streams to a single sampling rate.

    All outputs are tail-trimmed to the same length.

    Args:
        emg_2d: EMG matrix, shape ``(time, channels)``.
        emg_sr: EMG sampling rate.
        ekg_1d: Optional EKG reference, 1-D.
        ekg_sr: EKG sampling rate; required when ``ekg_1d`` is given.
        acc_by_emg: Map ``{emg_channel_index: acc_stream}``. Each value
            is 1-D or 2-D.
        acc_sr: Single sampling rate for every ACC stream, or
            ``{channel_index: sr}`` for per-channel rates.
        target_sr: Output sampling rate. Defaults to ``emg_sr`` (so
            EMG passes through, EKG / ACC are resampled to match).

    Returns:
        Dict with keys ``'sr'``, ``'emg'``, ``'ekg'``, ``'acc_by_emg'``,
        ``'n_samples'``.
    """
    emg = np.asarray(emg_2d)
    if emg.ndim != 2:
        raise ValueError("emg_2d must be 2D with shape (time_points, emg_channels).")

    out_sr = float(emg_sr if target_sr is None else target_sr)
    emg_h = _resample_with_pysampled(emg, emg_sr, out_sr)

    ekg_h = None
    if ekg_1d is not None:
        if ekg_sr is None:
            raise ValueError("ekg_sr must be provided when ekg_1d is provided.")
        ekg_h = _resample_with_pysampled(np.asarray(ekg_1d).reshape(-1), ekg_sr, out_sr)

    acc_h: Dict[int, np.ndarray] = {}
    if acc_by_emg is not None:
        for ch_idx, acc_sig in acc_by_emg.items():
            this_sr = acc_sr.get(ch_idx, None) if isinstance(acc_sr, dict) else acc_sr
            if this_sr is None:
                raise ValueError(
                    f"Missing accelerometer sampling rate for EMG channel {ch_idx}."
                )
            acc_h[int(ch_idx)] = _resample_with_pysampled(acc_sig, this_sr, out_sr)

    lengths = [emg_h.shape[0]]
    if ekg_h is not None:
        lengths.append(ekg_h.shape[0])
    lengths.extend([v.shape[0] for v in acc_h.values()])
    n = int(min(lengths))

    emg_h = emg_h[:n]
    if ekg_h is not None:
        ekg_h = ekg_h[:n]
    for key in list(acc_h.keys()):
        acc_h[key] = acc_h[key][:n]

    return {
        "sr": out_sr,
        "emg": emg_h,
        "ekg": ekg_h,
        "acc_by_emg": acc_h,
        "n_samples": n,
    }


def _preprocess_emg_stage(
    emg_2d: np.ndarray,
    sr: float,
    *,
    highpass_hz: Optional[float] = None,
    lowpass_hz: Optional[float] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    """Apply ``interpnan`` plus optional high/low-pass filters."""
    d = pysampled.Data(np.asarray(emg_2d), sr=float(sr)).interpnan()
    if highpass_hz is not None:
        d = d.highpass(float(highpass_hz), order=order)
    if lowpass_hz is not None:
        d = d.lowpass(float(lowpass_hz), order=order)
    return np.asarray(d())


def _fit_ecg_ica(
    pre_emg: np.ndarray,
    ekg_1d: np.ndarray,
    config: CleaningConfig,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit FastICA on the EMG+EKG matrix and score each IC against the EKG.

    The *expensive* half of the ECG stage. Concatenates the EKG as the trailing
    column of the EMG matrix so FastICA decomposes the joint structure, then (when
    ``ecg_auto_remove_components``) scores each IC by lagged correlation against
    the EKG. Split out from :func:`_run_ecg_stage` so an interactive
    :class:`CleaningSession` can fit once and re-zero components cheaply via
    :func:`_apply_ecg_components`.

    Returns a dict with ``ica_result``, the per-IC ``ic_corr_scores`` /
    ``ic_best_lags`` (``None`` when auto-detection is off), and
    ``ica_input_feature_names`` (the EMG names with ``"EKG"`` appended).
    """
    ecg_input = np.column_stack([pre_emg, np.asarray(ekg_1d).reshape(-1)])
    feat_with_ekg = (
        list(feature_names) + ["EKG"]
        if feature_names is not None
        else [f"ch{i}" for i in range(pre_emg.shape[1])] + ["EKG"]
    )
    ica_result = fit_ica(
        ecg_input,
        n_components=config.ecg_n_components,
        random_state=config.ica_random_state,
        feature_names=feat_with_ekg,
    )

    ic_corr_scores = None
    ic_best_lags = None
    if config.ecg_auto_remove_components:
        ic_corr_scores, ic_best_lags = score_components_against_ekg(
            ica_result.sources,
            ekg_1d=ecg_input[:, -1],
            max_lag_samples=config.ecg_corr_max_lag_samples,
        )

    return {
        "ica_result": ica_result,
        "ic_corr_scores": ic_corr_scores,
        "ic_best_lags": ic_best_lags,
        "ica_input_feature_names": feat_with_ekg,
    }


def _apply_ecg_components(
    ica_result: "ICAResult",
    ekg_1d: np.ndarray,
    components_to_remove: Optional[Union[str, List[int]]],
    config: CleaningConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reconstruct the EMG with ``components_to_remove`` zeroed — no refit.

    The *cheap* half of the ECG stage: re-zero the chosen ICs on an
    already-fitted :class:`ICAResult` (:func:`reconstruct_without_components`),
    drop the trailing EKG column, then optionally ridge-regress the lagged EKG
    out of every channel (``ecg_use_regression``). This is the primitive the
    interactive component picker re-runs on every toggle.

    Returns ``(cleaned_emg, regression_beta)`` — the latter ``None`` when the
    regression pass is off.
    """
    cleaned_full, _ = reconstruct_without_components(ica_result, components_to_remove)
    cleaned_emg = cleaned_full[:, :-1]  # drop the EKG column

    regression_beta = None
    if config.ecg_use_regression:
        cleaned_emg, regression_beta = regress_out_ekg_from_emg(
            cleaned_emg,
            ekg_1d=np.asarray(ekg_1d).reshape(-1),
            max_lag_samples=config.ecg_reg_max_lag_samples,
            ridge_alpha=config.ecg_reg_ridge_alpha,
        )
    return cleaned_emg, regression_beta


def _select_ecg_components(fit: Dict[str, Any], config: CleaningConfig) -> Dict[str, List[int]]:
    """Resolve the IC set to remove from a :func:`_fit_ecg_ica` result.

    Combines the manual override (``ecg_components_to_remove``) with the
    auto-detected EKG-correlated ICs. Returns the ``manual`` / ``auto`` / merged
    ``selected`` index lists.
    """
    n_components = fit["ica_result"].sources.shape[1]
    manual = _normalize_component_selection(config.ecg_components_to_remove, n_components)
    auto_components: List[int] = []
    if config.ecg_auto_remove_components and fit["ic_corr_scores"] is not None:
        auto_components = auto_select_ekg_components(
            fit["ic_corr_scores"],
            min_corr=config.ecg_corr_threshold,
            keep_at_least_one_when_strong=True,
            max_components=config.ecg_max_auto_components,
        )
    return {
        "manual": manual,
        "auto": auto_components,
        "selected": sorted(set(manual + auto_components)),
    }


def _run_ecg_stage(
    pre_emg: np.ndarray,
    ekg_1d: np.ndarray,
    config: CleaningConfig,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """ICA-based ECG suppression with optional residual regression.

    Composes the fit (:func:`_fit_ecg_ica`), component selection
    (:func:`_select_ecg_components`), and reconstruction
    (:func:`_apply_ecg_components`). Returns a dict with ``cleaned_emg``
    (n, n_emg_channels), the IC bookkeeping, and the regression beta when run.
    """
    fit = _fit_ecg_ica(pre_emg, ekg_1d, config, feature_names)
    ica_result = fit["ica_result"]
    sel = _select_ecg_components(fit, config)
    cleaned_emg, regression_beta = _apply_ecg_components(
        ica_result, ekg_1d, sel["selected"], config
    )

    return {
        "cleaned_emg": cleaned_emg,
        "components_removed": sel["selected"],
        "manual_components_removed": sel["manual"],
        "auto_ekg_components_removed": sel["auto"],
        "ic_ekg_corr_scores": fit["ic_corr_scores"],
        "ic_ekg_best_lags": fit["ic_best_lags"],
        "regression_beta": regression_beta,
        "ica_result": ica_result,
        "ica_input_feature_names": fit["ica_input_feature_names"],
    }


def _assemble_result(
    stages: Dict[str, np.ndarray],
    diagnostics: Dict[str, Any],
    coefficients: Dict[str, Any],
    sr: float,
    *,
    acc_by_emg: Optional[Dict[int, np.ndarray]],
    config: CleaningConfig,
    feature_names: Optional[List[str]],
    time: Optional[np.ndarray],
    ica_result: Optional["ICAResult"],
    ica_input_feature_names: Optional[List[str]],
    compute_motiononly: bool = True,
) -> CleaningResult:
    """Run the motion stage and assemble the final :class:`CleaningResult`.

    The shared tail of :func:`run_pipeline` and
    :meth:`CleaningSession.recompute`. On entry ``stages`` must already hold
    ``raw`` / ``preprocessed`` / ``post_ecg`` and ``diagnostics`` /
    ``coefficients`` their ECG entries. This adds the motion stage (in place on
    those dicts), computes the stage-isolated variants, fills ``stages["cleaned"]``,
    and builds the result. ``ica_result is not None`` is the "ECG stage ran" flag.

    ``compute_motiononly=False`` skips the second motion-regression pass that
    produces ``cleaned_emg_motiononly`` — an interactive preview that isn't showing
    the motion-only variant doesn't need it (the live picker passes this).
    """
    cfg = config
    post_ecg = stages["post_ecg"]

    post_motion = post_ecg
    motion_ran = False
    if cfg.use_motion_stage and acc_by_emg:
        post_motion, motion_betas, motion_diag = regress_out_motion_from_emg(
            post_ecg,
            acc_by_emg=acc_by_emg,
            max_lag_samples=cfg.motion_max_lag_samples,
            ridge_alpha=cfg.motion_ridge_alpha,
            include_magnitude=cfg.motion_include_magnitude,
            include_derivative=cfg.motion_include_derivative,
            min_variance_ratio=cfg.min_variance_ratio,
            min_power_ratio=cfg.min_power_ratio,
        )
        coefficients["motion_betas"] = motion_betas
        diagnostics["motion"] = motion_diag
        motion_ran = True
    else:
        diagnostics["motion"] = {"used": False}
    stages["cleaned"] = post_motion

    # Stage-isolated variants. ekg-only is a free read; motion-only
    # requires a second motion-regression pass on the preprocessed
    # signal (skips the ECG step) — cheap compared to ICA.
    cleaned_emg_ekgonly = stages["post_ecg"] if ica_result is not None else None
    cleaned_emg_motiononly: Optional[np.ndarray] = None
    if motion_ran and compute_motiononly:
        motiononly, _, _ = regress_out_motion_from_emg(
            stages["preprocessed"],
            acc_by_emg=acc_by_emg,
            max_lag_samples=cfg.motion_max_lag_samples,
            ridge_alpha=cfg.motion_ridge_alpha,
            include_magnitude=cfg.motion_include_magnitude,
            include_derivative=cfg.motion_include_derivative,
            min_variance_ratio=cfg.min_variance_ratio,
            min_power_ratio=cfg.min_power_ratio,
        )
        cleaned_emg_motiononly = motiononly

    if time is None:
        out_time = np.arange(stages["cleaned"].shape[0]) / float(sr)
    else:
        t_arr = np.asarray(time)
        out_time = (
            t_arr[: stages["cleaned"].shape[0]]
            if t_arr.shape[0] >= stages["cleaned"].shape[0]
            else np.arange(stages["cleaned"].shape[0]) / float(sr)
        )

    return CleaningResult(
        cleaned_emg=stages["cleaned"],
        sr=float(sr),
        time=out_time,
        stages=stages,
        diagnostics=diagnostics,
        coefficients=coefficients,
        cleaned_emg_ekgonly=cleaned_emg_ekgonly,
        cleaned_emg_motiononly=cleaned_emg_motiononly,
        feature_names=list(feature_names) if feature_names is not None else None,
        ica=ica_result,
        ica_input_feature_names=ica_input_feature_names,
    )


def run_pipeline(
    emg_2d: np.ndarray,
    sr: float,
    *,
    ekg_1d: Optional[np.ndarray] = None,
    acc_by_emg: Optional[Dict[int, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    time: Optional[np.ndarray] = None,
    config: Optional[CleaningConfig] = None,
) -> CleaningResult:
    """Run the staged cleaner over already-aligned numpy arrays.

    Inputs must already share a sampling rate; use
    :func:`harmonize_multirate_inputs` first if they do not (or call
    :class:`Log.clean_emg_ekg_artifact`, which harmonizes for you).

    Stage order: preprocess → ECG (optional) → motion (optional). Each
    stage stores its output in ``result.stages``.

    Args:
        emg_2d: EMG matrix, shape ``(time, channels)``.
        sr: Sampling rate (Hz) shared by every input array.
        ekg_1d: EKG reference, 1-D. Required when
            ``config.use_ecg_stage=True``.
        acc_by_emg: Map ``{emg_channel_index: acc_stream}``. Used only
            when ``config.use_motion_stage=True``.
        feature_names: Optional per-EMG-channel labels carried into
            ECG-stage diagnostics.
        time: Optional time grid for ``emg_2d``; reused on the result
            when shapes line up. When ``None``, a fresh grid is built
            from ``sr``.
        config: :class:`CleaningConfig`; defaults to ``CleaningConfig()``.

    Returns:
        :class:`CleaningResult` with the cleaned EMG, per-stage
        snapshots, and diagnostics.
    """
    cfg = CleaningConfig() if config is None else config
    if not isinstance(cfg, CleaningConfig):
        raise TypeError("config must be a CleaningConfig instance.")

    stages: Dict[str, np.ndarray] = {"raw": np.asarray(emg_2d)}
    diagnostics: Dict[str, Any] = {}
    coefficients: Dict[str, Any] = {}

    pre = _preprocess_emg_stage(
        stages["raw"],
        sr=sr,
        highpass_hz=cfg.preprocess_highpass_hz,
        lowpass_hz=cfg.preprocess_lowpass_hz,
        order=cfg.preprocess_order,
    )
    stages["preprocessed"] = pre

    post_ecg = pre
    ica_result: Optional[ICAResult] = None
    ica_input_feature_names: Optional[List[str]] = None
    if cfg.use_ecg_stage and ekg_1d is not None:
        ecg_stage = _run_ecg_stage(pre, np.asarray(ekg_1d), cfg, feature_names)
        post_ecg = ecg_stage["cleaned_emg"]
        coefficients["ecg_regression_beta"] = ecg_stage["regression_beta"]
        diagnostics["ecg"] = {
            "components_removed": ecg_stage["components_removed"],
            "manual_components_removed": ecg_stage["manual_components_removed"],
            "auto_ekg_components_removed": ecg_stage["auto_ekg_components_removed"],
            "ic_ekg_corr_scores": ecg_stage["ic_ekg_corr_scores"],
            "ic_ekg_best_lags": ecg_stage["ic_ekg_best_lags"],
            "corr_threshold": cfg.ecg_corr_threshold,
        }
        ica_result = ecg_stage["ica_result"]
        ica_input_feature_names = ecg_stage["ica_input_feature_names"]
    else:
        diagnostics["ecg"] = {"used": False}
    stages["post_ecg"] = post_ecg

    return _assemble_result(
        stages,
        diagnostics,
        coefficients,
        sr,
        acc_by_emg=acc_by_emg,
        config=cfg,
        feature_names=feature_names,
        time=time,
        ica_result=ica_result,
        ica_input_feature_names=ica_input_feature_names,
    )


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad ``arr`` (samples down rows) to exactly ``n`` rows.

    Used to fit a re-paired ACC predictor to the session's fixed sample count.
    ACC and EMG from the same recording match within rounding once resampled to a
    common rate, so any adjustment is a handful of tail samples.
    """
    m = arr.shape[0]
    if m == n:
        return arr
    if m > n:
        return arr[:n]
    pad = [(0, n - m)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad, mode="edge")


@dataclass(eq=False)
class CleaningSession:
    """Re-runnable EMG cleaning over one :class:`delsys.Log`, ICA fit cached.

    The interactive counterpart to :func:`run_pipeline`. :meth:`from_log` does the
    expensive setup once — gather + harmonize + preprocess + fit FastICA — and
    :meth:`recompute` then re-derives a full :class:`CleaningResult` for any
    IC-removal set and motion pairing *without refitting the ICA* (component
    removal is :func:`reconstruct_without_components`; the EKG residual regression
    and the ACC motion regression are cheap linear solves). That asymmetry is what
    lets the component picker in :mod:`delsys.clean_review` redraw live on every
    toggle. The source ``Log`` is held so a new motion pairing can re-resolve its
    per-sensor ACC predictors (the motion stage is downstream of the ICA fit).

    Attributes mirror the slices :func:`run_pipeline` derives internally; the
    interesting ones for a UI are :attr:`ica` (``None`` when the ECG stage didn't
    run), :attr:`ic_corr_scores`, and :meth:`auto_components` (the default set).
    """

    log: Any
    config: CleaningConfig
    sr: float
    emg_layout: List[Tuple[Any, str]]
    feature_names: List[str]
    raw_emg: np.ndarray  # harmonized, baseline-shifted EMG matrix (the "raw" stage)
    preprocessed_emg: np.ndarray  # post high/low-pass
    ekg_1d: Optional[np.ndarray]  # harmonized EKG (the ICA input's trailing column)
    ica: Optional[ICAResult]
    ic_corr_scores: Optional[np.ndarray]
    ic_best_lags: Optional[np.ndarray]
    ica_input_feature_names: Optional[List[str]]
    #: Resampled ACC predictors cached per motion pairing (so a component toggle
    #: never re-resamples the accelerometers — the dominant interactive cost).
    _acc_cache: Dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_log(
        cls, lf, *, config: Optional[CleaningConfig] = None
    ) -> "CleaningSession":
        """Build a session from a :class:`delsys.Log` (fits the ICA once).

        Mirrors :meth:`delsys.Log.clean_emg_ekg_artifact`'s gather + harmonize +
        preprocess, then fits FastICA on the EMG+EKG matrix. ACC is *not* gathered
        here — it's resolved per :meth:`recompute` so the motion pairing can change.
        """
        cfg = CleaningConfig() if config is None else config
        if not isinstance(cfg, CleaningConfig):
            raise TypeError("config must be a CleaningConfig instance.")
        if lf.emg is None:
            raise ValueError("Log has no EMG bundle to clean.")

        emg_2d, emg_sr, emg_layout, feature_names = lf._gather_emg()
        ekg_1d, ekg_sr = lf._gather_ekg()

        # Harmonize EMG + EKG only (no ACC): EMG passes through at its own rate,
        # EKG is resampled to match. ACC is downstream of the ICA fit and is
        # resolved/length-fit per recompute so the pairing stays editable.
        harmonized = harmonize_multirate_inputs(
            emg_2d=emg_2d,
            emg_sr=emg_sr,
            ekg_1d=ekg_1d,
            ekg_sr=ekg_sr,
            target_sr=emg_sr,
        )
        sr = float(harmonized["sr"])
        raw_emg = np.asarray(harmonized["emg"])
        ekg_h = harmonized["ekg"]

        pre = _preprocess_emg_stage(
            raw_emg,
            sr=sr,
            highpass_hz=cfg.preprocess_highpass_hz,
            lowpass_hz=cfg.preprocess_lowpass_hz,
            order=cfg.preprocess_order,
        )

        ica = ic_corr_scores = ic_best_lags = ica_feat = None
        if cfg.use_ecg_stage and ekg_h is not None:
            fit = _fit_ecg_ica(pre, ekg_h, cfg, feature_names)
            ica = fit["ica_result"]
            ic_corr_scores = fit["ic_corr_scores"]
            ic_best_lags = fit["ic_best_lags"]
            ica_feat = fit["ica_input_feature_names"]

        return cls(
            log=lf,
            config=cfg,
            sr=sr,
            emg_layout=emg_layout,
            feature_names=feature_names,
            raw_emg=raw_emg,
            preprocessed_emg=pre,
            ekg_1d=ekg_h,
            ica=ica,
            ic_corr_scores=ic_corr_scores,
            ic_best_lags=ic_best_lags,
            ica_input_feature_names=ica_feat,
        )

    @property
    def n_components(self) -> int:
        """Number of ICA components (0 when the ECG stage didn't run)."""
        return 0 if self.ica is None else int(self.ica.sources.shape[1])

    def auto_components(self) -> List[int]:
        """The default IC-removal set — auto-detected EKG-correlated ICs.

        Combines the config's manual override with auto-detection (same as a
        first-pass :func:`run_pipeline`); ``[]`` when the ECG stage didn't run.
        """
        if self.ica is None:
            return []
        fit = {"ica_result": self.ica, "ic_corr_scores": self.ic_corr_scores}
        return _select_ecg_components(fit, self.config)["selected"]

    def recompute(
        self,
        components_to_remove: Optional[Union[str, List[int]]],
        motion: Optional[Union[str, Dict[int, Union[int, str]]]] = "auto",
        motiononly: bool = True,
    ) -> CleaningResult:
        """Re-derive a :class:`CleaningResult` for a removal set + motion pairing.

        Cheap: the ICA fit from :meth:`from_log` is reused, and the resampled ACC
        predictors are cached per pairing (a component toggle never re-resamples).
        ``components_to_remove`` is treated as an explicit (manual) set —
        auto-detection does not re-fire — so the result reflects exactly what the
        user picked. ``motion`` follows the same grammar as
        :meth:`delsys.Log.clean_emg_ekg_artifact` (``"auto"`` / a
        ``{emg_sensor: target}`` dict / ``None``). ``motiononly=False`` skips the
        second motion-regression pass (the ``cleaned_emg_motiononly`` variant) — the
        live picker passes ``False`` unless it's previewing that variant.
        """
        n = self.preprocessed_emg.shape[0]
        stages: Dict[str, np.ndarray] = {
            "raw": self.raw_emg,
            "preprocessed": self.preprocessed_emg,
        }
        diagnostics: Dict[str, Any] = {}
        coefficients: Dict[str, Any] = {}

        if self.ica is not None:
            remove = _normalize_component_selection(
                components_to_remove, self.ica.sources.shape[1]
            )
            cleaned_emg, beta = _apply_ecg_components(
                self.ica, self.ekg_1d, remove, self.config
            )
            stages["post_ecg"] = cleaned_emg
            coefficients["ecg_regression_beta"] = beta
            diagnostics["ecg"] = {
                "components_removed": remove,
                "manual_components_removed": remove,
                "auto_ekg_components_removed": [],
                "ic_ekg_corr_scores": self.ic_corr_scores,
                "ic_ekg_best_lags": self.ic_best_lags,
                "corr_threshold": self.config.ecg_corr_threshold,
            }
        else:
            stages["post_ecg"] = self.preprocessed_emg
            diagnostics["ecg"] = {"used": False}

        acc_by_emg = self._acc_for_motion(motion, n)
        result = _assemble_result(
            stages,
            diagnostics,
            coefficients,
            self.sr,
            acc_by_emg=acc_by_emg or None,
            config=self.config,
            feature_names=self.feature_names,
            time=None,
            ica_result=self.ica,
            ica_input_feature_names=self.ica_input_feature_names,
            compute_motiononly=motiononly,
        )
        result.fname = getattr(self.log, "fname", None)
        return result

    @staticmethod
    def _motion_key(motion):
        """Hashable cache key for a motion pairing (dicts are unhashable)."""
        if isinstance(motion, dict):
            return ("dict", tuple(sorted(motion.items())))
        return motion

    def _acc_for_motion(
        self,
        motion: Optional[Union[str, Dict[int, Union[int, str]]]],
        n: int,
    ) -> Dict[int, np.ndarray]:
        """Resolve + resample + length-fit ACC predictors for ``motion`` (cached).

        Cached per pairing: resampling the accelerometers is the dominant per-call
        cost, and the predictors are constant for a fixed pairing, so a component
        toggle (which doesn't change the pairing) reuses them.
        """
        key = self._motion_key(motion)
        cached = self._acc_cache.get(key)
        if cached is not None:
            return cached
        acc_raw, acc_sr = self.log._acc_by_emg(self.emg_layout, motion)
        acc_by_emg: Dict[int, np.ndarray] = {}
        for col, arr in acc_raw.items():
            res = _resample_with_pysampled(np.asarray(arr), acc_sr[col], self.sr)
            acc_by_emg[col] = _fit_length(np.asarray(res), n)
        self._acc_cache[key] = acc_by_emg
        return acc_by_emg


# ---------------------------------------------------------------------------
# Reporting and review helpers
# ---------------------------------------------------------------------------
#
# Matplotlib is imported lazily inside these helpers so that callers that
# only use ``run_pipeline`` don't pay the matplotlib import cost.


# Frequency band used by the report's "ecg-band dB" column. Wide enough
# to capture the QRS spike and its first few harmonics, narrow enough
# that the EMG band (typically 30–500 Hz) doesn't dominate the integral.
_ECG_BAND_LO_HZ = 0.5
_ECG_BAND_HI_HZ = 30.0


def _rank_channels_by_attenuation(raw: np.ndarray, cleaned: np.ndarray) -> List[int]:
    """Return EMG column indices sorted by total-power dB attenuation.

    Most-attenuated (most-negative dB) first. Used to order the report's
    summary table and the per-channel pages.
    """
    var_raw = np.var(np.asarray(raw), axis=0)
    var_cleaned = np.var(np.asarray(cleaned), axis=0)
    db = 10.0 * np.log10(np.maximum(var_cleaned, 1e-30) / np.maximum(var_raw, 1e-30))
    return [int(i) for i in np.argsort(db)]


def _motion_outcome_for_channel(diagnostics: Dict[str, Any], ch_idx: int) -> str:
    """Look up the per-channel motion-stage reason from a result's diagnostics."""
    motion = diagnostics.get("motion") or {}
    if not motion or motion.get("used") is False:
        return "n/a"
    per_channel = motion.get("per_channel") or []
    for c in per_channel:
        if int(c.get("channel", -1)) == int(ch_idx):
            return str(c.get("reason", "n/a"))
    return "n/a"


def _channel_label(result: "CleaningResult", ch_idx: int) -> str:
    if result.feature_names and 0 <= ch_idx < len(result.feature_names):
        return str(result.feature_names[ch_idx])
    return f"ch{ch_idx}"


def _channel_total_db(raw_col: np.ndarray, cleaned_col: np.ndarray) -> float:
    return 10.0 * np.log10(
        max(float(np.var(cleaned_col)), 1e-30) / max(float(np.var(raw_col)), 1e-30)
    )


def _welch_nperseg(n_samples: int, sr: float) -> int:
    """Pick a sane Welch ``nperseg`` for short EMG segments."""
    target = int(sr * 2)
    nperseg = min(target, n_samples)
    nperseg = max(nperseg, min(64, n_samples))
    return max(int(nperseg), 8)


def _band_power(sig_1d: np.ndarray, sr: float, lo: float, hi: float) -> float:
    """Integrated PSD between ``[lo, hi]`` Hz via Welch."""
    from scipy.signal import welch

    sig = np.asarray(sig_1d).reshape(-1)
    nperseg = _welch_nperseg(sig.shape[0], sr)
    f, p = welch(sig, fs=float(sr), nperseg=nperseg)
    band = (f >= float(lo)) & (f <= float(hi))
    if not np.any(band):
        return 0.0
    # ``np.trapezoid`` is the NumPy 2.0 spelling (also present in 1.26+).
    # The 1.x ``np.trapz`` was removed in 2.0, so a ``getattr(..., default)``
    # would still trip the expired-attribute error on the default branch.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(p[band], f[band]))


def _channel_ecg_band_db(raw_col: np.ndarray, cleaned_col: np.ndarray, sr: float) -> float:
    p_raw = _band_power(raw_col, sr, _ECG_BAND_LO_HZ, _ECG_BAND_HI_HZ)
    p_cleaned = _band_power(cleaned_col, sr, _ECG_BAND_LO_HZ, _ECG_BAND_HI_HZ)
    return 10.0 * np.log10(max(p_cleaned, 1e-30) / max(p_raw, 1e-30))


def _channel_motion_db(post_ecg_col: np.ndarray, cleaned_col: np.ndarray) -> float:
    """Total-power dB attributable to the motion stage on one channel.

    Compares ``var(cleaned)`` to ``var(post_ecg)`` so the result reflects
    only what the motion regression added on top of the ECG-cleaned
    signal. When the ECG stage didn't run, ``post_ecg`` is the
    preprocessed signal and the metric collapses to "motion-only" dB.
    """
    return 10.0 * np.log10(
        max(float(np.var(cleaned_col)), 1e-30)
        / max(float(np.var(post_ecg_col)), 1e-30)
    )


def _resolve_default_report_path(result: "CleaningResult") -> Path:
    if result.fname is None:
        raise ValueError(
            "generate_report() needs an explicit path= when result.fname is unset."
        )
    src = Path(result.fname)
    return src.parent / f"{src.stem}_cleaning_report.pdf"


def _check_report_path_writable(source_fname: Optional[str]) -> None:
    """Raise :class:`PermissionError` early if the auto-report path is locked.

    Called from :class:`Log.clean_emg_ekg_artifact` before the cleaning
    pipeline runs, so a PDF that's open in another viewer fails the call
    up front instead of after a minute of ICA work plus an in-place
    splice. No-op when there's no source path (no auto-report would be
    written) or when the target file does not yet exist.
    """
    if source_fname is None:
        return
    src = Path(source_fname)
    target = src.parent / f"{src.stem}_cleaning_report.pdf"
    if not target.exists():
        return
    try:
        # ``r+b`` opens for read+write without truncating — touches the
        # OS lock the same way ``PdfPages`` will, so a held lock surfaces
        # here.
        with open(target, "r+b"):
            pass
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write the auto-report to {target} — the file is "
            "open in another program. Close it and re-run, or pass "
            "generate_report=False to skip the PDF step."
        ) from exc


def _build_summary_rows(result: "CleaningResult") -> Tuple[List[int], List[Dict[str, Any]]]:
    raw = np.asarray(result.stages["raw"])
    cleaned = np.asarray(result.cleaned_emg)
    sr = float(result.sr)

    # The motion-stage dB column compares cleaned vs post_ecg. When the
    # ECG stage didn't run, fall back to the preprocessed snapshot so
    # the value still reflects what the motion regression added.
    motion_baseline = np.asarray(
        result.stages.get("post_ecg", result.stages.get("preprocessed", cleaned))
    )

    ranked = _rank_channels_by_attenuation(raw, cleaned)
    rows: List[Dict[str, Any]] = []
    for rank, ch in enumerate(ranked):
        rows.append(
            {
                "rank": rank + 1,
                "channel": ch,
                "label": _channel_label(result, ch),
                "total_db": _channel_total_db(raw[:, ch], cleaned[:, ch]),
                "ecg_db": _channel_ecg_band_db(raw[:, ch], cleaned[:, ch], sr),
                "motion_db": _channel_motion_db(motion_baseline[:, ch], cleaned[:, ch]),
                "motion": _motion_outcome_for_channel(result.diagnostics, ch),
            }
        )
    return ranked, rows


def _draw_channel_panels(
    axes,
    raw_col: np.ndarray,
    ekgonly_col: Optional[np.ndarray],
    motiononly_col: Optional[np.ndarray],
    cleaned_col: np.ndarray,
    time: np.ndarray,
    *,
    show_ekgonly: bool = True,
    show_motiononly: bool = True,
    show_cleaned: bool = True,
) -> None:
    """Populate the three time-domain panels used by the PDF per-channel pages."""
    ax_e, ax_m, ax_c = axes
    for ax in axes:
        ax.cla()

    ax_e.plot(time, raw_col, color="0.5", lw=0.6, label="raw")
    if ekgonly_col is None:
        ax_e.text(
            0.5, 0.5, "ECG stage skipped",
            transform=ax_e.transAxes, ha="center", va="center", color="0.4",
        )
    elif show_ekgonly:
        ax_e.plot(time, ekgonly_col, color="C0", lw=0.6, label="ekg-only")
    ax_e.set_ylabel("ekg-only")
    ax_e.legend(loc="upper right", fontsize=7)

    ax_m.plot(time, raw_col, color="0.5", lw=0.6, label="raw")
    if motiononly_col is None:
        ax_m.text(
            0.5, 0.5, "motion stage skipped",
            transform=ax_m.transAxes, ha="center", va="center", color="0.4",
        )
    elif show_motiononly:
        ax_m.plot(time, motiononly_col, color="C1", lw=0.6, label="motion-only")
    ax_m.set_ylabel("motion-only")
    ax_m.legend(loc="upper right", fontsize=7)

    ax_c.plot(time, raw_col, color="0.5", lw=0.6, label="raw")
    if show_cleaned:
        ax_c.plot(time, cleaned_col, color="C2", lw=0.6, label="cleaned")
    ax_c.set_ylabel("cleaned")
    ax_c.set_xlabel("time (s)")
    ax_c.legend(loc="upper right", fontsize=7)


def _draw_psd(ax, raw_col: np.ndarray, cleaned_col: np.ndarray, sr: float) -> None:
    from scipy.signal import welch

    nperseg = _welch_nperseg(raw_col.shape[0], sr)
    f_r, p_r = welch(raw_col, fs=float(sr), nperseg=nperseg)
    f_c, p_c = welch(cleaned_col, fs=float(sr), nperseg=nperseg)
    ax.semilogy(f_r, np.maximum(p_r, 1e-30), color="0.5", lw=0.6, label="raw")
    ax.semilogy(f_c, np.maximum(p_c, 1e-30), color="C2", lw=0.6, label="cleaned")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper right", fontsize=7)


def _draw_summary_page(fig, result: "CleaningResult", rows: List[Dict[str, Any]]) -> None:
    fig.clf()
    components_removed = (
        result.diagnostics.get("ecg", {}).get("components_removed", "n/a")
    )
    title_lines = [
        f"source: {result.fname or '(unstamped)'}",
        f"sr: {result.sr:.2f} Hz | n_samples: {result.cleaned_emg.shape[0]} | "
        f"n_channels: {result.cleaned_emg.shape[1]}",
        f"ECG components removed: {components_removed}",
    ]
    fig.suptitle("\n".join(title_lines), fontsize=10, y=0.985)
    ax = fig.add_axes([0.05, 0.03, 0.9, 0.86])
    ax.axis("off")
    cell_text = [
        [
            str(r["rank"]),
            str(r["channel"]),
            r["label"],
            f"{r['total_db']:+.2f}",
            f"{r['ecg_db']:+.2f}",
            f"{r['motion_db']:+.2f}",
            r["motion"],
        ]
        for r in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=[
            "rank",
            "channel",
            "location",
            "total dB",
            "ecg-band dB",
            "motion dB",
            "motion outcome",
        ],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.1)


def _draw_diagnostics_page(
    fig, result: "CleaningResult", cfg_threshold: Optional[float] = None
) -> None:
    """Render the ECG-stage diagnostics page (page 1 of the PDF)."""
    fig.clf()

    ecg_diag = result.diagnostics.get("ecg") or {}
    ekg_used = ecg_diag.get("used", True)

    n_samples = result.cleaned_emg.shape[0]
    n_channels = result.cleaned_emg.shape[1]
    motion_diag = result.diagnostics.get("motion") or {}
    motion_used = motion_diag.get("used") is not False

    title_lines = [
        f"source: {result.fname or '(unstamped)'}",
        f"sr: {result.sr:.2f} Hz | n_samples: {n_samples} | "
        f"n_channels: {n_channels}",
        f"ECG stage: {'on' if ekg_used else 'off'} | "
        f"motion stage: {'on' if motion_used else 'off'}",
    ]
    fig.suptitle("\n".join(title_lines), fontsize=10, y=0.985)

    if not ekg_used:
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
        ax.axis("off")
        ax.text(
            0.5, 0.5, "ECG stage skipped",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="0.4",
        )
        return

    scores = ecg_diag.get("ic_ekg_corr_scores")
    components_removed = list(ecg_diag.get("components_removed") or [])
    manual = list(ecg_diag.get("manual_components_removed") or [])
    auto = list(ecg_diag.get("auto_ekg_components_removed") or [])
    best_lags = ecg_diag.get("ic_ekg_best_lags")

    ax_bar = fig.add_axes([0.08, 0.42, 0.86, 0.45])
    if scores is not None and len(scores) > 0:
        scores_arr = np.asarray(scores)
        x = np.arange(scores_arr.shape[0])
        colors = [
            "C3" if int(i) in components_removed else "0.6"
            for i in x
        ]
        ax_bar.bar(x, np.abs(scores_arr), color=colors)
        ax_bar.set_xlabel("IC index")
        ax_bar.set_ylabel("|corr| vs EKG")
        ax_bar.set_title("IC ↔ EKG lagged correlation scores")
        ax_bar.set_xticks(x)
        if cfg_threshold is not None:
            ax_bar.axhline(
                float(cfg_threshold), color="C0", lw=0.8, linestyle="--",
                label=f"threshold={cfg_threshold:.2f}",
            )
            ax_bar.legend(loc="upper right", fontsize=8)
    else:
        ax_bar.text(
            0.5, 0.5, "no IC scores (auto-detect off)",
            transform=ax_bar.transAxes, ha="center", va="center", color="0.4",
        )
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

    ax_text = fig.add_axes([0.08, 0.06, 0.86, 0.30])
    ax_text.axis("off")
    lines = [
        f"components_removed: {components_removed}",
        f"auto_ekg_components_removed: {auto}",
        f"manual_components_removed: {manual}",
        f"ic_ekg_best_lags: {list(best_lags) if best_lags is not None else 'n/a'}",
    ]
    ax_text.text(
        0.0, 1.0, "\n".join(lines),
        transform=ax_text.transAxes, ha="left", va="top",
        fontsize=9, family="monospace",
    )


def _write_report_pdf(
    result: "CleaningResult", path: Optional[Union[str, Path]]
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if path is None:
        out_path = _resolve_default_report_path(result)
    else:
        out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(result.stages["raw"])
    cleaned = np.asarray(result.cleaned_emg)
    ekgonly = result.cleaned_emg_ekgonly
    motiononly = result.cleaned_emg_motiononly
    time = np.asarray(result.time)
    sr = float(result.sr)

    _, rows = _build_summary_rows(result)

    threshold = (result.diagnostics.get("ecg") or {}).get("corr_threshold")

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        _draw_diagnostics_page(fig, result, cfg_threshold=threshold)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        _draw_summary_page(fig, result, rows)
        pdf.savefig(fig)
        plt.close(fig)

        for row in rows:
            ch = int(row["channel"])
            fig = plt.figure(figsize=(8.5, 11))
            gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.2], hspace=0.45)
            ax_e = fig.add_subplot(gs[0])
            ax_m = fig.add_subplot(gs[1], sharex=ax_e, sharey=ax_e)
            ax_c = fig.add_subplot(gs[2], sharex=ax_e, sharey=ax_e)
            ax_p = fig.add_subplot(gs[3])
            fig.suptitle(
                f"{row['label']} | rank {row['rank']}/{len(rows)} | "
                f"total {row['total_db']:+.2f} dB | "
                f"ecg-band {row['ecg_db']:+.2f} dB | "
                f"motion {row['motion_db']:+.2f} dB | "
                f"motion: {row['motion']}",
                fontsize=10,
            )
            _draw_channel_panels(
                (ax_e, ax_m, ax_c),
                raw[:, ch],
                ekgonly[:, ch] if ekgonly is not None else None,
                motiononly[:, ch] if motiononly is not None else None,
                cleaned[:, ch],
                time,
            )
            _draw_psd(ax_p, raw[:, ch], cleaned[:, ch], sr)
            pdf.savefig(fig)
            plt.close(fig)

    return out_path


__all__ = [
    "ICAResult",
    "CleaningConfig",
    "CleaningResult",
    "CleaningSession",
    "fit_ica",
    "score_components_against_ekg",
    "auto_select_ekg_components",
    "reconstruct_without_components",
    "regress_out_ekg_from_emg",
    "regress_out_motion_from_emg",
    "harmonize_multirate_inputs",
    "run_pipeline",
]
