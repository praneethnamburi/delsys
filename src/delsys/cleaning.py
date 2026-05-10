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
from typing import Any, Dict, List, Optional, Union

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
    """

    cleaned_emg: np.ndarray
    sr: float
    time: np.ndarray
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    coefficients: Dict[str, Any] = field(default_factory=dict)


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


def _run_ecg_stage(
    pre_emg: np.ndarray,
    ekg_1d: np.ndarray,
    config: CleaningConfig,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """ICA-based ECG suppression with optional residual regression.

    Concatenates the EKG as the trailing column of the EMG matrix so
    FastICA decomposes the joint structure. Components are zeroed by
    manual override (when ``config.ecg_components_to_remove`` is set)
    or auto-detection (lagged correlation against the EKG); the EKG
    column is then dropped from the reconstructed output. When
    ``ecg_use_regression=True``, a second pass ridge-regresses the
    lagged EKG out of every cleaned channel.

    Returns a dict with ``cleaned_emg`` (n, n_emg_channels), the IC
    bookkeeping, and the regression beta when run.
    """
    ecg_input = np.column_stack([pre_emg, np.asarray(ekg_1d).reshape(-1)])
    feat_with_ekg = (
        feature_names + ["EKG"] if feature_names is not None else None
    )
    ica_result = fit_ica(
        ecg_input,
        n_components=config.ecg_n_components,
        random_state=config.ica_random_state,
        feature_names=feat_with_ekg,
    )

    manual = _normalize_component_selection(
        config.ecg_components_to_remove, ica_result.sources.shape[1]
    )

    auto_components: List[int] = []
    ic_corr_scores = None
    ic_best_lags = None
    if config.ecg_auto_remove_components:
        ic_corr_scores, ic_best_lags = score_components_against_ekg(
            ica_result.sources,
            ekg_1d=ecg_input[:, -1],
            max_lag_samples=config.ecg_corr_max_lag_samples,
        )
        auto_components = auto_select_ekg_components(
            ic_corr_scores,
            min_corr=config.ecg_corr_threshold,
            keep_at_least_one_when_strong=True,
            max_components=config.ecg_max_auto_components,
        )

    selected_all = sorted(set(manual + auto_components))
    cleaned_full, _ = reconstruct_without_components(ica_result, selected_all)

    # Drop the EKG column from the reconstructed output.
    cleaned_emg = cleaned_full[:, :-1]

    regression_beta = None
    if config.ecg_use_regression:
        cleaned_emg, regression_beta = regress_out_ekg_from_emg(
            cleaned_emg,
            ekg_1d=ecg_input[:, -1],
            max_lag_samples=config.ecg_reg_max_lag_samples,
            ridge_alpha=config.ecg_reg_ridge_alpha,
        )

    return {
        "cleaned_emg": cleaned_emg,
        "components_removed": selected_all,
        "manual_components_removed": manual,
        "auto_ekg_components_removed": auto_components,
        "ic_ekg_corr_scores": ic_corr_scores,
        "ic_ekg_best_lags": ic_best_lags,
        "regression_beta": regression_beta,
    }


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
        }
    else:
        diagnostics["ecg"] = {"used": False}
    stages["post_ecg"] = post_ecg

    post_motion = post_ecg
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
    else:
        diagnostics["motion"] = {"used": False}
    stages["cleaned"] = post_motion

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
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases (one release window).
# ---------------------------------------------------------------------------

#: Alias for :class:`CleaningConfig` — original name from
#: ``pn-projects/projects/emg_ica_cleaning.py``. Kept for one release
#: window so downstream code switching to ``delsys.cleaning`` need only
#: change its import path. Prefer :class:`CleaningConfig` in new code.
EMGPipelineConfig = CleaningConfig

#: Alias for :class:`CleaningResult`. See :data:`EMGPipelineConfig`.
EMGPipelineResult = CleaningResult

#: Alias for :func:`fit_ica`. See :data:`EMGPipelineConfig`.
fit_ica_emg = fit_ica

#: Alias for :func:`score_components_against_ekg`. See :data:`EMGPipelineConfig`.
score_ica_components_against_ekg = score_components_against_ekg

#: Alias for :func:`run_pipeline`. See :data:`EMGPipelineConfig`.
run_emg_pipeline = run_pipeline


__all__ = [
    "ICAResult",
    "CleaningConfig",
    "CleaningResult",
    "fit_ica",
    "score_components_against_ekg",
    "auto_select_ekg_components",
    "reconstruct_without_components",
    "regress_out_ekg_from_emg",
    "regress_out_motion_from_emg",
    "harmonize_multirate_inputs",
    "run_pipeline",
    # back-compat aliases
    "EMGPipelineConfig",
    "EMGPipelineResult",
    "fit_ica_emg",
    "score_ica_components_against_ekg",
    "run_emg_pipeline",
]
