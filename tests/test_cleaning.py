"""Tests for ``delsys.cleaning`` and ``Log.clean_emg_ekg_artifact``.

The end-to-end coverage uses synthetic signals (band-passed white noise
EMG + ``neurokit2.ecg_simulate(method='simple')``) so the SNR-recovery
thresholds have a known ground truth. The Log-integration coverage uses
one fixture, since the cleaning algorithm is fixture-independent.
"""

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import neurokit2 as nk  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from delsys import CleaningConfig, CleaningResult, Log  # noqa: E402
from delsys.cleaning import (  # noqa: E402
    auto_select_ekg_components,
    fit_ica,
    harmonize_multirate_inputs,
    reconstruct_without_components,
    regress_out_ekg_from_emg,
    regress_out_motion_from_emg,
    run_pipeline,
    score_components_against_ekg,
)

# ---------------------------------------------------------------------------
# Synthetic-signal helpers
# ---------------------------------------------------------------------------


def _bandpassed_noise(n_samples: int, sr: float, lo: float, hi: float, rng) -> np.ndarray:
    """Tight-band white noise via FFT mask — keeps the EMG band realistic."""
    x = rng.standard_normal(n_samples)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    fft[~mask] = 0.0
    y = np.fft.irfft(fft, n=n_samples)
    return y / max(np.std(y), 1e-12)


def _simulate_ecg(n_samples: int, sr: float, hr_bpm: float, seed: int) -> np.ndarray:
    """Periodic ECG-like signal via neurokit2's ``method='simple'`` simulator.

    The default ``ecgsyn`` method has a slice-index bug on some installs
    when ``duration * sampling_rate`` isn't a whole number; the simple
    method avoids that and is more than realistic enough for SNR tests.
    """
    sig = nk.ecg_simulate(
        sampling_rate=int(round(sr)),
        length=int(n_samples),
        heart_rate=hr_bpm,
        method="simple",
        random_state=seed,
    )
    sig = np.asarray(sig)
    return (sig - np.mean(sig)) / max(np.std(sig), 1e-12)


def _build_synthetic_emg_with_ecg(
    *,
    n_samples: int,
    sr: float,
    n_channels: int = 4,
    snr_db: float = 0.0,
    hr_bpm: float = 70.0,
    seed: int = 0,
):
    """Return ``(clean_emg, contaminated_emg, ekg_1d)`` at the same sample count."""
    rng = np.random.default_rng(seed)
    clean = np.column_stack(
        [_bandpassed_noise(n_samples, sr, 30.0, 350.0, rng) for _ in range(n_channels)]
    )
    ekg = _simulate_ecg(n_samples, sr, hr_bpm, seed)

    # Mix at the requested SNR. Each channel gets a different (positive) gain
    # so component selection has to find one source rather than memorize a
    # constant scale.
    sig_power = np.mean(clean**2, axis=0)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    gains = rng.uniform(0.6, 1.4, size=n_channels) * np.sqrt(noise_power)
    contaminated = clean + np.outer(ekg, gains)
    return clean, contaminated, ekg


def _power_at_harmonic(sig_1d: np.ndarray, sr: float, freq_hz: float, half_bw: float = 1.5) -> float:
    """Mean power in a narrow band around ``freq_hz`` (used to score ECG removal)."""
    fft = np.fft.rfft(sig_1d)
    freqs = np.fft.rfftfreq(sig_1d.shape[0], d=1.0 / sr)
    band = (freqs >= freq_hz - half_bw) & (freqs <= freq_hz + half_bw)
    if not np.any(band):
        return 0.0
    return float(np.mean(np.abs(fft[band]) ** 2))


# ---------------------------------------------------------------------------
# Unit tests — building blocks
# ---------------------------------------------------------------------------


def test_fit_ica_recovers_two_sources():
    rng = np.random.default_rng(0)
    n = 2000
    s1 = np.sign(np.sin(2 * np.pi * np.arange(n) / 50.0))  # square
    s2 = rng.uniform(-1, 1, n)  # uniform noise
    sources = np.column_stack([s1, s2])
    mix = np.array([[1.0, 0.5], [0.5, 1.0]])
    x = sources @ mix.T

    res = fit_ica(x, n_components=2, random_state=0)
    assert res.sources.shape == (n, 2)
    assert res.mixing.shape == (2, 2)
    # ICA recovers sources up to permutation and sign — so check that
    # SOME pairing produces high absolute correlation on each row.
    corr = np.abs(np.corrcoef(sources.T, res.sources.T)[:2, 2:])
    best_per_truth = corr.max(axis=1)
    assert np.all(best_per_truth > 0.85), best_per_truth


def test_score_components_against_ekg_picks_lagged_match():
    n = 4000
    rng = np.random.default_rng(1)
    ekg = rng.standard_normal(n)
    # IC0 is a lag-3 copy of ekg with noise; IC1 is unrelated.
    ic0 = np.zeros(n)
    ic0[3:] = ekg[:-3]
    ic0 += 0.1 * rng.standard_normal(n)
    ic1 = rng.standard_normal(n)
    sources = np.column_stack([ic0, ic1])

    scores, lags = score_components_against_ekg(sources, ekg, max_lag_samples=10)
    assert scores[0] > 0.85
    assert scores[0] > scores[1]
    assert lags[0] == 3  # discovered the synthetic delay


def test_auto_select_ekg_components_threshold_and_cap():
    scores = np.array([0.5, 0.1, 0.4, 0.7])
    picked = auto_select_ekg_components(scores, min_corr=0.3, max_components=2)
    # Ranked by descending score; cap at 2.
    assert picked == [3, 0]


def test_auto_select_ekg_components_keeps_strong_when_below_threshold():
    scores = np.array([0.05, 0.22, 0.18])
    picked = auto_select_ekg_components(scores, min_corr=0.3, max_components=1)
    # No score clears 0.3, but max ≥ 0.2, so keep IC1.
    assert picked == [1]


def test_reconstruct_without_components_zeroes_selected():
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.standard_normal((n, 3))
    res = fit_ica(x, n_components=3, random_state=0)
    cleaned, src_clean = reconstruct_without_components(res, [0])
    assert cleaned.shape == x.shape
    assert np.allclose(src_clean[:, 0], 0.0)
    # IC1 / IC2 untouched.
    assert np.allclose(src_clean[:, 1], res.sources[:, 1])
    assert np.allclose(src_clean[:, 2], res.sources[:, 2])


def test_regress_out_ekg_from_emg_recovers_signal():
    rng = np.random.default_rng(3)
    n = 2000
    ekg = rng.standard_normal(n)
    target = rng.standard_normal(n)
    contaminated = (target + 1.5 * ekg).reshape(-1, 1)
    cleaned, beta = regress_out_ekg_from_emg(
        contaminated, ekg, max_lag_samples=2, ridge_alpha=1e-6
    )
    assert cleaned.shape == contaminated.shape
    # The lag-0 column of the design matrix is the middle of the lag block.
    # Regardless of which lag column carries it, the residual should be
    # close to the target.
    corr = np.corrcoef(cleaned[:, 0], target)[0, 1]
    assert corr > 0.9, f"residual should track the un-contaminated target; got {corr}"


def test_regress_out_motion_safety_gate_rejects_aggressive_cleaning():
    rng = np.random.default_rng(4)
    n = 1000
    # ACC is unrelated to EMG, so the regression has nothing to fit and
    # the safety gate should still pass (residual ≈ input).
    emg = rng.standard_normal((n, 1))
    acc = rng.standard_normal((n, 3))
    cleaned, betas, diag = regress_out_motion_from_emg(
        emg, {0: acc}, min_variance_ratio=0.1, min_power_ratio=0.1
    )
    assert cleaned.shape == emg.shape
    assert diag["per_channel"][0]["used"] is True

    # Now force a rejection: EMG is exactly equal to ACC's first axis, so
    # the regression collapses the residual to ~0, which trips the gate.
    emg_signal = acc[:, 0:1].copy()
    cleaned2, _, diag2 = regress_out_motion_from_emg(
        emg_signal,
        {0: acc},
        min_variance_ratio=0.5,
        min_power_ratio=0.5,
        max_lag_samples=0,
    )
    assert diag2["per_channel"][0]["reason"] == "rejected_by_safety_gate"
    np.testing.assert_array_equal(cleaned2, emg_signal)


def test_regress_out_motion_missing_predictor_passes_through():
    rng = np.random.default_rng(5)
    n = 500
    emg = rng.standard_normal((n, 2))
    cleaned, _, diag = regress_out_motion_from_emg(emg, {0: rng.standard_normal((n, 3))})
    assert diag["per_channel"][1]["reason"] == "missing_acc_predictor"
    np.testing.assert_array_equal(cleaned[:, 1], emg[:, 1])


# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------


def test_harmonize_multirate_inputs_aligns_lengths():
    rng = np.random.default_rng(6)
    sr_emg, sr_ekg, sr_acc = 1920.0, 120.0, 148.0
    emg = rng.standard_normal((int(sr_emg * 1.0), 2))
    ekg = rng.standard_normal(int(sr_ekg * 1.0))
    acc = rng.standard_normal((int(sr_acc * 1.0), 3))
    out = harmonize_multirate_inputs(
        emg_2d=emg,
        emg_sr=sr_emg,
        ekg_1d=ekg,
        ekg_sr=sr_ekg,
        acc_by_emg={0: acc, 1: acc},
        acc_sr=sr_acc,
        target_sr=sr_emg,
    )
    n = out["n_samples"]
    assert out["emg"].shape[0] == n
    assert out["ekg"].shape[0] == n
    assert all(v.shape[0] == n for v in out["acc_by_emg"].values())
    assert out["sr"] == sr_emg


# ---------------------------------------------------------------------------
# End-to-end SNR recovery — the headline test for the cleaner.
# ---------------------------------------------------------------------------


def test_run_pipeline_recovers_clean_emg_under_ecg_contamination():
    sr = 1920.0
    n = int(sr * 8)  # 8s — enough for FastICA to converge
    clean_emg, contaminated, ekg = _build_synthetic_emg_with_ecg(
        n_samples=n, sr=sr, n_channels=8, snr_db=-3.0, hr_bpm=70.0, seed=7
    )

    cfg = CleaningConfig(use_motion_stage=False, preprocess_highpass_hz=20.0)
    result = run_pipeline(contaminated, sr=sr, ekg_1d=ekg, config=cfg)
    assert isinstance(result, CleaningResult)
    assert result.cleaned_emg.shape == contaminated.shape

    # Per-channel correlation against the ground-truth clean EMG. The
    # preprocess high-pass discards <20 Hz, so the comparison must also
    # ignore that band — high-pass the truth before correlating.
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 20.0 / (sr / 2), btype="highpass", output="sos")
    truth_hp = sosfiltfilt(sos, clean_emg, axis=0)
    cleaned = result.cleaned_emg

    corrs = [
        np.corrcoef(cleaned[:, ch], truth_hp[:, ch])[0, 1] for ch in range(cleaned.shape[1])
    ]
    assert min(corrs) >= 0.85, f"per-channel correlation too low: {corrs}"

    # ECG harmonics should drop ≥ 6 dB on the contaminated channels.
    hr_hz = 70.0 / 60.0
    for ch in range(cleaned.shape[1]):
        before = sum(
            _power_at_harmonic(contaminated[:, ch], sr, h * hr_hz) for h in (1, 2, 3)
        )
        after = sum(
            _power_at_harmonic(cleaned[:, ch], sr, h * hr_hz) for h in (1, 2, 3)
        )
        ratio_db = 10.0 * np.log10(max(after, 1e-30) / max(before, 1e-30))
        assert ratio_db <= -6.0, f"channel {ch}: ECG-band attenuation only {ratio_db:.1f} dB"


# ---------------------------------------------------------------------------
# Log integration — invariants of clean_emg_ekg_artifact
# ---------------------------------------------------------------------------


def _load(fixtures_dir, name, tmp_path):
    src = fixtures_dir / name
    dst = tmp_path / name
    shutil.copy(src, dst)
    return Log(str(dst))


def test_log_clean_in_place_preserves_invariants(fixtures_dir, tmp_path):
    """Splice-back keeps every structural invariant: signal count,
    aggregate signal_names, sensors[*].emg labels, sensor count."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    pre_n_signals = len(lf.signals)
    pre_emg_names = list(lf.emg.signal_names)
    pre_emg_meta_sensors = list(lf.emg.meta["sensors"])
    pre_n_sensors = len(lf.sensors)
    per_sensor_emg_names = {
        s.number: list(s.emg.signal_names) for s in lf.sensors if hasattr(s, "emg")
    }
    raw_emg = lf.emg().copy()

    result = lf.clean_emg_ekg_artifact(generate_report=False)

    assert isinstance(result, CleaningResult)
    assert len(lf.signals) == pre_n_signals
    assert lf.emg.signal_names == pre_emg_names
    assert len(lf.emg.meta["sensors"]) == len(pre_emg_meta_sensors)
    assert len(lf.sensors) == pre_n_sensors
    for sensor in lf.sensors:
        if hasattr(sensor, "emg"):
            assert sensor.emg.signal_names == per_sensor_emg_names[sensor.number]

    # Cleaning actually changed the EMG, and lf.emg now matches the result.
    assert not np.array_equal(lf.emg(), raw_emg)
    np.testing.assert_allclose(lf.emg(), result.cleaned_emg)


def test_log_clean_in_place_false_does_not_mutate(fixtures_dir, tmp_path):
    """``in_place=False`` leaves ``lf.signals`` referentially identical."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    signals_id = id(lf.signals)
    raw_emg = lf.emg().copy()
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)

    assert id(lf.signals) == signals_id
    np.testing.assert_array_equal(lf.emg(), raw_emg)
    # Returned cleaned matrix has the right shape regardless of mutation.
    assert result.cleaned_emg.shape[1] == lf.emg().shape[1]


def test_log_clean_motion_auto_pairs_per_sensor(fixtures_dir, tmp_path):
    """``motion='auto'`` pairs each EMG sensor with its own ACC bundle.
    Sensors without ACC (e.g. EMGQ-only Quattro sensors in this fixture)
    show ``reason='missing_acc_predictor'`` in the diagnostics."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    result = lf.clean_emg_ekg_artifact(in_place=False, motion="auto", generate_report=False)

    per_channel = result.diagnostics["motion"]["per_channel"]
    paired = [c for c in per_channel if c["used"]]
    unpaired = [c for c in per_channel if not c["used"]]
    # At least one of each in this fixture (EMGS sensors carry ACC,
    # EMGQ sensors do not).
    assert paired, "expected at least one ACC-paired EMG channel"
    assert unpaired, "expected at least one EMG channel with no ACC pair"
    for c in unpaired:
        assert c["reason"] == "missing_acc_predictor"


def test_log_clean_motion_none_skips_motion_stage(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    result = lf.clean_emg_ekg_artifact(in_place=False, motion=None, generate_report=False)
    assert result.diagnostics["motion"] == {"used": False}


def test_log_clean_motion_dict_explicit_pairing(fixtures_dir, tmp_path):
    """``motion={emg_num: acc_num}`` resolves to the ACC bundle on the
    target sensor."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    # Build a single-key map: pair EMG sensor 1's motion to ACC sensor 2.
    # (Both exist in this fixture.)
    nums = {s.number for s in lf.sensors}
    assert {1, 2}.issubset(nums)
    result = lf.clean_emg_ekg_artifact(in_place=False, motion={1: 2}, generate_report=False)
    per_channel = result.diagnostics["motion"]["per_channel"]
    # Sensor 1 EMGS is the first EMG channel in the aggregate; it should
    # be marked used.
    assert per_channel[0]["used"] is True
    # No other EMG sensor was paired.
    assert all(c["reason"] == "missing_acc_predictor" for c in per_channel[1:])


def test_log_clean_motion_dict_unknown_sensor_raises(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    with pytest.raises(ValueError):
        lf.clean_emg_ekg_artifact(in_place=False, motion={1: 99999}, generate_report=False)


def test_log_clean_no_emg_raises(fixtures_dir, tmp_path, monkeypatch):
    """An EMG-less Log can't be cleaned."""
    lf = _load(fixtures_dir, "discover164_basic.csv", tmp_path)
    # Every committed fixture has EMG, so monkey-patch the property to
    # exercise the validation branch deterministically.
    monkeypatch.setattr(type(lf), "emg", property(lambda self: None))
    with pytest.raises(ValueError):
        lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)


def test_log_clean_manual_components_override(fixtures_dir, tmp_path):
    """Manual ``ecg_components_to_remove`` is honored alongside auto-detection."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    assert lf.emg is not None and lf.ekg is not None

    cfg = CleaningConfig(
        ecg_auto_remove_components=False,
        ecg_components_to_remove=[0],
        use_motion_stage=False,
    )
    result = lf.clean_emg_ekg_artifact(in_place=False, config=cfg, generate_report=False)
    assert 0 in result.diagnostics["ecg"]["components_removed"]
    assert result.diagnostics["ecg"]["auto_ekg_components_removed"] == []


# ---------------------------------------------------------------------------
# Stage-isolated variants and reporting (0.4.x)
# ---------------------------------------------------------------------------


def test_run_pipeline_populates_ekgonly_and_motiononly():
    """``run_pipeline`` exposes preprocess+ECG and preprocess+motion variants."""
    rng = np.random.default_rng(11)
    sr = 200.0
    n = int(sr * 4)
    emg = rng.standard_normal((n, 2)) * 0.3
    ekg = _simulate_ecg(n, sr, hr_bpm=70.0, seed=11)
    contaminated = emg + np.outer(ekg, np.array([0.6, 0.8]))
    acc = rng.standard_normal((n, 3))
    cfg = CleaningConfig(preprocess_highpass_hz=None)
    result = run_pipeline(
        contaminated, sr=sr, ekg_1d=ekg, acc_by_emg={0: acc, 1: acc}, config=cfg
    )

    # ekg-only equals the post-ECG snapshot in stages.
    np.testing.assert_array_equal(result.cleaned_emg_ekgonly, result.stages["post_ecg"])
    # motion-only is set when both stages would run.
    assert result.cleaned_emg_motiononly is not None
    # motion-only path differs from the combined path (the order
    # changes the residual).
    assert not np.allclose(result.cleaned_emg_motiononly, result.cleaned_emg)


def test_run_pipeline_skipped_stages_yield_none_variants():
    rng = np.random.default_rng(12)
    sr = 200.0
    n = int(sr * 2)
    emg = rng.standard_normal((n, 2))

    # No ekg -> ekgonly=None; no acc -> motiononly=None.
    result = run_pipeline(emg, sr=sr, ekg_1d=None, acc_by_emg=None, config=CleaningConfig())
    assert result.cleaned_emg_ekgonly is None
    assert result.cleaned_emg_motiononly is None


def test_generate_report_writes_pdf(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)

    out = result.generate_report(path=tmp_path / "out.pdf")
    assert out == tmp_path / "out.pdf"
    assert out.exists()
    assert out.stat().st_size > 0
    with open(out, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_generate_report_default_path_uses_fname(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)
    assert result.fname is not None

    out = result.generate_report()
    expected = tmp_path / "discover170_cleaning_report.pdf"
    assert out == expected
    assert out.exists()


def test_generate_report_raises_without_fname_or_path(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)
    result.fname = None
    with pytest.raises(ValueError):
        result.generate_report()


def test_log_clean_auto_report_default_writes_pdf(fixtures_dir, tmp_path):
    """The default ``generate_report=True`` produces a sibling PDF."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    expected = tmp_path / "discover170_cleaning_report.pdf"
    assert not expected.exists()
    lf.clean_emg_ekg_artifact(in_place=False)
    assert expected.exists()
    assert expected.stat().st_size > 0


def test_log_clean_auto_report_opt_out(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    expected = tmp_path / "discover170_cleaning_report.pdf"
    lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)
    assert not expected.exists()


def test_review_constructs_and_keys_advance_channel(fixtures_dir, tmp_path):
    """``review()`` builds a figure and the key handler advances the channel index."""
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)

    plt.close("all")
    result.review()
    fig = plt.gcf()
    state = fig._delsys_review_state
    assert state["idx"] == 0
    n = len(state["order"])
    assert n == result.cleaned_emg.shape[1]

    class _Ev:
        pass

    ev = _Ev()
    ev.key = "right"
    state["_on_key"](ev)
    assert state["idx"] == 1
    ev.key = "right"
    state["_on_key"](ev)
    assert state["idx"] == 2
    ev.key = "left"
    state["_on_key"](ev)
    assert state["idx"] == 1
    ev.key = "end"
    state["_on_key"](ev)
    assert state["idx"] == n - 1
    ev.key = "home"
    state["_on_key"](ev)
    assert state["idx"] == 0
    # Wrap on previous from 0.
    ev.key = "left"
    state["_on_key"](ev)
    assert state["idx"] == n - 1

    # Overlay toggles.
    assert state["show_ekgonly"] is True
    ev.key = "e"
    state["_on_key"](ev)
    assert state["show_ekgonly"] is False
    # 'o' folds all-on / all-off: if any are on, turn them all off.
    ev.key = "o"
    state["_on_key"](ev)
    assert state["show_ekgonly"] is False
    assert state["show_motiononly"] is False
    assert state["show_cleaned"] is False
    # second 'o' flips them all back on.
    ev.key = "o"
    state["_on_key"](ev)
    assert state["show_ekgonly"] is True
    assert state["show_motiononly"] is True
    assert state["show_cleaned"] is True
    plt.close("all")


def test_review_channels_arg_restricts_order(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, "discover170.csv", tmp_path)
    result = lf.clean_emg_ekg_artifact(in_place=False, generate_report=False)

    plt.close("all")
    result.review(channels=[3, 0, 1])
    state = plt.gcf()._delsys_review_state
    assert state["order"] == [3, 0, 1]
    plt.close("all")
