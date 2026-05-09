"""Synthetic-signal tests for ``EMG``: envelope extraction, TKEO, feature dicts."""
import numpy as np
import pandas as pd
import pytest

from delsys import EMG, SensorInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emg_sensor():
    return SensorInfo(
        name="emg1", modalities={"EMGS"}, number=1,
        type_sensorlog=None, lrc="C", location="Bicep",
    )


def _sine_burst_emg(sr=2000.0, dur=2.0, burst=(0.5, 1.5), freq=100.0, amp=0.3, noise=0.02, seed=0):
    """Return an EMG with silence -> sine burst -> silence (shape ``(N, 1)``)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    sig = noise * rng.randn(len(t))  # background noise
    burst_mask = (t >= burst[0]) & (t <= burst[1])
    sig[burst_mask] += amp * np.sin(2 * np.pi * freq * t[burst_mask])
    sig = sig.reshape(-1, 1)  # production EMG bundles are 2D (N, channels)
    return EMG(sig, sr=sr, t0=0.0, meta={"sensor": _emg_sensor()}), t, burst_mask


# ---------------------------------------------------------------------------
# .process — envelope2 should highlight the burst region
# ---------------------------------------------------------------------------

@pytest.fixture
def burst_emg():
    return _sine_burst_emg()


def test_emg_process_envelope2_returns_emg(burst_emg):
    emg, _, _ = burst_emg
    env = emg.process(amp_kind="envelope2")
    assert isinstance(env, EMG)
    assert env.sr == emg.sr
    # Same number of samples and channels.
    assert env.shape == emg.shape


def test_emg_process_burst_region_has_higher_amplitude(burst_emg):
    """envelope2 -> the burst window should have substantially higher mean amplitude
    than the silence window."""
    emg, t, _ = burst_emg
    env = emg.process(amp_kind="envelope2")
    e = env().flatten()
    silence_mask = (env.t >= 0.05) & (env.t < 0.4)
    burst_mask = (env.t >= 0.7) & (env.t < 1.3)
    silence_mean = e[silence_mask].mean()
    burst_mean = e[burst_mask].mean()
    assert burst_mean > 5 * silence_mean, (
        f"burst_mean={burst_mean:.4g} not >> silence_mean={silence_mean:.4g}"
    )


@pytest.mark.parametrize("kind", ["envelope2", "envelope", "rms", "mean"])
def test_emg_process_kinds_supported(burst_emg, kind):
    """All supported amp_kind values produce an EMG output of the correct sr."""
    emg, _, _ = burst_emg
    if kind in ("rms", "mean"):
        out = emg.process(amp_kind=kind, win_size=0.05, win_inc=0.025)
    else:
        out = emg.process(amp_kind=kind)
    assert isinstance(out, EMG)
    assert out.sr == emg.sr


def test_emg_process_invalid_kind_raises(burst_emg):
    emg, _, _ = burst_emg
    with pytest.raises(ValueError, match="Not supported kind"):
        emg.process(amp_kind="bogus")


def test_emg_process_lowpass_chains(burst_emg):
    """Passing ``lowpass`` kicks in a post-envelope lowpass filter; output stays EMG."""
    emg, _, _ = burst_emg
    out = emg.process(amp_kind="envelope2", lowpass=5)
    assert isinstance(out, EMG)


# ---------------------------------------------------------------------------
# .tkeo — Teager–Kaiser energy preserves shape and sensor metadata
# ---------------------------------------------------------------------------

def test_tkeo_preserves_shape_and_sensor(burst_emg):
    emg, _, _ = burst_emg
    out = emg.tkeo()
    assert isinstance(out, EMG)
    assert out.shape == emg.shape
    assert out.sensor is emg.sensor


def test_tkeo_zero_for_constant_signal():
    """TKEO of a perfectly constant signal is ~0 everywhere."""
    si = _emg_sensor()
    sig = np.full((500, 1), 0.7)
    emg = EMG(sig, sr=2000.0, t0=0.0, meta={"sensor": si})
    out = emg.tkeo()
    assert np.allclose(out(), 0, atol=1e-12)


# ---------------------------------------------------------------------------
# .get_features — temp + freq feature dictionary
# ---------------------------------------------------------------------------

def test_get_features_temporal_keys(burst_emg):
    """Temporal features include canonical keys (mean, rms, mav, zcr, ...)."""
    emg, _, _ = burst_emg
    feats = emg.get_features(kind="temp", win_size=0.25, win_inc=0.1)
    for key in ("mean", "rms", "mav", "zcr", "wamp"):
        assert key in feats, f"missing temporal feature {key!r}"
    # Time vector matches feature length.
    n = len(feats["mean"])
    assert len(feats["time"]) == n


def test_get_features_frequency_keys(burst_emg):
    """Frequency features include mean / peak / median frequency keys."""
    emg, _, _ = burst_emg
    feats = emg.get_features(kind="freq", win_size=0.25, win_inc=0.1)
    for key in ("mnf", "mwf", "mav", "pkf", "frr"):
        assert key in feats


def test_get_features_invalid_kind_raises(burst_emg):
    emg, _, _ = burst_emg
    with pytest.raises(ValueError, match="Not supported kind"):
        emg.get_features(kind="bogus")


def test_get_features_to_dataframe(burst_emg):
    """get_features returns a dict; callers wrap with pd.DataFrame for tabular form."""
    emg, _, _ = burst_emg
    feats = pd.DataFrame(emg.get_features(kind="temp", win_size=0.25, win_inc=0.1))
    assert isinstance(feats, pd.DataFrame)
    assert "mean" in feats.columns
