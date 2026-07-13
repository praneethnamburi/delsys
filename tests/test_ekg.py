"""Synthetic-ECG tests for ``EKG``: R-peak detection plus rate properties and
metadata round-trip."""

import neurokit2 as nk
import numpy as np
import pytest

from delsys import EKG, SensorInfo

# ---------------------------------------------------------------------------
# Synthetic ECG fixture (30 s of simulated ECG at a known heart rate)
# ---------------------------------------------------------------------------

SR = 1259.0
HEART_RATE_BPM = 70
DURATION_S = 30
EXPECTED_PEAKS = DURATION_S * HEART_RATE_BPM / 60  # ≈ 35


def _ekg_sensor():
    return SensorInfo(
        name="ekg1",
        modalities={"EKG"},
        number=1,
        type_sensorlog=None,
        lrc="C",
        location="Chest",
    )


@pytest.fixture(scope="module")
def synthetic_ekg():
    """30 s of simulated ECG at 70 bpm — wrapped in our ``EKG`` class."""
    sig = nk.ecg_simulate(
        duration=DURATION_S,
        sampling_rate=int(SR),
        heart_rate=HEART_RATE_BPM,
        random_state=0,
    )
    sig_2d = sig.reshape(-1, 1)  # production EKG bundles are 2D
    return EKG(sig_2d, sr=SR, t0=0.0, meta={"sensor": _ekg_sensor()})


# ---------------------------------------------------------------------------
# meta initialization
# ---------------------------------------------------------------------------


def test_ekg_initializes_rpeak_metadata(synthetic_ekg):
    """``EKG.__init__`` populates ``self.meta`` with empty rpeak slots."""
    for key in (
        "rpeaks_idx_default",
        "rpeaks_idx_removed",
        "rpeaks_idx_added",
        "noisy_segments_idx",
        "is_flipped",
        "tags",
    ):
        assert key in synthetic_ekg.meta


def test_ekg_preserves_sensor(synthetic_ekg):
    assert synthetic_ekg.sensor.name == "ekg1"


# ---------------------------------------------------------------------------
# R-peak detection
# ---------------------------------------------------------------------------


def _peaks_per_minute(n_peaks):
    return 60 * n_peaks / DURATION_S


def test_find_rpeaks_pn_recovers_simulated_rate(synthetic_ekg):
    """find_rpeaks_pn (alias find_rpeaks) is the highpass-then-prune version."""
    peaks = synthetic_ekg.find_rpeaks_pn()
    rate = _peaks_per_minute(len(peaks))
    assert abs(rate - HEART_RATE_BPM) < 5, f"PN rate {rate:.1f} bpm vs target {HEART_RATE_BPM}"


def test_find_rpeaks_alias_points_to_pn(synthetic_ekg):
    """``find_rpeaks`` is an alias for ``find_rpeaks_pn``."""
    assert EKG.find_rpeaks is EKG.find_rpeaks_pn


# ---------------------------------------------------------------------------
# hr / rr properties (HeartPy-backed)
# ---------------------------------------------------------------------------


def test_hr_property_matches_simulated_rate(synthetic_ekg):
    assert abs(synthetic_ekg.hr - HEART_RATE_BPM) < 5


def test_rr_property_returns_finite_value(synthetic_ekg):
    """Respiration rate via HeartPy. Don't assert tight bounds — the simulated
    ECG doesn't carry a known respiration component, just check it's finite."""
    rr = synthetic_ekg.rr
    assert np.isfinite(rr)
    assert rr > 0


# ---------------------------------------------------------------------------
# rpeak_times + ihr — instantaneous heart rate over time
# ---------------------------------------------------------------------------


def test_rpeak_times_returns_sorted_indices_and_idx_array(synthetic_ekg):
    """Calling ``rpeak_times`` populates default rpeaks if empty and returns a
    sorted array of sample indices plus a peak-index array of equal length."""
    # Reset state from earlier tests in this module-scoped fixture.
    synthetic_ekg.meta["rpeaks_idx_default"] = []
    synthetic_ekg.meta["rpeaks_idx_added"] = []
    synthetic_ekg.meta["rpeaks_idx_removed"] = []

    x, pk_idx = synthetic_ekg.rpeak_times()
    assert len(x) == len(pk_idx)
    assert np.all(np.diff(x) > 0), "rpeak indices not sorted"


def test_ihr_returns_finite_bpm(synthetic_ekg):
    """instantaneous-HR series should have at least one finite value near the simulated HR."""
    synthetic_ekg.meta["rpeaks_idx_default"] = []
    synthetic_ekg.meta["rpeaks_idx_added"] = []
    synthetic_ekg.meta["rpeaks_idx_removed"] = []
    xdata, ydata = synthetic_ekg.ihr()
    finite_y = [y for y in ydata if np.isfinite(y)]
    assert len(finite_y) > 0
    median_bpm = np.median(finite_y)
    assert abs(median_bpm - HEART_RATE_BPM) < 5


# ---------------------------------------------------------------------------
# flip_signal — toggles is_flipped flag and refinds peaks
# ---------------------------------------------------------------------------


def test_flip_signal_toggles_flag(synthetic_ekg):
    initial = synthetic_ekg.meta.get("is_flipped", False)
    synthetic_ekg.flip_signal()
    assert synthetic_ekg.meta["is_flipped"] is not initial
    synthetic_ekg.flip_signal()
    assert synthetic_ekg.meta["is_flipped"] is initial


# ---------------------------------------------------------------------------
# 0.2.0: find_rpeaks_pn rejects multi-channel input
# ---------------------------------------------------------------------------


def _two_channel_ekg(sr=SR, duration=DURATION_S):
    sig = nk.ecg_simulate(
        duration=duration, sampling_rate=int(sr), heart_rate=HEART_RATE_BPM, random_state=0
    )
    sig_2col = np.column_stack([sig, sig])
    sensors = [
        SensorInfo(
            name=f"ekg{i}",
            modalities={"EKG"},
            number=i + 1,
            type_sensorlog=None,
            lrc="C",
            location=f"Chest{i}",
        )
        for i in range(2)
    ]
    return EKG(
        sig_2col,
        sr=sr,
        t0=0.0,
        meta={"sensors": sensors},
        signal_names=["Chest0", "Chest1"],
        signal_coords=["ekg"],
    )


def test_ekg_find_rpeaks_pn_raises_on_multi_channel():
    """Multi-channel EKG aggregates can't go through HeartPy's 1D peak
    detector. Raise NotImplementedError that points at split."""
    agg = _two_channel_ekg()
    with pytest.raises(NotImplementedError, match="split_by_signal_name"):
        agg.find_rpeaks_pn()


def test_ekg_find_rpeaks_pn_works_on_split():
    """Documented migration path: split the aggregate by signal_name and
    detect peaks on each per-channel slice."""
    agg = _two_channel_ekg()
    parts = agg.split_by_signal_name()
    assert len(parts) == 2
    peaks = parts[0].find_rpeaks_pn()
    rate = 60 * len(peaks) / DURATION_S
    assert abs(rate - HEART_RATE_BPM) < 5


# ---------------------------------------------------------------------------
# Grid-independent review decision (rpeaks_decision / apply_rpeaks_decision)
# ---------------------------------------------------------------------------

from scipy.signal import resample_poly  # noqa: E402


def _fresh_ekg(sig, sr):
    return EKG(np.asarray(sig, dtype=float).reshape(-1, 1), sr=sr, t0=0.0,
               meta={"sensor": _ekg_sensor()})


def _curate(ekg):
    """Detect, then simulate a human edit: remove the 10th peak, add a fake beat."""
    ekg.find_rpeaks()
    removed_idx = ekg.meta["rpeaks_idx_default"][10]
    ekg.meta["rpeaks_idx_removed"] = sorted(
        set(ekg.meta["rpeaks_idx_removed"]) | {removed_idx}
    )
    mid_t = float((ekg.t[ekg.meta["rpeaks_idx_default"][5]]
                   + ekg.t[ekg.meta["rpeaks_idx_default"][6]]) / 2)
    ekg.meta["rpeaks_idx_added"] = [int(np.argmin(np.abs(ekg.t - mid_t)))]
    ekg.meta["tags"] = ["reviewed"]
    return removed_idx, mid_t


def test_rpeaks_decision_stores_times_and_only_human_removed():
    sr = 200
    sig = nk.ecg_simulate(duration=30, sampling_rate=sr, heart_rate=70, random_state=1)
    ekg = _fresh_ekg(sig, sr)
    removed_idx, mid_t = _curate(ekg)
    dec = ekg.rpeaks_decision()
    # added carries the fake-beat time (not an index)
    assert any(abs(mid_t - a) < 2.0 / sr for a in dec["added"])
    # removed excludes the detector's own auto-prune -> only the human removal
    assert dec["removed"] == [pytest.approx(float(ekg.t[removed_idx]), abs=1e-9)]
    assert dec["detector"]["name"] == "pn"
    assert dec["tags"] == ["reviewed"]


def test_apply_rpeaks_decision_reproduces_across_grids():
    sr1, sr2 = 200, 250
    sig = np.asarray(
        nk.ecg_simulate(duration=30, sampling_rate=sr1, heart_rate=70, random_state=1),
        dtype=float,
    )
    ekg1 = _fresh_ekg(sig, sr1)
    removed_idx, mid_t = _curate(ekg1)
    curated1 = sorted(float(ekg1.t[i]) for i in ekg1._get_rpeaks_from_meta())
    dec = ekg1.rpeaks_decision()

    ekg2 = _fresh_ekg(resample_poly(sig, sr2, sr1), sr2)
    final2 = ekg2.apply_rpeaks_decision(dec)
    curated2 = sorted(float(ekg2.t[i]) for i in final2)

    tol = 2.0 / sr1
    assert len(curated1) == len(curated2)
    for x in curated1:
        assert min(abs(x - y) for y in curated2) <= tol
    # human edits survived the grid change
    assert any(abs(mid_t - y) <= tol for y in curated2)          # added kept
    removed_t = float(ekg1.t[removed_idx])
    assert all(abs(removed_t - y) > tol for y in curated2)       # removed stays gone


def test_apply_rpeaks_decision_is_idempotent_same_grid():
    sr = 200
    sig = nk.ecg_simulate(duration=30, sampling_rate=sr, heart_rate=70, random_state=2)
    ekg = _fresh_ekg(sig, sr)
    _curate(ekg)
    expected = sorted(ekg._get_rpeaks_from_meta())
    dec = ekg.rpeaks_decision()
    again = _fresh_ekg(sig, sr).apply_rpeaks_decision(dec)
    assert sorted(again) == expected


def test_apply_rpeaks_decision_noise_windows_drop_peaks():
    sr = 200
    sig = nk.ecg_simulate(duration=30, sampling_rate=sr, heart_rate=70, random_state=3)
    ekg = _fresh_ekg(sig, sr)
    ekg.find_rpeaks()
    dec = ekg.rpeaks_decision()
    ekg2 = _fresh_ekg(sig, sr)
    ekg2.apply_rpeaks_decision(dec, noise_windows=[[12.0, 13.0]])
    clean_t = ekg2.t[ekg2.rpeak_times()[0]]
    assert all(not (12.0 <= float(t) <= 13.0) for t in clean_t)


def test_apply_rpeaks_decision_flip_reproduces():
    sr = 200
    sig = np.asarray(
        nk.ecg_simulate(duration=30, sampling_rate=sr, heart_rate=70, random_state=4),
        dtype=float,
    )
    ekg = _fresh_ekg(-sig, sr)
    ekg.flip_signal()  # toggles is_flipped + re-detects on negated
    dec = ekg.rpeaks_decision()
    assert dec["flipped"] is True
    ekg2 = _fresh_ekg(-sig, sr)
    ekg2.apply_rpeaks_decision(dec)
    assert ekg2.meta["is_flipped"] is True
    assert len(ekg2._get_rpeaks_from_meta()) > 20


def test_apply_rpeaks_decision_rejects_unknown_detector():
    sr = 200
    sig = nk.ecg_simulate(duration=10, sampling_rate=sr, heart_rate=70, random_state=5)
    ekg = _fresh_ekg(sig, sr)
    with pytest.raises(ValueError, match="unknown rpeak detector"):
        ekg.apply_rpeaks_decision({"detector": {"name": "iw"}, "added": [], "removed": []})


def test_rpeaks_decision_raises_on_multi_channel():
    agg = _two_channel_ekg()
    with pytest.raises(NotImplementedError):
        agg.rpeaks_decision()
