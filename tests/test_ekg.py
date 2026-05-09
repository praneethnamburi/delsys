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
