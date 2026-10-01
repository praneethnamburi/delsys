"""``EKG.nn_intervals`` / ``EKG.rmssd``: ectopic beats and noise gaps never reach an HRV index.

Peaks are planted directly in ``meta`` (no detection), so every expected value is exact.
"""

import numpy as np
import pytest

from delsys import EKG

SR = 1000.0


def _ekg(peak_times, duration=20.0):
    e = EKG(np.zeros((int(duration * SR), 1)), sr=SR, t0=0.0)
    e.meta["rpeaks_idx_default"] = [int(round(t * SR)) for t in peak_times]
    return e


def _rmssd_ms(rr_s):
    d = np.diff(np.asarray(rr_s) * 1000.0)
    return float(np.sqrt(np.mean(d ** 2)))


# alternating 0.80 / 0.84 s -> every successive difference is exactly 40 ms
SINUS = np.cumsum([1.0] + [0.80, 0.84] * 8)


def test_clean_record_matches_naive_rmssd():
    e = _ekg(SINUS)
    t, rr, ok = e.nn_intervals()
    assert ok.all() and len(rr) == len(SINUS) - 1
    assert e.rmssd() == pytest.approx(_rmssd_ms(np.diff(SINUS)))
    assert e.rmssd() == pytest.approx(40.0)


def test_ectopic_excludes_both_intervals_it_bounds():
    peaks = SINUS.copy()
    peaks[5] = peaks[4] + 0.45               # premature beat ...
    e = _ekg(peaks)
    e.meta["rpeaks_idx_ectopic"] = [int(round(peaks[5] * SR))]
    _, rr, ok = e.nn_intervals()
    assert list(np.flatnonzero(~ok)) == [4, 5]   # the short interval and the compensatory pause
    # the surviving differences are all the 40 ms sinus alternation -- the ectopic left no trace
    assert e.rmssd() == pytest.approx(40.0)
    # ... whereas the naive computation is badly inflated by it
    assert _rmssd_ms(np.diff(peaks)) > 100.0


def test_never_differences_across_an_excluded_interval():
    peaks = SINUS.copy()
    peaks[5] = peaks[4] + 0.45
    e = _ekg(peaks)
    e.meta["rpeaks_idx_ectopic"] = [int(round(peaks[5] * SR))]
    _, rr, ok = e.nn_intervals()
    pair_ok = ok[:-1] & ok[1:]
    # intervals 3 and 6 are both NN but not consecutive: no difference may join them
    assert not pair_ok[3:6].any()
    assert pair_ok.sum() == len(rr) - 1 - 3


def test_noisy_segment_breaks_the_series():
    e = _ekg(SINUS)
    i_mid = int(round(SINUS[6] * SR))
    e.meta["noisy_segments_idx"] = [[i_mid - 50, i_mid + 50]]   # swallows one peak
    t, rr, ok = e.nn_intervals()
    assert len(rr) == len(SINUS) - 2           # one peak gone -> one fewer interval
    gap = np.flatnonzero(~ok)
    assert len(gap) == 1 and rr[gap[0]] > 1.5  # the spanning "interval" is a gap, flagged
    assert e.rmssd() == pytest.approx(40.0)


def test_window_keeps_intervals_with_both_peaks_inside():
    e = _ekg(SINUS)
    t, rr, ok = e.nn_intervals(t_start=SINUS[2] - 1e-6, t_end=SINUS[10] + 1e-6)
    assert len(rr) == 8
    assert t[0] == pytest.approx(SINUS[3]) and t[-1] == pytest.approx(SINUS[10])


def test_rmssd_nan_when_nothing_survives():
    e = _ekg(SINUS[:3])
    e.meta["rpeaks_idx_ectopic"] = [int(round(SINUS[1] * SR))]
    assert np.isnan(e.rmssd())
