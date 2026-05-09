"""Tests for ``delsys.ica``: ICA-based artifact decomposition and cleaning.

Uses a tiny ``Log``-shaped mock (only ``__getitem__`` and ``fname`` are needed)
so we can supply controlled, multi-location accelerometer data without
fabricating a full CSV.
"""
import json

import matplotlib
matplotlib.use("Agg")  # disable interactive backend before any pyplot import

import numpy as np
import pytest

from delsys.ica import _ica_data_preprocess, ica_cleaning, ica_components


# ---------------------------------------------------------------------------
# Tiny Log/Sensor/IMU mock — only the surface ICA touches.
# ---------------------------------------------------------------------------

class _FakeIMU:
    """Mimics the slice-and-call shape that ``_ica_data_preprocess`` uses."""

    def __init__(self, data: np.ndarray):
        self._data = data  # shape (N, 3)

    def __call__(self):
        return self._data

    def __getitem__(self, time_slice):
        return _FakeIMU(self._data[time_slice])


class _FakeSensor:
    def __init__(self, acc_data: np.ndarray):
        self.acc = _FakeIMU(acc_data)


class _FakeLog:
    """``lf[location]`` -> _FakeSensor; ``lf.fname`` for ica_cleaning's JSON path."""

    def __init__(self, location_to_acc, fname):
        self._lookup = {loc: _FakeSensor(arr) for loc, arr in location_to_acc.items()}
        self.fname = fname

    def __getitem__(self, location):
        return self._lookup[location]


# ---------------------------------------------------------------------------
# Synthetic mixed-source data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_lf(tmp_path):
    """Two locations, each with a (N, 3) accelerometer signal that mixes a
    few independent sources. ICA should be able to separate the sources from
    the linear mixture."""
    rng = np.random.RandomState(42)
    n = 1000
    t = np.linspace(0, 5, n)

    # Three independent source signals:
    s1 = np.sin(2 * np.pi * 2.0 * t)
    s2 = np.sign(np.sin(2 * np.pi * 0.7 * t))
    s3 = rng.randn(n) * 0.1

    sources = np.column_stack([s1, s2, s3])

    # Two locations, each gets a different linear mixing matrix.
    mix_A = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    mix_B = np.array([[0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
    acc_A = sources @ mix_A
    acc_B = sources @ mix_B

    fname = str(tmp_path / "fake_log.csv")
    return _FakeLog({"LFoot": acc_A, "RFoot": acc_B}, fname)


# ---------------------------------------------------------------------------
# _ica_data_preprocess — stack locations × axes, mean-subtract
# ---------------------------------------------------------------------------

def test_preprocess_stacks_all_locations_and_axes(synthetic_lf):
    """Two locations × 3 axes -> 6 columns; row count matches the source signal length."""
    data = _ica_data_preprocess(synthetic_lf, ["LFoot", "RFoot"], time_slice=None)
    assert data.shape == (1000, 6)


def test_preprocess_mean_subtracts_each_channel(synthetic_lf):
    """Each output channel should have ~zero mean after preprocessing."""
    data = _ica_data_preprocess(synthetic_lf, ["LFoot", "RFoot"], time_slice=None)
    assert np.allclose(data.mean(axis=0), 0, atol=1e-10)


def test_preprocess_respects_time_slice(synthetic_lf):
    """When ``time_slice`` is supplied, only the slice is processed."""
    data = _ica_data_preprocess(synthetic_lf, ["LFoot"], time_slice=slice(100, 300))
    assert data.shape == (200, 3)


# ---------------------------------------------------------------------------
# ica_components — runs FastICA, returns transformed sources + mixing matrix
# ---------------------------------------------------------------------------

def test_ica_components_returns_correct_shapes(synthetic_lf):
    x_transformed, mixing = ica_components(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        time_slice=None, n_components=3, showplot=False,
    )
    assert x_transformed.shape == (1000, 3)
    assert mixing.shape == (6, 3)


def test_ica_components_default_n_components(synthetic_lf):
    """When ``n_components`` is None it defaults to the number of input columns."""
    x_transformed, mixing = ica_components(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        time_slice=None, n_components=None, showplot=False,
    )
    # 2 locations × 3 axes = 6 columns -> 6 components
    assert x_transformed.shape == (1000, 6)
    assert mixing.shape == (6, 6)


def test_ica_components_recovers_independent_sources(synthetic_lf):
    """For a known linear mixture, ICA should recover sources whose absolute
    correlation with the originals is high (signs and order can flip)."""
    # We fed in 3 independent sources, mixed into 6 channels (2 locs × 3 axes).
    # ICA with n_components=3 should recover the 3 sources up to sign/order.
    x_transformed, _ = ica_components(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        time_slice=None, n_components=3, showplot=False,
    )

    rng = np.random.RandomState(42)
    n = 1000
    t = np.linspace(0, 5, n)
    s1 = np.sin(2 * np.pi * 2.0 * t) - np.mean(np.sin(2 * np.pi * 2.0 * t))
    s2 = np.sign(np.sin(2 * np.pi * 0.7 * t))
    s2 = s2 - np.mean(s2)
    sources = np.column_stack([s1, s2])

    # For each original source, find the best-correlated recovered component.
    best_corrs = []
    for i in range(sources.shape[1]):
        corrs = [abs(np.corrcoef(sources[:, i], x_transformed[:, j])[0, 1])
                 for j in range(x_transformed.shape[1])]
        best_corrs.append(max(corrs))
    assert all(c > 0.9 for c in best_corrs), f"weak ICA recovery: {best_corrs}"


# ---------------------------------------------------------------------------
# ica_cleaning — settings JSON round-trip + signal shape preserved
# ---------------------------------------------------------------------------

def test_ica_cleaning_writes_settings_json(synthetic_lf):
    """First call writes ica_settings.json with the supplied parameters."""
    cleaned = ica_cleaning(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        components_to_remove=[0], time_slice=None, n_components=None,
    )
    settings_path = synthetic_lf.fname.split(".")[0] + "_ica_settings.json"
    with open(settings_path) as f:
        settings = json.load(f)
    assert settings["sensor_locs"] == ["LFoot", "RFoot"]
    assert settings["components_to_remove"] == [0]
    assert cleaned.shape == (1000, 6)


def test_ica_cleaning_loads_existing_settings(synthetic_lf, tmp_path):
    """A second call with no overrides reads from the JSON written by the first."""
    # First call seeds the JSON.
    ica_cleaning(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        components_to_remove=[1], time_slice=slice(0, 500), n_components=None,
    )
    # Second call without overrides — should pull params from JSON.
    cleaned = ica_cleaning(
        synthetic_lf,
        sensor_locs=None, components_to_remove=None,
        time_slice=None, n_components=None,
    )
    # Used time_slice 0..500 from JSON -> 500 rows.
    assert cleaned.shape == (500, 6)


def test_ica_cleaning_returns_finite_output(synthetic_lf):
    """Cleaning with at least one component removed produces finite reconstruction
    of the right shape. (We avoid asserting "different components → different output"
    because ``ica_cleaning`` forces ``n_components = data.shape[1]``, which over-
    decomposes our rank-3 synthetic data and yields near-noise-floor outputs.)"""
    out = ica_cleaning(
        synthetic_lf, sensor_locs=["LFoot", "RFoot"],
        components_to_remove=[0], time_slice=None, n_components=None,
    )
    assert out.shape == (1000, 6)
    assert np.all(np.isfinite(out))
