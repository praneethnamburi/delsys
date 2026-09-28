"""Synthetic-signal tests for amplitude normalisation and the Rudolph co-contraction index."""

import json
import warnings

import numpy as np
import pytest

from delsys import EMG, SensorInfo
from delsys.emg import ENVELOPE_SR, Reference, cocontraction, normalize, reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chan(name, sr=2000.0, dur=4.0, amp=0.3, active=(1.0, 3.0), freq=100.0, noise=1e-4, seed=0):
    """Single-channel EMG: quiet -> sine burst -> quiet, named ``name``."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(dur * sr)) / sr
    sig = noise * rng.randn(t.size)
    on = (t >= active[0]) & (t <= active[1])
    sig[on] += amp * np.sin(2 * np.pi * freq * t[on])
    sensor = SensorInfo(name=name, modalities={"EMGS"}, number=1, type_sensorlog=None,
                        lrc="R", location=name)
    return EMG(sig.reshape(-1, 1), sr=sr, t0=0.0, meta={"sensor": sensor}, signal_names=[name])


@pytest.fixture
def pair():
    """An agonist/antagonist pair at DIFFERENT native rates -- the case that used to raise.

    1259 Hz single-differential against a 2222 Hz Quattro is the real Delsys combination.
    """
    return _chan("flexor", sr=1259.2592592592594), _chan("extensor", sr=2222.222222222222, seed=1)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def test_reference_keys_by_channel_name(pair):
    ref = reference(pair[0])
    assert list(ref) == ["flexor"]
    assert ref["flexor"] > 0


def test_reference_records_its_processing(pair):
    ref = reference(pair[0], win_size=0.1)
    assert ref.rms_kw["win_size"] == 0.1
    assert ref.stat == "max"


def test_reference_max_is_the_envelope_peak_not_the_raw_peak(pair):
    flex = pair[0]
    assert reference(flex)["flexor"] < np.abs(np.asarray(flex())).max()


def test_reference_percentile_is_below_max(pair):
    assert reference(pair[0], stat="p90")["flexor"] < reference(pair[0], stat="max")["flexor"]


def test_reference_round_trips(tmp_path, pair):
    ref = reference(pair[0], source_path="trial_01.h5")
    out = Reference.load(ref.save(str(tmp_path / "ref.json")))
    assert out.values == ref.values and out.stat == ref.stat and out.rms_kw == ref.rms_kw
    assert json.loads((tmp_path / "ref.json").read_text())["source"] == "trial_01.h5"


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_gives_fraction_of_reference(pair):
    flex = pair[0]
    assert np.nanmax(np.asarray(normalize(flex.rms(), reference(flex))())) == pytest.approx(1.0)


def test_normalize_missing_channel_raises(pair):
    with pytest.raises(KeyError):
        normalize(pair[1].rms(), reference(pair[0]))


def test_normalize_without_reference_raises(pair):
    with pytest.raises(ValueError, match="no reference"):
        normalize(pair[0].rms())


def test_normalize_warns_when_processed_differently(pair):
    ref = reference(pair[0], win_size=0.05)
    with pytest.warns(RuntimeWarning, match="processed differently"):
        normalize(pair[0].rms(win_size=0.25), ref)


def test_normalize_silent_when_processing_matches(pair):
    ref = reference(pair[0], win_size=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        normalize(pair[0].rms(win_size=0.05), ref)


# ---------------------------------------------------------------------------
# cocontraction -- the cross-rate regression
# ---------------------------------------------------------------------------


def test_cocontraction_across_native_rates(pair):
    """Regression: rms() cannot deliver a requested rate exactly, so the pair needs resampling."""
    flex, ext = pair
    assert flex.rms().sr != ext.rms().sr          # the premise: the grids really do differ
    assert cocontraction(flex, ext, reference=None).sr == pytest.approx(ENVELOPE_SR)


def test_cocontraction_honours_requested_envelope_sr(pair):
    assert cocontraction(*pair, reference=None, envelope_sr=100.0).sr == pytest.approx(100.0)


def test_cocontraction_is_symmetric(pair):
    a, b = pair
    assert np.allclose(np.asarray(cocontraction(a, b, reference=None)()),
                       np.asarray(cocontraction(b, a, reference=None)()))


def test_cocontraction_method_form_matches(pair):
    a, b = pair
    assert np.allclose(np.asarray(a.cocontraction(b, reference=None)()),
                       np.asarray(cocontraction(a, b, reference=None)()))


def test_cocontraction_zero_when_one_muscle_is_silent():
    """Agonist alone -> the ratio term collapses, so the index goes to (near) zero."""
    loud = _chan("agonist", sr=2000.0, amp=0.5)
    quiet = _chan("antagonist", sr=2000.0, amp=0.0, noise=1e-6, seed=2)
    assert np.nanmax(np.asarray(cocontraction(loud, quiet, reference=None)())) < 1e-3


def test_cocontraction_equals_sum_when_balanced():
    """lo == hi -> the ratio is 1 -> CCI is exactly the sum of the two activations."""
    a = _chan("a", sr=2000.0, amp=0.4, seed=3)
    b = _chan("b", sr=2000.0, amp=0.4, seed=3)          # identical signal
    x = np.asarray(cocontraction(a, b, reference=None)()).reshape(-1)
    env = np.asarray(a.rms()()).reshape(-1)
    assert np.nanmax(x) == pytest.approx(2 * np.nanmax(env), rel=1e-3)


def test_cocontraction_min_activation_zeroes_the_baseline():
    a, b = _chan("a", sr=2000.0, seed=4), _chan("b", sr=2000.0, seed=5)
    assert np.all(np.asarray(cocontraction(a, b, reference=None, min_activation=1.0)()) == 0)


def test_cocontraction_rejects_multichannel():
    multi = EMG(np.zeros((1000, 2)), sr=2000.0, signal_names=["a", "b"])
    with pytest.raises(ValueError, match="single-channel"):
        cocontraction(multi, _chan("b", sr=2000.0), reference=None)


def test_cocontraction_requires_overlap():
    a, b = _chan("a", sr=2000.0), _chan("b", sr=2000.0)
    b = EMG(np.asarray(b()), sr=b.sr, t0=1000.0, meta=b.meta, signal_names=["b"])
    with pytest.raises(ValueError, match="share no time span"):
        cocontraction(a, b, reference=None)


# ---------------------------------------------------------------------------
# cocontraction -- how the reference is resolved
# ---------------------------------------------------------------------------


def _stamp(channels, ref):
    for ch in channels:
        ch.meta = dict(ch.meta or {}, reference=ref)
    return channels


def test_cocontraction_inherits_stamped_reference(pair):
    ref = Reference({"flexor": 0.1, "extensor": 0.1})
    flex, ext = _stamp(pair, ref)
    got = cocontraction(flex, ext)
    assert got.meta["cocontraction"]["normalized"] is True
    assert np.allclose(np.asarray(got()), np.asarray(cocontraction(flex, ext, reference=ref)()))


def test_cocontraction_explicit_none_overrides_a_stamped_reference(pair):
    """``reference=None`` must mean *unnormalised*, not *fall back to the attached one*."""
    ref = Reference({"flexor": 0.1, "extensor": 0.1})
    flex, ext = _stamp(pair, ref)
    got = cocontraction(flex, ext, reference=None)
    assert got.meta["cocontraction"]["normalized"] is False
    assert not np.allclose(np.asarray(got()), np.asarray(cocontraction(flex, ext)()))


def test_cocontraction_records_native_rates(pair):
    meta = cocontraction(*pair, reference=None).meta["cocontraction"]
    assert meta["channels"] == ["flexor", "extensor"]
    assert meta["native_sr"] == [pytest.approx(1259.259, rel=1e-4),
                                 pytest.approx(2222.222, rel=1e-4)]
