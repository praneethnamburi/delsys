"""Headless drive of the interactive EKG reviewer (``EKG.review()``).

Exercises construction + the edit/save actions under matplotlib Agg (no live
event loop): add / remove / noisy-segment / flip / tag / mode, then Save to the
``<stem>.delsys-events`` sidecar and a reload that reproduces the curation.
Skipped if ``datanavigator`` (the optional GUI dependency) isn't installed.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("datanavigator")

import neurokit2 as nk  # noqa: E402

from delsys import EKG, SensorInfo, _events  # noqa: E402


class _Ev:
    """Minimal stand-in for a matplotlib event (cursor x only)."""

    def __init__(self, t):
        self.xdata = float(t)
        self.inaxes = None


def _synth_ekg(sr=200, dur=30, number=12, location="Chest", seed=1):
    sig = np.asarray(
        nk.ecg_simulate(duration=dur, sampling_rate=sr, heart_rate=70, random_state=seed),
        dtype=float,
    ).reshape(-1, 1)
    s = SensorInfo(
        name=f"ekg{number}", modalities={"EKG"}, number=number,
        type_sensorlog=None, lrc="C", location=location,
    )
    return EKG(sig, sr=sr, t0=0.0, meta={"sensor": s},
               signal_names=[location], signal_coords=["ekg"])


@pytest.fixture
def reviewed(tmp_path):
    """Open a reviewer on a synthetic EKG, drive a full curation, return context."""
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    r = ekg.review()
    ch = r._cur()
    assert len(ch.meta["rpeaks_idx_default"]) > 20
    # flip first (re-detects) so subsequent edits survive; flip back to a clean base
    r._flip()
    assert ch.meta["is_flipped"] is True
    r._flip()
    assert ch.meta["is_flipped"] is False
    # add a beat between defaults 5 and 6
    mid_t = float((ch.t[ch.meta["rpeaks_idx_default"][5]] + ch.t[ch.meta["rpeaks_idx_default"][6]]) / 2)
    r._add_rpeak(_Ev(mid_t))
    # remove the peak nearest default[10]
    t10 = float(ch.t[ch.meta["rpeaks_idx_default"][10]])
    r._remove_rpeak(_Ev(t10))
    # noisy segment + tag
    r._mark_noise(_Ev(12.0))
    r._mark_noise(_Ev(13.0))
    r._tag("reviewed")
    return r, ch, ekg, mid_t, t10


def test_review_actions_update_meta(reviewed):
    _r, ch, _ekg, mid_t, _t10 = reviewed
    assert len(ch.meta["rpeaks_idx_added"]) == 1
    assert len(ch.meta["rpeaks_idx_removed"]) >= 1
    assert len(ch.meta["noisy_segments_idx"]) == 1
    assert "reviewed" in ch.meta["tags"]
    assert any(abs(mid_t - float(ch.t[i])) < 0.05 for i in ch.meta["rpeaks_idx_added"])


def test_review_save_writes_decision_and_noise(reviewed):
    r, _ch, ekg, mid_t, t10 = reviewed
    path = r.save()
    assert path == _events.events_path_for(ekg.meta["source"])
    rp = _events.read_rpeaks_signals(path)
    assert list(rp) == ["12.EKG.A | Chest"]
    dec = rp["12.EKG.A | Chest"]
    assert dec["tags"] == ["reviewed"]
    assert any(abs(mid_t - a) < 0.05 for a in dec["added"])
    assert any(abs(t10 - rmv) < 0.15 for rmv in dec["removed"])  # human removal persisted
    assert "12.EKG.A | Chest" in _events.read_noise_signals(path)


def test_review_reload_reproduces_curation(reviewed):
    r, _ch, ekg, mid_t, _t10 = reviewed
    r.save()
    fresh = _synth_ekg()
    fresh.meta["source"] = ekg.meta["source"]
    assert fresh.load_rpeaks() is True
    assert fresh.meta["tags"] == ["reviewed"]
    curated = sorted(float(fresh.t[i]) for i in fresh._get_rpeaks_from_meta())
    assert any(abs(mid_t - y) < 0.05 for y in curated)  # added reproduced


def test_review_mode_cycles(reviewed):
    r, _ch, _ekg, _mid, _t10 = reviewed
    m0 = r._mode
    r._cycle_mode()
    assert r._mode != m0


def test_review_multichannel_steps_channels(tmp_path):
    # two EKG sensors -> aggregate; review() splits + steps
    sig = nk.ecg_simulate(duration=30, sampling_rate=200, heart_rate=70, random_state=0)
    two = np.column_stack([sig, sig])
    sensors = [
        SensorInfo(name=f"ekg{i}", modalities={"EKG"}, number=i + 1,
                   type_sensorlog=None, lrc="C", location=f"Chest{i}")
        for i in range(2)
    ]
    agg = EKG(two, sr=200, t0=0.0, meta={"sensors": sensors},
              signal_names=["Chest0", "Chest1"], signal_coords=["ekg"])
    agg.meta["source"] = str(tmp_path / "Trial_2.h5")
    r = agg.review()
    assert len(r._channels) == 2
