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


def test_review_add_near_removed_restores_it(tmp_path):
    """Adding at a removed peak's location restores it (drops from removed) rather
    than creating a duplicate in added. Robust to the detector's own auto-prune,
    which pre-populates removed."""
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    r = ekg.review()
    ch = r._cur()
    removed0 = list(ch.meta["rpeaks_idx_removed"])  # baseline auto-prune
    i10 = int(ch.meta["rpeaks_idx_default"][10])
    t10 = float(ch.t[i10])
    r._remove_rpeak(_Ev(t10))
    assert i10 in ch.meta["rpeaks_idx_removed"]     # human removal recorded
    r._add_rpeak(_Ev(t10))
    assert i10 not in ch.meta["rpeaks_idx_removed"]  # restored, not shadowed
    assert ch.meta["rpeaks_idx_removed"] == removed0  # back to baseline
    assert ch.meta["rpeaks_idx_added"] == []          # no duplicate


def test_review_remove_added_undoes_it(tmp_path):
    """Removing a peak you added undoes the addition (drops from added) rather than
    pushing it into removed."""
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    r = ekg.review()
    ch = r._cur()
    removed0 = list(ch.meta["rpeaks_idx_removed"])
    mid_t = float((ch.t[ch.meta["rpeaks_idx_default"][5]] + ch.t[ch.meta["rpeaks_idx_default"][6]]) / 2)
    r._add_rpeak(_Ev(mid_t))
    assert len(ch.meta["rpeaks_idx_added"]) == 1
    added_idx = int(ch.meta["rpeaks_idx_added"][0])
    r._remove_rpeak(_Ev(float(ch.t[added_idx])))
    assert ch.meta["rpeaks_idx_added"] == []            # undone
    assert added_idx not in ch.meta["rpeaks_idx_removed"]  # not pushed to removed
    assert ch.meta["rpeaks_idx_removed"] == removed0       # removed unchanged


def test_review_has_help_button(tmp_path):
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    r = ekg.review()
    assert "Help (ctrl+k)" in r.buttons
    assert callable(getattr(r, "show_key_bindings", None))


def test_nav_added_steps_through_added_peaks_and_rolls_across_channels(tmp_path):
    """nav='added': w visits each added peak in order, rolls over to the next channel that has
    any (skipping one that has none), keeps the zoom; q walks back across the boundary."""
    a = _synth_ekg(number=1, seed=1)
    empty = _synth_ekg(number=2, seed=2)
    b = _synth_ekg(number=3, seed=3)
    for i, ch in enumerate((a, empty, b)):
        ch.meta["source"] = str(tmp_path / f"Trial_{i}.h5")
    from delsys import rpeak_review

    r = rpeak_review.launch([a, empty, b], nav="added")
    for ch, ts in ((a, (5.0, 12.0)), (b, (8.0,))):
        ch.meta["rpeaks_idx_added"] = [int(round(t * ch.sr)) for t in ts]
    r._ax_raw.set_xlim(0.0, 2.0)                      # a 2 s zoom, centred at 1 s
    visits = []
    for _ in range(3):
        r._jump_suspect(+1)
        lo, hi = r._ax_raw.get_xlim()
        visits.append((r._current_idx, round((lo + hi) / 2, 2), round(hi - lo, 2)))
    assert visits == [(0, 5.0, 2.0), (0, 12.0, 2.0), (2, 8.0, 2.0)]   # skipped the empty one
    r._jump_suspect(+1)                                # end of list: stays put
    assert r._current_idx == 2
    r._jump_suspect(-1)                                # back across the boundary to a's LAST
    lo, hi = r._ax_raw.get_xlim()
    assert r._current_idx == 0 and round((lo + hi) / 2, 2) == 12.0


def test_nav_review_window_restricts_targets(tmp_path):
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    from delsys import rpeak_review

    r = rpeak_review.launch([ekg], nav="added")
    ekg.meta["rpeaks_idx_added"] = [int(round(t * ekg.sr)) for t in (3.0, 15.0, 25.0)]
    ekg.meta["review_window"] = (10.0, 20.0)
    assert [t for t, _ in r._targets(ekg)] == [pytest.approx(15.0)]


def test_remove_on_restored_autopruned_peak_removes_the_beat(tmp_path):
    """A restored auto-prune is in BOTH added and default; `d` on it must remove the beat, and
    the removal must survive save + reload (it used to be re-recorded as added)."""
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    r = ekg.review()
    ch = r._cur()
    beat = ch.meta["rpeaks_idx_default"][7]
    ch.meta["rpeaks_idx_added"] = [beat]                 # the restored-auto-prune shape
    r._remove_rpeak(_Ev(float(ch.t[beat])))
    assert beat not in ch._get_rpeaks_from_meta()
    r.save()
    again = _synth_ekg()
    again.meta["source"] = ekg.meta["source"]
    again.load_rpeaks()
    assert beat not in again._get_rpeaks_from_meta()
