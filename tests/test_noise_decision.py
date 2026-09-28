"""The noise track as a decision (``EventData`` fields) + the cross-channel noise detector.

The noise track carries ``default`` / ``added`` / ``removed`` / ``tags`` / ``algorithm_name`` /
``params`` -- deliberately ``datanavigator.events.EventData``'s field names, so the on-disk shape
is already ``EventData.asdict()``. These tests pin the two things that matter: every field
survives a write (the old writer flattened them away), and a detector re-run cannot undo human
work.
"""

import numpy as np
import pytest

from delsys import EMG, SensorInfo, _events
from delsys._noise import _canonical_value, _normalize_signal_value, detect_noise

KEY = "s1:EMG"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, rec):
    return _events.write_events(str(path), {"noise": {"kind": "noise", "signals": {KEY: rec}}})


def _read(path):
    return _events.read_events(str(path))["events"]["noise"]["signals"][KEY]


def _effective(val):
    return [tuple(w) for w in _normalize_signal_value(val)[0]]


# ---------------------------------------------------------------------------
# Legacy shapes lift in rather than migrate
# ---------------------------------------------------------------------------


def test_bare_list_is_read_as_added():
    """Every window in an existing sidecar was placed by a human, so it IS ``added``."""
    assert _canonical_value([[1.0, 2.0]]) == {"added": [[1.0, 2.0]]}


def test_legacy_windows_key_is_read_as_added():
    assert _canonical_value({"windows": [[1.0, 2.0]]}) == {"added": [[1.0, 2.0]]}


def test_legacy_windows_still_masks_the_same_span():
    assert _effective({"windows": [[1.0, 2.0]], "dead": [[9.0, 10.0]]}) == [(1.0, 2.0)]
    assert _normalize_signal_value({"windows": [[1.0, 2.0]], "dead": [[9.0, 10.0]]})[1] == [
        (9.0, 10.0)
    ]


def test_dead_true_sugar_survives():
    assert _canonical_value({"dead": True}) == {"dead": [[None, None]]}


def test_legacy_and_new_added_do_not_duplicate():
    got = _canonical_value({"windows": [[1.0, 2.0]], "added": [[1.0, 2.0]]})
    assert got == {"added": [[1.0, 2.0]]}


# ---------------------------------------------------------------------------
# default - removed + added
# ---------------------------------------------------------------------------


def test_effective_is_default_minus_removed_plus_added():
    rec = {"default": [[1.0, 2.0], [3.0, 4.0]], "removed": [[3.0, 4.0]], "added": [[7.0, 8.0]]}
    assert _effective(rec) == [(1.0, 2.0), (7.0, 8.0)]


def test_removed_tolerates_json_rounding():
    """``removed`` is matched against ``default`` by value, so a float round-trip must not
    resurrect a rejected window."""
    rec = {"default": [[1.0, 2.0]], "removed": [[1.0 + 1e-9, 2.0 - 1e-9]]}
    assert _effective(rec) == []


def test_removed_does_not_suppress_an_unrelated_window():
    rec = {"default": [[1.0, 2.0]], "removed": [[5.0, 6.0]]}
    assert _effective(rec) == [(1.0, 2.0)]


# ---------------------------------------------------------------------------
# The write must not eat fields -- this is the regression
# ---------------------------------------------------------------------------


def test_every_field_survives_a_write(tmp_path):
    p = tmp_path / "t.delsys-events"
    _write(p, {"default": [[1.0, 2.0]], "added": [[3.0, 4.0]], "removed": [[5.0, 6.0]],
               "tags": ["reviewed"], "algorithm_name": "bt", "params": {"threshold": 8.0}})
    got = _read(p)
    assert got["default"] == [[1.0, 2.0]]
    assert got["added"] == [[3.0, 4.0]]
    assert got["removed"] == [[5.0, 6.0]]
    assert got["tags"] == ["reviewed"]
    assert got["algorithm_name"] == "bt"
    assert got["params"] == {"threshold": 8.0}


def test_write_is_idempotent(tmp_path):
    p = tmp_path / "t.delsys-events"
    rec = {"default": [[1.0, 2.0]], "removed": [[1.0, 2.0]], "added": [[7.0, 8.0]],
           "tags": ["reviewed"], "algorithm_name": "bt", "params": {"k": 1}}
    _write(p, rec)
    once = _read(p)
    _write(p, once)
    assert _read(p) == once


def test_schema_is_3(tmp_path):
    p = tmp_path / "t.delsys-events"
    _write(p, {"added": [[1.0, 2.0]]})
    assert _events.read_events(str(p))["schema"] == 3


def test_a_rerun_cannot_undo_human_work(tmp_path):
    """The whole point of separating ``default`` from ``added``."""
    p = tmp_path / "t.delsys-events"
    _write(p, {"default": [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], "algorithm_name": "bt"})
    rec = _read(p)
    rec["removed"] = [[20.0, 21.0]]           # human rejects one proposal
    rec["added"] = [[50.0, 51.0]]             # ... and marks their own
    rec["tags"] = ["reviewed"]
    _write(p, rec)

    rec = _read(p)
    rec["default"] = [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0]]   # detector re-runs
    _write(p, rec)

    eff = _effective(_read(p))
    assert (20.0, 21.0) not in eff            # still rejected, though re-proposed
    assert (30.0, 31.0) not in eff            # superseded
    assert (40.0, 41.0) in eff                # newly proposed
    assert (50.0, 51.0) in eff                # the human's own mark
    assert _read(p)["tags"] == ["reviewed"]


# ---------------------------------------------------------------------------
# detect_noise
# ---------------------------------------------------------------------------


def _bilateral_log(spike_at=None, spike_ms=50, burst=None, sr=2000.0, dur=20.0, n_per_side=3):
    """A Log-like object with EMG on both sides.

    ``spike_at`` puts a brief transient on EVERY channel (an artifact); ``burst`` puts a longer
    one on the LEFT only (a gesture).
    """
    rng = np.random.RandomState(0)
    t = np.arange(int(dur * sr)) / sr
    sensors = []
    for side in ("L", "R"):
        for i in range(n_per_side):
            x = 1e-3 * rng.randn(t.size)
            if spike_at is not None:
                m = (t >= spike_at) & (t < spike_at + spike_ms / 1000.0)
                x[m] += 0.5 * np.sin(2 * np.pi * 100 * t[m])
            if burst is not None and side == "L":
                m = (t >= burst) & (t < burst + 1.0)
                x[m] += 0.3 * np.sin(2 * np.pi * 100 * t[m])
            name = f"{side}Chan{i}"
            info = SensorInfo(name=name, modalities={"EMGS"}, number=i, type_sensorlog=None,
                              lrc=side, location=name)
            emg = EMG(x.reshape(-1, 1), sr=sr, t0=0.0, meta={"sensor": info}, signal_names=[name])
            sensors.append(type("S", (), {"emg": emg})())
    return type("L", (), {"sensors": sensors, "fname": "synthetic.h5"})()


def test_detects_a_bilateral_transient():
    got = detect_noise(_bilateral_log(spike_at=10.0))
    assert got, "a brief transient on every channel should be flagged"
    assert any(abs(c["t0"] - 10.0) < 0.3 for c in got)
    assert got[0]["n_left"] >= 2 and got[0]["n_right"] >= 2


def test_ignores_a_one_sided_burst():
    """A gesture is unilateral -- which is what kept the real 277.5 s event off the list."""
    assert detect_noise(_bilateral_log(burst=10.0)) == []


def test_quiet_recording_yields_nothing():
    assert detect_noise(_bilateral_log()) == []


def test_write_none_touches_no_disk(tmp_path):
    p = tmp_path / "x.delsys-events"
    detect_noise(_bilateral_log(spike_at=10.0), path=str(p))
    assert not p.exists()


def test_write_noise_lands_in_default(tmp_path, fixtures_dir):
    """Keys must come from the Log's Signals, and must RESOLVE -- a window under a key the
    masking core cannot resolve is silently inert, which is worse than a crash."""
    import delsys
    from delsys._noise import _signal_label, _write_detected, resolve_key

    lf = delsys.Log(str(fixtures_dir / "discover164_mvc.csv"))
    name = _signal_label(lf.signals[0])
    p = tmp_path / "x.delsys-events"
    _write_detected(lf, [{"t0": 1.0, "t1": 1.2, "channels": [name]}], str(p),
                    params={"threshold": 8.0})

    signals = _events.read_events(str(p))["events"]["noise"]["signals"]
    assert len(signals) == 1
    key, rec = next(iter(signals.items()))
    assert rec["default"] == [[1.0, 1.2]], "proposals belong in default, never in added"
    assert "added" not in rec
    assert rec["algorithm_name"] == "bilateral-transient"
    assert rec["params"]["threshold"] == pytest.approx(8.0)
    assert resolve_key(lf, key), f"key {key!r} does not resolve against the Log"


def test_write_refuses_a_log_with_no_addressable_signals():
    """The synthetic Log carries EMG bundles but no Signals, so nothing can be keyed."""
    from delsys._noise import _write_detected

    with pytest.raises(ValueError, match="no addressable signals"):
        _write_detected(_bilateral_log(spike_at=10.0),
                        [{"t0": 1.0, "t1": 1.2, "channels": ["LChan0"]}], "unused.json")


def test_rejects_an_unknown_write_target():
    with pytest.raises(ValueError, match="unknown write target"):
        detect_noise(_bilateral_log(spike_at=10.0), write="everything")


def test_one_sided_montage_warns():
    lf = _bilateral_log(spike_at=10.0)
    lf.sensors = [s for s in lf.sensors
                  if s.emg.meta["sensor"].lrc == "L"]      # drop the right arm
    with pytest.warns(RuntimeWarning, match="bilateral test"):
        detect_noise(lf)
