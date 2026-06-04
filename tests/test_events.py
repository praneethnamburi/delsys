"""Unified annotation sidecar (``<stem>.delsys-events``): format + collapse + bridge.

Companion to ``test_noise_sidecar.py`` (legacy ``.delsys-noise`` grammar + masking)
and ``test_clean.py`` (batch consumption). Here: the one-file ``{schema, events}``
round-trip across the built-in ``noise`` type and typed marker types, the
per-signal → trial-level collapse (keep-all + provenance + proximity dedupe), the
legacy ``.delsys-noise`` bridge, and noise masking off the unified file.
``discover170.csv`` is the end-to-end masking fixture.
"""

import shutil

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import delsys  # noqa: E402
from delsys import _events  # noqa: E402
from delsys._noise import (  # noqa: E402
    SIDECAR_SUFFIX,
    format_signal_key,
    sidecar_path_for,
    write_noise_sidecar,
)

FIXTURE = "discover170.csv"


def _load(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return delsys.Log(str(csv))


# ---------------------------------------------------------------------------
# Format round-trip
# ---------------------------------------------------------------------------


def test_events_path_for():
    assert _events.events_path_for("/a/b/Trial_5.h5") == "/a/b/Trial_5.delsys-events"


def test_write_read_roundtrip_mixed_types(tmp_path):
    p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "noise": {"signals": {"3.EMGS | T": [[1.0, 2.0]], "9.FSR.C | F": {"dead": True}}},
            "1": {"size": 1, "signals": {"3.EMGS | T": [[1.4], [2.1]]}},
            "2": {"size": 2, "signals": {"4.ACC | B": [[0.5, 0.9]]}},
        },
    )
    doc = _events.read_events(str(p))
    assert doc["schema"] == _events.EVENTS_SCHEMA
    ev = doc["events"]
    # noise canonicalized via _noise
    assert ev["noise"]["kind"] == "noise"
    assert ev["noise"]["signals"]["3.EMGS | T"] == {"windows": [[1.0, 2.0]]}
    assert ev["noise"]["signals"]["9.FSR.C | F"] == {"dead": [[None, None]]}
    # markers keep kind + size
    assert ev["1"]["kind"] == "marker" and ev["1"]["size"] == 1
    assert ev["1"]["signals"]["3.EMGS | T"] == [[1.4], [2.1]]
    assert ev["2"]["size"] == 2


def test_write_drops_empty_sections(tmp_path):
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "noise": {"signals": {"3.EMGS | T": {"windows": []}}},  # empty -> dropped
            "1": {"size": 1, "signals": {"3.EMGS | T": []}},  # empty -> dropped
            "2": {"size": 2, "signals": {"4.ACC | B": [[0.5, 0.9]]}},
        },
    )
    ev = _events.read_events(str(p))["events"]
    assert "noise" not in ev
    assert "1" not in ev
    assert "2" in ev


def test_read_missing_file_is_empty(tmp_path):
    doc = _events.read_events(str(tmp_path / "nope.delsys-events"))
    assert doc == {"schema": _events.EVENTS_SCHEMA, "events": {}}


def test_marker_note_tags_roundtrip(tmp_path):
    """An event may carry a per-event note/tags (object form); bare events stay bare."""
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "1": {
                "size": 1,
                "signals": {
                    "3.EMGS | T": [
                        [1.40],  # bare
                        {"seq": [2.10], "note": "good example", "tags": ["clean"]},
                    ]
                },
            }
        },
    )
    # on disk: bare stays a list, annotated stays an object
    disk = _events.read_events(str(p))["events"]["1"]["signals"]["3.EMGS | T"]
    assert disk[0] == [1.40]
    assert disk[1] == {"seq": [2.10], "note": "good example", "tags": ["clean"]}
    # records expose note/tags; seq-only view drops them
    recs = _events.read_marker_records(str(p), "1")["3.EMGS | T"]
    assert recs[0] == {"seq": [1.40], "note": None, "tags": []}
    assert recs[1]["note"] == "good example" and recs[1]["tags"] == ["clean"]
    assert _events.read_marker_signals(str(p), "1")["3.EMGS | T"] == [[1.40], [2.10]]
    # collapse surfaces note/tags per trial-level mark
    collapsed = _events.collapse_markers(str(p), "1")
    annotated = next(r for r in collapsed if r["seq"] == [2.10])
    assert annotated["note"] == "good example" and annotated["tags"] == ["clean"]


def test_marker_signals_accepts_eventdata_dict(tmp_path):
    """A marker value may arrive as a datanavigator ``EventData`` mapping; the
    concatenation of ``default`` + ``added`` is the sequence list."""
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {"1": {"size": 1, "signals": {"3.EMGS | T": {"default": [[0.5]], "added": [[1.4]]}}}},
    )
    assert _events.read_marker_signals(str(p), "1")["3.EMGS | T"] == [[0.5], [1.4]]


# ---------------------------------------------------------------------------
# Typed accessors + collapse
# ---------------------------------------------------------------------------


def test_marker_types_excludes_noise(tmp_path):
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "noise": {"signals": {"3.EMGS | T": [[1.0, 2.0]]}},
            "2": {"size": 2, "signals": {"4.ACC | B": [[0.5, 0.9]]}},
            "1": {"size": 1, "signals": {"3.EMGS | T": [[1.4]]}},
        },
    )
    assert _events.marker_types(str(p)) == ["1", "2"]


def test_collapse_keeps_all_with_provenance(tmp_path):
    """Marks placed from two different signals both survive (visible disagreement)."""
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "1": {
                "size": 1,
                "signals": {
                    "3.EMGS | RForearm": [[2.10], [1.40]],
                    "4.ACC | RBicep": [[1.42]],
                },
            }
        },
    )
    recs = _events.collapse_markers(str(p), "1")
    # sorted by start time; provenance retained
    assert [r["seq"][0] for r in recs] == [1.40, 1.42, 2.10]
    by_addr = {r["seq"][0]: (r["address"], r["label"]) for r in recs}
    assert by_addr[1.40] == ("3.EMGS", "RForearm")
    assert by_addr[1.42] == ("4.ACC", "RBicep")


def test_collapse_dedupe_proximity(tmp_path):
    p = tmp_path / ("x" + _events.EVENTS_SUFFIX)
    _events.write_events(
        str(p),
        {
            "1": {
                "size": 1,
                "signals": {"3.EMGS | A": [[1.40]], "4.ACC | B": [[1.42]], "5.ACC | C": [[3.0]]},
            }
        },
    )
    # 1.40 and 1.42 collapse (within 0.05 s); 3.0 stays.
    recs = _events.collapse_markers(str(p), "1", dedupe=0.05)
    assert [r["seq"][0] for r in recs] == [1.40, 3.0]
    # keep-all default retains both near marks
    assert len(_events.collapse_markers(str(p), "1")) == 3


# ---------------------------------------------------------------------------
# Legacy bridge
# ---------------------------------------------------------------------------


def test_noise_signals_for_prefers_unified(tmp_path):
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    legacy_p = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    _events.write_events(str(events_p), {"noise": {"signals": {"3.EMGS | T": [[1.0, 2.0]]}}})
    write_noise_sidecar(str(legacy_p), {"9.FSR.C | F": [[5.0, 6.0]]})
    # unified file present -> legacy ignored
    sigs = _events.noise_signals_for(str(events_p), str(legacy_p))
    assert "3.EMGS | T" in sigs and "9.FSR.C | F" not in sigs


def test_noise_signals_for_falls_back_to_legacy(tmp_path):
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    legacy_p = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(legacy_p), {"9.FSR.C | F": [[5.0, 6.0]]})
    sigs = _events.noise_signals_for(str(events_p), str(legacy_p))
    assert sigs["9.FSR.C | F"] == {"windows": [[5.0, 6.0]]}


def test_migrate_noise_sidecar_folds_legacy(tmp_path):
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    legacy_p = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(legacy_p), {"9.FSR.C | F": [[5.0, 6.0]]})
    out = _events.migrate_noise_sidecar(str(legacy_p), str(events_p))
    assert out == str(events_p)
    assert _events.read_noise_signals(str(events_p))["9.FSR.C | F"] == {"windows": [[5.0, 6.0]]}


def test_migrate_noop_when_unified_noise_present(tmp_path):
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    legacy_p = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    _events.write_events(str(events_p), {"noise": {"signals": {"3.EMGS | T": [[1.0, 2.0]]}}})
    write_noise_sidecar(str(legacy_p), {"9.FSR.C | F": [[5.0, 6.0]]})
    assert _events.migrate_noise_sidecar(str(legacy_p), str(events_p)) is None
    # unified noise untouched
    assert "9.FSR.C | F" not in _events.read_noise_signals(str(events_p))


def test_migrate_absent_legacy_is_noop(tmp_path):
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    assert _events.migrate_noise_sidecar(str(tmp_path / "missing.delsys-noise"), str(events_p)) is None


# ---------------------------------------------------------------------------
# Masking off the unified file
# ---------------------------------------------------------------------------


def test_apply_events_noise_masks_like_sidecar(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    before = [np.asarray(s()).copy() for s in lf.signals]

    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    _events.write_events(str(events_p), {"noise": {"signals": {key: [[0.02, 0.05]]}}})
    touched = _events.apply_events_noise(lf, str(events_p))

    assert touched >= 1
    after = np.asarray(lf.signals[0]())
    assert not np.array_equal(before[0], after)
    assert not np.isnan(after).any()  # interpolated back


def test_apply_events_noise_empty_when_no_noise_type(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    events_p = tmp_path / ("Trial_5" + _events.EVENTS_SUFFIX)
    _events.write_events(str(events_p), {"1": {"size": 1, "signals": {"3.EMGS | T": [[1.4]]}}})
    assert _events.apply_events_noise(lf, str(events_p)) == 0
