"""Per-signal noise sidecar (``<stem>.delsys-noise``): key grammar + masking.

Companion to ``test_clean.py`` (which owns the batch + manifest + datanavigator
Event consumption). Here: the ``"<sensor>.<modality>[.<coord>] | <label>"`` key
grammar (parse / format / resolve), the sidecar JSON round-trip, and per-signal
window / dead-span masking. ``discover170.csv`` is the end-to-end fixture (18
sensors, mixed modalities incl. multi-axis IMU).
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import delsys  # noqa: E402
from delsys._noise import (  # noqa: E402
    SIDECAR_SCHEMA,
    SIDECAR_SUFFIX,
    ParsedKey,
    _normalize_signal_value,
    apply_noise_sidecar,
    format_key,
    format_signal_key,
    parse_key,
    read_noise_sidecar,
    resolve_key,
    sidecar_path_for,
    write_noise_sidecar,
)

FIXTURE = "discover170.csv"


def _load(fixtures_dir, tmp_path):
    """Load the fixture as a Log (named Trial_5 so sidecar paths read naturally)."""
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return delsys.Log(str(csv))


# ---------------------------------------------------------------------------
# Key grammar
# ---------------------------------------------------------------------------


def test_parse_key_variants():
    assert parse_key("3.EMGS.A | Tricep_L") == ParsedKey(3, "EMGS", "A", "Tricep_L")
    assert parse_key("4.ACC | Bicep_R") == ParsedKey(4, "ACC", None, "Bicep_R")
    assert parse_key("7.EMGQ.A | Forearm").coord == "A"
    # Label is optional and informational.
    assert parse_key("9.FSR.C").label == ""
    assert parse_key("9.FSR.C").coord == "C"


def test_format_key_roundtrip():
    assert format_key(4, "ACC", "X", "Bicep_R") == "4.ACC.X | Bicep_R"
    assert format_key(4, "ACC") == "4.ACC"
    assert format_key(4, "ACC", None, "Bicep_R") == "4.ACC | Bicep_R"
    # parse(format(...)) round-trips the structural address.
    for sensor, mod, coord in [(3, "EMGS", "A"), (4, "ACC", None), (9, "FSR", "C")]:
        pk = parse_key(format_key(sensor, mod, coord, "label"))
        assert (pk.sensor, pk.modality, pk.coord) == (sensor, mod, coord)


def test_parse_key_rejects_bad_address():
    with pytest.raises(ValueError):
        parse_key("garbage")  # no modality component
    with pytest.raises(ValueError):
        parse_key("3 | only a label")


# ---------------------------------------------------------------------------
# Resolution against a real Log
# ---------------------------------------------------------------------------


def test_resolve_single_subchannel(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    sig = lf.signals[0]
    idxs = resolve_key(lf, format_signal_key(sig))
    assert 0 in idxs
    # Every resolved index shares the full address (coord pins to one channel).
    for j in idxs:
        s = lf.signals[j]
        assert (s.sensor.number, s.modality, s.subchannel) == (
            sig.sensor.number,
            sig.modality,
            sig.subchannel,
        )


def test_resolve_coordless_fans_out(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    counts = Counter(
        (s.sensor.number, s.modality) for s in lf.signals if s.sensor is not None
    )
    multi = [k for k, c in counts.items() if c > 1]
    assert multi, "discover170 should carry a multi-sub-channel modality (e.g. ACC)"
    sn, mod = multi[0]
    idxs = resolve_key(lf, f"{sn}.{mod} | whole modality")
    assert len(idxs) == counts[(sn, mod)]
    assert all(
        lf.signals[j].sensor.number == sn and lf.signals[j].modality == mod
        for j in idxs
    )


def test_resolve_unknown_key_is_empty(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    assert resolve_key(lf, "999.EMGS.A | nope") == []


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def test_sidecar_io_roundtrip_and_shorthand(tmp_path):
    p = tmp_path / ("x" + SIDECAR_SUFFIX)
    write_noise_sidecar(
        str(p),
        {
            "3.EMGS | T": [[1.0, 2.0]],  # bare-list shorthand -> windows
            "9.FSR.C | F": {"dead": True},  # whole-extent dead sugar
            "4.ACC | B": {"windows": [[0.5, 0.7]], "dead": [[3.0, None]]},
        },
    )
    doc = read_noise_sidecar(str(p))
    assert doc["schema"] == SIDECAR_SCHEMA
    sigs = doc["signals"]
    assert sigs["3.EMGS | T"] == {"windows": [[1.0, 2.0]]}
    assert sigs["9.FSR.C | F"] == {"dead": [[None, None]]}
    assert sigs["4.ACC | B"] == {"windows": [[0.5, 0.7]], "dead": [[3.0, None]]}


def test_normalize_signal_value_forms():
    assert _normalize_signal_value([[1.0, 2.0]]) == ([(1.0, 2.0)], [])
    assert _normalize_signal_value({"dead": True}) == ([], [(None, None)])
    w, d = _normalize_signal_value({"windows": [[0.5, 0.7]], "dead": [[3.0, None]]})
    assert w == [(0.5, 0.7)] and d == [(3.0, None)]


# ---------------------------------------------------------------------------
# Per-signal masking
# ---------------------------------------------------------------------------


def test_apply_sidecar_scopes_to_resolved_signal(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    resolved = set(resolve_key(lf, key))
    before = [np.asarray(s()).copy() for s in lf.signals]

    sidecar = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(sidecar), {key: [[0.02, 0.05]]})
    touched = apply_noise_sidecar(lf, str(sidecar))

    assert touched == len(resolved)
    after = [np.asarray(s()) for s in lf.signals]
    for i in range(len(lf.signals)):
        if i in resolved:
            assert not np.array_equal(before[i], after[i])
            assert not np.isnan(after[i]).any()  # interpolated back
        else:
            assert np.array_equal(before[i], after[i])  # untouched


def test_apply_sidecar_dead_whole_extent_zero_fills(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    sidecar = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(sidecar), {key: {"dead": True}})

    assert apply_noise_sidecar(lf, str(sidecar)) == 1
    assert np.all(np.asarray(lf.signals[0]()) == 0.0)


def test_apply_sidecar_dead_from_t_zero_fills_tail(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    sig0 = lf.signals[0]
    arr0 = np.asarray(sig0())
    n = arr0.shape[0]
    sr = float(sig0.sr)
    t0 = float(getattr(sig0, "_t0", 0.0) or 0.0)
    t_mid = t0 + (n // 2) / sr  # dead from the midpoint onward

    key = format_signal_key(sig0)
    sidecar = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(sidecar), {key: {"dead": [[t_mid, None]]}})
    apply_noise_sidecar(lf, str(sidecar))

    out = np.asarray(lf.signals[0]())
    i_mid = int(np.floor((t_mid - t0) * sr))
    assert np.all(out[i_mid:] == 0.0)  # tail zeroed
    assert not np.all(out[:i_mid] == 0.0)  # head left intact


def test_apply_sidecar_unknown_policy_raises(fixtures_dir, tmp_path):
    lf = _load(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    sidecar = tmp_path / ("Trial_5" + SIDECAR_SUFFIX)
    write_noise_sidecar(str(sidecar), {key: [[0.02, 0.05]]})
    with pytest.raises(ValueError):
        apply_noise_sidecar(lf, str(sidecar), policy="zero")


def test_sidecar_path_for():
    assert sidecar_path_for("/a/b/Trial_5.h5") == "/a/b/Trial_5" + SIDECAR_SUFFIX
