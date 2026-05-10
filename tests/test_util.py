"""Tests for ``delsys._util`` — modality dispatch and label-helper helpers."""

import numpy as np
import pytest

from delsys._metadata import SensorInfo
from delsys._util import (
    _DRIFT_TOLERANCE,
    _canonical_label,
    _mod_to_attr,
    _normalize_signal_lengths,
    _parse_fsr_quattro_positions,
    _trim_location,
)
from delsys.signals import Signal


def _make_signal(
    n_samples: int,
    *,
    modality: str = "EMGS",
    subchannel: str = "A",
    sensor_number: int = 1,
    sr: float = 1920.0,
    t0: float = 0.0,
) -> Signal:
    """Construct a synthetic :class:`Signal` for tests that need raw signals
    without going through a CSV fixture."""
    sensor_info = SensorInfo(
        name=f"S{sensor_number}",
        modalities={modality},
        number=sensor_number,
        type_sensorlog=None,
        lrc=None,
        location=None,
    )
    return Signal(
        np.arange(n_samples, dtype=float),
        sr,
        t0=t0,
        meta={"sensor": sensor_info, "modality": modality, "subchannel": subchannel},
    )


# ---------------------------------------------------------------------------
# _mod_to_attr — every EMG variant collapses to ``'emg'``; link devices remap.
# ---------------------------------------------------------------------------


def test_mod_to_attr_emg_uppercase_variants():
    assert _mod_to_attr("EMG") == "emg"
    assert _mod_to_attr("EMGS") == "emg"
    assert _mod_to_attr("EMGD") == "emg"
    assert _mod_to_attr("EMGQ") == "emg"


def test_mod_to_attr_link_overrides():
    assert _mod_to_attr("VO2") == "vo2master"
    assert _mod_to_attr("HR") == "hrstrap"


def test_mod_to_attr_passthrough():
    assert _mod_to_attr("ACC") == "acc"
    assert _mod_to_attr("GYRO") == "gyro"


# ---------------------------------------------------------------------------
# _canonical_label — strip non-alphanumeric chars, preserving case.
# ---------------------------------------------------------------------------


def test_canonical_label_strips_spaces():
    assert _canonical_label("Outer Edge") == "OuterEdge"


def test_canonical_label_strips_punctuation():
    assert _canonical_label("Foo-Bar.Baz") == "FooBarBaz"


def test_canonical_label_alnum_unchanged():
    assert _canonical_label("LFoot") == "LFoot"
    assert _canonical_label("ch5") == "ch5"


def test_canonical_label_empty_string():
    assert _canonical_label("") == ""


# ---------------------------------------------------------------------------
# _trim_location — channelmap location → short label, with chN fallback.
# ---------------------------------------------------------------------------


def test_trim_location_none_falls_back_to_ch_number():
    assert _trim_location(None, 5) == "ch5"


def test_trim_location_strips_parenthetical():
    assert _trim_location("LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)", 5) == "LFoot"


def test_trim_location_keeps_first_token_only():
    assert _trim_location("LPinkyReach (LPalmaris Longus)", 5) == "LPinkyReach"


def test_trim_location_canonicalises_special_chars():
    # `_trim_location` runs the first whitespace-token through `_canonical_label`.
    assert _trim_location("L-Foo!", 5) == "LFoo"


def test_trim_location_already_clean():
    assert _trim_location("LBrachialis", 7) == "LBrachialis"


# ---------------------------------------------------------------------------
# _parse_fsr_quattro_positions — parse the parenthetical key-name pairs.
# ---------------------------------------------------------------------------


def test_parse_fsr_positions_numeric_keys():
    out = _parse_fsr_quattro_positions(
        "LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)", "FSR"
    )
    assert out == ["Heel", "OuterEdge", "Ball", "Toe"]


def test_parse_quattro_positions_letter_keys():
    out = _parse_fsr_quattro_positions(
        "LForearmExtensors (A-Index, B-Middle, C-Ring, D-Little)", "EMGQ"
    )
    assert out == ["Index", "Middle", "Ring", "Little"]


def test_parse_quattro_positions_canonicalises_names():
    # Spaces inside a position name get stripped by `_canonical_label`.
    out = _parse_fsr_quattro_positions(
        "RLowerLeg (A-Medial Gastroc, B-Lateral Gastroc, C-Tibialis Anterior, D-Peroneus)",
        "EMGQ",
    )
    assert out == ["MedialGastroc", "LateralGastroc", "TibialisAnterior", "Peroneus"]


def test_parse_fsr_positions_no_parenthetical_returns_none():
    assert _parse_fsr_quattro_positions("LBrachialis", "EMGS") is None


def test_parse_fsr_positions_garbage_parenthetical_returns_none():
    assert _parse_fsr_quattro_positions("Garbage (foo, bar)", "FSR") is None


def test_parse_fsr_positions_none_location_returns_none():
    assert _parse_fsr_quattro_positions(None, "FSR") is None


def test_parse_fsr_positions_missing_keys_returns_none():
    # Only 3 of the 4 expected keys present.
    assert (
        _parse_fsr_quattro_positions("LFoot (1-Heel, 2-OuterEdge, 3-Ball)", "FSR") is None
    )


def test_parse_fsr_positions_collision_returns_none():
    # Two channels parse to the same canonical name.
    assert (
        _parse_fsr_quattro_positions(
            "Bad (A-Same, B-Same, C-Other, D-Toe)", "EMGQ"
        )
        is None
    )


# ---------------------------------------------------------------------------
# _normalize_signal_lengths — tail-trim same-(modality, sr) signals to a
# common length, eliminating the post-resample drift that otherwise trips
# Sensor.__init__'s len(np.unique([len(s)])) == 1 assert.
# ---------------------------------------------------------------------------


def test_normalize_lengths_within_group_trims_to_min():
    signals = [
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=1),
        _make_signal(99, modality="EMGS", subchannel="A", sensor_number=2),
    ]
    out = _normalize_signal_lengths(signals)
    assert [len(s) for s in out] == [99, 99]
    # _t0 unchanged for both — tail-trim preserves the start time.
    assert all(s._t0 == 0.0 for s in out)


def test_normalize_lengths_no_drift_is_noop():
    signals = [
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=1),
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=2),
    ]
    out = _normalize_signal_lengths(signals)
    assert [len(s) for s in out] == [100, 100]
    # Values match exactly — no tail-trim happened.
    for orig, new in zip(signals, out):
        np.testing.assert_array_equal(orig(), new())


def test_normalize_lengths_across_modality_independent():
    # Drift in EMGS group must not trim the ACC group.
    signals = [
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=1, sr=1920.0),
        _make_signal(99, modality="EMGS", subchannel="A", sensor_number=2, sr=1920.0),
        _make_signal(50, modality="ACC", subchannel="X", sensor_number=3, sr=148.0),
        _make_signal(50, modality="ACC", subchannel="X", sensor_number=4, sr=148.0),
    ]
    out = _normalize_signal_lengths(signals)
    lens = [len(s) for s in out]
    assert lens == [99, 99, 50, 50]


def test_normalize_lengths_across_sr_independent():
    # Same modality, different sr => independent groups.
    signals = [
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=1, sr=1920.0),
        _make_signal(99, modality="EMGS", subchannel="A", sensor_number=2, sr=1920.0),
        _make_signal(60, modality="EMGS", subchannel="A", sensor_number=3, sr=2000.0),
    ]
    out = _normalize_signal_lengths(signals)
    lens = [len(s) for s in out]
    assert lens == [99, 99, 60]


def test_normalize_lengths_warns_on_excessive_drift():
    n_long = 100
    n_short = 100 - (_DRIFT_TOLERANCE + 1)
    signals = [
        _make_signal(n_long, modality="EMGS", subchannel="A", sensor_number=1),
        _make_signal(n_short, modality="EMGS", subchannel="A", sensor_number=2),
    ]
    with pytest.warns(UserWarning):
        _normalize_signal_lengths(signals)


def test_normalize_lengths_preserves_meta_and_history():
    signals = [
        _make_signal(100, modality="EMGS", subchannel="A", sensor_number=1),
        _make_signal(99, modality="EMGS", subchannel="A", sensor_number=2),
    ]
    # Stamp some custom history on the longer signal so we can confirm it
    # propagates through the tail-trim clone.
    signals[0]._history.append(("custom-step", {"k": "v"}))
    out = _normalize_signal_lengths(signals)
    assert out[0].meta["modality"] == "EMGS"
    assert out[0].meta["subchannel"] == "A"
    assert out[0].sensor.number == 1
    assert ("custom-step", {"k": "v"}) in out[0]._history
