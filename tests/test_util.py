"""Tests for ``delsys._util`` — modality dispatch and label-helper helpers."""

from delsys._util import (
    _canonical_label,
    _mod_to_attr,
    _parse_fsr_quattro_positions,
    _trim_location,
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
