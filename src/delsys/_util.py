"""Small utility helpers used across the delsys package."""

import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from delsys.signals import Signal

#: Sample-count span (max - min within a same-(modality, sr) group) considered
#: "normal" floating-point drift from the per-channel resample step. Anything
#: above this is escalated to a :class:`UserWarning` because it usually means
#: a parser quirk rather than a quantization rounding.
_DRIFT_TOLERANCE: int = 4

#: Sub-channel keys we expect inside FSR / Quattro location parentheticals.
#: ``"FSR"`` channelmap entries use numeric keys (``1``/``2``/``3``/``4``);
#: ``"EMGQ"`` (Quattro) uses letters (``A``/``B``/``C``/``D``).
_SUBCHANNEL_KEYS = {
    "FSR": ("1", "2", "3", "4"),
    "EMGQ": ("A", "B", "C", "D"),
}

#: Match a single ``key-name`` token inside the channelmap parenthetical,
#: tolerating surrounding whitespace. ``name`` is captured greedily up to
#: the next comma or closing paren.
_KEY_NAME_PAIR_RE = re.compile(r"\s*([A-Za-z0-9]+)\s*-\s*([^,)]+?)\s*(?=,|\))")


def _mod_to_attr(name: str) -> str:
    """Convert a modality tag to the :class:`Sensor` attribute name that holds it.

    Maps every EMG variant (``'EMG'``, ``'EMGS'``, ``'EMGD'``, ``'EMGQ'``) to
    ``'emg'`` so EMG-bundle access is uniform regardless of sensor flavor.
    Link devices have longer attribute names than their modality tags
    (``'VO2'`` → ``'vo2master'``, ``'HR'`` → ``'hrstrap'``); everything else
    is just lower-cased.

    Args:
        name: Modality tag — case-insensitive.

    Returns:
        The lower-case attribute name used on :class:`Sensor` instances.
    """
    if name.upper().startswith("EMG"):
        return "emg"
    overrides = {"vo2": "vo2master", "hr": "hrstrap"}
    return overrides.get(name.lower(), name.lower())


def _modset_to_strlist(modset: Set[str]) -> List[str]:
    """Flatten a set of modality tags into the strings legacy lookup compares against.

    EMG variants collapse to ``'EMG'``. Each entry is included in both its
    original case and lower-cased form so substring matches are
    case-insensitive.

    Args:
        modset: Set of modality tags (e.g. ``{'EMGS', 'ACC', 'GYRO'}``).

    Returns:
        Flattened list including the canonical (case-folded EMG) labels and
        their lower-case equivalents.
    """
    modality_strings = ["EMG" if x.startswith("EMG") else x for x in modset]
    return modality_strings + [x.lower() for x in modality_strings]


def _canonical_label(s: str) -> str:
    """Strip every non-alphanumeric character, preserving case.

    Used to turn a channelmap location fragment (or position name) into a
    safe ``signal_names`` token: alphanumerics only, no spaces, no
    punctuation, no underscores. Underscores are reserved as the
    parent/sub-channel separator at the call site.

    Args:
        s: Arbitrary string.

    Returns:
        ``s`` with every character not in ``[A-Za-z0-9]`` removed.
    """
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def _trim_location(location: Optional[str], sensor_number: int) -> str:
    """Reduce a channelmap ``location`` to a short, sanitized label.

    Takes the first whitespace-delimited token of ``location`` and runs it
    through :func:`_canonical_label`. When ``location`` is ``None`` (i.e. no
    channelmap was supplied), falls back to ``f"ch{sensor_number}"`` so each
    sensor still has a unique, readable identifier.

    Args:
        location: ``SensorLog.location`` from the channelmap, or ``None``.
        sensor_number: Delsys sensor number — used for the no-channelmap
            fallback only.

    Returns:
        Alphanumeric-only short label (e.g. ``"LFoot"``, ``"ch5"``).
    """
    if location is None:
        return f"ch{sensor_number}"
    first_token = location.split(" ", 1)[0]
    return _canonical_label(first_token)


def _parse_fsr_quattro_positions(
    location: Optional[str], modality: str
) -> Optional[List[str]]:
    """Parse the FSR/Quattro channelmap parenthetical into per-channel names.

    Looks for the ``(<key>-<name>, <key>-<name>, …)`` block at the end of
    ``location``, where the keys are ``1``/``2``/``3``/``4`` for FSR or
    ``A``/``B``/``C``/``D`` for Quattro. Each name is run through
    :func:`_canonical_label` (stripping spaces and punctuation), and the
    result is returned in canonical sub-channel order (numeric or A→D).

    Returns ``None`` — signalling "fall back to ``A``/``B``/``C``/``D``" — on
    any parsing failure: no parenthetical, malformed entries, missing or
    duplicated keys, or post-canonicalisation name collisions.

    Args:
        location: ``SensorLog.location`` from the channelmap, or ``None``.
        modality: Either ``"FSR"`` (numeric keys) or ``"EMGQ"`` (letter
            keys). Any other modality returns ``None``.

    Returns:
        Canonicalised position names in sub-channel order, or ``None`` if
        parsing failed.
    """
    if location is None:
        return None
    if modality not in _SUBCHANNEL_KEYS:
        return None
    expected_keys = _SUBCHANNEL_KEYS[modality]

    open_idx = location.find("(")
    close_idx = location.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return None
    inner = location[open_idx + 1 : close_idx]

    pairs = _KEY_NAME_PAIR_RE.findall(inner + ")")
    if len(pairs) != len(expected_keys):
        return None

    by_key = {}
    for key, name in pairs:
        canon = _canonical_label(name)
        if not canon:
            return None
        by_key[key] = canon

    if set(by_key.keys()) != set(expected_keys):
        return None

    ordered = [by_key[k] for k in expected_keys]
    if len(set(ordered)) != len(ordered):
        return None

    return ordered


def _normalize_signal_lengths(signals: List["Signal"]) -> List["Signal"]:
    """Tail-trim same-(modality, sr) signals to a common length.

    Channels that nominally share a sampling rate can end up with sample
    counts that differ by 1-2 because each parser computes per-channel
    lengths from a slightly different rounding of ``sr * duration``. The
    drift trips :class:`Sensor`'s same-modality ``len`` assert when the
    user actually tries to stack. This helper closes the gap by trimming
    every signal in a group down to the group's minimum length, leaving
    ``_t0`` untouched (the drift is at the right edge anyway).

    Tail-trim, not pad: padding with zeros / NaN / extrapolated values
    would inject artifacts into downstream filters; a 1-2 sample tail
    trim at 1920 Hz is well below any analysis tolerance.

    A drift larger than :data:`_DRIFT_TOLERANCE` triggers a
    :class:`UserWarning` — that's the "something weird is going on"
    tripwire, not the normal case.

    Args:
        signals: Per-channel :class:`Signal` objects produced by the
            per-format parser.

    Returns:
        New list (same order as input). Signals already at ``min_len`` are
        passed through by identity; longer ones are replaced with a
        ``_clone(s()[:min_len])`` that preserves ``_t0``, ``meta``, and
        ``_history``.
    """
    groups: "defaultdict[tuple, List[int]]" = defaultdict(list)
    for idx, sig in enumerate(signals):
        groups[(sig.modality, sig.sr)].append(idx)

    out: List["Signal"] = list(signals)
    for (modality, sr), idxs in groups.items():
        lens = [len(out[i]) for i in idxs]
        min_len = min(lens)
        max_len = max(lens)
        if min_len <= 0:
            # Defensive: a zero-length channel will already trip the
            # downstream Sensor assert with a clearer signal.
            continue
        if max_len == min_len:
            continue
        if max_len - min_len > _DRIFT_TOLERANCE:
            warnings.warn(
                f"Same-rate length drift exceeds tolerance for modality="
                f"{modality!r} sr={sr}: span={max_len - min_len} samples "
                f"(min={min_len}, max={max_len}). Trimming to min.",
                UserWarning,
                stacklevel=2,
            )
        for i in idxs:
            sig = out[i]
            if len(sig) > min_len:
                out[i] = sig._clone(sig()[:min_len])
    return out
