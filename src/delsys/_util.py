"""Small utility helpers used across the delsys package."""

import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Set, Type

import numpy as np
import pysampled

from delsys._constants import _SUBCHANNEL_KEYS

if TYPE_CHECKING:
    from delsys.signals import Signal

#: Sample-count span (max - min within a same-(modality, sr) group) considered
#: "normal" floating-point drift from the per-channel resample step. Anything
#: above this is escalated to a :class:`UserWarning` because it usually means
#: a parser quirk rather than a quantization rounding.
_DRIFT_TOLERANCE: int = 4

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
        # Read modality defensively — signals with empty meta (very-old
        # pickles, or anything that bypassed the per-format parser) still
        # need to participate in length normalization or the per-Sensor
        # stack assert downstream will fire on them. They land in their
        # own ``(None, sr)`` group, which is fine.
        modality = sig.meta.get("modality") if sig.meta else None
        groups[(modality, sig.sr)].append(idx)

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


def _sensors_aligned_with_names(part: pysampled.Data) -> List:
    """Expand a part's sensor metadata to match its ``signal_names``.

    Per-Sensor bundles carry ``meta['sensor']`` (singular). Aggregate
    bundles carry ``meta['sensors']`` (plural). This helper returns a
    list of length ``len(part.signal_names)`` so the caller can
    concatenate aligned per-name sensor records into the new aggregate.
    """
    if isinstance(part.meta, dict) and "sensors" in part.meta:
        return list(part.meta["sensors"])
    sensor = part.meta.get("sensor") if isinstance(part.meta, dict) else None
    return [sensor] * len(part.signal_names)


def _aggregate_bundles(
    parts: List[pysampled.Data],
    bundle_cls: Optional[Type[pysampled.Data]] = None,
) -> Optional[pysampled.Data]:
    """Stack per-Sensor bundles along the signal axis into one Data.

    Returns ``None`` when ``parts`` is empty. With a single part, returns
    a label-preserved clone (so callers always get the canonical
    aggregate type — and its ``meta['sensors']`` plural convention —
    even when only one sensor has the modality).

    All parts must agree on ``signal_coords`` (channels-of-the-same-
    modality must mean the same coords) and ``axis``. ``t0`` agreement
    is required to within ``1 / target_sr``.

    Multi-rate input is downsampled to the lowest sampling rate present
    in ``parts``, with a :class:`UserWarning` listing the higher-rate
    sensors. Lowest-SR (rather than highest-SR) was chosen because
    upsampling can fabricate phantom high-frequency content, while a
    cheap downsample is information-lossy in a predictable way.

    Args:
        parts: Per-Sensor bundles to aggregate. Already in
            :attr:`Log.sensors` order.
        bundle_cls: Class to construct the aggregate as. Defaults to
            ``type(parts[0])`` so e.g. an aggregate of :class:`IMU`
            bundles is itself an :class:`IMU`.

    Returns:
        A single aggregate bundle, or ``None`` when ``parts`` is empty.

    Raises:
        ValueError: On ``signal_coords``, ``axis``, or ``t0`` mismatch.

    .. todo:: Migrate to ``pysampled.Data.merge_along_signal_name`` once
        pysampled ships those classmethods (deferred from pysampled
        1.2.0).
    """
    if not parts:
        return None

    if bundle_cls is None:
        bundle_cls = type(parts[0])

    axes = {p.axis for p in parts}
    if len(axes) > 1:
        raise ValueError(f"Cannot aggregate parts with mixed axis values: {axes}")
    axis = parts[0].axis

    coord_signatures = {tuple(p.signal_coords) for p in parts}
    if len(coord_signatures) > 1:
        raise ValueError(
            f"Cannot aggregate parts with mismatched signal_coords: {coord_signatures}"
        )
    signal_coords = list(parts[0].signal_coords)

    # Multi-rate: downsample higher-rate parts to the lowest sr present.
    srs = sorted({p.sr for p in parts})
    if len(srs) > 1:
        target_sr = srs[0]
        higher = [p for p in parts if p.sr != target_sr]
        higher_locations = [
            getattr(p.meta.get("sensor", None), "location", None) or "?"
            for p in higher
        ]
        warnings.warn(
            f"Multi-rate aggregate: resampling sensors {higher_locations} from "
            f"{sorted({p.sr for p in higher})} -> {target_sr} Hz",
            UserWarning,
            stacklevel=2,
        )
        parts = [p if p.sr == target_sr else p.resample(target_sr) for p in parts]
    else:
        target_sr = srs[0]

    # Validate t0 agreement (post-resample, since resample may shift t0).
    tol = 1.0 / target_sr
    t0 = parts[0]._t0
    for p in parts[1:]:
        if abs(p._t0 - t0) > tol:
            raise ValueError(
                f"Cannot aggregate parts with mismatched t0: {t0} vs {p._t0} "
                f"(tolerance {tol})"
            )

    # Defense in depth: post-resample lengths should already match given
    # parse-time normalization, but if they don't, tail-trim every part to
    # the shortest length so np.hstack can run.
    lens = [p().shape[0] if p().ndim > 1 else p().shape[0] for p in parts]
    min_len = min(lens)
    if max(lens) != min_len:
        parts = [
            p if p().shape[0] == min_len else p._clone(p()[:min_len]) for p in parts
        ]

    # Stack name-major. Each bundle's column layout is already
    # ``itertools.product(signal_names, signal_coords)`` per pysampled's
    # invariant, so concatenating preserves that as long as names don't
    # collide across parts (which they won't — each comes from a distinct
    # sensor location).
    arrays = [
        p() if p().ndim == 2 else np.atleast_2d(p()).T for p in parts
    ]
    stacked = np.hstack(arrays)

    signal_names: List[str] = []
    sensors_meta: List = []
    for p in parts:
        signal_names.extend(p.signal_names)
        sensors_meta.extend(_sensors_aligned_with_names(p))

    meta = {"sensors": sensors_meta}
    return bundle_cls(
        stacked,
        sr=target_sr,
        axis=axis,
        t0=t0,
        meta=meta,
        signal_names=signal_names,
        signal_coords=signal_coords,
    )
