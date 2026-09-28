"""Consume human-authored noise-window Events when cleaning a :class:`delsys.Log`.

The two kinds of EMG cleaning have separate owners (see ``CHANGELOG.md`` and the
``tutorials/workflow.md`` walkthrough): algorithmic ECG / motion suppression
lives in :mod:`delsys.cleaning`, while *human* noise-window marking is authored
upstream in datanavigator's ``SignalBrowser`` and merely **consumed** here. This
module is the consumption side.

It reads datanavigator's on-disk Event format directly as **plain JSON** — no
``datanavigator`` import — which sidesteps the deferred delsys↔datanavigator
dependency decision. The Event file is a 2-element list ``[metadata, data]``
where ``data`` maps a stringified trial-id tuple (e.g. ``"(2, 14, 17)"``) to a
record::

    {"default": [], "added": [[t0, t1], ...], "removed": [], "tags": [], ...}

The effective window set for a trial is ``default + added`` minus any ``removed``
interval, in seconds on the Log's clock.

The default policy is **NaN + interpolate**, applied modality-agnostically: a
noise window is a wall-clock span (a cable bump, a dropped-sample burst), not a
per-channel event, so every modality (EMG, ACC, ...) gets the same treatment.
This also covers the wobble accelerometer dropped-sample case.

This is the v1 surface — enough to wire the hook into :func:`delsys.clean`.
Per-modality / per-sensor scoping of windows and alternative fill policies are
follow-ups (see ``TODO.md``).
"""

import json
import os
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pysampled

from delsys._util import _trim_location

#: Trial-id key as accepted by the public helpers.
TrialId = Union[str, Sequence[int]]

#: Composite suffix of the per-log, per-signal noise sidecar. Composite (not
#: ``.json``) so it isn't swept by portfolio ``*.json`` tooling — same
#: convention as datanavigator's ``.dnav-toc`` (see the no-``.json``-sidecar
#: rule in CONVENTIONS / memory).
SIDECAR_SUFFIX = ".delsys-noise"

#: Bump when the sidecar layout changes incompatibly.
SIDECAR_SCHEMA = 1

#: Parsed form of a sidecar signal key
#: ``"<sensor>.<modality>[.<coord>] | <label>"``. ``coord`` is ``None`` for a
#: whole-modality (all sub-channels) address; ``label`` is informational and
#: ignored on resolve (a relabel never breaks lookup).
ParsedKey = namedtuple("ParsedKey", ["sensor", "modality", "coord", "label"])


def _normalize_key(trial_id: TrialId) -> str:
    """Coerce a trial id into the Event's stringified-tuple key form.

    ``(2, 14, 17)`` -> ``"(2, 14, 17)"`` (matching Python's ``str(tuple)``, which
    is how datanavigator serializes tuple keys); a string passes through verbatim.
    """
    if isinstance(trial_id, str):
        return trial_id
    if isinstance(trial_id, (tuple, list)):
        return str(tuple(trial_id))
    return str(trial_id)


def _as_pairs(seq) -> List[Tuple[float, float]]:
    """Coerce a list of ``[t0, t1]`` into validated ``(t0, t1)`` float tuples."""
    out: List[Tuple[float, float]] = []
    for iv in seq or []:
        if iv is None or len(iv) < 2:
            continue
        a, b = float(iv[0]), float(iv[1])
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out


def read_noise_intervals(path: str, trial_id: TrialId) -> List[Tuple[float, float]]:
    """Read the noise time-windows for one trial from a datanavigator Event JSON.

    Args:
        path: Path to the Event JSON authored in datanavigator's ``SignalBrowser``.
        trial_id: Trial key. A tuple/list is stringified to match the on-disk
            ``"(2, 14, 17)"`` key form; a string is used verbatim.

    Returns:
        Sorted, de-duplicated ``[(t_start, t_end), ...]`` in seconds. Empty when
        the trial has no marked noise (or no entry in the file).
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # datanavigator stores ``[metadata, data]``; tolerate a bare data dict too.
    if isinstance(doc, list):
        data = doc[1] if len(doc) > 1 and isinstance(doc[1], dict) else {}
    elif isinstance(doc, dict):
        data = doc
    else:
        data = {}

    entry = data.get(_normalize_key(trial_id))
    if not entry:
        return []

    intervals = _as_pairs(entry.get("default")) + _as_pairs(entry.get("added"))
    removed = set(_as_pairs(entry.get("removed")))
    return sorted({iv for iv in intervals if iv not in removed})


def _resolve_span(a, b, n: int, sr: float, t0: float) -> Tuple[int, int]:
    """Resolve a ``[start, end]`` span (seconds, ``None`` = open) to ``[i0, i1)``.

    ``None`` start clamps to the signal's first sample; ``None`` end to one past
    the last — so ``[T, None]`` means "from ``T`` to the end" and ``[None, None]``
    the whole extent.
    """
    i0 = 0 if a is None else max(0, int(np.floor((float(a) - t0) * sr)))
    i1 = n if b is None else min(n, int(np.ceil((float(b) - t0) * sr)) + 1)
    return i0, i1


def _mask_signals(lf, spec_by_index: Dict[int, dict], *, policy: str = "nan_interp") -> int:
    """Apply per-signal noise/dead spans to ``lf.signals``, in place.

    The shared core behind :func:`apply_noise_mask` (flat windows over a
    modality-filtered set) and :func:`apply_noise_sidecar` (per-resolved-address
    spans). For each touched signal:

    - ``"windows"`` spans (transient noise) are set to ``NaN`` then refilled via
      :meth:`pysampled.Data.interpnan` (policy ``"nan_interp"``).
    - ``"dead"`` spans (no usable signal) are **zero-filled**, applied *after*
      the interpolation so a dead span wins any overlap with a noise window.

    Each affected :class:`delsys.signals.Signal` is replaced with a ``_clone`` of
    the filled array (preserving ``sr`` / ``t0`` / ``meta``); the per-sensor
    bundles are rebuilt once at the end so ``lf.emg`` / ``lf.acc`` / ... reflect
    the mask on next access.

    Args:
        lf: A :class:`delsys.Log` (mutated in place — pass one you own).
        spec_by_index: ``{signal_index: {"windows": [...], "dead": [...]}}``.
            Span endpoints are seconds on the Log's clock; ``None`` is open.
        policy: Masking policy. Only ``"nan_interp"`` is supported in v1.

    Returns:
        The number of signals touched.
    """
    if policy != "nan_interp":
        raise ValueError(f"unknown noise policy {policy!r}; only 'nan_interp' in v1.")

    touched = 0
    for i, spec in spec_by_index.items():
        windows = spec.get("windows") or []
        dead = spec.get("dead") or []
        if not windows and not dead:
            continue

        sig = lf.signals[i]
        arr = np.asarray(sig(), dtype=float).copy()
        n = arr.shape[0]
        sr = float(sig.sr)
        t0 = float(getattr(sig, "_t0", 0.0) or 0.0)

        hit = False
        for a, b in windows:
            i0, i1 = _resolve_span(a, b, n, sr, t0)
            if i1 > i0:
                arr[i0:i1] = np.nan
                hit = True
        if hit:
            arr = np.asarray(pysampled.Data(arr, sr=sr).interpnan()())

        for a, b in dead:
            i0, i1 = _resolve_span(a, b, n, sr, t0)
            if i1 > i0:
                arr[i0:i1] = 0.0
                hit = True

        if not hit:
            continue
        lf.signals[i] = sig._clone(arr)
        touched += 1

    if touched:
        _rebuild_sensors(lf)
    return touched


def apply_noise_mask(
    lf,
    intervals: Sequence[Tuple[float, float]],
    *,
    policy: str = "nan_interp",
    modalities: Optional[Sequence[str]] = None,
) -> int:
    """Blank (and refill) flat noise windows across a Log's signals, in place.

    The modality-agnostic path: every signal (or every signal of a whitelisted
    modality) gets the same ``intervals`` masked with the ``nan_interp`` policy.
    For per-signal-addressed windows (and dead channels), see
    :func:`apply_noise_sidecar`. Both share the :func:`_mask_signals` core.

    Args:
        lf: A :class:`delsys.Log` (its ``signals`` / ``sensors`` are replaced in
            place — pass a Log you own, not one shared with a caller).
        intervals: ``[(t_start, t_end), ...]`` in seconds, on the Log's clock.
        policy: Masking policy. Only ``"nan_interp"`` is supported in v1.
        modalities: Optional modality whitelist (e.g. ``["EMGS", "ACC"]``);
            ``None`` (default) masks every signal regardless of modality.

    Returns:
        The number of signals touched.
    """
    if policy != "nan_interp":
        raise ValueError(f"unknown noise policy {policy!r}; only 'nan_interp' in v1.")

    windows = [(float(a), float(b)) for a, b in intervals if float(b) > float(a)]
    if not windows:
        return 0

    spec_by_index: Dict[int, dict] = {}
    for i, sig in enumerate(lf.signals):
        meta = getattr(sig, "meta", None) or {}
        if modalities is not None and meta.get("modality") not in modalities:
            continue
        spec_by_index[i] = {"windows": windows, "dead": []}
    return _mask_signals(lf, spec_by_index, policy=policy)


def apply_noise_events(
    lf,
    path: str,
    trial_id: TrialId,
    *,
    policy: str = "nan_interp",
    modalities: Optional[Sequence[str]] = None,
) -> int:
    """Read a noise Event JSON and mask the windows for ``trial_id`` on ``lf``.

    Convenience wrapper over :func:`read_noise_intervals` +
    :func:`apply_noise_mask`. Returns the number of signals touched (0 when the
    trial has no marked noise).
    """
    intervals = read_noise_intervals(path, trial_id)
    if not intervals:
        return 0
    return apply_noise_mask(lf, intervals, policy=policy, modalities=modalities)


# ---------------------------------------------------------------------------
# Per-signal noise sidecar (``<stem>.delsys-noise``) — key grammar + I/O
# ---------------------------------------------------------------------------
#
# A delsys-file-centric, per-signal noise record that travels next to one
# ``Trial_N.h5``. Unlike the datanavigator Event path above (trial-id-keyed,
# flat intervals), this is keyed by a structural *signal address* so windows
# (and dead spans) can be scoped to individual sensors / modalities / axes.


def sidecar_path_for(target: Union[str, "os.PathLike"]) -> str:
    """Sibling ``<stem>.delsys-noise`` path for a checkpoint / file ``target``."""
    return os.path.splitext(str(target))[0] + SIDECAR_SUFFIX


def parse_key(key: str) -> ParsedKey:
    """Parse a sidecar key ``"<sensor>.<modality>[.<coord>] | <label>"``.

    The part left of ``" | "`` is the authoritative structural address; the
    label on the right is informational and returned but ignored by
    :func:`resolve_key`. ``coord`` is ``None`` when the address omits it (a
    whole-modality / all-sub-channels address).
    """
    addr, _, label = key.partition(" | ")
    parts = addr.strip().split(".")
    if len(parts) < 2:
        raise ValueError(
            f"bad noise key {key!r}: address must be "
            f"'<sensor>.<modality>[.<coord>]' (got {addr!r})."
        )
    sensor = int(parts[0])
    modality = parts[1]
    coord = parts[2] if len(parts) > 2 else None
    return ParsedKey(sensor=sensor, modality=modality, coord=coord, label=label.strip())


def format_key(
    sensor: int, modality: str, coord: Optional[str] = None, label: str = ""
) -> str:
    """Build a sidecar key from address parts (inverse of :func:`parse_key`)."""
    addr = f"{sensor}.{modality}" + (f".{coord}" if coord else "")
    return f"{addr} | {label}" if label else addr


def _signal_label(sig) -> str:
    """Full per-channel display name for a Signal, matching the modality
    bundle's ``signal_names`` (side + body location + FSR/Quattro position).

    Reuses :meth:`delsys.sensor.Sensor._make_bundle_labels` so the label is
    byte-identical to what ``lf.<modality>.signal_names`` shows: single-name
    modalities (EMGS/EKG/ACC/GYRO) resolve to the trimmed location; multi-name
    modalities (EMGD/EMGQ/FSR) resolve to the per-sub-channel name (e.g.
    ``"LFoot_Ball"`` for FSR sub-channel ``C``).
    """
    from delsys._constants import SUBCHANNEL_MAP
    from delsys.sensor import Sensor

    keys = SUBCHANNEL_MAP.get(sig.modality, (sig.subchannel,))
    names, _ = Sensor._make_bundle_labels(sig.modality, sig.sensor, len(keys))
    if len(names) == len(keys) and sig.subchannel in keys:
        return names[keys.index(sig.subchannel)]
    return names[0] if names else ""


def format_signal_key(sig, *, include_coord: bool = True) -> str:
    """Build the sidecar key addressing a single :class:`delsys.signals.Signal`.

    With ``include_coord=True`` (default) the ``<label>`` is the full
    per-channel name from :func:`_signal_label` (matching the modality bundle's
    ``signal_names``). ``include_coord=False`` produces the whole-modality
    address (all sub-channels of the sensor+modality), labelled with just the
    trimmed body location via :func:`delsys._util._trim_location`.

    A per-channel :class:`Signal`'s own ``signal_names`` is a pysampled
    placeholder (``"s0"``), so it is deliberately *not* used.
    """
    sensor = sig.sensor
    if include_coord:
        label = _signal_label(sig)
    else:
        label = _trim_location(getattr(sensor, "location", None), sensor.number)
    return format_key(
        sensor.number,
        sig.modality,
        sig.subchannel if include_coord else None,
        label,
    )


def key_address(key: str) -> str:
    """Strip a key to its structural address ``"<sensor>.<modality>[.<coord>]"``.

    Drops the informational ``" | <label>"`` so two keys that differ only in
    label (e.g. after a relabel, or an older code version's placeholder) collapse
    to one. This is the stable identity to index annotations by — the label must
    never be load-bearing for lookup (only :func:`resolve_key` against a Log is
    authoritative).
    """
    pk = parse_key(key)
    return format_key(pk.sensor, pk.modality, pk.coord)


def relabel_key(lf, key: str) -> str:
    """Re-attach a current, human-readable label to ``key``'s address.

    Resolves the address against ``lf`` and rebuilds the full
    ``"<address> | <label>"`` from the matching signal (so a hand-edited or
    stale-labelled sidecar is rewritten with the right label on save). Falls back
    to the bare address when nothing on ``lf`` matches.
    """
    pk = parse_key(key)
    idxs = resolve_key(lf, key)
    if idxs:
        return format_signal_key(lf.signals[idxs[0]], include_coord=pk.coord is not None)
    return format_key(pk.sensor, pk.modality, pk.coord)


def resolve_key(lf, key: str) -> List[int]:
    """Resolve a sidecar key to the matching indices in ``lf.signals``.

    Delegates to :meth:`delsys.signals.Signal.matches` — the same predicate
    :meth:`delsys.Log._splice_emg_back` uses: a signal matches when its
    ``sensor.number`` and ``modality`` equal the address, and — when the address
    carries a ``coord`` — its ``subchannel`` matches too. A coord-less key fans
    out to every sub-channel of that sensor+modality.
    """
    pk = parse_key(key)
    return [
        i for i, sig in enumerate(lf.signals) if sig.matches(pk.sensor, pk.modality, pk.coord)
    ]


def _as_spans(seq) -> List[Tuple[Optional[float], Optional[float]]]:
    """Coerce ``[[a, b], ...]`` into ``(a, b)`` spans, preserving ``None`` ends.

    Unlike :func:`_as_pairs`, an endpoint may be ``None`` (open). Closed spans
    with ``b < a`` are normalized to ascending order.
    """
    out: List[Tuple[Optional[float], Optional[float]]] = []
    for iv in seq or []:
        if iv is None or len(iv) < 2:
            continue
        a = None if iv[0] is None else float(iv[0])
        b = None if iv[1] is None else float(iv[1])
        if a is not None and b is not None and b < a:
            a, b = b, a
        out.append((a, b))
    return out


def _span_eq(a, b, tol: float = 1e-6) -> bool:
    """Whether two spans name the same interval, tolerant of JSON round-tripping."""
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and abs(float(x) - float(y)) > tol:
            return False
    return True


def _noise_record(val) -> dict:
    """One channel's noise track as a **decision**, in ``datanavigator.events.EventData`` fields.

    ``default`` (what a detector proposed) / ``added`` (what a human marked) / ``removed`` (which
    proposals the human rejected), plus ``tags`` and the ``algorithm_name`` + ``params`` that
    produced ``default`` -- deliberately datanavigator's field names and meanings, not delsys
    inventions, so the on-disk shape is already ``EventData.asdict()`` and can be handed to the
    real class once importing it stops costing the video stack (see ``pyproject.toml``).

    ``dead`` is the one delsys addition: a whole-recording kill has no counterpart in EventData.

    Legacy shapes lift in rather than migrate: a bare ``[[t0, t1], ...]`` or a ``{"windows": ...}``
    entry becomes ``added``, which is what it always meant -- every window in an existing sidecar
    was placed by a human.
    """
    if isinstance(val, list):
        return {"default": [], "added": _as_spans(val), "removed": [], "tags": [],
                "algorithm_name": None, "params": {}, "dead": []}
    if not isinstance(val, dict):
        return {"default": [], "added": [], "removed": [], "tags": [],
                "algorithm_name": None, "params": {}, "dead": []}
    added = _as_spans(val.get("added"))
    for span in _as_spans(val.get("windows")):          # legacy alias
        if not any(_span_eq(span, a) for a in added):
            added.append(span)
    dead_raw = val.get("dead")
    return {
        "default": _as_spans(val.get("default")),
        "added": added,
        "removed": _as_spans(val.get("removed")),
        "tags": [str(t) for t in (val.get("tags") or [])],
        "algorithm_name": (None if val.get("algorithm_name") is None
                           else str(val["algorithm_name"])),
        "params": dict(val.get("params") or {}),
        "dead": [(None, None)] if dead_raw is True else _as_spans(dead_raw),
    }


def _effective_windows(rec: dict) -> list:
    """``(default - removed) + added`` -- the windows that actually get masked.

    Same rule as ``EventData``: a rejected proposal is remembered in ``removed`` rather than
    deleted, so re-running the detector cannot resurrect it, and an ``added`` window has no
    ``removed`` counterpart because deleting one just deletes it.
    """
    removed = rec.get("removed") or []
    kept = [w for w in (rec.get("default") or [])
            if not any(_span_eq(w, r) for r in removed)]
    return kept + list(rec.get("added") or [])


def _normalize_signal_value(val) -> Tuple[list, list]:
    """Split a sidecar signal value into ``(effective windows, dead)`` span lists.

    Accepts the decision form above and, unchanged, every shape that ever worked: a bare list
    ``[[t0, t1], ...]``, ``{"windows": [...], "dead": [...]}``, and ``"dead": true`` as sugar for
    one whole-extent dead span.
    """
    rec = _noise_record(val)
    return _effective_windows(rec), rec["dead"]


def _canonical_value(val) -> dict:
    """On-disk form of one channel's noise decision, dropping empty fields.

    Every field survives the round-trip -- which is the point. The previous version flattened
    whatever it was given down to ``{"windows", "dead"}``, so a detector's provenance, a human's
    rejections, and a ``reviewed`` tag were all silently discarded on the next save.
    """
    rec = _noise_record(val)
    out: dict = {}
    for key in ("default", "added", "removed"):
        if rec[key]:
            out[key] = [[a, b] for a, b in rec[key]]
    if rec["dead"]:
        out["dead"] = [[a, b] for a, b in rec["dead"]]
    if rec["tags"]:
        out["tags"] = list(rec["tags"])
    if rec["algorithm_name"]:
        out["algorithm_name"] = rec["algorithm_name"]
    if rec["params"]:
        out["params"] = dict(rec["params"])
    return out


def read_noise_sidecar(path: str) -> Dict:
    """Read a ``<stem>.delsys-noise`` sidecar as ``{"schema", "signals"}``.

    Tolerates a bare ``{key: value}`` mapping (no envelope) by wrapping it.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        return {"schema": SIDECAR_SCHEMA, "signals": {}}
    if "signals" not in doc and "schema" not in doc:
        # Bare {key: value} mapping — treat the whole doc as the signal map.
        return {"schema": SIDECAR_SCHEMA, "signals": doc}
    doc.setdefault("schema", SIDECAR_SCHEMA)
    doc.setdefault("signals", {})
    return doc


def write_noise_sidecar(path: str, signals: Dict[str, object]) -> str:
    """Write a ``<stem>.delsys-noise`` sidecar (stable, canonical-value order).

    ``signals`` maps a key to either a bare windows list or a
    ``{"windows", "dead"}`` object; values are normalized via
    :func:`_canonical_value` (empty entries dropped).
    """
    body = {k: _canonical_value(v) for k, v in signals.items()}
    body = {k: v for k, v in body.items() if v}
    doc = {"schema": SIDECAR_SCHEMA, "signals": body}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _apply_noise_signal_map(lf, signals: Dict[str, object], *, policy: str = "nan_interp") -> int:
    """Resolve a ``{address-key: value}`` noise map onto ``lf`` and mask it.

    The shared core behind the legacy :func:`apply_noise_sidecar` and the unified
    :func:`delsys._events.apply_events_noise`. Each key is resolved to concrete
    signal columns (:func:`resolve_key`) and its windows / dead spans are
    accumulated onto those columns, then applied via :func:`_mask_signals`. Keys
    that resolve to nothing on this Log are skipped.

    Returns the number of signals touched.
    """
    spec_by_index: Dict[int, dict] = {}
    for key, val in (signals or {}).items():
        windows, dead = _normalize_signal_value(val)
        if not windows and not dead:
            continue
        for idx in resolve_key(lf, key):
            slot = spec_by_index.setdefault(idx, {"windows": [], "dead": []})
            slot["windows"].extend(windows)
            slot["dead"].extend(dead)
    return _mask_signals(lf, spec_by_index, policy=policy)


def apply_noise_sidecar(lf, path: str, *, policy: str = "nan_interp") -> int:
    """Read a ``<stem>.delsys-noise`` sidecar and mask ``lf`` per signal address.

    Thin wrapper over :func:`_apply_noise_signal_map` reading the legacy
    per-signal sidecar. Returns the number of signals touched.
    """
    doc = read_noise_sidecar(path)
    return _apply_noise_signal_map(lf, doc.get("signals") or {}, policy=policy)


def _rebuild_sensors(lf) -> None:
    """Rebuild ``lf.sensors`` from ``lf.signals`` in the current canonical order.

    The aggregate bundle views (``lf.emg`` etc.) are derived from ``lf.sensors``,
    not ``lf.signals``, so an in-place edit of the signals must rebuild the
    sensors for the change to be visible downstream (e.g. to the cleaner).
    """
    info_by_num = {}
    for sig in lf.signals:
        si = (getattr(sig, "meta", None) or {}).get("sensor")
        if si is not None:
            info_by_num[si.number] = si
    order = [n for n in lf.sensor_numbers if n in info_by_num]
    lf.sensors = lf._signals_to_sensors([info_by_num[n] for n in order], lf.signals)


# ---------------------------------------------------------------------------
# Automatic noise-candidate detection
# ---------------------------------------------------------------------------
#: Recorded as ``algorithm_name`` on the proposals this detector writes, so a later run
#: can tell its own ``default`` entries from another detector's.
DETECTOR_NAME = "bilateral-transient"
#: Defaults for :func:`detect_noise`, tuned on the pia02 emgmax takes against four
#: hand-confirmed events (3 artifacts, 1 real maximal contraction; all four classified
#: correctly, the artifacts ranking 1-2-3 of 25 candidates).
DETECT_WIN_SIZE = 0.025       # envelope window: short, so a ~50 ms transient is resolved
DETECT_BASELINE = 2.0         # seconds; the local level a transient is measured against
DETECT_THRESHOLD = 8.0        # x the local level
DETECT_SIDE_MIN = 2           # channels per side that must fire together
DETECT_MERGE = 0.10           # seconds; gap below which two hits are one event
DETECT_PAD = 0.05             # seconds added each side, since the tails fall below threshold


def _detect_envelopes(lf, modality: str, win_size: float):
    """Per-channel amplitude envelopes on ONE common time grid, with each channel's side.

    The grid matters: a Quattro runs at 2222 Hz against a single-differential's 1259 Hz, and
    :meth:`delsys.emg.EMG.rms` cannot deliver a requested rate exactly (its step is a whole
    number of input samples), so the envelopes arrive on grids that differ in both rate and
    ``t0``. Everything here is a cross-channel comparison, so they are interpolated onto the
    first channel's grid before anything is compared.
    """
    import numpy as np

    env, side, grid, sr = {}, {}, None, None
    for sensor in getattr(lf, "sensors", []):
        bundle = getattr(sensor, modality.lower(), None)
        if bundle is None:
            continue
        singles = (bundle.split_by_signal_name() if bundle.n_signals() > 1 else [bundle])
        for ch in singles:
            name = (list(getattr(ch, "signal_names", None) or [None]) or [None])[0]
            if name is None or not hasattr(ch, "rms"):
                continue
            v = ch.rms(win_size=win_size)
            x = np.asarray(v()).reshape(-1)
            t = np.asarray(v.t, dtype=float)
            if grid is None:
                grid, sr = t, float(v.sr)
            env[name] = x if t.shape == grid.shape and np.allclose(t, grid) else np.interp(grid, t, x)
            side[name] = getattr(getattr(ch, "sensor", None), "lrc", None)
    return env, side, grid, sr


def detect_noise(
    lf,
    *,
    modality: str = "EMG",
    win_size: float = DETECT_WIN_SIZE,
    baseline: float = DETECT_BASELINE,
    threshold: float = DETECT_THRESHOLD,
    side_min: int = DETECT_SIDE_MIN,
    merge: float = DETECT_MERGE,
    pad: float = DETECT_PAD,
    write: "str | None" = None,
    path: "str | None" = None,
):
    """Find mechanical/electrical artifact candidates for a human to refine.

    Two properties separate an artifact from a contraction, and **neither is amplitude** --
    on a pia02 emgmax take a confirmed real contraction reached 121x the channel's median,
    *higher* than one of the confirmed artifacts, so any absolute threshold classifies them
    the same way:

    1. **It is a transient against its own local level.** The baseline here is the median of
       the surrounding ``baseline`` seconds, not of the whole take. A contraction ramps up and
       lifts its own baseline, so it scores low; a 50 ms spike cannot, and scores 100-400x.
       (A global median is useless on these files: most of a 15-minute take is rest, so the
       global median *is* the rest level and every real contraction clears any multiple of it.)
    2. **It crosses the body.** A gesture is one-sided -- during a confirmed left-hand
       contraction every right-arm channel sat below 2x its median -- while a cable tug or a
       bumped sensor lights up both arms at the same instant.

    Detection is per **event**, not per channel, which is the point: one artifact hitting 16
    channels is one row to review rather than 16 findings.

    Args:
        lf: A :class:`~delsys.log.Log`.
        modality: Which modality to scan (``"EMG"``).
        win_size: Envelope window, seconds. Short enough to resolve the transient.
        baseline: Seconds of local context the excursion is measured against.
        threshold: How many times the local level counts as hot.
        side_min: Channels per side that must be hot simultaneously. The bilateral test needs
            at least this many channels on each side to exist; when the montage is one-sided
            it is skipped and a warning says so, leaving only the transient test.
        merge: Hits closer than this are one event.
        pad: Seconds added each side of an event, because the transient's tails fall below
            threshold and a window that clips them still leaves part of the artifact in.
        write: ``None`` to return candidates without touching disk (the default -- look first).
            ``"noise"`` appends them to the sidecar's **noise** track for the affected channels
            only, so ``lf.view("sensor")`` shows them and ``alt+n`` removes a false positive.
            No schema change: a detected window is an ordinary ``windows`` entry -- which also
            means that once written it is **indistinguishable from one you marked by hand**.
        path: Sidecar path; defaults to the Log's own.

    Returns:
        Candidates, highest-scoring first. Each is a dict with ``t0``, ``t1``, ``dur``,
        ``peak`` (the largest local-level multiple in the event), ``channels`` (the hot ones),
        ``n_left`` / ``n_right``.

    Example:
        .. code-block:: python

            lf = delsys.Log("Trial_1.h5")
            for c in lf.detect_noise()[:10]:          # look before writing
                print(f"{c['t0']:8.1f}s  {c['dur']*1000:5.0f} ms  {c['peak']:6.0f}x")
            lf.detect_noise(write="noise")            # then seed the sidecar and refine
            lf.view("sensor")
    """
    import warnings

    import numpy as np

    env, side, grid, sr = _detect_envelopes(lf, modality, win_size)
    if not env:
        return []
    names = list(env)

    # Local level, held over non-overlapping blocks: cheap, and a block median is exactly the
    # "what is normal around here" that a transient has to stand out from.
    w = max(1, int(round(baseline * sr)))
    nb = max(1, len(grid) // w)
    n = nb * w
    ratio = np.empty((len(names), n), dtype=float)
    for i, nm in enumerate(names):
        x = env[nm][:n]
        base = np.repeat(np.median(x.reshape(nb, w), axis=1), w)
        ratio[i] = x / np.maximum(base, 1e-12)
    t = grid[:n]

    hot = ratio > float(threshold)
    sides = np.array([side[nm] for nm in names])
    n_left, n_right = hot[sides == "L"].sum(0), hot[sides == "R"].sum(0)
    per_side = {s: int((sides == s).sum()) for s in ("L", "R")}
    if min(per_side["L"], per_side["R"]) < side_min:
        warnings.warn(
            f"delsys.detect_noise: the bilateral test needs >= {side_min} {modality} channels "
            f"per side but the montage has L={per_side['L']}, R={per_side['R']}; falling back to "
            "the transient test alone, which cannot tell a brief artifact from a brief "
            "contraction. Expect false positives.", RuntimeWarning, stacklevel=2)
        keep = hot.sum(0) >= max(2, side_min)
    else:
        keep = (n_left >= side_min) & (n_right >= side_min)

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(t[idx]) > merge) + 1)

    out = []
    for g in groups:
        chans = [names[i] for i in np.flatnonzero(hot[:, g].any(axis=1))]
        out.append(dict(
            t0=float(t[g[0]] - pad), t1=float(t[g[-1]] + pad),
            dur=float(t[g[-1]] - t[g[0]] + 2 * pad),
            peak=float(ratio[:, g].max()), channels=chans,
            n_left=int(n_left[g].max()), n_right=int(n_right[g].max()),
        ))
    out.sort(key=lambda c: -c["peak"])

    if write == "noise":
        _write_detected(lf, out, path, params=dict(
            win_size=win_size, baseline=baseline, threshold=threshold,
            side_min=side_min, merge=merge, pad=pad, modality=modality))
    elif write is not None:
        raise ValueError(f"unknown write target {write!r}; use None or 'noise'")
    return out


def _write_detected(lf, candidates, path=None, params=None) -> str:
    """Write candidates as the noise track's ``default``, per channel.

    ``default`` is **replaced wholesale**, which is the whole point of separating it from
    ``added``: re-running the detector with different settings supersedes its own previous
    proposals while leaving every human ``added`` window and every ``removed`` rejection intact.
    A proposal the human already rejected stays rejected, because ``removed`` survives too.

    Per channel rather than per sensor: the reason to detect at all is that a corrupted maximum
    is per channel, and a narrow mark is easier to judge than a blanket one.
    """
    from delsys import _events

    path = path or _events.events_path_for(lf.fname)
    doc = _events.read_events(path)
    events = doc.get("events", {})
    section = events.get(_events.NOISE_TYPE) or {"kind": "noise", "signals": {}}
    signals = {k: _noise_record(v) for k, v in (section.get("signals") or {}).items()}

    # Keys come from the Log's Signal objects, which is what the annotator and the masking core
    # both address by -- NOT from the EMG bundles the envelopes were computed on. A bundle has no
    # modality/subchannel, so format_signal_key cannot key off one.
    by_name = {}
    for sig in (getattr(lf, "signals", None) or []):
        try:
            by_name[_signal_label(sig)] = format_signal_key(sig)
        except Exception:  # noqa: BLE001 -- a signal whose label cannot be built is not addressable
            continue
    if not by_name:
        raise ValueError(
            "cannot write detected noise: the Log exposes no addressable signals, so a window "
            "could not be keyed the way the annotator and the masking core read it")

    fresh = {}
    for cand in candidates:
        for nm in cand["channels"]:
            key = by_name.get(nm)
            if key is not None:
                fresh.setdefault(key, []).append([cand["t0"], cand["t1"]])
    for key, spans in fresh.items():
        rec = signals.setdefault(
            key, {"default": [], "added": [], "removed": [], "tags": [],
                  "algorithm_name": None, "params": {}, "dead": []})
        rec["default"] = spans                       # supersede, do not accumulate
        rec["algorithm_name"] = DETECTOR_NAME
        rec["params"] = dict(params or {})
    # a channel that USED to carry proposals and now has none must lose them too
    for key, rec in signals.items():
        if key not in fresh and rec.get("algorithm_name") == DETECTOR_NAME:
            rec["default"] = []
    events[_events.NOISE_TYPE] = {"kind": "noise", "signals": signals}
    return _events.write_events(path, events)
