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


def format_signal_key(sig, *, include_coord: bool = True) -> str:
    """Build the sidecar key addressing a single :class:`delsys.signals.Signal`.

    ``include_coord=False`` produces the whole-modality address (all
    sub-channels of the signal's sensor+modality).

    The ``<label>`` is the sensor's body location via
    :func:`delsys._util._trim_location` (the same source the modality bundles
    use — e.g. ``"Tricep"``; ``"ch<number>"`` when no channelmap was loaded).
    A per-channel :class:`Signal`'s own ``signal_names`` is a pysampled
    placeholder (``"s0"``), so it is deliberately *not* used.
    """
    sensor = sig.sensor
    label = _trim_location(getattr(sensor, "location", None), sensor.number)
    return format_key(
        sensor.number,
        sig.modality,
        sig.subchannel if include_coord else None,
        label,
    )


def resolve_key(lf, key: str) -> List[int]:
    """Resolve a sidecar key to the matching indices in ``lf.signals``.

    Reuses the sensor/modality/sub-channel matching that
    :meth:`delsys.Log._splice_emg_back` uses: a signal matches when its
    ``sensor.number`` and ``modality`` equal the address, and — when the address
    carries a ``coord`` — its ``subchannel`` matches too. A coord-less key fans
    out to every sub-channel of that sensor+modality.
    """
    pk = parse_key(key)
    out: List[int] = []
    for i, sig in enumerate(lf.signals):
        sensor = getattr(sig, "sensor", None)
        if sensor is None or sensor.number != pk.sensor:
            continue
        if sig.modality != pk.modality:
            continue
        if pk.coord is not None and sig.subchannel != pk.coord:
            continue
        out.append(i)
    return out


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


def _normalize_signal_value(val) -> Tuple[list, list]:
    """Split a sidecar signal value into ``(windows, dead)`` span lists.

    Accepts:
    - a bare list ``[[t0, t1], ...]`` — shorthand for windows-only;
    - an object ``{"windows": [...], "dead": [...]}`` (both optional);
    - ``"dead": true`` — sugar for one whole-extent dead span ``[None, None]``.
    """
    if isinstance(val, list):
        return _as_spans(val), []
    if isinstance(val, dict):
        windows = _as_spans(val.get("windows"))
        dead_raw = val.get("dead")
        if dead_raw is True:
            dead = [(None, None)]
        else:
            dead = _as_spans(dead_raw)
        return windows, dead
    return [], []


def _canonical_value(val) -> dict:
    """Normalize a sidecar value to ``{"windows": [...], "dead": [...]}`` form,
    omitting empty fields (used on write for stable, tidy output)."""
    windows, dead = _normalize_signal_value(val)
    out: dict = {}
    if windows:
        out["windows"] = [[a, b] for a, b in windows]
    if dead:
        out["dead"] = [[a, b] for a, b in dead]
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


def apply_noise_sidecar(lf, path: str, *, policy: str = "nan_interp") -> int:
    """Read a ``<stem>.delsys-noise`` sidecar and mask ``lf`` per signal address.

    Each key is resolved to concrete signal columns (:func:`resolve_key`) and
    its windows / dead spans are accumulated onto those columns, then applied via
    :func:`_mask_signals`. Keys that resolve to nothing on this Log are skipped.

    Returns the number of signals touched.
    """
    doc = read_noise_sidecar(path)
    spec_by_index: Dict[int, dict] = {}
    for key, val in (doc.get("signals") or {}).items():
        windows, dead = _normalize_signal_value(val)
        if not windows and not dead:
            continue
        for idx in resolve_key(lf, key):
            slot = spec_by_index.setdefault(idx, {"windows": [], "dead": []})
            slot["windows"].extend(windows)
            slot["dead"].extend(dead)
    return _mask_signals(lf, spec_by_index, policy=policy)


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
