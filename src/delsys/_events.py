"""Unified per-log annotation sidecar (``<stem>.delsys-events``).

One file holds **every** annotation a human places over a :class:`delsys.Log`,
split by *event type*:

- ``"noise"`` — the built-in quality track: per-signal noise ``windows`` (transient
  bursts, NaN+interpolated by :func:`delsys.clean`) and ``dead`` spans (no usable
  signal, zero-filled). Same ``{windows, dead}`` payload + signal-address key
  grammar as the legacy ``<stem>.delsys-noise`` sidecar (see :mod:`delsys._noise`).
- any other name (``"1"``, ``"2"``, …) — a *marker* track: typed point (``size=1``)
  or window (``size=2``) events. Authored **per signal** (the signal address is
  provenance — which trace the marker was placed from), but consumed at analysis
  time as **trial-level** markers via :func:`collapse_markers`.

Why one file (not datanavigator's one-file-per-event-type): the marks for a trial
travel together next to its ``Trial_N.h5``, and *how* each type is treated is the
consumer's business — :func:`delsys.clean` reads only the ``"noise"`` type; an
epoching/analysis step reads the marker types. datanavigator's ``Events`` machinery
is still the in-memory marking + display engine in the annotator (see
:mod:`delsys.annotate`); this module owns only the on-disk unified format + the
trial-level collapse, so the browser's per-``Event`` save is bypassed in favour of
one unified write.

On-disk shape (composite suffix, not ``.json`` — same rationale as
``<stem>.delsys-noise`` / datanavigator's ``.dnav-toc``)::

    {
      "schema": 1,
      "events": {
        "noise": {"kind": "noise",
                  "signals": {"3.EMGS | RForearm": {"windows": [[0.2, 0.3]]}}},
        "1":     {"kind": "marker", "size": 1,
                  "signals": {"3.EMGS | RForearm": [[1.40], [2.10]]}},
        "2":     {"kind": "marker", "size": 2,
                  "signals": {"3.EMGS | RForearm": [[0.50, 0.90]]}}
      }
    }

A marker type's per-signal value is a flat list of *sequences*, each a list of
``size`` times (datanavigator ``EventData.added`` shape; ``default`` / ``removed``
are unused here — these are hand-placed, not algorithm-seeded).
"""

import json
import os
from typing import Dict, List, Optional, Union

from delsys import _noise

#: Composite suffix of the unified per-log annotation sidecar.
EVENTS_SUFFIX = ".delsys-events"

#: Bump when the unified layout changes incompatibly. Schema 2 adds the
#: ``"rpeaks"`` type (an older reader would misread its per-signal dict as a
#: marker track — see :func:`marker_types`).
EVENTS_SCHEMA = 2

#: The built-in quality track name.
NOISE_TYPE = "noise"

#: The EKG R-peak review-decision track name. A reserved type (neither ``noise``
#: nor a marker) whose per-signal value is a curation *decision* — see
#: :func:`read_rpeaks_signals`.
RPEAKS_TYPE = "rpeaks"


def events_path_for(target: Union[str, "os.PathLike"]) -> str:
    """Sibling ``<stem>.delsys-events`` path for a checkpoint / file ``target``."""
    return os.path.splitext(str(target))[0] + EVENTS_SUFFIX


# ---------------------------------------------------------------------------
# Read / write the unified document
# ---------------------------------------------------------------------------


def read_events(path: str) -> Dict:
    """Read a ``<stem>.delsys-events`` doc as ``{"schema", "events": {...}}``.

    Tolerates a bare ``{type: {...}}`` mapping (no envelope) by wrapping it.
    Returns the empty shape for a missing file.
    """
    if not os.path.exists(path):
        return {"schema": EVENTS_SCHEMA, "events": {}}
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        return {"schema": EVENTS_SCHEMA, "events": {}}
    if "events" not in doc and "schema" not in doc:
        return {"schema": EVENTS_SCHEMA, "events": doc}
    doc.setdefault("schema", EVENTS_SCHEMA)
    doc.setdefault("events", {})
    return doc


def write_events(path: str, events: Dict[str, dict]) -> str:
    """Write a ``<stem>.delsys-events`` doc (stable, canonical order).

    ``events`` maps a type name to its section. The ``"noise"`` section is
    canonicalized via :mod:`delsys._noise` (empty entries dropped); marker
    sections keep their ``{kind, size, signals}`` shape with empty-signal entries
    dropped. A section that ends up with no signals is omitted entirely.
    """
    body: Dict[str, dict] = {}
    for name, section in events.items():
        if name == NOISE_TYPE:
            signals = {
                k: _noise._canonical_value(v)
                for k, v in (section.get("signals") or {}).items()
            }
            signals = {k: v for k, v in signals.items() if v}
            if signals:
                body[name] = {"kind": "noise", "signals": signals}
        elif name == RPEAKS_TYPE:
            signals = {
                k: _canonical_rpeaks_value(v)
                for k, v in (section.get("signals") or {}).items()
            }
            signals = {k: v for k, v in signals.items() if v}
            if signals:
                body[name] = {"kind": "rpeaks", "signals": signals}
        else:
            signals = _clean_marker_signals(section.get("signals") or {})
            if signals:
                body[name] = {
                    "kind": "marker",
                    "size": int(section.get("size", 1)),
                    "signals": signals,
                }
    doc = {"schema": EVENTS_SCHEMA, "events": body}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _clean_marker_signals(signals: Dict[str, object]) -> Dict[str, list]:
    """Coerce a marker section's per-signal value to its on-disk form, dropping
    empties. Each event serializes as a **bare** ``[t, ...]`` sequence, or the
    object form ``{"seq": [...], "note": ..., "tags": [...]}`` when it carries a
    note/tags (so plain marks stay terse and tidy)."""
    out: Dict[str, list] = {}
    for key, val in signals.items():
        recs = _as_records(val)
        if recs:
            out[key] = [_record_to_disk(r) for r in recs]
    return out


def _as_records(val) -> List[dict]:
    """Normalize a marker value into ``[{"seq": [t, ...], "note": str|None,
    "tags": [...]}, ...]``.

    Each event element may be a **bare** sequence ``[t0, t1]`` or the object form
    ``{"seq": [...], "note": ..., "tags": [...]}`` (per-event annotation). The
    container may be a list, or an ``EventData``-style mapping
    ``{"default": [...], "added": [...]}`` (their concatenation, manual ``added``
    after algorithm ``default``).
    """
    if isinstance(val, dict) and ("default" in val or "added" in val):
        raw = list(val.get("default") or []) + list(val.get("added") or [])
    else:
        raw = val or []
    out: List[dict] = []
    for ev in raw:
        if ev is None:
            continue
        if isinstance(ev, dict):
            seq_raw, note, tags = ev.get("seq"), ev.get("note"), list(ev.get("tags") or [])
        else:
            seq_raw, note, tags = ev, None, []
        try:
            seq = [float(x) for x in seq_raw]
        except (TypeError, ValueError):
            continue
        if not seq:
            continue
        out.append({"seq": seq, "note": note, "tags": tags})
    return out


def _record_to_disk(rec: dict):
    """Serialize one event record: a bare ``[t, ...]`` unless it carries a
    note/tags, in which case the object form (empty fields omitted)."""
    if rec.get("note") or rec.get("tags"):
        out: dict = {"seq": [float(x) for x in rec["seq"]]}
        if rec.get("note"):
            out["note"] = rec["note"]
        if rec.get("tags"):
            out["tags"] = list(rec["tags"])
        return out
    return [float(x) for x in rec["seq"]]


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------


def read_noise_signals(path: str) -> Dict[str, dict]:
    """Per-signal noise map from the unified doc's ``"noise"`` type.

    Returns ``{address-key: {"windows": [...], "dead": [...]}}`` (the same shape
    :func:`delsys._noise.read_noise_sidecar` returns under ``"signals"``), or an
    empty dict when the file or the noise type is absent.
    """
    section = read_events(path).get("events", {}).get(NOISE_TYPE)
    if not section:
        return {}
    return dict(section.get("signals") or {})


def marker_types(path: str) -> List[str]:
    """Names of the marker event types present in the doc, sorted.

    Excludes the two reserved non-marker tracks (``noise`` and ``rpeaks``), whose
    per-signal values are not marker records.
    """
    events = read_events(path).get("events", {})
    return sorted(name for name in events if name not in (NOISE_TYPE, RPEAKS_TYPE))


# ---------------------------------------------------------------------------
# R-peak review-decision track (``"rpeaks"``)
# ---------------------------------------------------------------------------
#
# Per-signal value is a *decision* (not marker records):
#
#     {"detector": {"name": "pn", "highpass": 5.0, "hr_max": 200.0},
#      "added":   [1.402, 2.101],   # peak times (s) merged into the result
#      "removed": [3.550],          # detector-default peak times suppressed
#      "flipped": false,            # polarity flip re-runs detection on load
#      "tags":    ["reviewed"]}     # free-text review tags
#
# Times, never indices — so the decision reproduces on any sample grid (native-
# rate ``.h5`` reload, a slice). ``final_peaks = f(raw_ekg, this_decision)``.


def _as_time_list(seq) -> List[float]:
    """Coerce a sequence of peak times to a sorted, de-duplicated float list."""
    out = []
    for t in seq or []:
        try:
            out.append(float(t))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _canonical_detector(det) -> dict:
    """Normalize a detector-provenance block, keeping ``name`` + numeric params."""
    if not isinstance(det, dict):
        return {}
    out: dict = {"name": str(det.get("name", "pn"))}
    for k in ("highpass", "hr_max"):
        if det.get(k) is not None:
            out[k] = float(det[k])
    return out


def _canonical_rpeaks_value(val) -> dict:
    """On-disk form of one channel's rpeak decision, dropping trivial entries.

    An entry is kept only if it carries a real curation (added/removed peaks, a
    polarity flip, or tags) — accepting the auto-detection verbatim with no tag is
    not worth persisting. The ``detector`` provenance rides along on kept entries
    (it makes ``removed`` and ``flipped`` reproducible on reload).
    """
    if not isinstance(val, dict):
        return {}
    added = _as_time_list(val.get("added"))
    removed = _as_time_list(val.get("removed"))
    flipped = bool(val.get("flipped", False))
    tags = [str(t) for t in (val.get("tags") or [])]
    out: dict = {}
    if added:
        out["added"] = added
    if removed:
        out["removed"] = removed
    if flipped:
        out["flipped"] = True
    if tags:
        out["tags"] = tags
    if out:
        out["detector"] = _canonical_detector(val.get("detector") or {})
    return out


def _read_rpeaks_value(val) -> dict:
    """Read one channel's decision into a fully-defaulted dict (all keys present)."""
    val = val if isinstance(val, dict) else {}
    return {
        "detector": _canonical_detector(val.get("detector") or {}) or {"name": "pn"},
        "added": _as_time_list(val.get("added")),
        "removed": _as_time_list(val.get("removed")),
        "flipped": bool(val.get("flipped", False)),
        "tags": [str(t) for t in (val.get("tags") or [])],
    }


def read_rpeaks_signals(path: str) -> Dict[str, dict]:
    """Per-signal R-peak decisions from the unified doc's ``"rpeaks"`` type.

    Returns ``{address-key: {"detector", "added", "removed", "flipped", "tags"}}``
    (all keys present, defaulted), or an empty dict when the file or the rpeaks
    type is absent.
    """
    section = read_events(path).get("events", {}).get(RPEAKS_TYPE)
    if not section:
        return {}
    return {
        key: _read_rpeaks_value(val)
        for key, val in (section.get("signals") or {}).items()
    }


def read_marker_records(path: str, event_type: str) -> Dict[str, List[dict]]:
    """Per-signal event **records** for one marker type:
    ``{address-key: [{"seq": [...], "note": str|None, "tags": [...]}, ...]}``."""
    section = read_events(path).get("events", {}).get(event_type)
    if not section:
        return {}
    out: Dict[str, List[dict]] = {}
    for key, val in (section.get("signals") or {}).items():
        recs = _as_records(val)
        if recs:
            out[key] = recs
    return out


def read_marker_signals(path: str, event_type: str) -> Dict[str, List[List[float]]]:
    """Per-signal sequences (times only) for one marker type:
    ``{address-key: [[t, ...], ...]}`` — note/tags dropped (see
    :func:`read_marker_records` to keep them)."""
    return {
        key: [r["seq"] for r in recs]
        for key, recs in read_marker_records(path, event_type).items()
    }


def collapse_markers(
    path: str, event_type: str, *, dedupe: Optional[float] = None
) -> List[dict]:
    """Flatten one marker type across signal addresses into trial-level markers.

    Each mark — placed per signal — becomes a trial-level record carrying its
    provenance (and any per-event note/tags)::

        [{"seq": [t] | [a, b], "address": "3.EMGS", "label": "RForearm",
          "note": str|None, "tags": [...]}, ...]

    sorted by ``seq[0]``. **Every** mark is kept by default (``dedupe=None``), so
    the same logical event placed from two signals shows up twice — *visible
    disagreement* rather than a silent merge. Pass ``dedupe=<seconds>`` to drop a
    mark whose start time is within that tolerance of an already-kept mark (any
    address), keeping the earliest-encountered one.

    Args:
        path: Unified ``<stem>.delsys-events`` path.
        event_type: A marker type slug (e.g. ``"1"``).
        dedupe: Proximity tolerance in seconds, or ``None`` to keep all.
    """
    records: List[dict] = []
    for key, recs in read_marker_records(path, event_type).items():
        pk = _noise.parse_key(key)
        addr = _noise.format_key(pk.sensor, pk.modality, pk.coord)
        for r in recs:
            records.append(
                {
                    "seq": r["seq"],
                    "address": addr,
                    "label": pk.label,
                    "note": r.get("note"),
                    "tags": r.get("tags") or [],
                }
            )
    records.sort(key=lambda r: (r["seq"][0] if r["seq"] else 0.0))
    if dedupe is None:
        return records
    kept: List[dict] = []
    for rec in records:
        start = rec["seq"][0] if rec["seq"] else 0.0
        if any(abs(start - (k["seq"][0] if k["seq"] else 0.0)) <= dedupe for k in kept):
            continue
        kept.append(rec)
    return kept


# ---------------------------------------------------------------------------
# Legacy ``<stem>.delsys-noise`` bridge
# ---------------------------------------------------------------------------


def noise_signals_for(
    events_path: str, legacy_noise_path: Optional[str] = None
) -> Dict[str, dict]:
    """Resolve the effective noise map, preferring the unified file.

    Returns the ``"noise"`` map from ``events_path`` when that file exists;
    otherwise falls back to a legacy ``<stem>.delsys-noise`` (``legacy_noise_path``,
    or the sibling of ``events_path`` when omitted). Empty when neither is present.
    """
    if os.path.exists(events_path):
        return read_noise_signals(events_path)
    if legacy_noise_path is None:
        legacy_noise_path = _noise.sidecar_path_for(events_path)
    if os.path.exists(legacy_noise_path):
        return dict(_noise.read_noise_sidecar(legacy_noise_path).get("signals") or {})
    return {}


def migrate_noise_sidecar(legacy_noise_path: str, events_path: str) -> Optional[str]:
    """Fold a legacy ``<stem>.delsys-noise`` into the unified file's ``"noise"`` type.

    No-op (returns ``None``) when the legacy file is absent. When the unified file
    already carries a noise type, the legacy file is left untouched (the unified
    file wins) and ``None`` is returned. Otherwise the legacy noise map is written
    as the unified ``"noise"`` section (merged with any existing marker types) and
    the written path is returned.
    """
    if not os.path.exists(legacy_noise_path):
        return None
    doc = read_events(events_path)
    if doc.get("events", {}).get(NOISE_TYPE):
        return None
    legacy = _noise.read_noise_sidecar(legacy_noise_path).get("signals") or {}
    if not legacy:
        return None
    events = dict(doc.get("events", {}))
    events[NOISE_TYPE] = {"kind": "noise", "signals": legacy}
    return write_events(events_path, events)


# ---------------------------------------------------------------------------
# Consumption (noise masking off the unified file)
# ---------------------------------------------------------------------------


def apply_events_noise(lf, path: str, *, policy: str = "nan_interp") -> int:
    """Mask ``lf`` with the noise windows / dead spans from a unified events file.

    The unified-file counterpart to :func:`delsys._noise.apply_noise_sidecar`;
    both resolve per-signal addresses and apply the shared masking core. Returns
    the number of signals touched (0 when there is no noise type).
    """
    return _noise._apply_noise_signal_map(lf, read_noise_signals(path), policy=policy)
