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
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pysampled

#: Trial-id key as accepted by the public helpers.
TrialId = Union[str, Sequence[int]]


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


def apply_noise_mask(
    lf,
    intervals: Sequence[Tuple[float, float]],
    *,
    policy: str = "nan_interp",
    modalities: Optional[Sequence[str]] = None,
) -> int:
    """Blank (and refill) noise windows across a Log's signals, in place.

    v1 policy ``"nan_interp"``: set every sample inside a window to ``NaN``, then
    linearly interpolate it back via :meth:`pysampled.Data.interpnan`. Each
    affected :class:`delsys.signals.Signal` is replaced with a ``_clone`` of the
    filled array (preserving ``sr`` / ``t0`` / ``meta``), and the per-sensor
    bundles are rebuilt so the bundle-level views (``lf.emg`` / ``lf.acc`` / ...)
    reflect the mask on next access.

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

    touched = 0
    for i, sig in enumerate(lf.signals):
        meta = getattr(sig, "meta", None) or {}
        if modalities is not None and meta.get("modality") not in modalities:
            continue

        arr = np.asarray(sig(), dtype=float).copy()
        n = arr.shape[0]
        sr = float(sig.sr)
        t0 = float(getattr(sig, "_t0", 0.0) or 0.0)

        hit = False
        for a, b in windows:
            i0 = max(0, int(np.floor((a - t0) * sr)))
            i1 = min(n, int(np.ceil((b - t0) * sr)) + 1)
            if i1 > i0:
                arr[i0:i1] = np.nan
                hit = True
        if not hit:
            continue

        filled = np.asarray(pysampled.Data(arr, sr=sr).interpnan()())
        lf.signals[i] = sig._clone(filled)
        touched += 1

    if touched:
        _rebuild_sensors(lf)
    return touched


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
