"""Glue between an :class:`delsys.ekg.EKG` review decision and the unified
``<stem>.delsys-events`` sidecar (its ``rpeaks`` type + shared ``noise`` track).

The :class:`~delsys.ekg.EKG` class owns the grid mechanics
(:meth:`~delsys.ekg.EKG.rpeaks_decision` /
:meth:`~delsys.ekg.EKG.apply_rpeaks_decision`); this module owns the *addressing*
(which sidecar key a bundle's channel maps to) and the read/merge/write against
the on-disk file. Kept out of :mod:`delsys.ekg` so the low-level bundle class
doesn't import the sidecar layer at module load.

``EKG.load_rpeaks`` / ``EKG.save_rpeaks`` and ``Log.ekg`` (auto-load) call in
here. An EKG bundle carries its source path in ``meta["source"]`` (stamped by
``Log.ekg`` / ``Log.ekg_raw``); the sidecar path is
``_events.events_path_for(source)``.
"""

import os
from typing import List, Optional

from delsys import _events, _noise

#: EKG's single sub-channel coord (``SUBCHANNEL_MAP['EKG'] == ('A',)``).
_EKG_COORD = "A"


def ekg_channel_address(ekg, i: int = 0, include_coord: bool = True) -> str:
    """Structural sidecar key for channel ``i`` of an EKG bundle.

    ``"<sensor>.EKG.A | <location>"`` — the same grammar the noise/marker tracks
    use (see :func:`delsys._noise.format_key`). The label is informational; lookup
    strips it.
    """
    sensors = ekg.sensors
    if not sensors:
        raise ValueError("EKG bundle carries no sensor metadata; cannot address it.")
    sensor = sensors[i]
    names = list(getattr(ekg, "signal_names", []) or [])
    label = names[i] if i < len(names) else (getattr(sensor, "location", "") or "")
    return _noise.format_key(sensor.number, "EKG", _EKG_COORD if include_coord else None, label)


def _match_sensor_ekg(signal_map: dict, sensor_num: int):
    """Value in a ``{address-key: value}`` map addressing ``sensor_num`` + EKG.

    Coord is ignored (a sensor+modality has one EKG channel), so a coord-ful or
    coord-less key both resolve. ``None`` when absent.
    """
    for key, val in (signal_map or {}).items():
        pk = _noise.parse_key(key)
        if pk.sensor == sensor_num and pk.modality == "EKG":
            return val
    return None


def _noise_windows_for(noise_map: dict, sensor_num: int) -> List[List[float]]:
    """Closed noise windows (seconds) for this sensor's EKG channel."""
    val = _match_sensor_ekg(noise_map, sensor_num)
    if not val:
        return []
    windows, _dead = _noise._normalize_signal_value(val)
    return [[a, b] for a, b in windows if a is not None and b is not None]


def apply_sidecar(ekg, path: str) -> bool:
    """Apply the sidecar's rpeaks decision + noise windows to a single-channel EKG.

    Returns ``True`` iff a decision was found and applied. A missing file, a
    multi-channel bundle, or no matching entry are all quiet no-ops (return
    ``False``) — signal access must never break on a sidecar.
    """
    if ekg.n_signals() > 1 or not path or not os.path.exists(path):
        return False
    sensors = ekg.sensors
    if not sensors:
        return False
    sensor_num = sensors[0].number
    decision = _match_sensor_ekg(_events.read_rpeaks_signals(path), sensor_num)
    if decision is None:
        return False
    windows = _noise_windows_for(_events.read_noise_signals(path), sensor_num)
    ekg.apply_rpeaks_decision(decision, noise_windows=windows or None)
    return True


def save_sidecar(ekg, path: str) -> str:
    """Merge this single-channel EKG's decision + noisy segments into ``path``.

    Preserves every other section (other channels' rpeaks, noise, markers);
    replaces only this channel's rpeaks entry and its noise windows. Returns the
    written path.
    """
    if ekg.n_signals() > 1:
        raise NotImplementedError("save_sidecar requires a single EKG channel.")
    addr = ekg_channel_address(ekg)
    addr_id = _noise.key_address(addr)
    events = dict(_events.read_events(path).get("events", {}))

    rp = {
        k: v
        for k, v in (events.get(_events.RPEAKS_TYPE, {}).get("signals") or {}).items()
        if _noise.key_address(k) != addr_id
    }
    rp[addr] = ekg.rpeaks_decision()
    events[_events.RPEAKS_TYPE] = {"signals": rp}

    windows = [
        [float(ekg.t[a]), float(ekg.t[b])]
        for a, b in (ekg.meta.get("noisy_segments_idx") or [])
    ]
    ns = {
        k: v
        for k, v in (events.get(_events.NOISE_TYPE, {}).get("signals") or {}).items()
        if _noise.key_address(k) != addr_id
    }
    if windows:
        ns[addr] = {"windows": windows}
    if ns:
        events[_events.NOISE_TYPE] = {"signals": ns}
    elif _events.NOISE_TYPE in events:
        events[_events.NOISE_TYPE] = {"signals": ns}

    return _events.write_events(path, events)


def events_path_for_ekg(ekg) -> Optional[str]:
    """The ``.delsys-events`` path for an EKG bundle's stamped source, or ``None``."""
    src = (ekg.meta or {}).get("source")
    return _events.events_path_for(src) if src else None
