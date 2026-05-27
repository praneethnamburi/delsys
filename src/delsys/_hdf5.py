"""HDF5 checkpoint I/O for :class:`delsys.Log`.

A checkpoint stores each channel's signal once, plus the embedded channelmap and
header metadata, so the ``.h5`` is self-contained and the source CSV becomes
disposable. Two storage modes, recorded per-signal via ``is_native``:

* **native** (Trigno-Base modalities exported with ``target_sr=None``) — stored at
  the acquisition rate and *re-resampled on load* to whatever ``target_sr`` /
  ``clock_mul`` the caller asks for. Alignment is therefore a load-time concern,
  not baked into storage.
* **terminal snapshot** (link devices, or a Log that was already resampled) —
  returned as-stored; load-time ``target_sr`` does not re-resample it.

Storage is ``float32`` + ``lzf`` (lossless vs the CSV's ~7 significant figures);
the reader upcasts to ``float64`` before any resampling so no downstream numeric
work runs in single precision. See ``CHANGELOG.md`` (0.5.0) and the portfolio
sandbox findings for the validation that motivated this design.
"""

import json

import numpy as np
import pysampled

from delsys._constants import TARGET_SR
from delsys._metadata import SensorInfo, SensorLog
from delsys.signals import Signal

#: Bump when the on-disk layout changes in a non-back-compatible way.
SCHEMA_VERSION = 1

#: Asynchronous link-device modalities. These are *not* stored at native rate
#: (they need a target rate by construction) — they are written as terminal snapshots.
LINK_MODS = frozenset({"VO2", "HR", "SmO2", "Thb"})

#: Per-modality target map that produces a NATIVE-rate :class:`Log`: ``None`` for
#: Trigno-Base modalities (kept at acquisition rate), the default rate for link devices.
NATIVE_SR = {k: (None if k not in LINK_MODS else v) for k, v in TARGET_SR.items()}


def _set(grp, key, val):
    """Set an attr, skipping ``None`` (h5 attrs reject None and VLEN-NULL strings)."""
    if val is not None:
        grp.attrs[key] = val


def _get(grp, key):
    """Read an attr, returning ``None`` when absent."""
    return grp.attrs[key] if key in grp.attrs else None


def write(lf, path):
    """Write ``lf`` to an HDF5 checkpoint at ``path``.

    Per-signal ``is_native`` is derived from ``lf.target_sr``: a modality mapped to
    ``None`` is stored native (re-resamplable); anything else is a terminal snapshot.
    """
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["name"] = lf.name
        f.attrs["clock_mul"] = float(lf.clock_mul)
        f.attrs["t0"] = float(lf.t0)
        f.attrs["t_min"] = float(lf.t_min)
        f.attrs["t_max"] = float(lf.t_max)
        f.attrs["hdr_json"] = json.dumps(lf.hdr, default=str)
        # Canonical sensor order from the source Log -> preserves bundle channel order.
        # (h5py iterates groups alphabetically, which would otherwise scramble it.)
        f.attrs["sensor_order_json"] = json.dumps([int(s.number) for s in lf.sensors])

        # EMGworks' time window depends on the requested target_sr (Discover's is
        # ``(0, duration)``, target-independent). A native export uses the widest
        # (min_sr=1) window; recording the raw un-snapped extent lets the reader trim
        # back to the exact window of whatever target_sr it's reloaded with.
        raw_window = getattr(lf, "_raw_window", None)
        if raw_window is not None:
            f.attrs["window_mode"] = "emgworks"
            f.attrs["raw_t_min"] = float(raw_window[0])
            f.attrs["raw_t_max"] = float(raw_window[1])

        cm = f.create_group("channelmap")
        for i, sl in enumerate(lf.sensor_map):
            g = cm.create_group(str(i))
            g.attrs["number"] = int(sl.number)
            _set(g, "type_tag", sl[1])
            _set(g, "lrc", sl[2])
            _set(g, "location", sl[3])

        # One SensorInfo per physical sensor, gathered from signal meta.
        seen = {}
        for sig in lf.signals:
            si = sig.meta["sensor"]
            seen[si.number] = si
        sens = f.create_group("sensors")
        for num, si in seen.items():
            g = sens.create_group(str(num))
            _set(g, "name", si.name)
            g.attrs["number"] = int(si.number)
            g.attrs["modalities_json"] = json.dumps(sorted(si.modalities))
            _set(g, "type_sensorlog", si.type_sensorlog)
            _set(g, "lrc", si.lrc)
            _set(g, "location", si.location)

        sigs = f.create_group("signals")
        for i, sig in enumerate(lf.signals):
            g = sigs.create_group(str(i))
            g.create_dataset(
                "data",
                data=np.asarray(sig._sig, dtype=np.float32),
                chunks=True,
                compression="lzf",
            )
            mod = sig.meta["modality"]
            g.attrs["sr"] = float(sig.sr)
            g.attrs["modality"] = mod
            _set(g, "subchannel", sig.meta["subchannel"])
            g.attrs["sensor_number"] = int(sig.meta["sensor"].number)
            g.attrs["is_native"] = bool(lf.target_sr.get(mod) is None)


def read_into(lf, path, target_sr=None, clock_mul=1.0, t0=0.0):
    """Populate an existing :class:`Log` instance ``lf`` from an HDF5 checkpoint.

    Native signals are resampled to ``target_sr`` (with ``clock_mul`` scaling the
    native rate); terminal snapshots are returned as stored. Arrays are upcast
    ``float32 -> float64`` before resampling.
    """
    import h5py

    from delsys.log import _normalize_signal_lengths

    if target_sr is None:
        target_sr = TARGET_SR
    lf.fname = path
    lf.clock_mul = float(clock_mul)
    lf.t0 = float(t0)
    lf.target_sr = target_sr
    lf.sensor_name_replace = {}

    with h5py.File(path, "r") as f:
        lf.name = f.attrs["name"]
        lf.hdr = json.loads(f.attrs["hdr_json"])
        lf.t_min = float(f.attrs["t_min"])
        lf.t_max = float(f.attrs["t_max"])
        sensor_order = json.loads(f.attrs["sensor_order_json"])

        # EMGworks: the checkpoint stored signals on the widest (min_sr=1) window;
        # trim back to the exact window this target_sr would produce so reload matches
        # Log(csv, target_sr) bitwise. (The native interp grid is baked at clock_mul=1,
        # so a clock-shifted reload can't be reproduced by reinterpretation.)
        window_mode = f.attrs["window_mode"] if "window_mode" in f.attrs else None
        emg_t_min = emg_t_max = None
        if window_mode == "emgworks":
            if abs(lf.clock_mul - 1.0) > 1e-12:
                raise NotImplementedError(
                    "Reloading an EMGworks checkpoint with clock_mul != 1 is not "
                    "supported: the native interpolation grid is fixed at clock_mul=1. "
                    "Reload at clock_mul=1 (or re-export). Discover checkpoints are exempt."
                )
            _grid = [v for v in target_sr.values() if v is not None]
            if _grid:  # a fully-native reload keeps the stored window (no trim)
                _min_sr = min(_grid)
                raw_t_min = float(f.attrs["raw_t_min"])
                raw_t_max = float(f.attrs["raw_t_max"])
                emg_t_min = np.floor(raw_t_min * _min_sr) / _min_sr
                emg_t_max = np.ceil(raw_t_max * _min_sr) / _min_sr
                if abs(emg_t_min - lf.t_min) > 1e-9:
                    raise NotImplementedError(
                        "EMGworks checkpoint reload with a nonzero start-time window "
                        f"(t_min={emg_t_min}) is not supported."
                    )

        cm = f["channelmap"]
        lf.sensor_map = [
            SensorLog(
                int(cm[k].attrs["number"]),
                _get(cm[k], "type_tag"),
                _get(cm[k], "lrc"),
                _get(cm[k], "location"),
            )
            for k in sorted(cm, key=int)
        ]

        sens = {}
        for k in f["sensors"]:
            g = f["sensors"][k]
            si = SensorInfo(
                _get(g, "name"),
                set(json.loads(g.attrs["modalities_json"])),
                int(g.attrs["number"]),
                _get(g, "type_sensorlog"),
                _get(g, "lrc"),
                _get(g, "location"),
            )
            sens[si.number] = si

        sr_orig, signals = [], []
        sg = f["signals"]
        for k in sorted(sg, key=int):
            g = sg[k]
            data = np.asarray(g["data"][()], dtype=np.float64)  # f32 disk -> f64 compute
            mod = g.attrs["modality"]
            sub = _get(g, "subchannel")
            snum = int(g.attrs["sensor_number"])
            stored_sr = float(g.attrs["sr"])
            if bool(g.attrs["is_native"]):
                if emg_t_max is not None:  # EMGworks: trim superset window to target's
                    n_ref = int((emg_t_max - emg_t_min) * stored_sr) + 1
                    data = data[:n_ref]
                sr_eff = stored_sr * lf.clock_mul
                sr_targ = target_sr.get(mod)
                base = pysampled.Data(data, sr=sr_eff)
                if sr_targ is None or abs(sr_targ - sr_eff) < 1e-9:
                    out = base
                else:
                    out = base.resample(sr_targ)
                sr_orig.append(sr_eff)
            else:  # terminal snapshot / link device
                out = pysampled.Data(data, sr=stored_sr)
                sr_orig.append(stored_sr)
            signals.append(
                Signal(
                    out(),
                    out.sr,
                    t0=lf.t0,
                    meta={"sensor": sens[snum], "modality": mod, "subchannel": sub},
                )
            )

    lf.sr_orig = sr_orig
    lf.signals = _normalize_signal_lengths(signals)
    lf.sensors = lf._signals_to_sensors([sens[n] for n in sensor_order], lf.signals)
    lf.sensor_groups = {}
    return lf
