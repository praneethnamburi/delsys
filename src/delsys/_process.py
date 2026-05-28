"""Batch conversion of Delsys CSVs to native HDF5 checkpoints, with smart
channelmap resolution.

``process()`` mirrors the shape of ``telemed.process``: walk a source for CSVs,
convert each to a self-contained ``.h5`` checkpoint (idempotently), and return a
``{path: status}`` dict. The hard part is picking the right channelmap per CSV:
a folder may hold several (``delsys_channelmap.txt`` plus
``delsys_channelmap_Trial_5_10.txt`` overrides for a sensor-config change), they
may live a level above the CSVs, and they can be misnamed or stale. So each
candidate is **content-checked** — its ``Ch n`` set vs the CSV's actual channel
numbers — and, under the default ``strict`` policy, a CSV whose name-chosen map
doesn't match is flagged and skipped rather than written with a doubtful map.
"""

import csv as _csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from delsys._constants import TARGET_SR
from delsys._metadata import SensorLog
from delsys._parse import _fix_corrupted_sensor_names, _parse_hdr, _parse_sig_name, _read_sensor_log

#: ``delsys_channelmap_Trial_5_10.txt`` -> applies to trials 5..10 (inclusive).
_TRIAL_RANGE_RE = re.compile(r"_Trial_(\d+)_(\d+)$")
_CHANNELMAP_GLOB = "delsys_channelmap*.txt"


def read_channelmap(path: str) -> Tuple[List[SensorLog], Dict[str, str]]:
    """Parse an (extended) channelmap file.

    The base format is the hand-rolled ``Ch <n> - <type> - <location>`` channelmap
    (see :func:`delsys._parse._read_sensor_log`). An optional trailing section::

        [sensor_name_replace]
        Avanti Sensor 1 (88016) = EMG 01 03558 (88016)

    carries the acquisition-time name corrections that used to live in project code.
    Old readers ignore it (the lines have no ``" - "`` separator), so the format is
    backward-compatible.

    Returns:
        ``(sensor_map, sensor_name_replace)``.
    """
    sensor_map = _read_sensor_log(path)
    name_replace: Dict[str, str] = {}
    in_section = False
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                in_section = s[1:-1].strip().lower() == "sensor_name_replace"
                continue
            if in_section and "=" in s:
                old, new = s.split("=", 1)
                name_replace[old.strip()] = new.strip()
    return sensor_map, name_replace


def _csv_channel_numbers(csv_path: str, sensor_name_replace: Optional[Dict[str, str]]) -> set:
    """The set of Delsys sensor numbers present in a CSV — from the header only."""
    hdr = _parse_hdr(csv_path, sensor_name_replace)
    app = hdr["application"]
    if app == "EMGworks":
        with open(csv_path, newline="") as f:
            cols = next(_csv.reader(f))
        cols = _fix_corrupted_sensor_names(cols, sensor_name_replace)
        sig_names = [c for c in cols if c and "X[s]" not in c]
    else:
        sig_names = [c for c in hdr["sensor_signal_names"] if "Time Series" not in c]
    nums = set()
    for sn in sig_names:
        try:
            nums.add(_parse_sig_name(sn, app, TARGET_SR).sensor_number)
        except Exception:  # pragma: no cover - defensive: skip unparseable columns
            pass
    return nums


def _csv_trial_number(csv_path: str) -> Optional[int]:
    """Trailing integer of the stem (``Trial_5`` -> 5), or ``None`` if absent."""
    tail = Path(csv_path).stem.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def _find_channelmaps(csv_path: str, search_parents: int) -> List[str]:
    """Candidate channelmaps in the CSV folder and up to ``search_parents`` levels up."""
    found: List[str] = []
    parents = Path(csv_path).resolve().parents
    for level in range(min(search_parents, len(parents) - 1) + 1):
        for p in sorted(glob.glob(str(parents[level] / _CHANNELMAP_GLOB))):
            if p not in found:
                found.append(p)
    return found


def _name_pick(csv_path: str, candidates: List[str]) -> Optional[str]:
    """Filename-hint pick: a matching ``_Trial_A_B`` override, else the plain default."""
    trial_no = _csv_trial_number(csv_path)
    default = None
    for cm in candidates:
        stem = Path(cm).stem
        m = _TRIAL_RANGE_RE.search(stem)
        if m:
            if trial_no is not None and int(m.group(1)) <= trial_no <= int(m.group(2)):
                return cm
        elif default is None:  # first non-trial-range map is the default
            default = cm
    return default


def _resolve_channelmap(
    csv_path: str,
    candidates: List[str],
    policy: str,
    nr_override: Optional[Dict[str, str]],
) -> Tuple[Optional[str], str]:
    """Pick the channelmap for ``csv_path``. Returns ``(path_or_None, status)``.

    ``status`` is ``"ok"`` (use the path) or a ``"flag: ..."`` / ``"warn: ..."``
    string. Under ``strict`` a flag means skip (path is ``None``).
    """
    pick = _name_pick(csv_path, candidates)
    if policy == "name_only":
        return pick, "ok" if pick else "flag: no channelmap found"

    def compatible(cm: str) -> bool:
        sensor_map, nr_map = read_channelmap(cm)
        nr = nr_override if nr_override is not None else nr_map
        return _csv_channel_numbers(csv_path, nr) == {sl.number for sl in sensor_map}

    if pick is not None and compatible(pick):
        return pick, "ok"

    matches = [cm for cm in candidates if compatible(cm)]
    pick_name = Path(pick).name if pick else "none"
    if len(matches) == 1:
        chosen = Path(matches[0]).name
        if policy == "lenient":
            return matches[0], f"warn: name-pick ({pick_name}) mismatched; used {chosen}"
        return None, f"flag: name-pick ({pick_name}) mismatches CSV channels; {chosen} matches"
    if not matches:
        return None, "flag: no channelmap matches the CSV's channels"
    return None, f"flag: {len(matches)} channelmaps match (ambiguous): " + ", ".join(
        Path(m).name for m in matches
    )


def _gather_csvs(
    source: Union[str, Path, Iterable[Union[str, Path]]], recursive: bool
) -> List[str]:
    """Expand ``source`` into a sorted list of Delsys ``*.csv`` paths."""
    if isinstance(source, (str, Path)):
        sources = [source]
    else:
        sources = list(source)
    out: List[str] = []
    skip = ("archive", "clips", "export")
    for s in sources:
        p = Path(s)
        if p.is_dir():
            it = p.rglob("*.csv") if recursive else p.glob("*.csv")
            for f in it:
                if not any(part.lower() in skip for part in f.parts):
                    out.append(str(f))
        elif p.suffix.lower() == ".csv":
            out.append(str(p))
    return sorted(set(out))


def process(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    channelmap: Optional[Union[str, List[SensorLog]]] = None,
    sensor_name_replace: Optional[Dict[str, str]] = None,
    channelmap_search_parents: int = 1,
    channelmap_policy: str = "strict",
    recursive: bool = True,
    skip_existing: bool = True,
    overwrite: bool = False,
    report: bool = True,
) -> Dict[str, str]:
    """Batch-convert Delsys CSVs to native ``.h5`` checkpoints.

    For each CSV under ``source``, resolve its channelmap (unless ``channelmap`` is
    given explicitly), then write ``<stem>.h5`` via :func:`delsys.to_native_h5`. The
    checkpoint is target-independent, so no ``target_sr`` is needed here — choose it
    later at ``Log(h5, target_sr=...)``.

    Args:
        source: A CSV path, a folder (walked for ``*.csv``), or an iterable of either.
        channelmap: Explicit channelmap (path or pre-parsed ``SensorLog`` list) used
            for *every* CSV; skips per-file discovery.
        sensor_name_replace: Acquisition-typo corrections; overrides any
            ``[sensor_name_replace]`` block found in a channelmap sidecar.
        channelmap_search_parents: How many parent folders above each CSV to search
            for ``delsys_channelmap*.txt`` (default 1 — the common "map one level up").
        channelmap_policy: ``"strict"`` (default) flags+skips a CSV whose name-chosen
            map doesn't match its channels; ``"lenient"`` picks the content-matching
            map and warns; ``"name_only"`` trusts the filename without a content check.
        recursive: Walk subfolders when ``source`` is a directory.
        skip_existing: Skip a CSV whose ``.h5`` already exists.
        overwrite: Rebuild even if the ``.h5`` exists.
        report: Write a ``delsys_process_report.txt`` per output folder.

    Returns:
        ``{csv_path: status}`` where status is ``"built"``, ``"hit"`` (already
        existed), ``"skipped: <reason>"``, or ``"error: <msg>"``.
    """
    from delsys.log import to_native_h5

    if channelmap_policy not in ("strict", "lenient", "name_only"):
        raise ValueError(
            f"channelmap_policy must be strict/lenient/name_only, got {channelmap_policy!r}"
        )

    results: Dict[str, str] = {}
    for csv_path in _gather_csvs(source, recursive):
        out_h5 = os.path.splitext(csv_path)[0] + ".h5"
        if skip_existing and os.path.exists(out_h5) and not overwrite:
            results[csv_path] = "hit"
            continue

        # Resolve the channelmap (path or explicit) + corrections for this CSV.
        if channelmap is not None:
            sm: Optional[Union[str, List[SensorLog]]] = channelmap
            nr = sensor_name_replace
        else:
            candidates = _find_channelmaps(csv_path, channelmap_search_parents)
            if not candidates:
                sm, nr = None, sensor_name_replace  # no map -> auto-build from CSV
            else:
                cm_path, status = _resolve_channelmap(
                    csv_path, candidates, channelmap_policy, sensor_name_replace
                )
                if cm_path is None:
                    results[csv_path] = f"skipped: {status}"
                    continue
                sm, nr_map = read_channelmap(cm_path)
                nr = sensor_name_replace if sensor_name_replace is not None else nr_map

        try:
            to_native_h5(csv_path, out_h5, sensor_map=sm, sensor_name_replace=nr)
            results[csv_path] = "built"
        except Exception as e:  # keep batch going; record the failure
            results[csv_path] = f"error: {type(e).__name__}: {e}"

    if report:
        _write_reports(results)
    return results


def _write_reports(results: Dict[str, str]) -> None:
    """Write a per-folder ``delsys_process_report.txt`` summarizing statuses."""
    by_folder: Dict[str, List[Tuple[str, str]]] = {}
    for csv_path, status in results.items():
        by_folder.setdefault(str(Path(csv_path).parent), []).append((Path(csv_path).name, status))
    for folder, rows in by_folder.items():
        lines = [f"{name} - {status}" for name, status in sorted(rows)]
        with open(os.path.join(folder, "delsys_process_report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
