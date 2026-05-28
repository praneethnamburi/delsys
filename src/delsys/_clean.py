"""Batch EMG/EKG-artifact cleaning of native HDF5 checkpoints, with a per-folder
decisions manifest for deterministic re-runs.

``clean()`` mirrors the shape of :func:`delsys.process`: walk a source for raw
``Trial_*.h5`` checkpoints, clean each into a ``Trial_*_cleaned.h5`` terminal
snapshot (idempotently), write a per-trial PDF report next to the checkpoint, and
return a ``{path: status}`` dict. The two kinds of cleaning have separate owners:
the algorithmic ECG / motion suppression here is :meth:`delsys.Log.clean_emg_ekg_artifact`;
human noise-window marking is authored in datanavigator and merely *consumed*
(see :mod:`delsys._noise`).

The reproducibility contract is ``cleaned.h5 = f(raw.h5, manifest)``. The raw
checkpoint is immutable; every per-trial decision — the ICA components to remove,
which cleaned variant to splice, the motion pairing, an optional noise-Event
reference, and the rest of the :class:`delsys.CleaningConfig` knob set — lives in
a per-folder ``delsys_cleaning.json`` keyed by trial id (the checkpoint stem). On
the first pass a trial with no manifest entry is cleaned with auto-detection and
its *resolved* decision is frozen into the manifest; subsequent passes replay
that frozen decision, so a re-run reproduces the cleaned checkpoint bit-for-bit
(the FastICA fit is seeded, and replaying the auto-chosen components with
auto-detection off reconstructs the same signal). Edit the manifest — e.g. swap
the auto-chosen IC after eyeballing the PDF, or point a trial at a noise Event —
and re-run with ``overwrite=True`` to regenerate.
"""

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from delsys.cleaning import CleaningConfig

#: Per-folder decisions manifest filename.
MANIFEST_NAME = "delsys_cleaning.json"

#: Bump when the manifest layout changes incompatibly.
MANIFEST_SCHEMA = 1

#: Suffix of a cleaned checkpoint (so the walk can skip its own outputs).
CLEANED_SUFFIX = "_cleaned.h5"

#: ``CleaningConfig`` fields stored at the manifest's *top level* (not inside
#: ``config``) because they encode the human decision the manifest exists for.
#: On apply they always win: auto-detection is forced off and the frozen
#: component list is used verbatim.
_SELECTION_FIELDS = ("ecg_components_to_remove", "ecg_auto_remove_components")


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def _is_cleaned(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith(CLEANED_SUFFIX)


def _cleaned_path(raw_h5: str) -> str:
    """``Trial_5.h5`` -> ``Trial_5_cleaned.h5``."""
    return os.path.splitext(raw_h5)[0] + CLEANED_SUFFIX


def _gather_h5s(source: Union[str, Path, Iterable[Union[str, Path]]], recursive: bool) -> List[str]:
    """Expand ``source`` into a sorted list of *raw* ``.h5`` checkpoints.

    ``*_cleaned.h5`` outputs are excluded so a re-run over the same folder never
    tries to clean its own results.
    """
    if isinstance(source, (str, Path)):
        sources = [source]
    else:
        sources = list(source)
    out: List[str] = []
    skip = ("archive", "clips", "export")
    for s in sources:
        p = Path(s)
        if p.is_dir():
            it = p.rglob("*.h5") if recursive else p.glob("*.h5")
            for f in it:
                if _is_cleaned(f):
                    continue
                if any(part.lower() in skip for part in f.parts):
                    continue
                out.append(str(f))
        elif p.suffix.lower() in (".h5", ".hdf5") and not _is_cleaned(p):
            out.append(str(p))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Manifest I/O + decision (de)serialization
# ---------------------------------------------------------------------------


def read_manifest(folder: Union[str, Path]) -> Dict[str, Any]:
    """Read a folder's ``delsys_cleaning.json`` (an empty manifest if absent)."""
    path = os.path.join(str(folder), MANIFEST_NAME)
    if not os.path.exists(path):
        return {"schema": MANIFEST_SCHEMA, "trials": {}}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("schema", MANIFEST_SCHEMA)
    manifest.setdefault("trials", {})
    return manifest


def write_manifest(folder: Union[str, Path], manifest: Dict[str, Any]) -> str:
    """Write ``manifest`` to a folder's ``delsys_cleaning.json`` (stable order)."""
    path = os.path.join(str(folder), MANIFEST_NAME)
    manifest.setdefault("schema", MANIFEST_SCHEMA)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _config_body(cfg: CleaningConfig) -> Dict[str, Any]:
    """Serialize a ``CleaningConfig`` minus the two selection fields."""
    body = asdict(cfg)
    for field in _SELECTION_FIELDS:
        body.pop(field, None)
    return body


def _coerce_motion(motion: Any) -> Union[str, Dict[int, Union[int, str]], None]:
    """Undo JSON's stringification of a motion dict's keys / numeric values."""
    if not isinstance(motion, dict):
        return motion  # "auto" / None / other
    out: Dict[int, Union[int, str]] = {}
    for k, v in motion.items():
        key = int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
        val = int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v
        out[key] = val
    return out


def _config_from_decision(decision: Dict[str, Any]) -> CleaningConfig:
    """Rebuild the exact ``CleaningConfig`` a frozen decision replays with.

    The stored ``config`` body supplies every knob except the two selection
    fields; auto-detection is forced *off* and the frozen component list is used
    verbatim, so the replay reconstructs the first pass's output bit-for-bit.
    """
    body = dict(decision.get("config") or {})
    for field in _SELECTION_FIELDS:
        body.pop(field, None)  # defensive: a stale hand-edit can't fight the top level
    return CleaningConfig(
        **body,
        ecg_auto_remove_components=False,
        ecg_components_to_remove=[int(c) for c in (decision.get("ecg_components_to_remove") or [])],
    )


def _freeze_decision(
    result, base_cfg: CleaningConfig, motion: Any, splice_source: str, noise_ref: Any
) -> Dict[str, Any]:
    """Capture the resolved per-trial decision after an auto (first-pass) run."""
    ecg = result.diagnostics.get("ecg") or {}
    components = [int(c) for c in (ecg.get("components_removed") or [])]
    return {
        "ecg_components_to_remove": components,
        "splice_source": splice_source,
        "motion": motion,
        "noise_event_ref": noise_ref,
        "accept": None,
        "config": _config_body(base_cfg),
    }


def _resolve_noise_ref(noise_ref: Any, raw_h5: str) -> Tuple[Optional[str], Any]:
    """Resolve a manifest ``noise_event_ref`` into ``(json_path, trial_key)``.

    ``noise_event_ref`` is either a ``{"path": ..., "key": ...}`` object (``key``
    is the datanavigator trial-id tuple, e.g. ``"(2, 14, 17)"``) or a bare path
    string (``key`` defaults to the checkpoint stem). A relative path resolves
    against the checkpoint's folder.
    """
    folder = os.path.dirname(os.path.abspath(raw_h5))
    if isinstance(noise_ref, dict):
        path = noise_ref.get("path")
        key = noise_ref.get("key", Path(raw_h5).stem)
    else:
        path = noise_ref
        key = Path(raw_h5).stem
    if path and not os.path.isabs(path):
        path = os.path.normpath(os.path.join(folder, path))
    return path, key


# ---------------------------------------------------------------------------
# Per-trial worker
# ---------------------------------------------------------------------------


def _clean_one(
    raw_h5: str,
    *,
    trial_manifest: Optional[Dict[str, Any]],
    target_sr: Optional[Dict[str, Optional[float]]],
    base_config: CleaningConfig,
    motion: Any,
    splice_source: str,
    skip_existing: bool,
    overwrite: bool,
    generate_pdf: bool,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Clean one raw checkpoint.

    Returns ``(status, frozen_decision_or_None, info)``. ``frozen_decision`` is
    non-None only for a first-pass (no-manifest-entry) trial that was cleaned —
    the caller folds it into the folder manifest. ``info`` carries report detail.
    """
    from delsys.log import Log

    entry = trial_manifest
    # A reviewer who marked this trial's cleaning bad (``accept: false`` in the
    # manifest, after eyeballing the PDF) blocks regeneration until they fix the
    # decision and flip it back — this overrides ``overwrite``.
    if entry is not None and entry.get("accept") is False:
        return "skipped: rejected (accept=false)", None, {}

    out_h5 = _cleaned_path(raw_h5)
    if skip_existing and os.path.exists(out_h5) and not overwrite:
        return "hit", None, {}

    try:
        # Always load a fresh Log we own — clean_emg_ekg_artifact mutates in
        # place, so we must never touch a Log a caller still holds.
        lf = Log(raw_h5, target_sr=target_sr, clock_mul=1.0, t0=0.0)

        if lf.emg is None:
            return "skipped: no EMG bundle", None, {}

        if entry is not None:
            cfg = _config_from_decision(entry)
            this_motion = _coerce_motion(entry.get("motion", "auto"))
            this_splice = entry.get("splice_source", "combined")
            noise_ref = entry.get("noise_event_ref")
            is_new = False
        else:
            cfg = base_config
            this_motion = motion
            this_splice = splice_source
            noise_ref = None
            is_new = True

        # Noise hook: consume human-authored noise windows. An explicit manifest
        # noise_event_ref wins; otherwise default to a sibling
        # <stem>.delsys-noise sidecar when present (recorded as provenance so a
        # replay re-consumes the same sidecar). Consumption dispatches by
        # suffix: a .delsys-noise path is the per-signal sidecar; anything else
        # is a trial-keyed datanavigator Event JSON.
        from delsys import _noise

        if not noise_ref:
            sidecar = _noise.sidecar_path_for(raw_h5)
            if os.path.exists(sidecar):
                noise_ref = os.path.basename(sidecar)

        noise_touched = 0
        if noise_ref:
            npath, nkey = _resolve_noise_ref(noise_ref, raw_h5)
            if npath and npath.endswith(_noise.SIDECAR_SUFFIX):
                if os.path.exists(npath):
                    noise_touched = _noise.apply_noise_sidecar(lf, npath)
            elif npath and os.path.exists(npath):
                noise_touched = _noise.apply_noise_events(lf, npath, nkey)

        result = lf.clean_emg_ekg_artifact(
            config=cfg,
            motion=this_motion,
            in_place=True,
            generate_report=generate_pdf,
            splice_source=this_splice,
        )
        lf.to_hdf5(out_h5)

        info = {
            "components": [
                int(c)
                for c in ((result.diagnostics.get("ecg") or {}).get("components_removed") or [])
            ],
            "splice": this_splice,
            "noise_touched": noise_touched,
        }
        frozen = (
            _freeze_decision(result, base_config, this_motion, this_splice, noise_ref)
            if is_new
            else None
        )
        return "cleaned", frozen, info
    except Exception as e:  # keep the batch going; record the failure
        return f"error: {type(e).__name__}: {e}", None, {}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summary(results: Dict[str, str]) -> str:
    """``cleaned X | hit Y | skipped Z | error W`` over a results dict."""
    c = Counter(v.split(":", 1)[0] for v in results.values())
    return " | ".join(f"{k} {c.get(k, 0)}" for k in ("cleaned", "hit", "skipped", "error"))


def _write_reports(results: Dict[str, str], infos: Dict[str, Dict[str, Any]]) -> None:
    """Write a per-folder ``delsys_cleaning_report.txt`` summarizing statuses."""
    by_folder: Dict[str, List[str]] = {}
    for raw_h5, status in results.items():
        name = Path(raw_h5).name
        info = infos.get(raw_h5, {})
        detail = ""
        if status == "cleaned":
            detail = f" (ecg={info.get('components', [])}, splice={info.get('splice')}"
            if info.get("noise_touched"):
                detail += f", noise_masked={info['noise_touched']}"
            detail += ")"
        by_folder.setdefault(str(Path(raw_h5).parent), []).append(f"{name} - {status}{detail}")
    for folder, lines in by_folder.items():
        with open(os.path.join(folder, "delsys_cleaning_report.txt"), "w") as f:
            f.write("\n".join(sorted(lines)) + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clean(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    target_sr: Optional[Dict[str, Optional[float]]] = None,
    config: Optional[CleaningConfig] = None,
    motion: Any = "auto",
    splice_source: str = "combined",
    recursive: bool = True,
    skip_existing: bool = True,
    overwrite: bool = False,
    manifest: bool = True,
    report: bool = True,
    generate_pdf: bool = True,
    progress: bool = True,
) -> Dict[str, str]:
    """Batch-clean Delsys native ``.h5`` checkpoints into ``*_cleaned.h5`` snapshots.

    For each raw ``Trial_*.h5`` under ``source``, load it (resampling to
    ``target_sr``), run :meth:`delsys.Log.clean_emg_ekg_artifact`, write a
    terminal-snapshot ``<stem>_cleaned.h5`` plus a ``<stem>_cleaning_report.pdf``
    next to the checkpoint, and record the decision in a per-folder
    ``delsys_cleaning.json``. See the module docstring for the reproducibility
    contract.

    Args:
        source: A ``.h5`` path, a folder (walked for raw ``*.h5``), or an iterable
            of either. ``*_cleaned.h5`` outputs are skipped automatically.
        target_sr: Per-modality target rates for loading the raw checkpoint
            (``None`` uses :data:`delsys.TARGET_SR`). The cleaned snapshot is
            written at these rates.
        config: Base :class:`delsys.CleaningConfig` for trials with *no* manifest
            entry (defaults to ``CleaningConfig()``). Trials with a manifest entry
            replay that entry's config instead — that is the reproducibility
            contract, so ``config`` does not override an existing decision.
        motion: Default motion (ACC) pairing for new trials — ``"auto"`` (default),
            a ``{emg_sensor: acc_target}`` dict, or ``None`` to skip the motion
            stage. Manifest entries carry their own.
        splice_source: Default spliced variant for new trials — ``"combined"``
            (default), ``"ekgonly"``, or ``"motiononly"``. Manifest entries carry
            their own.
        recursive: Walk subfolders when ``source`` is a directory.
        skip_existing: Skip a trial whose ``*_cleaned.h5`` already exists.
        overwrite: Re-clean even if the cleaned checkpoint exists (use after
            editing the manifest).
        manifest: Read/write the per-folder ``delsys_cleaning.json``. When
            ``False``, every trial is cleaned with the ``config`` / ``motion`` /
            ``splice_source`` defaults and nothing is recorded.
        report: Write a ``delsys_cleaning_report.txt`` per output folder.
        generate_pdf: Write the per-trial ``<stem>_cleaning_report.pdf``.
        progress: Print a triage line + a per-file progress bar (tqdm if
            available) + a final summary, surfacing skips/errors as they happen.

    Returns:
        ``{raw_h5_path: status}`` where status is ``"cleaned"``, ``"hit"`` (the
        cleaned checkpoint already existed), ``"skipped: <reason>"``, or
        ``"error: <msg>"``.
    """
    base_config = CleaningConfig() if config is None else config
    if not isinstance(base_config, CleaningConfig):
        raise TypeError("config must be a CleaningConfig instance.")
    if splice_source not in ("combined", "ekgonly", "motiononly"):
        raise ValueError(
            f"splice_source must be combined/ekgonly/motiononly, got {splice_source!r}"
        )

    h5s = _gather_h5s(source, recursive)
    if progress:
        if not h5s:
            print(
                f"delsys.clean: no raw .h5 checkpoints found under {source!r} "
                f"(recursive={recursive}). Nothing to do.",
                flush=True,
            )
        else:
            print(f"delsys.clean: {len(h5s)} checkpoint(s) under {source!r}.", flush=True)

    manifests: Dict[str, Dict[str, Any]] = {}
    dirty: set = set()

    def _folder_manifest(folder: str) -> Dict[str, Any]:
        if folder not in manifests:
            manifests[folder] = (
                read_manifest(folder) if manifest else {"schema": MANIFEST_SCHEMA, "trials": {}}
            )
        return manifests[folder]

    bar = None
    if progress and h5s:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(h5s), unit="file", leave=True)
        except ImportError:
            bar = None
    emit = bar.write if bar is not None else (lambda m: print(m, flush=True))

    results: Dict[str, str] = {}
    infos: Dict[str, Dict[str, Any]] = {}
    try:
        for raw_h5 in h5s:
            folder = str(Path(raw_h5).parent)
            trial_id = Path(raw_h5).stem
            m = _folder_manifest(folder)
            entry = m["trials"].get(trial_id) if manifest else None

            if bar is not None:
                bar.set_description(trial_id)
            status, frozen, info = _clean_one(
                raw_h5,
                trial_manifest=entry,
                target_sr=target_sr,
                base_config=base_config,
                motion=motion,
                splice_source=splice_source,
                skip_existing=skip_existing,
                overwrite=overwrite,
                generate_pdf=generate_pdf,
            )
            results[raw_h5] = status
            infos[raw_h5] = info
            if frozen is not None and manifest:
                m["trials"][trial_id] = frozen
                dirty.add(folder)
            if progress and (status.startswith("skipped") or status.startswith("error")):
                emit(f"  {Path(raw_h5).name}: {status}")
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    if manifest:
        for folder in dirty:
            write_manifest(folder, manifests[folder])
    if progress and h5s:
        print(f"delsys.clean: {_summary(results)}", flush=True)
    if report:
        _write_reports(results, infos)
    return results
