"""Batch EMG/EKG-artifact cleaning of native HDF5 checkpoints, with a per-log
decision sidecar for deterministic re-runs.

``clean()`` mirrors the shape of :func:`delsys.process`: walk a source for raw
``Trial_*.h5`` checkpoints, clean each into a ``Trial_*_cleaned.h5`` terminal
snapshot (idempotently), write a per-trial PDF report next to the checkpoint, and
return a ``{path: status}`` dict. The two kinds of cleaning have separate owners,
each with its own per-log sidecar that travels with the ``.h5``:

- ``<stem>.delsys-events`` — *human* annotations: noise-window / dead-channel marks
  (the ``noise`` type) plus typed marker tracks, authored in the annotator and
  merely *consumed* here (see :mod:`delsys._events`). A legacy per-signal
  ``<stem>.delsys-noise`` (see :mod:`delsys._noise`) is still read when no unified
  file is present.
- ``<stem>.delsys-artifact`` — the *algorithmic* cleaning decision (ICA components
  to remove, which cleaned variant to splice, the motion pairing, an optional
  noise reference, and the rest of the :class:`delsys.CleaningConfig` knob set).

The reproducibility contract is ``cleaned.h5 = f(raw.h5, <stem>.delsys-artifact)``.
The raw checkpoint is immutable; on the first pass a trial with no ``.delsys-artifact``
sidecar is cleaned with auto-detection and its *resolved* decision is frozen into a
fresh sidecar; subsequent passes replay that frozen decision, so a re-run reproduces
the cleaned checkpoint bit-for-bit (the FastICA fit is seeded, and replaying the
auto-chosen components with auto-detection off reconstructs the same signal). Edit
the sidecar — e.g. swap the auto-chosen IC after eyeballing the PDF, or via the
interactive :meth:`delsys.Log.clean` — and re-run with ``overwrite=True`` to
regenerate. A per-folder ``delsys_cleaning_report.txt`` summarizes a run (an
overview, not the source of truth).
"""

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from delsys.cleaning import CleaningConfig

#: Composite suffix of the per-log cleaning-decision sidecar. Composite (not
#: ``.json``) so it isn't swept by portfolio ``*.json`` tooling — same convention
#: as ``<stem>.delsys-noise`` and datanavigator's ``.dnav-toc``.
ARTIFACT_SUFFIX = ".delsys-artifact"

#: Bump when the decision-sidecar layout changes incompatibly.
ARTIFACT_SCHEMA = 1

#: Suffix of a cleaned checkpoint (so the walk can skip its own outputs).
CLEANED_SUFFIX = "_cleaned.h5"

#: Per-folder run-summary filename (an overview report, not the source of truth).
REPORT_NAME = "delsys_cleaning_report.txt"

#: ``CleaningConfig`` fields stored at the decision's *top level* (not inside
#: ``config``) because they encode the human decision the sidecar exists for.
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
# Per-log decision sidecar I/O + decision (de)serialization
# ---------------------------------------------------------------------------


def decision_path_for(checkpoint: Union[str, Path]) -> str:
    """Sibling ``<stem>.delsys-artifact`` path for a checkpoint ``.h5``."""
    return os.path.splitext(str(checkpoint))[0] + ARTIFACT_SUFFIX


def read_decision(checkpoint: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read a checkpoint's ``<stem>.delsys-artifact`` decision (``None`` if absent).

    Returns the inner decision dict (the ``ecg_components_to_remove`` /
    ``splice_source`` / ``motion`` / ``noise_event_ref`` / ``accept`` / ``config``
    record). Tolerates a bare decision dict written without the ``{schema, cleaning}``
    envelope.
    """
    path = decision_path_for(checkpoint)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        return None
    if "cleaning" in doc:
        return doc["cleaning"]
    # Bare decision dict (no envelope) — tolerate it.
    return {k: v for k, v in doc.items() if k != "schema"}


def write_decision(checkpoint: Union[str, Path], decision: Dict[str, Any]) -> str:
    """Write ``decision`` to ``<stem>.delsys-artifact`` (stable order). Returns the path."""
    path = decision_path_for(checkpoint)
    doc = {"schema": ARTIFACT_SCHEMA, "cleaning": decision}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
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
    """Resolve a decision's ``noise_event_ref`` into ``(json_path, trial_key)``.

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


def _default_noise_ref(target: str) -> Optional[str]:
    """Sibling annotation sidecar to consume by default (as a basename), or ``None``.

    Prefers the unified ``<stem>.delsys-events`` when it carries a ``noise`` type;
    falls back to a legacy ``<stem>.delsys-noise``. Stored as provenance in the
    decision's ``noise_event_ref`` so a replay re-consumes the same marks.
    """
    from delsys import _events, _noise

    events_path = _events.events_path_for(target)
    if os.path.exists(events_path) and _events.read_noise_signals(events_path):
        return os.path.basename(events_path)
    legacy = _noise.sidecar_path_for(target)
    if os.path.exists(legacy):
        return os.path.basename(legacy)
    return None


def _apply_noise_ref(lf, noise_ref: Any, raw_h5: str) -> int:
    """Mask ``lf`` from a decision's ``noise_event_ref``, dispatching by suffix.

    ``.delsys-events`` -> unified file's noise type; ``.delsys-noise`` -> legacy
    per-signal sidecar; anything else -> a trial-keyed datanavigator Event JSON.
    Returns the number of signals touched (0 when the ref resolves to nothing).
    """
    from delsys import _events, _noise

    npath, nkey = _resolve_noise_ref(noise_ref, raw_h5)
    if not npath or not os.path.exists(npath):
        return 0
    if npath.endswith(_events.EVENTS_SUFFIX):
        return _events.apply_events_noise(lf, npath)
    if npath.endswith(_noise.SIDECAR_SUFFIX):
        return _noise.apply_noise_sidecar(lf, npath)
    return _noise.apply_noise_events(lf, npath, nkey)


def upsert_decision(
    checkpoint: Union[str, Path],
    *,
    components: Iterable[int],
    config: CleaningConfig,
    splice_source: str = "combined",
    motion: Any = "auto",
    noise_ref: Any = None,
    accept: Optional[bool] = True,
    mark_stale: bool = True,
) -> str:
    """Write (or replace) a checkpoint's ``<stem>.delsys-artifact`` decision.

    The interactive write-back behind :meth:`delsys.Log.clean`'s Save: it
    records an *explicit* IC-removal set so a later :func:`clean` (``overwrite=True``,
    or after the stale snapshot is cleared) replays exactly what the reviewer
    previewed. The decision shape matches :func:`_freeze_decision`'s — auto-detection
    is forced off on replay (``ecg_components_to_remove`` lives at the decision's top
    level, the :data:`_SELECTION_FIELDS` contract) and the stored ``config`` body
    supplies the rest of the knobs.

    Args:
        checkpoint: Path to the raw ``Trial_*.h5``; the decision is written to the
            sibling ``<stem>.delsys-artifact``.
        components: IC indices to remove (stored verbatim, sorted/de-duplicated).
        config: The :class:`CleaningConfig` the preview used; its body (minus the
            selection fields) is stored.
        splice_source: ``"combined"`` / ``"ekgonly"`` / ``"motiononly"``.
        motion: Motion pairing (``"auto"`` / ``{emg_sensor: target}`` dict / ``None``).
        noise_ref: Stored ``noise_event_ref``. ``None`` (default) points at the
            sibling unified ``<stem>.delsys-events`` (its noise type) when one
            exists, else a legacy ``<stem>.delsys-noise``, so :func:`clean`
            re-consumes the same human noise windows.
        accept: Stored ``accept`` flag. ``True`` (default) marks the trial reviewed;
            ``False`` blocks regeneration until flipped (see :func:`_clean_one`).
        mark_stale: When ``True`` (default), delete a sibling ``*_cleaned.h5`` so the
            next :func:`clean` regenerates from this decision without ``overwrite=True``.

    Returns:
        The ``<stem>.delsys-artifact`` path written.
    """
    if splice_source not in ("combined", "ekgonly", "motiononly"):
        raise ValueError(
            f"splice_source must be combined/ekgonly/motiononly, got {splice_source!r}"
        )

    checkpoint = str(checkpoint)

    if noise_ref is None:
        noise_ref = _default_noise_ref(checkpoint)

    decision = {
        "ecg_components_to_remove": sorted({int(c) for c in components}),
        "splice_source": splice_source,
        "motion": motion,
        "noise_event_ref": noise_ref,
        "accept": accept,
        "config": _config_body(config),
    }

    path = write_decision(checkpoint, decision)

    if mark_stale:
        out_h5 = _cleaned_path(checkpoint)
        if os.path.exists(out_h5):
            os.remove(out_h5)

    return path


# ---------------------------------------------------------------------------
# Per-trial worker
# ---------------------------------------------------------------------------


def _clean_one(
    raw_h5: str,
    *,
    target_sr: Optional[Dict[str, Optional[float]]],
    base_config: CleaningConfig,
    motion: Any,
    splice_source: str,
    skip_existing: bool,
    overwrite: bool,
    generate_pdf: bool,
    record_decisions: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Clean one raw checkpoint, reading/writing its own ``.delsys-artifact`` sidecar.

    Reads the sibling decision sidecar (when ``record_decisions``): if present it is
    *replayed*; otherwise the trial is cleaned with the batch defaults and its
    resolved decision is *frozen* into a fresh sidecar. Returns ``(status, info)``;
    ``info`` carries report detail.
    """
    from delsys.log import Log

    entry = read_decision(raw_h5) if record_decisions else None
    # A reviewer who marked this trial's cleaning bad (``accept: false`` in the
    # decision sidecar, after eyeballing the PDF) blocks regeneration until they
    # fix the decision and flip it back — this overrides ``overwrite``.
    if entry is not None and entry.get("accept") is False:
        return "skipped: rejected (accept=false)", {}

    out_h5 = _cleaned_path(raw_h5)
    if skip_existing and os.path.exists(out_h5) and not overwrite:
        return "hit", {}

    try:
        # Always load a fresh Log we own — clean_emg_ekg_artifact mutates in
        # place, so we must never touch a Log a caller still holds.
        lf = Log(raw_h5, target_sr=target_sr, clock_mul=1.0, t0=0.0)

        if lf.emg is None:
            return "skipped: no EMG bundle", {}

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

        # Noise hook: consume human-authored noise windows. An explicit decision
        # noise_event_ref wins; otherwise default to a sibling annotation sidecar
        # (the unified <stem>.delsys-events noise type, else legacy
        # <stem>.delsys-noise) when present, recorded as provenance so a replay
        # re-consumes the same marks. Consumption dispatches by suffix
        # (see _apply_noise_ref).
        if not noise_ref:
            noise_ref = _default_noise_ref(raw_h5)

        noise_touched = _apply_noise_ref(lf, noise_ref, raw_h5) if noise_ref else 0

        result = lf.clean_emg_ekg_artifact(
            config=cfg,
            motion=this_motion,
            in_place=True,
            generate_report=generate_pdf,
            splice_source=this_splice,
        )
        lf.to_hdf5(out_h5)

        # Freeze a fresh decision sidecar for a first-pass trial so the next run
        # replays it (the reproducibility contract).
        if is_new and record_decisions:
            write_decision(
                raw_h5,
                _freeze_decision(result, base_config, this_motion, this_splice, noise_ref),
            )

        info = {
            "components": [
                int(c)
                for c in ((result.diagnostics.get("ecg") or {}).get("components_removed") or [])
            ],
            "splice": this_splice,
            "noise_touched": noise_touched,
        }
        return "cleaned", info
    except Exception as e:  # keep the batch going; record the failure
        return f"error: {type(e).__name__}: {e}", {}


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
        with open(os.path.join(folder, REPORT_NAME), "w") as f:
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
    record_decisions: bool = True,
    report: bool = True,
    generate_pdf: bool = True,
    progress: bool = True,
) -> Dict[str, str]:
    """Batch-clean Delsys native ``.h5`` checkpoints into ``*_cleaned.h5`` snapshots.

    For each raw ``Trial_*.h5`` under ``source``, load it (resampling to
    ``target_sr``), run :meth:`delsys.Log.clean_emg_ekg_artifact`, write a
    terminal-snapshot ``<stem>_cleaned.h5`` plus a ``<stem>_cleaning_report.pdf``
    next to the checkpoint, and record the decision in a sibling
    ``<stem>.delsys-artifact``. See the module docstring for the reproducibility
    contract.

    Args:
        source: A ``.h5`` path, a folder (walked for raw ``*.h5``), or an iterable
            of either. ``*_cleaned.h5`` outputs are skipped automatically.
        target_sr: Per-modality target rates for loading the raw checkpoint
            (``None`` uses :data:`delsys.TARGET_SR`). The cleaned snapshot is
            written at these rates.
        config: Base :class:`delsys.CleaningConfig` for trials with *no*
            ``.delsys-artifact`` sidecar (defaults to ``CleaningConfig()``). Trials
            with a sidecar replay that decision's config instead — that is the
            reproducibility contract, so ``config`` does not override an existing
            decision.
        motion: Default motion (ACC) pairing for new trials — ``"auto"`` (default),
            a ``{emg_sensor: acc_target}`` dict, or ``None`` to skip the motion
            stage. A trial's existing decision carries its own.
        splice_source: Default spliced variant for new trials — ``"combined"``
            (default), ``"ekgonly"``, or ``"motiononly"``. A trial's existing
            decision carries its own.
        recursive: Walk subfolders when ``source`` is a directory.
        skip_existing: Skip a trial whose ``*_cleaned.h5`` already exists.
        overwrite: Re-clean even if the cleaned checkpoint exists (use after
            editing a ``.delsys-artifact`` sidecar).
        record_decisions: Read/write the per-log ``<stem>.delsys-artifact`` sidecars.
            When ``False``, every trial is cleaned with the ``config`` / ``motion`` /
            ``splice_source`` defaults and nothing is read or recorded.
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
            if bar is not None:
                bar.set_description(Path(raw_h5).stem)
            status, info = _clean_one(
                raw_h5,
                target_sr=target_sr,
                base_config=base_config,
                motion=motion,
                splice_source=splice_source,
                skip_existing=skip_existing,
                overwrite=overwrite,
                generate_pdf=generate_pdf,
                record_decisions=record_decisions,
            )
            results[raw_h5] = status
            infos[raw_h5] = info
            if progress and (status.startswith("skipped") or status.startswith("error")):
                emit(f"  {Path(raw_h5).name}: {status}")
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    if progress and h5s:
        print(f"delsys.clean: {_summary(results)}", flush=True)
    if report:
        _write_reports(results, infos)
    return results
