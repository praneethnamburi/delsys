"""Per-project delsys configuration: ``delsys_project.toml``.

A project (a study / data collection) keeps its delsys settings in a
``delsys_project.toml`` committed in the project repo. It is the home for any
*per-project* setting — today the **event-type vocabulary** (see
:mod:`delsys._event_types`); later things like default ``target_sr`` or
channelmap policy. The file is TOML (not ``.py``) because the annotator *writes*
the event-type section interactively, and TOML round-trips machine edits cleanly;
``tomlkit`` is used so a write **preserves comments and the rest of the document**,
surgically touching only the table it changed.

Resolution order for "which project am I in" (:func:`find_project_config`):

1. the ``DELSYS_PROJECT_CONFIG`` environment variable, if it points at a file;
2. otherwise walk up from a start path (a trial file or folder) looking for a
   ``delsys_project.toml``;
3. otherwise ``None`` — callers fall back to built-in defaults.

File shape::

    [settings]
    # target_sr = { EMGS = 2000.0, ACC = 148.1 }   # wiring deferred

    [[event_types]]
    slug = "movement-onset"   # stable id written into .delsys-events
    label = "Movement onset"  # rename edits this only — no file migration
    key = "1"                 # single char to bind for marking
    size = 1                  # 1 = point, 2 = window
    color = "tab:green"
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tomlkit

#: Conventional filename of a project's delsys config.
PROJECT_CONFIG_NAME = "delsys_project.toml"

#: Environment variable naming an explicit project config path (wins over walk-up).
ENV_VAR = "DELSYS_PROJECT_CONFIG"


def find_project_config(start: Optional[Union[str, "os.PathLike"]] = None) -> Optional[str]:
    """Resolve the active ``delsys_project.toml`` path, or ``None``.

    ``DELSYS_PROJECT_CONFIG`` (when it points at an existing file) wins; otherwise
    walk up from ``start`` (a trial file or folder; defaults to the cwd) until a
    ``delsys_project.toml`` is found.
    """
    env = os.environ.get(ENV_VAR)
    if env and os.path.isfile(env):
        return env
    base = Path(start) if start is not None else Path.cwd()
    if base.is_file():
        base = base.parent
    base = base.resolve()
    for folder in (base, *base.parents):
        candidate = folder / PROJECT_CONFIG_NAME
        if candidate.is_file():
            return str(candidate)
    return None


class ProjectConfig:
    """A loaded ``delsys_project.toml`` document (thin wrapper over a tomlkit doc).

    Owns the file I/O + raw section access; the *meaning* of the ``[[event_types]]``
    array is layered on by :mod:`delsys._event_types` (which reads/writes it through
    this object), so this module stays free of event-type semantics.
    """

    def __init__(self, path: str, doc: "tomlkit.TOMLDocument") -> None:
        self.path = path
        self.doc = doc

    @classmethod
    def load(
        cls,
        path: Optional[Union[str, "os.PathLike"]] = None,
        *,
        start: Optional[Union[str, "os.PathLike"]] = None,
    ) -> Optional["ProjectConfig"]:
        """Load an explicit ``path``, or the one resolved from ``start`` — ``None``
        when no project config applies."""
        if path is None:
            path = find_project_config(start)
        if path is None:
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = tomlkit.parse(f.read())
        return cls(str(path), doc)

    @property
    def settings(self) -> Dict[str, Any]:
        """The ``[settings]`` table as a plain dict (empty when absent)."""
        return dict(self.doc.get("settings", {}))

    def event_types_raw(self) -> List[dict]:
        """The raw ``[[event_types]]`` array as a list of plain dicts."""
        return [dict(t) for t in self.doc.get("event_types", [])]

    def set_event_types_raw(self, types: List[dict]) -> None:
        """Replace the ``[[event_types]]`` array (surgical: other tables/comments
        in the document are untouched). Call :meth:`save` to persist."""
        array = tomlkit.aot()  # array-of-tables
        for t in types:
            table = tomlkit.table()
            for key, val in t.items():
                table[key] = val
            array.append(table)
        self.doc["event_types"] = array

    def save(self) -> str:
        """Write the document back to :attr:`path` (preserving comments/order)."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(self.doc))
        return self.path


def scaffold(path: Union[str, "os.PathLike"], *, event_types: Optional[List[dict]] = None) -> str:
    """Write a starter ``delsys_project.toml`` at ``path`` and return it.

    Used to seed a new project's config. ``event_types`` defaults to the template
    in :mod:`delsys._event_types` (resolved lazily to avoid an import cycle).
    """
    if event_types is None:
        from delsys import _event_types

        event_types = [t.asdict() for t in _event_types.default_event_types()]

    doc = tomlkit.document()
    doc.add(tomlkit.comment(" delsys per-project configuration. Committed with the project."))
    settings = tomlkit.table()
    settings.add(tomlkit.comment(" target_sr = { EMGS = 2000.0, ACC = 148.1 }  # wiring deferred"))
    doc["settings"] = settings

    array = tomlkit.aot()
    for t in event_types:
        table = tomlkit.table()
        for key, val in t.items():
            table[key] = val
        array.append(table)
    doc["event_types"] = array

    with open(path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))
    return str(path)
