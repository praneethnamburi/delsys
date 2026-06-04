"""Event-type vocabulary: the named marker tracks an annotation session offers.

A *project*'s event types live in its ``delsys_project.toml`` (see
:mod:`delsys._project`); this module is the semantic layer over the raw
``[[event_types]]`` array — the :class:`EventType` model, defaults/coercion, and
conversion to the annotator's internal marker spec.

Each type has a **stable slug** (the id written into ``.delsys-events``), a mutable
**label** (what a rename edits — so renaming never rewrites a trial file), a single
**key** to bind for marking, a **size** (1 = point, 2 = window), and a **color**.
``noise`` is *not* an event type here — it is the built-in quality track with its
own semantics (see :mod:`delsys._noise` / :mod:`delsys._events`).
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

#: Colors cycled across types that don't pin their own ``color``.
_PALETTE = ("tab:green", "tab:blue", "tab:purple", "tab:brown", "tab:olive")


@dataclass
class EventType:
    """One named marker track.

    Attributes:
        slug: Stable identifier, written into ``.delsys-events``. Never changes on
            a rename, so renaming the ``label`` requires no file migration.
        label: Human display name (what create/rename edits).
        key: Single character bound to add a mark of this type (``alt+<key>``
            removes). Multi-char keys won't fire on a keypress — see ``TODO.md``.
        size: ``1`` (point) or ``2`` (window — two presses fix start/end).
        color: Matplotlib color for the overlay.
    """

    slug: str
    label: str = ""
    key: str = ""
    size: int = 1
    color: str = ""

    def __post_init__(self) -> None:
        self.slug = str(self.slug)
        self.label = str(self.label) if self.label else self.slug
        self.key = str(self.key) if self.key else self.slug
        self.size = int(self.size)

    def asdict(self) -> dict:
        """Plain dict for serialization into ``[[event_types]]`` (stable field order)."""
        return {
            "slug": self.slug,
            "label": self.label,
            "key": self.key,
            "size": self.size,
            "color": self.color,
        }


def default_event_types() -> List[EventType]:
    """The built-in fallback vocabulary when no project config applies: a point
    track ``"1"`` and a window track ``"2"`` (the historical default)."""
    return [
        EventType(slug="1", label="Event 1", key="1", size=1, color="tab:green"),
        EventType(slug="2", label="Event 2", key="2", size=2, color="tab:blue"),
    ]


#: The ``events=`` argument accepted by ``Log.view`` / :func:`coerce`.
EventsArg = Union[None, Dict[str, int], Sequence[Union[str, Tuple[str, int], EventType, dict]]]


def coerce(events: EventsArg) -> List[EventType]:
    """Normalize an ad-hoc ``events=`` argument into ``[EventType]``.

    Accepts ``None`` (the :func:`default_event_types`), a ``{name: size}`` mapping,
    or a sequence of names / ``(name, size)`` pairs / dicts / :class:`EventType`.
    For the lightweight forms the slug, label, and key all default to the name.
    Colors fill in from the palette where unset.
    """
    if events is None:
        return default_event_types()
    items: list = list(events.items()) if isinstance(events, dict) else list(events)
    out: List[EventType] = []
    for i, it in enumerate(items):
        if isinstance(it, EventType):
            et = it
        elif isinstance(it, dict):
            et = EventType(**it)
        elif isinstance(it, (list, tuple)):
            name, size = it[0], (it[1] if len(it) > 1 else 1)
            et = EventType(slug=str(name), size=int(size))
        else:
            et = EventType(slug=str(it))
        if not et.color:
            et.color = _PALETTE[i % len(_PALETTE)]
        out.append(et)
    return out


def from_config(config) -> List[EventType]:
    """Parse a :class:`delsys._project.ProjectConfig`'s ``[[event_types]]`` array.

    Returns ``[]`` when the config declares no event types (callers then fall back
    to :func:`default_event_types`). Palette colors fill in where a row omits one.
    """
    out: List[EventType] = []
    for i, row in enumerate(config.event_types_raw()):
        if "slug" not in row:
            continue
        et = EventType(
            slug=row["slug"],
            label=row.get("label", ""),
            key=row.get("key", ""),
            size=row.get("size", 1),
            color=row.get("color", ""),
        )
        if not et.color:
            et.color = _PALETTE[i % len(_PALETTE)]
        out.append(et)
    return out


def save_to_config(config, types: Sequence[EventType]) -> str:
    """Write ``types`` into a :class:`ProjectConfig`'s ``[[event_types]]`` and save.

    Surgical (preserves the rest of the document); returns the written path. This
    is the persistence side of interactive create/rename/remove.
    """
    config.set_event_types_raw([t.asdict() for t in types])
    return config.save()


def to_marker_specs(types: Sequence[EventType]) -> List[Tuple[str, str, str, int, str]]:
    """Convert types to the annotator's internal spec ``(slug, label, key, size, color)``."""
    return [(t.slug, t.label, t.key, t.size, t.color) for t in types]


def resolve(lf, events: EventsArg = None) -> List[EventType]:
    """Resolve the event-type vocabulary for a ``view()`` over ``lf``.

    Precedence: an explicit ``events=`` argument wins; otherwise the project config
    resolved from the Log's file (``DELSYS_PROJECT_CONFIG`` / walk-up); otherwise
    the built-in :func:`default_event_types`.
    """
    if events is not None:
        return coerce(events)
    from delsys import _project

    config = _project.ProjectConfig.load(start=getattr(lf, "fname", None))
    if config is not None:
        types = from_config(config)
        if types:
            return types
    return default_event_types()
