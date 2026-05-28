"""Interactive per-signal noise annotation over a :class:`delsys.Log`.

A datanavigator ``SignalBrowser`` subclass that marks noise windows / dead spans
**per signal** and writes them to a ``<stem>.delsys-noise`` sidecar (see
:mod:`delsys._noise` for the key grammar + on-disk format).

datanavigator is an *optional* dependency: this module is imported only when
:meth:`delsys.Log.annotate_noise` runs (which lazy-imports it), so the delsys
core stays datanavigator-free — same posture as the ``_noise`` consume hook.
The subclass itself is built inside :func:`_build_noise_annotator_class` so even
importing this module doesn't touch datanavigator until launch.

Marking model (v1 — noise authoring; interactive cleaning is a follow-up):

- Browse the Log's per-channel signals; the sidebar dropdown lists them by their
  structural key ``"<sensor>.<modality>.<coord> | <label>"``.
- Drag-select a time span to mark a noise window on the current signal. The
  **"Mod scope"** toggle decides whether the window is recorded against the
  channel (coord-ful key) or the whole sensor+modality (coord-less key) — both
  are needed (a single dead axis vs a cable bump that hits every axis).
- **"Toggle dead"** marks the current scope dead for the whole recording.
- **"Undo window"** drops the last window on the current scope.
- **"Save noise"** writes the sidecar (also offered as ``save()`` for scripting).

Existing windows/dead spans affecting the displayed channel are shaded: the
channel's own windows in red, the whole-modality windows in orange, dead spans
in hatched gray.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from matplotlib.widgets import SpanSelector

from delsys import _noise

#: Sentinel span for a whole-recording dead channel (open both ends).
_WHOLE_EXTENT = [None, None]


def _build_noise_annotator_class():
    """Build the ``NoiseAnnotator`` class (datanavigator imported here, lazily)."""
    from datanavigator.signals import SignalBrowser

    class NoiseAnnotator(SignalBrowser):
        """SignalBrowser that marks per-signal noise into a ``.delsys-noise`` sidecar."""

        def __init__(self, lf, path: Optional[str] = None, figure_handle=None) -> None:
            self._lf = lf
            self._sidecar_path = path or _noise.sidecar_path_for(lf.fname)
            self._signals = list(lf.signals)
            # Dropdown labels = the coord-ful signal address for each channel.
            self._keys: List[str] = [_noise.format_signal_key(s) for s in self._signals]
            # In-memory annotation: {key: {"windows": [[a,b],...], "dead": [...]}}.
            self._ann: Dict[str, dict] = self._load_existing()
            self._overlay_artists: list = []
            self._spanselector = None
            # Must precede super().__init__: SignalBrowser.__init__ calls update(),
            # which our override extends to draw overlays from the state above.
            # titlefunc shows the structural key (with location) -- the default
            # falls back to "Plot number <i>" because a per-channel Signal has no
            # ``name`` attribute.
            super().__init__(
                plot_data=self._signals,
                signal_names=self._keys,
                titlefunc=lambda s: s._keys[s._current_idx],
                figure_handle=figure_handle,
            )
            self._add_controls()
            self.update()

        # -- annotation state -------------------------------------------------

        def _load_existing(self) -> Dict[str, dict]:
            """Seed the in-memory state from an existing sidecar, if present."""
            ann: Dict[str, dict] = {}
            if os.path.exists(self._sidecar_path):
                doc = _noise.read_noise_sidecar(self._sidecar_path)
                for key, val in (doc.get("signals") or {}).items():
                    windows, dead = _noise._normalize_signal_value(val)
                    ann[key] = {
                        "windows": [[a, b] for a, b in windows],
                        "dead": [[a, b] for a, b in dead],
                    }
            return ann

        @property
        def _mod_scope(self) -> bool:
            """Whether marking targets the whole sensor+modality (coord-less)."""
            return "Mod scope" in self.buttons and self.buttons["Mod scope"].state

        def _channel_key(self) -> str:
            """Coord-ful key of the currently displayed channel."""
            return self._keys[self._current_idx]

        def _modality_key(self) -> str:
            """Coord-less (whole sensor+modality) key for the current signal.

            Labelled with the trimmed body location (not the per-channel name),
            since the window applies to every sub-channel of the modality.
            """
            return _noise.format_signal_key(
                self._signals[self._current_idx], include_coord=False
            )

        def _current_key(self) -> str:
            """Key the current marking targets, per the scope toggle."""
            return self._modality_key() if self._mod_scope else self._channel_key()

        def _slot(self, key: str) -> dict:
            return self._ann.setdefault(key, {"windows": [], "dead": []})

        # -- marking actions --------------------------------------------------

        def add_window(self, a: float, b: float) -> None:
            """Record a noise window ``[a, b]`` on the current scope."""
            a, b = float(min(a, b)), float(max(a, b))
            if b <= a:
                return  # zero-width drag (a click) — ignore
            self._slot(self._current_key())["windows"].append([a, b])
            self.update()

        def toggle_dead(self, event=None) -> None:
            """Toggle whole-recording dead on the current scope."""
            dead = self._slot(self._current_key())["dead"]
            if _WHOLE_EXTENT in dead:
                dead.remove(_WHOLE_EXTENT)
            else:
                dead.append(list(_WHOLE_EXTENT))
            self.update()

        def undo(self, event=None) -> None:
            """Drop the last window on the current scope."""
            slot = self._ann.get(self._current_key())
            if slot and slot["windows"]:
                slot["windows"].pop()
            self.update()

        def save(self, event=None) -> str:
            """Write the sidecar (dropping empty entries). Returns the path."""
            payload = {
                k: v for k, v in self._ann.items() if v.get("windows") or v.get("dead")
            }
            return _noise.write_noise_sidecar(self._sidecar_path, payload)

        # -- UI wiring --------------------------------------------------------

        def _add_controls(self) -> None:
            # Drag-to-mark a window on the trace axes.
            self._spanselector = SpanSelector(
                self._ax,
                lambda a, b: self.add_window(a, b),
                "horizontal",
                useblit=False,
            )
            self.buttons.add(
                text="Save noise", type_="Push", action_func=lambda e: self.save()
            )
            self.buttons.add(text="Toggle dead", type_="Push", action_func=self.toggle_dead)
            self.buttons.add(text="Undo window", type_="Push", action_func=self.undo)
            # Scope toggle: False -> channel (coord-ful), True -> sensor+modality.
            self.buttons.add(text="Mod scope", type_="Toggle", start_state=False)

        # -- overlays ---------------------------------------------------------

        def update(self, event=None) -> None:
            super().update(event)
            self._redraw_overlays()

        def _redraw_overlays(self) -> None:
            ax = getattr(self, "_ax", None)
            if ax is None:
                return
            for art in self._overlay_artists:
                try:
                    art.remove()
                except Exception:  # noqa: BLE001 — artist already gone on a redraw
                    pass
            self._overlay_artists = []

            ch_key = self._channel_key()
            mod_key = self._modality_key()
            x0, x1 = ax.get_xlim()
            # Channel-scoped windows (red) + whole-modality windows (orange) both
            # affect the displayed channel; dead spans are hatched gray.
            for key, color in ((ch_key, "tab:red"), (mod_key, "tab:orange")):
                slot = self._ann.get(key)
                if not slot:
                    continue
                for a, b in slot.get("windows", []):
                    self._overlay_artists.append(ax.axvspan(a, b, alpha=0.2, color=color))
                for a, b in slot.get("dead", []):
                    lo = x0 if a is None else a
                    hi = x1 if b is None else b
                    self._overlay_artists.append(
                        ax.axvspan(lo, hi, alpha=0.15, color="gray", hatch="xx")
                    )

    return NoiseAnnotator


def launch_noise_annotator(lf, path: Optional[str] = None):
    """Build and show a :class:`NoiseAnnotator` over ``lf`` (returns the instance)."""
    annotator_cls = _build_noise_annotator_class()
    return annotator_cls(lf, path=path)
