"""Interactive annotation over a :class:`delsys.Log`: noise + typed event markers.

Two views onto the **same** ``<stem>.delsys-events`` sidecar (see
:mod:`delsys._events` for the unified on-disk format; :mod:`delsys._noise` for the
signal-address key grammar shared by every track):

- **signal-centric** (:func:`launch_annotator` ``view="signal"``, the default) — a
  ``datanavigator`` ``SignalBrowser`` subclass: flip through every channel, mark
  per channel or per whole sensor+modality.
- **sensor-centric** (``view="sensor"``) — a ``datanavigator`` ``PlotBrowser``
  subclass: see one sensor's modalities (EMG / ACC / GYRO / …) as stacked,
  time-aligned subplots. Built for spotting a blip shared across a sensor's
  channels (mechanical/cable artifact).

Both share :class:`_MarkingMixin` (state + unified ``.delsys-events`` I/O + the
keypress add/remove actions for both the **noise** track and the **typed marker**
tracks); each browser supplies only what's view-specific (which address a cursor
targets, and how to draw overlays).

``datanavigator`` is an *optional* dependency: it is imported only when a view is
launched (inside the per-class factories), so the delsys core stays
``datanavigator``-free.

Two kinds of mark (hover the cursor at the spot, then press):

- **noise** — a per-signal quality mask consumed by :func:`delsys.clean`:
  ``n`` adds a window (two presses fix start + end), ``alt+n`` removes the nearest,
  ``d`` toggles the hovered address dead for the whole recording.
- **typed markers** — point (``size=1``, one press) or window (``size=2``, two
  presses) events of an arbitrary type (``"1"``, ``"2"``, …), authored *per
  signal* (the address is provenance) and consumed at analysis time as trial-level
  markers (:func:`delsys._events.collapse_markers`). Each type's bound key adds
  it; ``alt+<key>`` removes the nearest.

A single **Save** button writes every track to the one ``<stem>.delsys-events``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt

from delsys import _event_types, _events, _noise
from delsys._util import _mod_to_attr, _trim_location

#: Sentinel span for a whole-recording dead channel (open both ends).
_WHOLE_EXTENT = [None, None]

#: Conventional Trigno display units per modality, for the y-axis label. These
#: are *conventions* (not read from the file — the loader doesn't carry units);
#: adjust if a particular export differs.
_MODALITY_UNITS = {
    "EMGS": "V",
    "EMGD": "V",
    "EMGQ": "V",
    "EKG": "V",
    "ACC": "g",
    "GYRO": "deg/s",
    "FSR": "a.u.",
    "Analog": "V",
}

#: Modalities the sensor-centric view stacks as subplots, in display order.
_MARKABLE_ORDER = ("EMGS", "EMGD", "EMGQ", "EKG", "ACC", "GYRO", "FSR", "Analog")

#: Modalities shown as one subplot *per sub-channel* in the sensor view (so each
#: can be marked individually). The rest (EMGS/EKG single-trace; ACC/GYRO X/Y/Z)
#: stay as one overlaid whole-modality subplot. Sub-channel count is read from the
#: actual signals present, so e.g. a single-trace Sync (Analog) shows one panel.
_SPLIT_MODALITIES = ("EMGQ", "FSR", "Analog")

#: Internal marker spec tuple: ``(slug, label, key, size, color)``. ``slug`` keys
#: the in-memory + on-disk track; ``key`` is the char bound for marking; ``label``
#: is the display name. Built from :class:`delsys._event_types.EventType`.
MarkerSpec = Tuple[str, str, str, int, str]


def _default_marker_specs() -> List[MarkerSpec]:
    return _event_types.to_marker_specs(_event_types.default_event_types())


class _MarkingMixin:
    """Shared marking state + actions over a ``.delsys-events`` sidecar.

    Mixed into a ``datanavigator`` browser (which supplies ``add_key_binding`` /
    ``buttons`` / ``memoryslots`` / ``update`` / ``statevariables``). Holds two
    kinds of in-memory annotation, both keyed by structural signal **address**
    (label stripped, so a relabel never breaks lookup):

    - ``_ann`` — the **noise** track: ``{address: {"windows": [...], "dead": [...]}}``.
    - ``_markers`` — the **typed marker** tracks:
      ``{slug: {address: [{"seq": [...], "note": ..., "tags": [...]}, ...]}}``.

    The browser subclass implements :meth:`_key_for_event` (which noise address a
    cursor targets) and its own overlay drawing; everything common lives here so
    the signal- and sensor-centric views don't re-implement the marking logic.
    """

    # -- state + I/O ------------------------------------------------------

    def _init_marking_state(
        self, lf, path: Optional[str], marker_specs: List[MarkerSpec]
    ) -> None:
        """Set up sidecar path + in-memory annotation. Call from ``__init__``."""
        self._lf = lf
        self._events_path = path or _events.events_path_for(lf.fname)
        self._marker_specs = marker_specs
        self._ann: Dict[str, dict] = self._load_noise()
        self._markers: Dict[str, Dict[str, list]] = self._load_markers()
        self._mark_buffer: list = []  # (track, key, x) for multi-press sequences
        self._overlay_artists: list = []
        # event.key -> spec, for the marking-key dispatch (bound methods only, so
        # datanavigator's add_key_binding passes the matplotlib event through).
        self._marker_add_by_key: Dict[str, Tuple[str, int]] = {
            key: (slug, size) for slug, _label, key, size, _color in marker_specs
        }
        self._marker_remove_by_key: Dict[str, str] = {
            f"alt+{key}": slug for slug, _label, key, _size, _color in marker_specs
        }

    def _load_noise(self) -> Dict[str, dict]:
        """Seed the noise track from the unified file (or a legacy ``.delsys-noise``).

        Keyed by structural **address** (label stripped) so a sidecar written with
        any label — including an older code version's placeholder — still matches
        the signals on render. Entries that collapse to the same address are merged.
        """
        signals = _events.noise_signals_for(
            self._events_path, _noise.sidecar_path_for(self._lf.fname)
        )
        ann: Dict[str, dict] = {}
        for key, val in signals.items():
            windows, dead = _noise._normalize_signal_value(val)
            slot = ann.setdefault(_noise.key_address(key), {"windows": [], "dead": []})
            slot["windows"].extend([a, b] for a, b in windows)
            slot["dead"].extend([a, b] for a, b in dead)
        return ann

    def _load_markers(self) -> Dict[str, Dict[str, list]]:
        """Seed each marker track from the unified file, keyed by address.

        Stored as **records** (``{"seq", "note", "tags"}``) so a per-event note/tag
        on disk survives an open→save round-trip even before the notes UI lands.
        """
        out: Dict[str, Dict[str, list]] = {slug: {} for slug, *_ in self._marker_specs}
        if not os.path.exists(self._events_path):
            return out
        for slug, _label, _key, _size, _color in self._marker_specs:
            recs = _events.read_marker_records(self._events_path, slug)
            slot = out[slug]
            for key, rs in recs.items():
                slot.setdefault(_noise.key_address(key), []).extend([dict(r) for r in rs])
        return out

    def save(self, event=None) -> str:
        """Write the unified ``.delsys-events`` (dropping empty entries). Returns the path.

        Every in-memory address is re-labelled from the current Log on the way out,
        so the file stays human-readable (and a stale-labelled sidecar is self-healed).
        """
        doc: Dict[str, dict] = {}
        noise_signals = {
            _noise.relabel_key(self._lf, addr): v
            for addr, v in self._ann.items()
            if v.get("windows") or v.get("dead")
        }
        if noise_signals:
            doc[_events.NOISE_TYPE] = {"signals": noise_signals}
        for slug, _label, _key, size, _color in self._marker_specs:
            sigs = {
                _noise.relabel_key(self._lf, addr): recs
                for addr, recs in self._markers.get(slug, {}).items()
                if recs
            }
            if sigs:
                doc[slug] = {"size": size, "signals": sigs}
        path = _events.write_events(self._events_path, doc)
        n = sum(len(s.get("signals", {})) for s in doc.values())
        print(f"delsys.view: saved {n} marked address(es) across {len(doc)} type(s) -> {path}")
        return path

    # -- multi-press collection (shared by noise windows + size>=2 markers) --

    def _collect(self, track: str, size: int, key: str, x: float) -> Optional[list]:
        """Buffer ``size`` cursor presses for one ``(track, key)``; return the
        completed sequence (and clear the buffer) once full, else ``None``.

        A press on a different ``(track, key)`` resets the buffer, so an
        interrupted multi-press never splices across signals/tracks.
        """
        if self._mark_buffer and self._mark_buffer[0][:2] != (track, key):
            self._mark_buffer = []
        self._mark_buffer.append((track, key, float(x)))
        if len(self._mark_buffer) < size:
            return None
        seq = [b[2] for b in self._mark_buffer[:size]]
        self._mark_buffer = []
        return seq

    # -- scope expansion (noise view hook) --------------------------------

    def _scope_keys(self, key: str) -> List[str]:
        """Noise addresses a marking action targets, given the cursor's ``key``.

        Defaults to the single ``key`` (the signal view, whose ``Mod scope``
        toggle already resolves which single key). The sensor view overrides this
        to fan a mark across every modality of the sensor when its ``Sensor
        scope`` toggle is on. Markers do *not* use scope — a marker records the one
        signal it was placed from.
        """
        return [key]

    # -- noise track: key-addressed actions (each redraws) ----------------

    def _mark_window(self, key: str, a: float, b: float) -> None:
        a, b = float(min(a, b)), float(max(a, b))
        if b > a:
            for k in self._scope_keys(key):
                self._slot(k)["windows"].append([a, b])
        self.update()

    def _slot(self, key: str) -> dict:
        return self._ann.setdefault(key, {"windows": [], "dead": []})

    def _remove_nearest(self, key: str, x: float) -> None:
        for k in self._scope_keys(key):
            slot = self._ann.get(k)
            windows = slot["windows"] if slot else []
            if windows:
                i = min(range(len(windows)), key=lambda i: abs(sum(windows[i]) / 2 - x))
                windows.pop(i)
        self.update()

    def _toggle_dead(self, key: str) -> None:
        for k in self._scope_keys(key):
            dead = self._slot(k)["dead"]
            if _WHOLE_EXTENT in dead:
                dead.remove(_WHOLE_EXTENT)
            else:
                dead.append(list(_WHOLE_EXTENT))
        self.update()

    def _drop_last(self, key: str) -> None:
        for k in self._scope_keys(key):
            slot = self._ann.get(k)
            if slot and slot["windows"]:
                slot["windows"].pop()
        self.update()

    # -- noise track: cursor-driven keypress handlers ---------------------

    def _mark_point(self, event=None) -> None:
        """Collect a cursor x; on the second press (same address) add the noise window."""
        key = self._key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is None or x is None:  # cursor not over a markable trace
            return
        seq = self._collect("noise", 2, key, x)
        if seq is not None:
            self._mark_window(key, seq[0], seq[1])

    def _remove_window(self, event=None) -> None:
        key = self._key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is not None and x is not None:
            self._remove_nearest(key, x)

    def _toggle_dead_at(self, event=None) -> None:
        key = self._key_for_event(event)
        if key is not None:
            self._toggle_dead(key)

    # -- typed marker tracks ----------------------------------------------

    def _marker_key_for_event(self, event) -> Optional[str]:
        """Provenance address a marker is placed on. Defaults to the noise key;
        the signal view overrides to the coord-ful channel (a marker records the
        one signal it was placed from, regardless of the noise Mod-scope toggle)."""
        return self._key_for_event(event)

    def _marker_slot(self, slug: str, key: str) -> list:
        return self._markers.setdefault(slug, {}).setdefault(key, [])

    def _mark_marker(self, slug: str, size: int, event=None) -> None:
        key = self._marker_key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is None or x is None:
            return
        key = _noise.key_address(key)
        seq = self._collect(slug, size, key, x)
        if seq is not None:
            self._marker_slot(slug, key).append({"seq": seq, "note": None, "tags": []})
            self.update()

    def _remove_marker(self, slug: str, event=None) -> None:
        key = self._marker_key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is None or x is None:
            return
        recs = self._markers.get(slug, {}).get(_noise.key_address(key))
        if recs:
            i = min(range(len(recs)), key=lambda i: abs(recs[i]["seq"][0] - x))
            recs.pop(i)
            self.update()

    def _mark_marker_event(self, event=None) -> None:
        spec = self._marker_add_by_key.get(getattr(event, "key", None))
        if spec is not None:
            self._mark_marker(spec[0], spec[1], event)

    def _remove_marker_event(self, event=None) -> None:
        slug = self._marker_remove_by_key.get(getattr(event, "key", None))
        if slug is not None:
            self._remove_marker(slug, event)

    # -- keybindings ------------------------------------------------------

    def _add_marking_keybindings(self, dead_key: Optional[str] = None) -> None:
        """Wire the noise + marker keys and the Save button. Frees digit keys
        from memoryslots so they can add marker types."""
        # GenericBrowser treats 1-9 as memory slots (storing _current_idx) and
        # re-shows the widget on every redraw; disable + neutralize the re-show so
        # the slots are gone and the digits are free for marker types.
        self.memoryslots.disable()
        self.memoryslots.hide()
        self.memoryslots.show = lambda *a, **k: None
        # Noise track (letter keys, so digits stay free for marker types).
        self.add_key_binding("n", self._mark_point, description="Add noise window (2 presses)")
        self.add_key_binding("alt+n", self._remove_window, description="Remove nearest noise window")
        if dead_key is not None:
            self.add_key_binding(
                dead_key, self._toggle_dead_at, description="Toggle dead (whole recording)"
            )
        # Typed marker tracks (key add / alt+key remove; bound-method dispatch).
        for slug, label, key, size, _color in self._marker_specs:
            kind = "point" if size == 1 else "window"
            self.add_key_binding(
                key, self._mark_marker_event, description=f"Add {label} ({kind})"
            )
            self.add_key_binding(
                f"alt+{key}", self._remove_marker_event, description=f"Remove nearest {label}"
            )
        self.buttons.add(text="Save", type_="Push", action_func=lambda e: self.save())

    # -- overlay helpers --------------------------------------------------

    def _clear_overlays(self) -> None:
        for art in self._overlay_artists:
            try:
                art.remove()
            except Exception:  # noqa: BLE001 — artist already gone (e.g. figure.clear)
                pass
        self._overlay_artists = []

    def _draw_key_spans(self, ax, key: str, color: str) -> None:
        """Shade one address's noise windows (``color``) + dead spans (hatched gray)."""
        slot = self._ann.get(key)
        if not slot:
            return
        x0, x1 = ax.get_xlim()
        for a, b in slot.get("windows", []):
            self._overlay_artists.append(ax.axvspan(a, b, alpha=0.2, color=color))
        for a, b in slot.get("dead", []):
            lo = x0 if a is None else a
            hi = x1 if b is None else b
            self._overlay_artists.append(
                ax.axvspan(lo, hi, alpha=0.15, color="gray", hatch="xx")
            )

    def _draw_marker_spans(self, ax, key: str) -> None:
        """Draw every marker track on ``ax`` for one address: size-1 as a dashed
        vertical line, size>=2 as a translucent span, in the track's color."""
        for slug, _label, _k, size, color in self._marker_specs:
            for rec in self._markers.get(slug, {}).get(key, []):
                seq = rec["seq"]
                if size >= 2 and len(seq) >= 2:
                    self._overlay_artists.append(
                        ax.axvspan(seq[0], seq[1], alpha=0.15, color=color)
                    )
                elif seq:
                    self._overlay_artists.append(
                        ax.axvline(seq[0], color=color, lw=1.2, ls="--")
                    )

    # -- to be implemented by the view ------------------------------------

    def _key_for_event(self, event) -> Optional[str]:
        raise NotImplementedError


def _build_signal_annotator_class():
    """Build the signal-centric annotator (datanavigator imported here, lazily)."""
    from datanavigator.signals import SignalBrowser

    class SignalAnnotator(_MarkingMixin, SignalBrowser):
        """SignalBrowser that marks noise + typed events into a ``.delsys-events`` sidecar."""

        def __init__(
            self,
            lf,
            path: Optional[str] = None,
            figure_handle=None,
            marker_specs: Optional[List[MarkerSpec]] = None,
        ) -> None:
            self._init_marking_state(lf, path, marker_specs or _default_marker_specs())
            self._signals = list(lf.signals)
            # Dropdown labels = the coord-ful signal address for each channel.
            self._keys: List[str] = [_noise.format_signal_key(s) for s in self._signals]
            if figure_handle is None:
                figure_handle = plt.figure(figsize=(14, 5))
            super().__init__(
                plot_data=self._signals,
                signal_names=self._keys,
                titlefunc=lambda s: s._keys[s._current_idx],
                figure_handle=figure_handle,
            )
            self._add_controls()
            if "Auto limits" in self.buttons:
                self.buttons["Auto limits"].set_state(True)
            self.update()

        # -- scope / current key ------------------------------------------

        @property
        def _mod_scope(self) -> bool:
            """Whether noise marking targets the whole sensor+modality (coord-less)."""
            return "Mod scope" in self.buttons and self.buttons["Mod scope"].state

        def _channel_key(self) -> str:
            # Address (label stripped); self._keys keeps the label for display.
            return _noise.key_address(self._keys[self._current_idx])

        def _modality_key(self) -> str:
            return _noise.key_address(
                _noise.format_signal_key(self._signals[self._current_idx], include_coord=False)
            )

        def _current_key(self) -> str:
            return self._modality_key() if self._mod_scope else self._channel_key()

        def _key_for_event(self, event) -> Optional[str]:
            # Single axes -> the current noise scope (the cursor's subplot is implicit).
            return self._current_key()

        def _marker_key_for_event(self, event) -> Optional[str]:
            # Markers always record the coord-ful channel (provenance), ignoring Mod scope.
            return self._channel_key()

        # -- public/button wrappers (event-less; act on the current scope) --

        def add_window(self, a: float, b: float) -> None:
            self._mark_window(self._current_key(), a, b)

        def toggle_dead(self, event=None) -> None:
            self._toggle_dead(self._current_key())

        def undo(self, event=None) -> None:
            self._drop_last(self._current_key())

        # -- controls + view ----------------------------------------------

        def _add_controls(self) -> None:
            self._add_marking_keybindings()
            self.buttons.add(text="Toggle dead", type_="Push", action_func=self.toggle_dead)
            self.buttons.add(text="Undo window", type_="Push", action_func=self.undo)
            # Scope toggle: False -> channel (coord-ful), True -> sensor+modality.
            self.buttons.add(text="Mod scope", type_="Toggle", start_state=False)

        def update(self, event=None) -> None:
            super().update(event)
            self._label_axes()
            self._clear_overlays()
            ax = getattr(self, "_ax", None)
            if ax is not None:
                # channel windows in red, whole-modality in orange (both affect
                # the displayed channel); dead spans hatched gray; markers per track.
                self._draw_key_spans(ax, self._channel_key(), "tab:red")
                self._draw_key_spans(ax, self._modality_key(), "tab:orange")
                self._draw_marker_spans(ax, self._channel_key())

        def _label_axes(self) -> None:
            ax = getattr(self, "_ax", None)
            if ax is None:
                return
            mod = self._signals[self._current_idx].modality
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"{mod} ({_MODALITY_UNITS.get(mod, 'a.u.')})")

    return SignalAnnotator


def _build_sensor_annotator_class():
    """Build the sensor-centric annotator (datanavigator imported here, lazily)."""
    from datanavigator.plots import PlotBrowser

    class SensorAnnotator(_MarkingMixin, PlotBrowser):
        """PlotBrowser showing one sensor's modalities stacked into time-aligned
        subplots, marking noise + typed events into the shared ``.delsys-events``
        sidecar.

        EMGQ / FSR / Analog get one subplot **per sub-channel** (so each Quattro
        channel, FSR pad, or Sync line — and a Sync that carries only one line — can
        be marked individually); EMGS/EKG (single trace) and ACC/GYRO (X/Y/Z
        overlaid) stay as one whole-modality subplot. A sensor that mixes, say, EMGQ
        with ACC/GYRO shows all of them. Marking targets the hovered subplot's
        address; the **Sensor scope** toggle instead fans a *noise* mark across
        every modality of the sensor (a wall-clock noise burst hits them all).
        """

        def __init__(
            self,
            lf,
            path: Optional[str] = None,
            figure_handle=None,
            marker_specs: Optional[List[MarkerSpec]] = None,
        ) -> None:
            self._init_marking_state(lf, path, marker_specs or _default_marker_specs())
            self._sensors = [s for s in lf.sensors if self._markable_modalities(s)]
            self._panel_axes: List = []  # [(Axes, key)] per update
            item_names = [self._sensor_label(s) for s in self._sensors]
            if figure_handle is None:
                figure_handle = plt.figure(figsize=(14, 8))
            # No setup_func -> PlotBrowser clears + redraws each sensor, which
            # accommodates sensors with different modality sets.
            super().__init__(
                plot_data=self._sensors,
                plot_func=self._plot_sensor,
                figure_handle=figure_handle,
                show_item_dropdown=False,  # we add our own (labelled by sensor)
            )
            self._add_controls()
            self.add_item_dropdown(item_names, var_name="sensor")
            if "Auto limits" in self.buttons:
                self.buttons["Auto limits"].set_state(True)
            # Subclass: PlotBrowser.__init__ skips the first draw for us.
            self.update()
            self.reset_axes()
            plt.show(block=False)

        # -- sensor / modality helpers ------------------------------------

        @staticmethod
        def _markable_modalities(sensor) -> List[str]:
            have = set(sensor.modalities)
            return [m for m in _MARKABLE_ORDER if m in have]

        @staticmethod
        def _sensor_label(sensor) -> str:
            return f"{sensor.number} | {_trim_location(sensor.location, sensor.number)}"

        @staticmethod
        def _modality_key_for(sensor, modality: str) -> str:
            # Whole-modality address (label-free); save() re-labels from the Log.
            return _noise.format_key(sensor.number, modality, None)

        def _panel_specs(self, sensor):
            """Build the stacked-panel layout: ``[(key, ylabel, [(t, y), ...]), ...]``.

            Split modalities contribute one spec per present sub-channel (keyed by
            its coord); the rest contribute one whole-modality spec (coord-less key,
            all sub-channels overlaid).
            """
            specs = []
            for mod in self._markable_modalities(sensor):
                unit = _MODALITY_UNITS.get(mod, "a.u.")
                if mod in _SPLIT_MODALITIES:
                    sigs = [s for s in self._lf.signals if s.matches(sensor.number, mod, None)]
                    for sig in sigs:
                        key = _noise.format_key(sensor.number, mod, sig.subchannel)
                        specs.append((key, f"{mod}.{sig.subchannel}\n({unit})", [(sig.t, sig())]))
                else:
                    bundle = getattr(sensor, _mod_to_attr(mod), None)
                    traces = [(bundle.t, bundle())] if bundle is not None else []
                    specs.append((self._modality_key_for(sensor, mod), f"{mod} ({unit})", traces))
            return specs

        def _key_for_event(self, event) -> Optional[str]:
            ax = getattr(event, "inaxes", None)
            if ax is None:
                return None
            for pax, key in self._panel_axes:
                if pax is ax:
                    return key
            return None

        # -- scope --------------------------------------------------------

        @property
        def _sensor_scope(self) -> bool:
            """Whether a noise mark fans across the whole sensor (all its modalities)."""
            return "Sensor scope" in self.buttons and self.buttons["Sensor scope"].state

        def _scope_keys(self, key: str) -> List[str]:
            if not self._sensor_scope:
                return [key]
            sensor = self._sensors[self._current_idx]
            return [self._modality_key_for(sensor, m) for m in self._markable_modalities(sensor)]

        # -- controls + view ----------------------------------------------

        def _add_controls(self) -> None:
            self._add_marking_keybindings(dead_key="d")
            # Off -> mark the hovered sub-channel/modality; On -> the whole sensor.
            self.buttons.add(text="Sensor scope", type_="Toggle", start_state=False)

        def _plot_sensor(self, sensor, figure, **kwargs) -> None:
            """Draw one sensor's per-sub-channel / per-modality stacked subplots."""
            specs = self._panel_specs(sensor)
            axes = figure.subplots(max(len(specs), 1), 1, sharex=True, squeeze=False)[:, 0]
            self._panel_axes = []
            for ax, (key, ylabel, traces) in zip(axes, specs):
                for t, y in traces:
                    ax.plot(t, y, lw=0.7)
                ax.set_ylabel(ylabel)
                self._panel_axes.append((ax, key))
            axes[-1].set_xlabel("time (s)")
            figure.suptitle(self._sensor_label(sensor))

        def update(self, event=None) -> None:
            super().update(event)  # PlotBrowser: clear + _plot_sensor (sets axes) + draw
            self._clear_overlays()
            for ax, key in self._panel_axes:
                # Own (coord-ful or whole-modality) windows in red; for a
                # sub-channel panel also show the whole-modality windows (orange).
                self._draw_key_spans(ax, key, "tab:red")
                pk = _noise.parse_key(key)
                if pk.coord is not None:
                    self._draw_key_spans(
                        ax, _noise.format_key(pk.sensor, pk.modality), "tab:orange"
                    )
                self._draw_marker_spans(ax, _noise.key_address(key))
            plt.draw()

    return SensorAnnotator


def launch_annotator(
    lf, path: Optional[str] = None, view: str = "signal", events=None
):
    """Build and show an annotator over ``lf`` (returns the instance).

    ``view="signal"`` (default) opens the per-channel SignalBrowser; ``"sensor"``
    opens the per-sensor stacked-modality PlotBrowser. Both author the same
    ``<stem>.delsys-events`` sidecar (noise + the marker tracks). The marker
    vocabulary is resolved by :func:`delsys._event_types.resolve` — an explicit
    ``events=`` wins, else the project config, else the built-in default.
    """
    marker_specs = _event_types.to_marker_specs(_event_types.resolve(lf, events))
    if view == "signal":
        cls = _build_signal_annotator_class()
    elif view == "sensor":
        cls = _build_sensor_annotator_class()
    else:
        raise ValueError(f"view must be 'signal' or 'sensor'; got {view!r}.")
    return cls(lf, path=path, marker_specs=marker_specs)


#: Back-compat alias — :func:`launch_annotator` is the canonical name.
launch_noise_annotator = launch_annotator
