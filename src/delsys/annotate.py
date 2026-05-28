"""Interactive noise annotation over a :class:`delsys.Log`.

Two views onto the **same** ``<stem>.delsys-noise`` sidecar (see
:mod:`delsys._noise` for the key grammar + on-disk format):

- **signal-centric** (:func:`launch_noise_annotator` ``view="signal"``, the
  default) — a ``datanavigator`` ``SignalBrowser`` subclass: flip through every
  channel, mark windows per channel or per whole sensor+modality.
- **sensor-centric** (``view="sensor"``) — a ``datanavigator`` ``PlotBrowser``
  subclass: see one sensor's modalities (EMG / ACC / GYRO / …) as stacked,
  time-aligned subplots, and mark whole-modality windows. Built for spotting a
  blip shared across a sensor's channels (mechanical/cable artifact).

Both share :class:`_NoiseMarkingMixin` (state + `.delsys-noise` I/O + the
keypress add/remove/dead actions); each browser supplies only what's
view-specific (which address a cursor targets, and how to draw overlays).

``datanavigator`` is an *optional* dependency: it is imported only when a view
is launched (inside the per-class factories), so the delsys core stays
``datanavigator``-free.

Marking (hover the cursor at the spot, then press):

- **``1``** — add a window: two presses fix its start and end;
- **``alt+1``** — remove the window nearest the cursor;
- **``d``** (sensor view) — toggle the hovered modality dead for the whole recording;
- **Save noise** button writes the sidecar (auto-seeded from an existing one).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from matplotlib import pyplot as plt

from delsys import _noise
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


class _NoiseMarkingMixin:
    """Shared noise-marking state + actions over a ``.delsys-noise`` sidecar.

    Mixed into a ``datanavigator`` browser (which supplies ``add_key_binding`` /
    ``buttons`` / ``memoryslots`` / ``update`` / ``statevariables``). The browser
    subclass implements :meth:`_key_for_event` (which signal address a cursor
    event targets) and its own overlay drawing; everything else lives here so the
    signal- and sensor-centric views don't re-implement the marking logic.
    """

    # -- state + I/O ------------------------------------------------------

    def _init_noise_state(self, lf, path: Optional[str]) -> None:
        """Set up sidecar path + in-memory annotation. Call from ``__init__``."""
        self._lf = lf
        self._sidecar_path = path or _noise.sidecar_path_for(lf.fname)
        self._ann: Dict[str, dict] = self._load_existing()
        self._mark_buffer: list = []  # (key, x) pairs for the two-press add
        self._overlay_artists: list = []

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

    def _slot(self, key: str) -> dict:
        return self._ann.setdefault(key, {"windows": [], "dead": []})

    def save(self, event=None) -> str:
        """Write the sidecar (dropping empty entries). Returns the path."""
        payload = {
            k: v for k, v in self._ann.items() if v.get("windows") or v.get("dead")
        }
        return _noise.write_noise_sidecar(self._sidecar_path, payload)

    # -- key-addressed actions (the cores; each redraws) ------------------

    def _mark_window(self, key: str, a: float, b: float) -> None:
        a, b = float(min(a, b)), float(max(a, b))
        if b > a:
            self._slot(key)["windows"].append([a, b])
        self.update()

    def _remove_nearest(self, key: str, x: float) -> None:
        slot = self._ann.get(key)
        windows = slot["windows"] if slot else []
        if windows:
            i = min(range(len(windows)), key=lambda i: abs(sum(windows[i]) / 2 - x))
            windows.pop(i)
        self.update()

    def _toggle_dead(self, key: str) -> None:
        dead = self._slot(key)["dead"]
        if _WHOLE_EXTENT in dead:
            dead.remove(_WHOLE_EXTENT)
        else:
            dead.append(list(_WHOLE_EXTENT))
        self.update()

    def _drop_last(self, key: str) -> None:
        slot = self._ann.get(key)
        if slot and slot["windows"]:
            slot["windows"].pop()
        self.update()

    # -- cursor-driven keypress handlers (use _key_for_event) -------------

    def _mark_point(self, event=None) -> None:
        """Collect a cursor x; on the second press (same address) add the window."""
        key = self._key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is None or x is None:  # cursor not over a markable trace
            return
        if self._mark_buffer and self._mark_buffer[0][0] != key:
            self._mark_buffer = []  # second press landed on a different signal
        self._mark_buffer.append((key, float(x)))
        if len(self._mark_buffer) < 2:
            return
        (k, x0), (_, x1) = self._mark_buffer[:2]
        self._mark_buffer = []
        self._mark_window(k, x0, x1)

    def _remove_window(self, event=None) -> None:
        key = self._key_for_event(event)
        x = getattr(event, "xdata", None)
        if key is not None and x is not None:
            self._remove_nearest(key, x)

    def _toggle_dead_at(self, event=None) -> None:
        key = self._key_for_event(event)
        if key is not None:
            self._toggle_dead(key)

    def _add_noise_keybindings(self, dead_key: Optional[str] = None) -> None:
        """Wire the marking keys + Save button. Frees digit keys from memoryslots."""
        # GenericBrowser otherwise treats 1-9 as memory slots (storing
        # _current_idx); we don't use those, and need '1' for marking.
        self.memoryslots.disable()
        self.add_key_binding("1", self._mark_point, description="Add noise window (2 presses)")
        self.add_key_binding(
            "alt+1", self._remove_window, description="Remove nearest noise window"
        )
        if dead_key is not None:
            self.add_key_binding(
                dead_key, self._toggle_dead_at, description="Toggle dead (whole recording)"
            )
        self.buttons.add(text="Save noise", type_="Push", action_func=lambda e: self.save())

    # -- overlay helpers --------------------------------------------------

    def _clear_overlays(self) -> None:
        for art in self._overlay_artists:
            try:
                art.remove()
            except Exception:  # noqa: BLE001 — artist already gone (e.g. figure.clear)
                pass
        self._overlay_artists = []

    def _draw_key_spans(self, ax, key: str, color: str) -> None:
        """Shade one key's windows (``color``) + dead spans (hatched gray) on ``ax``."""
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

    # -- to be implemented by the view ------------------------------------

    def _key_for_event(self, event) -> Optional[str]:
        raise NotImplementedError


def _build_signal_annotator_class():
    """Build the signal-centric annotator (datanavigator imported here, lazily)."""
    from datanavigator.signals import SignalBrowser

    class NoiseAnnotator(_NoiseMarkingMixin, SignalBrowser):
        """SignalBrowser that marks per-signal noise into a ``.delsys-noise`` sidecar."""

        def __init__(self, lf, path: Optional[str] = None, figure_handle=None) -> None:
            self._init_noise_state(lf, path)
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
            """Whether marking targets the whole sensor+modality (coord-less)."""
            return "Mod scope" in self.buttons and self.buttons["Mod scope"].state

        def _channel_key(self) -> str:
            return self._keys[self._current_idx]

        def _modality_key(self) -> str:
            return _noise.format_signal_key(
                self._signals[self._current_idx], include_coord=False
            )

        def _current_key(self) -> str:
            return self._modality_key() if self._mod_scope else self._channel_key()

        def _key_for_event(self, event) -> Optional[str]:
            # Single axes -> the current scope (the cursor's subplot is implicit).
            return self._current_key()

        # -- public/button wrappers (event-less; act on the current scope) --

        def add_window(self, a: float, b: float) -> None:
            self._mark_window(self._current_key(), a, b)

        def toggle_dead(self, event=None) -> None:
            self._toggle_dead(self._current_key())

        def undo(self, event=None) -> None:
            self._drop_last(self._current_key())

        # -- controls + view ----------------------------------------------

        def _add_controls(self) -> None:
            self._add_noise_keybindings()
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
                # the displayed channel); dead spans hatched gray.
                self._draw_key_spans(ax, self._channel_key(), "tab:red")
                self._draw_key_spans(ax, self._modality_key(), "tab:orange")

        def _label_axes(self) -> None:
            ax = getattr(self, "_ax", None)
            if ax is None:
                return
            mod = self._signals[self._current_idx].modality
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"{mod} ({_MODALITY_UNITS.get(mod, 'a.u.')})")

    return NoiseAnnotator


def _build_sensor_annotator_class():
    """Build the sensor-centric annotator (datanavigator imported here, lazily)."""
    from datanavigator.plots import PlotBrowser

    class SensorNoiseAnnotator(_NoiseMarkingMixin, PlotBrowser):
        """PlotBrowser showing one sensor's modalities stacked; marks whole-modality
        noise into the shared ``.delsys-noise`` sidecar."""

        def __init__(self, lf, path: Optional[str] = None, figure_handle=None) -> None:
            self._init_noise_state(lf, path)
            self._sensors = [s for s in lf.sensors if self._markable_modalities(s)]
            self._subplot_axes: Dict[str, object] = {}  # modality -> Axes (per update)
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
            loc = _trim_location(sensor.location, sensor.number)
            return _noise.format_key(sensor.number, modality, None, loc)

        def _key_for_event(self, event) -> Optional[str]:
            ax = getattr(event, "inaxes", None)
            if ax is None:
                return None
            sensor = self._sensors[self._current_idx]
            for mod, mod_ax in self._subplot_axes.items():
                if mod_ax is ax:
                    return self._modality_key_for(sensor, mod)
            return None

        # -- controls + view ----------------------------------------------

        def _add_controls(self) -> None:
            self._add_noise_keybindings(dead_key="d")

        def _plot_sensor(self, sensor, figure, **kwargs) -> None:
            """Draw one sensor's modalities as stacked, time-aligned subplots."""
            mods = self._markable_modalities(sensor)
            axes = figure.subplots(len(mods), 1, sharex=True, squeeze=False)[:, 0]
            self._subplot_axes = {}
            for ax, mod in zip(axes, mods):
                bundle = getattr(sensor, _mod_to_attr(mod), None)
                if bundle is not None:
                    ax.plot(bundle.t, bundle())
                ax.set_ylabel(f"{mod} ({_MODALITY_UNITS.get(mod, 'a.u.')})")
                self._subplot_axes[mod] = ax
            axes[-1].set_xlabel("time (s)")
            figure.suptitle(self._sensor_label(sensor))

        def update(self, event=None) -> None:
            super().update(event)  # PlotBrowser: clear + _plot_sensor (sets axes) + draw
            self._clear_overlays()
            sensor = self._sensors[self._current_idx]
            for mod, ax in self._subplot_axes.items():
                self._draw_key_spans(ax, self._modality_key_for(sensor, mod), "tab:red")
            plt.draw()

    return SensorNoiseAnnotator


def launch_noise_annotator(lf, path: Optional[str] = None, view: str = "signal"):
    """Build and show a noise annotator over ``lf`` (returns the instance).

    ``view="signal"`` (default) opens the per-channel SignalBrowser; ``"sensor"``
    opens the per-sensor stacked-modality PlotBrowser. Both author the same
    ``<stem>.delsys-noise`` sidecar.
    """
    if view == "signal":
        cls = _build_signal_annotator_class()
    elif view == "sensor":
        cls = _build_sensor_annotator_class()
    else:
        raise ValueError(f"view must be 'signal' or 'sensor'; got {view!r}.")
    return cls(lf, path=path)
