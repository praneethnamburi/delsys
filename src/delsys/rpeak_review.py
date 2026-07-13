"""Interactive EKG R-peak reviewer (``EKG.review()``).

The graduated, package-native successor to the bespoke ``EKGBrowser`` that lived
in ``pn-projects/projects/gibson/gib01.py``. Opens a ``datanavigator``
``PlotBrowser`` over an EKG's channel(s) so a human can correct auto-detected
R-peaks and persist the result to the ``<stem>.delsys-events`` sidecar — from
which any later load reproduces the curated peaks (see
:meth:`delsys.ekg.EKG.load_rpeaks` and ``Log.ekg`` auto-load).

Curate-and-reload loop::

    ekg = lf.ekg          # auto-detected (and auto-applies a prior decision)
    ekg.review()          # add/remove peaks, mark noise, flip, then Save
    # ... any later session:
    peaks = lf.ekg.rpeak_times()   # the curated final peaks, reproduced

Controls (keys mirrored by buttons where noted):

- **a** — add an R-peak at the cursor (snapped per the *edit mode*).
- **d** — remove the nearest R-peak to the cursor.
- **n** — mark a noisy segment (two presses = start/end); peaks inside are
  dropped from :meth:`~delsys.ekg.EKG.rpeak_times`.
- **f** / **Flip** button — flip polarity **and re-detect** (the one edit that
  changes the detector baseline, not just the human diff).
- **m** / **Mode** — cycle the add snap: ``peak`` / ``valley`` / ``exact``.
- **1** / **2** / **3** — tag ``reviewed`` / ``representative`` / ``interesting``.
- **s** / **Save** button — write the decision(s) to ``<stem>.delsys-events``.

``datanavigator`` is imported lazily (inside the factory) so the delsys core
stays dnav-free.
"""

from typing import List, Optional

import numpy as np

#: Tag names bound to the digit keys 1 / 2 / 3.
_TAG_KEYS = {"1": "reviewed", "2": "representative", "3": "interesting"}


def _build_rpeak_reviewer_class():
    """Build the reviewer class (``datanavigator`` imported here, lazily)."""
    import matplotlib.pyplot as plt
    from datanavigator.plots import PlotBrowser

    class RPeakReviewer(PlotBrowser):
        """PlotBrowser whose browse axis is the EKG channel; edits the rpeak meta
        of the current channel and saves a per-channel decision to the sidecar."""

        def __init__(
            self,
            channels: List,
            path: Optional[str] = None,
            figure_handle=None,
        ) -> None:
            if not channels:
                raise ValueError("RPeakReviewer needs at least one EKG channel.")
            self._channels = list(channels)
            self._events_path = path
            # Snap / match windows (seconds) around the cursor.
            self._win_add = (-0.025, 0.025)
            self._win_remove = (-0.10, 0.10)
            self._noise_buffer: Optional[float] = None
            self._ax_raw = None  # rebound each draw (the shared browse x-axis)
            self._ax_ibi = None
            self._xlim = None  # preserved raw/IBI zoom across redraws

            # Seed each channel: replay a saved decision if present, else detect.
            for ch in self._channels:
                if not ch.load_rpeaks(self._events_path):
                    if len(ch.meta.get("rpeaks_idx_default", [])) == 0:
                        ch.find_rpeaks()

            if figure_handle is None:
                figure_handle = plt.figure(figsize=(15, 9))
            super().__init__(
                plot_data=self._channels,
                plot_func=self._plot,
                figure_handle=figure_handle,
                show_item_dropdown=False,  # we add our own (labelled "channel")
            )
            self._add_controls()
            self.add_item_dropdown([self._label(ch) for ch in self._channels], var_name="channel")
            # Subclass: PlotBrowser.__init__ skips the first draw for us.
            self.update()
            self.reset_axes()
            plt.show(block=False)
            print(
                f"delsys EKG review: {len(self._channels)} channel(s). "
                "a add / d remove / n noisy-segment (2 presses) / f flip+redetect / "
                "m mode / 1-3 tag / s save. Zoom persists across edits."
            )

        # -- helpers -------------------------------------------------------

        @staticmethod
        def _label(ch) -> str:
            from delsys import _rpeaks

            try:
                return _rpeaks.ekg_channel_address(ch)
            except Exception:  # noqa: BLE001 — best-effort label
                return "EKG"

        def _cur(self):
            """The EKG channel currently browsed."""
            return self.get_current_data()

        @property
        def _mode(self) -> str:
            var = getattr(self, "_mode_var", None)
            return var.current_state if var is not None else "peak"

        # -- controls ------------------------------------------------------

        def _add_controls(self) -> None:
            # Free the digit keys (GenericBrowser binds 1-9 as memory slots and
            # re-shows the widget every redraw).
            self.memoryslots.disable()
            self.memoryslots.hide()
            self.memoryslots.show = lambda *a, **k: None

            edit = "R-peak editing"
            self.add_key_binding(
                "a", self._add_rpeak,
                description="Add peak at cursor (or restore a removed one)", group=edit,
            )
            self.add_key_binding(
                "d", self._remove_rpeak,
                description="Remove nearest peak (or undo an added one)", group=edit,
            )
            self.add_key_binding(
                "n", self._mark_noise, description="Mark noisy segment (two presses)", group=edit
            )
            self.add_key_binding(
                "f", self._flip, description="Flip polarity + re-detect", group=edit
            )
            self.add_key_binding(
                "m", self._cycle_mode, description="Cycle add mode (peak/valley/exact)", group=edit
            )
            for key, tag in _TAG_KEYS.items():
                self.add_key_binding(
                    key, (lambda e=None, t=tag: self._tag(t)), description=f"Tag {tag}", group="Tags"
                )
            self.add_key_binding(
                "s", self.save, description="Save decision to sidecar", group="File"
            )

            self.buttons.add(text="Flip (f)", type_="Push", action_func=self._flip)
            self.buttons.add(text="Save (s)", type_="Push", action_func=self.save)
            self.buttons.add(text="Mode (m)", type_="Push", action_func=self._cycle_mode)
            self.buttons.add(text="Help (ctrl+k)", type_="Push", action_func=self._help)
            self._mode_var = self.statevariables.add(
                "edit mode", ["peak", "valley", "exact"], widget="dropdown"
            )

        def _help(self, event=None) -> None:
            """Open datanavigator's grouped key-binding cheatsheet (Help / ctrl+k)."""
            self.show_key_bindings()

        # -- actions -------------------------------------------------------

        def _add_rpeak(self, event=None) -> None:
            if event is None or event.xdata is None:
                return
            ch = self._cur()
            t = float(event.xdata)
            # If a currently-*removed* peak sits near the cursor, restore it (drop
            # it from the removed set) rather than adding a near-duplicate. Add is
            # the inverse of remove: re-adding what you removed should undo it.
            removed = list(ch.meta.get("rpeaks_idx_removed", []))
            if removed:
                rt = np.asarray(ch.t[np.array(removed, dtype=int)], dtype=float)
                j = int(np.argmin(np.abs(rt - t)))
                if abs(float(rt[j]) - t) <= self._win_remove[1]:
                    ch.meta["rpeaks_idx_removed"] = [r for k, r in enumerate(removed) if k != j]
                    print(f"  restored peak @ {float(rt[j]):.3f}s")
                    self.update()
                    return
            i_marked = int(np.argmin(np.abs(ch.t - t)))
            lo = max(i_marked + round(self._win_add[0] * ch.sr), 0)
            hi = min(i_marked + round(self._win_add[1] * ch.sr) + 1, len(ch))
            seg = np.asarray(ch())[lo:hi].reshape(-1)
            if self._mode == "peak" and seg.size:
                i_peak = int(np.argmax(seg)) + lo
            elif self._mode == "valley" and seg.size:
                i_peak = int(np.argmin(seg)) + lo
            else:
                i_peak = i_marked
            ch.meta.setdefault("rpeaks_idx_added", [])
            if i_peak not in ch.meta["rpeaks_idx_added"]:
                ch.meta["rpeaks_idx_added"].append(int(i_peak))
            print(f"  + peak @ {float(ch.t[i_peak]):.3f}s")
            self.update()

        def _remove_rpeak(self, event=None) -> None:
            if event is None or event.xdata is None:
                return
            ch = self._cur()
            t_marked = float(event.xdata)
            added = list(ch.meta.get("rpeaks_idx_added", []))
            default = list(ch.meta.get("rpeaks_idx_default", []))
            # Added first so a manual addition wins a tie and gets *undone* rather
            # than shadowed by a stale entry in both added and removed.
            idx = np.array(added + default, dtype=int)
            if idx.size == 0:
                return
            t_idx = np.asarray(ch.t[idx], dtype=float)
            nearest = int(idx[int(np.argmin(np.abs(t_idx - t_marked)))])
            if not (self._win_remove[0] < (t_marked - float(ch.t[nearest])) < self._win_remove[1]):
                return
            if nearest in added:
                ch.meta["rpeaks_idx_added"] = [a for a in added if a != nearest]
                print(f"  undo added peak @ {float(ch.t[nearest]):.3f}s")
            else:
                ch.meta.setdefault("rpeaks_idx_removed", [])
                if nearest not in ch.meta["rpeaks_idx_removed"]:
                    ch.meta["rpeaks_idx_removed"].append(nearest)
                print(f"  - peak @ {float(ch.t[nearest]):.3f}s")
            self.update()

        def _mark_noise(self, event=None) -> None:
            if event is None or event.xdata is None:
                return
            x = float(event.xdata)
            if self._noise_buffer is None:
                self._noise_buffer = x
                print(f"  noisy segment start @ {x:.3f}s (press n again for end)")
                return
            a, b = self._noise_buffer, x
            self._noise_buffer = None
            if b <= a:
                print("  noisy segment cancelled (end <= start)")
                return
            ch = self._cur()
            ch.meta.setdefault("noisy_segments_idx", [])
            ch.meta["noisy_segments_idx"].extend(ch._windows_to_idx_pairs([[a, b]]))
            print(f"  noisy segment {a:.3f}-{b:.3f}s")
            self.update()

        def _flip(self, event=None) -> None:
            """Flip polarity of the current channel and re-detect (button / f)."""
            self._cur().flip_signal()
            print("  flipped polarity + re-detected")
            self.update()

        def _cycle_mode(self, event=None) -> None:
            var = getattr(self, "_mode_var", None)
            if var is not None:
                var.cycle()

        def _tag(self, tag: str) -> None:
            ch = self._cur()
            ch.meta.setdefault("tags", [])
            if tag not in ch.meta["tags"]:
                ch.meta["tags"].append(tag)
                print(f"  tagged {tag}")
                self.update()

        # -- drawing -------------------------------------------------------

        def _plot(self, ch, figure, **kwargs) -> None:
            gs = figure.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.0], hspace=0.45)
            ax_raw = figure.add_subplot(gs[0, 0])
            ax_ibi = figure.add_subplot(gs[1, 0], sharex=ax_raw)
            ax_hist = figure.add_subplot(gs[2, 0])
            self._ax_raw, self._ax_ibi = ax_raw, ax_ibi

            t = np.asarray(ch.t, dtype=float)
            y = np.asarray(ch()).reshape(-1)
            ax_raw.plot(t, y, lw=0.5, color="0.4")
            default = np.asarray(ch.meta.get("rpeaks_idx_default", []), dtype=int)
            added = np.asarray(ch.meta.get("rpeaks_idx_added", []), dtype=int)
            removed = np.asarray(ch.meta.get("rpeaks_idx_removed", []), dtype=int)
            if default.size:
                ax_raw.plot(t[default], y[default], "*", color="darkorange", ms=6, label="default")
            if added.size:
                ax_raw.plot(t[added], y[added], "+", color="seagreen", ms=10, label="added")
            if removed.size:
                ax_raw.plot(t[removed], y[removed], "x", color="0.6", ms=7, label="removed")
            for a, b in ch.meta.get("noisy_segments_idx", []):
                ax_raw.axvspan(float(t[a]), float(t[b]), color="0.5", alpha=0.15)
            ax_raw.set_ylabel("EKG (mV)")
            ax_raw.legend(loc="upper right", fontsize=7)

            # IBI over time (clean peaks, noisy segments excluded).
            clean_idx = ch.rpeak_times()[0]
            rt = np.asarray(ch.t[clean_idx], dtype=float)
            if rt.size > 1:
                ax_ibi.plot(rt[1:], np.diff(rt) * 1000.0, "o-", ms=3, color="C0")
                ax_hist.hist(np.diff(rt) * 1000.0, bins=30, color="C2")
            ax_ibi.set_ylabel("IBI (ms)")
            ax_ibi.set_xlabel("time (s)")
            ax_hist.set_xlabel("IBI (ms)")
            ax_hist.set_ylabel("count")

            tags = ch.meta.get("tags", [])
            figure.suptitle(
                f"{self._label(ch)}   |   mode={self._mode}   |   "
                f"flipped={bool(ch.meta.get('is_flipped', False))}   |   "
                f"tags={','.join(tags) if tags else '-'}",
                fontsize=10,
            )
            # Persistent on-figure shortcut legend (the command-line hint stays too).
            figure.text(
                0.008, 0.004,
                "a add  ·  d remove  ·  n noise(2 presses)  ·  f flip+redetect  ·  "
                "m mode  ·  1/2/3 tag  ·  s save        Help button / ctrl+k = full list",
                fontsize=7.5, family="monospace", color="0.4", va="bottom",
            )

        def update(self, event=None) -> None:
            # Preserve the raw/IBI zoom across redraws (edits shouldn't reset the
            # view). "Auto limits" ON snaps back to the full extent.
            auto = "Auto limits" in self.buttons and self.buttons["Auto limits"].state
            if not auto and self._ax_raw is not None:
                try:
                    self._xlim = self._ax_raw.get_xlim()
                except Exception:  # noqa: BLE001 — stale axis
                    self._xlim = None
            super().update(event)
            if not auto and self._xlim is not None and self._ax_raw is not None:
                self._ax_raw.set_xlim(self._xlim)
            plt.draw()

        # -- save ----------------------------------------------------------

        def save(self, event=None) -> Optional[str]:
            """Write every channel's decision (+ noisy segments) to the sidecar.

            Each channel merges its ``rpeaks`` entry + noise windows into the shared
            ``<stem>.delsys-events``; other sections are preserved. Returns the
            written path (``None`` if no channel has a source to key on).
            """
            path = None
            for ch in self._channels:
                try:
                    path = ch.save_rpeaks(self._events_path)
                except ValueError as exc:  # no source on this channel
                    print(f"delsys EKG review: {exc}")
                    return None
            if path:
                print(f"delsys EKG review: saved -> {path}")
            return path

    return RPeakReviewer


def launch(channels: List, path: Optional[str] = None, figure_handle=None):
    """Build and show the reviewer over ``channels`` (returns the instance)."""
    cls = _build_rpeak_reviewer_class()
    return cls(channels, path=path, figure_handle=figure_handle)
