"""Interactive ICA-cleaning tool over a :class:`delsys.Log`.

One window, one entry point (:meth:`delsys.Log.clean`) for the whole ECG/ICA
cleaning decision — it replaces the old read-only ``CleaningResult.review`` /
``review_components`` viewers. The window has two regions:

- **Picker (top + left)** — an all-components bar of each IC's EKG correlation
  (which IC is the heartbeat?), colored by what's removed; **click a bar to inspect
  that IC** (``1`` / the button toggles its removal). Below it, the inspected IC's
  detail (time course + strongest input contributor).
- **Reviewer (right)** — the previewed EMG channel, raw vs the chosen cleaned
  variant, plus its PSD. Step channels with the arrow keys / the ``channel`` dropdown.

The three time-domain panels (IC, its contributor, the channel preview) share one
x-axis, and its zoom is preserved across redraws (flip **Auto limits** on to reset).

A single **Motion** auto/off toggle and a **splice** selector drive the rest;
**Save** writes the decision to the sibling ``<stem>.delsys-artifact`` (and clears
the stale ``*_cleaned.h5``) so the next :func:`delsys.clean` reproduces it.

The cheap-iteration core is :class:`delsys.cleaning.CleaningSession`: FastICA is fit
*once* when the window opens, and every toggle / motion change is a refit-free
:meth:`~delsys.cleaning.CleaningSession.recompute`. ``datanavigator`` is an optional
dependency, imported only when the reviewer is launched.

Interaction:

- **click a bar** — inspect that IC (switch the detail panel; non-destructive);
- **``1``** — toggle the inspected IC's removal → live re-clean;
- **``j`` / ``k``** — inspect the previous / next IC (non-destructive);
- **arrow keys / ``channel`` dropdown** — preview a different EMG channel;
- **``Motion`` toggle** — auto (each sensor's own ACC) vs off;
- **``splice`` dropdown** — combined / ekgonly / motiononly variant;
- **Save decision** — write ``<stem>.delsys-artifact``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from delsys.cleaning import (
    CleaningSession,
    _channel_label,
    _draw_psd,
    score_components_against_ekg,
)

#: Splice variants offered in the sidebar, in dropdown order.
_SPLICE_CHOICES = ("combined", "ekgonly", "motiononly")

#: The cleaned-variant attribute on a CleaningResult for each splice choice.
_SPLICE_ATTR = {
    "combined": "cleaned_emg",
    "ekgonly": "cleaned_emg_ekgonly",
    "motiononly": "cleaned_emg_motiononly",
}


def _build_clean_reviewer_class():
    """Build the reviewer class (``datanavigator`` imported here, lazily)."""
    from datanavigator.plots import PlotBrowser

    class CleanReviewer(PlotBrowser):
        """PlotBrowser whose browse axis is the previewed EMG channel.

        The component decision lives in :attr:`_remove` (toggled with ``1`` / the
        button on the *inspected* IC; clicking a bar switches which IC is inspected)
        and the motion pairing in the ``motion`` state variable; both feed
        :meth:`CleaningSession.recompute`, whose cached result every panel reads. The
        ICA fit never re-runs. The shared time-axis zoom is preserved across redraws
        (so you can compare ICs / channels / splices at the same window).
        """

        def __init__(self, session: CleaningSession, figure_handle=None) -> None:
            if session.ica is None:
                raise ValueError(
                    "Log.clean(): the ECG/ICA stage didn't run (no EKG, or "
                    "use_ecg_stage=False) — there are no components to pick."
                )
            self._session = session
            self._remove = set(session.auto_components())
            self._inspect_ic = sorted(self._remove)[0] if self._remove else 0
            self._splice = "combined"
            self._result = None  # set by the first _do_recompute()
            self._ax_bar = None  # the all-ICs bar axes (rebound each draw)
            self._ax_time = None  # the shared time-axis parent (rebound each draw)
            self._time_axes = []  # the time-domain axes whose y tracks the x-window
            self._xlim = None  # preserved time-axis view across redraws

            # Per-IC EKG-correlation scores for the overview bar. Computed here
            # when the session fit with auto-detection off (so the picker is
            # always informative regardless of config).
            if session.ic_corr_scores is not None:
                self._scores = np.abs(np.asarray(session.ic_corr_scores))
            else:
                scores, _ = score_components_against_ekg(
                    session.ica.sources,
                    np.asarray(session.ekg_1d).reshape(-1),
                    max_lag_samples=session.config.ecg_corr_max_lag_samples,
                )
                self._scores = np.abs(scores)

            # Cache the ICA inputs (the inverse-transformed sources): the per-IC
            # contributor traces don't change with the removal set.
            self._ica_input = np.asarray(
                session.ica.model.inverse_transform(session.ica.sources)
            )
            self._ica_feat = session.ica_input_feature_names or [
                f"in{i}" for i in range(self._ica_input.shape[1])
            ]

            if figure_handle is None:
                figure_handle = plt.figure(figsize=(14, 9))
            super().__init__(
                plot_data=list(range(len(session.feature_names))),  # browse = channel
                plot_func=self._plot,
                figure_handle=figure_handle,
                show_item_dropdown=False,  # we add our own (labelled "channel")
            )
            self._do_recompute()
            self._add_controls()
            self.add_item_dropdown(list(session.feature_names), var_name="channel")
            # Leave "Auto limits" OFF so the time-axis zoom persists across redraws
            # (update() captures/restores it); flip it ON to snap back to full extent.
            self._click_cid = self.figure.canvas.mpl_connect(
                "button_press_event", self._on_bar_click
            )
            # Subclass: PlotBrowser.__init__ skips the first draw for us.
            self.update()
            self.reset_axes()
            plt.show(block=False)
            print(
                f"delsys.clean: ready — {session.n_components} components, "
                f"{len(session.feature_names)} EMG channels. Click a bar (or j/k) to "
                "inspect an IC; '1' toggles its removal (red); arrows step channels; "
                "zoom persists (Auto limits resets); Save to write."
            )

        # -- recompute (the cheap, refit-free core) -----------------------

        def _do_recompute(self) -> None:
            self._result = self._session.recompute(
                sorted(self._remove),
                motion=self._current_motion(),
                motiononly=(self._splice == "motiononly"),
            )

        @property
        def _n_components(self) -> int:
            return self._session.n_components

        @property
        def _preview_ch(self) -> int:
            return self._current_idx

        # -- actions -------------------------------------------------------

        def _toggle_ic(self, ic: int) -> None:
            if ic in self._remove:
                self._remove.discard(ic)
            else:
                self._remove.add(ic)
            self._do_recompute()
            self.update()

        def toggle_remove(self, event=None) -> None:
            """Toggle the *inspected* IC's removal (button / ``1`` key)."""
            self._toggle_ic(self._inspect_ic)

        def _inspect_next(self, event=None) -> None:
            """Inspect the next IC (``k``) — non-destructive, just redraws the detail."""
            self._inspect_ic = (self._inspect_ic + 1) % self._n_components
            self.update()

        def _inspect_prev(self, event=None) -> None:
            """Inspect the previous IC (``j``)."""
            self._inspect_ic = (self._inspect_ic - 1) % self._n_components
            self.update()

        def _on_bar_click(self, event) -> None:
            """Click a bar in the overview → inspect that IC (non-destructive).

            Switching the inspected IC only redraws the detail (no recompute); use
            ``1`` / the Toggle-remove button to actually add/drop it from the set.
            """
            if event.inaxes is not self._ax_bar or event.xdata is None:
                return
            ic = int(round(event.xdata))
            if 0 <= ic < self._n_components:
                self._inspect_ic = ic
                self.update()

        def _current_motion(self) -> Optional[str]:
            var = getattr(self, "_motion_var", None)
            if var is None:
                return "auto"
            return "auto" if var.current_state == "auto" else None

        def _on_motion_change(self) -> None:
            self._do_recompute()
            self.update()

        def _on_splice_change(self) -> None:
            self._splice = self._splice_var.current_state
            # Recompute so the motiononly variant is materialized when selected
            # (cheap: the ICA fit + ACC are cached).
            self._do_recompute()
            self.update()

        # -- controls ------------------------------------------------------

        def _add_controls(self) -> None:
            # GenericBrowser treats 1-9 as memory slots and PlotBrowser.update
            # re-shows the (empty) widget every redraw; disable + neutralize the
            # re-show so the slots are gone and '1' is free for us.
            self.memoryslots.disable()
            self.memoryslots.hide()
            self.memoryslots.show = lambda *a, **k: None
            self.add_key_binding(
                "1", self.toggle_remove, description="Toggle remove (inspected IC)"
            )
            self.add_key_binding("k", self._inspect_next, description="Inspect next IC")
            self.add_key_binding("j", self._inspect_prev, description="Inspect previous IC")
            self.buttons.add(
                text="Toggle remove", type_="Push", action_func=self.toggle_remove
            )
            self.buttons.add(text="Save decision", type_="Push", action_func=self.save)

            self._motion_var = self.statevariables.add(
                "motion", ["auto", "off"], widget="dropdown"
            )
            self._motion_var.add_on_change(self._on_motion_change)
            self._splice_var = self.statevariables.add(
                "splice", list(_SPLICE_CHOICES), widget="dropdown"
            )
            self._splice_var.add_on_change(self._on_splice_change)

        # -- drawing -------------------------------------------------------

        def _plot(self, channel_idx, figure, **kwargs) -> None:
            gs = figure.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.55, wspace=0.22)
            self._ax_bar = figure.add_subplot(gs[0, :])
            ax_ic = figure.add_subplot(gs[1, 0])
            # The three time-domain panels (IC, its contributor, the channel preview)
            # share one x-axis; the PSD (frequency) stays independent.
            ax_contrib = figure.add_subplot(gs[2, 0], sharex=ax_ic)
            ax_prev = figure.add_subplot(gs[1, 1], sharex=ax_ic)
            ax_psd = figure.add_subplot(gs[2, 1])
            self._ax_time = ax_ic
            self._time_axes = [ax_ic, ax_contrib, ax_prev]

            self._draw_bar(self._ax_bar)
            self._draw_ic_detail(ax_ic, ax_contrib)
            self._draw_channel(ax_prev, ax_psd, channel_idx)
            figure.suptitle(self._title(channel_idx), fontsize=10)
            # Rescale each time panel's y to the data in the *visible* x-window,
            # live on interactive pan/zoom too.
            ax_ic.callbacks.connect("xlim_changed", self._autoscale_y_to_xlim)

        def _draw_bar(self, ax) -> None:
            x = np.arange(self._n_components)
            colors = [
                "C3" if i in self._remove else "0.6" for i in range(self._n_components)
            ]
            bars = ax.bar(x, self._scores, color=colors)
            # Outline the inspected IC.
            if 0 <= self._inspect_ic < len(bars):
                bars[self._inspect_ic].set_edgecolor("C0")
                bars[self._inspect_ic].set_linewidth(2.0)
            thr = self._session.config.ecg_corr_threshold
            ax.axhline(thr, color="C0", lw=0.8, ls="--", label=f"threshold={thr:.2f}")
            ax.set_xticks(x)
            ax.set_xlabel("IC index  (click a bar to inspect; '1' toggles remove)")
            ax.set_ylabel("|corr| vs EKG")
            ax.set_title("components — red = removed", fontsize=9, loc="left")
            ax.legend(loc="upper right", fontsize=7)

        def _draw_ic_detail(self, ax_ic, ax_contrib) -> None:
            c = self._inspect_ic
            sources = np.asarray(self._session.ica.sources)
            time = np.asarray(self._result.time)
            t_ic = time if time.shape[0] == sources.shape[0] else np.arange(sources.shape[0])

            removed = c in self._remove
            ax_ic.plot(t_ic, sources[:, c], color="C3" if removed else "0.4", lw=0.6)
            ax_ic.set_ylabel(f"IC {c}")
            score = float(self._scores[c]) if c < len(self._scores) else float("nan")
            ax_ic.set_title(
                f"IC {c}  [{'REMOVED' if removed else 'kept'}]  |corr|={score:.2f}",
                fontsize=9, loc="left",
            )

            col = np.asarray(self._session.ica.mixing)[:, c]
            row = int(np.argmax(np.abs(col)))
            label = self._ica_feat[row] if 0 <= row < len(self._ica_feat) else f"in{row}"
            ax_contrib.plot(t_ic, self._ica_input[:, row], color="0.3", lw=0.6)
            ax_contrib.set_ylabel(f"top: {label}")
            ax_contrib.set_title(
                f"top contributor: {label}  (|A|={abs(float(col[row])):.3f})",
                fontsize=8, loc="left",
            )

        def _draw_channel(self, ax_prev, ax_psd, ch: int) -> None:
            result = self._result
            raw = np.asarray(result.stages["raw"])
            time = np.asarray(result.time)
            variant = getattr(result, _SPLICE_ATTR[self._splice])

            ax_prev.plot(time, raw[:, ch], color="0.5", lw=0.6, label="raw")
            if variant is None:
                ax_prev.text(
                    0.5, 0.5, f"{self._splice} variant not available (stage off)",
                    transform=ax_prev.transAxes, ha="center", va="center", color="0.4",
                )
            else:
                cleaned_col = np.asarray(variant)[:, ch]
                ax_prev.plot(time, cleaned_col, color="C2", lw=0.6, label=self._splice)
                _draw_psd(ax_psd, raw[:, ch], cleaned_col, float(result.sr))
            ax_prev.set_ylabel(f"ch {ch}: {_channel_label(result, ch)}")
            ax_prev.set_xlabel("time (s)")
            ax_prev.legend(loc="upper right", fontsize=7)

        def _title(self, ch: int) -> str:
            return (
                f"channel {ch + 1}/{len(self._session.feature_names)}: "
                f"{_channel_label(self._result, ch)}  "
                f"| removing {sorted(self._remove)}  "
                f"| motion={self._current_motion() or 'off'}  | splice={self._splice}"
            )

        def update(self, event=None) -> None:
            # Preserve the shared time-axis zoom across redraws so ICs / channels /
            # splices can be compared at the same window. "Auto limits" ON opts out
            # (snap back to full extent). PlotBrowser.update clears + rebuilds axes.
            auto = "Auto limits" in self.buttons and self.buttons["Auto limits"].state
            if not auto and self._ax_time is not None:
                try:
                    self._xlim = self._ax_time.get_xlim()
                except Exception:  # noqa: BLE001 — stale axis; fall back to full
                    self._xlim = None
            super().update(event)
            if not auto and self._xlim is not None and self._ax_time is not None:
                self._ax_time.set_xlim(self._xlim)
            self._autoscale_y_to_xlim()
            plt.draw()

        def _autoscale_y_to_xlim(self, event=None) -> None:
            """Fit each time panel's y-limits to the data within the visible x-window.

            matplotlib leaves y at the full-data extent when x is zoomed; this hugs y
            to what's actually on screen. Fired on every ``xlim_changed`` (interactive
            pan/zoom and the :meth:`update` restore) and per redraw.
            """
            if self._ax_time is None:
                return
            x0, x1 = self._ax_time.get_xlim()
            for ax in self._time_axes:
                lo, hi = np.inf, -np.inf
                for line in ax.get_lines():
                    xd = np.asarray(line.get_xdata(), dtype=float)
                    yd = np.asarray(line.get_ydata(), dtype=float)
                    if xd.shape != yd.shape or xd.size == 0:
                        continue
                    m = (xd >= min(x0, x1)) & (xd <= max(x0, x1)) & np.isfinite(yd)
                    if m.any():
                        lo = min(lo, float(yd[m].min()))
                        hi = max(hi, float(yd[m].max()))
                if np.isfinite(lo) and hi > lo:
                    pad = 0.05 * (hi - lo)
                    ax.set_ylim(lo - pad, hi + pad)

        # -- save ----------------------------------------------------------

        def save(self, event=None) -> Optional[str]:
            """Write the current decision to the sibling ``<stem>.delsys-artifact``.

            Records the explicit removal set + splice + motion keyed by the Log's
            checkpoint, and clears any stale ``*_cleaned.h5`` so the next
            :func:`delsys.clean` regenerates from it. Returns the sidecar path (or
            ``None`` when the Log has no source path to key on).
            """
            from delsys import _clean

            fname = getattr(self._session.log, "fname", None)
            if not fname:
                print("delsys.clean: Log has no source path; cannot save a decision.")
                return None
            if not str(fname).lower().endswith((".h5", ".hdf5")):
                print(
                    f"delsys.clean: note — {fname!r} is not an .h5 checkpoint; the "
                    "decision sidecar sits beside it and delsys.clean() walks .h5 files."
                )
            path = _clean.upsert_decision(
                fname,
                components=sorted(self._remove),
                config=self._session.config,
                splice_source=self._splice,
                motion=self._current_motion(),
                accept=True,
            )
            print(
                f"delsys.clean: saved decision (remove={sorted(self._remove)}, "
                f"splice={self._splice}, motion={self._current_motion() or 'off'}) -> {path}"
            )
            return path

    return CleanReviewer


def launch_clean_reviewer(session: CleaningSession):
    """Build and show a :class:`CleanReviewer` over ``session`` (returns it)."""
    cls = _build_clean_reviewer_class()
    return cls(session)
