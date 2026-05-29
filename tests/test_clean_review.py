"""Interactive ICA-cleaning tool (``delsys.clean_review`` / ``Log.clean``).

Exercises the reviewer headlessly (matplotlib Agg, no live event loop): the GUI
builds but its bar-click / toggle / motion / splice / save actions are driven
directly, then the recompute result and the written ``<stem>.delsys-artifact``
sidecar are inspected. Skipped if datanavigator (the optional GUI dependency)
isn't importable.
"""

import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("datanavigator")

import delsys  # noqa: E402
from delsys import CleaningConfig, Log  # noqa: E402
from delsys._clean import read_decision  # noqa: E402

FIXTURE = "discover170.csv"  # carries EMG + EKG, so the ICA stage runs


def _csv_log(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return Log(str(csv))


def _h5_log(fixtures_dir, tmp_path):
    """A Log loaded from a native .h5 checkpoint (so save() writes a sidecar)."""
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    raw = delsys.to_native_h5(str(csv))
    return Log(str(raw)), tmp_path


def _first_unremoved(rev):
    return next(c for c in range(rev._session.n_components) if c not in rev._remove)


def test_clean_opens_with_auto_defaults(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()

    assert rev._current_idx == 0  # browse axis = previewed channel
    assert len(rev.data) == len(rev._session.feature_names)
    assert rev._remove == set(rev._session.auto_components())
    assert rev._splice == "combined"
    assert rev._current_motion() == "auto"
    assert rev._result is not None
    assert "channel" in rev.statevariables  # the item dropdown


def test_bar_click_inspects_ic_without_toggling(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    target = _first_unremoved(rev)
    removed_before = set(rev._remove)

    rev._on_bar_click(SimpleNamespace(inaxes=rev._ax_bar, xdata=float(target)))
    assert rev._inspect_ic == target          # switched the inspected IC
    assert rev._remove == removed_before       # but did NOT change the removal set

    # Toggling is the explicit action.
    rev.toggle_remove()
    assert target in rev._remove


def test_bar_click_outside_axes_ignored(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    before = set(rev._remove)
    rev._on_bar_click(SimpleNamespace(inaxes=None, xdata=1.0))
    assert rev._remove == before


def test_toggle_remove_button_acts_on_inspected_ic(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    rev._inspect_ic = _first_unremoved(rev)

    rev.toggle_remove()
    assert rev._inspect_ic in rev._remove
    rev.toggle_remove()
    assert rev._inspect_ic not in rev._remove


def test_j_k_inspect_components_without_toggling(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    rev._inspect_ic = 0
    removed_before = set(rev._remove)

    rev._inspect_next()
    assert rev._inspect_ic == 1
    rev._inspect_prev()
    rev._inspect_prev()
    assert rev._inspect_ic == rev._session.n_components - 1  # wrapped
    assert rev._remove == removed_before  # inspecting never changes the removal set


def test_motion_toggle_off_disables_stage(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    assert "per_channel" in rev._result.diagnostics["motion"]  # auto -> ran

    rev._motion_var.set_state("off")  # fires _on_motion_change -> recompute
    assert rev._current_motion() is None
    assert rev._result.diagnostics["motion"] == {"used": False}


def test_splice_change_updates_variant(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    rev._splice_var.set_state("ekgonly")  # fires _on_splice_change
    assert rev._splice == "ekgonly"


def test_channel_browse_redraws(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    rev._current_idx = 3
    rev.update()  # must not raise; previews channel 3
    assert rev._preview_ch == 3


def test_time_axis_zoom_persists_across_updates(fixtures_dir, tmp_path):
    """A zoom on the shared time-axis survives inspecting a different IC / channel."""
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    # Auto limits is off by default, so the zoom should be preserved.
    assert rev.buttons["Auto limits"].state is False
    rev._ax_time.set_xlim(0.01, 0.04)  # user zooms

    rev._inspect_next()  # a redraw (inspect another IC)
    assert rev._ax_time.get_xlim() == (0.01, 0.04)
    rev._current_idx = 2
    rev.update()  # a redraw (step channel)
    assert rev._ax_time.get_xlim() == (0.01, 0.04)


def test_time_axis_panels_share_x(fixtures_dir, tmp_path):
    """The IC detail, its contributor, and the channel preview share one x-axis."""
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    rev._ax_time.set_xlim(0.0, 0.05)
    # Every axis sharing x with _ax_time moves together; expect >= 3 such axes
    # (IC, contributor, preview) — but NOT the 4th (PSD, frequency).
    shared = [ax for ax in rev.figure.axes if ax.get_xlim() == (0.0, 0.05) and ax is not rev._ax_bar]
    assert len(shared) >= 3


def test_y_autoscales_to_visible_x_window(fixtures_dir, tmp_path):
    """Zooming x rescales each time panel's y to the data within the window."""
    lf = _csv_log(fixtures_dir, tmp_path)
    rev = lf.clean()
    ax = rev._ax_time
    line = ax.get_lines()[0]  # the inspected-IC trace
    xd = np.asarray(line.get_xdata(), dtype=float)
    yd = np.asarray(line.get_ydata(), dtype=float)

    x0, x1 = float(xd[0]), float(xd[len(xd) // 4])
    ax.set_xlim(x0, x1)  # fires xlim_changed -> y autoscale

    m = (xd >= x0) & (xd <= x1)
    wmin, wmax = float(yd[m].min()), float(yd[m].max())
    pad = 0.05 * (wmax - wmin)
    assert ax.get_ylim() == pytest.approx((wmin - pad, wmax + pad), rel=1e-6, abs=1e-9)


def test_save_writes_artifact_sidecar_and_marks_stale(fixtures_dir, tmp_path):
    lf, folder = _h5_log(fixtures_dir, tmp_path)
    delsys.clean(str(folder), progress=False)
    cleaned = folder / "Trial_5_cleaned.h5"
    assert cleaned.exists()

    rev = lf.clean()
    rev._inspect_ic = _first_unremoved(rev)
    rev.toggle_remove()  # force a non-default decision
    rev._splice_var.set_state("ekgonly")

    path = rev.save()
    assert path == str(folder / "Trial_5.delsys-artifact")
    assert not cleaned.exists()  # stale snapshot cleared

    entry = read_decision(folder / "Trial_5.h5")
    assert entry["ecg_components_to_remove"] == sorted(rev._remove)
    assert entry["splice_source"] == "ekgonly"
    assert entry["accept"] is True


def test_clean_raises_without_ecg_stage(fixtures_dir, tmp_path):
    lf = _csv_log(fixtures_dir, tmp_path)
    with pytest.raises(ValueError):
        lf.clean(config=CleaningConfig(use_ecg_stage=False))
