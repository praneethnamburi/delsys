"""Interactive noise annotator (``delsys.annotate`` / ``Log.annotate_noise``).

Exercises the marking + save logic headlessly (matplotlib Agg, no live event
loop): the GUI launches but its mark/dead/undo/save methods are driven directly,
then the resulting ``<stem>.delsys-noise`` sidecar is read back. Skipped if
datanavigator (the optional GUI dependency) isn't importable.
"""

import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

pytest.importorskip("datanavigator")

import delsys  # noqa: E402
from delsys._noise import (  # noqa: E402
    format_signal_key,
    parse_key,
    read_noise_sidecar,
    sidecar_path_for,
    write_noise_sidecar,
)

FIXTURE = "discover170.csv"


def _log(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return delsys.Log(str(csv))


def test_annotate_marks_channel_window_and_saves(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)

    doc = read_noise_sidecar(ann.save())
    key = format_signal_key(lf.signals[0])
    assert doc["signals"][key]["windows"] == [[0.02, 0.05]]


def test_annotate_mod_scope_writes_coordless_key(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.buttons["Mod scope"].set_state(True)  # whole sensor+modality
    ann.add_window(0.10, 0.12)

    doc = read_noise_sidecar(ann.save())
    # Whole-modality key = coord-less address, labelled with the body location.
    mod_key = format_signal_key(lf.signals[0], include_coord=False)
    assert doc["signals"][mod_key]["windows"] == [[0.10, 0.12]]


def test_annotate_toggle_dead_roundtrips(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    key = format_signal_key(lf.signals[0])

    ann.toggle_dead()
    assert read_noise_sidecar(ann.save())["signals"][key]["dead"] == [[None, None]]

    # Toggling again clears it; an all-empty entry is dropped on save.
    ann.toggle_dead()
    assert key not in read_noise_sidecar(ann.save())["signals"]


def test_annotate_undo_drops_last_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)
    ann.add_window(0.06, 0.08)
    ann.undo()

    key = format_signal_key(lf.signals[0])
    assert ann._ann[key]["windows"] == [[0.02, 0.05]]


def test_annotate_zero_width_drag_ignored(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.03, 0.03)  # a click, not a drag
    assert ann._ann.get(format_signal_key(lf.signals[0]), {"windows": []})["windows"] == []


def test_annotate_seeds_from_existing_sidecar(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    write_noise_sidecar(sidecar_path_for(lf.fname), {key: [[0.2, 0.3]]})

    ann = lf.annotate_noise()
    assert ann._ann[key]["windows"] == [[0.2, 0.3]]


def test_annotate_keypress_marks_window(fixtures_dir, tmp_path):
    """Two '1' presses (start, then end) at the cursor add a window."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann._mark_point(SimpleNamespace(xdata=0.02))
    ann._mark_point(SimpleNamespace(xdata=0.05))

    key = format_signal_key(lf.signals[0])
    assert ann._ann[key]["windows"] == [[0.02, 0.05]]


def test_annotate_keypress_removes_nearest_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)
    ann.add_window(0.10, 0.12)
    ann._remove_window(SimpleNamespace(xdata=0.11))  # nearest the second window

    key = format_signal_key(lf.signals[0])
    assert ann._ann[key]["windows"] == [[0.02, 0.05]]


def test_annotate_auto_limits_on_by_default(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    assert ann.buttons["Auto limits"].state is True


# ---------------------------------------------------------------------------
# Sensor-centric view (PlotBrowser subclass; stacked modality subplots)
# ---------------------------------------------------------------------------


def test_sensor_view_marks_modality_window(fixtures_dir, tmp_path):
    """Pressing '1' twice over a modality subplot adds a whole-modality window."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise(view="sensor")
    ann._current_idx = 0
    ann.update()  # build subplots / _subplot_axes for sensor 0
    mod = next(iter(ann._subplot_axes))
    ax = ann._subplot_axes[mod]
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.02))
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.05))

    key = ann._modality_key_for(ann._sensors[0], mod)
    doc = read_noise_sidecar(ann.save())
    assert doc["signals"][key]["windows"] == [[0.02, 0.05]]
    assert parse_key(key).coord is None  # whole-modality (coord-less) address


def test_sensor_view_shares_sidecar_with_signal_view(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    assert (
        lf.annotate_noise(view="signal")._sidecar_path
        == lf.annotate_noise(view="sensor")._sidecar_path
    )


def test_sensor_view_dead_toggle(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise(view="sensor")
    ann._current_idx = 0
    mod = next(iter(ann._subplot_axes))
    ann._toggle_dead_at(SimpleNamespace(inaxes=ann._subplot_axes[mod], xdata=0.0))

    key = ann._modality_key_for(ann._sensors[0], mod)
    assert read_noise_sidecar(ann.save())["signals"][key]["dead"] == [[None, None]]


def test_annotate_invalid_view_rejected(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    with pytest.raises(ValueError):
        lf.annotate_noise(view="bogus")
