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
    key_address,
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


def _addr0(lf):
    """Address (label-free) of the first signal — the in-memory ``_ann`` key."""
    return key_address(format_signal_key(lf.signals[0]))


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

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_zero_width_drag_ignored(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.03, 0.03)  # a click, not a drag
    assert ann._ann.get(_addr0(lf), {"windows": []})["windows"] == []


def test_annotate_seeds_from_existing_sidecar(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    write_noise_sidecar(sidecar_path_for(lf.fname), {key: [[0.2, 0.3]]})

    ann = lf.annotate_noise()
    assert ann._ann[_addr0(lf)]["windows"] == [[0.2, 0.3]]


def test_annotate_renders_existing_windows_regardless_of_label(fixtures_dir, tmp_path):
    """Regression: a sidecar whose key carries a *different* label than the
    current one (e.g. written by an older code version) must still render — the
    annotator indexes by structural address, not the full labelled key."""
    lf = _log(fixtures_dir, tmp_path)
    addr = _addr0(lf)
    # Deliberately stale label on the same address.
    write_noise_sidecar(sidecar_path_for(lf.fname), {f"{addr} | stale_label": [[0.02, 0.05]]})

    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.update()
    # The window loaded under the address and an overlay span was drawn for it.
    assert ann._ann[addr]["windows"] == [[0.02, 0.05]]
    assert len(ann._overlay_artists) >= 1
    # Saving rewrites the key with the current label (self-heal).
    doc = read_noise_sidecar(ann.save())
    assert format_signal_key(lf.signals[0]) in doc["signals"]


def test_annotate_keypress_marks_window(fixtures_dir, tmp_path):
    """Two '1' presses (start, then end) at the cursor add a window."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann._mark_point(SimpleNamespace(xdata=0.02))
    ann._mark_point(SimpleNamespace(xdata=0.05))

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_keypress_removes_nearest_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)
    ann.add_window(0.10, 0.12)
    ann._remove_window(SimpleNamespace(xdata=0.11))  # nearest the second window

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_auto_limits_on_by_default(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise()
    assert ann.buttons["Auto limits"].state is True


# ---------------------------------------------------------------------------
# Sensor-centric view (PlotBrowser subclass; stacked modality subplots)
# ---------------------------------------------------------------------------


def test_sensor_view_marks_panel_window(fixtures_dir, tmp_path):
    """Two '1' presses over a sensor-view panel add a window on that panel's address."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise(view="sensor")
    ann._current_idx = 0
    ann.update()  # build panels / _panel_axes for sensor 0
    ax, key = ann._panel_axes[0]
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.02))
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.05))

    saved = {key_address(k): v for k, v in read_noise_sidecar(ann.save())["signals"].items()}
    assert saved[key_address(key)]["windows"] == [[0.02, 0.05]]


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
    ann.update()
    ax, key = ann._panel_axes[0]
    ann._toggle_dead_at(SimpleNamespace(inaxes=ax, xdata=0.0))

    saved = {key_address(k): v for k, v in read_noise_sidecar(ann.save())["signals"].items()}
    assert saved[key_address(key)]["dead"] == [[None, None]]


def test_sensor_view_splits_quattro_fsr_analog_into_subchannels(fixtures_dir, tmp_path):
    """EMGQ / FSR / Analog get one coord-ful panel per present sub-channel."""
    from delsys.annotate import _SPLIT_MODALITIES

    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise(view="sensor")
    found_multi = False
    for i, s in enumerate(ann._sensors):
        split_mods = [m for m in ann._markable_modalities(s) if m in _SPLIT_MODALITIES]
        if not split_mods:
            continue
        ann._current_idx = i
        ann.update()
        for m in split_mods:
            n_sig = sum(1 for sg in lf.signals if sg.matches(s.number, m, None))
            panel_keys = [k for _, k in ann._panel_axes if parse_key(k).modality == m]
            assert len(panel_keys) == n_sig  # one panel per sub-channel
            assert all(parse_key(k).coord is not None for k in panel_keys)
            if n_sig > 1:
                found_multi = True
    if not found_multi:
        pytest.skip("fixture has no split modality with >1 sub-channel")


def test_sensor_scope_marks_across_all_modalities(fixtures_dir, tmp_path):
    """With Sensor scope on, a mark lands on every modality of the sensor."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.annotate_noise(view="sensor")
    ann._current_idx = 0
    ann.update()
    ann.buttons["Sensor scope"].set_state(True)
    ax, _ = ann._panel_axes[0]
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.02))
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.05))

    sensor = ann._sensors[0]
    saved = {key_address(k): v for k, v in read_noise_sidecar(ann.save())["signals"].items()}
    for m in ann._markable_modalities(sensor):
        mk = key_address(ann._modality_key_for(sensor, m))
        assert saved[mk]["windows"] == [[0.02, 0.05]]


def test_annotate_invalid_view_rejected(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    with pytest.raises(ValueError):
        lf.annotate_noise(view="bogus")
