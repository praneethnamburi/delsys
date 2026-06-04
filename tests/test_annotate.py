"""Interactive annotator (``delsys.annotate`` / ``Log.view``).

Exercises the marking + save logic headlessly (matplotlib Agg, no live event
loop): the GUI launches but its mark/dead/undo/marker methods are driven directly,
then the resulting unified ``<stem>.delsys-events`` sidecar is read back (noise +
typed marker tracks). Skipped if datanavigator (the optional GUI dependency) isn't
importable.
"""

import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

pytest.importorskip("datanavigator")

import delsys  # noqa: E402
from delsys import _events  # noqa: E402
from delsys._noise import (  # noqa: E402
    format_signal_key,
    key_address,
    parse_key,
    sidecar_path_for,
    write_noise_sidecar,
)

FIXTURE = "discover170.csv"


def _log(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return delsys.Log(str(csv))


def _addr0(lf):
    """Address (label-free) of the first signal — the in-memory key."""
    return key_address(format_signal_key(lf.signals[0]))


def _saved_noise(ann):
    """Save the annotator and read back its unified file's noise track."""
    return _events.read_noise_signals(ann.save())


# ---------------------------------------------------------------------------
# Noise track — signal-centric view
# ---------------------------------------------------------------------------


def test_annotate_marks_channel_window_and_saves(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)

    sigs = _saved_noise(ann)
    key = format_signal_key(lf.signals[0])
    assert sigs[key]["windows"] == [[0.02, 0.05]]


def test_annotate_mod_scope_writes_coordless_key(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.buttons["Mod scope"].set_state(True)  # whole sensor+modality
    ann.add_window(0.10, 0.12)

    sigs = _saved_noise(ann)
    mod_key = format_signal_key(lf.signals[0], include_coord=False)
    assert sigs[mod_key]["windows"] == [[0.10, 0.12]]


def test_annotate_toggle_dead_roundtrips(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    key = format_signal_key(lf.signals[0])

    ann.toggle_dead()
    assert _saved_noise(ann)[key]["dead"] == [[None, None]]

    # Toggling again clears it; an all-empty entry is dropped on save.
    ann.toggle_dead()
    assert key not in _saved_noise(ann)


def test_annotate_undo_drops_last_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)
    ann.add_window(0.06, 0.08)
    ann.undo()

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_zero_width_drag_ignored(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.add_window(0.03, 0.03)  # a click, not a drag
    assert ann._ann.get(_addr0(lf), {"windows": []})["windows"] == []


def test_annotate_keypress_marks_window(fixtures_dir, tmp_path):
    """Two 'n' presses (start, then end) at the cursor add a noise window."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann._mark_point(SimpleNamespace(xdata=0.02))
    ann._mark_point(SimpleNamespace(xdata=0.05))

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_keypress_removes_nearest_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)
    ann.add_window(0.10, 0.12)
    ann._remove_window(SimpleNamespace(xdata=0.11))  # nearest the second window

    assert ann._ann[_addr0(lf)]["windows"] == [[0.02, 0.05]]


def test_annotate_auto_limits_on_by_default(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    assert ann.buttons["Auto limits"].state is True


# ---------------------------------------------------------------------------
# Legacy seeding + path
# ---------------------------------------------------------------------------


def test_annotate_seeds_from_legacy_noise_sidecar(fixtures_dir, tmp_path):
    """No unified file yet -> the annotator seeds noise from a legacy
    ``<stem>.delsys-noise`` (the migration bridge)."""
    lf = _log(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    write_noise_sidecar(sidecar_path_for(lf.fname), {key: [[0.2, 0.3]]})

    ann = lf.view()
    assert ann._ann[_addr0(lf)]["windows"] == [[0.2, 0.3]]
    # Saving migrates the marks into the unified file.
    assert format_signal_key(lf.signals[0]) in _saved_noise(ann)


def test_annotate_seeds_from_unified_events(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    key = format_signal_key(lf.signals[0])
    _events.write_events(
        _events.events_path_for(lf.fname), {"noise": {"signals": {key: [[0.4, 0.5]]}}}
    )
    ann = lf.view()
    assert ann._ann[_addr0(lf)]["windows"] == [[0.4, 0.5]]


def test_annotate_renders_existing_windows_regardless_of_label(fixtures_dir, tmp_path):
    """A sidecar whose key carries a *different* label than the current one must
    still render — the annotator indexes by structural address, not the full key."""
    lf = _log(fixtures_dir, tmp_path)
    addr = _addr0(lf)
    write_noise_sidecar(sidecar_path_for(lf.fname), {f"{addr} | stale_label": [[0.02, 0.05]]})

    ann = lf.view()
    ann._current_idx = 0
    ann.update()
    assert ann._ann[addr]["windows"] == [[0.02, 0.05]]
    assert len(ann._overlay_artists) >= 1
    # Saving rewrites the key with the current label (self-heal).
    assert format_signal_key(lf.signals[0]) in _saved_noise(ann)


def test_view_path_shared_across_kinds(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    assert lf.view("signal")._events_path == lf.view("sensor")._events_path
    assert lf.view()._events_path.endswith(".delsys-events")


# ---------------------------------------------------------------------------
# Typed marker tracks
# ---------------------------------------------------------------------------


def test_marker_point_event_saved_per_signal(fixtures_dir, tmp_path):
    """A '1' press places a size-1 marker on the current channel address."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann._mark_marker("1", 1, SimpleNamespace(xdata=1.40))

    sigs = _events.read_marker_signals(ann.save(), "1")
    key = format_signal_key(lf.signals[0])
    assert sigs[key] == [[1.40]]


def test_marker_window_event_two_presses(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann._mark_marker("2", 2, SimpleNamespace(xdata=0.50))
    ann._mark_marker("2", 2, SimpleNamespace(xdata=0.90))

    sigs = _events.read_marker_signals(ann.save(), "2")
    assert sigs[format_signal_key(lf.signals[0])] == [[0.50, 0.90]]


def test_marker_remove_nearest(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann._mark_marker("1", 1, SimpleNamespace(xdata=1.0))
    ann._mark_marker("1", 1, SimpleNamespace(xdata=2.0))
    ann._remove_marker("1", SimpleNamespace(xdata=1.9))  # nearest the second

    assert [r["seq"] for r in ann._markers["1"][_addr0(lf)]] == [[1.0]]


def test_marker_dispatch_by_event_key(fixtures_dir, tmp_path):
    """The digit-key handler dispatches on event.key to the right track."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann._mark_marker_event(SimpleNamespace(key="1", xdata=1.4))  # point: one press
    sigs = _events.read_marker_signals(ann.save(), "1")
    assert sigs[format_signal_key(lf.signals[0])] == [[1.4]]


def test_markers_and_noise_coexist_in_one_file(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view()
    ann._current_idx = 0
    ann.add_window(0.02, 0.05)  # noise
    ann._mark_marker("1", 1, SimpleNamespace(xdata=1.4))  # marker

    path = ann.save()
    doc = _events.read_events(path)["events"]
    assert "noise" in doc and "1" in doc
    # Both addressable / collapsible.
    assert _events.read_noise_signals(path)[format_signal_key(lf.signals[0])]["windows"] == [
        [0.02, 0.05]
    ]
    recs = _events.collapse_markers(path, "1")
    assert recs[0]["seq"] == [1.4] and recs[0]["address"].startswith(
        str(lf.signals[0].sensor.number)
    )


def test_custom_marker_specs(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view(events={"onset": 1, "phrase": 2})
    slugs = [slug for slug, *_ in ann._marker_specs]
    assert slugs == ["onset", "phrase"]
    # add/remove dispatch is keyed by the bound key (= the name for ad-hoc specs).
    assert "onset" in ann._marker_add_by_key and "alt+phrase" in ann._marker_remove_by_key


# ---------------------------------------------------------------------------
# Sensor-centric view (PlotBrowser subclass; stacked modality subplots)
# ---------------------------------------------------------------------------


def test_sensor_view_marks_panel_window(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view("sensor")
    ann._current_idx = 0
    ann.update()  # build panels / _panel_axes for sensor 0
    ax, key = ann._panel_axes[0]
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.02))
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.05))

    saved = {key_address(k): v for k, v in _saved_noise(ann).items()}
    assert saved[key_address(key)]["windows"] == [[0.02, 0.05]]


def test_sensor_view_marks_typed_event(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view("sensor")
    ann._current_idx = 0
    ann.update()
    ax, key = ann._panel_axes[0]
    ann._mark_marker("1", 1, SimpleNamespace(inaxes=ax, xdata=1.4))

    sigs = {key_address(k): v for k, v in _events.read_marker_signals(ann.save(), "1").items()}
    assert sigs[key_address(key)] == [[1.4]]


def test_sensor_view_dead_toggle(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view("sensor")
    ann._current_idx = 0
    ann.update()
    ax, key = ann._panel_axes[0]
    ann._toggle_dead_at(SimpleNamespace(inaxes=ax, xdata=0.0))

    saved = {key_address(k): v for k, v in _saved_noise(ann).items()}
    assert saved[key_address(key)]["dead"] == [[None, None]]


def test_sensor_view_splits_quattro_fsr_analog_into_subchannels(fixtures_dir, tmp_path):
    """EMGQ / FSR / Analog get one coord-ful panel per present sub-channel."""
    from delsys.annotate import _SPLIT_MODALITIES

    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view("sensor")
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
    """With Sensor scope on, a noise mark lands on every modality of the sensor."""
    lf = _log(fixtures_dir, tmp_path)
    ann = lf.view("sensor")
    ann._current_idx = 0
    ann.update()
    ann.buttons["Sensor scope"].set_state(True)
    ax, _ = ann._panel_axes[0]
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.02))
    ann._mark_point(SimpleNamespace(inaxes=ax, xdata=0.05))

    sensor = ann._sensors[0]
    saved = {key_address(k): v for k, v in _saved_noise(ann).items()}
    for m in ann._markable_modalities(sensor):
        mk = key_address(ann._modality_key_for(sensor, m))
        assert saved[mk]["windows"] == [[0.02, 0.05]]


def test_view_invalid_kind_rejected(fixtures_dir, tmp_path):
    lf = _log(fixtures_dir, tmp_path)
    with pytest.raises(ValueError):
        lf.view("bogus")
