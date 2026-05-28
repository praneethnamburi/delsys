"""Batch ``delsys.clean()`` + the decisions manifest + noise-event consumption.

The headline property is the reproducibility contract
``cleaned.h5 = f(raw.h5, manifest)``: a first pass freezes the auto-chosen
decision into ``delsys_cleaning.json`` and a replay reproduces the cleaned
checkpoint bit-for-bit. ``discover170.csv`` is the one fixture used end-to-end
(it carries both EMG and EKG, so the ECG stage actually runs); the cleaning
algorithm itself is covered in ``test_cleaning.py``.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import delsys  # noqa: E402
import delsys.log  # noqa: E402
from delsys import CleaningConfig  # noqa: E402
from delsys._clean import _config_body, read_manifest  # noqa: E402
from delsys._noise import apply_noise_mask, read_noise_intervals  # noqa: E402

FIXTURE = "discover170.csv"  # 18 sensors, EMG + EKG present


@pytest.fixture
def raw_checkpoint(fixtures_dir, tmp_path):
    """A native ``.h5`` checkpoint built from the fixture, named Trial_5.h5."""
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    raw = delsys.to_native_h5(str(csv))
    return Path(raw), tmp_path


def _signature(h5_path):
    """Concatenate every signal array of a checkpoint — a deterministic fingerprint."""
    lf = delsys.Log(str(h5_path))
    return np.concatenate([np.asarray(s()).ravel() for s in lf.signals])


# ---------------------------------------------------------------------------
# Batch mechanics
# ---------------------------------------------------------------------------


def test_clean_builds_artifacts_and_is_idempotent(raw_checkpoint):
    raw, folder = raw_checkpoint

    res = delsys.clean(str(folder), progress=False)
    assert res[str(raw)] == "cleaned"

    cleaned = folder / "Trial_5_cleaned.h5"
    assert cleaned.exists()
    assert (folder / "Trial_5_cleaning_report.pdf").exists()
    assert (folder / "delsys_cleaning.json").exists()
    assert (folder / "delsys_cleaning_report.txt").exists()
    # The cleaned checkpoint reloads as a normal Log.
    assert delsys.Log(str(cleaned)).emg is not None

    # Second run skips the existing cleaned checkpoint...
    assert delsys.clean(str(folder), progress=False)[str(raw)] == "hit"
    # ...unless forced.
    assert delsys.clean(str(folder), overwrite=True, progress=False)[str(raw)] == "cleaned"


def test_clean_does_not_walk_its_own_outputs(raw_checkpoint):
    """The walk excludes ``*_cleaned.h5`` so a re-run never cleans its outputs."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    res = delsys.clean(str(folder), overwrite=True, progress=False)
    # Only the raw checkpoint appears as a key; the cleaned output never does.
    assert set(res) == {str(raw)}


def test_clean_report_lists_decision_detail(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    report = (folder / "delsys_cleaning_report.txt").read_text()
    assert "Trial_5.h5 - cleaned" in report
    assert "splice=combined" in report


def test_clean_opt_out_of_pdf_and_report(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), generate_pdf=False, report=False, progress=False)
    assert not (folder / "Trial_5_cleaning_report.pdf").exists()
    assert not (folder / "delsys_cleaning_report.txt").exists()


def test_clean_skips_checkpoint_with_no_emg(raw_checkpoint, monkeypatch):
    raw, folder = raw_checkpoint
    # Build the raw .h5 happened in the fixture; now make every Log EMG-less.
    monkeypatch.setattr(delsys.log.Log, "emg", property(lambda self: None))
    res = delsys.clean(str(folder), progress=False)
    assert res[str(raw)].startswith("skipped")
    assert not (folder / "Trial_5_cleaned.h5").exists()


def test_clean_invalid_splice_source_rejected(raw_checkpoint):
    raw, folder = raw_checkpoint
    with pytest.raises(ValueError):
        delsys.clean(str(folder), splice_source="bogus", progress=False)


def test_clean_progress_summary_and_silence(raw_checkpoint, capsys):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=True)
    out = capsys.readouterr().out
    assert "delsys.clean:" in out and "cleaned 1" in out

    delsys.clean(str(folder), overwrite=True, progress=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Manifest + reproducibility contract
# ---------------------------------------------------------------------------


def test_manifest_captures_auto_decision(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)

    manifest = read_manifest(folder)
    assert manifest["schema"] == 1
    entry = manifest["trials"]["Trial_5"]
    # The four top-level decision fields plus the config body.
    assert isinstance(entry["ecg_components_to_remove"], list)
    assert entry["splice_source"] == "combined"
    assert entry["motion"] == "auto"
    assert entry["noise_event_ref"] is None
    assert entry["accept"] is None  # not yet reviewed
    assert "preprocess_highpass_hz" in entry["config"]
    # The selection fields never leak into the config body.
    assert "ecg_components_to_remove" not in entry["config"]
    assert "ecg_auto_remove_components" not in entry["config"]


def test_replay_from_manifest_is_byte_identical(raw_checkpoint):
    """First pass (auto) then a forced replay (manifest) produce identical output."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    sig1 = _signature(folder / "Trial_5_cleaned.h5")

    delsys.clean(str(folder), overwrite=True, progress=False)
    sig2 = _signature(folder / "Trial_5_cleaned.h5")

    assert np.array_equal(sig1, sig2)


def test_manifest_drives_output(raw_checkpoint):
    """``cleaned.h5 = f(raw.h5, manifest)``: a hand-authored manifest reproduces
    a direct ``clean_emg_ekg_artifact`` run with the same explicit decision."""
    raw, folder = raw_checkpoint

    # Author a manifest that pins ECG component [0] (not what auto would pick).
    manifest = {
        "schema": 1,
        "trials": {
            "Trial_5": {
                "ecg_components_to_remove": [0],
                "splice_source": "combined",
                "motion": "auto",
                "noise_event_ref": None,
                "config": _config_body(CleaningConfig()),
            }
        },
    }
    (folder / "delsys_cleaning.json").write_text(json.dumps(manifest, indent=2))

    delsys.clean(str(folder), progress=False)
    via_clean = _signature(folder / "Trial_5_cleaned.h5")

    # The manifest must be preserved verbatim (existing entry, not re-frozen).
    assert read_manifest(folder)["trials"]["Trial_5"]["ecg_components_to_remove"] == [0]

    # Independent reference: the same explicit decision applied directly.
    ref = delsys.Log(str(raw))
    cfg = CleaningConfig(ecg_auto_remove_components=False, ecg_components_to_remove=[0])
    ref.clean_emg_ekg_artifact(
        config=cfg, motion="auto", splice_source="combined", in_place=True, generate_report=False
    )
    ref_h5 = folder / "reference.h5"
    ref.to_hdf5(str(ref_h5))
    via_direct = _signature(ref_h5)

    assert np.array_equal(via_clean, via_direct)


def test_editing_manifest_changes_output(raw_checkpoint):
    """Editing the frozen decision changes the cleaned checkpoint on re-run."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    auto_sig = _signature(folder / "Trial_5_cleaned.h5")

    manifest = read_manifest(folder)
    auto_components = manifest["trials"]["Trial_5"]["ecg_components_to_remove"]
    # Force a different decision: remove no ECG components at all.
    manifest["trials"]["Trial_5"]["ecg_components_to_remove"] = []
    (folder / "delsys_cleaning.json").write_text(json.dumps(manifest, indent=2))

    delsys.clean(str(folder), overwrite=True, progress=False)
    edited_sig = _signature(folder / "Trial_5_cleaned.h5")

    # Removing a real component changed the output; if auto picked nothing this
    # would be a no-op, so only assert a difference when auto removed something.
    if auto_components:
        assert not np.array_equal(auto_sig, edited_sig)


def test_clean_accept_false_blocks_regeneration(raw_checkpoint):
    """A reviewer marking ``accept: false`` skips the trial even under overwrite."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)

    manifest = read_manifest(folder)
    manifest["trials"]["Trial_5"]["accept"] = False
    (folder / "delsys_cleaning.json").write_text(json.dumps(manifest, indent=2))

    res = delsys.clean(str(folder), overwrite=True, progress=False)
    assert res[str(raw)].startswith("skipped: rejected")


def test_manifest_false_writes_no_manifest(raw_checkpoint):
    raw, folder = raw_checkpoint
    res = delsys.clean(str(folder), manifest=False, progress=False)
    assert res[str(raw)] == "cleaned"
    assert (folder / "Trial_5_cleaned.h5").exists()
    assert not (folder / "delsys_cleaning.json").exists()


# ---------------------------------------------------------------------------
# Noise-event consumption
# ---------------------------------------------------------------------------


def _write_noise_event(path, mapping):
    """Write a datanavigator-style ``[metadata, data]`` Event JSON."""
    meta = {"name": "noise", "size": 2, "all_keys_are_tuples": True}
    data = {
        key: {"default": [], "added": added, "removed": removed}
        for key, (added, removed) in mapping.items()
    }
    Path(path).write_text(json.dumps([meta, data]))


def test_read_noise_intervals_keys_and_set_algebra(tmp_path):
    nf = tmp_path / "noise.json"
    _write_noise_event(
        nf,
        {
            "(2, 14, 17)": ([[1.0, 2.0], [5.0, 6.0]], [[5.0, 6.0]]),  # one removed
            "Trial_5": ([[0.1, 0.2]], []),
        },
    )
    # Tuple and string keys resolve identically; removed intervals are dropped.
    assert read_noise_intervals(str(nf), (2, 14, 17)) == [(1.0, 2.0)]
    assert read_noise_intervals(str(nf), "(2, 14, 17)") == [(1.0, 2.0)]
    assert read_noise_intervals(str(nf), "Trial_5") == [(0.1, 0.2)]
    # A trial with no entry yields no windows.
    assert read_noise_intervals(str(nf), (9, 9, 9)) == []


def test_read_noise_intervals_real_datanavigator_file():
    """Smoke-read the real on-disk Event file (skipped if not mounted)."""
    real = Path(r"S:\2201000537 - Operator\event_noise_acc_bicep.json")
    if not real.exists():
        pytest.skip("real datanavigator Event file not mounted")
    intervals = read_noise_intervals(str(real), (2, 14, 17))
    assert len(intervals) > 0
    assert all(b > a for a, b in intervals)


def test_apply_noise_mask_interpolates_and_propagates(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    lf = delsys.Log(str(csv))
    before = lf.emg().copy()

    # A window inside the (short) fixture's extent.
    touched = apply_noise_mask(lf, [(0.02, 0.05)], policy="nan_interp")
    assert touched > 0
    after = lf.emg()
    assert before.shape == after.shape
    assert not np.array_equal(before, after)  # the window was rewritten
    assert not np.isnan(after).any()  # interpolated, no NaNs left


def test_apply_noise_mask_modality_filter(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    lf = delsys.Log(str(csv))
    n_all = sum(1 for _ in lf.signals)
    n_emg = apply_noise_mask(lf, [(0.02, 0.05)], modalities=["EMGS"])
    assert 0 < n_emg < n_all  # only EMGS channels masked, not every signal


def test_apply_noise_mask_unknown_policy_raises(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_5.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    lf = delsys.Log(str(csv))
    with pytest.raises(ValueError):
        apply_noise_mask(lf, [(0.0, 0.1)], policy="zero")


def test_clean_consumes_noise_event(raw_checkpoint):
    raw, folder = raw_checkpoint
    _write_noise_event(folder / "noise.json", {"Trial_5": ([[0.02, 0.05]], [])})

    # First pass freezes the manifest; then point the trial at the noise event.
    delsys.clean(str(folder), progress=False)
    manifest = read_manifest(folder)
    manifest["trials"]["Trial_5"]["noise_event_ref"] = {"path": "noise.json", "key": "Trial_5"}
    (folder / "delsys_cleaning.json").write_text(json.dumps(manifest, indent=2))

    delsys.clean(str(folder), overwrite=True, progress=False)
    report = (folder / "delsys_cleaning_report.txt").read_text()
    assert "noise_masked=" in report


def test_clean_auto_consumes_sibling_delsys_noise(raw_checkpoint):
    """A sibling ``<stem>.delsys-noise`` is consumed by default and recorded as
    provenance in the manifest's ``noise_event_ref``."""
    from delsys._noise import format_signal_key, sidecar_path_for, write_noise_sidecar

    raw, folder = raw_checkpoint

    # Author a sidecar next to the checkpoint, addressing one real signal.
    key = format_signal_key(delsys.Log(str(raw)).signals[0])
    write_noise_sidecar(sidecar_path_for(str(raw)), {key: [[0.02, 0.05]]})

    res = delsys.clean(str(folder), progress=False)
    assert res[str(raw)] == "cleaned"

    # Defaulted ref points at the sibling sidecar (provenance, not the windows).
    entry = read_manifest(folder)["trials"]["Trial_5"]
    assert entry["noise_event_ref"] == "Trial_5" + ".delsys-noise"
    assert "noise_masked=" in (folder / "delsys_cleaning_report.txt").read_text()
