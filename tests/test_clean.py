"""Batch ``delsys.clean()`` + the per-log decision sidecar + noise-event consumption.

The headline property is the reproducibility contract
``cleaned.h5 = f(raw.h5, <stem>.delsys-artifact)``: a first pass freezes the
auto-chosen decision into a per-log ``Trial_*.delsys-artifact`` sidecar and a
replay reproduces the cleaned checkpoint bit-for-bit. ``discover170.csv`` is the
one fixture used end-to-end
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
from delsys._clean import (  # noqa: E402
    _config_body,
    read_decision,
    upsert_decision,
    write_decision,
)
from delsys._noise import (  # noqa: E402
    apply_noise_mask,
    read_noise_intervals,
    sidecar_path_for,
    write_noise_sidecar,
)

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
    assert (folder / "Trial_5.delsys-artifact").exists()
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
# Per-log decision sidecar + reproducibility contract
# ---------------------------------------------------------------------------


def test_decision_sidecar_captures_auto_decision(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)

    # The decision lives in a per-log sibling, not a per-folder manifest.
    assert (folder / "Trial_5.delsys-artifact").exists()
    entry = read_decision(raw)
    assert isinstance(entry["ecg_components_to_remove"], list)
    assert entry["splice_source"] == "combined"
    assert entry["motion"] == "auto"
    assert entry["noise_event_ref"] is None
    assert entry["accept"] is None  # not yet reviewed
    assert "preprocess_highpass_hz" in entry["config"]
    # The selection fields never leak into the config body.
    assert "ecg_components_to_remove" not in entry["config"]
    assert "ecg_auto_remove_components" not in entry["config"]


def test_replay_from_sidecar_is_byte_identical(raw_checkpoint):
    """First pass (auto) then a forced replay (sidecar) produce identical output."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    sig1 = _signature(folder / "Trial_5_cleaned.h5")

    delsys.clean(str(folder), overwrite=True, progress=False)
    sig2 = _signature(folder / "Trial_5_cleaned.h5")

    assert np.array_equal(sig1, sig2)


def test_sidecar_drives_output(raw_checkpoint):
    """``cleaned.h5 = f(raw.h5, .delsys-artifact)``: a hand-authored decision
    reproduces a direct ``clean_emg_ekg_artifact`` run with the same explicit set."""
    raw, folder = raw_checkpoint

    # Author a decision that pins ECG component [0] (not what auto would pick).
    write_decision(
        raw,
        {
            "ecg_components_to_remove": [0],
            "splice_source": "combined",
            "motion": "auto",
            "noise_event_ref": None,
            "config": _config_body(CleaningConfig()),
        },
    )

    delsys.clean(str(folder), progress=False)
    via_clean = _signature(folder / "Trial_5_cleaned.h5")

    # The decision must be preserved verbatim (existing sidecar, not re-frozen).
    assert read_decision(raw)["ecg_components_to_remove"] == [0]

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


def test_editing_sidecar_changes_output(raw_checkpoint):
    """Editing the frozen decision changes the cleaned checkpoint on re-run."""
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    auto_sig = _signature(folder / "Trial_5_cleaned.h5")

    entry = read_decision(raw)
    auto_components = entry["ecg_components_to_remove"]
    # Force a different decision: remove no ECG components at all.
    entry["ecg_components_to_remove"] = []
    write_decision(raw, entry)

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

    entry = read_decision(raw)
    entry["accept"] = False
    write_decision(raw, entry)

    res = delsys.clean(str(folder), overwrite=True, progress=False)
    assert res[str(raw)].startswith("skipped: rejected")


def test_record_decisions_false_writes_no_sidecar(raw_checkpoint):
    raw, folder = raw_checkpoint
    res = delsys.clean(str(folder), record_decisions=False, progress=False)
    assert res[str(raw)] == "cleaned"
    assert (folder / "Trial_5_cleaned.h5").exists()
    assert not (folder / "Trial_5.delsys-artifact").exists()


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

    # First pass freezes the decision; then point the trial at the noise event.
    delsys.clean(str(folder), progress=False)
    entry = read_decision(raw)
    entry["noise_event_ref"] = {"path": "noise.json", "key": "Trial_5"}
    write_decision(raw, entry)

    delsys.clean(str(folder), overwrite=True, progress=False)
    report = (folder / "delsys_cleaning_report.txt").read_text()
    assert "noise_masked=" in report


def test_clean_auto_consumes_sibling_delsys_noise(raw_checkpoint):
    """A sibling ``<stem>.delsys-noise`` is consumed by default and recorded as
    provenance in the decision's ``noise_event_ref``."""
    from delsys._noise import format_signal_key, sidecar_path_for, write_noise_sidecar

    raw, folder = raw_checkpoint

    # Author a sidecar next to the checkpoint, addressing one real signal.
    key = format_signal_key(delsys.Log(str(raw)).signals[0])
    write_noise_sidecar(sidecar_path_for(str(raw)), {key: [[0.02, 0.05]]})

    res = delsys.clean(str(folder), progress=False)
    assert res[str(raw)] == "cleaned"

    # Defaulted ref points at the sibling sidecar (provenance, not the windows).
    entry = read_decision(raw)
    assert entry["noise_event_ref"] == "Trial_5" + ".delsys-noise"
    assert "noise_masked=" in (folder / "delsys_cleaning_report.txt").read_text()


# ---------------------------------------------------------------------------
# 0.5.0 — upsert_decision (interactive review-cleaning write-back)
# ---------------------------------------------------------------------------


def test_upsert_decision_writes_explicit_entry(raw_checkpoint):
    raw, folder = raw_checkpoint
    cfg = CleaningConfig()
    path = upsert_decision(
        str(raw),
        components=[2, 0, 2],  # de-duplicated + sorted on write
        config=cfg,
        splice_source="ekgonly",
        motion=None,
    )
    assert path == str(folder / "Trial_5.delsys-artifact")

    entry = read_decision(raw)
    assert entry["ecg_components_to_remove"] == [0, 2]
    assert entry["splice_source"] == "ekgonly"
    assert entry["motion"] is None
    assert entry["accept"] is True
    assert entry["noise_event_ref"] is None  # no sibling sidecar present
    # Stored config body matches and omits the top-level selection fields.
    assert entry["config"] == _config_body(cfg)
    assert "ecg_components_to_remove" not in entry["config"]


def test_upsert_decision_defaults_noise_ref_to_sibling_sidecar(raw_checkpoint):
    raw, folder = raw_checkpoint
    write_noise_sidecar(sidecar_path_for(str(raw)), {"3.EMGS": [[0.1, 0.2]]})

    upsert_decision(str(raw), components=[1], config=CleaningConfig())
    entry = read_decision(raw)
    assert entry["noise_event_ref"] == "Trial_5.delsys-noise"


def test_upsert_decision_marks_cleaned_snapshot_stale(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    cleaned = folder / "Trial_5_cleaned.h5"
    assert cleaned.exists()

    upsert_decision(str(raw), components=[1], config=CleaningConfig())
    assert not cleaned.exists()  # stale snapshot cleared


def test_upsert_decision_mark_stale_false_keeps_snapshot(raw_checkpoint):
    raw, folder = raw_checkpoint
    delsys.clean(str(folder), progress=False)
    cleaned = folder / "Trial_5_cleaned.h5"

    upsert_decision(str(raw), components=[1], config=CleaningConfig(), mark_stale=False)
    assert cleaned.exists()


def test_upsert_decision_is_replayed_by_clean(raw_checkpoint):
    """A hand-written decision is honored verbatim on the next clean: the entry
    is replayed (not re-frozen) and the cleaned snapshot regenerates."""
    raw, folder = raw_checkpoint
    upsert_decision(str(raw), components=[0], config=CleaningConfig(), motion=None)

    res = delsys.clean(str(folder), progress=False)
    assert res[str(raw)] == "cleaned"
    assert (folder / "Trial_5_cleaned.h5").exists()
    # clean replayed our explicit decision rather than overwriting it.
    entry = read_decision(raw)
    assert entry["ecg_components_to_remove"] == [0]
    assert entry["motion"] is None


def test_upsert_decision_rejects_bad_splice(raw_checkpoint):
    raw, _ = raw_checkpoint
    with pytest.raises(ValueError):
        upsert_decision(str(raw), components=[], config=CleaningConfig(), splice_source="bogus")
