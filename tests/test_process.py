"""Batch process() + smart channelmap resolution."""

import shutil
from pathlib import Path

import pytest

import delsys
from delsys._process import _csv_channel_numbers, read_channelmap

FIXTURE = "discover164_basic.csv"  # 10 sensors, no link devices


def _write_channelmap(path, numbers, name_replace=None):
    lines = [f"Ch {n} - EMG - Loc{n}" for n in sorted(numbers)]
    if name_replace:
        lines += ["", "[sensor_name_replace]"]
        lines += [f"{k} = {v}" for k, v in name_replace.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def trial_csv(fixtures_dir, tmp_path):
    """A copy of the fixture named Trial_1.csv, plus its true channel-number set."""
    csv = tmp_path / "Trial_1.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    return csv, _csv_channel_numbers(str(csv), None)


def test_process_builds_and_is_idempotent(trial_csv, tmp_path):
    csv, nums = trial_csv
    _write_channelmap(tmp_path / "delsys_channelmap.txt", nums)

    res = delsys.process(str(tmp_path))
    assert res[str(csv)] == "built"
    h5 = tmp_path / "Trial_1.h5"
    assert h5.exists()
    assert delsys.Log(str(h5)).emg is not None  # reloads

    # second run skips the existing checkpoint
    assert delsys.process(str(tmp_path))[str(csv)] == "hit"
    # ...unless forced
    assert delsys.process(str(tmp_path), overwrite=True)[str(csv)] == "built"


def test_process_report_written(trial_csv, tmp_path):
    csv, nums = trial_csv
    _write_channelmap(tmp_path / "delsys_channelmap.txt", nums)
    delsys.process(str(tmp_path))
    report = tmp_path / "delsys_process_report.txt"
    assert report.exists() and "Trial_1.csv - built" in report.read_text()


def test_strict_skips_mismatched_channelmap(trial_csv, tmp_path):
    csv, _ = trial_csv
    _write_channelmap(tmp_path / "delsys_channelmap.txt", {91, 92, 93})  # wrong numbers
    res = delsys.process(str(tmp_path))
    assert res[str(csv)].startswith("skipped")
    assert not (tmp_path / "Trial_1.h5").exists()


def test_lenient_picks_content_matching_map(trial_csv, tmp_path):
    csv, nums = trial_csv
    # default (name-pick) is wrong; an alternate map actually matches the CSV.
    _write_channelmap(tmp_path / "delsys_channelmap.txt", {91, 92})
    _write_channelmap(tmp_path / "delsys_channelmap_alt.txt", nums)
    assert delsys.process(str(tmp_path))[str(csv)].startswith("skipped")  # strict
    assert delsys.process(str(tmp_path), channelmap_policy="lenient")[str(csv)] == "built"


def test_trial_range_override(trial_csv, tmp_path):
    csv, nums = trial_csv  # Trial_1
    _write_channelmap(tmp_path / "delsys_channelmap.txt", {91, 92})  # wrong default
    _write_channelmap(tmp_path / "delsys_channelmap_Trial_1_5.txt", nums)  # covers trial 1
    assert delsys.process(str(tmp_path))[str(csv)] == "built"


def test_channelmap_one_folder_up(fixtures_dir, tmp_path):
    """The pia02/chi01 layout: CSVs in a subfolder, channelmap one level above."""
    sub = tmp_path / "delsys"
    sub.mkdir()
    csv = sub / "Trial_1.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    _write_channelmap(tmp_path / "delsys_channelmap.txt", _csv_channel_numbers(str(csv), None))
    res = delsys.process(str(sub), channelmap_search_parents=1)
    assert res[str(csv)] == "built"


def test_read_channelmap_parses_corrections(tmp_path):
    cm = tmp_path / "delsys_channelmap.txt"
    _write_channelmap(cm, {1, 2}, name_replace={"Avanti Sensor 1 (88016)": "EMG 01 03558 (88016)"})
    sensor_map, name_replace = read_channelmap(str(cm))
    assert {sl.number for sl in sensor_map} == {1, 2}
    assert name_replace == {"Avanti Sensor 1 (88016)": "EMG 01 03558 (88016)"}


def test_invalid_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        delsys.process(str(tmp_path), channelmap_policy="bogus")


def test_progress_summary_and_silence(trial_csv, tmp_path, capsys):
    csv, nums = trial_csv
    _write_channelmap(tmp_path / "delsys_channelmap.txt", nums)
    delsys.process(str(tmp_path), progress=True)
    out = capsys.readouterr().out
    assert "delsys.process:" in out and "built 1" in out

    delsys.process(str(tmp_path), overwrite=True, progress=False)
    assert capsys.readouterr().out == ""
