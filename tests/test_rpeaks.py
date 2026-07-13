"""EKG rpeak sidecar glue (:mod:`delsys._rpeaks`) + ``Log.ekg`` auto-load.

Addressing, save/apply against ``<stem>.delsys-events`` (the ``rpeaks`` type +
shared ``noise`` track), and the ``Log.ekg`` / ``Log.ekg_raw`` accessors. The
grid-independent *reproduction* of a decision is covered in ``test_ekg.py``;
here we exercise the file glue and the accessor wiring. Detection needs a long
signal, so real-Log cases use short fixtures only for stamping / no-op / guard
behaviour (their EKG is too short for HeartPy).
"""

import shutil
import warnings

import matplotlib

matplotlib.use("Agg")

import neurokit2 as nk  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import delsys  # noqa: E402
from delsys import EKG, SensorInfo, _events, _rpeaks  # noqa: E402

FIXTURE = "discover170.csv"


def _synth_ekg(sr=200, dur=30, number=12, location="Chest", seed=1):
    sig = np.asarray(
        nk.ecg_simulate(duration=dur, sampling_rate=sr, heart_rate=70, random_state=seed),
        dtype=float,
    ).reshape(-1, 1)
    s = SensorInfo(
        name=f"ekg{number}", modalities={"EKG"}, number=number,
        type_sensorlog=None, lrc="C", location=location,
    )
    return EKG(sig, sr=sr, t0=0.0, meta={"sensor": s},
               signal_names=[location], signal_coords=["ekg"])


def _curate(ekg):
    ekg.find_rpeaks()
    removed_idx = ekg.meta["rpeaks_idx_default"][3]
    ekg.meta["rpeaks_idx_removed"] = sorted(set(ekg.meta["rpeaks_idx_removed"]) | {removed_idx})
    mid_t = float((ekg.t[ekg.meta["rpeaks_idx_default"][1]]
                   + ekg.t[ekg.meta["rpeaks_idx_default"][2]]) / 2)
    ekg.meta["rpeaks_idx_added"] = [int(np.argmin(np.abs(ekg.t - mid_t)))]
    ekg.meta["tags"] = ["reviewed"]
    ekg.meta["noisy_segments_idx"] = ekg._windows_to_idx_pairs([[12.0, 13.0]])
    return removed_idx, mid_t


def test_ekg_channel_address():
    assert _rpeaks.ekg_channel_address(_synth_ekg()) == "12.EKG.A | Chest"
    assert _rpeaks.ekg_channel_address(_synth_ekg(number=7, location="Neck")) == "7.EKG.A | Neck"


def test_save_apply_roundtrip_reproduces_curation(tmp_path):
    src = str(tmp_path / "Trial_1.h5")
    sidecar = str(tmp_path / ("Trial_1" + _events.EVENTS_SUFFIX))
    ekg = _synth_ekg()
    ekg.meta["source"] = src
    removed_idx, mid_t = _curate(ekg)
    curated = sorted(float(ekg.t[i]) for i in ekg._get_rpeaks_from_meta())

    assert ekg.save_rpeaks() == sidecar
    assert list(_events.read_rpeaks_signals(sidecar)) == ["12.EKG.A | Chest"]
    assert list(_events.read_noise_signals(sidecar)) == ["12.EKG.A | Chest"]

    fresh = _synth_ekg()
    fresh.meta["source"] = src
    assert fresh.load_rpeaks() is True
    assert fresh.meta["tags"] == ["reviewed"]
    curated2 = sorted(float(fresh.t[i]) for i in fresh._get_rpeaks_from_meta())
    tol = 2.0 / 200
    assert len(curated) == len(curated2)
    for x in curated:
        assert min(abs(x - y) for y in curated2) <= tol
    # noisy segment carried through the sidecar's shared noise track
    clean = sorted(float(t) for t in fresh.t[fresh.rpeak_times()[0]])
    assert all(not (12.0 <= t <= 13.0) for t in clean)


def test_load_rpeaks_no_matching_sensor_is_noop(tmp_path):
    src = str(tmp_path / "Trial_1.h5")
    ekg = _synth_ekg(number=12)
    ekg.meta["source"] = src
    _curate(ekg)
    ekg.save_rpeaks()

    other = _synth_ekg(number=99)
    other.meta["source"] = src
    assert other.load_rpeaks() is False


def test_save_rpeaks_preserves_other_sections(tmp_path):
    sidecar = str(tmp_path / ("Trial_1" + _events.EVENTS_SUFFIX))
    # pre-existing marker + a different sensor's noise
    _events.write_events(sidecar, {
        "1": {"size": 1, "signals": {"3.EMGS | Fore": [[1.0]]}},
        "noise": {"signals": {"3.EMGS | Fore": {"windows": [[0.1, 0.2]]}}},
    })
    ekg = _synth_ekg()
    ekg.meta["source"] = str(tmp_path / "Trial_1.h5")
    _curate(ekg)
    ekg.save_rpeaks()
    assert _events.marker_types(sidecar) == ["1"]  # marker preserved
    noise = _events.read_noise_signals(sidecar)
    assert "3.EMGS | Fore" in noise and "12.EKG.A | Chest" in noise  # both kept


def test_save_rpeaks_without_source_raises():
    ekg = _synth_ekg()
    with pytest.raises(ValueError, match="no source path"):
        ekg.save_rpeaks()


def test_log_ekg_stamps_source_and_bypass(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_9.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    lf = delsys.Log(str(csv))
    e = lf.ekg
    assert e is not None and e.n_signals() == 1
    assert e.meta.get("source") == str(csv)
    # no sidecar -> no detection triggered
    assert len(e.meta.get("rpeaks_idx_default", [])) == 0
    assert lf.ekg_raw.meta.get("source") == str(csv)


def test_log_ekg_guards_bad_sidecar(fixtures_dir, tmp_path):
    csv = tmp_path / "Trial_9.csv"
    shutil.copy(fixtures_dir / FIXTURE, csv)
    lf = delsys.Log(str(csv))
    addr = _rpeaks.ekg_channel_address(lf.ekg_raw)
    # the fixture EKG is too short for HeartPy: apply must warn, not raise
    _events.write_events(
        _events.events_path_for(str(csv)),
        {"rpeaks": {"signals": {addr: {"added": [0.01], "detector": {"name": "pn"}}}}},
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e = lf.ekg
        assert e is not None
        assert any("rpeak sidecar" in str(wi.message) for wi in w)
