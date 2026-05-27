"""Round-trip parity for the HDF5 native checkpoint (csv -> .h5 -> Log)."""

import numpy as np
import pytest

import delsys
from delsys._constants import TARGET_SR

# Trigno Discover fixtures span the supported header versions + link devices.
DISCOVER_FIXTURES = [
    "discover142.csv",
    "discover150.csv",
    "discover164_basic.csv",
    "discover164_mvc.csv",
    "discover164_link.csv",  # VO2 / HR link devices
    "discover170.csv",
]

# Every modality bundle Log can expose.
BUNDLES = ("emg", "ekg", "acc", "gyro", "fsr", "analog", "vo2master", "hrstrap")


@pytest.mark.parametrize("name", DISCOVER_FIXTURES)
def test_native_h5_roundtrip_parity(name, fixtures_dir, tmp_path):
    """Log(.h5) must reproduce Log(.csv) bitwise (within float32) for every bundle."""
    csv = str(fixtures_dir / name)
    ref = delsys.Log(csv, target_sr=TARGET_SR)

    h5 = str(tmp_path / name.replace(".csv", ".h5"))
    delsys.to_native_h5(csv, h5)
    got = delsys.Log(h5, target_sr=TARGET_SR)

    for m in BUNDLES:
        rb, gb = getattr(ref, m, None), getattr(got, m, None)
        assert (rb is None) == (gb is None), f"{m}: presence mismatch"
        if rb is None:
            continue
        ra, ga = np.asarray(rb._sig), np.asarray(gb._sig)
        assert ra.shape == ga.shape, f"{m}: shape {ra.shape} vs {ga.shape}"
        assert np.allclose(ra, ga, rtol=1e-4, atol=1e-5), f"{m}: values differ"
        assert list(rb.signal_names) == list(gb.signal_names), f"{m}: names differ"
        assert abs(rb.sr - gb.sr) < 1e-9, f"{m}: sr differs"

    # Sensor metadata + side queries survive the round trip.
    rmeta = sorted((s.number, s.location, s.lrc, tuple(sorted(s.modalities))) for s in ref.sensors)
    gmeta = sorted((s.number, s.location, s.lrc, tuple(sorted(s.modalities))) for s in got.sensors)
    assert rmeta == gmeta
    assert [s.number for s in ref.find(side="R")] == [s.number for s in got.find(side="R")]


def test_native_h5_default_outpath(fixtures_dir, tmp_path):
    """to_native_h5 defaults the output to <csv stem>.h5 and returns the path."""
    import shutil

    csv = tmp_path / "discover170.csv"
    shutil.copy(fixtures_dir / "discover170.csv", csv)
    out = delsys.to_native_h5(str(csv))
    assert out == str(csv.with_suffix(".h5"))
    assert (tmp_path / "discover170.h5").exists()


def test_h5_dispatch_via_suffix(fixtures_dir, tmp_path):
    """Constructing Log with an .h5 path takes the checkpoint path, not the CSV parser."""
    csv = str(fixtures_dir / "discover150.csv")
    h5 = str(tmp_path / "d150.h5")
    delsys.to_native_h5(csv, h5)
    lf = delsys.Log(h5)  # default target_sr
    # name is the recording's identity (source CSV stem), preserved across the
    # checkpoint regardless of the .h5 filename.
    assert lf.name == "discover150"
    assert lf.emg is not None


def test_clock_mul_applied_on_load(fixtures_dir, tmp_path):
    """clock_mul is a load-time argument: it scales the reconstructed sampling rate."""
    csv = str(fixtures_dir / "discover150.csv")
    h5 = str(tmp_path / "d150.h5")
    delsys.to_native_h5(csv, h5)
    # Native EMG (target None) lets clock_mul flow into sr_orig.
    native = {k: None for k in TARGET_SR}
    base = delsys.Log(h5, target_sr=native, clock_mul=1.0)
    scaled = delsys.Log(h5, target_sr=native, clock_mul=1.01)
    assert np.allclose(np.array(scaled.sr_orig), np.array(base.sr_orig) * 1.01)


# EMGworks' time window depends on target_sr, so the checkpoint stores the widest
# (min_sr=1) window and trims on reload. Verify parity at several reload targets,
# including a custom one whose min rate differs from the export's.
_EMG_DEFAULT = TARGET_SR
_EMG_CUSTOM = {
    "EMGS": 1440,
    "EMGD": 1440,
    "EMGQ": 1440,
    "ACC": 240,
    "GYRO": 240,
    "FSR": 240,
    "EKG": 1440,
    "Analog": 4800,
}  # min_sr=240


@pytest.mark.parametrize("target", [_EMG_DEFAULT, _EMG_CUSTOM], ids=["default", "custom_min240"])
def test_emgworks_roundtrip_parity(target, fixtures_dir, tmp_path):
    """EMGworks Log(.h5) reproduces Log(.csv) bitwise (within float32) for any target,
    via the superset-window-then-trim reload path."""
    csv = str(fixtures_dir / "emgworks.csv")
    ref = delsys.Log(csv, target_sr=target)
    h5 = str(tmp_path / "emgworks.h5")
    delsys.to_native_h5(csv, h5)
    got = delsys.Log(h5, target_sr=target)
    for m in BUNDLES:
        rb, gb = getattr(ref, m, None), getattr(got, m, None)
        assert (rb is None) == (gb is None), f"{m}: presence"
        if rb is None:
            continue
        ra, ga = np.asarray(rb._sig), np.asarray(gb._sig)
        assert ra.shape == ga.shape, f"{m}: shape {ra.shape} vs {ga.shape}"
        assert np.allclose(ra, ga, rtol=1e-4, atol=1e-5), f"{m}: values"


def test_emgworks_clock_mul_reload_rejected(fixtures_dir, tmp_path):
    """An EMGworks checkpoint's interp grid is fixed at clock_mul=1; a clock-shifted
    reload is rejected rather than silently returning a mismatched signal."""
    csv = str(fixtures_dir / "emgworks.csv")
    h5 = str(tmp_path / "emgworks.h5")
    delsys.to_native_h5(csv, h5)
    delsys.Log(h5, target_sr=TARGET_SR, clock_mul=1.0)  # ok
    with pytest.raises(NotImplementedError):
        delsys.Log(h5, target_sr=TARGET_SR, clock_mul=1.0001)
