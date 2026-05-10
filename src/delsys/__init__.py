"""Load and analyze Delsys CSV exports as ``pysampled.Data`` time series.

The package centers on :class:`Log`, which reads a CSV exported from EMGworks
or Trigno Discover, normalizes its many per-format quirks (header layouts,
sub-channel orderings, link-device asynchrony), resamples each channel to a
configurable per-modality target rate, and groups the result into per-sensor
modality bundles (:class:`EMG`, :class:`EKG`, :class:`IMU`, :class:`FSR`,
:class:`VO2Master`).

Classes:
    Log: Top-level loader; the only entry point most users need.

    Sensor: One physical Delsys sensor with all its modality bundles.
    Signal: Per-channel time series tagged with sensor / modality / sub-channel.

    EMG: Single, Duo, and Quattro EMG bundle with preprocessing and feature extraction.
    EKG: ECG bundle with R-peak detection and rate properties.
    IMU: Accelerometer / gyroscope bundle (X, Y, Z axes).
    FSR: Force-sensitive-resistor bundle (4 channels).
    VO2Master: Respiratory gas analyzer link device (8 channels).

    SensorInfo, SensorLog, SigInfoDelsys: Metadata namedtuples.

Constants:
    TARGET_SR: Default per-modality target sampling rate.
    SUBCHANNEL_MAP: Canonical sub-channel ordering per modality.
    APPLICATIONS: Recognized acquisition applications.
    LINK_DEVICE_REGISTRY: Sensor-name substring → (modality, synthetic
        sensor number) for link devices (VO2 Master, HR Strap).

Example:
    .. code-block:: python

        import delsys

        lf = delsys.Log("path/to/Trial.csv")
        for emg in lf.emg.split_by_signal_name():
            envelope = emg.process(amp_kind="envelope2")
        right_forearm = lf.find(side="R", location="Forearm")
"""

from delsys._constants import (
    APPLICATIONS,
    LINK_DEVICE_REGISTRY,
    SUBCHANNEL_MAP,
    TARGET_SR,
)
from delsys._metadata import SensorInfo, SensorLog, SigInfoDelsys
from delsys.cleaning import CleaningConfig, CleaningResult
from delsys.ekg import EKG
from delsys.emg import EMG
from delsys.log import Log
from delsys.sensor import Sensor
from delsys.signals import FSR, IMU, Signal, VO2Master

__version__ = "0.4.0"

__all__ = [
    # primary entry point
    "Log",
    # signal classes
    "Signal",
    "Sensor",
    "EMG",
    "EKG",
    "IMU",
    "FSR",
    "VO2Master",
    # metadata records
    "SensorInfo",
    "SensorLog",
    "SigInfoDelsys",
    # cleaning containers
    "CleaningConfig",
    "CleaningResult",
    # constants
    "TARGET_SR",
    "SUBCHANNEL_MAP",
    "APPLICATIONS",
    "LINK_DEVICE_REGISTRY",
]
