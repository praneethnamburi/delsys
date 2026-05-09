"""delsys — Load and analyze Delsys EMG-system CSV exports.

Top-level exports::

    from delsys import Log
    lf = Log("path/to/Trial.csv")

    lf.emg          # list of EMG bundles
    lf.find(...)    # query for sensors / signals / modality bundles
"""
from delsys._constants import (
    APPLICATIONS,
    HR_SENSOR_NUM,
    SUBCHANNEL_MAP,
    TARGET_SR,
    VO2_SENSOR_NUM,
)
from delsys._metadata import SensorInfo, SensorLog, SigInfoDelsys
from delsys._util import decreturn
from delsys.ekg import EKG
from delsys.emg import EMG
from delsys.log import Log
from delsys.sensor import Sensor
from delsys.signals import FSR, IMU, Signal, VO2Master

__all__ = [
    # primary
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
    # utilities
    "decreturn",
    # constants
    "TARGET_SR",
    "SUBCHANNEL_MAP",
    "APPLICATIONS",
    "VO2_SENSOR_NUM",
    "HR_SENSOR_NUM",
]

__version__ = "0.1.0"
