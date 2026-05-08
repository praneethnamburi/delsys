"""Module-level constants and shared types for the delsys package."""
from collections import namedtuple


# Standard target sampling rate per modality. Used to normalize raw signals
# during loading; users can override by passing target_sr= to Log().
TARGET_SR = {
    'EMGS': 1920, 'EMGD': 1920, 'EMGQ': 1920,
    'ACC': 120, 'GYRO': 120,
    'FSR': 120, 'EKG': 120,
    'Analog': 2400,
    'SmO2': 5, 'Thb': 5,
    'VO2': 1, 'HR': 1,
}

# For each modality, the canonical sub-channel order. Used when stacking
# signals into multi-channel modality bundles (IMU, FSR, VO2Master).
SUBCHANNEL_MAP = {
    'EMGS': ('A',),
    'EMGD': ('A', 'B'),
    'EMGQ': ('A', 'B', 'C', 'D'),
    'ACC': ('X', 'Y', 'Z'),
    'GYRO': ('X', 'Y', 'Z'),
    'FSR': ('A', 'B', 'C', 'D'),
    'EKG': ('A',),
    'Analog': ('A',),
    'VO2': (
        'BreathingCycle', 'Resp.Rate', 'TidalVol.', 'Ventilation(L/min)',
        'FeO2(%)', 'VO2Absolute', 'AmbientPressure', 'FlowSensor', 'OxygenSensor',
    ),
    'HR': ('HeartRate',),
}

APPLICATIONS = ('EMGworks', 'Trigno Discover')

# Placeholder sensor numbers for Delsys link devices (VO2 Master, HR Strap).
# Chosen to not collide with Trigno-base sensor numbers (typically 1-16).
VO2_SENSOR_NUM = 900
HR_SENSOR_NUM = 901

# Parsed signal info from a CSV column header
SigInfoDelsys = namedtuple('SigInfoDelsys', 'sensor_name modality sensor_number subchannel')
