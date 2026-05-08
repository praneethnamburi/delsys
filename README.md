# delsys

Load CSV exports from the Delsys EMG system (EMGworks and Trigno Discover) into Python and prepare them for analysis.

## Install

```
pip install delsys
```

For local development:

```
git clone https://github.com/praneethnamburi/delsys
pip install -e ./delsys
```

## Quick start

```python
from delsys import Log

lf = Log("path/to/exported.csv", sensor_map="path/to/delsys_channelmap.txt")

# Standard views
lf.signals          # flat list, one entry per (sensor, modality, sub-channel)
lf.sensors          # one entry per physical sensor, modalities attached as attributes

# Typed accessors
lf.emg              # list of EMG bundles
lf.ekg              # list of EKG bundles
lf.acc, lf.gyro     # IMU bundles
lf.fsr              # FSR bundles
lf.analog
lf.vo2master        # VO2 Master link device
lf.hrstrap          # HR Strap link device

# Side accessors
lf.left, lf.right, lf.center

# Query API
lf.find(modality="EMG", side="R")
lf.find(location="Forearm")
lf.find(sensor_number=5)
lf.find(modality="EMG", as_="signal")    # raw Signal objects (one per sub-channel)
```

## Supported sensors

EMG (Avanti, Duo, Quattro, Snap-Lead), EKG, ACC, GYRO, FSR, Analog, VO2 Master (link), HR Strap (link).

## Supported export formats

- EMGworks
- Trigno Discover v1.4.2
- Trigno Discover v1.5.0+
- Trigno Discover v1.6.4 (with link devices and time-series columns)
- Trigno Discover v1.7.0

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `heartpy`, `neurokit2`, [`pysampled`](https://github.com/praneethnamburi/pysampled).

## License

MIT — see [LICENSE](LICENSE).
