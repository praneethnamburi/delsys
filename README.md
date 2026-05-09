# delsys

[![src](https://img.shields.io/badge/src-github-blue)](https://github.com/praneethnamburi/delsys)
[![PyPI - Version](https://img.shields.io/pypi/v/delsys.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/delsys/)
[![Documentation Status](https://readthedocs.org/projects/delsys/badge/?version=latest)](https://delsys.readthedocs.io)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/delsys/main/LICENSE)

*Load Delsys CSV exports into Python as `pysampled.Data` time series.*

`delsys` reads CSV files exported from EMGworks and Trigno Discover, normalizes their
many per-format quirks (header layouts, sub-channel orderings, link-device
asynchrony), resamples each channel to a configurable per-modality target
rate, and groups the result into per-sensor modality bundles (EMG, EKG,
IMU, FSR, VO2 Master, HR Strap, Analog) ready for analysis.

## Installation

```sh
pip install delsys
```

For local development:

```sh
git clone https://github.com/praneethnamburi/delsys
pip install -e "./delsys[dev]"
```

## Quickstart

```python
import delsys

lf = delsys.Log("path/to/Trial.csv", sensor_map="path/to/delsys_channelmap.txt")

# Direct accessors — list of typed bundles, one per sensor with that modality.
lf.emg                            # list of EMG bundles
lf.ekg                            # list of EKG bundles
lf.acc, lf.gyro                   # tri-axial IMU bundles
lf.fsr                            # 4-channel FSR bundles
lf.analog                         # raw Analog signals
lf.vo2master                      # VO2 Master link device (8 channels)
lf.hrstrap                        # HR Strap link device

# Side accessors return whole Sensor objects.
lf.left, lf.right, lf.center

# Filtered queries.
lf.find(modality="EMG", side="R")              # right-side EMG bundles
lf.find(location="Forearm")                    # any sensor at "Forearm"
lf.find(sensor_number=5)
lf.find(modality="EMG", as_="signal")          # raw per-channel Signal objects

# A typical EMG envelope pipeline.
for emg in lf.emg:
    envelope = emg.process(amp_kind="envelope2")
    rms = emg.rms(envelope_sr=240)             # clean RMS amplitude pipeline
```

See the full API reference at <https://delsys.readthedocs.io>.

## Channelmap files (optional)

When you pass `sensor_map="path/to/delsys_channelmap.txt"` to `Log()`, that
file labels each sensor number with a sensor type and a body-location tag.
This lets you query by side (`lf.find(side="R")`) and by location
(`lf.find(location="Forearm")`).

The format is one sensor per line, three fields separated by `" - "`:

```
Ch 1 - EMG - LBicep
Ch 2 - EMG - RBicep
Ch 11 - EKG - Chest
Ch 12 - Sync - Optitrack Recording Gate
Ch 19 - Quattro - LForearmExtensors (A-Index, B-Middle, C-Ring, D-Little)
Ch 21 - FSR - LFoot (1-Heel, 2-OuterEdge, 3-Ball, 4-Toe)
```

- **Field 1**: any text whose last whitespace-token is the sensor's channel
  number (`Ch 1`, `Channel 01`, `1` all work).
- **Field 2**: a type tag (free text — common values: `EMG`, `Quattro`,
  `Snap`, `EKG`, `FSR`, `Sync`).
- **Field 3**: a location label. Its *first character* is interpreted as the
  side (`L`/`R`/`C` for left/right/center). Anything else still loads but
  won't match `lf.find(side=...)`.

Trailing parenthetical notes are informational only — they remain in
`location` but the parser doesn't extract sub-channel labels from them.
Blank lines and lines without two `" - "` separators are silently skipped.

A more comprehensive reference file lives at
[`examples/delsys_channelmap.txt`](https://github.com/praneethnamburi/delsys/blob/main/examples/delsys_channelmap.txt).

## Supported export formats

- EMGworks
- Trigno Discover 1.4.2
- Trigno Discover 1.5.0
- Trigno Discover 1.6.4 (with and without link devices)
- Trigno Discover 1.7.0

## Supported sensors

EMG (Avanti single, Duo, Quattro, Snap-Lead), EKG, ACC, GYRO, FSR, Analog,
VO2 Master (link), HR Strap (link).

## Scope and contributions

Supported sensor types are limited to those the maintainer has access to.
Delsys ships other hardware (e.g. SmO2/Thb appear as stubs in `TARGET_SR`
but are not exercised end-to-end). Contributions adding parsers and tests
for additional sensors are very welcome.

If you'd like to contribute, the dev install above pulls `pytest`, `black`,
and `isort`. Format with `black` and `isort` before opening a PR:

```sh
isort src/ tests/ scripts/
black src/ tests/ scripts/
pytest
```

## License

Distributed under the MIT License. See [LICENSE](https://github.com/praneethnamburi/delsys/blob/main/LICENSE) for details.

## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project link: <https://github.com/praneethnamburi/delsys>

## Acknowledgments

This package was developed as part of the ImmersionToolbox initiative at the
[MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for
supporting this initiative.
