"""Foundational metadata records used throughout the package.

Three small namedtuples that travel through the loader pipeline:

* :data:`SensorLog` — one row of a manual channelmap text file
  (``delsys_channelmap.txt``).
* :data:`SensorInfo` — combined metadata from the channelmap and the CSV
  header. Held by every :class:`Signal` and every :class:`Sensor`.
* :data:`SigInfoDelsys` — parsed information for a single CSV column,
  produced by :func:`_parse_sig_name`.

Living in their own module avoids circular imports between ``signals.py``
(which needs ``SensorInfo`` for ``Signal``) and ``sensor.py`` (which uses
``Signal`` and also needs ``SensorInfo``).
"""

from collections import namedtuple

# Manually entered sensor map record (one row per sensor in delsys_channelmap.txt).
# ``lrc`` is one of ``'L'``, ``'R'``, ``'C'`` (left/right/center) or ``None``.
SensorLog = namedtuple("SensorLog", "number type_sensorlog lrc location")

# Sensor metadata combined from the channelmap and the CSV header.
SensorInfo = namedtuple("SensorInfo", "name modalities number type_sensorlog lrc location")

# Parsed info for a single CSV column header.
SigInfoDelsys = namedtuple("SigInfoDelsys", "sensor_name modality sensor_number subchannel")
