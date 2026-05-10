"""CSV header parsing, version detection, and per-format dataframe parsers.

This module is internal: ``Log`` orchestrates everything in this file and
the public API never references it directly. Functions are exposed so the
parsers can be tested in isolation.

The three per-format dataframe parsers (``_parse_dataframe_emgworks``,
``_parse_dataframe_discover``, ``_parse_dataframe_discover_with_link``)
all return the same tuple shape::

    (t_min: float, t_max: float, sr_list: List[float], signals: List[Signal])
"""

import contextlib
import csv
import io
import re
from typing import IO, Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pysampled
from scipy.interpolate import interp1d
from scipy.signal import resample

from delsys._constants import APPLICATIONS, LINK_DEVICE_REGISTRY
from delsys._metadata import SensorInfo, SensorLog, SigInfoDelsys
from delsys.signals import Signal

# Trigno Base receives an integer number of samples per 13.5 ms frame, so
# every sampling rate is a multiple of 1/0.0135 Hz. We round measured /
# header-reported rates to that grid.
_TRIGNO_FRAME_INTERVAL_S = 0.0135

# Subchannel name mapping for Discover sensors that report 1/2/3/4 instead of A/B/C/D.
_NUMERIC_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

# Per-column sampling-rate extraction. The Discover-with-link variant accepts
# a leading minus because VO2 Master columns are reported with a "-1 Hz" tag.
_COLUMN_SR_RE = re.compile(r"\(([^)]+)Hz")
_COLUMN_SR_LINK_RE = re.compile(r"\(-?([^)]+)Hz")

# Pulls the first word after the colon in a Discover column header — used by
# the link parser to find the matching ``Time Series`` column.
_LINK_NAME_TAG_RE = re.compile(r":\s*(\w+)")


def _quantize_to_trigno_frame(sr_hz: float) -> float:
    """Round a sampling rate to the nearest Trigno-Base 13.5 ms frame multiple."""
    return round(sr_hz * _TRIGNO_FRAME_INTERVAL_S) / _TRIGNO_FRAME_INTERVAL_S


def _parse_sig_name(
    sensor_sig_name: str,
    application: str,
    target_sr: Dict[str, Optional[float]],
) -> SigInfoDelsys:
    """Parse one CSV column header into a :data:`SigInfoDelsys` record.

    Each Delsys CSV column header encodes ``"<sensor_name>: <signal_info>"``.
    EMGworks and Discover use different signal-info conventions, both handled
    here.

    Args:
        sensor_sig_name: One column header from the CSV — e.g.
            ``"EMG 01 04498 (54717): EMG 1 (mV)"`` for Discover or
            ``"EMG 1.A: EMG 1 ..."`` for EMGworks.
        application: ``'EMGworks'`` or ``'Trigno Discover'``.
        target_sr: Modality → target sampling rate map. Used to validate that
            the parsed modality is one we know how to handle.

    Returns:
        Namedtuple with ``sensor_name``, ``modality``, ``sensor_number``,
        ``subchannel``. ``modality`` is normalized: EMG single/duo/quattro
        become ``'EMGS'`` / ``'EMGD'`` / ``'EMGQ'``; FSR analog channels
        get reclassified to ``'FSR'``; link devices listed in
        :data:`LINK_DEVICE_REGISTRY` (VO2 Master, HR Strap) get their
        synthetic ``(modality, sensor_number)`` from the registry.

    Raises:
        AssertionError: If ``application`` is not in :data:`APPLICATIONS`.
        ValueError: If the parsed modality is not a key of ``target_sr``,
            or if the sub-channel for FSR / ACC / GYRO is unexpected.

    Example:
        >>> info = _parse_sig_name(
        ...     'EMG 01 04498 (54717): EMG 1 (mV)', 'Trigno Discover',
        ...     {'EMGS': 2000},
        ... )
        >>> info.modality, info.sensor_number, info.subchannel
        ('EMGS', 1, 'A')
    """
    assert application in APPLICATIONS

    if application == "EMGworks":
        sensor_name, modality, sensor_number, subchannel = _parse_sig_name_emgworks(sensor_sig_name)
    else:
        sensor_name, modality, sensor_number, subchannel = _parse_sig_name_discover(sensor_sig_name)

    if modality == "EMG":
        if "Quattro" in sensor_name:
            modality = "EMGQ"
            subchannel = _NUMERIC_TO_LETTER.get(subchannel, subchannel)
        elif "Duo" in sensor_name:
            modality = "EMGD"
            subchannel = _NUMERIC_TO_LETTER.get(subchannel, subchannel)
        else:
            modality = "EMGS"

    if modality not in target_sr:
        raise ValueError(
            f"Modality {modality!r} parsed from column {sensor_sig_name!r} "
            f"is not in target_sr keys {list(target_sr)}."
        )

    if modality == "FSR":
        assert subchannel in ("A", "B", "C", "D"), f"FSR subchannel {subchannel!r} not in A/B/C/D"
    if modality in ("ACC", "GYRO"):
        assert subchannel in ("X", "Y", "Z"), f"{modality} subchannel {subchannel!r} not in X/Y/Z"

    return SigInfoDelsys(sensor_name, modality, sensor_number, subchannel)


def _parse_sig_name_emgworks(ss_name: str) -> Tuple[str, str, int, str]:
    """Decode an EMGworks column header into ``(sensor_name, modality, sensor_number, subchannel)``."""
    sensor_name, sig_name = ss_name.split(": ")
    modality = sig_name.split(" ")[0].split(".")[0]
    sensor_number = int(sig_name.split(" ")[1])
    if "." not in sig_name:
        subchannel = "A"
    else:
        subchannel = sig_name.split(" ")[0].split(".")[1]
    return sensor_name, modality, sensor_number, subchannel


def _parse_sig_name_discover(ss_name: str) -> Tuple[str, str, int, str]:
    """Decode a Trigno Discover column header into ``(sensor_name, modality, sensor_number, subchannel)``."""
    sensor_name, sig_name = ss_name.split(": ")
    modality, subchannel = sig_name.split(" ")[:2]

    if "FSR" in sensor_name and modality == "Analog":
        modality = "FSR"

    # Link-device dispatch is registry-driven; first substring match wins.
    # Falls through to the Trigno-Base sensor-number parse only when no
    # link device matches.
    for substr, (link_mod, link_num) in LINK_DEVICE_REGISTRY.items():
        if substr in sensor_name:
            sensor_number = link_num
            subchannel = modality + subchannel
            modality = link_mod
            break
    else:
        try:
            sensor_number = int(sensor_name.split(" ")[1])
        except ValueError:
            sensor_number = int(sensor_name.split(" ")[-2].split("/")[0])

    if modality == "Analog":
        # Discover sometimes outputs 'Analog I' / 'Analog A' / etc; normalize to 'A'.
        subchannel = "A"
    if subchannel in _NUMERIC_TO_LETTER:
        subchannel = _NUMERIC_TO_LETTER[subchannel]

    return sensor_name, modality, sensor_number, subchannel


def _read_sensor_log(sensor_map_file: str) -> List[SensorLog]:
    """Read a Delsys channelmap text file into a list of :class:`SensorLog` records.

    The channelmap format is hand-rolled (not a Delsys export). One sensor
    per non-empty line, three fields separated by ``" - "`` (space-dash-space):

    .. code-block:: text

        Ch 1 - EMG - LBicep
        Ch 2 - EMG - RBicep
        Ch 5 - snap lead - RTricep
        Ch 11 - EKG - Chest
        Ch 12 - Sync - Optitrack Recording Gate

    Field semantics:

    - **Field 1** — any text whose last whitespace-token is the sensor's
      channel number (``"Ch 1"``, ``"Channel 01"``, and ``"1"`` all work).
    - **Field 2** — type tag, free text. Common values are ``"EMG"``,
      ``"snap lead"``, ``"EKG"``, ``"FSR"``, ``"Sync"``. The Sensor builder
      uses ``"FSR"`` to reclassify Discover Analog channels as FSR.
    - **Field 3** — location label. Its **first character** is taken as the
      ``lrc`` field (``"L"``, ``"R"``, ``"C"`` for left/right/center).
      Anything else still loads but won't match :meth:`Log.find` ``side=``
      filters.

    Lines without two ``" - "`` separators are silently skipped, as are
    blank lines. The file is read with ``utf-8-sig`` so a leading BOM is
    handled.

    A reference file lives at :file:`examples/delsys_channelmap.txt` in the
    repository.

    Args:
        sensor_map_file: Path to the channelmap text file.

    Returns:
        List of :class:`SensorLog` namedtuples — one per recognized line.

    Raises:
        ValueError: If a recognized line's first field has no parseable
            channel-number token (e.g. ``"Ch X - EMG - LBicep"``).
    """
    with open(sensor_map_file, "r", encoding="utf-8-sig") as f:
        sensor_map_raw = [x.split(" - ") for x in f.read().splitlines() if x]
    return [
        SensorLog(int(x[0].split(" ")[-1]), x[1], x[2][0], x[2].rstrip())
        for x in sensor_map_raw
        if len(x) > 1
    ]


def _parse_hdr(fname: str, sensor_name_replace: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Read the CSV header rows and return a dict describing the file format.

    Detects EMGworks vs. Trigno Discover from the first row and dispatches
    to the per-format reader.

    Args:
        fname: Path to the CSV file.
        sensor_name_replace: ``{corrupted_name: new_name}`` map applied to
            signal column names to repair sensor labels misspelled during
            data acquisition.

    Returns:
        Always contains ``'application'`` (``'EMGworks'`` or
        ``'Trigno Discover'``) and ``'skiprows'`` (rows pandas should skip
        when reading the data block). For Trigno Discover files it also
        contains:

        - ``'application_full'`` — full version string from row 1.
        - ``'application_version'`` — extracted from parens: ``'1.4.2'``,
          ``'1.5.0'``, ``'1.6.4'``, ``'1.7.0'``.
        - ``'datetime'`` — collection start time from row 2.
        - ``'duration_s'`` — collection length in seconds from row 3.
        - ``'sensor_name_mode'`` — ``{sensor_name: mode_int}``.
        - ``'sensor_signal_names'`` — synthesized list of per-column
          ``"<sensor>: <signal>"`` strings used as pandas column names.
    """
    if sensor_name_replace is None:
        sensor_name_replace = {}
    with open(fname, newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)
        if any("Trigno Discover" in x for x in first_row):
            return _parse_hdr_discover(reader, first_row, sensor_name_replace)
    return {"application": "EMGworks", "skiprows": 0}


def _parse_hdr_discover(
    reader: Iterator[List[str]],
    first_row: List[str],
    sensor_name_replace: Dict[str, str],
) -> Dict[str, Any]:
    """Decode the rows of a Trigno Discover CSV header (caller has already consumed row 1)."""
    hdr: Dict[str, Any] = {"application": "Trigno Discover"}
    hdr["application_full"] = first_row[-1].strip()
    s = hdr["application_full"]
    hdr["application_version"] = s[s.find("(") + 1 : s.find(")")]
    hdr["datetime"] = next(reader)[-1].strip()
    hdr["duration_s"] = float(next(reader)[-1].strip())

    sensor_names = [x.strip() for x in next(reader)]
    sensor_modes = [x.strip().removeprefix("sensor mode: ") for x in next(reader)]
    hdr["sensor_name_mode"] = {
        s_name: int(s_mode) for s_name, s_mode in zip(sensor_names, sensor_modes) if s_name
    }
    # Forward-fill blank cells in the sensor-name row (each sensor only labels
    # its first column; subsequent channels of the same sensor leave it blank).
    for idx, sensor_name in enumerate(sensor_names[1:]):
        if not sensor_name:
            sensor_names[idx + 1] = sensor_names[idx]

    signal_names_raw = [x.strip() for x in next(reader)]
    signal_names: List[str] = []
    analog_letters = ("A", "B", "C", "D")
    curr_letter_idx = 0
    for sn in signal_names_raw:
        if "Analog ?" in sn:  # Discover-export quirk: "?" placeholder for analog channel letter
            sn = sn.replace("?", analog_letters[curr_letter_idx])
            curr_letter_idx = (curr_letter_idx + 1) % len(analog_letters)
        signal_names.append(sn)

    signal_names = _fix_corrupted_sensor_names(signal_names, sensor_name_replace)
    sensor_names += [sensor_names[-1]] * (len(signal_names) - len(sensor_names))

    if hdr["application_version"] == "1.4.2":
        hdr["sensor_signal_names"] = [
            f"{sensor_name}: {signal_name}"
            for sensor_name, signal_name in zip(sensor_names, signal_names)
        ]
        hdr["skiprows"] = 6
    else:  # 1.5.0+, also covers 1.6.x and 1.7.0
        sampling_rates = next(reader)
        hdr["sensor_signal_names"] = [
            f"{sensor_name}: {signal_name} - ({sampling_rate.strip()})"
            for sensor_name, signal_name, sampling_rate in zip(
                sensor_names, signal_names, sampling_rates
            )
        ]
        hdr["skiprows"] = 7
    return hdr


def _fix_corrupted_sensor_names(
    sig_names: List[str],
    sensor_name_replace: Optional[Dict[str, str]],
) -> List[str]:
    """Apply prefix replacements to repair sensor names misspelled during acquisition.

    Args:
        sig_names: Column-name list to repair.
        sensor_name_replace: ``{corrupted_prefix: replacement_prefix}`` map.
            ``None`` is treated as empty (no-op).

    Returns:
        New list with prefixes rewritten in place; non-matching names pass
        through unchanged.
    """
    if not sensor_name_replace:
        return list(sig_names)
    sig_names_new: List[str] = []
    for sn in sig_names:
        sn_new = sn
        for corrupted_sensor_name, new_sensor_name in sensor_name_replace.items():
            if sn.startswith(corrupted_sensor_name):
                sn_new = sn.replace(corrupted_sensor_name, new_sensor_name)
        sig_names_new.append(sn_new)
    return sig_names_new


@contextlib.contextmanager
def _open_dropped_samples_log(path: Optional[str]) -> Iterator[IO[str]]:
    """Yield a writable text stream — file at ``path`` if given, discard buffer otherwise.

    Lets the discover parsers be called outside the :class:`Log` orchestrator
    without creating a side-channel file (useful in tests).
    """
    if path is None:
        yield io.StringIO()
    else:
        with open(path, "w") as f:
            yield f


def _detect_parser(hdr: Dict[str, Any], time_names: List[str]) -> str:
    """Inspect parsed header info and return a tag indicating which parser to use.

    Args:
        hdr: Header dict produced by :func:`_parse_hdr`.
        time_names: List of time-series column names (empty for EMGworks /
            Discover-basic).

    Returns:
        One of ``'emgworks'``, ``'discover_basic'``, ``'discover_link'``.

    Raises:
        ValueError: If link sensors (VO2 Master / HR Strap) are present but
            ``Time Series`` columns are missing — the link parser cannot
            resample link signals without timestamps.
    """
    if hdr["application"] == "EMGworks":
        return "emgworks"
    has_link = any("VO2" in x.strip() or "HR" in x.strip() for x in hdr["sensor_signal_names"])
    has_timestamps = bool(time_names)
    if has_link and not has_timestamps:
        raise ValueError(
            "Found link data (VO2 Master / HR Strap) but no Time Series columns. "
            "Re-export this file from Discover with Time Series enabled."
        )
    if has_link:
        return "discover_link"
    return "discover_basic"


# ---------------------------------------------------------------------------
# Per-channel helpers shared by the three dataframe parsers
# ---------------------------------------------------------------------------


def _sensors_by_number(sensors_info: Iterable[SensorInfo]) -> Dict[int, SensorInfo]:
    """Index ``sensors_info`` by sensor number for O(1) lookup inside per-channel loops."""
    return {s.number: s for s in sensors_info}


def _make_signal(
    sig_array: np.ndarray,
    sr: float,
    sig_info: SigInfoDelsys,
    sensors_by_number: Dict[int, SensorInfo],
    t0: float,
) -> Signal:
    """Construct a :class:`Signal` carrying its sensor / modality / subchannel in ``meta``."""
    return Signal(
        sig_array,
        sr,
        t0=t0,
        meta={
            "sensor": sensors_by_number[sig_info.sensor_number],
            "modality": sig_info.modality,
            "subchannel": sig_info.subchannel,
        },
    )


def _log_dropped_samples(f: IO[str], d: np.ndarray, sig_info: SigInfoDelsys) -> None:
    """Write one per-channel zero-sample report line to the open stream."""
    n_zeros = int(np.sum(d == 0))
    n_total = len(d)
    pct = (n_zeros / n_total) * 100 if n_total else 0.0
    f.write(
        f"{sig_info.sensor_name} {sig_info.sensor_number} {sig_info.modality} {sig_info.subchannel} - "
        f"{n_zeros} / {n_total} = {pct:5.2f}% \n"
    )


def _check_frame_count(duration: float, sr_list: List[float], n_rows: int) -> None:
    """Validate that ``n_rows`` matches the expected sample count for ``duration`` × ``max(sr)``.

    Uses ``raise`` instead of ``assert`` so the check survives ``python -O``.
    """
    expected = round(duration * max(sr_list))
    if expected != n_rows:
        raise ValueError(
            f"Frame-count invariant failed: expected round(duration * max_sr) = "
            f"round({duration} * {max(sr_list)}) = {expected}, got len(df) = {n_rows}."
        )


# ---------------------------------------------------------------------------
# Per-format dataframe parsers
# ---------------------------------------------------------------------------


def _parse_dataframe_emgworks(
    df: "Any",  # pandas.DataFrame; quoted to avoid an import-time pandas dep here
    signal_map: List[SigInfoDelsys],
    sensors_info: List[SensorInfo],
    target_sr: Dict[str, Optional[float]],
    sig_names: List[str],
    time_names: List[str],
    clock_mul: float,
    t0: float,
) -> Tuple[float, float, List[float], List[Signal]]:
    """Build :class:`Signal` objects from an EMGworks dataframe.

    EMGworks pairs every signal column with a per-channel time column.
    Each channel is independently sorted by time, dropped of NaNs,
    interpolated onto a uniform grid, and resampled to its modality's
    target rate.

    Returns:
        ``(t_min, t_max, sr_list, signals)``. ``sr_list`` holds the
        deduced (pre-resample) sampling rate per channel.
    """
    sensors_by_number = _sensors_by_number(sensors_info)
    t_min_list: List[float] = []
    t_max_list: List[float] = []
    sr_list: List[float] = []
    ts_list: List[np.ndarray] = []
    for t_name, s_name in zip(time_names, sig_names):
        ts = df[[t_name, s_name]].copy()
        ts.sort_values(by=t_name, inplace=True)
        ts = ts.dropna().to_numpy()
        t = ts[:, 0]
        t_min_list.append(t[0] / clock_mul)
        t_max_list.append(t[-1] / clock_mul)
        # Median (not mean) of np.diff(t) — robust to dropped samples.
        sr_deduced = 1 / np.median(np.diff(t))
        sr_list.append(_quantize_to_trigno_frame(sr_deduced) * clock_mul)
        ts_list.append(ts)
    min_sr = np.min(list(target_sr.values()))
    t_min = np.floor(np.min(t_min_list) * min_sr) / min_sr
    t_max = np.ceil(np.max(t_max_list) * min_sr) / min_sr

    signals: List[Signal] = []
    for ts, sr, sig_info in zip(ts_list, sr_list, signal_map):
        n_samples = int((t_max - t_min) * sr) + 1
        this_t_max = t_min + (n_samples - 1) / sr
        t = np.linspace(t_min, this_t_max, n_samples)
        sig = interp1d(ts[:, 0] / clock_mul, ts[:, 1], fill_value="extrapolate")(t)
        sr_targ = target_sr[sig_info.modality]
        sig_resampled = resample(sig, round(n_samples * sr_targ / sr))
        signals.append(_make_signal(sig_resampled, sr_targ, sig_info, sensors_by_number, t0))

    return t_min, t_max, sr_list, signals


def _parse_dataframe_discover(
    df: "Any",
    signal_map: List[SigInfoDelsys],
    sensors_info: List[SensorInfo],
    target_sr: Dict[str, Optional[float]],
    duration_hdr: float,
    clock_mul: float,
    t0: float,
    *,
    dropped_samples_path: Optional[str] = None,
) -> Tuple[float, float, List[float], List[Signal]]:
    """Build :class:`Signal` objects from a Trigno Discover dataframe (no link devices).

    Discover stores all channels on a fixed grid at the highest sensor's
    sampling rate; per-channel rates are recovered from the column headers
    and the actual sample count is bounded by ``round(sr * duration)``.

    Args:
        dropped_samples_path: If given, a per-channel zero-sample report is
            written to this path. ``None`` discards the report (useful for
            tests).

    Returns:
        ``(t_min, t_max, sr_list, signals)``.

    Raises:
        ValueError: If the row count doesn't match ``round(duration * max_sr)``.
    """
    sensors_by_number = _sensors_by_number(sensors_info)
    duration = duration_hdr / clock_mul
    column_sr = [float(_COLUMN_SR_RE.findall(x)[0].strip()) for x in list(df)]
    sr_list = [_quantize_to_trigno_frame(x) * clock_mul for x in column_sr]
    _check_frame_count(duration, sr_list, len(df))

    t_min = 0.0
    t_max = duration
    with _open_dropped_samples_log(dropped_samples_path) as f:
        signals: List[Signal] = []
        for ts_name, sr, sig_info in zip(df, sr_list, signal_map):
            sr_targ = target_sr[sig_info.modality]
            d = df[ts_name][: round(sr * duration) + 1].to_numpy()
            _log_dropped_samples(f, d, sig_info)
            base = pysampled.Data(d, sr=sr).interpnan()
            sig_resampled = base if sr_targ is None else base.resample(sr_targ)
            signals.append(
                _make_signal(
                    sig_resampled(),
                    sig_resampled.sr,
                    sig_info,
                    sensors_by_number,
                    t0,
                )
            )

    return t_min, t_max, sr_list, signals


def _parse_dataframe_discover_with_link(
    df: "Any",
    signal_map: List[SigInfoDelsys],
    sensors_info: List[SensorInfo],
    target_sr: Dict[str, Optional[float]],
    duration_hdr: float,
    clock_mul: float,
    t0: float,
    sig_names: List[str],
    time_names: List[str],
    *,
    dropped_samples_path: Optional[str] = None,
) -> Tuple[float, float, List[float], List[Signal]]:
    """Build :class:`Signal` objects from a Trigno Discover dataframe with link devices.

    Trigno-Base channels are handled like in :func:`_parse_dataframe_discover`
    (uniform grid at the per-column sampling rate). Link devices (VO2 Master,
    HR Strap) are resampled against their own ``Time Series`` columns via
    :func:`pysampled.uniform_resample`.

    Args:
        dropped_samples_path: See :func:`_parse_dataframe_discover`.

    Returns:
        ``(t_min, t_max, sr_list, signals)``.

    Raises:
        ValueError: If a link column has no matching ``Time Series`` column,
            or if the row count doesn't match ``round(duration * max_sr)``.

    Note:
        The 13.5 ms Trigno frame quantization only applies to base sensors,
        not link devices — link sampling rates pass through unchanged.
        See ``TODO.md`` for the planned cleanup.
    """
    sensors_by_number = _sensors_by_number(sensors_info)
    duration = duration_hdr / clock_mul
    column_sr = [float(_COLUMN_SR_LINK_RE.findall(x)[0].strip()) for x in sig_names]
    sr_list: List[float] = []
    t_min = 0.0
    t_max = duration

    link_time_names = [x for x in time_names if "VO2" in x.strip() or "HR" in x.strip()]

    with _open_dropped_samples_log(dropped_samples_path) as f:
        signals: List[Signal] = []
        for ts_name, sr_raw, sig_info in zip(sig_names, column_sr, signal_map):
            sr_targ = target_sr[sig_info.modality]
            if sig_info.modality in ("VO2", "HR"):
                # Link device: resample against its own time-series column.
                # NOTE: link signals can start delayed and end before the rest
                # of the recording; see TODO.md for proper zero-fill handling.
                if " Breathing Cycle" in ts_name:
                    continue  # data from breathing cycle is not useful

                tag_match = _LINK_NAME_TAG_RE.findall(ts_name)
                if not tag_match:
                    raise ValueError(f"Could not parse link-channel tag from column {ts_name!r}.")
                time_name = next(
                    (name for name in link_time_names if tag_match[0] in name),
                    None,
                )
                if time_name is None:
                    raise ValueError(
                        f"No Time Series column found matching tag {tag_match[0]!r} "
                        f"for link channel {ts_name!r}. Available: {link_time_names!r}."
                    )

                sr = sr_raw
                d_signal = df[ts_name][: round(sr * duration) + 1].to_numpy()
                d_time = df[time_name][: round(sr * duration) + 1].to_numpy()
                sig_resampled = pysampled.uniform_resample(d_time, d_signal, sr_targ, t_min, t_max)
                d = d_signal  # for the dropped-samples report below

            else:
                # Trigno Base: 13.5 ms frame quantization applies.
                sr = _quantize_to_trigno_frame(sr_raw) * clock_mul
                d = df[ts_name][: round(sr * duration) + 1].to_numpy()
                base = pysampled.Data(d, sr=sr).interpnan()
                sig_resampled = base if sr_targ is None else base.resample(sr_targ)

            sr_list.append(sr)
            _log_dropped_samples(f, d, sig_info)
            signals.append(
                _make_signal(
                    sig_resampled(),
                    sig_resampled.sr,
                    sig_info,
                    sensors_by_number,
                    t0,
                )
            )

        _check_frame_count(duration, sr_list, len(df))

    return t_min, t_max, sr_list, signals
