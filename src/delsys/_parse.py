"""CSV header parsing, version detection, and per-format dataframe parsers."""
import contextlib
import csv
import io
import re

import numpy as np
import pysampled
from scipy.interpolate import interp1d
from scipy.signal import resample

from delsys._constants import APPLICATIONS, SigInfoDelsys, VO2_SENSOR_NUM, HR_SENSOR_NUM
from delsys.signals import Signal, SensorLog


def _parse_sig_name(sensor_sig_name, application, target_sr):
    """Parse one CSV column header into a ``SigInfoDelsys`` record."""
    assert application in APPLICATIONS

    def _parse_sig_name_emgworks(ss_name):
        """Parse signal name written by EMGWorks."""
        sensor_name, sig_name = ss_name.split(': ')
        modality = sig_name.split(' ')[0].split('.')[0]
        sensor_number = int(sig_name.split(' ')[1])
        if '.' not in sig_name:
            subchannel = 'A'
        else:
            subchannel = sig_name.split(' ')[0].split('.')[1]
        return sensor_name, modality, sensor_number, subchannel

    def _parse_sig_name_discover(ss_name):
        sensor_name, sig_name = ss_name.split(': ')
        modality, subchannel = sig_name.split(' ')[:2]

        if 'FSR' in sensor_name and modality == 'Analog':
            modality = 'FSR'

        if 'VO2 Master' not in sensor_name and 'HR Strap' not in sensor_name:
            try:
                sensor_number = int(sensor_name.split(' ')[1])
            except ValueError:
                sensor_number = int(sensor_name.split(' ')[-2].split('/')[0])
        else:
            if 'VO2 Master' in sensor_name:
                sensor_number = VO2_SENSOR_NUM
                subchannel = modality + subchannel
                modality = 'VO2'
            if 'HR' in sensor_name:
                sensor_number = HR_SENSOR_NUM
                subchannel = modality + subchannel
                modality = 'HR'

        if modality == 'Analog':
            # Discover sometimes outputs 'Analog I' / 'Analog A' / etc; normalize to 'A'
            subchannel = 'A'
        subchannel_remap = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        if subchannel in subchannel_remap:
            subchannel = subchannel_remap[subchannel]

        return sensor_name, modality, sensor_number, subchannel

    if application == 'EMGworks':
        ret = _parse_sig_name_emgworks(sensor_sig_name)
    else:
        ret = _parse_sig_name_discover(sensor_sig_name)

    sensor_name, modality, sensor_number, subchannel = ret

    if modality == 'EMG':
        if 'Quattro' in sensor_name:
            modality += 'Q'
            # New Discover sometimes uses 1234 instead of ABCD; normalize.
            subchannel = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}.get(subchannel, subchannel)
        elif 'Duo' in sensor_name:
            modality += 'D'
            subchannel = {'1': 'A', '2': 'B'}.get(subchannel, subchannel)
        else:
            modality += 'S'

    try:
        assert modality in tuple(target_sr.keys())
    except AssertionError:
        print("Your target SR does not match the modalities found in your data!")

    if modality == 'FSR':
        assert subchannel in ('A', 'B', 'C', 'D')
    if modality in ('ACC', 'GYRO'):
        assert subchannel in ('X', 'Y', 'Z')

    return SigInfoDelsys(sensor_name, modality, sensor_number, subchannel)


def _read_sensor_log(sensor_map_file):
    """Read and parse a Delsys channelmap text file."""
    with open(sensor_map_file, 'r', encoding='utf-8-sig') as f:
        sensor_map_raw = [x.split(' - ') for x in f.read().splitlines() if x]
    return [
        SensorLog(int(x[0].split(' ')[-1]), x[1], x[2][0], x[2].rstrip())
        for x in sensor_map_raw if len(x) > 1
    ]


def _parse_hdr(fname, sensor_name_replace=None):
    """Read the CSV header rows and return a dict describing the application,
    version, sampling info, and the synthesized per-column ``sensor: signal`` names.

    Detects EMGworks vs. Trigno Discover (and Discover version) on the first row.
    """
    if sensor_name_replace is None:
        sensor_name_replace = {}
    hdr = {}
    hdr['application'] = 'EMGworks'
    hdr['skiprows'] = 0
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        first_row = next(reader)
        if any(['Trigno Discover' in x for x in first_row]):
            hdr['application'] = 'Trigno Discover'
            hdr['application_full'] = first_row[-1].strip()
            s = hdr['application_full']
            hdr['application_version'] = s[s.find('(') + 1:s.find(')')]
            hdr['datetime'] = next(reader)[-1].strip()
            hdr['duration_s'] = float(next(reader)[-1].strip())
            sensor_names = [x.strip() for x in next(reader)]
            sensor_modes = [x.strip().removeprefix('sensor mode: ') for x in next(reader)]
            hdr['sensor_name_mode'] = {
                s_name: int(s_mode)
                for s_name, s_mode in zip(sensor_names, sensor_modes) if s_name
            }
            for idx, sensor_name in enumerate(sensor_names[1:]):
                if not sensor_name:
                    sensor_names[idx + 1] = sensor_names[idx]
            signal_names_raw = [x.strip() for x in next(reader)]
            signal_names = []
            analog_letters = ('A', 'B', 'C', 'D')
            curr_letter_idx = 0
            for sn in signal_names_raw:
                if 'Analog ?' in sn:  # Annoying bug in the discover software export
                    sn = sn.replace('?', analog_letters[curr_letter_idx])
                    curr_letter_idx = (curr_letter_idx + 1) % len(analog_letters)
                signal_names.append(sn)

            signal_names = _fix_corrupted_sensor_names(signal_names, sensor_name_replace)
            sensor_names += [sensor_names[-1]] * (len(signal_names) - len(sensor_names))
            if hdr['application_version'] == '1.4.2':
                hdr['sensor_signal_names'] = [
                    f'{sensor_name}: {signal_name}'
                    for sensor_name, signal_name in zip(sensor_names, signal_names)
                ]
                hdr['skiprows'] = 6
            else:  # 1.5.0+, also works for 1.6.x and 1.7.0
                sampling_rates = next(reader)
                hdr['sensor_signal_names'] = [
                    f'{sensor_name}: {signal_name} - ({sampling_rate.strip()})'
                    for sensor_name, signal_name, sampling_rate in zip(sensor_names, signal_names, sampling_rates)
                ]
                hdr['skiprows'] = 7
    return hdr


def _fix_corrupted_sensor_names(sig_names: list, sensor_name_replace: dict) -> list:
    """Apply ``sensor_name_replace`` (a ``{corrupted_name: new_name}`` dict) to fix
    sensor names that were misspelled during data acquisition."""
    sig_names_new = []
    for sn in sig_names:
        sn_new = sn
        for corrupted_sensor_name, new_sensor_name in sensor_name_replace.items():
            if sn.startswith(corrupted_sensor_name):
                sn_new = sn.replace(corrupted_sensor_name, new_sensor_name)
        sig_names_new.append(sn_new)
    return sig_names_new


@contextlib.contextmanager
def _open_dropped_samples_log(path):
    """Context manager: yields a writable file at ``path``, or a discard buffer if ``path`` is None.

    Lets the discover parsers be called outside the ``Log`` orchestrator without
    creating a side-channel file (useful in tests).
    """
    if path is None:
        yield io.StringIO()
    else:
        with open(path, 'w') as f:
            yield f


def _detect_parser(hdr, time_names):
    """Inspect parsed header info and return a tag indicating which parser to use.

    Returns one of: ``'emgworks'``, ``'discover_basic'``, ``'discover_link'``.
    Raises if link sensors are present but time-series columns are missing
    (Discover-with-link without ``Time Series`` columns can't be resampled).
    """
    if hdr['application'] == 'EMGworks':
        return 'emgworks'
    has_link = any('VO2' in x.strip() or 'HR' in x.strip() for x in hdr['sensor_signal_names'])
    has_timestamps = bool(time_names)
    if has_link and not has_timestamps:
        raise Exception("Found link data. Please export this file with Time Series!!")
    if has_link:
        return 'discover_link'
    return 'discover_basic'


def _parse_dataframe_emgworks(df, signal_map, sensors_info, target_sr, sig_names, time_names, clock_mul, t0):
    t_min_list = []
    t_max_list = []
    sr_list = []
    ts_list = []
    for t_name, s_name in zip(time_names, sig_names):
        ts = df[[t_name, s_name]].copy()
        ts.sort_values(by=t_name, inplace=True)
        ts = ts.dropna().to_numpy()
        t = ts[:, 0]
        t_min_list.append(t[0] / clock_mul)
        t_max_list.append(t[-1] / clock_mul)
        # Use median rather than mean of np.diff(t) — robust to dropped samples.
        sr_deduced = 1 / np.median(np.diff(t))
        # Per Delsys: an integer number of samples is received per 13.5 ms frame.
        sr_list.append((round(sr_deduced * 0.0135) / 0.0135) * clock_mul)
        ts_list.append(ts)
    min_sr = np.min(list(target_sr.values()))
    t_min = np.floor(np.min(t_min_list) * min_sr) / min_sr
    t_max = np.ceil(np.max(t_max_list) * min_sr) / min_sr

    signals = []
    for ts, sr, sig_info in zip(ts_list, sr_list, signal_map):
        n_samples = int((t_max - t_min) * sr) + 1
        this_t_max = t_min + (n_samples - 1) / sr
        t = np.linspace(t_min, this_t_max, n_samples)
        sig = interp1d(ts[:, 0] / clock_mul, ts[:, 1], fill_value='extrapolate')(t)
        sr_targ = target_sr[sig_info.modality]
        sig_resampled = resample(sig, round(n_samples * sr_targ / sr))
        this_sensor, = [sensor for sensor in sensors_info if sensor.number == sig_info.sensor_number]
        signals.append(Signal(sig_resampled, sr_targ, this_sensor, sig_info.modality, sig_info.subchannel, t0=t0))

    return t_min, t_max, sr_list, signals


def _parse_dataframe_discover(df, signal_map, sensors_info, target_sr, duration_hdr, clock_mul, t0, *, dropped_samples_path=None):
    duration = duration_hdr / clock_mul
    pattern = re.compile(r'\(([^)]+)Hz')
    column_sr = [float(pattern.findall(x)[0].strip()) for x in list(df)]
    sr_list = [(round(x * 0.0135) / 0.0135) * clock_mul for x in column_sr]  # exact sampling rate
    assert round(duration * max(sr_list)) == len(df)

    t_min = 0.
    t_max = duration
    with _open_dropped_samples_log(dropped_samples_path) as f:
        signals = []
        for ts_name, sr, sig_info in zip(df, sr_list, signal_map):
            sr_targ = target_sr[sig_info.modality]
            d = df[ts_name][:round(sr * duration) + 1].to_numpy()
            f.write(
                f'{sig_info.sensor_name} {sig_info.sensor_number} {sig_info.modality} {sig_info.subchannel} - '
                f'{np.sum(d == 0)} / {len(d)} = {(np.sum(d == 0) / len(d)) * 100:5.2f}% \n'
            )
            if sr_targ is None:  # no resampling
                sig_resampled = pysampled.Data(d, sr=sr).interpnan()
            else:
                sig_resampled = pysampled.Data(d, sr=sr).interpnan().resample(sr_targ)
            this_sensor, = [sensor for sensor in sensors_info if sensor.number == sig_info.sensor_number]
            signals.append(Signal(sig_resampled(), sig_resampled.sr, this_sensor, sig_info.modality, sig_info.subchannel, t0=t0))

    return t_min, t_max, sr_list, signals


def _parse_dataframe_discover_with_link(df, signal_map, sensors_info, target_sr, duration_hdr, clock_mul, t0, sig_names, time_names, *, dropped_samples_path=None):
    duration = duration_hdr / clock_mul
    pattern = re.compile(r'\(-?([^)]+)Hz')  # VO2 Master frequency appears as -1 for some reason
    column_sr = [float(pattern.findall(x)[0].strip()) for x in sig_names]
    # TODO: make the exception explicit. The 13.5 ms sampling-rate calc only applies for trigno base, not link devices.
    sr_list = []
    t_min = 0.
    t_max = duration

    link_time_names = [x for x in time_names if 'VO2' in x.strip() or 'HR' in x.strip()]

    with _open_dropped_samples_log(dropped_samples_path) as f:
        signals = []
        for ts_name, sr_raw, sig_info in zip(sig_names, column_sr, signal_map):
            sr_targ = target_sr[sig_info.modality]
            if 'VO2' in ts_name or 'HR' in ts_name:
                # Resample with the time series provided by VO2 / HR.
                # NOTES: VO2 can start delayed (TODO: fill those points with zeros) and can finish before the other system.
                pattern = re.compile(r':\s*(\w+)')
                match = re.findall(pattern, ts_name)
                time_name = next((name for name in link_time_names if match[0] in name), None)
                sr = sr_raw
                if ' Breathing Cycle' in ts_name:
                    continue  # data from breathing cycle is not useful

                d_signal = df[ts_name][:round(sr * duration) + 1].to_numpy()
                d_time = df[time_name][:round(sr * duration) + 1].to_numpy()
                sig_resampled = pysampled.uniform_resample(d_time, d_signal, sr_targ, t_min, t_max)
                d = d_signal  # for the dropped-samples report below

            else:
                sr = (round(sr_raw * 0.0135) / 0.0135) * clock_mul  # 13.5 ms only for trigno base, not link devices
                d = df[ts_name][:round(sr * duration) + 1].to_numpy()
                if sr_targ is None:  # no resampling
                    sig_resampled = pysampled.Data(d, sr=sr).interpnan()
                else:
                    sig_resampled = pysampled.Data(d, sr=sr).interpnan().resample(sr_targ)

            sr_list.append(sr)
            f.write(
                f'{sig_info.sensor_name} {sig_info.sensor_number} {sig_info.modality} {sig_info.subchannel} - '
                f'{np.sum(d == 0)} / {len(d)} = {(np.sum(d == 0) / len(d)) * 100:5.2f}% \n'
            )
            this_sensor, = [sensor for sensor in sensors_info if sensor.number == sig_info.sensor_number]
            signals.append(Signal(sig_resampled(), sig_resampled.sr, this_sensor, sig_info.modality, sig_info.subchannel, t0=t0))

        assert round(duration * max(sr_list)) == len(df)

    return t_min, t_max, sr_list, signals
