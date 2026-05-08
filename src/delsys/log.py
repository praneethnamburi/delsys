"""Top-level loader. ``Log`` reads a Delsys CSV export and routes to the
appropriate per-format parser, then groups parsed signals into ``Sensor``
objects keyed by sensor number.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from delsys._constants import TARGET_SR
from delsys._parse import (
    _detect_parser,
    _fix_corrupted_sensor_names,
    _parse_dataframe_discover,
    _parse_dataframe_discover_with_link,
    _parse_dataframe_emgworks,
    _parse_hdr,
    _parse_sig_name,
    _read_sensor_log,
)
from delsys._util import _mod_to_attr, _modset_to_strlist
from delsys.sensor import Sensor
from delsys.signals import SensorInfo, SensorLog


class Log:
    """Read Delsys log files (CSV export).

    Example::

        lf = Log(fname)
    """

    def __init__(self, fname, sensor_map=None, target_sr=None, clock_mul=1., t0=0., sensor_name_replace=None):
        """
        Inputs:
            fname (str) - full path to a csv file exported from delsys
            sensor_map (str, list, None) -
                - (str) full path to a file that maps sensors to names and placements (e.g. delsys_channelmap.txt)
                - (list) list of entries if the text file delsys_channelmap.txt has already been read in
                - (None) default
            target_sr (dict) - {modality (str): target_sampling_rate (float)}
                - for default, see module variable TARGET_SR
            clock_mul (float) - multiplier for sampling rates to 'sample' in a clock form a different system
                - for example, if the duration of an optitrack recording is 190.12 s,
                    and the optitrack recording gate in delsys is 190.27 s,
                    then clock_mul will be (duration in delsys)/(duration in optitrack) = 190.27/190.12.
                    Inferred sampling rates will be assumed to be (original sampling rate in delsys)*(duration in delsys)/(duration in optitrack)
            t0 (float) -
                - If the delsys file is synchronized to another system, then t0 will specify the start of the trial in the system that is counting time
                - If this is non-zero, and clock_mul != 1., then it is assumed that t0 is specified in the new clock!
        """
        if target_sr is None:
            target_sr = TARGET_SR
        if sensor_name_replace is None:
            sensor_name_replace = {}
        self.target_sr = target_sr
        self.clock_mul = clock_mul
        self.t0 = t0
        self.sensor_name_replace = sensor_name_replace
        self.fname = fname
        self.name = os.path.splitext(os.path.split(fname)[1])[0]
        self.hdr = _parse_hdr(self.fname, self.sensor_name_replace)

        df, sig_names, time_names = self._read_csv_file(self.fname, self.hdr, self.sensor_name_replace)

        self.sensor_map = self._parse_sensor_map(sensor_map, sig_names)
        self.signal_map, sensors_info = self._combine_signal_sensor_info(
            self.hdr['application'], self.target_sr, sig_names, self.sensor_map
        )

        parser_tag = _detect_parser(self.hdr, time_names)
        dropped_samples_path = os.path.join(
            Path(self.fname).parent, Path(self.fname).stem + '_dropped_samples.txt'
        )

        if parser_tag == 'emgworks':
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_emgworks(
                df, self.signal_map, sensors_info, self.target_sr, sig_names, time_names, self.clock_mul, self.t0
            )
        elif parser_tag == 'discover_link':
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_discover_with_link(
                df, self.signal_map, sensors_info, self.target_sr, self.hdr['duration_s'], self.clock_mul, self.t0,
                sig_names, time_names, dropped_samples_path=dropped_samples_path,
            )
        else:  # 'discover_basic' — TODO: CHECK if this works for a file with timestamps exported
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_discover(
                df, self.signal_map, sensors_info, self.target_sr, self.hdr['duration_s'], self.clock_mul, self.t0,
                dropped_samples_path=dropped_samples_path,
            )

        self.sensors = self._signals_to_sensors(sensors_info, self.signals)
        self.sensor_groups = {}

    @staticmethod
    def _read_csv_file(fname, hdr, sensor_name_replace=None):
        if hdr['application'] == 'EMGworks':
            df = pd.read_csv(fname)
            column_names = list(df)
            assert len(column_names) % 2 == 0

            time_col_pos = ['X[s]' in c for c in column_names]
            assert np.unique([x % 2 for x in np.where(time_col_pos)[0]]) == np.array([0])

            time_names = [c for c in column_names if 'X[s]' in c]
            sig_names = [c for c in column_names if 'X[s]' not in c]
            assert len(time_names) == len(sig_names)
        else:  # trigno discover
            df = pd.read_csv(
                fname, skiprows=hdr['skiprows'], names=hdr['sensor_signal_names'], skipinitialspace=True,
            ).dropna(axis=1, how='all').dropna(axis=0, how='all')
            hdr['sensor_signal_names'] = list(df)
            time_names = [c for c in hdr['sensor_signal_names'] if 'Time Series' in c]
            sig_names = [c for c in hdr['sensor_signal_names'] if 'Time Series' not in c]

        sig_names = _fix_corrupted_sensor_names(sig_names, sensor_name_replace)

        return df, sig_names, time_names

    def _parse_sensor_map(self, sensor_map, sig_names):
        if sensor_map is None:
            sensor_numbers = list(np.unique(
                [_parse_sig_name(s_name, self.hdr['application'], self.target_sr).sensor_number
                 for s_name in sig_names]
            ))
            sensor_numbers.sort()
            sensor_map = [SensorLog(int(x), None, None, None) for x in sensor_numbers]
        elif isinstance(sensor_map, str):
            assert os.path.exists(sensor_map)
            sensor_map = _read_sensor_log(sensor_map)
        else:
            assert isinstance(sensor_map, list)
        for sensor in sensor_map:
            assert isinstance(sensor, SensorLog)
        return sensor_map

    @staticmethod
    def _combine_signal_sensor_info(application, target_sr, sig_names, sensor_map):
        signal_map = [_parse_sig_name(sig_name, application, target_sr) for sig_name in sig_names]

        sensors_info = []
        for sensor in sensor_map:
            try:
                name, = set([x.sensor_name for x in signal_map if x.sensor_number == sensor.number])
            except ValueError as e:
                print("Signal name mismatch in channelmap. Found these in log file:")
                print('\n'.join(list(np.unique([x.sensor_name for x in signal_map]))))
                print("These are specified in sensor map (delsys_channelmap.txt)")
                print('\n'.join([f'{s.number} - {s.type_sensorlog} - {s.location}' for s in sensor_map]))
                raise e
            modalities = {x.modality for x in signal_map if x.sensor_number == sensor.number}
            if sensor.type_sensorlog == 'FSR':
                modalities = {'FSR' if mod == 'Analog' else mod for mod in modalities}
            sensors_info.append(SensorInfo(name=name, modalities=modalities, **sensor._asdict()))

        return signal_map, sensors_info

    @staticmethod
    def _signals_to_sensors(sensors_info, signals):
        sensors = []
        for sensor_info in sensors_info:
            sensors.append(Sensor(sensor_info, [s for s in signals if s.sensor.number == sensor_info.number]))
        return sensors

    modalities = property(lambda self: set([x.modality for x in self.signals]))
    sampling_rates = property(lambda self: set([x.sr for x in self.signals]))
    locations = property(lambda self: set([x.location for x in self.signals]))

    sensor_names = property(lambda self: set([x.name for x in self.sensors]))
    sensor_modalities = property(lambda self: {sensor.name: sensor.modalities for sensor in self.sensors})

    @property
    def dur(self):
        """Duration of the total recording."""
        return self.t_max - self.t_min

    @property
    def modality_sensors(self):
        """For each modality, return which sensors have that modality."""
        ret = {}
        for modality in self.modalities:
            ret[modality] = set([x.sensor_name for x in self.signals if x.modality == modality])
        return ret

    @property
    def sensor_numbers(self):
        return [s.number for s in self.sensors]

    # Typed retrieval accessors. Computed at access time, so no pickled state
    # is added — existing pickled Logs gain these accessors automatically.
    emg       = property(lambda self: [s.emg       for s in self.sensors if hasattr(s, 'emg')])
    ekg       = property(lambda self: [s.ekg       for s in self.sensors if hasattr(s, 'ekg')])
    acc       = property(lambda self: [s.acc       for s in self.sensors if hasattr(s, 'acc')])
    gyro      = property(lambda self: [s.gyro      for s in self.sensors if hasattr(s, 'gyro')])
    fsr       = property(lambda self: [s.fsr       for s in self.sensors if hasattr(s, 'fsr')])
    analog    = property(lambda self: [s.analog    for s in self.sensors if hasattr(s, 'analog')])
    vo2master = property(lambda self: [s.vo2master for s in self.sensors if hasattr(s, 'vo2master')])
    hrstrap   = property(lambda self: [s.hrstrap   for s in self.sensors if hasattr(s, 'hrstrap')])

    left   = property(lambda self: [s for s in self.sensors if s.lrc == 'L'])
    right  = property(lambda self: [s for s in self.sensors if s.lrc == 'R'])
    center = property(lambda self: [s for s in self.sensors if s.lrc == 'C'])

    def find(self, modality=None, side=None, location=None, sensor_number=None, name=None, as_='auto'):
        """Query for sensors, modality bundles, or raw signals using named filters.

        Parameters
        ----------
        modality : str, optional
            Modality to filter by; case-insensitive and matches EMG variants
            (``'emg'`` matches sensors with EMGS/EMGD/EMGQ). Recognized values:
            ``'EMG'``, ``'EKG'``, ``'ACC'``, ``'GYRO'``, ``'FSR'``, ``'Analog'``,
            ``'VO2'``, ``'HR'``.
        side : {'L', 'R', 'C'}, optional
            Sensor side (``lrc`` field).
        location : str, optional
            Substring match on ``sensor.location`` (``'Forearm'`` matches both
            ``'LForearm'`` and ``'RForearm'``).
        sensor_number : int, optional
            Specific sensor number.
        name : str, optional
            Substring match on ``sensor.name``.
        as_ : {'auto', 'modality', 'sensor', 'signal'}, optional
            Return shape. ``'auto'`` returns modality bundles when ``modality``
            is set, otherwise sensors. ``'signal'`` always returns raw
            ``Signal`` objects (one per sub-channel).

        Example usage::

            lf.find(modality='EMG')                  # list of EMG bundles
            lf.find(modality='emg')                  # case-insensitive
            lf.find(modality='EMGS')                 # variant matching (EMG/EMGS/EMGD/EMGQ unify)
            lf.find(modality='VO2')                  # link devices via the same API
            lf.find(side='R')                        # list of right-side sensors
            lf.find(location='Forearm')              # substring match (matches LForearm + RForearm)
            lf.find(sensor_number=5)                 # by number
            lf.find(name='Avanti sensor 2')          # substring match
            lf.find(modality='EMG', as_='signal')    # raw Signal objects, one per sub-channel
            lf.find(modality='EMG', as_='sensor')    # whole Sensor objects with the modality
            lf.find()                                # all sensors, no filter

        Returns
        -------
        list
            Matching items, possibly empty.
        """
        sensors = list(self.sensors)

        if side is not None:
            sensors = [s for s in sensors if s.lrc == side]
        if location is not None:
            sensors = [s for s in sensors if s.location and location in s.location]
        if sensor_number is not None:
            sensors = [s for s in sensors if s.number == sensor_number]
        if name is not None:
            sensors = [s for s in sensors if name in s.name]
        if modality is not None:
            target_attr = _mod_to_attr(modality)
            sensors = [s for s in sensors if any(_mod_to_attr(m) == target_attr for m in s.modalities)]

        if as_ == 'auto':
            as_ = 'modality' if modality is not None else 'sensor'

        if as_ == 'sensor':
            return sensors

        if as_ == 'modality':
            if modality is None:
                raise ValueError("as_='modality' requires the modality argument")
            attr = _mod_to_attr(modality)
            return [getattr(s, attr) for s in sensors if hasattr(s, attr)]

        if as_ == 'signal':
            sensor_nums = {s.number for s in sensors}
            signals = [sig for sig in self.signals if sig.sensor.number in sensor_nums]
            if modality is not None:
                target_attr = _mod_to_attr(modality)
                signals = [sig for sig in signals if _mod_to_attr(sig.modality) == target_attr]
            return signals

        raise ValueError(f"Unknown as_={as_!r}; expected 'auto', 'modality', 'sensor', or 'signal'")

    def __getitem__(self, key) -> list:
        """Retrieve a list of sensors/signals."""
        if not isinstance(key, list):
            key = [key]
        ret = [item for k in key for item in self._getitem_onekey(k)]
        if len(ret) == 1:
            return ret[0]
        return ret

    def _getitem_onekey(self, key):
        """Retrieve items based on a key, in order of priority:
        sensor number, left/right/center, modality, location (partial OK),
        sensor name (partial OK).
        """
        if isinstance(key, int):
            return [s for s in self.sensors if s.number == key]

        assert isinstance(key, str)
        if len(key) == 1 and key in set([s.lrc for s in self.sensors]):
            return [s for s in self.sensors if s.lrc == key]

        if key in _modset_to_strlist(self.modalities):
            return [getattr(s, _mod_to_attr(key)) for s in self.sensors if key in _modset_to_strlist(s.modalities)]

        if key in [s.location for s in self.sensors]:
            return [s for s in self.sensors if key in s.location]

        if key in self.sensor_names:
            return [s for s in self.sensors if key in s.name]

        return []

    def add_sensor_group(self, name, sensor_list):
        """Example: ``lf.add_sensor_group('LBFL', (14, 7, 4))``."""
        if not hasattr(self, 'sensor_groups'):  # pickle backward compatibility
            self.sensor_groups = {}
        all_sensors = self.sensor_numbers
        for s_num in sensor_list:
            assert s_num in all_sensors
        self.sensor_groups[name] = sensor_list

    def is_resampled(self) -> bool:
        return self.clock_mul != 1.

    def is_shifted(self) -> bool:
        return self.t0 != 0.

    def is_adjusted(self) -> bool:
        return self.is_resampled() or self.is_shifted()

    def export_to_csv(self, modality='emg', process_func=None, process_name='_processed', export_dir=None):
        """Export a modality (e.g. 'emg', 'acc') to a CSV file.

        export_dir: directory to write the CSV into. Defaults to ``<input-csv-parent>/export``.
        """
        if process_func is None:
            process_func = lambda x: x
            process_name = ''

        if export_dir is None:
            export_dir = os.path.join(Path(self.fname).parent, 'export')
        os.makedirs(export_dir, exist_ok=True)
        all_signals = self[modality]
        if not isinstance(all_signals, list):
            all_signals = [all_signals]
        assert len(set([x.sr for x in all_signals])) == 1
        emg_dict = {'Time': []}
        for sensor in self.sensors:
            if modality in _modset_to_strlist(sensor.modalities):
                sig = sensor.__dict__[modality]
                for signal_count, signal in enumerate(sig.split_to_1d()):
                    name = f'{sensor.name} {sensor.location} :{signal_count}'
                    emg_dict[name] = process_func(signal)

        n_samples = min([x().shape[0] for x in list(emg_dict.values())[1:]])
        emg_dict['Time'] = list(emg_dict.values())[1].t[:n_samples]
        for signal_name, signal in emg_dict.items():
            if signal_name != 'Time':
                emg_dict[signal_name] = signal()[:n_samples]

        save_name = os.path.join(export_dir, Path(self.fname).stem + f'_{modality}{process_name}.csv')
        print('data was saved:', save_name)
        df = pd.DataFrame.from_dict(emg_dict)
        return df.to_csv(save_name, index=None)
