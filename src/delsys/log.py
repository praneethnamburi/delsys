"""Top-level loader.

The ``Log`` class reads a Delsys CSV export, dispatches to the appropriate
per-format parser (EMGworks, Trigno Discover basic, Trigno Discover with
link devices), and groups the parsed signals into ``Sensor`` objects keyed
by sensor number.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from delsys._constants import TARGET_SR
from delsys._metadata import SensorInfo, SensorLog
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
from delsys._util import (
    _aggregate_bundles,
    _mod_to_attr,
    _modset_to_strlist,
    _normalize_signal_lengths,
)
from delsys.ekg import EKG
from delsys.emg import EMG
from delsys.sensor import Sensor
from delsys.signals import FSR, IMU, Signal, VO2Master


class Log:
    """Load a Delsys CSV file and expose its signals as typed bundles.

    A ``Log`` is the single entry point of the package. Construct one with the
    path to a CSV file produced by EMGworks or Trigno Discover, and the
    constructor parses the header, picks the right per-format dataframe parser,
    resamples each channel to the target sampling rate, and assembles the
    results into per-sensor modality bundles (``EMG``, ``EKG``, ``IMU``,
    ``FSR``, ``VO2Master``, etc.).

    After construction, three families of accessors are available:

    * Direct attribute access:
      ``lf.emg``, ``lf.ekg``, ``lf.acc``, ``lf.gyro``, ``lf.fsr``, ``lf.analog``,
      ``lf.vo2master``, ``lf.hrstrap`` — each returns a single aggregated
      :class:`pysampled.Data` per modality (channels stacked across all
      sensors that have that modality), or ``None`` if no sensor does.
      Use ``bundle.split_by_signal_name()`` to recover the per-Sensor list.
      Side accessors ``lf.left`` / ``lf.right`` / ``lf.center`` return
      lists of ``Sensor``.
    * The ``find()`` method for filtered queries.
    * ``__getitem__`` for legacy lookups by sensor number, side, modality,
      location, or sensor name. **Deprecated as of 0.3.0** — retained
      indefinitely for backward compatibility, but new code should prefer
      ``find``, which is more explicit about its return shape.

    Args:
        fname: Path to a CSV file exported from Delsys.
        sensor_map: Manual sensor-to-name-and-placement map. May be:

            - ``None`` (default) — auto-build from the sensor numbers found in the file.
            - ``str`` — path to a channelmap text file (e.g. ``delsys_channelmap.txt``).
            - ``list`` — a pre-parsed list of :class:`SensorLog` records.
        target_sr: Per-modality target sampling rates, e.g.
            ``{"EMGS": 2000, "ACC": 200, ...}``. Defaults to
            :data:`delsys.TARGET_SR`.
        clock_mul: Sampling-rate multiplier used when re-clocking against
            another acquisition system. For example, if a Delsys recording's
            duration is 190.27 s but the synchronized OptiTrack recording is
            190.12 s, set ``clock_mul = 190.27 / 190.12`` so the inferred
            sampling rates are scaled to the OptiTrack clock.
        t0: Trial start time in the synchronized clock. If non-zero and
            ``clock_mul != 1.0``, ``t0`` is interpreted in the new clock.
        sensor_name_replace: ``{corrupted_name: new_name}`` map applied to
            sensor names that were misspelled during data acquisition.

    Attributes:
        fname (str): The CSV path that was loaded.
        name (str): File stem (without extension), used as the recording's name.
        hdr (dict): Parsed header information (see :func:`_parse_hdr`).
        sensors (List[Sensor]): One entry per physical sensor in the recording.
        signals (List[Signal]): One entry per per-channel signal (i.e. an EMG
            sensor contributes one ``Signal``, an ACC sensor contributes three).
        sensor_map (List[SensorLog]): The channelmap used during loading.
        signal_map (List[SigInfoDelsys]): Per-column metadata records.
        target_sr (Dict[str, Optional[float]]): Effective target-sr map.
        clock_mul (float): Effective clock multiplier.
        t0 (float): Effective trial start time.
        sensor_groups (Dict[str, Sequence[int]]): User-defined groupings
            (see :meth:`add_sensor_group`).
        t_min (float): Earliest valid time across all signals (seconds).
        t_max (float): Latest valid time across all signals (seconds).
        sr_orig (List[float]): Original (pre-resample) sampling rate per channel.

    Example:
        .. code-block:: python

            import delsys

            lf = delsys.Log("trial_01.csv")
            for emg in lf.emg.split_by_signal_name():
                processed = emg.process(amp_kind="envelope2")
            right_sensors = lf.find(side="R")
            forearm_emg = lf.find(modality="EMG", location="Forearm")
    """

    def __init__(
        self,
        fname: str,
        sensor_map: Optional[Union[str, List[SensorLog]]] = None,
        target_sr: Optional[Dict[str, Optional[float]]] = None,
        clock_mul: float = 1.0,
        t0: float = 0.0,
        sensor_name_replace: Optional[Dict[str, str]] = None,
    ) -> None:
        if target_sr is None:
            target_sr = TARGET_SR
        if sensor_name_replace is None:
            sensor_name_replace = {}
        self.target_sr: Dict[str, Optional[float]] = target_sr
        self.clock_mul: float = clock_mul
        self.t0: float = t0
        self.sensor_name_replace: Dict[str, str] = sensor_name_replace
        self.fname: str = fname
        self.name: str = os.path.splitext(os.path.split(fname)[1])[0]
        self.hdr: Dict[str, Any] = _parse_hdr(self.fname, self.sensor_name_replace)

        df, sig_names, time_names = self._read_csv_file(
            self.fname, self.hdr, self.sensor_name_replace
        )

        self.sensor_map: List[SensorLog] = self._parse_sensor_map(sensor_map, sig_names)
        self.signal_map, sensors_info = self._combine_signal_sensor_info(
            self.hdr["application"], self.target_sr, sig_names, self.sensor_map
        )

        parser_tag = _detect_parser(self.hdr, time_names)
        dropped_samples_path = os.path.join(
            Path(self.fname).parent, Path(self.fname).stem + "_dropped_samples.txt"
        )

        if parser_tag == "emgworks":
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_emgworks(
                df,
                self.signal_map,
                sensors_info,
                self.target_sr,
                sig_names,
                time_names,
                self.clock_mul,
                self.t0,
            )
        elif parser_tag == "discover_link":
            self.t_min, self.t_max, self.sr_orig, self.signals = (
                _parse_dataframe_discover_with_link(
                    df,
                    self.signal_map,
                    sensors_info,
                    self.target_sr,
                    self.hdr["duration_s"],
                    self.clock_mul,
                    self.t0,
                    sig_names,
                    time_names,
                    dropped_samples_path=dropped_samples_path,
                )
            )
        else:  # 'discover_basic' — TODO: confirm behavior on a file with timestamps exported.
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_discover(
                df,
                self.signal_map,
                sensors_info,
                self.target_sr,
                self.hdr["duration_s"],
                self.clock_mul,
                self.t0,
                dropped_samples_path=dropped_samples_path,
            )

        # Tail-trim per-(modality, sr) drift before the per-Sensor stack
        # asserts equal lengths. Drift originates upstream in the parsers'
        # rounding of ``sr * duration``; normalizing here keeps the
        # parsers' per-format quirks intact.
        self.signals = _normalize_signal_lengths(self.signals)

        self.sensors: List[Sensor] = self._signals_to_sensors(sensors_info, self.signals)
        self.sensor_groups: Dict[str, Sequence[int]] = {}

    # ------------------------------------------------------------------
    # Internal CSV-reading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv_file(
        fname: str,
        hdr: Dict[str, Any],
        sensor_name_replace: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Read the CSV body and return the dataframe along with column-name lists.

        Args:
            fname: Path to the CSV file.
            hdr: Header dict produced by :func:`_parse_hdr`.
            sensor_name_replace: Optional sensor-name correction map.

        Returns:
            ``(df, sig_names, time_names)``. ``sig_names`` are the columns
            holding signal data; ``time_names`` are the per-channel time-stamp
            columns (empty list for Discover-basic files).
        """
        if hdr["application"] == "EMGworks":
            df = pd.read_csv(fname)
            column_names = list(df)
            assert len(column_names) % 2 == 0

            time_col_pos = ["X[s]" in c for c in column_names]
            assert np.unique([x % 2 for x in np.where(time_col_pos)[0]]) == np.array([0])

            time_names = [c for c in column_names if "X[s]" in c]
            sig_names = [c for c in column_names if "X[s]" not in c]
            assert len(time_names) == len(sig_names)
        else:  # Trigno Discover
            df = (
                pd.read_csv(
                    fname,
                    skiprows=hdr["skiprows"],
                    names=hdr["sensor_signal_names"],
                    skipinitialspace=True,
                )
                .dropna(axis=1, how="all")
                .dropna(axis=0, how="all")
            )
            hdr["sensor_signal_names"] = list(df)
            time_names = [c for c in hdr["sensor_signal_names"] if "Time Series" in c]
            sig_names = [c for c in hdr["sensor_signal_names"] if "Time Series" not in c]

        sig_names = _fix_corrupted_sensor_names(sig_names, sensor_name_replace)
        return df, sig_names, time_names

    def _parse_sensor_map(
        self,
        sensor_map: Optional[Union[str, List[SensorLog]]],
        sig_names: List[str],
    ) -> List[SensorLog]:
        """Resolve the user-supplied ``sensor_map`` into a list of ``SensorLog``.

        When ``sensor_map`` is None, build one from the sensor numbers found
        in the CSV (location/lrc fields default to ``None``). When it's a
        string, treat it as a path to a Delsys channelmap file.
        """
        if sensor_map is None:
            sensor_numbers = list(
                np.unique(
                    [
                        _parse_sig_name(
                            s_name, self.hdr["application"], self.target_sr
                        ).sensor_number
                        for s_name in sig_names
                    ]
                )
            )
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
    def _combine_signal_sensor_info(
        application: str,
        target_sr: Dict[str, Optional[float]],
        sig_names: List[str],
        sensor_map: List[SensorLog],
    ) -> Tuple[List, List[SensorInfo]]:
        """Merge per-column header metadata with the channelmap into ``SensorInfo`` records.

        Args:
            application: ``'EMGworks'`` or ``'Trigno Discover'``.
            target_sr: Per-modality target sampling rates.
            sig_names: Column names from the CSV (signal columns, no time
                columns).
            sensor_map: Pre-parsed channelmap.

        Returns:
            ``(signal_map, sensors_info)`` — the per-column ``SigInfoDelsys``
            list and the per-sensor ``SensorInfo`` list.

        Raises:
            ValueError: If the channelmap names a sensor number that does not
                appear unambiguously in the CSV (e.g. duplicate or missing).
        """
        signal_map = [_parse_sig_name(sig_name, application, target_sr) for sig_name in sig_names]

        sensors_info: List[SensorInfo] = []
        for sensor in sensor_map:
            try:
                (name,) = set(
                    [x.sensor_name for x in signal_map if x.sensor_number == sensor.number]
                )
            except ValueError as e:
                print("Signal name mismatch in channelmap. Found these in log file:")
                print("\n".join(list(np.unique([x.sensor_name for x in signal_map]))))
                print("These are specified in sensor map (delsys_channelmap.txt)")
                print(
                    "\n".join(
                        [f"{s.number} - {s.type_sensorlog} - {s.location}" for s in sensor_map]
                    )
                )
                raise e
            modalities = {x.modality for x in signal_map if x.sensor_number == sensor.number}
            if sensor.type_sensorlog == "FSR":
                modalities = {"FSR" if mod == "Analog" else mod for mod in modalities}
            sensors_info.append(SensorInfo(name=name, modalities=modalities, **sensor._asdict()))

        return signal_map, sensors_info

    @staticmethod
    def _signals_to_sensors(
        sensors_info: List[SensorInfo],
        signals: List[Signal],
    ) -> List[Sensor]:
        """Group signals by sensor number and wrap each group in a ``Sensor``."""
        sensors: List[Sensor] = []
        for sensor_info in sensors_info:
            sensors.append(
                Sensor(sensor_info, [s for s in signals if s.sensor.number == sensor_info.number])
            )
        return sensors

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    modalities: Set[str] = property(  # type: ignore[assignment]
        lambda self: set([x.modality for x in self.signals])
    )
    sampling_rates: Set[float] = property(  # type: ignore[assignment]
        lambda self: set([x.sr for x in self.signals])
    )
    locations: Set[Optional[str]] = property(  # type: ignore[assignment]
        lambda self: set([x.location for x in self.signals])
    )

    sensor_names: Set[str] = property(  # type: ignore[assignment]
        lambda self: set([x.name for x in self.sensors])
    )
    sensor_modalities: Dict[str, Set[str]] = property(  # type: ignore[assignment]
        lambda self: {sensor.name: sensor.modalities for sensor in self.sensors}
    )

    @property
    def dur(self) -> float:
        """Duration of the recording, in seconds (``t_max - t_min``)."""
        return self.t_max - self.t_min

    @property
    def modality_sensors(self) -> Dict[str, Set[str]]:
        """Map each modality to the set of sensor names that have it."""
        ret: Dict[str, Set[str]] = {}
        for modality in self.modalities:
            ret[modality] = set([x.sensor_name for x in self.signals if x.modality == modality])
        return ret

    @property
    def sensor_numbers(self) -> List[int]:
        """Sensor numbers in their order of appearance in :attr:`sensors`."""
        return [s.number for s in self.sensors]

    # ------------------------------------------------------------------
    # Typed retrieval accessors. Computed at access time, so no pickled state
    # is added — existing pickled Logs gain these accessors automatically.
    # ------------------------------------------------------------------

    emg: Optional[EMG] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.emg for s in self.sensors if hasattr(s, "emg")], EMG
        )
    )
    ekg: Optional[EKG] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.ekg for s in self.sensors if hasattr(s, "ekg")], EKG
        )
    )
    acc: Optional[IMU] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.acc for s in self.sensors if hasattr(s, "acc")], IMU
        )
    )
    gyro: Optional[IMU] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.gyro for s in self.sensors if hasattr(s, "gyro")], IMU
        )
    )
    fsr: Optional[FSR] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.fsr for s in self.sensors if hasattr(s, "fsr")], FSR
        )
    )
    analog = property(
        lambda self: _aggregate_bundles(
            [s.analog for s in self.sensors if hasattr(s, "analog")]
        )
    )
    vo2master: Optional[VO2Master] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.vo2master for s in self.sensors if hasattr(s, "vo2master")],
            VO2Master,
        )
    )
    hrstrap = property(
        lambda self: _aggregate_bundles(
            [s.hrstrap for s in self.sensors if hasattr(s, "hrstrap")]
        )
    )

    left: List[Sensor] = property(lambda self: [s for s in self.sensors if s.lrc == "L"])  # type: ignore[assignment]
    right: List[Sensor] = property(lambda self: [s for s in self.sensors if s.lrc == "R"])  # type: ignore[assignment]
    center: List[Sensor] = property(lambda self: [s for s in self.sensors if s.lrc == "C"])  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def find(
        self,
        modality: Optional[str] = None,
        side: Optional[str] = None,
        location: Optional[str] = None,
        sensor_number: Optional[int] = None,
        name: Optional[str] = None,
        as_: str = "auto",
    ) -> list:
        """Query for sensors, modality bundles, or raw signals.

        Args:
            modality: Modality to filter by. Case-insensitive and matches EMG
                variants (``'emg'`` matches sensors with ``EMGS``/``EMGD``/
                ``EMGQ``). Recognized values: ``'EMG'``, ``'EKG'``, ``'ACC'``,
                ``'GYRO'``, ``'FSR'``, ``'Analog'``, ``'VO2'``, ``'HR'``.
            side: Sensor side — one of ``'L'``, ``'R'``, ``'C'``.
            location: Substring match on ``sensor.location`` (e.g.
                ``'Forearm'`` matches both ``'LForearm'`` and ``'RForearm'``).
            sensor_number: Specific sensor number.
            name: Substring match on ``sensor.name``.
            as_: Return shape — one of ``'auto'``, ``'modality'``,
                ``'sensor'``, ``'signal'``. ``'auto'`` returns modality
                bundles when ``modality`` is set, otherwise sensors.
                ``'signal'`` returns raw :class:`Signal` objects (one per
                sub-channel).

        Returns:
            A list of matching items (possibly empty).

        Raises:
            ValueError: If ``as_='modality'`` is requested without a
                ``modality`` argument, or if ``as_`` is unrecognized.

        Example:
            .. code-block:: python

                lf.find(modality="EMG")                  # list of EMG bundles
                lf.find(modality="emg")                  # case-insensitive
                lf.find(modality="EMGS")                 # variant matching
                lf.find(modality="VO2")                  # link devices
                lf.find(side="R")                        # right-side sensors
                lf.find(location="Forearm")              # substring match
                lf.find(sensor_number=5)                 # by number
                lf.find(name="Avanti sensor 2")          # substring match
                lf.find(modality="EMG", as_="signal")    # raw Signal objects
                lf.find(modality="EMG", as_="sensor")    # whole Sensor objects
                lf.find()                                # all sensors
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
            sensors = [
                s for s in sensors if any(_mod_to_attr(m) == target_attr for m in s.modalities)
            ]

        if as_ == "auto":
            as_ = "modality" if modality is not None else "sensor"

        if as_ == "sensor":
            return sensors

        if as_ == "modality":
            if modality is None:
                raise ValueError("as_='modality' requires the modality argument")
            attr = _mod_to_attr(modality)
            return [getattr(s, attr) for s in sensors if hasattr(s, attr)]

        if as_ == "signal":
            sensor_nums = {s.number for s in sensors}
            signals = [sig for sig in self.signals if sig.sensor.number in sensor_nums]
            if modality is not None:
                target_attr = _mod_to_attr(modality)
                signals = [sig for sig in signals if _mod_to_attr(sig.modality) == target_attr]
            return signals

        raise ValueError(f"Unknown as_={as_!r}; expected 'auto', 'modality', 'sensor', or 'signal'")

    # ------------------------------------------------------------------
    # Legacy bracket lookup (deprecated, retained indefinitely)
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[int, str, list]) -> Any:
        """Look up sensors or modality bundles by sensor number, side, modality, location, or name.

        .. deprecated:: 0.3.0
           Use :meth:`find` instead. ``__getitem__`` overloads five key
           types (int sensor number, single-letter side, modality string,
           location substring, sensor-name substring) and collapses
           single-match results, which makes the return shape hard to
           predict at the call site. ``find`` takes named filters and
           always returns a list. The legacy method is retained
           indefinitely for backward compatibility — there is no removal
           plan — but new code should prefer ``find``.

        When ``key`` is a list, the result is the OR of the per-key matches.

        Args:
            key: One of:

                - ``int`` — sensor number.
                - ``str`` of length 1 matching ``L``/``R``/``C`` — side.
                - modality string (``'EMG'``, ``'ACC'``, ``'analog'``, ...).
                - location substring (``'Foot'`` matches ``'LFoot'`` and
                  ``'RFoot'``).
                - sensor-name substring.
                - ``list`` of any of the above.

        Returns:
            The single matching item if exactly one match was found, else a
            list of all matches (possibly empty).
        """
        if not isinstance(key, list):
            key = [key]
        ret = [item for k in key for item in self._getitem_onekey(k)]
        if len(ret) == 1:
            return ret[0]
        return ret

    def _getitem_onekey(self, key: Union[int, str]) -> list:
        """Look up by a single key. See :meth:`__getitem__` for priority order."""
        if isinstance(key, int):
            return [s for s in self.sensors if s.number == key]

        assert isinstance(key, str)
        if len(key) == 1 and key in set([s.lrc for s in self.sensors]):
            return [s for s in self.sensors if s.lrc == key]

        if key in _modset_to_strlist(self.modalities):
            return [
                getattr(s, _mod_to_attr(key))
                for s in self.sensors
                if key in _modset_to_strlist(s.modalities)
            ]

        if key in [s.location for s in self.sensors]:
            return [s for s in self.sensors if key in s.location]

        if key in self.sensor_names:
            return [s for s in self.sensors if key in s.name]

        return []

    # ------------------------------------------------------------------
    # Sensor groups
    # ------------------------------------------------------------------

    def add_sensor_group(self, name: str, sensor_list: Sequence[int]) -> None:
        """Register a named group of sensor numbers.

        Args:
            name: Group label (e.g. ``'LBFL'``).
            sensor_list: Sensor numbers to include. All numbers must exist in
                :attr:`sensor_numbers`.

        Raises:
            AssertionError: If ``sensor_list`` references an unknown sensor
                number.

        Example:
            .. code-block:: python

                lf.add_sensor_group("LBFL", (14, 7, 4))
        """
        if not hasattr(self, "sensor_groups"):  # pickle backward compatibility
            self.sensor_groups = {}
        all_sensors = self.sensor_numbers
        for s_num in sensor_list:
            assert s_num in all_sensors
        self.sensor_groups[name] = sensor_list

    # ------------------------------------------------------------------
    # Synchronization-state predicates
    # ------------------------------------------------------------------

    def is_resampled(self) -> bool:
        """``True`` if a non-unity ``clock_mul`` was applied at load time."""
        return self.clock_mul != 1.0

    def is_shifted(self) -> bool:
        """``True`` if a non-zero ``t0`` was applied at load time."""
        return self.t0 != 0.0

    def is_adjusted(self) -> bool:
        """``True`` if either :meth:`is_resampled` or :meth:`is_shifted` is true."""
        return self.is_resampled() or self.is_shifted()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(
        self,
        modality: str = "emg",
        process_func: Optional[Callable] = None,
        process_name: str = "_processed",
        export_dir: Optional[str] = None,
    ) -> None:
        """Write all signals of one modality to a CSV file.

        Args:
            modality: Modality attribute name (``'emg'``, ``'acc'``, ...).
            process_func: Optional per-signal processing callable applied
                before writing. Default is the identity.
            process_name: Suffix appended to the output filename when a
                processing function was supplied. Ignored when
                ``process_func`` is ``None``.
            export_dir: Output directory. Defaults to
                ``<input-csv-parent>/export``, created if missing.

        Raises:
            AssertionError: If signals selected by ``modality`` have
                heterogeneous sampling rates.
        """
        if process_func is None:
            process_func = lambda x: x  # noqa: E731
            process_name = ""

        if export_dir is None:
            export_dir = os.path.join(Path(self.fname).parent, "export")
        os.makedirs(export_dir, exist_ok=True)
        all_signals = self.find(modality=modality, as_="modality")
        assert len(set([x.sr for x in all_signals])) == 1
        emg_dict: Dict[str, Any] = {"Time": []}
        for sensor in self.sensors:
            if modality in _modset_to_strlist(sensor.modalities):
                sig = sensor.__dict__[modality]
                for signal_count, signal in enumerate(sig.split_to_1d()):
                    name = f"{sensor.name} {sensor.location} :{signal_count}"
                    emg_dict[name] = process_func(signal)

        n_samples = min([x().shape[0] for x in list(emg_dict.values())[1:]])
        emg_dict["Time"] = list(emg_dict.values())[1].t[:n_samples]
        for signal_name, signal in emg_dict.items():
            if signal_name != "Time":
                emg_dict[signal_name] = signal()[:n_samples]

        save_name = os.path.join(
            export_dir, Path(self.fname).stem + f"_{modality}{process_name}.csv"
        )
        print("data was saved:", save_name)
        df = pd.DataFrame.from_dict(emg_dict)
        return df.to_csv(save_name, index=None)
