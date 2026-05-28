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

from delsys._constants import SUBCHANNEL_MAP, TARGET_SR
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
from delsys.cleaning import (
    CleaningConfig,
    CleaningResult,
    harmonize_multirate_inputs,
    run_pipeline,
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
        if str(fname).lower().endswith((".h5", ".hdf5")):
            # HDF5 checkpoint: rebuild from the stored signals, resampling native
            # modalities to ``target_sr`` and applying ``clock_mul`` / ``t0`` at load.
            # ``sensor_map`` / ``sensor_name_replace`` are ignored (already embedded).
            from delsys import _hdf5

            _hdf5.read_into(self, fname, target_sr=target_sr, clock_mul=clock_mul, t0=t0)
            return
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

        # Raw (un-snapped) time extent — only EMGworks needs it (target-dependent
        # window); Discover's window is ``(0, duration)`` and target-independent.
        self._raw_window: Optional[Tuple[float, float]] = None
        if parser_tag == "emgworks":
            _win: Dict[str, float] = {}
            self.t_min, self.t_max, self.sr_orig, self.signals = _parse_dataframe_emgworks(
                df,
                self.signal_map,
                sensors_info,
                self.target_sr,
                sig_names,
                time_names,
                self.clock_mul,
                self.t0,
                out_window=_win,
            )
            self._raw_window = (_win["raw_t_min"], _win["raw_t_max"])
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

    #: Attributes whose first access hydrates a lazily-loaded HDF5 checkpoint
    #: (see :func:`delsys._hdf5.read_into`). Everything signal-derived flows through
    #: ``sensors`` (the bundle properties, ``find``, ``__getitem__``), so this set is
    #: the complete trigger surface.
    _H5_LAZY_ATTRS = ("sensors", "signals", "sr_orig", "sensor_groups")

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup misses. For a Log built from an HDF5
        # checkpoint, the signal datasets are read on the first such access.
        if name in Log._H5_LAZY_ATTRS and "_h5_deferred" in self.__dict__:
            from delsys import _hdf5

            _hdf5.hydrate(self)
            return self.__dict__[name]
        raise AttributeError(name)

    def to_hdf5(self, path: str) -> str:
        """Write this ``Log`` to a self-contained HDF5 checkpoint.

        Each modality is stored in whatever rate this ``Log`` currently holds:
        modalities loaded with ``target_sr=None`` are stored native and can be
        re-resampled on reload; everything else is a terminal snapshot. The
        channelmap and header are embedded, so the source CSV is no longer needed.
        Reload with ``Log(path, target_sr=..., clock_mul=..., t0=...)``.

        For the canonical CSV → native-checkpoint conversion, prefer
        :func:`delsys.to_native_h5`, which loads at native rate for you.

        Args:
            path: Output ``.h5`` path.

        Returns:
            ``path`` (for chaining).
        """
        from delsys import _hdf5

        _hdf5.write(self, path)
        return path

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

    # Derived from ``self.sensors`` (not ``self.signals``) so very-old pickles
    # with empty per-:class:`Signal` ``meta`` still resolve — the Sensor
    # metadata is repaired by :meth:`Sensor.__setstate__`, the per-Signal
    # meta is not. Same rationale as the 0.4.0 ``_normalize_signal_lengths``
    # / ``_splice_emg_back`` fixes.
    modalities: Set[str] = property(  # type: ignore[assignment]
        lambda self: {m for s in self.sensors for m in s.modalities}
    )
    sampling_rates: Set[float] = property(  # type: ignore[assignment]
        lambda self: set([x.sr for x in self.signals])
    )
    locations: Set[Optional[str]] = property(  # type: ignore[assignment]
        lambda self: {s.location for s in self.sensors}
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
        for sensor in self.sensors:
            for modality in sensor.modalities:
                ret.setdefault(modality, set()).add(sensor.name)
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
        lambda self: _aggregate_bundles([s.emg for s in self.sensors if hasattr(s, "emg")], EMG)
    )
    ekg: Optional[EKG] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles([s.ekg for s in self.sensors if hasattr(s, "ekg")], EKG)
    )
    acc: Optional[IMU] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles([s.acc for s in self.sensors if hasattr(s, "acc")], IMU)
    )
    gyro: Optional[IMU] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles([s.gyro for s in self.sensors if hasattr(s, "gyro")], IMU)
    )
    fsr: Optional[FSR] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles([s.fsr for s in self.sensors if hasattr(s, "fsr")], FSR)
    )
    analog = property(
        lambda self: _aggregate_bundles([s.analog for s in self.sensors if hasattr(s, "analog")])
    )
    vo2master: Optional[VO2Master] = property(  # type: ignore[assignment]
        lambda self: _aggregate_bundles(
            [s.vo2master for s in self.sensors if hasattr(s, "vo2master")],
            VO2Master,
        )
    )
    hrstrap = property(
        lambda self: _aggregate_bundles([s.hrstrap for s in self.sensors if hasattr(s, "hrstrap")])
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
    # EMG / EKG artifact cleaning
    # ------------------------------------------------------------------

    def annotate_noise(self, path: Optional[str] = None):
        """Open the interactive per-signal noise annotator over this Log.

        Launches the delsys ``SignalBrowser`` subclass (see
        :mod:`delsys.annotate`): browse the signals by their structural key,
        drag-select noise windows / mark dead channels, and save them to a
        ``<stem>.delsys-noise`` sidecar that :func:`delsys.clean` auto-consumes.

        datanavigator is imported lazily here, so the delsys core stays
        datanavigator-free until this method is called.

        Args:
            path: Sidecar path to read/write. Defaults to a sibling
                ``<stem>.delsys-noise`` of this Log's source file.

        Returns:
            The annotator instance (a ``SignalBrowser`` subclass).
        """
        from delsys.annotate import launch_noise_annotator

        return launch_noise_annotator(self, path)

    def clean_emg_ekg_artifact(
        self,
        *,
        config: Optional[CleaningConfig] = None,
        motion: Optional[Union[str, Dict[int, Union[int, str]]]] = "auto",
        in_place: bool = True,
        generate_report: bool = True,
        splice_source: str = "combined",
    ) -> CleaningResult:
        """Clean ECG and motion artifact from every EMG channel in this Log.

        Stages:

        1. **Gather** — :attr:`emg` (with :meth:`pysampled.Data.shift_baseline`
           applied so the per-channel dB metrics in the report reflect
           the cleaning's effect on the AC signal rather than a constant
           offset in the raw input) + :attr:`ekg` + (optional) per-EMG
           ACC predictors resolved from ``motion``.
        2. **Harmonize** — defensively resample EMG / EKG / ACC to the
           EMG bundle's sampling rate. EMG passes through unchanged; the
           rest are tail-trimmed to a common length.
        3. **Pipeline** — preprocess → ICA-based ECG suppression with
           auto component detection by lagged correlation → optional
           ACC-guided motion regression with safety gates. See
           :func:`delsys.cleaning.run_pipeline`.
        4. **Splice-back** — when ``in_place=True`` (default), replace
           the EMG samples in :attr:`signals` and rebuild every sensor
           that carries EMG so :attr:`emg` and :attr:`sensors[*].emg`
           reflect the cleaned data on next access. Note that the
           baseline-shift in step 1 is part of what gets spliced back,
           so any DC offset in the raw EMG is permanently removed once
           ``clean_emg_ekg_artifact()`` runs in place.

        Args:
            config: Pipeline knobs. Defaults to :class:`CleaningConfig`'s
                defaults.
            motion: ACC predictor source.

                * ``"auto"`` (default) — pair each EMG sensor with its
                  own ACC bundle when present (Trigno Avanti sensors
                  carry both modalities). EMG sensors with no ACC pair
                  pass through the ECG stage only.
                * ``dict`` of ``{emg_sensor_number: target}`` — explicit
                  mapping. ``target`` is either an integer sensor number
                  whose ACC bundle to use, or a string matched against
                  :attr:`Sensor.location`.
                * ``None`` — skip the motion stage regardless of
                  ``config.use_motion_stage``.

            in_place: If ``True`` (default), mutate :attr:`signals` and
                rebuild every EMG sensor; the returned
                :class:`CleaningResult` is for diagnostics. If ``False``,
                do not mutate; just return the result so the caller can
                splice manually.
            generate_report: If ``True`` (default), write a multi-page
                PDF report to ``<source_csv_stem>_cleaning_report.pdf``
                next to the input CSV after the run completes (and
                after the in-place splice-back, when applicable).
                Equivalent to calling
                :meth:`CleaningResult.generate_report` on the returned
                result. Pass ``False`` to skip the PDF step.
            splice_source: Which cleaned variant to splice back into
                :attr:`signals` when ``in_place=True``. One of
                ``"combined"`` (default — :attr:`CleaningResult.cleaned_emg`,
                preprocess + ECG + motion), ``"ekgonly"``
                (:attr:`CleaningResult.cleaned_emg_ekgonly`, preprocess +
                ECG only), or ``"motiononly"``
                (:attr:`CleaningResult.cleaned_emg_motiononly`,
                preprocess + motion only). Use
                ``splice_source="ekgonly"`` when the motion stage is
                doing more harm than good on a particular trial, or
                ``"motiononly"`` to keep just the motion regression.
                Ignored when ``in_place=False``. The auto-report runs
                *after* the splice-back, so the per-channel pages
                reflect what ``lf.emg`` will look like.

        Returns:
            :class:`CleaningResult` containing the cleaned EMG matrix,
            per-stage snapshots, and diagnostics.

        Raises:
            ValueError: If the Log has no EMG bundle, or if a
                ``motion`` dict references a sensor number that does
                not exist or has no ACC modality.

        Example:
            .. code-block:: python

                import delsys
                from delsys import CleaningConfig

                lf = delsys.Log("trial.csv")

                # Default: auto ECG component removal, auto ACC pairing
                result = lf.clean_emg_ekg_artifact()

                # Manual ECG component override
                cfg = CleaningConfig(
                    ecg_auto_remove_components=False,
                    ecg_components_to_remove=[2, 5],
                )
                result = lf.clean_emg_ekg_artifact(config=cfg)

                # Drive motion with explicit ACC pairing
                result = lf.clean_emg_ekg_artifact(motion={4: 4, 14: 11})

                # Dry-run, do not mutate
                result = lf.clean_emg_ekg_artifact(in_place=False)
        """
        if config is None:
            config = CleaningConfig()
        if self.emg is None:
            raise ValueError("Log has no EMG bundle to clean.")

        # Pre-flight the report path so a locked PDF (file open in another
        # viewer) fails fast — before we burn time on the ICA fit and
        # before the splice-back mutates ``lf.signals``.
        if generate_report:
            from delsys.cleaning import _check_report_path_writable

            _check_report_path_writable(self.fname)

        # Shift the baseline up front so the per-channel dB metrics in
        # the report reflect the cleaning's effect on the AC signal rather
        # than a constant offset in the raw input. This also feeds a
        # better-conditioned matrix into FastICA.
        emg_bundle = self.emg.shift_baseline()
        emg_sr = float(emg_bundle.sr)
        emg_2d = np.asarray(emg_bundle())

        # Per-sensor channel layout in the aggregate EMG matrix: each EMG
        # sensor contributes ``len(SUBCHANNEL_MAP[mod])`` columns in
        # ``Log.sensors`` order (matching ``_aggregate_bundles`` /
        # ``Sensor.__init__``). Cache this so the splice-back, the
        # feature_names list, and the acc_by_emg map all index by the
        # same column ordering.
        emg_layout: List[Tuple[Any, str]] = []  # (sensor, modality) per EMG sensor
        feature_names: List[str] = []
        for sensor in self.sensors:
            mod = next((m for m in sensor.modalities if m.startswith("EMG")), None)
            if mod is None:
                continue
            emg_layout.append((sensor, mod))
            feature_names.extend(sensor.emg.signal_names)

        # EKG reference: collapse to 1-D. Multi-EKG logs use the first
        # column — a deliberate simplification; users wanting to mix
        # multiple EKG references can pre-process and pass the result
        # through delsys.cleaning.run_pipeline directly.
        ekg_1d: Optional[np.ndarray] = None
        ekg_sr: Optional[float] = None
        if self.ekg is not None:
            ekg_arr = np.asarray(self.ekg())
            ekg_1d = ekg_arr if ekg_arr.ndim == 1 else ekg_arr[:, 0]
            ekg_sr = float(self.ekg.sr)

        # Resolve the per-EMG-sensor ACC predictor according to ``motion``.
        # Build acc_by_emg keyed by *EMG column index* in ``emg_2d``;
        # all sub-channels of one EMG sensor share the same predictor.
        acc_by_emg: Dict[int, np.ndarray] = {}
        acc_sr: Dict[int, float] = {}
        if motion is not None:
            col_idx = 0
            for sensor, mod in emg_layout:
                n_subch = len(SUBCHANNEL_MAP[mod])
                acc_sensor = self._resolve_acc_for_emg(sensor, motion)
                if acc_sensor is not None and hasattr(acc_sensor, "acc"):
                    acc_arr = np.asarray(acc_sensor.acc())
                    rate = float(acc_sensor.acc.sr)
                    for k in range(n_subch):
                        acc_by_emg[col_idx + k] = acc_arr
                        acc_sr[col_idx + k] = rate
                col_idx += n_subch

        harmonized = harmonize_multirate_inputs(
            emg_2d=emg_2d,
            emg_sr=emg_sr,
            ekg_1d=ekg_1d,
            ekg_sr=ekg_sr,
            acc_by_emg=acc_by_emg if acc_by_emg else None,
            acc_sr=acc_sr if acc_sr else None,
            target_sr=emg_sr,
        )

        result = run_pipeline(
            harmonized["emg"],
            harmonized["sr"],
            ekg_1d=harmonized["ekg"],
            acc_by_emg=harmonized["acc_by_emg"] or None,
            feature_names=feature_names,
            time=None,
            config=config,
        )
        result.diagnostics["harmonization"] = {
            "target_sr": harmonized["sr"],
            "n_samples": harmonized["n_samples"],
            "has_ekg": harmonized["ekg"] is not None,
            "n_acc_streams": len(harmonized["acc_by_emg"]),
            "backend": "pysampled",
        }
        result.fname = self.fname
        result.feature_names = list(feature_names)

        if in_place:
            splice_choices = {
                "combined": result.cleaned_emg,
                "ekgonly": result.cleaned_emg_ekgonly,
                "motiononly": result.cleaned_emg_motiononly,
            }
            if splice_source not in splice_choices:
                raise ValueError(
                    f"splice_source must be one of {sorted(splice_choices)}; "
                    f"got {splice_source!r}."
                )
            chosen = splice_choices[splice_source]
            if chosen is None:
                raise ValueError(
                    f"splice_source={splice_source!r} requested but "
                    f"the corresponding stage didn't run (variant is None)."
                )
            self._splice_emg_back(np.asarray(chosen), emg_layout)

        if generate_report:
            result.generate_report()

        return result

    def _resolve_acc_for_emg(
        self,
        emg_sensor: Sensor,
        motion: Union[str, Dict[int, Union[int, str]]],
    ) -> Optional[Sensor]:
        """Look up the ACC source sensor for one EMG sensor.

        ``"auto"`` returns the EMG sensor itself when it carries an
        ACC modality (typical for Trigno Avanti). A dict maps EMG sensor
        numbers to either another sensor's number (int) or a sensor
        ``location`` string (case-sensitive substring match — same
        convention as :meth:`find`'s ``location`` filter).

        Returns ``None`` when no ACC pair is available for this EMG
        sensor.
        """
        if motion == "auto":
            return emg_sensor if hasattr(emg_sensor, "acc") else None
        if isinstance(motion, dict):
            target = motion.get(emg_sensor.number)
            if target is None:
                return None
            if isinstance(target, int):
                for s in self.sensors:
                    if s.number == target and hasattr(s, "acc"):
                        return s
                raise ValueError(
                    f"motion={{{emg_sensor.number}: {target}}}: "
                    f"sensor {target} not found or has no ACC modality."
                )
            if isinstance(target, str):
                for s in self.sensors:
                    if s.location and target in s.location and hasattr(s, "acc"):
                        return s
                raise ValueError(
                    f"motion={{{emg_sensor.number}: {target!r}}}: "
                    f"no sensor with that location carries ACC."
                )
            raise TypeError(f"motion dict values must be int or str, got {type(target).__name__}.")
        raise ValueError(f"motion must be 'auto', a dict, or None; got {motion!r}.")

    def _splice_emg_back(
        self,
        cleaned_2d: np.ndarray,
        emg_layout: List[Tuple[Sensor, str]],
    ) -> None:
        """Replace EMG sample arrays in ``self.signals`` and rebuild EMG sensors.

        Walks ``emg_layout`` (the same per-sensor / per-modality / per-
        sub-channel ordering used to build the EMG matrix passed into
        the pipeline) and column-pairs it with ``cleaned_2d``. Each
        target :class:`Signal` is replaced by ``signal._clone(col)``,
        preserving ``_t0`` / ``meta`` / ``_history``. Then each affected
        sensor's ``emg`` bundle is replaced with a fresh
        :class:`pysampled.Data._clone` of the cleaned block, so
        ``lf.emg`` reflects the cleaning even on very-old pickles where
        per-:class:`Signal` ``meta`` is empty (and the per-Signal splice
        above silently no-ops).
        """
        col = 0
        for sensor, mod in emg_layout:
            for subchannel in SUBCHANNEL_MAP[mod]:
                for i, sig in enumerate(self.signals):
                    if (
                        sig.sensor is not None
                        and sig.sensor.number == sensor.number
                        and sig.modality == mod
                        and sig.subchannel == subchannel
                    ):
                        self.signals[i] = sig._clone(cleaned_2d[:, col])
                        break
                col += 1

        # Defensive: in the rare case the pipeline shifts sample count
        # (shouldn't happen since we run at one canonical rate inside),
        # absorb any drift before the per-Sensor stack assert fires.
        self.signals = _normalize_signal_lengths(self.signals)

        # Update each affected sensor's ``emg`` bundle directly from the
        # cleaned block. ``_clone`` preserves the bundle's
        # ``signal_names`` / ``signal_coords`` / ``meta``, so the
        # aggregate ``lf.emg`` view stays well-labelled.
        col = 0
        sensors_by_number = {s.number: s for s in self.sensors}
        for sensor, mod in emg_layout:
            n_subch = len(SUBCHANNEL_MAP[mod])
            block = cleaned_2d[:, col : col + n_subch]
            target = sensors_by_number.get(sensor.number)
            if target is not None and hasattr(target, "emg"):
                target.emg = target.emg._clone(block)
            col += n_subch

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


def to_native_h5(
    csv_path: str,
    out_h5: Optional[str] = None,
    *,
    sensor_map: Optional[Union[str, List[SensorLog]]] = None,
    sensor_name_replace: Optional[Dict[str, str]] = None,
) -> str:
    """Convert a Delsys CSV to a NATIVE-rate HDF5 checkpoint (Trigno Discover only).

    The canonical ``csv -> .h5`` step: loads at the acquisition rate (no resampling
    of Trigno-Base modalities; link devices keep their target rate), embeds the
    channelmap and header, and writes ``float32`` + ``lzf``. ``clock_mul`` and ``t0``
    are intentionally left at identity — alignment to another clock is applied later,
    on load (``Log(out_h5, clock_mul=..., t0=...)``), not baked into storage.

    The resulting ``.h5`` is self-contained, so the source CSV can be regenerated on
    demand or discarded.

    Args:
        csv_path: Path to the Delsys CSV.
        out_h5: Output ``.h5`` path. Defaults to ``csv_path`` with an ``.h5`` suffix.
        sensor_map: Channelmap (path or pre-parsed list); ``None`` auto-builds from the CSV.
        sensor_name_replace: Optional sensor-name correction map.

    Note:
        EMGworks checkpoints store signals on the widest (min_sr=1) window and trim to
        the requested window on reload; reload at ``clock_mul=1`` reproduces
        ``Log(csv, target_sr)`` bitwise. A ``clock_mul != 1`` reload of an EMGworks
        checkpoint is rejected (its native interpolation grid is fixed at export time);
        Discover checkpoints have no such restriction.

    Returns:
        The output ``.h5`` path.
    """
    from delsys import _hdf5

    if out_h5 is None:
        out_h5 = os.path.splitext(csv_path)[0] + ".h5"
    lf = Log(
        csv_path,
        sensor_map=sensor_map,
        target_sr=_hdf5.NATIVE_SR,
        clock_mul=1.0,
        t0=0.0,
        sensor_name_replace=sensor_name_replace,
    )
    _hdf5.write(lf, out_h5)
    return out_h5
