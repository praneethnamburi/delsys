# TODO

Items deferred during the initial standalone-package extraction.

## Restructure / API

- **`Sensor.__init__` modality dispatch.** It uses an if/elif chain on modality strings to choose which class to instantiate. A small `MODALITY_REGISTRY` mapping `{'EMG': EMG, 'EKG': EKG, ...}` would be cleaner and would make adding modalities (e.g. SmO2/Thb that already appear in `TARGET_SR`) one-line changes.

- **`Log.__getitem__` overloading.** Mixes sensor-lookup and signal-lookup based on key type (int, single-letter, modality string, location, name). Works but is undiscoverable. The new `lf.find(...)` is the public-facing replacement; `__getitem__` could be deprecated when there's a transition window.

- **EKG annotation I/O.** `EKG.find_rpeaks_hp` and `EKG.clean_rpeaks` read/write `*_ekginfo.json` next to the source CSV. Mixing file I/O into signal-processing makes EKG awkward to test (every test needs a tmpdir). Extract into a small `RPeakAnnotations` helper that EKG composes with.

- **VO2 / HR identity.** `VO2_SENSOR_NUM` and `HR_SENSOR_NUM` are placeholder integers chosen to not collide with Trigno-base sensor numbers. Brittle if Delsys ever ships a third link device. A real fix identifies link sensors by type rather than number — would also clean up the special-cased branches in `_parse_sig_name_discover` and `Sensor.__init__`.

## Pre-existing in-code TODOs

- **EKG**: "Slicing will not work well with the cached rpeak indices. TODO: modify the `__getitem__` method."
- **EKG**: "Modify path to be an attribute from Signal so there will be no need to include it as an argument" (R-peak annotation file path).
- **`_parse_dataframe_discover_with_link`**: "Make the exception explicit. The 13.5 ms sampling-rate calculation only applies for trigno base, not link devices."
- **VO2 link**: "VO2 can start delayed (fill these data points with zeros), VO2 can finish before the other system."
- **`discover_basic` parser path**: "Check if this works for a file with timestamps exported." Needs a fixture with timestamps but no link sensors.

## Domain / units

- VO2Master `VO2_absolute` returns raw CSV values (~37000 for moderate exercise). Likely a units issue, not a column-mapping one — verify against a known-correct VO2 reading and add unit conversion if needed.

## Tests to add (Stage 4 of the original extraction plan)

1. CSV header parsing — `_parse_hdr`, `_parse_sig_name`, `_detect_parser` per version.
2. End-to-end `Log` loading — one trimmed fixture per header format.
3. Signal classes — IMU axis access, FSR channel access, VO2Master column mapping (regression test for the bug fixed on 2026-05-08).
4. `EMG.process` envelope pipeline.
5. `EKG` R-peak detection (NeuroKit's `ecg_simulate` for synthetic input).
6. ICA cleaning.
