# Plan 0.5.0 — in-log noise marking + per-log sidecar + SignalBrowser

Fresh-session design doc. Builds on the shipped batch-cleaning layer
(`delsys.clean` + `delsys_cleaning.json` manifest + `delsys._noise` consume
hook, commit `7fa41cd` on `feature/hdf5-checkpoint`). Goal: let a human mark
noisy segments **per signal** inside a single Delsys file, store them in a
per-log sidecar, and adopt datanavigator's `SignalBrowser` (with a new signal
dropdown) — subclassed in delsys — as the authoring + interactive-cleaning UI.

This is a **delsys-file / delsys-folder centric** view: a `.delsys-noise`
sidecar lives next to one `Trial_N.h5` and travels with it. We deliberately do
**not** impose or draw on any project/session organization (unlike wobble's
trial-id-tuple Events).

## Locked decisions (from the 2026-05-28 design Q&A)

1. **API name: `lf.annotate_noise()`** is the entry point (not `mark_noise`).
   Opens the delsys `SignalBrowser` subclass for the loaded `Log`.

2. **Sidecar key = one string combining the structural address AND the human
   label** (combine proposal options #1 + #2 into a single key string):

   ```
   "<sensor>.<modality>[.<coord>] | <label>"
   ```

   - Left of `" | "` = the **authoritative, channelmap-stable** structural
     address. `<modality>` ∈ EMGS/EMGD/EMGQ/ACC/GYRO/FSR; `<coord>` ∈ that
     modality's `SUBCHANNEL_MAP` entry (EMGS=`A`; EMGD=`A,B`; EMGQ=`A,B,C,D`;
     ACC/GYRO=`X,Y,Z`; FSR=`A,B,C,D`). **Omitting `<coord>` = the whole
     modality / all axes.**
   - Right of `" | "` = the bundle **label** (the location-derived
     `signal_names`), informational only (display + sanity cross-check), ignored
     on read so a relabel never breaks lookup.
   - Worked examples (covers every required case):
     - single EMG → `"3.EMGS | Tricep_L"`
     - ACC all 3 axes → `"4.ACC | Bicep_R"`
     - ACC one axis → `"4.ACC.X | Bicep_R"`  *(both granularities required)*
     - gyro all axes → `"4.GYRO | Bicep_R"`
     - Quattro sub-channel → `"7.EMGQ.A | Forearm"`
     - dead FSR channel → `"9.FSR.C | LFoot_Ball"` (window = whole extent)
   - Resolution of a key → concrete signal columns reuses the same
     sensor/modality/sub-channel matching `clean_emg_ekg_artifact`'s splice-back
     uses (`sig.sensor.number` / `sig.modality` / `sig.subchannel`). A
     coord-less key fans out to every sub-channel of that sensor+modality.

3. **Sidecar format: `<stem>.delsys-noise`** (JSON content, composite suffix —
   avoids portfolio `*.json` sweeps per the no-`.json`-sidecar convention). The
   delsys `SignalBrowser` subclass owns read/write. Value per key = list of
   `[t_start, t_end]` windows in seconds on the Log's clock (same NaN+interp
   consumption as `delsys._noise` today, but addressed per-signal).

4. **Two artifacts, not one** (the clean-code argument the user asked for):
   - `delsys_cleaning.json` (per-**folder**) — *algorithmic* cleaning decisions
     keyed by trial stem; the `clean()` reproducibility contract; mostly
     machine-written.
   - `<stem>.delsys-noise` (per-**log**) — *human* noise windows keyed by signal
     address; GUI-authored; meaningful even if `clean()` never runs.
   They differ in **granularity** (folder/trial-keyed vs file/signal-keyed),
   **author/lifecycle** (programmatic vs hand-drawn in a GUI), and **scope**
   (delsys-file-centric, travels with the `.h5`). Folding noise into the
   per-folder manifest would nest a per-signal map under every trial and mix two
   key spaces. The manifest's existing **`noise_event_ref`** field is exactly the
   decoupling: it should default to / point at the sibling `<stem>.delsys-noise`,
   so the manifest records *which* noise sidecar was applied (provenance) while
   the sidecar is the single source of truth for the windows. `clean()` should
   auto-consume a sibling `<stem>.delsys-noise` when present.
   - *Open to revisit:* if the user prefers full file-centricity, the
     alternative is collapsing the per-folder manifest into a per-log
     `Trial_N.delsys-clean` holding both cleaning + noise. Bigger refactor of
     just-shipped code; recommended only if the per-folder view is unwanted.

5. **delsys `SignalBrowser` subclass scope: noise + interactive cleaning** (one
   integrated review tool). Mark/edit noise windows per signal AND drive
   `clean_emg_ekg_artifact` (choose ICA components, preview cleaned vs raw,
   write the `delsys_cleaning.json` decision) from the same browser.

## Build phases

**Phase A — datanavigator: signal dropdown in the Qt sidebar.**
Locate `SignalBrowser` in the datanavigator repo. Add a dropdown to the Qt
sidebar that switches the displayed signal among a provided list (general
feature, lands in datanavigator core — not delsys-specific). Keep it generic:
the browser is handed a list of named signals and the dropdown selects one.

**Phase B — delsys: data layer (no GUI).**
- `<stem>.delsys-noise` read/write + the `"<sensor>.<modality>[.<coord>] |
  <label>"` key grammar (parse/format/resolve-to-columns).
- Extend `delsys._noise.apply_noise_mask` to apply windows **per resolved
  signal address** (today it masks all signals for a flat interval list).
- `clean()`: default `noise_event_ref` to a sibling `<stem>.delsys-noise`.
- Tests for parse/format/resolve + per-signal masking (mirror `test_clean.py`).

**Phase C — delsys: `SignalBrowser` subclass + `lf.annotate_noise()`.**
- Subclass datanavigator's `SignalBrowser`; populate the Phase-A dropdown from
  `lf` (every signal, labelled by the key grammar).
- Noise marking writes `<stem>.delsys-noise`.
- Interactive cleaning: component pick / preview / write `delsys_cleaning.json`.
- `lf.annotate_noise()` launches it.

**Docs/changelog:** update `tutorials/workflow.md` (the noise-authoring section
currently describes the dnav-Event path; add the `lf.annotate_noise()` +
`.delsys-noise` per-log flow). CHANGELOG `[Unreleased]`. Still no version bump.

## Open questions for the fresh session

- **Dropdown UX granularity** in the browser: list one entry per *sub-channel*
  (e.g. `4.ACC.X/Y/Z` separately) or per *sensor+modality* with axis overlay?
  Marking should still support both whole-modality and per-axis windows.
- **Default marking granularity**: does dragging a window default to the whole
  sensor+modality (all axes/sub-channels) or just the displayed channel, with a
  modifier for the other? (User: "we will need both.")
- **Interactive cleaning ↔ manifest**: exact UX for choosing ICA components and
  the write-back to `delsys_cleaning.json` (and how a noise edit triggers a
  re-clean / marks the cleaned `.h5` stale).
- **FSR "dead channel"**: window = full `[t_min, t_max]`, or a dedicated
  `dead: true` flag in the sidecar instead of a degenerate window?
- **datanavigator dependency**: `annotate_noise()` needs datanavigator at
  import time. Keep it an optional/lazy import (delsys core stays dnav-free, as
  `_noise` consumption is today) — confirm the dep posture.
