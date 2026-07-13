# Reviewing and persisting EKG R-peaks

Auto-detection gets most R-peaks right, but real recordings need a human pass —
a missed beat, a double-counted QRS, a stretch of motion noise, an inverted
lead. `delsys` gives you an **interactive reviewer** to fix those, and a
**sidecar** that stores the correction so every later load reproduces the
*curated* peaks — no re-review, no pickle.

```python
import delsys

lf = delsys.Log("Trial_1.h5")     # or Trial_1.csv
ekg = lf.ekg                       # aggregate EKG bundle (auto-detected)
ekg.review()                       # opens the reviewer
```

## The reviewer

`EKG.review()` opens a window over the EKG channel(s):

- **top** — the raw trace with peak markers: `*` auto-detected, `+` you added,
  `x` you removed; noisy segments are shaded.
- **middle** — inter-beat interval (ms) over time.
- **bottom** — a histogram of inter-beat intervals.

Controls (buttons mirror the keys where noted):

| key | button | action |
|-----|--------|--------|
| `a` | | add an R-peak at the cursor (snapped per the *edit mode*) — or, if you're on a peak you removed, **restore** it |
| `d` | | remove the R-peak nearest the cursor — or **undo** it if you added it |
| `n` | | mark a noisy segment — press once for the start, again for the end |
| `f` | **Flip** | flip polarity **and re-detect** |
| `m` | **Mode** | cycle the add-snap: `peak` / `valley` / `exact` |
| `1` `2` `3` | | tag `reviewed` / `representative` / `interesting` |
| `s` | **Save** | write the decision to the sidecar |
| `ctrl+k` | **Help** | show the full key-binding cheatsheet |

`a` and `d` are inverses — re-adding a peak you removed just restores it (no
duplicate), and removing one you added undoes the addition. The shortcut legend
is drawn on the figure, and the **Help** button (or `ctrl+k`) shows the full
list. The raw/IBI zoom persists across edits, so you can work zoomed in.

> **Flip first.** Flipping polarity re-runs detection from scratch, which resets
> your manual removals. If a lead is inverted, press **Flip** *before* you start
> adding/removing peaks.

A multi-channel EKG (more than one EKG sensor) is split automatically — step
between channels with the **channel** dropdown; each is saved separately.

## What gets saved

**Save** writes to a sibling `<stem>.delsys-events` file (the same unified
sidecar `lf.view()` uses for noise/markers). R-peak curation lands in its
`rpeaks` type, one entry per channel:

```json
{
  "events": {
    "rpeaks": {
      "signals": {
        "12.EKG.A | Chest": {
          "detector": {"name": "pn", "highpass": 5.0, "hr_max": 200.0},
          "added":   [5.51],
          "removed": [9.39],
          "flipped": false,
          "tags":    ["reviewed"]
        }
      }
    },
    "noise": {"signals": {"12.EKG.A | Chest": {"windows": [[12.0, 13.0]]}}}
  }
}
```

Peaks are stored as **times, not sample indices**, so the decision reproduces on
any sample grid — the native-rate `.h5`, a resampled load, or a slice all give
the same curated peaks. Only your *removals* are stored; the detector's own
double-peak prune is regenerated on load. Noisy segments go into the shared
`noise` track, so `delsys.clean` sees them too (scoped to the EKG channel).

## Loading the curated peaks

Just load the file again — `lf.ekg` applies the saved decision automatically:

```python
lf = delsys.Log("Trial_1.h5")
peaks = lf.ekg.rpeak_times()[0]        # the curated final peaks
t_peaks = lf.ekg.t[peaks]
tx, bpm = lf.ekg.ihr()                 # instantaneous heart rate
```

- `lf.ekg` reproduces the curation (re-detects with the stored settings, applies
  your add/remove/flip, drops peaks inside noisy segments).
- `lf.ekg_raw` bypasses it and gives you the plain auto-detection.

Programmatic access to the same round-trip:

```python
ekg = lf.ekg_raw
ekg.load_rpeaks()                      # apply the sidecar (returns True if found)
# ... edit ekg.meta ...
ekg.save_rpeaks()                      # write it back
```
