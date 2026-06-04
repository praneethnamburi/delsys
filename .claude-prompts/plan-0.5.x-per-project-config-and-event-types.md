# Plan: per-project config + interactive event-type management

Status: **designed, not yet built** (design session 2026-06-03/04). The unified
`.delsys-events` sidecar + `lf.view()` shipped first (commits `ba517d8`..`8ebb6fc`,
0.5.0a2). This doc captures the agreed next build so a fresh session can execute
without re-deriving the decisions.

## Problem being solved

`lf.view()` today marks noise + **anonymous numbered** marker tracks (`"1"`, `"2"`),
fixed at launch via `events=`. There is no way to:
- name a type ("movement-onset" vs "noise"),
- create / rename / remove types from inside the annotator,
- share a type vocabulary across a study's trials.

Marking "this is where the participant started moving" is therefore not first-class.

## Decisions (locked this session)

- **Per-project config subsystem**, not just an event-type registry. `target_sr`
  and future per-project settings live here too.
- **One file, TOML** (`delsys_project.toml`) in the project repo, committed. The
  GUI *writes* the event-type section, so the format must be machine-round-trippable
  — `.py` is out (can't safely rewrite). Use **`tomlkit`** (new dep) so a GUI write
  surgically updates the `[[event_types]]` tables and **preserves comments / other
  sections**. Works on the 3.10 floor.
- **Project resolution:** `DELSYS_PROJECT_CONFIG` env var → else walk up from the
  trial folder for a `delsys_project.toml` → else built-in defaults (current
  `("1",1),("2",2)` point+window pair).
- **Types are slug-keyed.** `.delsys-events` references a type by stable `slug`;
  the config maps `slug → {label, key, size, color}`. **Rename = edit `label`**,
  zero file migration, correct everywhere. **Remove** = drop from config; trials
  still carrying that slug render as "unknown/retired", not silently dropped.
- **Modal style:** adopt dustrack's `ConfirmOverlay` (translucent, blocking,
  role-styled buttons, default-button safety) as datanavigator's **canonical**
  modal. Promote the proven code up rather than rewrite; dustrack then consumes it.
- **Per-event notes/tags:** make the `.delsys-events` event record *format* carry an
  optional `note` / `tags` (an event becomes `[t]` **or** `{"seq":[t],"note":...,
  "tags":[...]}`; readers normalize, `collapse_markers` surfaces them). UI to author
  notes is **deferred** — just don't design it out.

### Config file shape
```toml
[settings]
target_sr = { EMGS = 2000.0, ACC = 148.1 }   # wiring deferred (see below)

[[event_types]]
slug = "movement-onset"   # stable id written into .delsys-events
label = "Movement onset"  # rename edits this only
key = "1"                 # single char (digit) to fire on keypress
size = 1                  # 1 = point, 2 = window
color = "tab:green"
```

## Build cuts (each a commit; mirror the last cut's phasing)

1. **delsys, no Qt — foundation (testable down payment).**
   - `delsys/_project.py`: resolve + load `delsys_project.toml` via `tomlkit`
     (env → walk-up → defaults); cache by path.
   - `delsys/_event_types.py`: slug model + read/write the `[[event_types]]` tables
     (surgical, comment-preserving) + a `default_event_types()` template +
     `scaffold(path)` to write a starter config.
   - `view()` loads its marker tracks from the resolved registry (explicit
     `events=` arg still overrides); `_normalize_marker_specs` grows a registry path.
   - `.delsys-events`: make the marker record note/tag-ready (extend `_events`
     `_as_sequences`/`write_events`/`collapse_markers`; annotator can stay seq-only
     for now since no UI authors notes yet — but prefer storing records to avoid a
     future drop-on-save footgun when the notes UI lands).
   - Tests: config resolution, registry round-trip (comments preserved), slug→spec,
     note/tag passthrough.
   - **Keep `target_sr` *wiring* OUT of this cut** (user steer). The TOML has a home
     for it, but feeding it into `Log(target_sr=...)` intersects the open
     "reconsider TARGET_SR defaults" item in `TODO.md` and deserves its own pass.

2. **datanavigator — modal primitives.** Lift dustrack's `_make_confirm_overlay_class`
   (in `dustrack/dustrack/_overlays.py`, lines ~370–603) up as a public canonical
   `confirm(figure, title, message, buttons, default, severity, checkboxes)` built on
   the Qt window datanavigator already holds (`find_qt_window`); add a **text-input**
   dialog (`prompt_text(figure, title, prompt, default)`) in the same style. Off-Qt
   fallback returns the default / None so headless tests don't block. CHANGELOG on
   `master`. (This is also the primitive the deferred **warn-on-close** needs.)

3. **dustrack — consume the promoted overlay.** Re-point `dustrack/_overlays` /
   `_close_guard` at datanavigator's confirm; retire the local copy. Re-run
   `tests/test_save_on_close.py`.

4. **delsys — interactive type management in `view()`.** New / Rename / Remove type
   buttons using the datanavigator dialogs; live key rebinding; writes back to the
   project `delsys_project.toml`.

## Carry-over follow-ups (already in TODO.md)
- Multi-char marker-track keybindings (today only single-key names fire).
- Warn-on-close (unblocked once cut 2 lands the dialogs).
- Public `Log`-level trial-marker reader; marker reconciliation policy.
- Per-event notes/tags **UI**.

## How to test the *already-shipped* part (needs a real display)
The `lf.view()` GUI couldn't be verified headlessly — run it on a real screen:
`b4` env, `C:\Users\praneeth\anaconda3\envs\b4\python.exe`, open a Trigno `.h5`,
`lf = delsys.Log(...); lf.view()`, mark noise (`n`/`alt+n`/`d`) + markers (`1`/`2`),
**Save**, and confirm the sibling `.delsys-events` + that `delsys.clean()` consumes it.
