# Plan delsys 0.4.0 — Log.clean_emg_ekg_artifact

You're starting a fresh session in `C:\dev\delsys`. The task this
session is to **design** the EMG/EKG artifact cleaning feature for the
next minor release (0.4.0). Do **not** implement code yet — produce a
concrete plan we can execute in a follow-up session. Stay in plan mode
until the user is satisfied with the design.

## Authoritative reading list

Read these in order. They give you the full state of the project — do
not rely on what's in this prompt as the canonical version.

1. `C:\dev\delsys\CLAUDE.md` — repo-local instructions, points at the
   spec and conventions.
2. `C:\dev\pn-specs\specs\delsys.md` — product intent, audience, what
   "success" looks like, the roadmap entry for 0.4 that you're now
   filling in.
3. `C:\dev\pn-specs\CONVENTIONS.md` — cross-cutting conventions
   (docstring style, formatter, build backend) — assume they apply.
4. `C:\dev\delsys\CHANGELOG.md` — release history; note 0.1.1, 0.2.0,
   0.3.0 patterns for breaking-change framing and migration tables.
5. `C:\dev\delsys\TODO.md` — deferred work, especially "Followups
   (target 0.4.0)" and "Post-0.1.0 roadmap".
6. `C:\dev\pn-projects\projects\emg_ica_cleaning.py` (961 lines) — the
   source pipeline to port. Skim the `def `/`class ` index first, then
   read in detail.

## Environment

- Working dir: `C:\dev\delsys`. Git repo, default branch `main`.
- Conda env name: **`b4`** — Python 3.10.13, has delsys 0.3.0
  installed editable. Run tests via
  `conda run -n b4 pytest -q` (Bash sees `conda` only via PowerShell;
  prefer the PowerShell tool for env-aware commands).
- Platform: Windows; shell is PowerShell.
- `pysampled` lives at `C:\dev\pysampled` and is imported from there
  in dev. delsys depends on `pysampled>=1.2.0`.

## What 0.4.0 needs to deliver

The headline feature is a `Log`-integrated method (working name
`Log.clean_emg_ekg_artifact`) that ports the multi-stage pipeline from
`emg_ica_cleaning.py` and runs it end-to-end on a loaded `Log`. The
spec roadmap entry summarises the pipeline as:

> harmonization → preprocess → ICA-based ECG suppression with
> auto-component-detection by lagged correlation → ACC-guided motion
> regression with safety gates

Concrete integration tasks the spec already names:

1. **Gather inputs** from `lf`: all EMG `Signal`s + the EKG `Signal`,
   plus optional per-EMG ACC predictors.
2. **Run the pipeline** (port of `emg_ica_cleaning.py`).
3. **Splice cleaned samples back** into `lf.signals` per channel and
   rebuild affected `EMG` bundles in `lf.sensors[*].emg` (since
   `EMG._sig` is built by stacking signals at `Sensor.__init__` time).

Pending non-headline work that may ride along (decide whether 0.4.0
or punt):

- Migrate `_aggregate_bundles` to `pysampled.Data.merge_along_signal_name`
  once pysampled ships those classmethods. (Check pysampled's TODO at
  `C:\dev\pysampled\TODO.md` to see whether 1.3.0 is close.)

## Design questions you must resolve in the plan

Don't punt these. Answer each with a recommendation + the trade-off:

1. **Public API surface.** One `Log` method, or a method plus a
   lower-level `delsys.cleaning` module exposing the building blocks
   (ICA fit, component selection, regression) for power users? What do
   the kwargs look like — one big config dict, a dataclass like
   `EMGPipelineConfig`, or flat kwargs?
2. **Component selection UX.** The source code supports three modes:
   manual selection by index list, automated selection via lagged
   correlation against EKG, and an interactive prompt. Which of these
   ship in 0.4.0? An interactive prompt is a poor fit for a library
   call; the auto path is the obvious default; the manual override
   needs a clean kwarg.
3. **Splice-back contract.** Three options for how cleaned data gets
   back to the `Log`:
   - mutate `lf` in place (`lf.clean_emg_ekg_artifact(...)`),
   - return a new `Log` (immutable),
   - return cleaned bundles and let the caller decide.
   Pick one with rationale. Mutating in place needs to also rebuild
   `lf.sensors[*].emg` and refresh `lf.emg` (the aggregate is computed
   on access so that part is free, but `Sensor.emg` is not).
4. **Optional ACC predictors.** How does the user opt in to motion
   regression? Auto-discover via `lf.find(modality='ACC')`, or require
   an explicit map? What's the safety gate from the source pipeline,
   and does it survive translation?
5. **Multi-rate handling.** Source has `harmonize_multirate_inputs`
   that uses `pysampled.uniform_resample`. delsys already resamples at
   load time per `TARGET_SR`. Decide: trust load-time rates and assume
   inputs are already commensurate, or run the harmonization stage
   defensively?
6. **Realtime variant.** Source has `_run_pipeline_realtime` and
   `_run_pipeline_offline`. Recommend offline-only for v1; if you
   disagree, justify.
7. **Plotting helpers.** Source has `plot_ica_components` and
   `plot_signals_before_after`. delsys is currently plotting-free at
   the API level (uses `pysampled` which has its own helpers).
   Recommend: ship plotting in a separate module, or skip for v1?
8. **Test strategy.** No real fixture has known ECG contamination
   ground-truth. Plan a synthetic test: generate clean EMG +
   superimpose a synthetic ECG onto it via a known mixing matrix +
   verify ICA recovers the EMG within tolerance. Specify what "within
   tolerance" means (correlation threshold against the known clean
   signal? Reduction in power at heart-rate harmonics?).
9. **Breaking changes?** Likely none in the public Log API, but
   double-check whether the splice-back affects any of the 0.2.0
   aggregate accessor invariants (especially `meta["sensors"]` and
   `signal_names` ordering after a rebuild).

## Output

Write the plan to
`C:\Users\praneeth\.claude\plans\<descriptive-name>.md`. Structure:

- **Context** — why this is happening (point at spec roadmap + the
  source file).
- **Public API proposal** — signatures, kwargs, return type. Show
  example call sites.
- **Internal architecture** — what new module(s), what gets moved
  where, which utilities to reuse from `pysampled` and from
  `delsys._util`.
- **Per-design-question decisions** — answer each of the 9 questions
  above with a recommendation and the trade-off considered.
- **Splice-back contract** — explicit description of how
  `lf.signals` and `lf.sensors[*].emg` get updated.
- **Test plan** — what's synthetic, what's fixture-based, what
  invariants to assert.
- **CHANGELOG draft** — Removed (BREAKING) / Deprecated / Added /
  Changed / Internal sections, in the same style as 0.3.0.
- **TODO.md updates** — mark "Shipped in 0.4.0" entries, move
  remaining 0.3.0 followups appropriately.
- **File-by-file change list** — exact paths to be touched, ready to
  hand to an implementation session.

When the plan is solid, exit plan mode for user approval. Don't start
implementing — that's the next session.

## Don'ts

- Don't restate this prompt in the plan; assume the reader sees the
  plan, not the prompt.
- Don't propose features that aren't in `emg_ica_cleaning.py` or the
  spec roadmap. Scope creep here is dangerous because the source
  pipeline is already big.
- Don't block on unanswered questions — make a recommendation and
  flag it; the user will redirect if needed.
- Don't run `git tag` or `git push` or `flit publish`. Tagging and
  PyPI publish stays manual (user runs them after approval).
