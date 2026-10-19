# Lower-level validation

Phase 3A completion: the reviewed `phase3a-frame-feedback.zip` reports all four
Pick/Place benchmark trials passing, all retreats successful, and maximum true
placement error 9.67 mm (unchanged limit: 25 mm). The operator subsequently
confirmed both final Isaac groups (two integration scenarios and six
regressions) and manual GUI testing passed, and approved merging. See the
[Phase 3A record](control_diagnostic.md#latest-workstation-result-and-merge-checks)
for the distinction between uploaded benchmark evidence and operator-reported
final checks. Latest cloud validation passes 221 discovered unit tests, all
eight standalone offline checks and the architecture check.

Historical results recorded on the Isaac Sim workstation on 2026-09-26
(not revalidated by the Phase 1 cloud run):

- Perception V1: complete suite PASS; single-cube localization approximately 2 mm and same-ID reacquisition after Pick/Place.
- Robust execution (Phase A): all five stages PASS; deterministic workspace benchmark mean localization error about 1.85 mm, worst observed about 2.41 mm; 4/4 Pick+Place benchmark trials; mean physical placement error about 4.93 mm; reactive Pick, BT recovery and multi-object manipulation PASS.
- Grasp planning (Phase B): unit and Isaac stages PASS; 90 mm cube rejected before motion with `NO_VALID_GRASP`; normal 50 mm perception-driven cube still Pick+Place successful with about 6 mm physical target error.

The regression suites are retained under `tests/` and runners under `scripts/`.
Historical experiments and Phase 1 records are recoverable as described in
[history.md](history.md).

## Single regression entry point

From the repository root, use Python 3.11/3.12 with `requirements.txt` installed:

```bash
python scripts/run_all_regressions.py --unit-only
python scripts/run_all_regressions.py --group isaac
python scripts/run_all_regressions.py --group regression
# All groups, retaining optional logs, JSON and scenario artifacts:
python scripts/run_all_regressions.py --artifacts-dir outputs/regressions
```

- `unit` / `--unit-only`: architecture and syntax checks, unittest discovery under
  `tests/unit`, and all eight retained standalone unit scripts. Uses the invoking
  Python without the Isaac wrapper. Fake SDK/UI tests do not validate real Isaac.
- `isaac`: both scenarios under `tests/isaac` (discovery and manipulation).
- `regression`: all six scenarios under `tests/regression` (benchmark, grasp
  planning, reactive pick, recovery, multi-object manipulation and placement).

The last two groups require the Isaac workstation. The wrapper still prepends
`.deps`; each scenario script imports/verifies NumPy 1.26.4 before importing
`SimulationApp`. The runner invokes those files directly, without `-c` or
`runpy`, preserving the normal script `__main__` and argument context. Direct
script invocation has the same NumPy guard. The working GUI launch is unchanged.
See [isaac_runner_startup.md](isaac_runner_startup.md) for the startup assessment
and the workstation procedure if a native crash persists.

Each selected scenario runs once. A failed command or missing/non-PASS Isaac
result fails the run, while the remaining selected checks still execute. Reports
are written in fresh directories so prior successes cannot be reused. Without
`--artifacts-dir`, scenario JSON is temporary and terminal diagnostics remain
visible. With it, each command has a log and `summary.json` records its status
and actual exit code. No tar packaging or milestone wrapper scripts are needed.

The cloud `.venv` is an offline test environment, not a replacement for Isaac's
`.deps` launch path. Unit tests can also be run directly, but discovery alone
omits the eight standalone scripts; the consolidated runner includes them.
The static check excludes local virtual environments and validates generic-package
imports/syntax, not every semantic architectural dependency.

## Manual GUI smoke test

Phase 3A migrates standalone commands and the recovery BT to the same async
runtime used by the GUI. The existing Kit bridge pumps application updates;
do not use `asyncio.run()` to control live Isaac. Direct-script dispatch and
pre-SimulationApp NumPy verification are unchanged. Re-run both workstation
groups after this execution migration; cloud doubles cannot validate rendering,
controller timing, physics callbacks or camera updates during commands.

Run on the university Isaac Sim 5.1.0-rc.19 workstation when revalidating GUI
or runtime changes:

1. Confirm `ISAAC_SIM` points to the installation and `.deps` has the pinned
   NumPy 1.26.4 and py_trees 2.6.0. Launch `./run_isaac_sim.sh`. Check the logged
   NumPy version/path, full experience startup, ROS bridge startup overrides,
   absence of an unsupported-option error, and the **Robot Skill Stack** window.
2. Open `scenes/playground.usd`, select `config/scenes/playground.toml`, and press
   **Initialize Runtime**. Confirm readiness, object rows, selectable IDs and live
   cached poses. Check logs for camera or updater failures.
3. With separated supported objects in the configured workspace, confirm V1
   geometry observations for cubes, spheres and cylinders. Check lost/visible
   state after a controlled occlusion; do not treat an offline fixture as this test.
4. Select a graspable object, run Pick, confirm physical attachment and held state,
   then Place at a known free tabletop target. Confirm physical release, pose,
   same-ID reacquisition, and held-state clearing. Exercise Move and Home in a
   clear workspace. Check GUI responsiveness during motion.
5. Check an occupied placement target and an oversized/unreachable grasp are
   rejected before the corresponding unsafe motion. Review both the result and
   `retreat_ok`; Place can currently report success despite failed retreat.
6. Create the local ground-truth profile described in [gui.md](gui.md), restart
   Isaac, open the same scene and initialize that profile. Repeat WorldModel
   initialization and basic manipulation; no core or GUI edits should be needed.
7. Exercise Stop/Play, reset/reinitialize, and shutdown. Check for duplicate
   callbacks, stale controls, camera cleanup errors, and residual tasks.
8. Repeat sequential commands and both BT tasks. Confirm rendering/perception
   continue during motion and GUI actions stay disabled while a command is busy.
   Recovery must run the existing Home-and-retry sequence without restarting
   Pick while Home is running. Coroutine cancellation is not an emergency stop;
   uncertain physical/held state remains outside this phase.
