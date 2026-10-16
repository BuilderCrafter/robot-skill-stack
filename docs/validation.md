# Lower-level validation

Historical results recorded on the Isaac Sim workstation on 2026-09-26
(not revalidated by the Phase 1 cloud run):

- Perception V1: complete suite PASS; single-cube localization approximately 2 mm and same-ID reacquisition after Pick/Place.
- Robust execution (Phase A): all five stages PASS; deterministic workspace benchmark mean localization error about 1.85 mm, worst observed about 2.41 mm; 4/4 Pick+Place benchmark trials; mean physical placement error about 4.93 mm; reactive Pick, BT recovery and multi-object manipulation PASS.
- Grasp planning (Phase B): unit and Isaac stages PASS; 90 mm cube rejected before motion with `NO_VALID_GRASP`; normal 50 mm perception-driven cube still Pick+Place successful with about 6 mm physical target error.

The regression suites are retained under `tests/` and runners under `scripts/`.
Historical experiments and Phase 1 records are recoverable as described in
[history.md](history.md).

## Plain-Python checks

From the repository root, use a Python 3.11/3.12 environment with
`requirements.txt` installed. The cloud run used the existing `.venv`; this is
an offline test environment, not a substitute for Isaac's `.deps` launch path.

```bash
python -m unittest discover -s tests/unit -t .
for module in test_async_runtime test_grasp_planning test_perception test_placement \
              test_primitive_perception test_reactive_pick test_world_model_update_rate \
              test_world_model_view; do
    python -m "tests.unit.$module" || exit
done
python scripts/check_architecture.py
```

The eight standalone modules are not executed by unittest discovery. Main unit
tests include fake Isaac/UI adapters; they do not start the simulator or
prove real camera, controller, or GUI behavior. The static checker compiles Python
and checks generic-package imports; it does not verify all architectural coupling.

## Isaac-dependent checks

`tests/isaac/` and `tests/regression/` start `SimulationApp`; they require the
workstation. Suite parent scripts may run under ordinary Python, but invoke
`run_isaac_python.sh` for children. `run_all_regressions.py --unit-only` still
uses that wrapper. It is not a
standalone plain-Python test command.


## Course-polish suite

```bash
python3 scripts/run_course_polish_suite.py
```

It validates placement occupancy logic, the simulator-independent WorldModel GUI
view model, and an Isaac regression proving an occupied placement is rejected
before robot motion while a free target still succeeds.

The canonical full regression command remains:

```bash
python3 scripts/run_all_regressions.py
```

This includes architecture checks, the primitive unit suite,
Perception V1, Phase A, Phase B, and the course-polish suite. Historical suites
overlap; no runners or execution paths were consolidated in Phase 1.

## Manual GUI smoke test

Run on the university Isaac Sim 5.1.0-rc.19 workstation; these checks are pending
until performed there:

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
