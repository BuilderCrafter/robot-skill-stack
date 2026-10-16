# Phase 1 baseline and low-risk cleanup

## Starting state and scope

Branch: `refactor/phase-1-baseline`.
Starting commit: `5cd7dc8d303c5b9604b17c18391aabc5c3baaefe`.
The checkout was clean and identical to the audited commit before changes.
No AGENTS.md instructions were found under `/workspace`.

Phase 1 fixes launch dispatch, removes verified inert API/configuration, and
documents the implemented system. It does not implement the audit's proposed
architecture. V1 retains cubes, spheres and cylinders; V2, both grasp planners,
classifiers, sync/async paths, tracking, behavior trees, and Isaac lifecycle
mechanisms remain. No dependency pins or SSH configuration were changed.

## Commands and before/after results

All commands ran from the repository root under Python 3.12.14, using the
existing `.venv` with NumPy 1.26.4 and py_trees 2.6.0. Before any code edits, the
offline dataset tests additionally received their pinned dependencies:

```bash
.venv/bin/python -m pip install --no-cache-dir \
    opencv-python==4.11.0.86 pillow==11.2.1 PyYAML==6.0.2
```

These packages were installed only in the ignored cloud test environment, not
Isaac or `.deps`. No PyTorch/Ultralytics installation or training was performed.

| Command / check | Before | After |
|---|---|---|
| `.venv/bin/python -m unittest discover -s tests/unit -t .` | 362 passed | 363 passed |
| Eight standalone unit modules listed below | All 8 passed | All 8 passed |
| `.venv/bin/python -m unittest discover -s tools/perception_v2_data/tests -v` | 71 passed | 71 passed |
| `.venv/bin/python scripts/check_architecture.py` | Boundary and syntax PASS | Boundary and syntax PASS |
| `.venv/bin/python -m pip check` | PASS | PASS |
| `bash -n run_isaac_python.sh run_isaac_sim.sh` | Not separately recorded | PASS |
| `git diff --check` | Clean starting tree | PASS |

The additional main unit test exercises GUI shell dispatch through the real
repository wrappers in a temporary fixture. It verifies argument forwarding
from another working directory with spaces, `.deps` precedence, and exit status
propagation using a fake Isaac launcher and sentinel Python script. It does not
launch or validate Isaac Sim.

The standalone baseline run enumerated test files without `unittest`; this
matched the following explicit list, repeated after the edits:

```bash
for module in test_async_runtime test_grasp_planning test_perception test_placement \
              test_primitive_perception test_reactive_pick test_world_model_update_rate \
              test_world_model_view; do
    .venv/bin/python -m "tests.unit.$module" || exit
done
```

Unittest discovery alone does not execute these eight script entry points. Main
unit tests include V2 and fake simulator/UI adapters. The offline dataset tests
use fake USD/Isaac components and do not validate real rendering or training.
No skips or expected failures were reported in the two discovery runs.

Additional after-change checks loaded both checked-in scene profiles using the
audited and current loaders, comparing all effective dataclass values after
excluding the three removed inert fields. Both profiles matched. All retained
`manipulation.grasping.__all__` exports imported successfully.

`run_all_regressions.py --unit-only` was deliberately not used: it invokes
`run_primitive_perception_suite.py`, which still dispatches through Isaac Python
and includes V2 tests. Tests under `tests/isaac/` and `tests/regression/` start
SimulationApp and were not run. The default Isaac installation and NVIDIA GPU
tooling were unavailable in this cloud machine.

## Files changed and evidence for removals

References were searched across tracked Python, TOML, tests, scripts, docs and
public exports before removal. No repository callers explicitly supplied the
removed Pick argument or consumed the geometry protocol. External callers, if
any, must stop importing/supplying these removed inert APIs.

| Files | Change and evidence |
|---|---|
| `run_isaac_sim.sh` | Remove unsupported `--no-ros-env` dispatch argument and correct comments for the reported Isaac Sim 5.1.0-rc.19 launcher. |
| `tests/unit/test_project_runners.py` | Add shell-dispatch regression test; existing runner behavior and tests remain. |
| `robot_skill_stack/manipulation/skills/pick.py` | Remove `grasp_lift_threshold`, which was accepted and stored but never read. |
| `robot_skill_stack/integrations/isaac/manipulation/franka_backend.py`, `bootstrap.py` | Remove `joint_tolerance` and `max_home_steps`; constructor did not read/store them, and bootstrap was the only explicit caller. |
| `robot_skill_stack/integrations/isaac/config.py`, `config/scenes/playground.toml`, `playground_v2.toml` | Remove `cube_ratio_max`, `cube_min_size`, `cube_max_size`: only declared, parsed and listed in profiles; not passed to a classifier or otherwise consumed. Active semantic-history settings remain. |
| `robot_skill_stack/manipulation/grasping/geometry.py`, `grasping/__init__.py` | Delete `ObjectGeometryProvider` and its public re-export. It was only a future protocol with no consumers. Concrete lazy mask/point-cloud methods remain. |
| `robot_skill_stack/world/perception/factory.py` | Remove two unused local aliases for tracking/semantic settings; `build_tracker()` still reads the settings it uses. |
| `README.md`, `docs/architecture.md`, `perception.md`, `manipulation.md`, `gui.md`, `validation.md` | Correct production planner, multi-primitive V1, actual integration couplings, launch contract, missing ground-truth profile guidance, test requirements, and workstation smoke checklist. |
| `docs/phase_1_baseline.md` | Record this baseline, verification, scope and outstanding checks. |

The launch strategy remains unchanged: `run_isaac_python.sh` prepends `.deps`
and the repository to `PYTHONPATH`; `launch_isaac_gui.py` verifies NumPy 1.26.4
before SimulationApp, selects the full experience, and retains the ROS bridge
startup overrides. Neither of those two files was modified.

## Pending workstation validation

Use the short [manual Isaac smoke checklist](validation.md#manual-gui-smoke-test)
for GUI startup, WorldModel initialization, V1 primitive observations, ground
truth, Home/Move/Pick/Place, rejection cases and lifecycle checks. Follow the
[local ground-truth profile instructions](gui.md#worldmodel-panel); no missing
checked-in profile is assumed. Simulator, physical manipulation, and real GUI
validation are not claimed by the cloud results.

## Deliberately deferred

- V2 archival and its test-fixture dependencies.
- Alternative planner/classifier removal, module merges and runner consolidation.
- Sync/async consolidation and completing the async backend interface.
- GUI/backend coupling, configuration redesign and tracking/lifecycle policies.
- Franka `speed` remains diagnostic; `home` is the initialization TCP pose.
- Held state after a failed lift, unconditional Home-and-retry recovery, and
  Place success after retreat failure remain unresolved behavior questions.
- Test entry points still differ from GUI pre-SimulationApp NumPy initialization.
- The static checker still walks `.venv` and checks imports rather than every
  semantic dependency.

The audit's cube-only suggestion is not adopted: the approved long-term scope
retains all three primitive families and minimal Behavior Trees. No candidate
from the approved inert cleanup required a high-risk refactor or deferral.
