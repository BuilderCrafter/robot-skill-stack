# Robot Skill Stack

Deterministic lower-level robotics stack for the course project, structured so
future thesis-level planning can integrate without rewriting the robot layer.

## Architecture

```text
orchestration / Behavior Trees
            │
        RobotRuntime
            │
    manipulation skills
      │             │
 GraspPlanner   PlacementPlanner   ManipulationBackend
                      │
               Isaac integration

RGB-D sensor → perception → WorldModel
                         ↑       │
                         └ action-aware association hints
```

The generic package does not import Isaac Sim. Simulator-specific code is
isolated under `robot_skill_stack/integrations/isaac`.

## Repository layout

```text
robot_skill_stack/
  common/                     shared types
  runtime/                    skill API, registry, runtime
  world/
    model/                    WorldModel, observations, provider/updater
    perception/               RGB-D discovery, tracking, geometry
  manipulation/
    backend.py                simulator-agnostic manipulation contract
    skills/                   home, move, pick, place
    grasping/                 replaceable feasibility-aware grasp planners
    placement/                target-occupancy feasibility
  presentation/               simulator-independent WorldModel UI view
  orchestration/
    behavior_trees/           BT nodes, tasks, executor
  integrations/
    isaac/                    Isaac bootstrap, camera, Franka backend, GT oracle

config/                       scene/runtime profiles
scenes/                       Isaac USD scenes
tests/                        unit, Isaac integration, regression
scripts/                      regression-suite runners
docs/                         architecture and subsystem documentation
exts/                         optional Isaac UI extension
```

## Current course-level capabilities

- geometry-first RGB-D discovery and primitive fitting for cubes/boxes, spheres,
  and upright or lying cylinders, without Isaac semantic labels
- persistent object IDs, held-track retention, and placement association hints
- WorldModel with swappable perception / ground-truth providers
- lazy object point-cloud access
- action-aware reacquisition after robot-mediated transport
- reactive Pick with bounded local replanning
- gripper-feedback grasp verification
- placement target occupancy rejection before motion
- stable Place verification
- Behavior Tree Pick+Place and recovery
- multi-object manipulation
- replaceable grasp-planner interface with feasibility rejection
- Isaac GUI with selectable live WorldModel objects

Isaac startup selects the production `SimplePrimitiveGraspPlanner`.
V1 is the supported perception source. ROS 2 and external robot/source adapters are
future work, not implemented integrations.

## Dependencies

The tested repo-local dependencies are:

```text
numpy==1.26.4
py_trees==2.6.0
```

They are intentionally not vendored into the repository. If `.deps` is not
already present on the Isaac machine, recreate it from the repo root:

```bash
./run_isaac_python.sh -m pip install --target .deps -r requirements.txt
```

## Isaac GUI launch

Set `ISAAC_SIM` to the installation directory if it differs from
`/home/etfrobotics/isaacsim`, then run `./run_isaac_sim.sh` from the repository.
The launcher verifies NumPy 1.26.4 before importing `SimulationApp`, uses the
full Isaac experience, and disables its ROS bridge startup. Isaac Sim
5.1.0-rc.19's `python.sh` does not support `--no-ros-env`; do not add it.
Open `scenes/playground.usd` and initialize `config/scenes/playground.toml`.
See `docs/gui.md` for switching to ground truth.

## Validation

Manipulation has one cooperative command API:
`await runtime.execute_async("pick", object_id=...)` (likewise Home, Move and
Place). The GUI and Behavior Trees use this path; standalone Isaac tests drive
the same coroutine with Kit's existing bridge. `runtime.busy` reports an active
command; overlapping commands fail with `FailureCode.BUSY` rather than queuing.
See `docs/phase_3a_execution.md` for interface migration and validation.
The merged held-state upgrade is described in [Phase 3B](docs/phase_3b_held_state.md).
The observation cleanup and its pending workstation validation are described in
[Phase 4A](docs/phase_4a_world_model.md).

Use one runner from the repository root with Python dependencies installed:

```bash
python3 scripts/run_all_regressions.py --unit-only
python3 scripts/run_all_regressions.py --group isaac
python3 scripts/run_all_regressions.py --group regression
```

`--unit-only` runs static checks, unittest discovery, and the eight standalone
unit modules with the invoking Python; it does not require Isaac. The `isaac`
and `regression` groups require the workstation and dispatch through the existing
Isaac wrapper with pinned NumPy pre-import. Omit the group to run everything.
Add `--artifacts-dir outputs/regressions` to retain logs, scenario results and
artifacts in a fresh run directory; otherwise output goes to the terminal.

See `docs/validation.md` for the manual Isaac smoke checklist,
`docs/phase_2_pruning.md` for before/after results, and `docs/history.md` for
recovering removed experiments from the preservation tag.

See `docs/architecture.md`, `docs/perception.md`, `docs/manipulation.md`,
`docs/validation.md`, `docs/gui.md`, and `docs/ros2.md`.
