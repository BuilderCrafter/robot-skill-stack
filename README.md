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
    perception/               RGB-D discovery, tracking, semantics, geometry
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

On an Isaac workstation, run every retained lower-level regression suite:

```bash
python3 scripts/run_all_regressions.py
```

Or run them separately:

```bash
python3 scripts/run_perception_v1_suite.py
python3 scripts/run_phase_a_suite.py
python3 scripts/run_phase_b_suite.py
python3 scripts/run_course_polish_suite.py
```

The milestone suites write logs/JSON under ignored `outputs/` and package their
results as `.tar.gz` archives. The aggregate also runs the primitive suite,
which runs simulator-independent unit modules. Even `--unit-only` invokes the Isaac Python
wrapper. For direct plain-Python checks, baseline results, and the manual Isaac
smoke checklist, see `docs/validation.md`.

See `docs/architecture.md`, `docs/perception.md`, `docs/manipulation.md`,
`docs/validation.md`, `docs/gui.md`, and `docs/ros2.md`.
