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
 GraspPlanner   ManipulationBackend
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

- geometry-first RGB-D cube discovery without Isaac semantic labels
- persistent object IDs and repeated semantic belief
- WorldModel with swappable perception / ground-truth providers
- lazy object point-cloud access
- action-aware reacquisition after robot-mediated transport
- reactive Pick with bounded local replanning
- gripper-feedback grasp verification
- stable Place verification
- Behavior Tree Pick+Place and recovery
- multi-object manipulation
- replaceable grasp-planner interface with feasibility rejection

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

## Validation

Run every retained lower-level regression suite:

```bash
python3 scripts/run_all_regressions.py
```

Or run them separately:

```bash
python3 scripts/run_perception_v1_suite.py
python3 scripts/run_phase_a_suite.py
python3 scripts/run_phase_b_suite.py
```

Each suite writes logs/JSON under ignored `outputs/` and packages its results as
a `.tar.gz` archive.

See `docs/architecture.md`, `docs/perception.md`, `docs/manipulation.md`,
`docs/validation.md`, and `docs/ros2.md`.
