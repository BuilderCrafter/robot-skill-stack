# Robot Skill Stack

A modular Isaac Sim stack for the Franka Panda: asynchronous manipulation skills,
Behavior Trees, RGB-D perception, cached semantic observations, spatial occupancy
mapping and optional Lula RRT global planning.

MoveToPose, Home, Pick and Place retain the direct RMPflow manipulation baseline.
Static ground-truth global RRT is independently selectable. Frozen RGB-D global
RRT supports occupied-cell proxies and optional measured-box reconstruction; it
is experimental, opt-in and can refuse incomplete or stale observations.

## Quick start

Use the existing university workstation installation: **Isaac Sim 5.1.0-rc.19,
Python 3.11, NumPy 1.26.4 and py_trees 2.6.0**. From the repository root:

```bash
./run_isaac_sim.sh
```

Set `ISAAC_SIM` if the installation differs from `/home/etfrobotics/isaacsim`.
Open `scenes/playground.usd`, select `config/scenes/playground.toml` in the
**Robot Skill Stack** panel and press **Initialize Runtime**. Existing launchers,
repo-local `.deps`, USD scenes and default direct configuration are preserved.
Restart Isaac before switching setups. Do not introduce `--no-ros-env`.

The multi-view demonstration uses `scenes/multiview_perception_final.usda` and
`config/scenes/multiview_perception_final.toml`, with the original two 640×480
cameras. See [configuration](docs/configuration.md) for explicit source selection.

## Motion options

Direct RMPflow remains the default and the complete Pick/Place baseline.
For independently selectable static GT global planning:

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "static"
```

The nominal reconstructed RGB-D option uses:

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "frozen_rgbd"
global_rgbd_representation = "reconstructed"
max_obstacles = 64
```

It additionally requires enabled `rgbd` Spatial Mapping with automatic robot
self-filtering. `occupied` remains available. Reconstruction preserves the entire
original selected occupied union, protected floor evidence and the 64-proxy bound.
It never imports GT obstacle geometry. Refusals have no automatic GT/direct fallback.
Global motion retains intended-contact/carrying restrictions; use direct for
ordinary manipulation. See [global planning](docs/motion_planning.md) for limits.

## Validation

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group global --artifacts-dir outputs/global
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping --artifacts-dir outputs/mapping
```

Native cases run sequentially and write fresh results. The global group covers
GT and nominal reconstructed RGB-D, with an independent cooked-geometry veto
before selected execution. Inspect actual motion, endpoint errors, contacts,
source identity and cleanup; process startup alone is not a passing scenario.
See [validation](docs/validation.md) and the [integration report](docs/integration-rgbd-global-rrt.md).

## Architecture and documentation

```text
GUI / Behavior Trees → RobotRuntime → skills → ManipulationBackend → MotionStrategy
RGB-D / GT → bounded observations → WorldModel → skills / GUI
RGB-D / GT → independent Spatial Mapping → immutable frozen planning source
```

Generic modules contain no Isaac imports. Bootstrap owns SDK construction,
cameras, workers and lifecycle; motion strategies own planner/controller cleanup.
The optional reactive obstacle-avoidance strategy is retired and rejected by
configuration. Ordinary direct RMPflow and manipulation recovery remain available.

- [Architecture](docs/architecture.md), [configuration](docs/configuration.md), [GUI](docs/gui.md).
- [Manipulation](docs/manipulation.md), [perception](docs/perception.md), [spatial mapping](docs/spatial_mapping.md).
- [Motion planning](docs/motion_planning.md), [validation](docs/validation.md), [control diagnostics](docs/control_diagnostic.md).

Home is a captured startup TCP reference, not a certified safe configuration.
Finger positions do not prove object identity or prevent slipping. Sampled
clearance/contact evidence does not certify continuous swept collision safety.
Historical experiment tooling and reports remain on the preserved
[`feature/rgbd-rrt-autonomous`](https://github.com/BuilderCrafter/robot-skill-stack/tree/feature/rgbd-rrt-autonomous) branch.
