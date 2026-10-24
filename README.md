# Robot Skill Stack

Course-level robotics stack for Isaac Sim and the Franka Panda, with a small
foundation for later planning research. It uses geometry-based RGB-D V1 or
Isaac ground truth, one cached WorldModel and one cooperative async command path.

Capabilities include cube/box, sphere and cylinder fitting; persistent object
IDs and placement reacquisition; gripper-position grasp verification and loss
detection; reactive Pick; stable Place or intentional release through the API;
target-occupancy rejection; two small Behavior Trees; and an Isaac GUI.
Only Franka is implemented. YOLO/V2, ROS and additional robot adapters are not
part of the supported runtime.

## Quick start

On the configured Isaac Sim **5.1.0-rc.19 / Python 3.11** workstation, from the
repository root:

```bash
./run_isaac_sim.sh
```

Set `ISAAC_SIM` if the installation differs from `/home/etfrobotics/isaacsim`.
Open `scenes/playground.usd`, select `config/scenes/playground.toml` in the
**Robot Skill Stack** panel and press **Initialize Runtime**.

The wrappers prepend repository-local `.deps` and import/verify **NumPy 1.26.4
before `SimulationApp`**. The other runtime pin is **py_trees 2.6.0**. Use the
existing workstation installation and dependencies; do not bypass the wrappers
or add unsupported launcher options. See [validation](docs/validation.md) for
launch constraints and [configuration](docs/configuration.md) to select ground
truth or another setup. Restart Isaac before changing setups.

For plain-Python development, use a Python 3.11/3.12 environment with
`requirements.txt` installed. The cloud `.venv` is for offline tests and does
not replace Isaac's `.deps` path.

## Tests and diagnostics

One runner covers all retained checks:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/regression
```

The first command runs architecture/syntax checks, discovered tests and eight
standalone offline modules. The other groups require Isaac and contain two
integration scenarios, direct-strategy acceptance and six regressions. Omit the group to run everything.
Artifacts are written to fresh run directories; a NumPy line or open window is
not a passing scenario. See [validation](docs/validation.md) for the final
acceptance checklist, injected-loss diagnostic and native results.

An opt-in [multi-view RGB-D experiment](docs/multiview_rgbd.md) adds a separate
`multi_view` source and two-camera scene. Its research results, native tests,
limitations and reproduction commands are documented separately; the ordinary
playground continues to use single-camera V1.

The [final cube/sphere perception checkpoint](docs/multiview_perception_final.md)
records the camera study, frozen validation and remaining limitations. For the
presentation, open [the experimental demo scene](scenes/multiview_perception_final.usda)
with [its opt-in configuration](config/scenes/multiview_perception_final.toml).
Both cameras remain editable in Isaac's Stage. Autonomous multi-view manipulation
remains deferred.

## Architecture and layout

```text
GUI / Behavior Trees → RobotRuntime → skills → ManipulationBackend
                                        ↘ grasp / placement checks
RGB-D V1 / ground truth → WorldModelUpdater → WorldModel → skills / GUI
```

The generic runtime, manipulation, world, presentation and orchestration code
has no Isaac imports. `integrations/isaac/bootstrap.py` assembles the robot and
source through explicit factories, using one small TOML per setup. New adapters
implement these existing contracts; no plugin system is needed.
RGB-D processing runs in bounded SDK-free workers. Optional spatial mapping
starts from its TOML section and publishes independent occupancy snapshots;
skills and motion planning do not consume the map yet.
The [motion strategy boundary](docs/motion_planning.md) selects direct RMPflow by
default, preserving existing skills and exposing optional map access for future
strategies. Reactive, global and hybrid execution are not implemented.

| Path | Responsibility |
| --- | --- |
| `robot_skill_stack/runtime/` | Skill interface, registry, async commands, bounded observation workers |
| `robot_skill_stack/manipulation/` | Skills, motion strategy contract, held-state policy, grasp/placement feasibility |
| `robot_skill_stack/world/` | Observations, cached state, perception/tracking, independent occupancy |
| `robot_skill_stack/orchestration/` | Minimal BT tasks and executor |
| `robot_skill_stack/presentation/` | Read-only WorldModel display model |
| `robot_skill_stack/integrations/isaac/` | Composition, Franka backend, camera, ground truth, lifecycle |
| `exts/` | Isaac GUI extension |
| `config/scenes/`, `scenes/` | Setup TOML and USD scene |
| `tests/`, `scripts/` | Offline/native validation and focused diagnostics |

## Documentation

- [Architecture](docs/architecture.md): contracts, execution and lifecycle ownership.
- [Configuration](docs/configuration.md): current TOML schema and source selection.
- [GUI](docs/gui.md): existing controls and explicit operator recovery.
- [Manipulation](docs/manipulation.md): skills, Home, Place results and held state.
- [Perception / WorldModel](docs/perception.md): geometry, observations and identity.
- [Spatial mapping](docs/spatial_mapping.md): automatic runtime mapping, shared frames and V5 validation.
- [Validation](docs/validation.md): commands, acceptance gates and release readiness.
- [Control diagnostics](docs/control_diagnostic.md): instability correction and trace capture.
- [History](docs/history.md): retired experiments and earlier phase reports.

Home is a captured startup TCP reference, not a certified safe configuration.
Endpoint/clearance checks do not establish collision-free transport. Finger
positions do not prove object identity or prevent slipping. Read the limitations
before relying on recovery or manipulating a new scene.
