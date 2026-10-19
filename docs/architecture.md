# Architecture

```text
Orchestration (Behavior Trees / future external planner)
                 │
             RobotRuntime
                 │
       Manipulation skills
          │            │
   SimplePrimitiveGraspPlanner / SimplePlacementPlanner
          └──────┬─────┘
                 │
              Backend
                 │
          Isaac integration

RGB-D → Discovery → PrimitiveEstimator → ObjectTracker
                                             │
                                  ObservationBatch
                                             │
Isaac ground truth ───────────────→ WorldModelUpdater → WorldModel
                                     ↑                   │
                                     └─ held / placement hints
```

## Boundaries

`runtime`, `world`, `manipulation`, and `orchestration` are simulator-agnostic.
`integrations/isaac` owns Omniverse/Isaac imports. Perception and ground truth
implement `observe(context) -> ObservationBatch`, `forget(id)` and `close()`.
A batch distinguishes a fresh sample (including an empty frame) from no new
sample; source errors raise. `WorldModelUpdater` owns scheduling, source health,
visibility, timestamp expiration and coordinated forgetting through the existing
Isaac physics callback. Sources own their resources, including camera cleanup.
WorldModel stores object/attachment state and placement hints. Skills query
cached WorldModel state and call a manipulation backend, without querying Isaac.

The composition root is `integrations/isaac/bootstrap.py`. It loads one TOML
setup and validates the robot plus only the selected source's required prims.
`manipulation/factory.py` selects the robot SDK wrapper, backend and gripper
geometry; `world/factory.py` selects the observation source. Franka is the only
implemented robot; RGB-D V1 and ground truth are the two sources. Unsupported
selectors fail explicitly. No registry or configuration inheritance is involved.

Construction retains its order: World/context, robot wrapper, reset/play, 30 Kit
updates, source (RGB-D camera initialization and 60 updates), updater, backend,
planners, skills/runtime, initial observation and the existing physics callback.
The runtime and skills share the same WorldModel. Production uses
`SimplePrimitiveGraspPlanner`. Simple placement feasibility checks target
occupancy, not the entire transport path. Callback ownership and cleanup remain
with the existing bundle/updater lifecycle.

`[scene]` owns support height and workspace bounds. Algorithm thresholds use
their existing Python defaults; Franka construction supplies its gripper spec.
RGB-D requires neither ground-truth objects nor their SDK wrappers. Optional
`[ground_truth]` references apply only to oracle startup. Native tests own their
playground cube handles in `tests/support/isaac_helpers.py`; these handles are
not part of the production bundle. See the [Phase 4B migration](phase_4b_config_adapters_ui.md).

There is one cooperative asynchronous command path:
`RobotRuntime.execute_async` → skill `execute_async` → backend async command.
The runtime rejects overlapping commands with `BUSY` and releases command
ownership after results, exceptions or cancellation. `BaseSkill` and
`ManipulationBackend` declare this API explicitly. Backend readiness, pose,
reachability and grasp feedback are synchronous queries; gripper min/max width
are simple capability values. Only the Isaac Franka backend is implemented.

`SkillNode` schedules that same runtime coroutine; `BTExecutor.run_async` ticks
without blocking Kit. The recovery selector remembers a running recovery branch
instead of re-entering Pick during Home. Tick-budget exhaustion/cancellation
cancels and drains pending node tasks before returning. This is coroutine
cleanup, not a physical robot stop or held-state recovery policy.

Standalone Isaac tests use the existing `tests/support/isaac_async.py` Kit bridge,
which pumps `SimulationApp.update()` until the coroutine finishes. The backend
yields on Kit updates and never advances simulation through blocking `world.step`
loops or worker threads. No synchronous robot-command implementation remains.

V1 is the geometry-based source for boxes, spheres and cylinders.

A future thesis layer should depend primarily on `RobotRuntime` and `WorldModel`. ROS 2, if required, should be added as another adapter under `integrations/` rather than becoming the internal architecture.


## Presentation boundary

`robot_skill_stack.presentation` contains a simulator-independent read model for
WorldModel state. The Isaac extension renders that read model and invokes
`RobotRuntime`. Object display and manipulation work with either source, but
the extension also reads backend `is_ready` and runtime `busy`, and accesses
an optional bundle `capture_frame(path)` callback for diagnostics. Ordinary
refresh is read-only. Clear Lost requests updater cleanup for either source;
Clear Held keeps its confirmation and calls the runtime's software-only reset,
which rejects busy commands and changed object IDs. Source health is displayed
in the existing source label and gates observation-dependent controls. The
extension closes the bundle rather than managing a source or camera directly.

## Launch and lifecycle

`run_isaac_sim.sh` delegates to `run_isaac_python.sh`, which prepends `.deps` and
the repository to `PYTHONPATH`. `scripts/launch_isaac_gui.py` imports and verifies
NumPy 1.26.4 before `SimulationApp`, selects the full experience, disables ROS
bridge startup, and enables the UI extension. No `--no-ros-env` argument is used.
Callback ownership, deferred World construction, camera cleanup, and deferred
UI disposal remain in the existing Isaac integration/lifecycle code.

See [Phase 4A](phase_4a_world_model.md) for source contracts and accepted tracking
tradeoffs, and [Phase 4B](phase_4b_config_adapters_ui.md) for adapter construction
and its outstanding native validation.
