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
                                 ObjectObservation[]
                                             │
Isaac ground truth ───────────────→ WorldModelUpdater → WorldModel
                                     ↑                   │
                                     └─ held / placement hints
```

## Boundaries

`runtime`, `world`, `manipulation`, and `orchestration` are simulator-agnostic.
`integrations/isaac` owns Omniverse/Isaac imports. Perception and ground truth
implement `WorldObservationProvider.observe(context)`; `WorldModelUpdater`
samples the selected source through an Isaac physics callback. Skills query
cached WorldModel state and call a manipulation backend, without querying Isaac.

The composition root is `integrations/isaac/bootstrap.py`. It loads a TOML
profile, validates stage prims, constructs Franka and the selected source, and
registers Home, MoveToPose, Pick and Place. Production uses
`SimplePrimitiveGraspPlanner`. Simple placement feasibility checks target
occupancy, not the entire transport path.

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
the extension also reads backend `is_ready` and runtime `busy`, invokes updater cleanup,
and accesses perception capture. These are existing
integration couplings, not capabilities of the generic presentation model.

## Launch and lifecycle

`run_isaac_sim.sh` delegates to `run_isaac_python.sh`, which prepends `.deps` and
the repository to `PYTHONPATH`. `scripts/launch_isaac_gui.py` imports and verifies
NumPy 1.26.4 before `SimulationApp`, selects the full experience, disables ROS
bridge startup, and enables the UI extension. No `--no-ros-env` argument is used.
Callback ownership, deferred World construction, camera cleanup, and deferred
UI disposal remain in the existing Isaac integration/lifecycle code.
