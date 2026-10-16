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
`SimplePrimitiveGraspPlanner`; the older cube-focused `TopDownGraspPlanner`
remains in the package and tests. Simple placement feasibility checks target
occupancy, not the entire transport path.

Both synchronous and cooperative asynchronous execution exist. The GUI uses
async skills and BT nodes; standalone tests also use synchronous execution.
The abstract backend currently declares only the synchronous methods, although
the GUI path requires async backend methods as well. Only the Isaac Franka
backend is implemented; replacing it requires composition-root changes.

V1 is the geometry-based source for boxes, spheres and cylinders.

A future thesis layer should depend primarily on `RobotRuntime` and `WorldModel`. ROS 2, if required, should be added as another adapter under `integrations/` rather than becoming the internal architecture.


## Presentation boundary

`robot_skill_stack.presentation` contains a simulator-independent read model for
WorldModel state. The Isaac extension renders that read model and invokes
`RobotRuntime`. Object display and manipulation work with either source, but
the extension also checks Franka articulation readiness, invokes updater cleanup,
and accesses perception capture. These are existing
integration couplings, not capabilities of the generic presentation model.

## Launch and lifecycle

`run_isaac_sim.sh` delegates to `run_isaac_python.sh`, which prepends `.deps` and
the repository to `PYTHONPATH`. `scripts/launch_isaac_gui.py` imports and verifies
NumPy 1.26.4 before `SimulationApp`, selects the full experience, disables ROS
bridge startup, and enables the UI extension. No `--no-ros-env` argument is used.
Callback ownership, deferred World construction, camera cleanup, and deferred
UI disposal remain in the existing Isaac integration/lifecycle code.
