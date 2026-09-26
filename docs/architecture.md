# Architecture

```text
Orchestration (Behavior Trees / future external planner)
                 │
             RobotRuntime
                 │
       Manipulation skills
                 │
      GraspPlanner + Backend
                 │
          Isaac integration

RGB-D sensor → Perception → WorldModel
                         ↑        │
                         └─ action-aware association hints
```

## Boundaries

`runtime`, `world`, `manipulation`, and `orchestration` are simulator-agnostic. `integrations/isaac` owns Omniverse/Isaac imports. Perception produces observations; the WorldModel is the cached state queried by skills and orchestration. Skills never query Isaac directly.

A future thesis layer should depend primarily on `RobotRuntime` and `WorldModel`. ROS 2, if required, should be added as another adapter under `integrations/` rather than becoming the internal architecture.
