# Manipulation and grasp planning

The manipulation module owns robot-facing skills, the generic backend contract,
and replaceable grasp planners. Isaac-specific execution lives under
`robot_skill_stack/integrations/isaac/manipulation`.

## Skill path

```text
RobotRuntime
   ↓
Pick / Place / MoveToPose / Home
   ↓
GraspPlanner + ManipulationBackend
   ↓
IsaacFrankaBackend (current integration)
```

`PickSkill` is reactive: after pre-grasp and grasp approach it compares the
latest visible WorldModel position with the position used to create the grasp.
Movement above 20 mm causes a local replan, bounded to two replans before
`OBJECT_MOVED` is returned.

Physical grasp success is verified by backend gripper feedback rather than by
requiring camera visibility while the object is occluded by the hand.

## Grasp planner interface

`GraspPlanner.plan(...)` returns a `GraspResult` containing either a valid
`GraspPlan` or a structured `GraspFailureReason`.

Isaac bootstrap constructs `SimplePrimitiveGraspPlanner` for cubes/boxes,
spheres, and upright or lying cylinders. It checks that:

- the object pose is known,
- the object is marked graspable,
- supported primitive geometry and dimensions are known,
- the requested strategy is top-down,
- a candidate fits the configured parallel-jaw gripper,
- contact height clears the support plane and palm,
- known-object swept envelopes and IK allow the approach and lift.

Unknown primitives and unsupported tilted orientations are rejected.

The Franka integration configures a 75 mm usable maximum width. A 50 mm cube is
accepted; a 90 mm cube is rejected before robot motion.

## Future grasp planners

A future planner can implement `GraspPlanner`. The perception provider already
offers lazy `get_point_cloud(object_id)` access, but current planners do not use
it. General learned or arbitrary-object grasp planning is not implemented.


## Placement feasibility

`SimplePlacementPlanner` performs a conservative AABB overlap check before any
placement motion. It compares the held object's estimated size and requested
target against other currently visible WorldModel objects. An occupied target
returns `TARGET_OCCUPIED` before the end effector moves. Invisible/stale objects
are not treated as hard obstacles in this V1 checker.

## Current execution limitations

All command callers use `await runtime.execute_async(...)`. Synchronous skill and
backend commands have been removed; standalone Isaac scripts drive the same
coroutine through the external Kit bridge. Read-only queries stay synchronous.
The existing async command tolerances, update budgets and sequence are unchanged.
Franka's `speed` argument is diagnostic and does not scale motion.
Its `home` is the TCP pose captured during backend construction, not a separately
configured safe joint pose. Gripper width feedback verifies grasping; it is not
a force/contact sensor. Place reports `retreat_ok` separately and can currently
report success despite a failed retreat. Uncertain held state, recovery motion,
retreat reporting, speed and Home semantics remain reserved for Phase 3B.
