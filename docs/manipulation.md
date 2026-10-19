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
returns `TARGET_OCCUPIED` before issuing any backend motion or gripper command;
physical settling can continue during simulation updates. Invisible/stale
objects are not treated as hard obstacles in this V1 checker.

## Held-state evidence

Finger-position feedback distinguishes Free, Held and Uncertain. Pick records
the grasp before lifting; carrying motions check for loss; failed/interrupted
gripper operations attempt bounded resolution. Uncertain state blocks commands
and protects the candidate object until resolved. See
[Phase 3B held state](phase_3b_held_state.md) for policy, API changes and pending
workstation validation. No force sensing is used.

## Current execution limitations

Place supports `mode="stable"` (default) and `mode="release"`. Stable mode checks
the observed final object position; release mode allows an intentional midair
drop after reaching the requested object-center release pose and opening the
gripper. Failed approach/descent/open commands remain failures in either mode.
The GUI Place button uses stable mode. No release/drop toggle is planned;
intentional release remains available through the skill API. See the
[controller diagnostic](control_diagnostic.md#latest-workstation-result-and-merge-checks)
for the passing native benchmark and operator-confirmed final validation.

All command callers use `await runtime.execute_async(...)`. Synchronous skill and
backend commands have been removed; standalone Isaac scripts drive the same
coroutine through the external Kit bridge. Read-only queries stay synchronous.
Motion tolerances and update budgets are unchanged; bounded gripper inspection
adds observation updates around grasp/release and ambiguous feedback.
Franka's `speed` argument is diagnostic and does not scale motion.
Its `home` is the TCP pose captured during backend construction, not a separately
configured safe joint pose. Gripper width feedback verifies grasping; it is not
a force/contact sensor. Place reports `retreat_ok` separately and can currently
report success despite a failed retreat. Broader recovery policy, retreat
reporting, speed and Home semantics remain deferred beyond this held-state step.
