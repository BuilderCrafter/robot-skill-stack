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

The current `TopDownGraspPlanner` is intentionally cube-focused. It checks that:

- the object pose is known,
- the object is marked graspable,
- object size is known,
- the requested strategy is top-down,
- the horizontal object extent fits the configured parallel-jaw gripper.

The Franka integration configures a 75 mm usable maximum width. A 50 mm cube is
accepted; a 90 mm cube is rejected before robot motion.

## Future grasp planners

A future geometry or ML planner can implement the same protocol. The existing
`ObjectGeometryProvider` protocol matches the perception provider's lazy
`get_point_cloud(object_id)` boundary, so a point-cloud planner can request
geometry only when needed without putting point clouds in the WorldModel.
