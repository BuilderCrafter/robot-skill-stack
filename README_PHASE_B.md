# Phase B — Grasp planning as a replaceable subsystem

Phase B turns the cube-focused grasp planner into a standalone module with
structured feasibility results while preserving the working lower-level stack.

## Architecture

```text
PickSkill
   ↓
GraspPlanner protocol
   ↓
┌──────────────────────┐
│ TopDownGraspPlanner  │  current implementation
└──────────────────────┘
   ↓
GraspResult
   ├── valid GraspPlan
   └── or structured failure reason
```

Future planners can implement the same protocol, for example:

```text
PointCloudGraspPlanner
   ↓
ObjectGeometryProvider.get_point_cloud(object_id)
   ↓
candidate generation / scoring / feasibility
   ↓
GraspResult
```

No point-cloud planner is implemented in Phase B.

## Structured result

`GraspResult` returns either:

- `plan` with pre-grasp / grasp / lift poses, or
- a `GraspFailureReason`.

Current failure reasons include:

- `object_pose_unknown`
- `size_unknown`
- `object_not_graspable`
- `object_too_large`
- `unsupported_hint`
- `no_feasible_grasp`

`PickSkill` maps planner rejection to the existing high-level
`FailureCode.NO_VALID_GRASP` and preserves the detailed planner reason in
`SkillResult.details`.

## Top-down feasibility

The current `TopDownGraspPlanner` remains deliberately simple and cube-focused.

It checks:

- object pose exists,
- object is marked graspable,
- object size exists,
- the requested grasp hint is top-down,
- the object fits the configured parallel-jaw gripper width.

The Franka runtime wires the planner with a 75 mm usable maximum width, derived
from the existing backend grasp-width configuration.

Because arbitrary object orientation is not estimated yet, the planner uses the
larger horizontal AABB extent as the required jaw clearance. This is
conservative by design.

A 50 mm cube is feasible.

A 90 mm cube is rejected before the robot moves.

## Compatibility

`manipulation/grasp_planner.py` remains as a compatibility shim so older tests
and callers importing `TopDownGraspPlanner` from the old path continue to work.

## Lazy point-cloud hook

`ObjectGeometryProvider` is only a protocol in this phase:

```python
get_point_cloud(object_id) -> np.ndarray | None
```

The existing perception provider already exposes a compatible method, so a
future point-cloud/ML planner can receive that provider by dependency injection
without changing `PickSkill`.

## Run

Apply over the Phase A repo, then run:

```bash
python3 scripts/run_phase_b_suite.py
```

The suite validates:

1. structured planner failures,
2. oversized-object rejection,
3. planner replaceability,
4. `PickSkill` failure mapping,
5. rejection before robot motion in Isaac,
6. normal perception-driven 5 cm cube Pick + Place regression.

Results are packaged as:

```text
outputs/phase_b_suite_<timestamp>.tar.gz
```
