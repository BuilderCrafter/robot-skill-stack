from __future__ import annotations

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.grasping.types import (
    GraspFailureReason,
    GraspPlan,
    GraspResult,
    ParallelJawGripperSpec,
)
from robot_skill_stack.world.model.entities import WorldObject


class TopDownGraspPlanner:
    """
    Conservative cube-focused top-down planner.

    It owns its own feasibility checks. For V1/V2, an object must:
    - have a known pose,
    - be marked graspable,
    - have known size,
    - support the requested top-down strategy,
    - fit inside the configured parallel-jaw gripper width.

    Because arbitrary object orientation is not yet estimated, width feasibility
    uses the larger horizontal AABB extent. This is intentionally conservative.
    """

    def __init__(
        self,
        *,
        approach_height: float = 0.10,
        default_lift_height: float = 0.12,
        grasp_z_offset: float = 0.0,
        grasp_orientation=None,
        gripper: ParallelJawGripperSpec | None = None,
    ):
        self.approach_height = float(approach_height)
        self.default_lift_height = float(default_lift_height)
        self.grasp_z_offset = float(grasp_z_offset)
        self.gripper = gripper or ParallelJawGripperSpec(
            max_width=0.075,
        )

        if grasp_orientation is None:
            grasp_orientation = np.asarray(
                [0.0, 0.0, 1.0, 0.0],
                dtype=float,
            )

        self.grasp_orientation = np.asarray(
            grasp_orientation,
            dtype=float,
        )

    def _feasibility(self, obj: WorldObject, grasp_hint: str | None):
        if obj.pose is None:
            return GraspResult.failure(
                GraspFailureReason.OBJECT_POSE_UNKNOWN,
                "Object pose is unknown.",
            )

        if not obj.graspable:
            return GraspResult.failure(
                GraspFailureReason.OBJECT_NOT_GRASPABLE,
                "Object is marked non-graspable.",
            )

        if grasp_hint not in (None, "top", "top_down"):
            return GraspResult.failure(
                GraspFailureReason.UNSUPPORTED_HINT,
                f"Unsupported grasp hint '{grasp_hint}'.",
                grasp_hint=grasp_hint,
            )

        if obj.size is None:
            return GraspResult.failure(
                GraspFailureReason.SIZE_UNKNOWN,
                "Object size is unknown, so top-down fit cannot be verified.",
            )

        size = np.asarray(obj.size, dtype=float)
        required_width = float(np.max(size[:2]))

        if required_width > self.gripper.max_width:
            return GraspResult.failure(
                GraspFailureReason.OBJECT_TOO_LARGE,
                (
                    f"Object requires {required_width:.3f} m horizontal "
                    f"clearance, exceeding gripper max width "
                    f"{self.gripper.max_width:.3f} m."
                ),
                object_size_m=size.tolist(),
                required_width_m=required_width,
                gripper_max_width_m=self.gripper.max_width,
            )

        return GraspResult(
            details={
                "object_size_m": size.tolist(),
                "required_width_m": required_width,
                "gripper_max_width_m": self.gripper.max_width,
                "feasibility_checked": True,
                "strategy": "top_down",
            }
        )

    def plan(
        self,
        obj: WorldObject,
        *,
        lift_height: float | None = None,
        grasp_hint: str | None = None,
    ) -> GraspResult:
        feasibility = self._feasibility(obj, grasp_hint)
        if feasibility.failure_reason is not None:
            return feasibility

        if lift_height is None:
            lift_height = self.default_lift_height

        grasp_position = obj.pose.position.copy()
        grasp_position[2] += self.grasp_z_offset

        grasp = Pose(
            position=grasp_position,
            orientation=self.grasp_orientation.copy(),
            frame=obj.pose.frame,
        )
        pre_grasp = grasp.translated(
            [0.0, 0.0, self.approach_height]
        )
        lift = grasp.translated(
            [0.0, 0.0, float(lift_height)]
        )

        plan = GraspPlan(
            pre_grasp=pre_grasp,
            grasp=grasp,
            lift=lift,
        )

        return GraspResult.success(
            plan,
            **feasibility.details,
            approach_height_m=self.approach_height,
            lift_height_m=float(lift_height),
            grasp_z_offset_m=self.grasp_z_offset,
        )
