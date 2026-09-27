from __future__ import annotations

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.placement import (
    PlacementFailureReason,
    PlacementPlanner,
    SimplePlacementPlanner,
)
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)
from robot_skill_stack.world.model.world_model import WorldModel


class PlaceSkill(BaseSkill):
    SPEC = SkillSpec(
        name="place",
        description="Place or release the currently held object.",
        inputs=("target", "mode"),
        preconditions=(
            "an object is held",
            "target is unoccupied",
            "target is reachable",
        ),
        effects=("object is released",),
        failures=(
            FailureCode.NOT_HOLDING_OBJECT,
            FailureCode.TARGET_OCCUPIED,
            FailureCode.UNREACHABLE,
            FailureCode.PLACE_FAILED,
            FailureCode.TIMEOUT,
        ),
    )

    def __init__(
        self,
        backend: ManipulationBackend,
        world_model: WorldModel,
        placement_planner: PlacementPlanner | None = None,
        *,
        approach_height=0.10,
        retreat_height=0.12,
        placement_tolerance=0.05,
        reacquire_radius=0.08,
        reacquire_ttl=3.0,
    ):
        self.backend = backend
        self.world_model = world_model
        self.placement_planner = (
            placement_planner or SimplePlacementPlanner()
        )
        self.approach_height = approach_height
        self.retreat_height = retreat_height
        self.placement_tolerance = placement_tolerance
        self.reacquire_radius = reacquire_radius
        self.reacquire_ttl = reacquire_ttl

    @staticmethod
    def _motion_result(result, message):
        if result.ok:
            return None
        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            message,
            FailureCode.TIMEOUT
            if result.timed_out
            else FailureCode.PLACE_FAILED,
            result.details,
        )

    def _prepare(self, target, mode):
        if mode == "default":
            mode = "stable"
        if mode not in ("stable", "release"):
            return None, None, SkillResult(
                SkillStatus.FAILED,
                f"Unknown placement mode '{mode}'.",
                FailureCode.PLACE_FAILED,
            )

        object_id = self.world_model.held_object_id
        if object_id is None:
            return None, None, SkillResult(
                SkillStatus.FAILED,
                "No object is currently held.",
                FailureCode.NOT_HOLDING_OBJECT,
            )

        obj = self.world_model.require(object_id)
        placement = self.placement_planner.evaluate(
            obj,
            target,
            self.world_model,
        )
        if not placement.ok:
            code = (
                FailureCode.TARGET_OCCUPIED
                if placement.failure_reason
                == PlacementFailureReason.TARGET_OCCUPIED
                else FailureCode.PLACE_FAILED
            )
            return None, None, SkillResult(
                SkillStatus.FAILED,
                placement.message
                or "Placement target is not feasible.",
                code,
                {
                    "placement_failure_reason": (
                        placement.failure_reason.value
                    ),
                    "placement_planner_details": placement.details,
                },
            )

        return (object_id, obj, placement, mode), target, None

    def _poses(self, target):
        orientation = (
            target.orientation
            if target.orientation is not None
            else self.backend.get_end_effector_pose().orientation
        )
        position = target.position.copy()
        offset = self.world_model.held_object_offset_in_ee
        if offset is not None:
            position -= quat_matrix(orientation) @ offset
        release = Pose(
            position,
            orientation,
            target.frame,
        )
        return (
            release.translated([0, 0, self.approach_height]),
            release,
            release.translated([0, 0, self.retreat_height]),
        )

    def _check_reachability(self, poses, placement):
        for pose in poses:
            if not self.backend.check_reachability(pose):
                return SkillResult(
                    SkillStatus.FAILED,
                    "Required placement pose is unreachable.",
                    FailureCode.UNREACHABLE,
                    {
                        "placement_planner_details": (
                            placement.details
                        )
                    },
                )
        return None

    def _finish(
        self,
        object_id,
        obj,
        target,
        mode,
        placement,
        retreat_result,
    ):
        if mode == "release":
            return SkillResult(
                SkillStatus.SUCCESS,
                f"Released '{object_id}'.",
                details={
                    "object_id": object_id,
                    "release_position": target.position.tolist(),
                    "retreat_ok": retreat_result.ok,
                    "placement_planner_details": placement.details,
                },
            )

        if obj.pose is None or not obj.visible:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not reacquire object after placement.",
                FailureCode.PLACE_FAILED,
                {
                    "object_id": object_id,
                    "target_position": target.position.tolist(),
                    "visible": obj.visible,
                    "retreat_ok": retreat_result.ok,
                    "placement_planner_details": placement.details,
                },
            )

        error = float(np.linalg.norm(obj.pose.position - target.position))
        if error > self.placement_tolerance:
            return SkillResult(
                SkillStatus.FAILED,
                f"Placement error {error:.3f} m.",
                FailureCode.PLACE_FAILED,
                {
                    "target_position": target.position.tolist(),
                    "final_position": obj.pose.position.tolist(),
                    "placement_error_m": error,
                    "retreat_ok": retreat_result.ok,
                    "placement_planner_details": placement.details,
                },
            )

        return SkillResult(
            SkillStatus.SUCCESS,
            f"Placed '{object_id}'.",
            details={
                "target_position": target.position.tolist(),
                "final_position": obj.pose.position.tolist(),
                "placement_error_m": error,
                "retreat_ok": retreat_result.ok,
                "placement_planner_details": placement.details,
            },
        )

    def execute(self, target: Pose, mode="stable"):
        prepared, target, failure = self._prepare(target, mode)
        if failure:
            return failure
        object_id, obj, placement, mode = prepared

        poses = self._poses(target)
        failure = self._check_reachability(poses, placement)
        if failure:
            return failure
        pre, release, retreat = poses

        result = self.backend.move_to_pose(
            pre,
            speed=0.4,
            position_tolerance=0.020,
            orientation_tolerance=0.10,
        )
        failure = self._motion_result(
            result,
            "Pre-place motion failed.",
        )
        if failure:
            failure.details["placement_planner_details"] = (
                placement.details
            )
            return failure

        result = self.backend.move_to_pose(
            release,
            speed=0.15,
            position_tolerance=0.015,
            orientation_tolerance=0.10,
        )
        failure = self._motion_result(
            result,
            "Release approach failed.",
        )
        if failure:
            failure.details["placement_planner_details"] = (
                placement.details
            )
            return failure

        self.world_model.expect_object_at(
            object_id,
            target.position,
            radius=self.reacquire_radius,
            ttl=self.reacquire_ttl,
            reason="place",
        )

        result = self.backend.open_gripper()
        if not result.ok:
            self.world_model.clear_association_hint(object_id)
            return SkillResult(
                SkillStatus.FAILED,
                "Could not release object.",
                FailureCode.PLACE_FAILED,
                {
                    **result.details,
                    "placement_planner_details": placement.details,
                },
            )

        self.world_model.set_held(None)
        retreat_result = self.backend.move_to_pose(
            retreat,
            speed=0.3,
            position_tolerance=0.020,
            orientation_tolerance=0.10,
        )
        return self._finish(
            object_id,
            obj,
            target,
            mode,
            placement,
            retreat_result,
        )

    async def execute_async(self, target: Pose, mode="stable"):
        prepared, target, failure = self._prepare(target, mode)
        if failure:
            return failure
        object_id, obj, placement, mode = prepared

        poses = self._poses(target)
        failure = self._check_reachability(poses, placement)
        if failure:
            return failure
        pre, release, retreat = poses

        result = await self.backend.move_to_pose_async(
            pre,
            speed=0.4,
            position_tolerance=0.020,
            orientation_tolerance=0.10,
        )
        failure = self._motion_result(
            result,
            "Pre-place motion failed.",
        )
        if failure:
            failure.details["placement_planner_details"] = (
                placement.details
            )
            return failure

        result = await self.backend.move_to_pose_async(
            release,
            speed=0.15,
            position_tolerance=0.015,
            orientation_tolerance=0.10,
        )
        failure = self._motion_result(
            result,
            "Release approach failed.",
        )
        if failure:
            failure.details["placement_planner_details"] = (
                placement.details
            )
            return failure

        self.world_model.expect_object_at(
            object_id,
            target.position,
            radius=self.reacquire_radius,
            ttl=self.reacquire_ttl,
            reason="place",
        )

        result = await self.backend.open_gripper_async()
        if not result.ok:
            self.world_model.clear_association_hint(object_id)
            return SkillResult(
                SkillStatus.FAILED,
                "Could not release object.",
                FailureCode.PLACE_FAILED,
                {
                    **result.details,
                    "placement_planner_details": placement.details,
                },
            )

        self.world_model.set_held(None)
        retreat_result = await self.backend.move_to_pose_async(
            retreat,
            speed=0.3,
            position_tolerance=0.020,
            orientation_tolerance=0.10,
        )
        return self._finish(
            object_id,
            obj,
            target,
            mode,
            placement,
            retreat_result,
        )
