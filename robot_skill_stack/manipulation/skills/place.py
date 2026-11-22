from __future__ import annotations

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.held_state import HeldState, motion_failure_code
from robot_skill_stack.manipulation.placement import (
    PlacementFailureReason,
    PlacementPlanner,
    SimplePlacementPlanner,
)
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillStatus,
)
from robot_skill_stack.world.model.world_model import WorldModel


class PlaceSkill(BaseSkill):
    name = "place"

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
        self.holding = HeldState(backend, world_model)
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
            result.message if result.details.get('grasp_failure') else message,
            motion_failure_code(result, FailureCode.PLACE_FAILED),
            result.details,
        )

    def _prepare(self, target, mode):
        if self.world_model.held_object_uncertain:
            return None, None, SkillResult(
                SkillStatus.FAILED, 'Grasp is uncertain; inspect/reset the gripper first.',
                FailureCode.GRASP_UNCERTAIN,
            )
        if not self.world_model.observations_healthy:
            return None, None, SkillResult(
                SkillStatus.FAILED, "Observation source is not fresh; Place was not started.",
                FailureCode.WORLD_STATE_UNAVAILABLE,
            )
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
        details = {
            "object_id": object_id,
            "release_confirmed": True,
            "placement_verified": None,  # Release mode deliberately skips landing verification.
            "retreat_ok": retreat_result.ok,
            "retreat_message": retreat_result.message,
            "retreat_details": retreat_result.details,
            "placement_planner_details": placement.details,
        }
        message = f"Released '{object_id}'."
        verification_message = None
        if mode == "release":
            details["release_position"] = target.position.tolist()
        else:
            details.update(target_position=target.position.tolist(), visible=obj.visible)
            if obj.pose is None or not obj.visible:
                details["placement_verified"] = False
                verification_message = "Could not reacquire object after placement."
            else:
                error = float(np.linalg.norm(obj.pose.position - target.position))
                details.update(final_position=obj.pose.position.tolist(), placement_error_m=error,
                               placement_verified=error <= self.placement_tolerance)
                if not details["placement_verified"]:
                    verification_message = f"Placement error {error:.3f} m."
            message = f"Placed '{object_id}'."
        if verification_message:
            details["verification_message"] = verification_message

        # Release is already confirmed: a retreat failure must not restore attachment.
        # Give the motion failure precedence, retaining verification evidence above.
        if not retreat_result.ok:
            details["failed_stage"] = "retreat"
            return SkillResult(
                SkillStatus.TIMEOUT if retreat_result.timed_out else SkillStatus.FAILED,
                "Object released, but retreat failed: " + (retreat_result.message or "motion did not complete."),
                motion_failure_code(retreat_result, FailureCode.PLACE_FAILED),
                details,
            )
        if verification_message:
            details["failed_stage"] = "verification"
            return SkillResult(SkillStatus.FAILED, verification_message, FailureCode.PLACE_FAILED, details)
        return SkillResult(SkillStatus.SUCCESS, message, details=details)

    async def execute_async(self, target: Pose, mode="stable"):
        if getattr(self.backend, 'supports_contact_motion', True) is False:
            return SkillResult(SkillStatus.FAILED,
                'Selected motion strategy does not support intended-contact Place.',
                FailureCode.MOTION_UNSUPPORTED)
        prepared, target, failure = self._prepare(target, mode)
        if failure:
            return failure
        object_id, obj, placement, mode = prepared

        poses = self._poses(target)
        failure = self._check_reachability(poses, placement)
        if failure:
            return failure
        pre, release, retreat = poses

        result = await self.holding.run_motion(
            self.backend.move_to_pose_async,
            pre,
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

        result = await self.holding.run_motion(
            self.backend.move_to_pose_async,
            release,
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

        result = await self.holding.release()
        if not result.ok:
            self.world_model.clear_association_hint(object_id)
            return SkillResult(
                SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
                result.message or "Could not release object.",
                (FailureCode.GRASP_UNCERTAIN if self.world_model.held_object_uncertain
                 else motion_failure_code(result, FailureCode.PLACE_FAILED)),
                {
                    **result.details,
                    "object_id": object_id,
                    "failed_stage": "release",
                    "release_confirmed": False,
                    "placement_verified": None,
                    "retreat_ok": None,
                    "gripper_state": self.world_model.gripper_state.value,
                    "placement_planner_details": placement.details,
                },
            )

        retreat_result = await self.holding.run_motion(
            self.backend.move_to_pose_async,
            retreat,
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
