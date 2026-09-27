from __future__ import annotations

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix

from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.grasping.planner import GraspPlanner
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)
from robot_skill_stack.world.model.world_model import WorldModel


class PickSkill(BaseSkill):
    SPEC = SkillSpec(
        name="pick",
        description="Grasp an object and lift it.",
        inputs=("object_id", "grasp_hint", "lift_height"),
        preconditions=("object exists", "pose known", "gripper free"),
        effects=("object is held", "object is lifted"),
        failures=(
            FailureCode.OBJECT_NOT_FOUND,
            FailureCode.OBJECT_POSE_UNKNOWN,
            FailureCode.OBJECT_MOVED,
            FailureCode.UNREACHABLE,
            FailureCode.NO_VALID_GRASP,
            FailureCode.GRASP_FAILED,
            FailureCode.TIMEOUT,
        ),
    )

    def __init__(
        self,
        backend: ManipulationBackend,
        world_model: WorldModel,
        grasp_planner: GraspPlanner,
        *,
        grasp_lift_threshold=0.03,
        movement_threshold=0.02,
        max_local_replans=2,
    ):
        self.backend = backend
        self.world_model = world_model
        self.grasp_planner = grasp_planner
        self.grasp_lift_threshold = grasp_lift_threshold
        self.movement_threshold = float(movement_threshold)
        self.max_local_replans = int(max_local_replans)

    @staticmethod
    def _motion_result(result, message):
        if result.ok:
            return None
        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            message,
            FailureCode.TIMEOUT
            if result.timed_out
            else FailureCode.GRASP_FAILED,
            result.details,
        )

    def _motion(self, pose, speed, tolerance, message):
        return self._motion_result(
            self.backend.move_to_pose(
                pose,
                speed=speed,
                position_tolerance=tolerance,
                orientation_tolerance=0.10,
            ),
            message,
        )

    async def _motion_async(self, pose, speed, tolerance, message):
        return self._motion_result(
            await self.backend.move_to_pose_async(
                pose,
                speed=speed,
                position_tolerance=tolerance,
                orientation_tolerance=0.10,
            ),
            message,
        )

    @staticmethod
    def _movement(obj, planned_position):
        if obj.pose is None or not obj.visible:
            return None
        return float(np.linalg.norm(obj.pose.position - planned_position))

    def _moved_result(
        self,
        object_id,
        planned_position,
        current_position,
        displacement,
        local_replans,
    ):
        return SkillResult(
            SkillStatus.FAILED,
            f"'{object_id}' kept moving during pick.",
            FailureCode.OBJECT_MOVED,
            {
                "planned_object_position": planned_position.tolist(),
                "current_object_position": current_position.tolist(),
                "object_displacement_m": displacement,
                "movement_threshold_m": self.movement_threshold,
                "local_replans": local_replans,
                "max_local_replans": self.max_local_replans,
            },
        )

    def _start(self, object_id):
        obj = self.world_model.get(object_id)
        if obj is None:
            return None, SkillResult(
                SkillStatus.FAILED,
                f"Unknown object '{object_id}'.",
                FailureCode.OBJECT_NOT_FOUND,
            )
        if self.world_model.held_object_id is not None:
            return None, SkillResult(
                SkillStatus.FAILED,
                "Gripper is already holding an object.",
                FailureCode.GRASP_FAILED,
            )
        if obj.pose is None or not obj.visible:
            return None, SkillResult(
                SkillStatus.FAILED,
                f"Pose of '{object_id}' is unavailable.",
                FailureCode.OBJECT_POSE_UNKNOWN,
            )
        return obj, None

    def _plan(self, obj, lift_height, grasp_hint, local_replans):
        if obj.pose is None:
            return None, SkillResult(
                SkillStatus.FAILED,
                f"Pose of '{obj.object_id}' became unavailable.",
                FailureCode.OBJECT_POSE_UNKNOWN,
                {"local_replans": local_replans},
            )

        planned_position = obj.pose.position.copy()
        result = self.grasp_planner.plan(
            obj,
            lift_height=lift_height,
            grasp_hint=grasp_hint,
        )
        if not result.ok:
            return None, SkillResult(
                SkillStatus.FAILED,
                result.message or "No valid grasp.",
                FailureCode.NO_VALID_GRASP,
                {
                    "grasp_failure_reason": (
                        None
                        if result.failure_reason is None
                        else result.failure_reason.value
                    ),
                    "grasp_planner_details": result.details,
                    "local_replans": local_replans,
                },
            )

        for name, pose in (
            ("pre_grasp", result.plan.pre_grasp),
            ("grasp", result.plan.grasp),
            ("lift", result.plan.lift),
        ):
            if not self.backend.check_reachability(pose):
                return None, SkillResult(
                    SkillStatus.FAILED,
                    f"{name} pose is unreachable.",
                    FailureCode.UNREACHABLE,
                    {
                        "local_replans": local_replans,
                        "grasp_planner_details": result.details,
                    },
                )
        orientation = result.plan.grasp.orientation
        if orientation is None:
            orientation = self.backend.get_end_effector_pose().orientation
        result.details["object_offset_in_ee_m"] = (
            quat_matrix(orientation).T @ (planned_position-result.plan.grasp.position)
        ).tolist()
        return (planned_position, result), None

    def _record_attachment(self, grasp_result, planned_position):
        actual = self.backend.get_end_effector_pose()
        orientation = actual.orientation
        if orientation is None:
            orientation = grasp_result.plan.grasp.orientation
        grasp_result.details["object_offset_in_ee_m"] = (
            quat_matrix(orientation).T @ (planned_position-actual.position)
        ).tolist()
        grasp_result.details["attachment_reference"] = "pre-close measured TCP / planned object center"

    def _check_replan(
        self,
        obj,
        object_id,
        planned_position,
        local_replans,
    ):
        displacement = self._movement(obj, planned_position)
        if (
            displacement is None
            or displacement <= self.movement_threshold
        ):
            return False, local_replans, None

        if local_replans >= self.max_local_replans:
            return (
                False,
                local_replans,
                self._moved_result(
                    object_id,
                    planned_position,
                    obj.pose.position.copy(),
                    displacement,
                    local_replans,
                ),
            )
        return True, local_replans + 1, None

    def _success(
        self,
        object_id,
        obj,
        initial,
        local_replans,
        grasp_details,
        grasp_at_contact,
        grasp_after_lift,
    ):
        visual_lift = None
        final_position = None
        if obj.pose is not None:
            final_position = obj.pose.position.copy()
            if obj.visible:
                visual_lift = float(final_position[2] - initial[2])

        self.world_model.set_held(
            object_id, object_offset_in_ee=grasp_details.get("object_offset_in_ee_m"),
        )
        return SkillResult(
            SkillStatus.SUCCESS,
            f"Picked '{object_id}'.",
            details={
                "initial_position": initial.tolist(),
                "final_cached_position": (
                    None
                    if final_position is None
                    else final_position.tolist()
                ),
                "visual_lift_distance_m": visual_lift,
                "object_visible_after_lift": obj.visible,
                "local_replans": local_replans,
                "movement_threshold_m": self.movement_threshold,
                "grasp_planner_details": grasp_details,
                "grasp_at_contact": grasp_at_contact.details,
                "grasp_after_lift": grasp_after_lift.details,
            },
        )

    def execute(self, object_id, grasp_hint=None, lift_height=0.12):
        obj, failure = self._start(object_id)
        if failure:
            return failure

        initial = obj.pose.position.copy()
        local_replans = 0
        gripper_open = False

        while True:
            planned, failure = self._plan(
                obj,
                lift_height,
                grasp_hint,
                local_replans,
            )
            if failure:
                return failure
            planned_position, grasp_result = planned
            plan = grasp_result.plan

            if not gripper_open:
                result = self.backend.open_gripper()
                if not result.ok:
                    return SkillResult(
                        SkillStatus.FAILED,
                        "Could not open gripper.",
                        FailureCode.GRASP_FAILED,
                        result.details,
                    )
                gripper_open = True

            failure = self._motion(
                plan.pre_grasp,
                0.4,
                0.020,
                "Pre-grasp failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            replan, local_replans, failure = self._check_replan(
                obj, object_id, planned_position, local_replans
            )
            if failure:
                return failure
            if replan:
                continue

            failure = self._motion(
                plan.grasp,
                0.15,
                0.010,
                "Grasp approach failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            replan, local_replans, failure = self._check_replan(
                obj, object_id, planned_position, local_replans
            )
            if failure:
                return failure
            if replan:
                continue

            self._record_attachment(grasp_result, planned_position)
            close = self.backend.close_gripper()
            if not close.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Could not close gripper.",
                    FailureCode.GRASP_FAILED,
                    {**close.details, "local_replans": local_replans},
                )

            grasp_at_contact = self.backend.verify_grasp()
            if not grasp_at_contact.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Gripper closed without detecting an object.",
                    FailureCode.GRASP_FAILED,
                    {
                        **grasp_at_contact.details,
                        "local_replans": local_replans,
                    },
                )

            failure = self._motion(
                plan.lift,
                0.2,
                0.020,
                "Lift failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            grasp_after_lift = self.backend.verify_grasp()
            if not grasp_after_lift.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Object was lost during lift.",
                    FailureCode.GRASP_FAILED,
                    {
                        **grasp_after_lift.details,
                        "local_replans": local_replans,
                    },
                )

            return self._success(
                object_id,
                obj,
                initial,
                local_replans,
                grasp_result.details,
                grasp_at_contact,
                grasp_after_lift,
            )

    async def execute_async(
        self,
        object_id,
        grasp_hint=None,
        lift_height=0.12,
    ):
        obj, failure = self._start(object_id)
        if failure:
            return failure

        initial = obj.pose.position.copy()
        local_replans = 0
        gripper_open = False

        while True:
            planned, failure = self._plan(
                obj,
                lift_height,
                grasp_hint,
                local_replans,
            )
            if failure:
                return failure
            planned_position, grasp_result = planned
            plan = grasp_result.plan

            if not gripper_open:
                result = await self.backend.open_gripper_async()
                if not result.ok:
                    return SkillResult(
                        SkillStatus.FAILED,
                        "Could not open gripper.",
                        FailureCode.GRASP_FAILED,
                        result.details,
                    )
                gripper_open = True

            failure = await self._motion_async(
                plan.pre_grasp,
                0.4,
                0.020,
                "Pre-grasp failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            replan, local_replans, failure = self._check_replan(
                obj, object_id, planned_position, local_replans
            )
            if failure:
                return failure
            if replan:
                continue

            failure = await self._motion_async(
                plan.grasp,
                0.15,
                0.010,
                "Grasp approach failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            replan, local_replans, failure = self._check_replan(
                obj, object_id, planned_position, local_replans
            )
            if failure:
                return failure
            if replan:
                continue

            self._record_attachment(grasp_result, planned_position)
            close = await self.backend.close_gripper_async()
            if not close.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Could not close gripper.",
                    FailureCode.GRASP_FAILED,
                    {**close.details, "local_replans": local_replans},
                )

            grasp_at_contact = self.backend.verify_grasp()
            if not grasp_at_contact.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Gripper closed without detecting an object.",
                    FailureCode.GRASP_FAILED,
                    {
                        **grasp_at_contact.details,
                        "local_replans": local_replans,
                    },
                )

            failure = await self._motion_async(
                plan.lift,
                0.2,
                0.020,
                "Lift failed.",
            )
            if failure:
                failure.details.update(
                    local_replans=local_replans,
                    grasp_planner_details=grasp_result.details,
                )
                return failure

            grasp_after_lift = self.backend.verify_grasp()
            if not grasp_after_lift.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Object was lost during lift.",
                    FailureCode.GRASP_FAILED,
                    {
                        **grasp_after_lift.details,
                        "local_replans": local_replans,
                    },
                )

            return self._success(
                object_id,
                obj,
                initial,
                local_replans,
                grasp_result.details,
                grasp_at_contact,
                grasp_after_lift,
            )
