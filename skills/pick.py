from __future__ import annotations

import numpy as np

from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus
from manipulation.grasping.planner import GraspPlanner
from world_model.world_model import WorldModel


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

    def _motion(self, pose, speed, tolerance, message):
        result = self.backend.move_to_pose(
            pose,
            speed=speed,
            position_tolerance=tolerance,
            orientation_tolerance=0.10,
        )
        if result.ok:
            return None

        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            message,
            FailureCode.TIMEOUT if result.timed_out else FailureCode.GRASP_FAILED,
            result.details,
        )

    def _movement(self, obj, planned_position):
        if obj.pose is None or not obj.visible:
            return None
        return float(
            np.linalg.norm(obj.pose.position - planned_position)
        )

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

    def execute(self, object_id, grasp_hint=None, lift_height=0.12):
        obj = self.world_model.get(object_id)
        if obj is None:
            return SkillResult(
                SkillStatus.FAILED,
                f"Unknown object '{object_id}'.",
                FailureCode.OBJECT_NOT_FOUND,
            )

        if self.world_model.held_object_id is not None:
            return SkillResult(
                SkillStatus.FAILED,
                "Gripper is already holding an object.",
                FailureCode.GRASP_FAILED,
            )

        if obj.pose is None or not obj.visible:
            return SkillResult(
                SkillStatus.FAILED,
                f"Pose of '{object_id}' is unavailable.",
                FailureCode.OBJECT_POSE_UNKNOWN,
            )

        initial = obj.pose.position.copy()
        local_replans = 0
        gripper_open = False
        last_grasp_details = {}

        while True:
            if obj.pose is None:
                return SkillResult(
                    SkillStatus.FAILED,
                    f"Pose of '{object_id}' became unavailable.",
                    FailureCode.OBJECT_POSE_UNKNOWN,
                    {"local_replans": local_replans},
                )

            planned_position = obj.pose.position.copy()
            grasp_result = self.grasp_planner.plan(
                obj,
                lift_height=lift_height,
                grasp_hint=grasp_hint,
            )

            if not grasp_result.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    grasp_result.message or "No valid grasp.",
                    FailureCode.NO_VALID_GRASP,
                    {
                        "grasp_failure_reason": (
                            None
                            if grasp_result.failure_reason is None
                            else grasp_result.failure_reason.value
                        ),
                        "grasp_planner_details": grasp_result.details,
                        "local_replans": local_replans,
                    },
                )

            plan = grasp_result.plan
            last_grasp_details = grasp_result.details

            for name, pose in (
                ("pre_grasp", plan.pre_grasp),
                ("grasp", plan.grasp),
                ("lift", plan.lift),
            ):
                if not self.backend.check_reachability(pose):
                    return SkillResult(
                        SkillStatus.FAILED,
                        f"{name} pose is unreachable.",
                        FailureCode.UNREACHABLE,
                        {
                            "local_replans": local_replans,
                            "grasp_planner_details": last_grasp_details,
                        },
                    )

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
                failure.details["local_replans"] = local_replans
                failure.details["grasp_planner_details"] = last_grasp_details
                return failure

            displacement = self._movement(obj, planned_position)
            if (
                displacement is not None
                and displacement > self.movement_threshold
            ):
                if local_replans >= self.max_local_replans:
                    return self._moved_result(
                        object_id,
                        planned_position,
                        obj.pose.position.copy(),
                        displacement,
                        local_replans,
                    )
                local_replans += 1
                continue

            failure = self._motion(
                plan.grasp,
                0.15,
                0.010,
                "Grasp approach failed.",
            )
            if failure:
                failure.details["local_replans"] = local_replans
                failure.details["grasp_planner_details"] = last_grasp_details
                return failure

            displacement = self._movement(obj, planned_position)
            if (
                displacement is not None
                and displacement > self.movement_threshold
            ):
                if local_replans >= self.max_local_replans:
                    return self._moved_result(
                        object_id,
                        planned_position,
                        obj.pose.position.copy(),
                        displacement,
                        local_replans,
                    )
                local_replans += 1
                continue

            close = self.backend.close_gripper()
            if not close.ok:
                return SkillResult(
                    SkillStatus.FAILED,
                    "Could not close gripper.",
                    FailureCode.GRASP_FAILED,
                    {
                        **close.details,
                        "local_replans": local_replans,
                    },
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
                failure.details["local_replans"] = local_replans
                failure.details["grasp_planner_details"] = last_grasp_details
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

            visual_lift = None
            final_position = None
            if obj.pose is not None:
                final_position = obj.pose.position.copy()
                if obj.visible:
                    visual_lift = float(
                        final_position[2] - initial[2]
                    )

            self.world_model.set_held(object_id)

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
                    "grasp_planner_details": last_grasp_details,
                    "grasp_at_contact": grasp_at_contact.details,
                    "grasp_after_lift": grasp_after_lift.details,
                },
            )
