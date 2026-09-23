from __future__ import annotations

from core.manipulation import ManipulationBackend

from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus

from manipulation.grasp_planner import TopDownGraspPlanner

from world_model.world_model import WorldModel


class PickSkill(BaseSkill):
    """
    Semantic object-picking capability.

    This class contains no Isaac-specific code.

    It composes generic manipulation primitives:

        open gripper
        move to pre-grasp
        move to grasp
        close gripper
        lift
        verify object actually moved
    """

    SPEC = SkillSpec(
        name="pick",
        description="Grasp an object and lift it from its supporting surface.",
        inputs=("object_id", "optional grasp_hint", "lift_height"),

        preconditions=("object exists", "object pose is known", "gripper is free", "valid grasp exists"),

        effects=("object is held by the gripper", "object is lifted"),

        failures=(
            FailureCode.OBJECT_NOT_FOUND,
            FailureCode.OBJECT_POSE_UNKNOWN,
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
        grasp_planner: TopDownGraspPlanner,
        *,
        grasp_lift_threshold: float = 0.03,
        approach_speed: float = 0.4,
        grasp_speed: float = 0.15,
        lift_speed: float = 0.2,
    ) -> None:

        self.backend = backend
        self.world_model = world_model
        self.grasp_planner = grasp_planner
        self.grasp_lift_threshold = grasp_lift_threshold
        self.approach_speed = approach_speed
        self.grasp_speed = grasp_speed
        self.lift_speed = lift_speed

    # Internal helper
    def _motion_failure(self, backend_result, message: str) -> SkillResult:

        if backend_result.timed_out:
            return SkillResult(
                status=SkillStatus.TIMEOUT,
                failure_code=FailureCode.TIMEOUT,
                message=message,
                details=backend_result.details,
            )


        return SkillResult(
            status=SkillStatus.FAILED,
            failure_code=FailureCode.GRASP_FAILED,
            message=message,
            details=backend_result.details,
        )


    # Execute
    def execute(self, object_id: str, grasp_hint: str | None = None, lift_height: float = 0.12) -> SkillResult:
        # Preconditions: object exists
        obj = self.world_model.get(object_id)

        if obj is None:

            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.OBJECT_NOT_FOUND,
                message=f"Unknown object: '{object_id}'.",
            )

        # Preconditions: gripper should be free
        if self.world_model.held_object_id is not None:
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.GRASP_FAILED,
                message=f"Pick requested while another object is already held: '{self.world_model.held_object_id}'.",
            )


        # Refresh actual object pose.
        refreshed = self.world_model.refresh_object(object_id)

        if not refreshed or obj.pose is None:
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.OBJECT_POSE_UNKNOWN,
                message=f"Pose of '{object_id}' is unknown.",
            )

        initial_position = obj.pose.position.copy()
        initial_z = float(initial_position[2])

        # Ask grasp planner for candidate.
        plan = self.grasp_planner.plan(obj, lift_height=lift_height, grasp_hint=grasp_hint)

        if plan is None:
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.NO_VALID_GRASP,
                message=f"No valid grasp was found for '{object_id}'.",
            )


        # Reachability pre-check.
        for name, pose in (("pre_grasp", plan.pre_grasp), ("grasp", plan.grasp), ("lift", plan.lift)):
            if not self.backend.check_reachability(pose):
                return SkillResult(
                    status=SkillStatus.FAILED,
                    failure_code=FailureCode.UNREACHABLE,
                    message=f"Required grasp pose '{name}' is unreachable.",
                    details={
                        "pose": pose.position.tolist(),
                        "object_id": object_id,
                    },
                )

        # 1. Open gripper
        result = self.backend.open_gripper()

        if not result.ok:
            return self._motion_failure(result, "Could not open gripper.")

        # 2. Move above object
        result = self.backend.move_to_pose(plan.pre_grasp, speed=self.approach_speed)

        if not result.ok:
            return self._motion_failure(result, "Could not reach pre-grasp pose.")

        # 3. Descend to grasp pose
        result = self.backend.move_to_pose(plan.grasp, speed=(self.grasp_speed))

        if not result.ok:
            return self._motion_failure(result, "Could not reach grasp pose.")


        # 4. Close gripper
        result = self.backend.close_gripper()

        if not result.ok:
            return self._motion_failure(result, "Could not close gripper.")

        # 5. Lift
        result = self.backend.move_to_pose(plan.lift, speed=self.lift_speed)

        if not result.ok:
            return self._motion_failure(result, "Could not execute lift motion.")

        # 6. Refresh real world state.
        if not self.world_model.refresh_object(object_id):
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.GRASP_FAILED,
                message="Lift motion completed, but object state could not be verified.",
            )

        final_position = obj.pose.position.copy()

        lift_distance = float(final_position[2] - initial_z)

        # 7. Verify object physically moved upward.
        if lift_distance < self.grasp_lift_threshold:
            return SkillResult(
                status=SkillStatus.FAILED,
                failure_code=FailureCode.GRASP_FAILED,
                message="Gripper motion completed, but the object was not lifted.",
                details={
                    "object_id": object_id,
                    "initial_position": initial_position.tolist(),
                    "final_position": final_position.tolist(),
                    "lift_distance_m": lift_distance,
                    "required_lift_m": self.grasp_lift_threshold,
                },
            )

        # Semantic world-model update.
        self.world_model.set_held(object_id)


        # Success
        return SkillResult(
            status=SkillStatus.SUCCESS,
            message=f"Picked '{object_id}'.",
            details={
                "object_id": object_id,
                "initial_position": initial_position.tolist(),
                "final_position": final_position.tolist(),
                "lift_distance_m": lift_distance,
                "grasp_hint": grasp_hint,
                "lift_height_requested_m": lift_height,
            },
        )