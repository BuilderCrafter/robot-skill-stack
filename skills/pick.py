from __future__ import annotations

from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus
from manipulation.grasp_planner import TopDownGraspPlanner
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
        grasp_lift_threshold=0.03,
    ):
        self.backend = backend
        self.world_model = world_model
        self.grasp_planner = grasp_planner
        self.grasp_lift_threshold = grasp_lift_threshold

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
        plan = self.grasp_planner.plan(
            obj,
            lift_height=lift_height,
            grasp_hint=grasp_hint,
        )

        if plan is None:
            return SkillResult(
                SkillStatus.FAILED,
                "No valid grasp.",
                FailureCode.NO_VALID_GRASP,
            )

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
                )

        result = self.backend.open_gripper()
        if not result.ok:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not open gripper.",
                FailureCode.GRASP_FAILED,
            )

        failure = self._motion(plan.pre_grasp, 0.4, 0.020, "Pre-grasp failed.")
        if failure:
            return failure

        failure = self._motion(plan.grasp, 0.15, 0.010, "Grasp approach failed.")
        if failure:
            return failure

        if not self.backend.close_gripper().ok:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not close gripper.",
                FailureCode.GRASP_FAILED,
            )

        failure = self._motion(plan.lift, 0.2, 0.020, "Lift failed.")
        if failure:
            return failure

        if obj.pose is None:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not verify object state.",
                FailureCode.GRASP_FAILED,
            )

        final = obj.pose.position.copy()
        lift = float(final[2] - initial[2])

        if lift < self.grasp_lift_threshold:
            return SkillResult(
                SkillStatus.FAILED,
                "Object was not lifted.",
                FailureCode.GRASP_FAILED,
                {"lift_distance_m": lift},
            )

        self.world_model.set_held(object_id)

        return SkillResult(
            SkillStatus.SUCCESS,
            f"Picked '{object_id}'.",
            details={
                "initial_position": initial.tolist(),
                "final_position": final.tolist(),
                "lift_distance_m": lift,
            },
        )