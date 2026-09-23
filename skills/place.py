from __future__ import annotations

import numpy as np

from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus
from core.types import Pose
from world_model.world_model import WorldModel


class PlaceSkill(BaseSkill):
    SPEC = SkillSpec(
        name="place",
        description="Place the currently held object at a requested pose.",
        inputs=("target", "mode"),
        preconditions=("robot is holding an object", "target is reachable"),
        effects=("object is released", "object is stable near target"),
        failures=(
            FailureCode.NOT_HOLDING_OBJECT,
            FailureCode.UNREACHABLE,
            FailureCode.PATH_BLOCKED,
            FailureCode.PLACE_FAILED,
            FailureCode.TIMEOUT,
        ),
    )

    def __init__(
        self,
        backend: ManipulationBackend,
        world_model: WorldModel,
        *,
        approach_height: float = 0.10,
        retreat_height: float = 0.12,
        placement_tolerance: float = 0.05,
        approach_speed: float = 0.4,
        place_speed: float = 0.15,
        retreat_speed: float = 0.3,
    ) -> None:
        self.backend = backend
        self.world_model = world_model
        self.approach_height = approach_height
        self.retreat_height = retreat_height
        self.placement_tolerance = placement_tolerance
        self.approach_speed = approach_speed
        self.place_speed = place_speed
        self.retreat_speed = retreat_speed

    def _motion_failure(self, result, message: str) -> SkillResult:
        if result.timed_out:
            return SkillResult(
                SkillStatus.TIMEOUT,
                message,
                FailureCode.TIMEOUT,
                result.details,
            )

        return SkillResult(
            SkillStatus.FAILED,
            message,
            FailureCode.PATH_BLOCKED,
            result.details,
        )

    def execute(
        self,
        target: Pose,
        mode: str = "default",
    ) -> SkillResult:
        object_id = self.world_model.held_object_id

        if object_id is None:
            return SkillResult(
                SkillStatus.FAILED,
                "Place requested while no object is held.",
                FailureCode.NOT_HOLDING_OBJECT,
            )

        obj = self.world_model.get(object_id)

        if obj is None:
            return SkillResult(
                SkillStatus.FAILED,
                f"Held object '{object_id}' is not present in the world model.",
                FailureCode.PLACE_FAILED,
            )

        # If the caller only specifies a position, preserve the current
        # end-effector orientation from the grasp.
        orientation = target.orientation
        if orientation is None:
            orientation = self.backend.get_end_effector_pose().orientation

        release_pose = Pose(
            position=target.position.copy(),
            orientation=orientation,
            frame=target.frame,
        )

        pre_place = release_pose.translated([0.0, 0.0, self.approach_height])
        retreat = release_pose.translated([0.0, 0.0, self.retreat_height])

        for name, pose in (
            ("pre_place", pre_place),
            ("release", release_pose),
            ("retreat", retreat),
        ):
            if not self.backend.check_reachability(pose):
                return SkillResult(
                    SkillStatus.FAILED,
                    f"Required placement pose '{name}' is unreachable.",
                    FailureCode.UNREACHABLE,
                    {"pose": pose.position.tolist()},
                )

        result = self.backend.move_to_pose(pre_place, speed=self.approach_speed)
        if not result.ok:
            return self._motion_failure(result, "Could not reach pre-place pose.")

        result = self.backend.move_to_pose(release_pose, speed=self.place_speed)
        if not result.ok:
            return self._motion_failure(result, "Could not reach release pose.")

        result = self.backend.open_gripper()
        if not result.ok:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not release object.",
                FailureCode.PLACE_FAILED,
                result.details,
            )

        # Once the gripper opens, the semantic state is no longer "held".
        self.world_model.set_held(None)

        result = self.backend.move_to_pose(retreat, speed=self.retreat_speed)
        if not result.ok:
            return self._motion_failure(result, "Object released, but retreat failed.")

        if not self.world_model.refresh_object(object_id) or obj.pose is None:
            return SkillResult(
                SkillStatus.FAILED,
                "Object was released, but its final pose could not be verified.",
                FailureCode.PLACE_FAILED,
            )

        error = float(np.linalg.norm(obj.pose.position - target.position))

        if error > self.placement_tolerance:
            return SkillResult(
                SkillStatus.FAILED,
                f"Placement error {error:.3f} m exceeded tolerance.",
                FailureCode.PLACE_FAILED,
                {
                    "object_id": object_id,
                    "target_position": target.position.tolist(),
                    "final_position": obj.pose.position.tolist(),
                    "placement_error_m": error,
                },
            )

        return SkillResult(
            SkillStatus.SUCCESS,
            f"Placed '{object_id}'.",
            details={
                "object_id": object_id,
                "target_position": target.position.tolist(),
                "final_position": obj.pose.position.tolist(),
                "placement_error_m": error,
                "mode": mode,
            },
        )