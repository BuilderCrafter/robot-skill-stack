from __future__ import annotations

import numpy as np

from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus
from core.types import Pose
from world_model.world_model import WorldModel


class PlaceSkill(BaseSkill):
    SPEC = SkillSpec(
        name="place",
        description="Place or release the currently held object.",
        inputs=("target", "mode"),
        preconditions=("an object is held", "target is reachable"),
        effects=("object is released",),
        failures=(
            FailureCode.NOT_HOLDING_OBJECT,
            FailureCode.UNREACHABLE,
            FailureCode.PLACE_FAILED,
            FailureCode.TIMEOUT,
        ),
    )

    def __init__(
        self,
        backend: ManipulationBackend,
        world_model: WorldModel,
        *,
        approach_height=0.10,
        retreat_height=0.12,
        placement_tolerance=0.05,
    ):
        self.backend = backend
        self.world_model = world_model
        self.approach_height = approach_height
        self.retreat_height = retreat_height
        self.placement_tolerance = placement_tolerance

    def _move(self, pose, speed, tolerance, message):
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
            FailureCode.TIMEOUT if result.timed_out else FailureCode.PLACE_FAILED,
            result.details,
        )

    def execute(self, target: Pose, mode="stable"):
        if mode == "default":
            mode = "stable"

        if mode not in ("stable", "release"):
            return SkillResult(
                SkillStatus.FAILED,
                f"Unknown placement mode '{mode}'.",
                FailureCode.PLACE_FAILED,
            )

        object_id = self.world_model.held_object_id
        if object_id is None:
            return SkillResult(
                SkillStatus.FAILED,
                "No object is currently held.",
                FailureCode.NOT_HOLDING_OBJECT,
            )

        obj = self.world_model.require(object_id)
        orientation = target.orientation or self.backend.get_end_effector_pose().orientation

        release = Pose(target.position.copy(), orientation, target.frame)
        pre = release.translated([0, 0, self.approach_height])
        retreat = release.translated([0, 0, self.retreat_height])

        for pose in (pre, release, retreat):
            if not self.backend.check_reachability(pose):
                return SkillResult(
                    SkillStatus.FAILED,
                    "Required placement pose is unreachable.",
                    FailureCode.UNREACHABLE,
                )

        failure = self._move(pre, 0.4, 0.020, "Pre-place motion failed.")
        if failure:
            return failure

        failure = self._move(release, 0.15, 0.015, "Release approach failed.")
        if failure:
            return failure

        result = self.backend.open_gripper()
        if not result.ok:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not release object.",
                FailureCode.PLACE_FAILED,
            )

        self.world_model.set_held(None)

        retreat_result = self.backend.move_to_pose(
            retreat,
            speed=0.3,
            position_tolerance=0.020,
            orientation_tolerance=0.10,
        )

        if mode == "release":
            return SkillResult(
                SkillStatus.SUCCESS,
                f"Released '{object_id}'.",
                details={
                    "object_id": object_id,
                    "release_position": target.position.tolist(),
                    "retreat_ok": retreat_result.ok,
                },
            )

        if not self.world_model.refresh_object(object_id) or obj.pose is None:
            return SkillResult(
                SkillStatus.FAILED,
                "Could not verify final object position.",
                FailureCode.PLACE_FAILED,
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
            },
        )