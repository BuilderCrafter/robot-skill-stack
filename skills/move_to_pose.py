from __future__ import annotations

from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus
from core.types import Pose


class MoveToPoseSkill(BaseSkill):
    SPEC = SkillSpec(
        name="move_to_pose",
        description="Move the robot end effector to a requested pose.",
        inputs=("target", "speed"),
        preconditions=("target pose is valid", "target is reachable"),
        effects=("end effector reaches target pose",),
        failures=(
            FailureCode.UNREACHABLE,
            FailureCode.PATH_BLOCKED,
            FailureCode.TIMEOUT,
        ),
    )

    def __init__(
        self,
        backend: ManipulationBackend,
        *,
        position_tolerance=0.025,
        orientation_tolerance=0.10,
    ):
        self.backend = backend
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

    def execute(
        self,
        target: Pose,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ) -> SkillResult:
        if not self.backend.check_reachability(target):
            return SkillResult(
                SkillStatus.FAILED,
                "Requested pose is unreachable.",
                FailureCode.UNREACHABLE,
            )

        result = self.backend.move_to_pose(
            target,
            speed=speed,
            position_tolerance=position_tolerance or self.position_tolerance,
            orientation_tolerance=(
                orientation_tolerance or self.orientation_tolerance
            ),
        )

        if result.ok:
            return SkillResult(
                SkillStatus.SUCCESS,
                "End effector reached target pose.",
                details=result.details,
            )

        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            result.message,
            FailureCode.TIMEOUT if result.timed_out else FailureCode.PATH_BLOCKED,
            result.details,
        )