from __future__ import annotations

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)


class MoveToPoseSkill(BaseSkill):
    SPEC = SkillSpec(
        "move_to_pose",
        "Move the robot end effector to a requested pose.",
        ("target", "speed"),
        ("target pose is valid", "target is reachable"),
        ("end effector reaches target pose",),
        (
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

    def _result(self, result):
        if result.ok:
            return SkillResult(
                SkillStatus.SUCCESS,
                "End effector reached target pose.",
                details=result.details,
            )
        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            result.message,
            FailureCode.TIMEOUT
            if result.timed_out
            else FailureCode.PATH_BLOCKED,
            result.details,
        )

    def execute(
        self,
        target: Pose,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        if not self.backend.check_reachability(target):
            return SkillResult(
                SkillStatus.FAILED,
                "Requested pose is unreachable.",
                FailureCode.UNREACHABLE,
            )
        result = self.backend.move_to_pose(
            target,
            speed=speed,
            position_tolerance=(
                position_tolerance or self.position_tolerance
            ),
            orientation_tolerance=(
                orientation_tolerance or self.orientation_tolerance
            ),
        )
        return self._result(result)

    async def execute_async(
        self,
        target: Pose,
        speed=0.5,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        if not self.backend.check_reachability(target):
            return SkillResult(
                SkillStatus.FAILED,
                "Requested pose is unreachable.",
                FailureCode.UNREACHABLE,
            )
        result = await self.backend.move_to_pose_async(
            target,
            speed=speed,
            position_tolerance=(
                position_tolerance or self.position_tolerance
            ),
            orientation_tolerance=(
                orientation_tolerance or self.orientation_tolerance
            ),
        )
        return self._result(result)
