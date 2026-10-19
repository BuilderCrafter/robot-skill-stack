from __future__ import annotations

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.held_state import HeldState, motion_failure_code
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillStatus,
)


class MoveToPoseSkill(BaseSkill):
    name = "move_to_pose"

    def __init__(
        self,
        backend: ManipulationBackend,
        world_model: WorldModel,
        *,
        position_tolerance=0.025,
        orientation_tolerance=0.10,
    ):
        self.backend = backend
        self.holding = HeldState(backend, world_model)
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
            motion_failure_code(result, FailureCode.PATH_BLOCKED),
            result.details,
        )

    async def execute_async(
        self,
        target: Pose,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        if not self.backend.check_reachability(target):
            return SkillResult(
                SkillStatus.FAILED,
                "Requested pose is unreachable.",
                FailureCode.UNREACHABLE,
            )
        result = await self.holding.run_motion(
            self.backend.move_to_pose_async,
            target,
            position_tolerance=(
                position_tolerance or self.position_tolerance
            ),
            orientation_tolerance=(
                orientation_tolerance or self.orientation_tolerance
            ),
        )
        return self._result(result)
