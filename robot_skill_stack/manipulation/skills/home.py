from robot_skill_stack.manipulation.backend import ManipulationBackend
from robot_skill_stack.manipulation.held_state import HeldState, motion_failure_code
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.runtime.skill import (
    BaseSkill,
    FailureCode,
    SkillResult,
    SkillSpec,
    SkillStatus,
)


class HomeSkill(BaseSkill):
    SPEC = SkillSpec(
        "home",
        "Return to the backend's named startup pose; path safety is not guaranteed.",
        ("optional named pose",),
        (),
        ("robot reaches the named pose",),
        (FailureCode.PATH_BLOCKED, FailureCode.TIMEOUT, FailureCode.GRASP_LOST, FailureCode.GRASP_UNCERTAIN),
    )

    def __init__(self, backend: ManipulationBackend, world_model: WorldModel):
        self.backend = backend
        self.holding = HeldState(backend, world_model)

    @staticmethod
    def _result(result):
        if result.ok:
            return SkillResult(
                SkillStatus.SUCCESS,
                result.message,
                details=result.details,
            )
        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            result.message,
            motion_failure_code(result, FailureCode.PATH_BLOCKED),
            result.details,
        )

    async def execute_async(self, name="home"):
        return self._result(await self.holding.run_motion(self.backend.home_async, name))
