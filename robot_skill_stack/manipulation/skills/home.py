from robot_skill_stack.manipulation.backend import ManipulationBackend
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
        "Return the robot to a known safe pose.",
        ("optional named pose",),
        (),
        ("robot reaches a safe pose",),
        (FailureCode.PATH_BLOCKED, FailureCode.TIMEOUT),
    )

    def __init__(self, backend: ManipulationBackend):
        self.backend = backend

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
            FailureCode.TIMEOUT
            if result.timed_out
            else FailureCode.PATH_BLOCKED,
            result.details,
        )

    def execute(self, name="home"):
        return self._result(self.backend.home(name))

    async def execute_async(self, name="home"):
        return self._result(await self.backend.home_async(name))
