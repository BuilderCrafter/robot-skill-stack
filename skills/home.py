from core.manipulation import ManipulationBackend
from core.skill import BaseSkill, FailureCode, SkillResult, SkillSpec, SkillStatus


class HomeSkill(BaseSkill):
    SPEC = SkillSpec(
        name="home",
        description="Return the robot to a known safe pose.",
        inputs=("optional named pose",),
        preconditions=(),
        effects=("robot reaches a safe pose",),
        failures=(FailureCode.PATH_BLOCKED, FailureCode.TIMEOUT),
    )

    def __init__(self, backend: ManipulationBackend):
        self.backend = backend

    def execute(self, name="home") -> SkillResult:
        result = self.backend.home(name)

        if result.ok:
            return SkillResult(
                SkillStatus.SUCCESS,
                result.message,
                details=result.details,
            )

        return SkillResult(
            SkillStatus.TIMEOUT if result.timed_out else SkillStatus.FAILED,
            result.message,
            FailureCode.TIMEOUT if result.timed_out else FailureCode.PATH_BLOCKED,
            result.details,
        )