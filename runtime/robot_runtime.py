from core.skill import FailureCode, SkillResult, SkillStatus
from runtime.skill_registry import SkillRegistry


class RobotRuntime:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, skill_name: str, **kwargs) -> SkillResult:
        try:
            return self.registry.get(skill_name).execute(**kwargs)
        except Exception as exc:
            return SkillResult(
                SkillStatus.FAILED,
                f"{skill_name} raised {type(exc).__name__}: {exc}",
                FailureCode.INTERNAL_ERROR,
            )

    def available_skills(self) -> list[str]:
        return self.registry.names()