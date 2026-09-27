from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.skill import (
    FailureCode,
    SkillResult,
    SkillStatus,
)


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

    async def execute_async(
        self,
        skill_name: str,
        **kwargs,
    ) -> SkillResult:
        try:
            skill = self.registry.get(skill_name)
            execute_async = getattr(skill, "execute_async", None)
            if execute_async is None:
                return SkillResult(
                    SkillStatus.FAILED,
                    f"Skill '{skill_name}' has no cooperative async path.",
                    FailureCode.INTERNAL_ERROR,
                )
            return await execute_async(**kwargs)
        except Exception as exc:
            return SkillResult(
                SkillStatus.FAILED,
                f"{skill_name} raised {type(exc).__name__}: {exc}",
                FailureCode.INTERNAL_ERROR,
            )

    def available_skills(self):
        return self.registry.names()
