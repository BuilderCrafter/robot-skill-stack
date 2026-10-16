from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.skill import FailureCode, SkillResult, SkillStatus


class RobotRuntime:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    async def execute_async(self, skill_name: str, **kwargs) -> SkillResult:
        # No await between checking and reserving: commands on the same
        # cooperative event loop are rejected, not queued behind stale targets.
        if self._busy:
            return SkillResult(SkillStatus.FAILED, "Another command is executing.", FailureCode.BUSY)
        self._busy = True
        try:
            return await self.registry.get(skill_name).execute_async(**kwargs)
        except Exception as exc:
            return SkillResult(
                SkillStatus.FAILED,
                f"{skill_name} raised {type(exc).__name__}: {exc}",
                FailureCode.INTERNAL_ERROR,
            )
        finally:
            # Also release on cancellation; CancelledError is not converted to success/failure.
            self._busy = False

    def available_skills(self):
        return self.registry.names()
