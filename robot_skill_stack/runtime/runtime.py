from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.runtime.skill import FailureCode, SkillResult, SkillStatus


class RobotRuntime:
    def __init__(self, registry: SkillRegistry, world_model: WorldModel):
        self.registry = registry
        self.world_model = world_model
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

    def clear_held_state(self, expected_id: str):
        """Explicit software-only operator reset, never during a robot command."""
        if self.busy:
            raise RuntimeError('Another command is executing; held state was not cleared')
        if self.world_model.held_object_id != expected_id:
            raise RuntimeError('Held object changed; nothing was cleared')
        return self.world_model.clear_held_state()

    def available_skills(self):
        return self.registry.names()
