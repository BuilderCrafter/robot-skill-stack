"""Gripper evidence and attachment bookkeeping shared by manipulation skills."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from numpy.typing import ArrayLike

from robot_skill_stack.common.types import GripperState
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback, ManipulationBackend
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.world_model import WorldModel


def motion_failure_code(result: BackendResult, fallback: FailureCode) -> FailureCode:
    """Prefer structured source/grasp failures over timeout or a skill's fallback."""
    if result.details.get('world_state_unavailable'):
        return FailureCode.WORLD_STATE_UNAVAILABLE
    if result.details.get('unsupported_motion'):
        return FailureCode.MOTION_UNSUPPORTED
    if result.details.get('grasp_failure'):
        return (FailureCode.GRASP_LOST if result.details['grasp_state'] == 'free'
                else FailureCode.GRASP_UNCERTAIN)
    return FailureCode.TIMEOUT if result.timed_out else fallback


class HeldState:
    """Keep attachment state consistent with gripper evidence across skill commands."""

    def __init__(self, backend: ManipulationBackend, world_model: WorldModel) -> None:
        self.backend, self.model = backend, world_model

    def _clear_lost_attachment(self) -> None:
        obj = self.model.get(self.model.held_object_id)
        if obj is not None:
            # Require a fresh provider observation before another Pick.
            obj.visible = False
        self.model.clear_held_state()

    def failure(self, state: GripperState, object_id: str | None = None) -> BackendResult:
        message = ('Object lost from gripper.' if state == GripperState.FREE else
                   'Grasp is uncertain; inspect the gripper before continuing.')
        return BackendResult(False, message, details={
            'grasp_failure': True, 'grasp_state': state.value,
            'object_id': object_id or self.model.held_object_id,
        })

    async def resolve(self, *, keep_placement_hint: bool = False) -> GripperFeedback:
        """Resolve feedback; confirmed Free can preserve a successful release's hint."""
        if self.model.held_object_id is not None:
            self.model.held_object_uncertain = True
        feedback = await self.backend.inspect_gripper_async()
        if self.model.held_object_id is not None:
            if feedback.state == GripperState.FREE:
                if keep_placement_hint:
                    self.model.set_held(None)
                else:
                    self._clear_lost_attachment()
            else:
                self.model.held_object_uncertain = feedback.state == GripperState.UNCERTAIN
        return feedback

    async def _after_error(self) -> None:
        # Preserve the original exception/cancellation even if feedback fails.
        if self.model.held_object_id is not None:
            self.model.held_object_uncertain = True
            try:
                await self.resolve()
            except BaseException:
                pass

    async def close(self, object_id: str, offset: ArrayLike | None) -> tuple[BackendResult, GripperFeedback]:
        """Reserve an uncertain attachment before closing, then resolve its evidence."""
        self.model.set_held(object_id, object_offset_in_ee=offset, uncertain=True)
        try:
            result = await self.backend.close_gripper_async()
            feedback = await self.resolve()
            if not result.ok:
                result.details['grasp_state'] = feedback.state.value
                return result, feedback
            if feedback.state != GripperState.HELD:
                return self.failure(feedback.state, object_id), feedback
            return result, feedback
        except BaseException:
            await self._after_error()
            raise

    async def release(self) -> BackendResult:
        """Open and confirm Free; preserve the original error if inspection also fails."""
        object_id = self.model.held_object_id
        self.model.held_object_uncertain = True
        try:
            result = await self.backend.open_gripper_async()
            feedback = await self.resolve(keep_placement_hint=result.ok)
            if result.ok and feedback.state != GripperState.FREE:
                self.model.held_object_uncertain = True
                return self.failure(GripperState.UNCERTAIN)
            return result
        except BaseException:
            self.model.clear_association_hint(object_id)
            await self._after_error()
            raise

    async def run_motion(self, command: Callable[..., Awaitable[BackendResult]],
                         *args, **kwargs) -> BackendResult:
        """Stop after three suspect samples; inspect any remaining suspicion at completion."""
        if self.model.held_object_uncertain:
            return self.failure(GripperState.UNCERTAIN)
        object_id = self.model.held_object_id
        if object_id is None:
            return await command(*args, **kwargs)

        suspect_states: list[GripperState] = []

        def check_grasp() -> BackendResult | None:
            feedback = self.backend.gripper_feedback()
            if feedback.state == GripperState.HELD:
                suspect_states.clear()
                return None
            suspect_states.append(feedback.state)
            if len(suspect_states) < 3:
                return None
            state = (GripperState.FREE if all(s == GripperState.FREE for s in suspect_states)
                     else GripperState.UNCERTAIN)
            if state == GripperState.FREE:
                self._clear_lost_attachment()
            else:
                self.model.held_object_uncertain = True
            return self.failure(state, object_id)

        try:
            # Resolve suspect feedback before issuing a carrying command.
            if self.backend.gripper_feedback().state != GripperState.HELD:
                feedback = await self.resolve()
                if feedback.state != GripperState.HELD:
                    return self.failure(feedback.state, object_id)
            result = await command(*args, grasp_check=check_grasp, **kwargs)
            if result.ok and suspect_states:
                feedback = await self.resolve()
                if feedback.state != GripperState.HELD:
                    return self.failure(feedback.state, object_id)
            if not result.ok:
                await self._after_error()
                result.details['resolved_gripper_state'] = self.model.gripper_state.value
            return result
        except BaseException:
            await self._after_error()
            raise
