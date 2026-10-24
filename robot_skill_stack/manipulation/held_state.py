"""Gripper evidence and attachment bookkeeping shared by manipulation skills."""
from robot_skill_stack.common.types import GripperState
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.runtime.skill import FailureCode
from robot_skill_stack.world.model.world_model import WorldModel


def motion_failure_code(result, fallback):
    if result.details.get('world_state_unavailable'):
        return FailureCode.WORLD_STATE_UNAVAILABLE
    if result.details.get('unsupported_motion'):
        return FailureCode.MOTION_UNSUPPORTED
    if result.details.get('grasp_failure'):
        return (FailureCode.GRASP_LOST if result.details['grasp_state'] == 'free'
                else FailureCode.GRASP_UNCERTAIN)
    return FailureCode.TIMEOUT if result.timed_out else fallback


class HeldState:
    def __init__(self, backend, world_model: WorldModel):
        self.backend, self.model = backend, world_model

    def _clear_lost_attachment(self):
        obj = self.model.get(self.model.held_object_id)
        if obj is not None:
            # Require a fresh provider observation before another Pick.
            obj.visible = False
        self.model.clear_held_state()

    def failure(self, state, object_id=None):
        message = ('Object lost from gripper.' if state == GripperState.FREE else
                   'Grasp is uncertain; inspect the gripper before continuing.')
        return BackendResult(False, message, details={
            'grasp_failure': True, 'grasp_state': state.value,
            'object_id': object_id or self.model.held_object_id,
        })

    async def resolve(self, *, keep_placement_hint=False):
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

    async def _after_error(self):
        # Preserve the original exception/cancellation even if feedback fails.
        if self.model.held_object_id is not None:
            self.model.held_object_uncertain = True
            try:
                await self.resolve()
            except BaseException:
                pass

    async def close(self, object_id, offset):
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

    async def release(self):
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

    async def run_motion(self, command, *args, **kwargs):
        if self.model.held_object_uncertain:
            return self.failure(GripperState.UNCERTAIN)
        object_id = self.model.held_object_id
        if object_id is None:
            return await command(*args, **kwargs)

        suspect = []

        def check():
            feedback = self.backend.gripper_feedback()
            if feedback.state == GripperState.HELD:
                suspect.clear()
                return None
            suspect.append(feedback.state)
            if len(suspect) < 3:
                return None
            state = (GripperState.FREE if all(s == GripperState.FREE for s in suspect)
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
            result = await command(*args, grasp_check=check, **kwargs)
            if result.ok and suspect:
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
