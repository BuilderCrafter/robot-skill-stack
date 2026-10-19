"""Failure/cancellation state transitions and carrying guards, without Isaac."""
import asyncio
from robot_skill_stack.world.model.world_model import WorldModel
import unittest
from unittest.mock import AsyncMock

import py_trees

from robot_skill_stack.common.types import GripperState, Pose
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback
from robot_skill_stack.manipulation.held_state import HeldState
from robot_skill_stack.manipulation.skills.home import HomeSkill
from robot_skill_stack.manipulation.skills.move_to_pose import MoveToPoseSkill
from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place_recovery import create_pick_and_place_recovery_tree
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.runtime.skill import FailureCode, SkillResult, SkillStatus
from tests.unit.test_execution_runtime import Command, next_tick
from tests.unit import test_grasp_attachment as attachment_tests


class HeldStateTests(unittest.IsolatedAsyncioTestCase):
    def stack(self):
        model, backend, runtime = attachment_tests.AttachmentTests().stack()
        runtime.registry.register(HomeSkill(backend, model))
        runtime.registry.register(MoveToPoseSkill(backend, model))
        return model, backend, runtime

    async def picked(self):
        model, backend, runtime = self.stack()
        result = await runtime.execute_async('pick', object_id='target')
        self.assertTrue(result.ok, result)
        backend.calls.clear()
        return model, backend, runtime

    async def test_grasp_is_recorded_before_lift_and_failed_lift_keeps_original_timeout(self):
        model, backend, runtime = self.stack()
        move = backend.move_to_pose_async
        count = 0
        async def fail_lift(target, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                self.assertEqual(model.gripper_state, GripperState.HELD)
                self.assertIsNotNone(model.held_object_offset_in_ee)
                return BackendResult(False, 'controller timeout', timed_out=True, details={'steps': 1000})
            return await move(target, **kwargs)
        backend.move_to_pose_async = fail_lift
        result = await runtime.execute_async('pick', object_id='target')
        self.assertEqual(result.status, SkillStatus.TIMEOUT)
        self.assertEqual(result.failure_code, FailureCode.TIMEOUT)
        self.assertEqual(result.details['steps'], 1000)
        self.assertTrue(result.details['gripper_engaged'])
        self.assertEqual(model.gripper_state, GripperState.HELD)
        self.assertEqual([c[0] for c in backend.calls].count('open'), 1)

    async def test_failed_approach_does_not_reserve_an_attachment(self):
        model, backend, runtime = self.stack()
        backend.move_to_pose_async = AsyncMock(return_value=BackendResult(False, timed_out=True))
        result = await runtime.execute_async('pick', object_id='target')
        self.assertEqual(result.failure_code, FailureCode.TIMEOUT)
        self.assertEqual(model.gripper_state, GripperState.FREE)
        self.assertIsNone(model.held_object_offset_in_ee)

    async def test_empty_and_invalid_closure_are_distinct_without_automatic_release(self):
        for state, code in ((GripperState.FREE, FailureCode.GRASP_FAILED),
                            (GripperState.UNCERTAIN, FailureCode.GRASP_UNCERTAIN)):
            with self.subTest(state=state):
                model, backend, runtime = self.stack()
                backend.gripper_feedback = lambda: GripperFeedback(state)
                result = await runtime.execute_async('pick', object_id='target')
                self.assertEqual(result.failure_code, code)
                self.assertEqual(model.gripper_state, state)
                self.assertEqual([c[0] for c in backend.calls], ['open', 'move', 'move', 'close'])
                self.assertEqual(model.held_object_id, 'target' if state == GripperState.UNCERTAIN else None)

    async def test_cancellation_during_closing_resolves_feedback_but_remains_cancelled(self):
        for state in (GripperState.HELD, GripperState.FREE, GripperState.UNCERTAIN):
            with self.subTest(state=state):
                model, backend, runtime = self.stack()
                async def cancel_close(width=None):
                    self.assertEqual(model.gripper_state, GripperState.UNCERTAIN)
                    self.assertEqual(model.held_object_id, 'target')
                    raise asyncio.CancelledError('during closing')
                backend.close_gripper_async = cancel_close
                backend.gripper_feedback = lambda: GripperFeedback(state)
                with self.assertRaisesRegex(asyncio.CancelledError, 'during closing'):
                    await runtime.execute_async('pick', object_id='target')
                self.assertEqual(model.gripper_state, state)
                self.assertFalse(runtime.busy)

    async def test_error_during_closing_is_preserved_when_recheck_also_fails(self):
        model, backend, runtime = self.stack()
        backend.close_gripper_async = AsyncMock(side_effect=ValueError('original close error'))
        backend.inspect_gripper_async = AsyncMock(side_effect=RuntimeError('feedback unavailable'))
        result = await runtime.execute_async('pick', object_id='target')
        self.assertEqual(result.failure_code, FailureCode.INTERNAL_ERROR)
        self.assertIn('original close error', result.message)
        self.assertNotIn('feedback unavailable', result.message)
        self.assertEqual(model.gripper_state, GripperState.UNCERTAIN)

    async def test_cancellation_during_lift_retains_confirmed_grasp(self):
        model, backend, runtime = self.stack()
        move = backend.move_to_pose_async
        entered = asyncio.Event()
        async def wait_in_lift(target, **kwargs):
            if backend.closed:
                entered.set()
                await asyncio.Event().wait()
            return await move(target, **kwargs)
        backend.move_to_pose_async = wait_in_lift
        task = asyncio.create_task(runtime.execute_async('pick', object_id='target'))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(model.gripper_state, GripperState.HELD)
        self.assertEqual(model.held_object_id, 'target')
        self.assertFalse(runtime.busy)

    async def test_slip_during_lift_stops_pick_and_clears_attachment(self):
        model, backend, runtime = self.stack()
        move = backend.move_to_pose_async
        async def lose_cube(target, **kwargs):
            if backend.closed:
                backend.closed = False
            return await move(target, **kwargs)
        backend.move_to_pose_async = lose_cube
        result = await runtime.execute_async('pick', object_id='target')
        self.assertEqual(result.failure_code, FailureCode.GRASP_LOST)
        self.assertEqual(model.gripper_state, GripperState.FREE)
        self.assertIsNone(model.held_object_offset_in_ee)
        self.assertTrue(result.details['gripper_engaged'])
        self.assertFalse(model.require('target').visible)
        previous_calls = list(backend.calls)
        retry = await runtime.execute_async('pick', object_id='target')
        self.assertEqual(retry.failure_code, FailureCode.OBJECT_POSE_UNKNOWN)
        self.assertEqual(backend.calls, previous_calls)

    async def test_slip_during_place_carry_stops_before_release_or_retreat(self):
        model, backend, runtime = await self.picked()
        move = backend.move_to_pose_async
        async def lose_cube(target, **kwargs):
            backend.closed = False
            return await move(target, **kwargs)
        backend.move_to_pose_async = lose_cube
        result = await runtime.execute_async('place', target=Pose([.45, .25, .05]))
        self.assertEqual(result.failure_code, FailureCode.GRASP_LOST)
        self.assertEqual([c[0] for c in backend.calls], ['move'])
        self.assertEqual(model.gripper_state, GripperState.FREE)
        self.assertFalse(model.association_hints())

    async def test_uncertainty_detected_in_motion_does_not_resume_after_recheck_resolves_held(self):
        model, backend, runtime = await self.picked()
        # One preflight query, then three invalid in-motion samples, then recovery.
        states = iter([GripperState.HELD]+[GripperState.UNCERTAIN]*3+[GripperState.HELD])
        backend.gripper_feedback = lambda: GripperFeedback(next(states), .05)
        result = await runtime.execute_async('move_to_pose', target=Pose([.45, .1, .2], backend.ee.orientation))
        self.assertEqual(result.failure_code, FailureCode.GRASP_UNCERTAIN)
        self.assertEqual(model.gripper_state, GripperState.HELD)
        self.assertEqual([c[0] for c in backend.calls], ['move'])

    async def test_brief_bad_feedback_is_tolerated_but_loss_at_motion_end_is_resolved(self):
        for readings, resolved in (([GripperState.FREE, GripperState.HELD, GripperState.HELD],
                                     GripperState.HELD),
                                    ([GripperState.HELD, GripperState.FREE, GripperState.FREE],
                                     GripperState.FREE)):
            with self.subTest(resolved=resolved):
                model, backend, runtime = await self.picked()
                states = iter([GripperState.HELD]+readings+[resolved])
                backend.gripper_feedback = lambda: GripperFeedback(next(states), .05)
                result = await runtime.execute_async('move_to_pose', target=Pose([.45, .1, .2], backend.ee.orientation))
                self.assertEqual(result.ok, resolved == GripperState.HELD)
                self.assertEqual(model.gripper_state, resolved)

    async def test_already_lost_grasp_is_resolved_before_any_place_command(self):
        model, backend, runtime = await self.picked()
        backend.closed = False
        result = await runtime.execute_async('place', target=Pose([.45, .25, .05]))
        self.assertEqual(result.failure_code, FailureCode.GRASP_LOST)
        self.assertEqual(backend.calls, [])
        self.assertEqual(model.gripper_state, GripperState.FREE)

    async def test_uncertainty_blocks_all_commands_and_protects_candidate_until_reset(self):
        model, backend, runtime = self.stack()
        model.set_held('target', object_offset_in_ee=[0, 0, .03], uncertain=True)
        for name, args in (('pick', {'object_id': 'target'}), ('place', {'target': Pose([.45, .25, .05])}),
                           ('home', {}), ('move_to_pose', {'target': Pose([.45, 0, .2])})):
            result = await runtime.execute_async(name, **args)
            self.assertEqual(result.failure_code, FailureCode.GRASP_UNCERTAIN)
        self.assertEqual(backend.calls, [])
        obj = model.require('target')
        obj.visible, obj.last_seen = False, 0.
        self.assertFalse(model.remove('target'))
        from tests.unit.test_world_observations import source
        from robot_skill_stack.world.model.updater import WorldModelUpdater
        self.assertFalse(WorldModelUpdater(model, source()).cleanup(now=1e9))
        model.clear_held_state()
        self.assertEqual(model.gripper_state, GripperState.FREE)
        self.assertFalse(model.held_object_uncertain)
        self.assertIsNone(model.held_object_offset_in_ee)

    async def test_home_preserves_held_attachment_and_stops_on_in_motion_loss(self):
        for loses_object in (False, True):
            with self.subTest(loses_object=loses_object):
                model, backend, runtime = await self.picked()
                if loses_object:
                    move = backend.move_to_pose_async
                    async def slip(target, **kwargs):
                        backend.closed = False
                        return await move(target, **kwargs)
                    backend.move_to_pose_async = slip
                result = await runtime.execute_async('home')
                self.assertEqual(result.ok, not loses_object)
                if loses_object:
                    self.assertEqual(result.failure_code, FailureCode.GRASP_LOST)
                    self.assertEqual(model.gripper_state, GripperState.FREE)
                    self.assertIsNone(model.held_object_offset_in_ee)
                else:
                    self.assertEqual(model.gripper_state, GripperState.HELD)
                    self.assertEqual(model.held_object_id, 'target')
                    self.assertIsNotNone(model.held_object_offset_in_ee)
                self.assertEqual([c[0] for c in backend.calls], ['move'])

    async def test_interrupted_opening_resolves_state_without_masking_cancellation(self):
        for state in (GripperState.FREE, GripperState.HELD, GripperState.UNCERTAIN):
            with self.subTest(state=state):
                model, backend, runtime = await self.picked()
                async def cancel_open():
                    self.assertEqual(model.gripper_state, GripperState.UNCERTAIN)
                    backend.gripper_feedback = lambda: GripperFeedback(state)
                    raise asyncio.CancelledError('during opening')
                backend.open_gripper_async = cancel_open
                with self.assertRaisesRegex(asyncio.CancelledError, 'during opening'):
                    await runtime.execute_async('place', target=Pose([.45, .25, .05]))
                self.assertEqual(model.gripper_state, state)
                self.assertFalse(model.association_hints())
                self.assertFalse(runtime.busy)

    async def test_failed_open_can_leave_free_and_failed_retreat_never_reasserts_held(self):
        for failed_open in (True, False):
            with self.subTest(failed_open=failed_open):
                model, backend, runtime = await self.picked()
                open_gripper = backend.open_gripper_async
                async def release():
                    await open_gripper()
                    return BackendResult(not failed_open, 'original opening failure' if failed_open else '')
                backend.open_gripper_async = release
                move = backend.move_to_pose_async
                async def fail_retreat(target, **kwargs):
                    return (await move(target, **kwargs) if backend.closed else
                            BackendResult(False, 'retreat timeout', timed_out=True))
                backend.move_to_pose_async = fail_retreat
                result = await runtime.execute_async('place', target=Pose([.45, .25, .05]))
                self.assertEqual(model.gripper_state, GripperState.FREE)
                self.assertIsNone(model.held_object_offset_in_ee)
                if failed_open:
                    self.assertEqual(result.failure_code, FailureCode.PLACE_FAILED)
                    self.assertIn('original opening failure', result.message)
                    self.assertFalse(model.association_hints())
                else:
                    self.assertFalse(result.details['retreat_ok'])

    async def test_recovery_does_not_home_or_retry_after_gripper_engagement_or_loss(self):
        for code, details in ((FailureCode.GRASP_FAILED, {'gripper_engaged': True}),
                               (FailureCode.GRASP_LOST, {}), (FailureCode.GRASP_UNCERTAIN, {}),
                               (FailureCode.INTERNAL_ERROR, {}), (FailureCode.BUSY, {}),
                               (FailureCode.WORLD_STATE_UNAVAILABLE, {})):
            with self.subTest(code=code):
                registry, calls = SkillRegistry(), []
                for name in ('pick', 'home', 'place'):
                    async def command(_name=name, **kwargs):
                        calls.append(_name)
                        return SkillResult(SkillStatus.FAILED, failure_code=code, details=details)
                    registry.register(Command(name, command))
                root = create_pick_and_place_recovery_tree(RobotRuntime(registry, WorldModel()), 'target', Pose([.45, .25, .05]))
                status = await BTExecutor(root).run_async(next_tick)
                self.assertEqual(status, py_trees.common.Status.FAILURE)
                self.assertEqual(calls, ['pick'])
