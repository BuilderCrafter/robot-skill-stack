"""Cooperative command ownership and BT sequencing, without Isaac."""
import asyncio
import unittest
from unittest.mock import AsyncMock

import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.skills.home import HomeSkill
from robot_skill_stack.manipulation.skills.move_to_pose import MoveToPoseSkill
from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place import create_pick_and_place_tree
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place_recovery import create_pick_and_place_recovery_tree
from robot_skill_stack.runtime.registry import SkillRegistry
from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.runtime.skill import BaseSkill, FailureCode, SkillResult, SkillStatus
from tests.unit.test_async_runtime import FakeAsyncBackend
from robot_skill_stack.world.model.world_model import WorldModel


async def next_tick():
    await asyncio.sleep(0)


class Command(BaseSkill):
    """A yielding command to exercise orchestration independently of geometry."""
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback

    async def execute_async(self, **kwargs):
        return await self.callback(**kwargs)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_preserves_names_and_rejects_duplicate_or_missing_commands(self):
        registry = SkillRegistry()
        command = Command('home', AsyncMock())
        registry.register(command)
        self.assertEqual(registry.names(), ['home'])
        self.assertIs(registry.get('home'), command)
        with self.assertRaisesRegex(ValueError, 'already registered'):
            registry.register(Command('home', AsyncMock()))
        with self.assertRaisesRegex(KeyError, 'Unknown skill'):
            registry.get('missing')
        self.assertIs(registry.get('home'), command)

    def stack(self):
        backend, registry = FakeAsyncBackend(), SkillRegistry()
        model = WorldModel()
        registry.register(HomeSkill(backend, model))
        registry.register(MoveToPoseSkill(backend, model))
        return backend, RobotRuntime(registry, model)

    async def test_operator_reset_is_software_only_and_rejects_busy_or_changed_attachment(self):
        from robot_skill_stack.world.model.entities import WorldObject
        model=WorldModel()
        model.register(WorldObject('cube'))
        model.set_held('cube',object_offset_in_ee=[0,0,.03],uncertain=True)
        registry=SkillRegistry()
        entered,release=asyncio.Event(),asyncio.Event()
        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return SkillResult(SkillStatus.SUCCESS)
        registry.register(Command('blocked',blocked))
        runtime=RobotRuntime(registry,model)
        task=asyncio.create_task(runtime.execute_async('blocked'))
        await entered.wait()
        with self.assertRaisesRegex(RuntimeError,'command is executing'):
            runtime.clear_held_state('cube')
        self.assertEqual(model.held_object_id,'cube')
        self.assertTrue(model.held_object_uncertain)
        release.set()
        await task
        with self.assertRaisesRegex(RuntimeError,'Held object changed'):
            runtime.clear_held_state('different')
        self.assertEqual(runtime.clear_held_state('cube'),'cube')
        self.assertIsNone(model.held_object_id)
        self.assertFalse(model.held_object_uncertain)
        self.assertIsNone(model.held_object_offset_in_ee)
        self.assertTrue(model.exists('cube'))

    async def test_busy_rejects_overlap_without_motion_then_accepts_next_command(self):
        backend, runtime = self.stack()
        entered, release = asyncio.Event(), asyncio.Event()
        async def home(name="home", **kwargs):
            entered.set()
            await release.wait()
            return BackendResult(True)
        backend.home_async = AsyncMock(side_effect=home)
        running = asyncio.create_task(runtime.execute_async('home'))
        await asyncio.wait_for(entered.wait(), 2)
        self.assertTrue(runtime.busy)
        rejected = await runtime.execute_async('move_to_pose', target=Pose([.4, 0, .2]))
        self.assertEqual(rejected.failure_code, FailureCode.BUSY)
        self.assertEqual(backend.async_calls, 0)
        release.set()
        self.assertTrue((await running).ok)
        self.assertFalse(runtime.busy)
        self.assertTrue((await runtime.execute_async('move_to_pose', target=Pose([.4, 0, .2]))).ok)

    async def test_exceptions_and_unknown_commands_release_busy_state(self):
        backend, runtime = self.stack()
        backend.home_async = AsyncMock(side_effect=ValueError('controller failed'))
        failed = await runtime.execute_async('home')
        self.assertEqual(failed.failure_code, FailureCode.INTERNAL_ERROR)
        self.assertIn('controller failed', failed.message)
        self.assertFalse(runtime.busy)
        self.assertEqual((await runtime.execute_async('missing')).failure_code, FailureCode.INTERNAL_ERROR)
        backend.home_async = AsyncMock(return_value=BackendResult(True))
        self.assertTrue((await runtime.execute_async('home')).ok)

    async def test_timeout_and_unreachable_results_propagate_without_poisoning_next_command(self):
        backend, runtime = self.stack()
        backend.move_to_pose_async = AsyncMock(return_value=BackendResult(False, timed_out=True, details={'steps': 1000}))
        timed_out = await runtime.execute_async('move_to_pose', target=Pose([.4, 0, .2]))
        self.assertEqual(timed_out.status, SkillStatus.TIMEOUT)
        self.assertEqual(timed_out.failure_code, FailureCode.TIMEOUT)
        self.assertEqual(timed_out.details['steps'], 1000)
        backend.check_reachability = lambda pose: False
        failed = await runtime.execute_async('move_to_pose', target=Pose([.4, 0, .2]))
        self.assertEqual(failed.failure_code, FailureCode.UNREACHABLE)
        self.assertEqual(backend.move_to_pose_async.await_count, 1)
        self.assertFalse(runtime.busy)
        self.assertTrue((await runtime.execute_async('home')).ok)

    async def test_home_forwards_name_and_preserves_backend_failure(self):
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out):
                backend, runtime = self.stack()
                backend.home_async = AsyncMock(return_value=BackendResult(
                    False, 'named motion failed', timed_out=timed_out, details={'pose_name': 'home'}))
                result = await runtime.execute_async('home', name='home')
                backend.home_async.assert_awaited_once_with('home')
                self.assertEqual(result.status, SkillStatus.TIMEOUT if timed_out else SkillStatus.FAILED)
                self.assertEqual(result.failure_code, FailureCode.TIMEOUT if timed_out else FailureCode.PATH_BLOCKED)
                self.assertEqual(result.message, 'named motion failed')
                self.assertEqual(result.details['pose_name'], 'home')

    async def test_cancelled_command_releases_ownership_and_propagates_cancellation(self):
        backend, runtime = self.stack()
        entered = asyncio.Event()
        async def home(name="home", **kwargs):
            entered.set()
            await asyncio.Event().wait()
        backend.home_async = AsyncMock(side_effect=home)
        running = asyncio.create_task(runtime.execute_async('home'))
        await asyncio.wait_for(entered.wait(), 2)
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertFalse(runtime.busy)
        backend.home_async = AsyncMock(return_value=BackendResult(True))
        self.assertTrue((await runtime.execute_async('home')).ok)

    def task_runtime(self, fail_first=False):
        calls, registry = [], SkillRegistry()
        for name in ('pick', 'place', 'home'):
            async def callback(_name=name, **kwargs):
                calls.append(_name)
                if fail_first and calls == ['pick']:
                    return SkillResult(SkillStatus.FAILED, 'not visible', FailureCode.OBJECT_POSE_UNKNOWN)
                # Multiple ticks expose selectors that incorrectly restart Pick during Home.
                for _ in range(3):
                    await next_tick()
                return SkillResult(SkillStatus.SUCCESS)
            registry.register(Command(name, callback))
        return RobotRuntime(registry, WorldModel()), calls

    async def test_bt_sequence_uses_canonical_runtime_and_keeps_results(self):
        runtime, calls = self.task_runtime()
        root = create_pick_and_place_tree(runtime, 'object', Pose([.4, .2, .025]))
        self.assertEqual(await BTExecutor(root).run_async(next_tick), py_trees.common.Status.SUCCESS)
        self.assertEqual(calls, ['pick', 'place', 'home'])
        self.assertFalse(runtime.busy)

    async def test_recovery_resumes_running_branch_without_restarting_first_pick(self):
        runtime, calls = self.task_runtime(fail_first=True)
        root = create_pick_and_place_recovery_tree(runtime, 'object', Pose([.4, .2, .025]))
        self.assertEqual(await BTExecutor(root).run_async(next_tick), py_trees.common.Status.SUCCESS)
        self.assertEqual(calls, ['pick', 'home', 'pick', 'place', 'home'])
        first = next(node for node in root.iterate() if node.name == 'Pick object')
        self.assertEqual(first.result.failure_code, FailureCode.OBJECT_POSE_UNKNOWN)

    async def test_recovery_uses_real_home_skill_and_stops_if_it_times_out(self):
        for succeeds in (False, True):
            with self.subTest(succeeds=succeeds):
                model, registry = WorldModel(), SkillRegistry()
                backend = FakeAsyncBackend()
                backend.home_async = AsyncMock(return_value=BackendResult(
                    succeeds, 'home timeout' if not succeeds else '', timed_out=not succeeds))
                pick = AsyncMock(side_effect=[
                    SkillResult(SkillStatus.FAILED, failure_code=FailureCode.OBJECT_POSE_UNKNOWN),
                    SkillResult(SkillStatus.SUCCESS),
                ])
                place = AsyncMock(return_value=SkillResult(SkillStatus.SUCCESS))
                registry.register(Command('pick', pick))
                registry.register(Command('place', place))
                registry.register(HomeSkill(backend, model))
                runtime = RobotRuntime(registry, model)
                tree = create_pick_and_place_recovery_tree(
                    runtime, 'object', Pose([.4, .2, .025]), return_home=False)
                status = await BTExecutor(tree).run_async(next_tick)
                self.assertEqual(status, py_trees.common.Status.SUCCESS if succeeds else py_trees.common.Status.FAILURE)
                backend.home_async.assert_awaited_once_with('home')
                self.assertEqual(pick.await_count, 2 if succeeds else 1)
                self.assertEqual(place.await_count, 1 if succeeds else 0)
                home = next(node for node in tree.iterate() if node.name == 'Recovery Home')
                if not succeeds:
                    self.assertEqual(home.result.status, SkillStatus.TIMEOUT)
                    self.assertEqual(home.result.failure_code, FailureCode.TIMEOUT)
                self.assertFalse(runtime.busy)

    async def test_bt_failure_stops_later_commands(self):
        runtime, calls = self.task_runtime(fail_first=True)
        root = create_pick_and_place_tree(runtime, 'object', Pose([.4, .2, .025]))
        self.assertEqual(await BTExecutor(root).run_async(next_tick), py_trees.common.Status.FAILURE)
        self.assertEqual(calls, ['pick'])
        self.assertFalse(runtime.busy)

    async def test_bt_tick_budget_cancels_and_drains_pending_command(self):
        registry = SkillRegistry()
        async def blocked(**kwargs):
            await asyncio.Event().wait()
        registry.register(Command('pick', blocked))
        runtime = RobotRuntime(registry, WorldModel())
        root = create_pick_and_place_tree(runtime, 'object', Pose([.4, .2, .025]))
        status = await BTExecutor(root).run_async(next_tick, max_ticks=2)
        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertFalse(runtime.busy)
        node = next(node for node in root.iterate() if node.name == 'Pick object')
        self.assertTrue(node._task.cancelled())
        registry.get('pick').callback = AsyncMock(return_value=SkillResult(SkillStatus.SUCCESS))
        self.assertTrue((await runtime.execute_async('pick')).ok)

    async def test_bt_executor_cancellation_drains_child_before_returning(self):
        registry, entered = SkillRegistry(), asyncio.Event()
        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        registry.register(Command('pick', blocked))
        runtime = RobotRuntime(registry, WorldModel())
        root = create_pick_and_place_tree(runtime, 'object', Pose([.4, .2, .025]))
        task = asyncio.create_task(BTExecutor(root).run_async(next_tick))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(runtime.busy)


if __name__ == '__main__':
    unittest.main()
