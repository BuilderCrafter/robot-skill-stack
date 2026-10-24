"""Reactive motion keeps the SDK command contract while fencing invalid maps."""
import asyncio
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.common.types import GripperState, Pose
from robot_skill_stack.manipulation.backend import BackendResult, GripperFeedback
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleBox, ObstaclePacket
from tests.unit import test_franka_async_backend as fixtures
from tests.unit.test_rmpflow_obstacle_registry import Policy
from robot_skill_stack.integrations.isaac.manipulation.motion.obstacles import RmpflowObstacleRegistry


def load_reactive():
    modules = {name: ModuleType(name) for name in (
        'isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.types')}
    modules['isaacsim.core.utils.types'].ArticulationAction = NS
    name = 'robot_skill_stack.integrations.isaac.manipulation.motion.reactive_under_test'
    path = Path(__file__).resolve().parents[2]/'robot_skill_stack/integrations/isaac/manipulation/motion/reactive.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module.ReactiveRmpflowStrategy


@dataclass(frozen=True)
class View:
    timestamp: float | None
    stale_after: float
    generation: int
    revision: int
    obstacles: object
    health: object

    @property
    def states(self):
        raise AssertionError('Reactive control scanned the occupancy grid')

    @property
    def probabilities(self):
        raise AssertionError('Reactive control copied the occupancy grid')


def view(*, timestamp=99.5, generation=0, revision=1, status='ok', fresh=True, boxes=()):
    packet = ObstaclePacket(generation, revision, timestamp, 99.6, boxes, {})
    return View(timestamp, 1., generation, revision, packet,
                MappingProxyType(dict(status=status, fresh=fresh)))


class Registry:
    def __init__(self):
        self.calls, self.count, self.closed = [], 0, False

    @property
    def diagnostics(self):
        return dict(proxy_count=self.count, closed=self.closed)

    def reset_tracking(self):
        self.calls.append(('reset',))
        self.count = 0

    def sync(self, packet, *, now):
        self.calls.append(('sync', packet, now))
        self.count = sum(box.expires_at > now for box in packet.boxes)

    def clear(self):
        self.calls.append(('clear',))
        self.count = 0

    def close(self):
        self.calls.append(('close',))
        self.closed = True
        self.clear()


class ReactiveStrategyTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy_class = load_reactive()

    def context(self, *, reaches=True, playing=True, registry=None):
        backend, state, target = fixtures.FrankaCommandTests().adapter(reaches=reaches, playing=playing)
        backend.robot.handles_initialized = True
        backend.gripper_feedback = Mock(return_value=GripperFeedback(GripperState.FREE, .08))
        holder, clock = NS(view=view()), NS(now=100.)
        source = NS(snapshot=Mock(side_effect=lambda: holder.view))
        registry = Registry() if registry is None else registry
        strategy = self.strategy_class(backend, mapping_source_provider=lambda: source,
            registry=registry, clock=lambda: clock.now)
        backend._motion_strategy = strategy
        return NS(backend=backend, state=state, target=target, strategy=strategy,
                  holder=holder, clock=clock, source=source, registry=registry)

    async def test_invalid_robot_state_fails_before_ik_or_gripper_reads(self):
        c = self.context()
        c.backend.robot.handles_initialized = False
        c.backend.check_reachability = Mock(side_effect=AssertionError('invalid IK state'))
        c.backend.gripper_feedback = Mock(side_effect=AssertionError('invalid gripper state'))
        result = await c.backend.move_to_pose_async(c.target)
        self.assertFalse(result.ok)
        self.assertTrue(result.details['world_state_unavailable'])
        self.assertIn('reinitialize', result.message)
        c.backend.robot.apply_action.assert_not_called()
        c.backend.check_reachability.assert_not_called()
        c.backend.gripper_feedback.assert_not_called()

    def assert_hold(self, context):
        fixtures.FrankaCommandTests().assert_arm_hold(context.backend)
        self.assertFalse(context.backend._motion_active)

    async def test_success_preserves_request_tolerances_yields_callback_order_and_final_hold(self):
        c = self.context()
        c.target = Pose(c.target.position, [0., 0., 1., 0.])
        c.state.position = Pose([0., 0., 0.], c.target.orientation)
        update = c.backend._next_update
        events = []
        async def next_update():
            await update()
            events.append(('update', c.state.updates))
            if c.state.updates >= 2:
                c.state.position = c.target
        c.backend._next_update = next_update
        def guard():
            events.append(('guard', c.state.updates))
        result = await c.backend.move_to_pose_async(c.target, .007, .03, guard)
        self.assertTrue(result.ok)
        self.assertEqual(events, [('update', 1), ('guard', 1), ('update', 2), ('guard', 2)])
        self.assertEqual(result.details['position_tolerance_m'], .007)
        self.assertEqual(result.details['orientation_tolerance_rad'], .03)
        self.assertEqual(result.details['map_timestamp'], 99.5)
        self.assertEqual(result.details['map_generation'], 0)
        self.assertEqual(result.details['map_revision'], 1)
        self.assertEqual(c.backend.robot.apply_action.call_count, 3)
        c.backend.world.step.assert_not_called()
        self.assert_hold(c)
        self.assertEqual(c.registry.calls[-1], ('clear',))

    async def test_frame_ik_and_unavailable_mapping_reject_before_controller_commands(self):
        c = self.context()
        bad_frame = await c.backend.move_to_pose_async(Pose(c.target.position, frame='robot'))
        self.assertIn('Unsupported frame', bad_frame.message)
        c.backend.check_reachability = lambda target: False
        bad_ik = await c.backend.move_to_pose_async(c.target)
        self.assertIn('IK solution', bad_ik.message)
        c.source.snapshot.assert_not_called()
        c.backend.check_reachability = lambda target: True
        c.strategy._mapping_source_provider = None
        missing = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(missing.details['world_state_unavailable'])
        c.backend.motion_controller.reset.assert_not_called()
        c.backend.robot.apply_action.assert_not_called()

    async def test_initial_waiting_stale_failed_closed_future_and_mismatched_sources_never_fallback(self):
        variants = [view(timestamp=None, status='waiting', fresh=False),
            view(timestamp=98.), view(timestamp=101.), view(status='error', fresh=False),
            view(status='closed', fresh=False), view(status='no_frame', fresh=False),
            replace(view(), obstacles=None),
            replace(view(), obstacles=replace(view().obstacles, generation=1)),
            replace(view(), obstacles=replace(view().obstacles, revision=2)),
            replace(view(), obstacles=replace(view().obstacles, timestamp=99.6))]
        for source_view in variants:
            c = self.context()
            c.holder.view = source_view
            with self.subTest(view=source_view):
                result = await c.backend.move_to_pose_async(c.target)
                self.assertFalse(result.ok)
                self.assertTrue(result.details['world_state_unavailable'])
                c.backend.robot.apply_action.assert_not_called()
                c.backend.motion_controller.forward.assert_not_called()
                self.assertFalse(c.backend._motion_active)

    async def test_held_or_uncertain_gripper_is_explicitly_unsupported(self):
        for state in (GripperState.HELD, GripperState.UNCERTAIN):
            c = self.context()
            c.backend.gripper_feedback.return_value = GripperFeedback(state, .05)
            result = await c.backend.move_to_pose_async(c.target)
            self.assertFalse(result.ok)
            self.assertTrue(result.details['unsupported_motion'])
            self.assertIn('not modeled', result.message)
            c.backend.robot.apply_action.assert_not_called()
            c.backend.motion_controller.reset.assert_not_called()

    async def test_timeout_preserves_budget_and_exact_hold(self):
        c = self.context(reaches=False)
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.details['max_steps'], 3)
        self.assertEqual(c.state.updates, 3)
        self.assertEqual(c.backend.robot.apply_action.call_count, 4)
        self.assert_hold(c)

    async def test_source_failure_staleness_generation_and_revision_regression_after_update_hold_immediately(self):
        variants = [('error', view(status='error', fresh=False), 100.),
            ('stale', view(), 101.), ('generation', view(generation=1), 100.),
            ('revision', view(revision=0), 100.), ('future', view(timestamp=100.1), 100.)]
        for label, replacement, now in variants:
            c = self.context()
            update = c.backend._next_update
            async def next_update():
                await update()
                c.holder.view, c.clock.now = replacement, now
            c.backend._next_update = next_update
            with self.subTest(condition=label):
                result = await c.backend.move_to_pose_async(c.target)
                self.assertFalse(result.ok)
                self.assertTrue(result.details['world_state_unavailable'])
                self.assertEqual(c.backend.motion_controller.forward.call_count, 1)
                self.assertEqual(c.backend.robot.apply_action.call_count, 2)
                self.assert_hold(c)
                self.assertEqual(c.registry.calls[-1], ('clear',))

    async def test_frozen_no_frame_can_reuse_original_evidence_only_inside_freshness_budget(self):
        c = self.context()
        update = c.backend._next_update
        async def next_update():
            await update()
            c.holder.view = view(status='no_frame', fresh=False)
            c.clock.now = 100.2
        c.backend._next_update = next_update
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.ok)
        self.assertEqual(result.details['map_timestamp'], 99.5)
        self.assertAlmostEqual(result.details['map_age_seconds'], .7)
        self.assert_hold(c)

    async def test_new_revision_is_applied_on_next_step_without_waiting_for_a_worker(self):
        c = self.context()
        update = c.backend._next_update
        async def next_update():
            await update()
            c.holder.view = view(revision=2, timestamp=99.7)
        c.backend._next_update = next_update
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.ok)
        synced = [row[1] for row in c.registry.calls if row[0] == 'sync']
        self.assertEqual([row.revision for row in synced], [1, 2])
        self.assertEqual(result.details['map_revision'], 2)
        self.assertEqual(result.details['map_timestamp'], 99.7)

    async def test_original_cell_expiry_removes_proxy_while_global_map_is_still_fresh(self):
        policy = Policy()
        registry = RmpflowObstacleRegistry(policy, meters_per_unit=1.)
        c = self.context(registry=registry)
        boxes = (ObstacleBox('A', (.4, 0., .3), (.1, .1, .1), 100.05, 8),
                 ObstacleBox('B', (.8, 0., .3), (.1, .1, .1), 105., 8))
        c.holder.view = view(timestamp=99.7, boxes=boxes)
        policy.set_robot_base_pose = Mock()
        c.backend.motion_controller.get_motion_policy = lambda: policy
        update = c.backend._next_update
        counts = []
        original_forward = c.backend.motion_controller.forward
        def forward(**kwargs):
            counts.append(len(policy.registered))
            return original_forward.return_value
        c.backend.motion_controller.forward = forward
        async def next_update():
            await update()
            c.clock.now = 100.1
        c.backend._next_update = next_update
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.ok)
        self.assertEqual(counts, [2, 1])
        self.assertEqual(policy.registered, {})  # Remaining proxy removed after final hold.

    async def test_grasp_failure_returns_original_result_after_yield_and_before_tcp_feedback(self):
        c = self.context()
        failure = BackendResult(False, 'Object lost.', details={'grasp_failure': True})
        guard = Mock(return_value=failure)
        c.backend.get_end_effector_pose = Mock(side_effect=AssertionError('feedback before grasp check'))
        result = await c.backend.move_to_pose_async(c.target, grasp_check=guard)
        self.assertIs(result, failure)
        self.assertEqual(c.state.updates, 1)
        guard.assert_called_once_with()
        c.backend.get_end_effector_pose.assert_not_called()
        self.assert_hold(c)

    async def test_stop_before_or_after_update_fails_without_extra_progress(self):
        c = self.context(playing=False)
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.details['world_state_unavailable'])
        c.backend.robot.apply_action.assert_not_called()
        c = self.context()
        update = c.backend._next_update
        async def stopped():
            await update()
            c.state.position = c.target  # Even reaching the goal cannot hide the stop.
            c.backend.world.is_playing = lambda: False
        c.backend._next_update = stopped
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.details['world_state_unavailable'])
        self.assertEqual(c.backend.motion_controller.forward.call_count, 1)
        self.assert_hold(c)

    async def test_actual_task_cancellation_preserves_hold_cleanup_and_backend_ownership(self):
        c = self.context()
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked_update():
            entered.set()
            await release.wait()
        c.backend._next_update = blocked_update
        task = asyncio.create_task(c.backend.move_to_pose_async(c.target))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_hold(c)
        self.assertEqual(c.registry.calls[-1], ('clear',))
        self.assertEqual(c.strategy.diagnostics['state'], 'cancelled')

    async def test_sdk_exception_after_an_action_keeps_original_exception_despite_cleanup_failure(self):
        c = self.context()
        update = c.backend._next_update
        async def failed_update():
            await update()
            raise RuntimeError('original Kit update exception')
        c.backend._next_update = failed_update
        c.registry.clear = Mock(side_effect=RuntimeError('secondary registry cleanup exception'))
        with self.assertRaisesRegex(RuntimeError, 'original Kit update exception') as caught:
            await c.backend.move_to_pose_async(c.target)
        self.assertTrue(any('secondary registry cleanup' in note for note in caught.exception.__notes__))
        self.assert_hold(c)
        c.registry.clear.assert_called_once_with()

    async def test_controller_exception_after_first_action_holds_and_unregisters(self):
        c = self.context()
        action = c.backend.motion_controller.forward.return_value
        c.backend.motion_controller.forward.side_effect = [action, RuntimeError('controller failed')]
        with self.assertRaisesRegex(RuntimeError, 'controller failed'):
            await c.backend.move_to_pose_async(c.target)
        self.assert_hold(c)
        self.assertEqual(c.registry.calls[-1], ('clear',))

    async def test_source_access_exception_is_descriptive_and_recovered_source_accepts_new_request(self):
        c = self.context()
        c.source.snapshot.side_effect = RuntimeError('camera worker unavailable')
        failed = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(failed.details['world_state_unavailable'])
        self.assertIn('camera worker unavailable', failed.message)
        c.source.snapshot.side_effect = lambda: c.holder.view
        self.assertTrue((await c.backend.move_to_pose_async(c.target)).ok)
        self.assert_hold(c)

    async def test_close_is_idempotent_and_prevents_further_motion(self):
        c = self.context()
        c.strategy.close()
        c.strategy.close()
        self.assertEqual(c.registry.calls, [('close',), ('clear',), ('close',), ('clear',)])
        self.assertTrue(c.registry.closed)
        self.assertEqual(c.registry.count, 0)
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.details['world_state_unavailable'])
        self.assertIn('closed', result.message)
        c.backend.robot.apply_action.assert_not_called()

    async def test_failed_close_stays_closed_to_motion_and_retries_owned_cleanup(self):
        c = self.context()
        c.registry.count = 1
        original_close = c.registry.close
        attempts = []

        def fails_once():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError('temporary SDK obstacle removal failure')
            original_close()

        c.registry.close = fails_once
        with self.assertRaisesRegex(RuntimeError, 'temporary SDK obstacle removal failure'):
            c.strategy.close()
        self.assertTrue(c.strategy.diagnostics['closed'])
        self.assertEqual(c.registry.count, 1)
        result = await c.backend.move_to_pose_async(c.target)
        self.assertTrue(result.details['world_state_unavailable'])
        c.backend.robot.apply_action.assert_not_called()
        c.strategy.close()
        self.assertEqual(len(attempts), 2)
        self.assertTrue(c.registry.closed)
        self.assertEqual(c.registry.count, 0)


if __name__ == '__main__':
    unittest.main()
