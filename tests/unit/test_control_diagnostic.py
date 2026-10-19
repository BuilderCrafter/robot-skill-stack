"""Validate diagnostic forwarding and cleanup without loading native Isaac."""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from tests.unit.test_regression_diagnostics import scenario
from tests.support.control_trace import record_control


class ControlDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def stack(self):
        module = scenario('diagnose_isaac_control', folder='scripts')
        module.record_control = record_control
        world = NS(current_time=0., current_time_step_index=0,
                   step=Mock(side_effect=AssertionError('unexpected blocking world.step')))
        action = NS(joint_positions=np.array([.1, .2]), joint_velocities=np.array([.3, .4]),
                    joint_indices=np.array([0, 1]))
        robot = NS(get_joint_positions=Mock(return_value=np.array([0., 0.])),
                   get_joint_velocities=Mock(return_value=np.array([.3, .4])),
                   get_applied_action=Mock(return_value=action), apply_action=Mock())
        backend = NS(world=world, robot=robot, motion_controller=NS(forward=Mock(return_value=action)),
                     get_end_effector_pose=Mock(return_value=Pose([.45, 0, .25], [0, 0, 1, 0])))
        callbacks = {}
        scheduler = NS(add_physics_callback=Mock(side_effect=lambda name, callback_fn: callbacks.update({name: callback_fn})),
                       remove_physics_callback=Mock(side_effect=lambda name: callbacks.pop(name)))
        async def update():
            await asyncio.sleep(0)
            for callback in callbacks.values():
                callback(1/60)
            world.current_time_step_index += 1
            world.current_time += 1/60
        backend._next_update = AsyncMock(side_effect=update)
        async def move(target, **kwargs):
            result = backend.motion_controller.forward(target_end_effector_position=target.position,
                                                       target_end_effector_orientation=target.orientation)
            backend.robot.apply_action(result)
            await backend._next_update()
            return BackendResult(True)
        async def gripper():
            backend.robot.apply_action(action)
            await backend._next_update()
            return BackendResult(True)
        backend.move_to_pose_async = AsyncMock(side_effect=move)
        backend.open_gripper_async = AsyncMock(side_effect=gripper)
        backend.close_gripper_async = AsyncMock(side_effect=gripper)
        return module, backend, scheduler

    async def test_recorder_preserves_actions_wait_count_and_event_order(self):
        module, backend, scheduler = self.stack()
        forward, apply, wait = backend.motion_controller.forward, backend.robot.apply_action, backend._next_update
        samples = []
        with module.record_control(backend, scheduler, samples) as set_stage:
            set_stage('move')
            result = backend.motion_controller.forward(target_end_effector_position=[.45, 0, .25])
            backend.robot.apply_action(result)
            await backend._next_update()
        self.assertIs(backend.motion_controller.forward, forward)
        self.assertIs(backend.robot.apply_action, apply)
        self.assertIs(backend._next_update, wait)
        forward.assert_called_once_with(target_end_effector_position=[.45, 0, .25])
        apply.assert_called_once_with(result)
        wait.assert_awaited_once_with()
        self.assertEqual([s['event'] for s in samples],
                         ['before_forward', 'after_forward', 'before_apply', 'physics_callback', 'after_update'])
        self.assertEqual(samples[-1]['physics_step'], 1)
        self.assertEqual(samples[-1]['applied_action']['joint_velocities'], [.3, .4])
        self.assertTrue(all(s['stage'] == 'move' for s in samples))
        scheduler.remove_physics_callback.assert_called_once_with('control_diagnostic')
        backend.world.step.assert_not_called()

    async def test_empty_gripper_sequence_records_idle_without_extra_actions(self):
        module, backend, scheduler = self.stack()
        metrics = {'samples': [], 'stages': []}
        await module.run_diagnostic(backend, scheduler, metrics)
        self.assertEqual([s['stage'] for s in metrics['stages']],
                         ['hover', 'open_empty', 'close_empty', 'side', 'idle_after_side', 'back', 'idle_after_back'])
        self.assertEqual(backend.move_to_pose_async.await_count, 3)
        self.assertEqual(backend.robot.apply_action.call_count, 5)
        self.assertEqual(backend.world.current_time_step_index, 125)  # 5 fake commands + 120 idle updates.
        self.assertTrue(all(s['event'] in ('physics_callback', 'after_update')
                            for s in metrics['samples'] if s['stage'].startswith('idle_')))
        for call in backend.move_to_pose_async.await_args_list:
            self.assertEqual(call.args[0].position[2], .25)
            self.assertEqual(call.kwargs['orientation_tolerance'], .10)

    async def test_loaded_trace_records_true_cube_motion_and_positional_controller_arguments(self):
        module, backend, scheduler = self.stack()
        cube = NS(get_world_pose=lambda: (np.array([.4, 0, .025+.1*backend.world.current_time_step_index]),
                                          np.array([1, 0, 0, 0])))
        position, orientation = np.array([.4, 0, .15]), np.array([0, 0, 1, 0])
        samples = []
        with module.record_control(backend, scheduler, samples, cube=cube) as set_stage:
            set_stage('trial_3:pick')
            backend.motion_controller.forward(position, orientation)
            await backend._next_update()
        backend.motion_controller.forward.assert_called_once_with(position, orientation)
        self.assertEqual(samples[1]['target_position'], position.tolist())
        self.assertEqual(samples[1]['target_orientation'], orientation.tolist())
        self.assertEqual(samples[0]['cube_position'], [.4, 0, .025])
        self.assertEqual(samples[-1]['cube_position'], [.4, 0, .125])
        self.assertTrue(all(s['stage'] == 'trial_3:pick' for s in samples))

    async def test_partial_gripper_action_and_retained_arm_velocity_targets_are_distinguishable(self):
        module, backend, scheduler = self.stack()
        fingers = NS(joint_positions=np.array([.04, .04]), joint_velocities=None,
                     joint_indices=np.array([7, 8]))
        samples = []
        with module.record_control(backend, scheduler, samples) as set_stage:
            set_stage('open_empty')
            backend.robot.apply_action(fingers)
            await backend._next_update()
        self.assertEqual(samples[0]['action']['joint_indices'], [7, 8])
        self.assertIsNone(samples[0]['action']['joint_velocities'])
        self.assertEqual(samples[-1]['applied_action']['joint_velocities'], [.3, .4])
        backend.robot.apply_action.assert_called_once_with(fingers)

    async def test_failed_hover_stops_before_gripper_commands_and_retains_failure(self):
        module, backend, scheduler = self.stack()
        backend.move_to_pose_async.side_effect = None
        backend.move_to_pose_async.return_value = BackendResult(False, 'orientation timeout', timed_out=True)
        metrics = {'samples': [], 'stages': []}
        with self.assertRaisesRegex(RuntimeError, 'hover: orientation timeout'):
            await module.run_diagnostic(backend, scheduler, metrics)
        self.assertTrue(metrics['stages'][0]['result']['timed_out'])
        backend.open_gripper_async.assert_not_called()
        backend.close_gripper_async.assert_not_called()
        scheduler.remove_physics_callback.assert_called_once()

    async def test_callback_error_surfaces_and_cancellation_restores_wrappers(self):
        for failure in (RuntimeError('feedback unavailable'), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                module, backend, scheduler = self.stack()
                original = backend._next_update
                with self.assertRaises(type(failure)) as raised:
                    with module.record_control(backend, scheduler, []) as set_stage:
                        set_stage('failure')
                        if isinstance(failure, asyncio.CancelledError):
                            raise failure
                        backend.robot.get_joint_positions.side_effect = failure
                        await backend._next_update()
                self.assertIs(raised.exception, failure)
                self.assertIs(backend._next_update, original)
                scheduler.remove_physics_callback.assert_called_once()
