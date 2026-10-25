"""Observation validity must not become a synthetic motion success."""
import asyncio
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.manipulation.backend import BackendResult
from scripts.diagnose_global_stabilization_v4_1 import (
    ARM, DIFFERENT_GOAL, MODES, arm_target_error, callback_for, cancel_start,
    controlled_acceptance, dynamics_feedback, reduce_observations,
)


class ControlledObservationTests(unittest.TestCase):
    def test_ahead_hold_target_has_negative_actual_minus_target_sign(self):
        positions = np.arange(9, dtype=float)
        targets = positions.copy(); targets[0] += .03; targets[8] += 7.
        error = arm_target_error(positions, dict(joint_positions=targets, joint_indices=None), tuple(range(7)))
        np.testing.assert_allclose(error, [-.03, 0, 0, 0, 0, 0, 0])
        partial = dict(joint_positions=[positions[3]+.2, positions[0]+.1], joint_indices=[3, 0])
        np.testing.assert_allclose(arm_target_error(positions, partial, (0, 3)), [-.1, -.2])
        self.assertIsNone(arm_target_error(None, partial, (0, 3)))
        self.assertIsNone(arm_target_error(positions, partial, (0, 1, 3)))

    def test_modes_keep_home_and_post_cancel_other_goal_distinct(self):
        self.assertFalse(MODES['B']['cancel'])
        self.assertTrue(MODES['C']['cancel'])
        self.assertEqual(MODES['C']['followup'], 'home')
        self.assertEqual(MODES['D']['followup'], 'different_goal')
        self.assertEqual(MODES['E']['strategy'], 'direct')
        self.assertEqual(MODES['E_cancel']['strategy'], 'direct')
        self.assertNotEqual(DIFFERENT_GOAL, (.45, -.25, .35))

    def test_native_drive_type_fallback_preserves_real_enum_values(self):
        native = SimpleNamespace(get_drive_types=lambda: np.array([[1]*7+[1, 0]]),
            get_dof_drive_model_properties=lambda: np.zeros((1, 9, 3)))
        robot = SimpleNamespace(_articulation_view=SimpleNamespace(_physics_view=native))
        values = dynamics_feedback(robot)
        self.assertTrue(values['get_drive_types']['available'])
        self.assertEqual(values['get_drive_types']['value'], [[1]*8+[0]])
        self.assertEqual(values['get_drive_types']['source'], 'public_tensor_view.get_drive_types')

    def test_acceptance_requires_every_expected_operation_and_strict_outcome(self):
        rows = [dict(label=name, strict_success=True) for name in ('setup_start', 'move_to_goal', 'home')]
        self.assertTrue(controlled_acceptance('B', rows))
        self.assertFalse(controlled_acceptance('B', rows[:-1]))
        failed = copy.deepcopy(rows); failed[-1]['strict_success'] = False
        self.assertFalse(controlled_acceptance('B', failed))
        cancelled = dict(label='cancel_start', cancelled=True, ownership_released=True,
            overlap_rejected=True, final_applied_arm_velocity_zero=True,
            contacts=dict(reported_points=0), clearance=dict(minimum_m=.02))
        self.assertTrue(controlled_acceptance('C', [*rows, cancelled]))
        self.assertFalse(controlled_acceptance('C', [*rows, {**cancelled, 'ownership_released': False}]))
        self.assertFalse(controlled_acceptance('C', [*rows, {**cancelled, 'final_applied_arm_velocity_zero': False}]))

    def test_reduction_keeps_tiny_physical_contact_and_proximity_separate(self):
        operation = dict(id=0, label='home', kind='motion', cancelled=False,
            returned_success=True, measured_full_pose_within_tolerances=True,
            first_sample=0, last_sample=1, first_action=0, last_action=1,
            before=dict(simulation_time=0., physics_step=0, actual_minus_applied_target_arm_rad=[0.]*7),
            after=dict(simulation_time=.1, physics_step=1, actual_minus_applied_target_arm_rad=[0.]*7,
                live_drive_feedback=dict(robot_applied_action=dict(available=True,
                    value=dict(joint_velocities=[0.]*9)))), strategy_diagnostics={})
        pair = dict(sensor='/Franka/panda_hand', obstacle='/World/Post', count=1,
            minimum_separation_m=-1e-12, impulse_magnitude=0.,
            separations_m=[-1e-12], normal_impulses_ns=[0.], points_world_m=[[0., 0, 0]],
            normals_world=[[1., 0, 0]])
        sample = dict(simulation_time=.1, physics_step=1, physics_dt_s=.1,
            poses=[[0., 0, 0, 0, 0, 0, 1]], body_velocities=[[0.]*6], contacts=[pair])
        observer = SimpleNamespace(operations=[operation], samples=[sample], indices=tuple(range(7)),
            metadata=dict(body_paths=['/Franka/panda_hand'], local_coms=[[0., 0, 0, 0, 0, 0, 1]]))
        capture = dict(pose_matrices=lambda poses, units: np.array([np.eye(4)]), meshes=[],
            snapshot=SimpleNamespace(meters_per_unit=1., obstacles=[]))
        with patch('scripts.diagnose_global_stabilization_v4_1.fixture.reduce_clearances',
                   return_value=dict(minimum_m=.01)):
            physical = reduce_observations(observer, capture)[0]
            self.assertEqual(physical['observed_physical_contact_steps'], 1)
            self.assertFalse(physical['strict_success'])
            pair.update(minimum_separation_m=.02, separations_m=[.02])
            proximity = reduce_observations(observer, capture)[0]
            self.assertEqual(proximity['observed_physical_contact_steps'], 0)
            self.assertEqual(proximity['contacts']['reported_points'], 1)
            self.assertFalse(proximity['strict_success'])


class ControlledSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_home_does_not_invalidate_completed_observation_callback(self):
        observer = SimpleNamespace(operations=[dict(label='move_to_goal', returned_success=True,
            measured_full_pose_within_tolerances=True)])
        execute = AsyncMock(return_value=BackendResult(False, 'tracking failure'))
        state = {'observer': observer}
        callback = callback_for('B', state)
        result = await callback(None, SimpleNamespace(runtime=SimpleNamespace(execute_async=execute)), {}, Pose)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(state['callback_complete'])
        execute.assert_awaited_once_with('home')

    async def test_d_uses_other_full_pose_after_cancel_without_changing_home(self):
        observer = SimpleNamespace(operations=[dict(label='move_to_goal', returned_success=True,
            measured_full_pose_within_tolerances=True)])
        execute = AsyncMock(return_value=BackendResult(True))
        with patch('scripts.diagnose_global_stabilization_v4_1.cancel_start', new_callable=AsyncMock) as cancel:
            await callback_for('D', {'observer': observer})(None,
                SimpleNamespace(runtime=SimpleNamespace(execute_async=execute)), {}, Pose)
        cancel.assert_awaited_once_with(observer, Pose)
        self.assertEqual(execute.await_args.args, ('move_to_pose',))
        np.testing.assert_array_equal(execute.await_args.kwargs['target'].position, DIFFERENT_GOAL)
        np.testing.assert_array_equal(execute.await_args.kwargs['target'].orientation, [0., 0, 1., 0])

    async def test_failed_primary_never_runs_unmatched_followup(self):
        observer = SimpleNamespace(operations=[dict(label='move_to_goal', returned_success=False,
            measured_full_pose_within_tolerances=False)])
        execute = AsyncMock()
        state = {'observer': observer}
        with patch('scripts.diagnose_global_stabilization_v4_1.cancel_start', new_callable=AsyncMock) as cancel:
            await callback_for('C', state)(None, SimpleNamespace(runtime=SimpleNamespace(execute_async=execute)), {}, Pose)
        cancel.assert_not_awaited(); execute.assert_not_awaited()
        self.assertTrue(state['followup_precondition_failed'])

    async def test_cancellation_is_observed_before_followup_and_releases_owner(self):
        observer = SimpleNamespace(actions=[], backend=None)
        backend = SimpleNamespace(_motion_active=False)
        async def move(target, **kwargs):
            if backend._motion_active:
                return BackendResult(False, 'Arm motion is already in progress.')
            backend._motion_active = True
            observer.actions.append('first command')
            try:
                await asyncio.Future()
            finally:
                observer.actions.append('zero velocity hold')
                backend._motion_active = False
        backend.move_to_pose_async = move
        backend._next_update = lambda: asyncio.sleep(0)
        observer.backend = backend
        observer.begin_operation = lambda *args, **kwargs: {}
        observer.state = lambda: dict(joints=[.1]*7)
        observer.end_operation = lambda row, result, cancelled: row.update(cancelled=cancelled)
        row = await cancel_start(observer, Pose)
        self.assertTrue(row['first_command_observed'])
        self.assertTrue(row['active_before_cancel'])
        self.assertTrue(row['overlap_rejected'])
        self.assertTrue(row['cancelled'])
        self.assertTrue(row['ownership_released'])
        self.assertEqual(observer.actions, ['first command', 'zero velocity hold'])


if __name__ == '__main__':
    unittest.main()
