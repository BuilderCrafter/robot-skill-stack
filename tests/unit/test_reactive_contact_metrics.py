"""Physical units and missing-data behavior for observational severity metrics."""
import unittest
from types import SimpleNamespace

import numpy as np

from tests.support.reactive_contact_metrics import (
    analyze_contact_samples, analyze_joint_feedback, capture_joint_feedback,
    contact_point_velocity)


def sample(step, *, separation=.01, impulse=0., contact=True):
    return dict(physics_step=step, physics_dt_s=.01, simulation_time=step*.01,
        poses=[[0., 0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0., 1.]],
        body_velocities=[[.2, 0., 0., 0., 0., 0.], [.1, 0., 0., 0., 0., 0.]],
        contacts=[] if not contact else [dict(sensor='/robot', obstacle='/box', count=1,
            normal_impulses_ns=[impulse], separations_m=[separation],
            points_world_m=[[.5, 0., 0.]], normals_world=[[1., 0., 0.]])])


def analyze(samples, **kwargs):
    return analyze_contact_samples(samples, ['/robot', '/box'], np.zeros((2, 3)), **kwargs)


class ReactiveContactMetricsTests(unittest.TestCase):
    def test_rotating_body_uses_transformed_com_not_body_origin(self):
        pose=np.eye(4)
        pose[:3, :3]=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        pose[:3, 3]=[3., 4., 0.]
        # Local COM [1,0,0] -> world [3,5,0]; point [4,5,0] is 1m from COM.
        actual=contact_point_velocity([[4., 5., 0.]], pose, [1., 0., 0.], [1., 0., 0., 0., 0., 2.])
        np.testing.assert_allclose(actual, [[1., 2., 0.]])

    def test_relative_speed_force_units_and_duplicate_hold_sample(self):
        first=sample(0, contact=False)
        second=sample(1, separation=-.0008, impulse=.02)
        result=analyze([first, second, second], minimum_clearance_m=0., strict_v2_success=False)
        self.assertEqual(result['duplicate_samples_removed'], 1)
        self.assertEqual(result['reported_points'], 1)
        self.assertAlmostEqual(result['normal_pair_step_average_force_n']['maximum'], 2.)
        self.assertAlmostEqual(result['relative_contact_point_speed_m_s']['maximum'], .1)
        self.assertAlmostEqual(result['robot_contact_point_speed_m_s']['maximum'], .2)
        self.assertAlmostEqual(result['depth_m']['maximum'], .0008)
        self.assertEqual(result['strict_v2_success'], False)
        self.assertEqual(result['physical_classification'], 'measured_physical_contact_ungraded')
        self.assertIsNone(result['severity_pass'])

    def test_skipped_physics_steps_do_not_invent_average_force_or_contact_duration(self):
        result=analyze([sample(0, contact=False), sample(2, impulse=.02)])
        self.assertEqual(result['normal_pair_step_average_force_n']['count'], 0)
        self.assertEqual(result['force_unavailable_reasons']['physics_steps_skipped_or_repeated'], 1)
        self.assertEqual(result['sampled_reported_contact_duration_s'], 0.)

    def test_sdk_dt_must_agree_with_simulation_time_not_wall_time(self):
        second=sample(1, impulse=.02)
        second['simulation_time']=.5
        second['at']=1000000.
        result=analyze([sample(0, contact=False), second])
        self.assertEqual(result['normal_pair_step_average_force_n']['count'], 0)
        self.assertIn('simulation_time_does_not_match_sdk_physics_dt', result['force_unavailable_reasons'])

    def test_proximity_is_separate_from_strict_zero_and_source_failure(self):
        result=analyze([sample(0), sample(1)], minimum_clearance_m=.01,
            strict_v2_success=False, goal_status='success')
        self.assertEqual(result['physical_classification'], 'contact_offset_proximity')
        self.assertFalse(result['strict_zero_reported_contacts'])
        self.assertFalse(result['strict_v2_success'])
        self.assertIsNone(result['severity_pass'])
        aborted=analyze([sample(0, separation=-.005, impulse=.3)],
            goal_status='failed', failure_code='world_state_unavailable', minimum_clearance_m=0.)
        self.assertEqual(aborted['failure_code'], 'world_state_unavailable')
        self.assertEqual(aborted['physical_classification'], 'measured_physical_contact_ungraded')

    def test_reference_disagreement_and_saturation_cannot_claim_clear_contact(self):
        result=analyze([sample(0, contact=False)], minimum_clearance_m=0.)
        self.assertEqual(result['physical_classification'], 'reference_contact_disagreement')
        full=sample(0)
        full['contact_buffer_full']=True
        self.assertEqual(analyze([full])['physical_classification'], 'contact_buffer_saturated_severity_incomplete')

    def test_old_v2_pair_impulse_remains_raw_without_invented_kinematics(self):
        legacy=dict(simulation_time=0., contacts=[dict(sensor='/robot', count=4,
            minimum_separation_m=-.002, impulse_magnitude=.1)])
        result=analyze([legacy])
        self.assertEqual(result['reported_points'], 4)
        self.assertEqual(result['normal_pair_impulse_ns']['maximum'], .1)
        self.assertEqual(result['normal_point_impulse_ns']['count'], 0)
        self.assertEqual(result['robot_contact_point_speed_m_s']['count'], 0)
        self.assertEqual(result['normal_pair_step_average_force_n']['count'], 0)

    def test_joint_feedback_is_read_only_and_unavailable_loads_are_not_zero(self):
        class Robot:
            def __init__(self):
                self.positions=np.array([.1]*9)
                self.action=SimpleNamespace(joint_positions=[.2, .3], joint_velocities=None,
                    joint_efforts=None, joint_indices=[0, 7])
            def get_joint_positions(self): return self.positions
            def get_joint_velocities(self): return np.ones(9)
            def get_measured_joint_efforts(self): return np.arange(9.)
            def get_applied_joint_efforts(self): return np.zeros(9)
            def get_measured_joint_forces(self): raise RuntimeError('invalid handles')
            def get_applied_action(self): return self.action
        robot=Robot()
        feedback=capture_joint_feedback(robot, controller_velocity_feedback=np.zeros(7))
        robot.positions[:]=9.
        self.assertEqual(feedback['positions'], [.1]*9)
        self.assertIsNone(feedback['incoming_joint_wrenches'])
        self.assertIn('incoming_joint_wrenches', feedback['errors'])
        metrics=analyze_joint_feedback([dict(joint_feedback=feedback)])
        self.assertEqual(metrics['arm_position_tracking_error_rad']['count'], 1)
        self.assertAlmostEqual(metrics['arm_position_tracking_error_rad']['maximum'], .1)
        self.assertEqual(metrics['arm_solver_projected_effort_abs_nm']['maximum'], 6.)
        self.assertFalse(metrics['contact_detector_validated'])


if __name__ == '__main__':
    unittest.main()
