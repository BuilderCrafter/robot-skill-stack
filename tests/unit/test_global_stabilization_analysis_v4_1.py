import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_global_stabilization_v4_1 import (
    ARM, control_state_summary, dynamics_summary, hand_fk_summary, joint1_body_summary, joint_axis_from_urdf,
    target_summaries, xyzw_rotation,
)


def qmul(first, last):
    a, b = np.asarray(first[:3]), np.asarray(last[:3])
    return np.r_[first[3]*b+last[3]*a+np.cross(a, b), first[3]*last[3]-a@b]


def quat(axis, angle):
    return np.r_[np.asarray(axis)*math.sin(angle/2), math.cos(angle/2)]


class PhysicalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(robot_dof_names=[*ARM, 'panda_finger_joint1', 'panda_finger_joint2'],
            body_paths=['/Franka/panda_hand', '/Franka/panda_link1', '/Franka/panda_link0'])
        self.proof = dict(joint_name='panda_joint1', parent='panda_link0', child='panda_link1',
            axis_in_parent=[0., 0., 1.])

    def sample(self, physical_angle, native_angle=None, base_angle=0., sign=1.):
        base = quat((1, 0, 0), base_angle)
        child = qmul(base, qmul(quat((0, 0, 1), physical_angle), quat((1, 0, 0), .4)))
        hand = qmul(child, quat((0, 1, 0), .2))
        pose = np.r_[[.2, .3, .4], sign*hand]
        rotation = xyzw_rotation(hand)
        native = physical_angle if native_angle is None else native_angle
        q = [native, *([0.]*8)]
        # A fixed 20mm hand-to-model origin offset must not appear as drift.
        model_position = pose[:3]+rotation@np.array([.02, 0, 0])
        return dict(poses=[pose.tolist(), np.r_[[0., 0., .333], sign*child].tolist(),
            np.r_[[0., 0., 0.], sign*base].tolist()], feedback=dict(positions=q),
            lula_model_fk=dict(frames=dict(panda_hand=dict(available=True,
                value=[model_position.tolist(), rotation.tolist()]))))

    def test_xyzw_rotation_and_sign_are_unambiguous(self):
        self.assertTrue(np.allclose(xyzw_rotation([1., 0, 0, 0]), np.diag([1., -1., -1.])))
        self.assertTrue(np.array_equal(xyzw_rotation([0, 0, 0, 1]), xyzw_rotation([0, 0, 0, -1])))
        with self.assertRaises(ValueError):
            xyzw_rotation([0, 0, 0, 0])

    def test_physical_yaw_strips_base_and_child_frame_offsets(self):
        rows = [self.sample(angle, base_angle=.3*i, sign=(-1.)**i)
            for i, angle in enumerate((.2, .4, -.1))]
        result = joint1_body_summary(rows, self.meta, self.proof)
        self.assertAlmostEqual(result['physical_delta_rad'], -.3)
        self.assertLess(result['maximum_physical_minus_native_delta_rad'], 1e-12)
        self.assertLess(result['maximum_off_axis_rotation_residual_rad'], 1e-7)

    def test_body_angle_unwraps_and_detects_encoder_only_drift(self):
        rows = [self.sample(angle) for angle in (0., 1., 2., 3., 4.)]
        self.assertAlmostEqual(joint1_body_summary(rows, self.meta, self.proof)['physical_delta_rad'], 4.)
        wrong = [self.sample(0., native_angle=x) for x in (0., .1, .4)]
        result = joint1_body_summary(wrong, self.meta, self.proof)
        self.assertAlmostEqual(result['maximum_physical_minus_native_delta_rad'], .4)

    def test_hand_comparison_distinguishes_fixed_frame_offset_from_motion_error(self):
        rows = [self.sample(x, base_angle=x/3) for x in (.1, .3, -.2)]
        result = hand_fk_summary(rows, self.meta, meters_per_unit=1.)
        self.assertAlmostEqual(result['maximum_absolute_position_difference_m'], .02)
        self.assertLess(result['maximum_hand_to_fk_translation_offset_variation_m'], 1e-12)
        rows[-1]['lula_model_fk']['frames']['panda_hand']['value'][0][0] += .1
        self.assertAlmostEqual(hand_fk_summary(rows, self.meta, meters_per_unit=1.)[
            'maximum_hand_to_fk_translation_offset_variation_m'], .1)

    def test_axis_comes_from_urdf_mechanics_and_origin_rotation(self):
        text = '''<robot name="test"><joint name="panda_joint1" type="revolute">
        <parent link="panda_link0"/><child link="panda_link1"/>
        <origin rpy="0 1.5707963267948966 0"/><axis xyz="0 0 1"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'robot.urdf';path.write_text(text)
            proof = joint_axis_from_urdf(path)
            self.assertTrue(np.allclose(proof['axis_in_parent'], [1, 0, 0], atol=1e-12))

    def test_applied_target_mapping_keeps_physical_indices_and_missing_readings(self):
        names = self.meta['robot_dof_names']
        requested = np.arange(7.)
        full = [0.]*len(names)
        for value, index in zip(requested, reversed(range(7))):
            full[index] = value
        action = dict(requested_positions=requested.tolist(), requested_velocities=[0.]*7,
            requested_indices=list(reversed(range(7))), actual_physx_targets=dict(available=True,
                value=dict(joint_positions=full, joint_velocities=[0.]*9, joint_indices=None)))
        sample = dict(feedback=dict(positions=full), live_drive_feedback=dict(
            robot_applied_action=action['actual_physx_targets']))
        result = target_summaries([sample], [action], self.meta)
        self.assertEqual(result['requested_vs_actual_physx_positions_rad']['maximum_absolute'], 0.)
        self.assertEqual(result['measured_vs_latched_physx_position_targets_rad']['maximum_absolute'], 0.)
        result = target_summaries([], [dict(actual_physx_targets=dict(available=False))], self.meta)
        self.assertFalse(result['requested_vs_actual_physx_positions_rad']['available'])

    def test_dynamics_keep_mass_submatrix_coupling_and_unavailable_signals_separate(self):
        matrix = np.eye(9);matrix[0, 2] = matrix[2, 0] = .99
        rows = [dict(dynamics=dict(get_mass_matrices=dict(available=True, value=[matrix.tolist()]),
            get_generalized_gravity_forces=dict(available=False, error='unsupported')))]
        result = dynamics_summary(rows, self.meta)
        self.assertAlmostEqual(result['joint1_joint3_mass_submatrix_condition_number']['maximum'], 199.)
        self.assertAlmostEqual(result['joint1_joint3_mass_correlation']['maximum'], .99)
        self.assertEqual(result['availability_counts']['get_mass_matrices'], 1)
        self.assertIsNone(result['signal_summaries']['get_generalized_gravity_forces'])
        self.assertEqual(result['unavailable_reasons']['get_generalized_gravity_forces'], ['unsupported'])

    def test_control_summary_detects_actual_solver_changes_and_keeps_effort_semantics(self):
        rows = []
        for iterations in (1, 1, 4):
            rows.append(dict(dynamics=dict(
                get_solver_velocity_iteration_counts=dict(available=True, value=[iterations]),
                get_drive_types=dict(available=True, value=[[1]*9])),
                live_drive_feedback=dict(controller_get_gains=dict(available=False, error='unsupported')),
                feedback=dict(explicit_applied_efforts=[0.]*9, measured_efforts=[3.]*9)))
        result = control_state_summary(rows, self.meta)
        native = result['observed_states']
        self.assertEqual(native['get_solver_velocity_iteration_counts']['distinct_states'], 2)
        self.assertFalse(native['get_solver_velocity_iteration_counts']['unchanged'])
        self.assertEqual(native['get_solver_velocity_iteration_counts']['states'], [
            dict(value=[1], samples=2), dict(value=[4], samples=1)])
        self.assertIn('USD attribute', native['get_solver_velocity_iteration_counts']['provenance'])
        self.assertTrue(native['get_drive_types']['unchanged'])
        self.assertIn('Native drive buffer', native['get_drive_types']['provenance'])
        self.assertFalse(native['controller_get_gains']['available'])
        self.assertEqual(native['controller_get_gains']['unavailable_reasons'], ['unsupported'])
        self.assertEqual(result['effort_signals']['explicit_applied_efforts']['maximum_absolute'], 0.)
        self.assertEqual(result['effort_signals']['measured_efforts']['maximum_absolute'], 3.)


if __name__ == '__main__':
    unittest.main()
