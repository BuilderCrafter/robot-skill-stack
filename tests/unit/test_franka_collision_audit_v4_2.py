"""Measured-state selection and frame/clearance semantics for the offline audit."""
import unittest

import numpy as np

from scripts.audit_franka_collision_model_v4_2 import (
    measured_finger_pose, physical_clearance, query_timing, representative_indices,
)
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle, StaticWorldSnapshot


class FrankaCollisionAuditTests(unittest.TestCase):
    def test_first_middle_last_are_selected_per_actual_phase_after_deduplication(self):
        samples = [dict(phase=phase, physics_step=step, simulation_time=step/60)
                   for phase, step in [('Home', 0), ('Home', 0), ('Home', 1), ('Home', 2),
                                       ('Low', 3), ('Low', 4), ('Low', 5), ('Low', 6)]]
        self.assertEqual(representative_indices(samples, 3), [0, 2, 3, 4, 6, 7])
        with self.assertRaises(ValueError):
            representative_indices(samples, 0)

    def test_measured_finger_correction_uses_world_rotated_prismatic_direction(self):
        native = np.eye(4)
        native[:3, 3] = [1., 2., 3.]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        corrected = measured_finger_pose(native, rotation, [0., -1., 0.], .002, .025)
        np.testing.assert_allclose(corrected[:3, 3], [.977, 2., 3.])
        np.testing.assert_array_equal(native[:3, 3], [1., 2., 3.])

    def test_physical_plane_penetration_is_signed_and_cannot_look_clear(self):
        part = dict(body='/Franka/panda_hand', vertices=np.array([[0., 0., -.001], [0., 0., .1]]))
        plane = StaticObstacle('/Ground', 'cuboid', (0., 0., -.5), dimensions=(10., 10., 1.),
            source_kind='Plane', approximation='plane_halfspace_clipped_to_certified_domain')
        snapshot = StaticWorldSnapshot((plane,), 0., 1., '/Franka')
        result = physical_clearance([part], {'/Franka/panda_hand': np.eye(4)}, snapshot)
        self.assertAlmostEqual(result['minimum_m'], -.001)
        self.assertEqual(result['closest_pair']['body'], '/Franka/panda_hand')
        self.assertIsNone(physical_clearance([part], {}, None)['minimum_m'])

    def test_query_timing_is_explicitly_optional_and_bounded(self):
        self.assertEqual(query_timing(None, np.zeros(7), 0)['status'], 'NOT_MEASURED')
        with self.assertRaises(ValueError):
            query_timing(None, np.zeros(7), 102)


if __name__ == '__main__':
    unittest.main()
