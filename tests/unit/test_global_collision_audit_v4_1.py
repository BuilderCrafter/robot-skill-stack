import unittest

import numpy as np

from scripts.audit_global_collision_v4_1 import (
    body_mesh_vertices, coverage_statistics, pose_matrix_xyzw, sphere_union_gaps, surface_quadrature,
    unique_state_indices,
)


class CollisionAuditTests(unittest.TestCase):
    def test_equal_area_quadrature_preserves_area_and_linear_centroid(self):
        vertices = np.array([[0., 0, 0], [2., 0, 0], [0., 3., 0]])
        for subdivisions in (1, 2, 4, 8):
            points, weights = surface_quadrature(vertices, np.array([[0, 1, 2]]), subdivisions)
            self.assertEqual(len(points), subdivisions**2)
            self.assertAlmostEqual(weights.sum(), 3.)
            self.assertTrue(np.allclose(np.average(points, axis=0, weights=weights), [2/3, 1, 0]))
            self.assertTrue(np.all(points[:, 0] > 0))
            self.assertTrue(np.all(points[:, 1] > 0))
            self.assertTrue(np.all(points[:, 0]/2+points[:, 1]/3 < 1))

    def test_union_coverage_can_come_from_neighboring_body_sphere(self):
        points = np.array([[0., 0, 0], [2., 0, 0]])
        gaps = sphere_union_gaps(points, [[0., 0, 0], [2., 0, 0]], [.5, .5])
        self.assertTrue(np.allclose(gaps, [-.5, -.5]))
        empty = sphere_union_gaps(points, np.empty((0, 3)), [])
        self.assertTrue(np.isinf(empty).all())
        self.assertEqual(coverage_statistics(empty, [1., 1.])["estimated_covered_surface_fraction"], 0.)

    def test_surface_fraction_uses_area_weights_not_point_count(self):
        result = coverage_statistics([-.1, .1], [9., 1.])
        self.assertEqual(result["estimated_covered_surface_fraction"], .9)
        self.assertEqual(result["uncovered_samples"], 1)
        self.assertEqual(result["maximum_sampled_outside_gap_m"], .1)

    def test_cooked_mesh_scale_is_applied_before_surface_area_weights(self):
        transform = np.eye(4); transform[:3, :3] *= .01; transform[:3, 3] = [1., 2, 3]
        physical = body_mesh_vertices([[0., 0, 0], [100., 0, 0], [0., 100., 0]], transform)
        self.assertTrue(np.allclose(physical, [[1, 2, 3], [2, 2, 3], [1, 3, 3]]))
        _, weights = surface_quadrature(physical, np.array([[0, 1, 2]]))
        self.assertAlmostEqual(weights.sum(), .5)

    def test_invalid_geometry_never_creates_a_coverage_claim(self):
        with self.assertRaises(ValueError):
            surface_quadrature([[0., 0, 0]]*3, np.array([[0, 1, 2]]))
        with self.assertRaises(ValueError):
            surface_quadrature([[0., 0, 0]]*3, np.array([[0, 1, 8]]))
        with self.assertRaises(ValueError):
            sphere_union_gaps([[0., 0, 0]], [[0., 0, 0]], [-1.])

    def test_xyzw_pose_uses_actual_physics_quaternion_convention(self):
        value = pose_matrix_xyzw([1, 2, 3, 0, 0, np.sqrt(.5), np.sqrt(.5)])
        self.assertTrue(np.allclose(value[:3, :3]@[1., 0, 0], [0, 1, 0]))
        self.assertTrue(np.allclose(value[:3, 3], [1, 2, 3]))

    def test_physics_states_are_deduplicated_before_fixed_subsampling(self):
        samples = [dict(physics_step=n, simulation_time=n/60) for n in (0, 0, 1, 2, 3, 3)]
        self.assertEqual(unique_state_indices(samples, 3), [0, 3, 4])


if __name__ == "__main__":
    unittest.main()
