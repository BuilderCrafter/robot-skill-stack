"""Exact ray-depth, mask and bounded-work regression tests for packet BVH."""
from unittest.mock import patch
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping import robot_filter as filtering
from tests.unit.test_spatial_mapping_robot_filter import box_mesh, frame, snapshot


class RobotFilterPacketTests(unittest.TestCase):
    def assert_depths_equal(self, mesh, origin, directions, limit):
        expected = filtering._mesh_depth_bvh(mesh, origin, directions, limit)
        actual = filtering._mesh_depth(mesh, origin, directions, limit)
        np.testing.assert_array_equal(actual, expected)
        return actual

    def test_known_surface_hits_misses_and_finite_initial_limits(self):
        mesh = box_mesh(lower=(-1., -1., 1.), upper=(1., 1., 2.))
        directions = np.array([[0., 0., 1.], [1., 1., 1.], [2., 0., 1.],
                               [0., 0., -1.], [0., 0., 0.]])
        limits = np.array([np.inf, np.inf, np.inf, 2., .25])
        actual = self.assert_depths_equal(mesh, np.zeros(3), directions, limits)
        np.testing.assert_array_equal(actual, [1., 1., np.inf, 2., .25])
        np.testing.assert_array_equal(limits, [np.inf, np.inf, np.inf, 2., .25])

    def test_grazing_parallel_inside_origins_and_affine_local_rays(self):
        mesh = box_mesh(lower=(-1., -1., 1.), upper=(1., 1., 2.))
        rng = np.random.default_rng(518)
        targeted = np.array([[0., 0., 1.], [1., 1., 1.], [-1., -1., 1.],
            [1., 0., 0.], [0., 0., -1.], [0., 0., 0.], [-0., 0., 1.],
            [1.+1e-11, 1., 1.], [1.-1e-11, 1., 1.], [1., 1e-15, 1e-15]])
        directions = np.vstack((targeted, rng.normal(size=(128, 3))))
        for origin in (np.zeros(3), np.array([0., 0., 1.5]), np.array([1., 1., 1.])):
            for transform in (np.eye(3), np.diag([.001, 3., 2.]),
                    np.array([[2., .5, 0.], [0., .01, .3], [0., 0., 4.]])):
                inverse = np.linalg.inv(transform)
                for limit in (np.inf, 0., .5, 1.5):
                    with self.subTest(origin=origin, transform=transform, limit=limit):
                        self.assert_depths_equal(mesh, origin @ inverse.T,
                            directions @ inverse.T, np.full(len(directions), limit))

    def test_random_multilevel_mesh_and_degenerate_triangles(self):
        rng = np.random.default_rng(916)
        vertices = rng.uniform([-1., -1., .2], [1., 1., 3.], (360, 3))
        triangles = np.arange(len(vertices)).reshape(-1, 3)
        triangles = np.vstack((triangles, [0, 0, 1]))
        mesh = filtering.TriangleMesh(vertices, triangles)
        directions = rng.normal(size=(513, 3))
        limits = rng.uniform(.1, 4., len(directions))
        limits[::3] = np.inf
        for origin in (np.zeros(3), np.array([.3, -.2, 1.])):
            self.assert_depths_equal(mesh, origin, directions, limits)

    def test_empty_input_and_multiple_root_ray_packets(self):
        mesh = box_mesh()
        self.assert_depths_equal(mesh, np.zeros(3), np.empty((0, 3)), np.empty(0))
        directions = np.tile([0., 0., 1.], (2051, 1))
        limits = np.full(len(directions), np.inf)
        actual = self.assert_depths_equal(mesh, np.zeros(3), directions, limits)
        np.testing.assert_array_equal(actual, np.ones(len(directions)))

    def test_leaf_intersection_temporaries_are_bounded_and_all_hits_retained(self):
        vertices = np.array([[-1., -1., 1.], [1., -1., 1.], [0., 1., 1.]])
        mesh = filtering.TriangleMesh(vertices, np.tile([0, 1, 2], (128, 1)))
        directions = np.tile([0., 0., 1.], (128, 1))
        cross = np.cross
        batches = []
        def observed_cross(first, second):
            batches.append(len(first))
            return cross(first, second)
        with patch.object(filtering.np, 'cross', side_effect=observed_cross):
            result = filtering._mesh_depth(mesh, np.zeros(3), directions,
                                           np.full(len(directions), np.inf))
        self.assertTrue(batches)
        self.assertLessEqual(max(batches), 1024)
        np.testing.assert_array_equal(result, np.ones(len(directions)))

    def test_pending_frontier_limit_falls_back_without_mutating_inputs(self):
        mesh = box_mesh()
        directions = np.tile([0., 0., 1.], (128, 1))
        limits = np.full(len(directions), np.inf)
        expected = filtering._mesh_depth_bvh(mesh, np.zeros(3), directions, limits)
        with patch.object(filtering, '_MAX_PENDING_NODE_PAIRS', 1), patch.object(
                filtering, '_mesh_depth_bvh', wraps=filtering._mesh_depth_bvh) as fallback:
            result = filtering._mesh_depth(mesh, np.zeros(3), directions, limits)
        fallback.assert_called_once()
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(np.isinf(limits).all())
        np.testing.assert_array_equal(directions, np.tile([0., 0., 1.], (128, 1)))

    def test_complete_masks_match_near_depth_tolerance_and_on_foreground(self):
        mesh = box_mesh(lower=(-1., -1., 1.), upper=(1., 1., 2.))
        part = filtering.MeshInstance(mesh, np.diag([2., .5, 1.5, 1.]))
        model = snapshot([part])
        depths = [[1.5-.005000001, 1.5-.005, 1.5, 1.5+.005, 1.5+.005000001,
                   .9, 2., np.nan, 0.]]
        current = frame(depths)
        source = filtering.RobotSelfFilter(stride=1)
        with patch.object(filtering, '_mesh_depth', filtering._mesh_depth_bvh):
            expected = source.masks((current,), {'A': model})['A']
        actual = source.masks((current,), {'A': model})['A']
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual[0, 0])
        self.assertTrue(actual[0, 2])
        self.assertFalse(actual[0, 5])


if __name__ == '__main__':
    unittest.main()
