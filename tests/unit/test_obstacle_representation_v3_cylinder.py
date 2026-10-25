import unittest
from types import SimpleNamespace

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh
from scripts.obstacle_representation_v3_cylinder import (
    convex_distance, convex_distance_details, depth_reference_scores_convex,
    reduce_cylinder_clearances, unit_prism)
from tests.support.reactive_geometry import CollisionMeshReference, mesh_box_distance


VERTICES = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
TRIANGLES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


class CylinderReferenceTests(unittest.TestCase):
    def test_regular_prism_dimensions_and_topology(self):
        vertices, counts, indices = unit_prism()
        self.assertEqual(vertices.shape, (64, 3))
        np.testing.assert_allclose(np.linalg.norm(vertices[:, :2], axis=1), .5)
        np.testing.assert_allclose(vertices.max(0)-vertices.min(0), [1., 1., 1.])
        self.assertEqual(int(counts.sum()), len(indices))
        self.assertEqual(len(counts), 34)

    def test_axis_aligned_separation_overlap_and_containment(self):
        for shift, expected in (([3., 0, 0], 1.), ([3., 4., 0], np.sqrt(5)),
                                ([0., 0, 0], 0.), ([1., 0, 0], 0.)):
            with self.subTest(shift=shift):
                self.assertAlmostEqual(convex_distance(VERTICES, VERTICES+shift), expected, places=8)
        self.assertEqual(convex_distance(VERTICES, VERTICES*.1), 0.)

    def test_matches_independent_triangle_box_reference_for_rotated_boxes(self):
        generator = np.random.default_rng(721)
        mesh = TriangleMesh(VERTICES, TRIANGLES)
        for index in range(35):
            rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = generator.uniform(-4, 4, 3)
            half = generator.uniform(.15, 1.8, 3)
            points = VERTICES@rotation.T+transform[:3, 3]
            expected = mesh_box_distance(mesh, transform, half)
            actual = convex_distance_details(points, VERTICES*half)
            with self.subTest(index=index):
                self.assertLessEqual(actual['upper_bound_m']-actual['lower_bound_m'], 1.001e-9)
                self.assertAlmostEqual(actual['distance_m'], expected, places=7)

    def test_prism_distance_uses_curved_polygon_boundary_not_enclosing_box(self):
        vertices, _, _ = unit_prism()
        # This point is inside the enclosing square but outside the prism.
        point = np.array([[.49, .49, 0.]])
        expected = np.sqrt(2)*.49-.5
        self.assertAlmostEqual(convex_distance(vertices, point), expected, places=7)
        self.assertAlmostEqual(convex_distance(vertices, vertices+[0, 0, 1.2]), .2, places=8)

    def test_rigid_transform_preserves_distance(self):
        hull, _, _ = unit_prism()
        second = hull+[1.3, .3, .2]
        angle = .713
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        expected = convex_distance(hull, second)
        actual = convex_distance(hull@rotation.T+[2., -3., .7], second@rotation.T+[2., -3., .7])
        self.assertAlmostEqual(actual, expected, places=8)

    def test_depth_reference_optical_intersection_and_occlusion(self):
        hull = unit_prism()
        transform = np.eye(4)
        transform[2, 3] = -2.
        frame = SimpleNamespace(camera_id='camera', token=(1, 1), acquired_at=9.,
            depth=np.array([[1.5]]), intrinsics=np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]]),
            world_from_camera=transform)
        row = depth_reference_scores_convex([frame], [hull], [0., 0, 0])['rows'][0]
        self.assertEqual(row['potential_stride_grid_rays'], 1)
        self.assertEqual(row['visible_stride_grid_returns'], 1)
        frame.depth[:] = 1.
        self.assertEqual(depth_reference_scores_convex([frame], [hull], np.eye(4))['rows'][0]['occluded_stride_grid_returns'], 1)

    def test_reducer_requires_cooked_geometry_and_tracks_actual_body_transform(self):
        mesh = TriangleMesh(VERTICES*.1, TRIANGLES)
        part = CollisionMeshReference('/robotmesh', '/robot', 'wrist', mesh, np.eye(4), 'physx_cooked:convexHull')
        pose = np.repeat(np.eye(4)[None], 2, axis=0)
        pose[0, 2, 3] = 1.
        hull = unit_prism()
        boxes = [[('/cylinder', pose[1], [1, 1, 1])]]
        result = reduce_cylinder_clearances([part], ['/robot', '/cylinder'], [pose], boxes, [hull], '/cylinder')
        self.assertAlmostEqual(result['minimum_m'], .4, places=8)
        self.assertEqual(result['mesh_box_evaluations'], 0)
        bad = CollisionMeshReference('/robotmesh', '/robot', 'wrist', mesh, np.eye(4), 'authored_fallback')
        with self.assertRaises(ValueError):
            reduce_cylinder_clearances([bad], ['/robot', '/cylinder'], [pose], boxes, [hull], '/cylinder')

    def test_invalid_inputs_and_bounded_iteration_fail_explicitly(self):
        for value in ([], [[np.nan, 0, 0]], [[0, 0]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                convex_distance(value, VERTICES)
        for settings in ({'tolerance':0}, {'max_iterations':0}, {'max_iterations':True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                convex_distance(VERTICES, VERTICES, **settings)


if __name__ == '__main__':
    unittest.main()
