"""Connectivity and conservative splitting regressions, independent of Isaac."""
from itertools import product
import unittest

import numpy as np

from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from robot_skill_stack.world.perception_multiview.segmentation import (
    cluster_points, _metric_components, refine_cluster, supported_object_cloud,
)


def reference_groups(points, *, cell_size=.008, radius=None):
    """Small all-pairs oracle for connectivity, used only on tiny test clouds."""
    if radius is None:
        cells = np.floor(points/cell_size).astype(int)
        adjacency = abs(cells[:, None]-cells).max(2) <= 1
    else:
        adjacency = np.linalg.norm(points[:, None]-points, axis=2) <= radius
    remaining, groups = set(range(len(points))), []
    while remaining:
        stack, selected = [remaining.pop()], []
        while stack:
            index = stack.pop()
            selected.append(index)
            neighbors = remaining.intersection(np.flatnonzero(adjacency[index]))
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        groups.append(points[selected])
    return groups


def canonical(groups):
    return sorted(tuple(sorted(map(tuple, group))) for group in groups)


class SegmentationTests(unittest.TestCase):
    @staticmethod
    def sphere(center, phi=None):
        if phi is None:
            phi = np.linspace(.12, 2.9, 35)
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 65, endpoint=False), phi)
        directions = np.c_[np.sin(phi.ravel())*np.cos(theta.ravel()),
                           np.sin(phi.ravel())*np.sin(theta.ravel()), np.cos(phi.ravel())]
        points = np.asarray(center)+.025*directions
        return points[points[:, 2] > center[2]-.021]

    def test_optimized_broad_graph_preserves_reference_connectivity(self):
        rng = np.random.default_rng(76)
        for scale in (.001, .008, .035):
            points = rng.uniform(-scale, scale, (90, 3))
            self.assertEqual(canonical(cluster_points(points, min_points=1)),
                             canonical(reference_groups(points)))

    def test_negative_cell_boundaries_and_diagonal_neighbors(self):
        points = np.array(list(product((-.008, 0., .008), repeat=3))+[(.04, 0., 0.)])
        actual = cluster_points(points, min_points=1)
        self.assertEqual(sorted(map(len, actual)), [1, 27])
        self.assertEqual(canonical(actual), canonical(reference_groups(points)))
        self.assertEqual(canonical(actual), canonical(cluster_points(points[::-1], min_points=1)))

    def test_metric_graph_uses_distances_inside_same_and_neighbor_cells(self):
        points = np.array([[0., 0., 0.], [.0039, .0039, .0039],
                           [.0041, .0039, .0039], [.011, 0., 0.]])
        self.assertEqual(canonical(_metric_components(points, min_points=1)),
                         canonical(reference_groups(points, radius=.004)))
        self.assertEqual(sorted(map(len, _metric_components(points, min_points=1))), [1, 1, 2])

    def test_metric_graph_matches_exact_reference_under_density_variation(self):
        rng = np.random.default_rng(81)
        sparse = rng.uniform(-.012, .012, (90, 3))
        for points in (sparse, np.r_[sparse, np.zeros((70, 3))]):
            self.assertEqual(canonical(_metric_components(points, min_points=1)),
                             canonical(reference_groups(points, radius=.004)))

    def test_metric_graph_bounds_dense_duplicate_cells_without_changing_connectivity(self):
        points = np.r_[np.zeros((70, 3)), np.full((70, 3), .01)]
        self.assertEqual(sorted(map(len, _metric_components(points))), [70, 70])

    def test_nearby_spheres_split_only_with_supported_separate_models(self):
        points = np.r_[self.sphere((0., 0., .125)), self.sphere((.055, 0., .125))]
        self.assertEqual(len(cluster_points(points)), 1)
        original = points.copy()
        refined = refine_cluster(points, SurfaceFitter(), support_plane_z=.1)
        self.assertEqual(len(refined), 2)
        self.assertTrue(all(fit.geometry.shape.value == 'sphere' for _, fit in refined))
        self.assertLess(np.linalg.norm(refined[0][1].center-[0., 0., .125]), .001)
        self.assertLess(np.linalg.norm(refined[1][1].center-[.055, 0., .125]), .001)
        self.assertGreater(sum(len(cloud) for cloud, _ in refined)/len(points), .95)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(canonical([np.concatenate([cloud for cloud, _ in refined])]), canonical([points]))

    def test_fragmented_spheres_are_unknown_without_a_recovery_retry(self):
        # Each sphere's hemispheres are separated by 4.2 mm of missing samples.
        phi = np.r_[np.linspace(.12, np.pi/2-.084, 20), np.linspace(np.pi/2+.084, 2.9, 20)]
        points = np.r_[self.sphere((0., 0., .125), phi), self.sphere((.06, 0., .125), phi)]
        refined = refine_cluster(points, SurfaceFitter(), support_plane_z=.1)
        self.assertEqual(len(refined), 1)
        self.assertIs(refined[0][0], points)
        self.assertEqual(refined[0][1].geometry.shape.value, 'unknown')

    def test_disconnected_sparse_unknown_patches_do_not_invent_objects(self):
        y, z = np.meshgrid(np.linspace(0., .04, 15), np.linspace(.005, .04, 15))
        first = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
        points = np.r_[first, first+[.006, 0., 0.]]
        refined = refine_cluster(points, SurfaceFitter())
        self.assertEqual(len(refined), 1)
        self.assertIs(refined[0][0], points)
        self.assertEqual(refined[0][1].geometry.shape.value, 'unknown')

    def test_background_connected_cluster_is_rejected_whole(self):
        x, z = np.meshgrid(np.linspace(-.1, .1, 70), np.linspace(.005, .20, 70))
        background = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        sphere = self.sphere((0., -.031, .025))
        points = np.r_[background, sphere]
        self.assertEqual(len(cluster_points(points)), 1)
        refined = refine_cluster(points, SurfaceFitter())
        self.assertEqual(len(refined), 1)
        self.assertIs(refined[0][0], points)
        self.assertIsNone(refined[0][1])
        self.assertFalse(supported_object_cloud(points, 0.)[0])


if __name__ == '__main__':
    unittest.main()
