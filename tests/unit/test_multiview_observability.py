"""Geometric edge evidence is distinct from primitive-class consistency."""
from dataclasses import replace
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.clouds import project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.perception_multiview.observability import assess
from tests.support.multiview_fixtures import analytic_frames
from tests.support.primitive_scene import Solid


def box_fit(center=(.445, -.01, .025), size=(.09, .09, .05), yaw=0.):
    size = np.asarray(size, float)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return PrimitiveFit(PrimitiveGeometry('cube', .999, box_size=size, yaw=yaw),
                        np.asarray(center), abs(rotation) @ size)


def candidate_points(frames, fit):
    """Measured fitted-surface points; analytic truth is not passed to assess."""
    size = fit.geometry.box_size
    c, s = np.cos(fit.geometry.yaw), np.sin(fit.geometry.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    clouds = []
    for frame in frames:
        points = project_depth(frame)[0]
        q = abs((points-fit.center) @ rotation)-size/2
        distance = abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))
        clouds.append(voxelize(points[(distance < .0015) & (points[:, 2] > .004)]))
    return voxelize(np.concatenate(clouds))


class BoxObservabilityTests(unittest.TestCase):
    def frames(self, solids):
        return analytic_frames(solids, eyes=((1.49, -.15, 1.13), (.19, -.74, .66)))

    def test_clear_box_has_bounded_edges_without_changing_input(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points = candidate_points(frames, fit)
        old_center, old_size = fit.center.copy(), fit.geometry.box_size.copy()
        result = assess(fit, points, frames)
        self.assertIsNotNone(result)
        center, size = result
        self.assertTrue(np.all(size >= old_size))
        self.assertLess(max(size-old_size), .004)
        self.assertLess(np.linalg.norm(center-old_center), .002)
        np.testing.assert_array_equal(fit.center, old_center)
        np.testing.assert_array_equal(fit.geometry.box_size, old_size)
        self.assertNotIn('geometric_observability', fit.geometry.metadata)

    def test_one_clear_view_can_bound_edge_hidden_in_other_view(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points = candidate_points(frames, fit)
        blocked = replace(frames[0], depth=np.full_like(frames[0].depth, .05))
        # A different clear view of each edge is sufficient; a camera's foreground
        # is not a veto against genuine physical-edge evidence in its companion.
        result = assess(fit, points, (blocked, frames[1]))
        self.assertIsNotNone(result)
        self.assertIsNone(assess(fit, points, (blocked,)))
        reverse = assess(fit, points, (frames[1], blocked))
        np.testing.assert_array_equal(reverse, result)

    def test_physical_both_view_cover_rejects_confident_partial_box(self):
        fit = box_fit(center=(.445, -.0333, .025), size=(.09, .0434, .05))
        solids = [Solid('cube', center=(.445, -.01, .025), size=(.09, .09, .05)),
                  Solid('cube', center=(.445, .04, .085), size=(.30, .17, .014)),
                  Solid('cube', center=(.37, -.015, .105), size=(.012, .19, .21))]
        frames = self.frames(solids)
        points = candidate_points(frames, fit)
        result = assess(fit, points, frames)
        self.assertIsNone(result)
        self.assertEqual(fit.geometry.shape.value, 'cube')
        self.assertEqual(fit.geometry.confidence, .999)

    def test_invalid_depth_does_not_establish_free_space(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points = candidate_points(frames, fit)
        for value in (np.nan, 0., -1.):
            with self.subTest(depth=value):
                invalid = [replace(frame, depth=np.full_like(frame.depth, value)) for frame in frames]
                self.assertIsNone(assess(fit, points, invalid))

    def test_out_of_frame_is_unknown_instead_of_free(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points = candidate_points(frames, fit)
        outside = []
        for frame in frames:
            k = frame.intrinsics.copy()
            k[0, 2] += 10000
            outside.append(replace(frame, intrinsics=k))
        result = assess(fit, points, outside)
        self.assertIsNone(result)

    def test_small_hidden_extension_is_rejected_or_carried_in_geometry(self):
        fit = box_fit(center=(.445, -.011, .025), size=(.071, .069, .05))
        frames = self.frames([Solid('cube', center=(.445, -.01, .025), size=(.071, .071, .05))])
        points = candidate_points(frames, fit)
        result = assess(fit, points, frames)
        if result is not None:
            center, size = result
            self.assertGreaterEqual(center[1]+size[1]/2, -.01+.071/2)

    def test_oblique_plane_height_error_does_not_make_hidden_extent_free(self):
        fit = box_fit(center=(.445, -.0333, .025), size=(.09, .0434, .05))
        solids = [Solid('cube', center=(.445, -.01, .025), size=(.09, .09, .05)),
                  Solid('cube', center=(.445, .04, .085), size=(.30, .17, .014)),
                  Solid('cube', center=(.37, -.015, .105), size=(.012, .19, .21))]
        frames = self.frames(solids)
        points = candidate_points(frames, fit)
        # A correlated 1.5 mm top-plane overestimate is within the surface
        # consistency tolerance, and oblique ray intersections amplify it.
        fit.center[2] += .00075
        fit.geometry.box_size[2] += .0015
        fit.size[2] += .0015
        result = assess(fit, points, frames)
        self.assertIsNone(result)

    def test_missing_frames_or_non_box_geometry_has_no_box_bound(self):
        self.assertIsNone(assess(box_fit(), np.zeros((40, 3)), ()))
        for shape in ('sphere', 'cylinder', 'unknown'):
            fit = PrimitiveFit(PrimitiveGeometry(shape, .99), np.array([.45, 0., .025]), np.full(3, .05))
            self.assertIsNone(assess(fit, [], ()))


if __name__ == '__main__':
    unittest.main()
