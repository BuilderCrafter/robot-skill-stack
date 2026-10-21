"""Geometric edge evidence is distinct from primitive-class consistency."""
from dataclasses import replace
import unittest

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.clouds import project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.perception_multiview.observability import assess
from tests.support.multiview_research import analytic_frames
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
    return voxelize(np.concatenate(clouds)), tuple(clouds)


class BoxObservabilityTests(unittest.TestCase):
    def frames(self, solids):
        return analytic_frames(solids, eyes=((1.49, -.15, 1.13), (.19, -.74, .66)))

    def test_clear_box_has_bounded_edges_without_changing_input(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points, views = candidate_points(frames, fit)
        old_center, old_size = fit.center.copy(), fit.geometry.box_size.copy()
        result = assess(fit, points, frames, view_clouds=views)
        self.assertTrue(result['observable'], result)
        self.assertTrue(all(axis[side]['status'] == 'bounded' for axis in result['axes']
                            for side in ('negative', 'positive')))
        bounds = result['conservative_geometry']
        self.assertTrue(np.all(np.asarray(bounds['box_size']) >= old_size))
        self.assertLess(max(np.asarray(bounds['box_size'])-old_size), .004)
        np.testing.assert_array_equal(fit.center, old_center)
        np.testing.assert_array_equal(fit.geometry.box_size, old_size)
        self.assertNotIn('geometric_observability', fit.geometry.metadata)

    def test_one_clear_view_can_bound_edge_hidden_in_other_view(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points, views = candidate_points(frames, fit)
        blocked = replace(frames[0], depth=np.full_like(frames[0].depth, .05))
        # A different clear view of each edge is sufficient; a camera's foreground
        # is not a veto against genuine physical-edge evidence in its companion.
        result = assess(fit, points, (blocked, frames[1]), view_clouds=views)
        self.assertTrue(result['observable'], result)
        self.assertTrue(any(side['cameras'][0]['occluded'] > 0 for axis in result['axes'][:2]
                            for side in (axis['negative'], axis['positive'])))

    def test_physical_both_view_cover_rejects_confident_partial_box(self):
        fit = box_fit(center=(.445, -.0333, .025), size=(.09, .0434, .05))
        solids = [Solid('cube', center=(.445, -.01, .025), size=(.09, .09, .05)),
                  Solid('cube', center=(.445, .04, .085), size=(.30, .17, .014)),
                  Solid('cube', center=(.37, -.015, .105), size=(.012, .19, .21))]
        frames = self.frames(solids)
        points, views = candidate_points(frames, fit)
        result = assess(fit, points, frames, view_clouds=views)
        self.assertFalse(result['observable'], result)
        self.assertEqual(result['axes'][1]['positive']['status'], 'occluded')
        self.assertIsNone(result['conservative_geometry'])
        self.assertEqual(fit.geometry.shape.value, 'cube')
        self.assertEqual(fit.geometry.confidence, .999)

    def test_invalid_depth_does_not_establish_free_space(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points, _ = candidate_points(frames, fit)
        invalid = [replace(frame, depth=np.full_like(frame.depth, np.nan)) for frame in frames]
        result = assess(fit, points, invalid)
        self.assertFalse(result['observable'])
        self.assertTrue(all(axis[side]['status'] == 'missing' for axis in result['axes'][:2]
                            for side in ('negative', 'positive')))

    def test_out_of_frame_is_unknown_instead_of_free(self):
        fit = box_fit()
        frames = self.frames([Solid('cube', tuple(fit.center), tuple(fit.geometry.box_size))])
        points, _ = candidate_points(frames, fit)
        outside = []
        for frame in frames:
            k = frame.intrinsics.copy()
            k[0, 2] += 10000
            outside.append(replace(frame, intrinsics=k))
        result = assess(fit, points, outside)
        self.assertFalse(result['observable'])
        self.assertTrue(all(axis[side]['status'] == 'out_of_frame' for axis in result['axes'][:2]
                            for side in ('negative', 'positive')))

    def test_small_hidden_extension_is_rejected_or_carried_in_geometry(self):
        fit = box_fit(center=(.445, -.011, .025), size=(.071, .069, .05))
        frames = self.frames([Solid('cube', center=(.445, -.01, .025), size=(.071, .071, .05))])
        points, views = candidate_points(frames, fit)
        result = assess(fit, points, frames, view_clouds=views)
        if result['observable']:
            geometry = result['conservative_geometry']
            center, size = np.asarray(geometry['center']), np.asarray(geometry['box_size'])
            self.assertGreaterEqual(center[1]+size[1]/2, -.01+.071/2)
        else:
            self.assertIsNone(result['conservative_geometry'])

    def test_oblique_plane_height_error_does_not_make_hidden_extent_free(self):
        fit = box_fit(center=(.445, -.0333, .025), size=(.09, .0434, .05))
        solids = [Solid('cube', center=(.445, -.01, .025), size=(.09, .09, .05)),
                  Solid('cube', center=(.445, .04, .085), size=(.30, .17, .014)),
                  Solid('cube', center=(.37, -.015, .105), size=(.012, .19, .21))]
        frames = self.frames(solids)
        points, views = candidate_points(frames, fit)
        # A correlated 1.5 mm top-plane overestimate is within the surface
        # consistency tolerance, and oblique ray intersections amplify it.
        fit.center[2] += .00075
        fit.geometry.box_size[2] += .0015
        fit.size[2] += .0015
        result = assess(fit, points, frames, view_clouds=views)
        self.assertFalse(result['observable'], result)
        self.assertGreater(result['top_spread_m'], .001)
        self.assertIsNone(result['conservative_geometry'])

    def test_missing_frames_fail_closed_and_curves_keep_existing_path(self):
        fit = box_fit()
        self.assertFalse(assess(fit, np.zeros((40, 3)), ())['observable'])
        sphere = PrimitiveFit(PrimitiveGeometry('sphere', .99, radius=.025), np.array([.45, 0., .025]),
                              np.full(3, .05))
        result = assess(sphere, [], ())
        self.assertFalse(result['applicable'])
        self.assertEqual(result['reason'], 'not_a_box')
        self.assertIsNone(result['conservative_geometry'])


if __name__ == '__main__':
    unittest.main()
