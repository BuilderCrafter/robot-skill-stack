"""Geometric and rejection guards for the bounded orientation searches."""
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.config import SceneGeometry
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


class OptimizedSurfaceFittingTests(unittest.TestCase):
    def cloud(self, solid, *, noise=0.):
        scene = SceneGeometry(0., [.2, -.4, 0], [.7, .4, .22])
        clouds = [crop_workspace(project_depth(f, stride=1)[0], scene)
                  for f in analytic_frames([solid], noise=noise, seed=97)]
        return fuse_clouds(clouds)

    def test_box_refinement_preserves_dimensions_across_yaw_wrap(self):
        for yaw in (-.004, .4321, np.pi/2-.004):
            with self.subTest(yaw=yaw):
                solid = Solid('cube', center=(.45, 0., .025), size=(.075, .035, .05), yaw=yaw)
                fit = SurfaceFitter().fit(self.cloud(solid))
                self.assertEqual(fit.geometry.shape.value, 'cube', fit.geometry.scores)
                error = abs((fit.geometry.yaw-yaw+np.pi/4) % (np.pi/2)-np.pi/4)
                self.assertLess(error, np.deg2rad(.15))
                self.assertLess(np.max(abs(np.sort(fit.geometry.box_size)-np.sort(solid.size))), .001)
                self.assertLess(np.linalg.norm(fit.center-solid.center), .001)

    def test_lying_axes_cover_full_interval_and_preserve_radius_length(self):
        for yaw in (.071, .762, 1.648, 3.083):
            with self.subTest(yaw=yaw):
                axis = np.array([np.cos(yaw), np.sin(yaw), 0.])
                solid = Solid('cylinder', center=(.45, 0., .018), radius=.018, length=.09, axis=axis)
                fit = SurfaceFitter().fit(self.cloud(solid))
                self.assertEqual(fit.geometry.shape.value, 'cylinder', fit.geometry.scores)
                error = np.arccos(np.clip(abs(fit.geometry.axis @ axis), 0., 1.))
                self.assertLess(error, np.deg2rad(.3))
                self.assertLess(abs(fit.geometry.radius-solid.radius), .001)
                self.assertLess(abs(fit.geometry.length-solid.length), .0015)

    def test_noisy_and_sparse_curved_surfaces_still_compete(self):
        for solid in (Solid('sphere', radius=.025),
                      Solid('cylinder', center=(.45, 0., .045), radius=.02, length=.09)):
            with self.subTest(shape=solid.shape):
                cloud = self.cloud(solid, noise=.0002)
                fit = SurfaceFitter().fit(cloud[::2])
                self.assertEqual(fit.geometry.shape.value, solid.shape, fit.geometry.scores)
                self.assertLess(abs(fit.geometry.radius-solid.radius), .002)

    def test_planar_or_insufficient_evidence_remains_unknown(self):
        rng = np.random.default_rng(78)
        for count in (0, 12, 29, 300):
            points = np.c_[rng.uniform([.4, -.03], [.5, .03], (count, 2)), np.full(count, .05)]
            with self.subTest(points=count):
                fit = SurfaceFitter().fit(points)
                self.assertEqual(fit.geometry.shape.value, 'unknown', fit.geometry.scores)
                self.assertEqual(fit.geometry.metadata['reason'], 'insufficient_or_ambiguous_surface')


if __name__ == '__main__':
    unittest.main()
