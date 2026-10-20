import unittest
import numpy as np

from robot_skill_stack.integrations.isaac.config import SceneGeometry
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


class SurfaceFittingTests(unittest.TestCase):
    def setUp(self):
        self.fitter = SurfaceFitter()
        self.scene = SceneGeometry(0., [.2, -.4, 0], [.7, .4, .22])

    def clouds(self, solid):
        return [crop_workspace(project_depth(f, stride=1)[0], self.scene)
                for f in analytic_frames([solid])]

    def test_fused_boxes_spheres_and_upright_and_lying_cylinders(self):
        for solid in (Solid('cube', size=(.06, .04, .05), yaw=.43),
                      Solid('sphere', radius=.025),
                      Solid('cylinder', center=(.45, 0, .05), radius=.02, length=.10),
                      Solid('cylinder', center=(.45, 0, .02), radius=.02, length=.10,
                            axis=(np.cos(.43), np.sin(.43), 0.))):
            with self.subTest(shape=solid.shape, axis=solid.axis):
                fit = self.fitter.fit(fuse_clouds(self.clouds(solid)))
                self.assertEqual(fit.geometry.shape.value, solid.shape, fit.geometry.scores)
                self.assertLess(np.linalg.norm(fit.center-solid.center), .004)
                if solid.shape == 'cylinder':
                    self.assertLess(abs(fit.geometry.radius-solid.radius), .003)
                    self.assertGreater(abs(np.dot(fit.geometry.axis, solid.axis)), .995)

    def test_duplicate_views_do_not_inflate_confidence(self):
        clouds = self.clouds(Solid('sphere'))
        first = self.fitter.fit(fuse_clouds(clouds))
        duplicate = self.fitter.fit(fuse_clouds(clouds+[np.repeat(clouds[0], 5, axis=0)]))
        self.assertEqual(first.geometry.shape, duplicate.geometry.shape)
        self.assertLess(abs(first.geometry.confidence-duplicate.geometry.confidence), .03)

    def test_small_planar_and_unsupported_geometry_remains_unknown(self):
        for points in (np.zeros((3, 3)), np.c_[np.random.default_rng(1).uniform(.4, .45, (100, 2)),
                                             np.full(100, .05)]):
            self.assertEqual(self.fitter.fit(points).geometry.shape.value, 'unknown')
        fit = self.fitter.fit(fuse_clouds(self.clouds(Solid('ellipsoid', size=(.08, .04, .05)))))
        self.assertEqual(fit.geometry.shape.value, 'unknown', fit.geometry.scores)


if __name__ == '__main__':
    unittest.main()
