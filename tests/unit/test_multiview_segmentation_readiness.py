"""Conservative refinement, shared fitting budget and actual camera-to-object flow."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from robot_skill_stack.world.perception_multiview.provider import _FitBudget
from robot_skill_stack.world.perception_multiview.segmentation import refine_cluster
from tests.support.multiview_fixtures import analytic_frames
from tests.support.primitive_scene import Solid
from tests.unit import test_multiview_segmentation as fixtures


def box_surface():
    grid = np.linspace(-.025, .025, 22)
    points = []
    for fixed in range(3):
        other = [axis for axis in range(3) if axis != fixed]
        a, b = np.meshgrid(grid, grid)
        for sign in (-1, 1):
            face = np.zeros((a.size, 3))
            face[:, fixed] = sign*.025
            face[:, other[0]], face[:, other[1]] = a.ravel(), b.ravel()
            points.append(face)
    return np.unique(np.concatenate(points), axis=0)+[.45, 0., .025]


class SegmentationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.fitter = SurfaceFitter()
        self.spheres = np.r_[fixtures.SegmentationTests.sphere((0., 0., .025)),
                             fixtures.SegmentationTests.sphere((.055, 0., .025))]

    def test_missing_band_is_unknown_and_not_observable(self):
        points = box_surface()
        points = points[abs(points[:, 0]-.45) > .004]
        result = refine_cluster(points, self.fitter)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0][0], points)
        self.assertEqual(result[0][1].geometry.shape.value, 'unknown')
        finalized = self.fitter.finalize(result[0][1], points)
        self.assertFalse(finalized.geometry.metadata['parameter_observable'])

    def test_continuous_box_is_not_split(self):
        points = box_surface()
        result = refine_cluster(points, self.fitter)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0][0], points)
        fit = result[0][1] or self.fitter.fit(points)
        self.assertEqual(fit.geometry.shape.value, 'cube')

    def test_insufficient_shared_budget_returns_unknown_without_fitting(self):
        fitter = Mock(remaining=2)
        result = refine_cluster(self.spheres, fitter)
        self.assertEqual(result[0][1].geometry.shape.value, 'unknown')
        fitter.fit.assert_not_called()

    def test_child_and_parent_fits_share_the_budget_and_cached_results(self):
        budget = _FitBudget(self.fitter, max_calls=3)
        with patch.object(self.fitter, 'fit', wraps=self.fitter.fit) as fit:
            result = refine_cluster(self.spheres, budget)
            self.assertEqual(len(result), 2)
            self.assertTrue(all(cached.geometry.shape.value == 'sphere' for _, cached in result))
            self.assertEqual(fit.call_count, 3)
            self.assertEqual(budget.remaining, 0)
            refused = refine_cluster(self.spheres, budget)
            self.assertEqual(refused[0][1].geometry.shape.value, 'unknown')
            self.assertEqual(fit.call_count, 3)

    def test_excessive_fragments_are_unknown_without_child_fits(self):
        x, z = np.meshgrid(np.linspace(-.007, .007, 7), np.linspace(.025, .043, 6))
        points = np.concatenate([np.c_[x.ravel(), np.full(x.size, .007*i), z.ravel()] for i in range(16)])
        fitter = Mock(remaining=24)
        result = refine_cluster(points, fitter)
        self.assertIs(result[0][0], points)
        self.assertEqual(result[0][1].geometry.shape.value, 'unknown')
        fitter.fit.assert_not_called()

    def test_multicamera_provider_detects_supported_scene_shapes(self):
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
        from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
        from tests.unit.test_multiview_provider import Camera
        solids = [Solid('cube', center=(.32, -.18, .025), yaw=.37), Solid('sphere'),
                  Solid('cylinder', center=(.38, .18, .045), radius=.018, length=.09),
                  Solid('cylinder', center=(.58, -.18, .018), radius=.018, length=.09, axis=(1, 0, 0))]
        cameras = {str(i): Camera(frame) for i, frame in enumerate(analytic_frames(solids))}
        scene = load_scene_config('config/scenes/multiview_playground.toml').scene
        provider = MultiViewProvider(FrameSynchronizer(cameras), scene)
        self.addCleanup(provider.close)
        for _ in range(3):
            for camera in cameras.values():
                camera.number += 1
            batch = provider.observe()
        self.assertCountEqual([obj.class_name for obj in batch.observations], ['cube', 'sphere', 'cylinder', 'cylinder'])
        for solid in solids:
            nearest = min(batch.observations, key=lambda obj: np.linalg.norm(obj.pose.position-solid.center))
            self.assertEqual(nearest.class_name, solid.shape)
            self.assertLess(np.linalg.norm(nearest.pose.position-solid.center), .004)
            self.assertTrue(nearest.metadata['contributing_views'])

    def test_rendered_sphere_survives_missing_depth_and_separate_foreground(self):
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
        from robot_skill_stack.world.perception_multiview.provider import MultiViewInput, MultiViewProcessor
        from tests.support.primitive_scene import render
        scene = load_scene_config('config/scenes/multiview_perception_final.toml').scene
        cases = [([Solid('sphere')], ((1.05, .55, .90), (.25, -.85, .65)), .2),
                 ([Solid('sphere'), Solid('cube', center=(.45, .145, .04), size=(.045, .24, .05))],
                  ((.85, .35, .60), (.2, -.5, .45)), 0.)]
        for solids, eyes, dropout in cases:
            with self.subTest(dropout=dropout):
                frames = []
                for i, eye in enumerate(eyes):
                    raw = render(solids, eye=eye, resolution=(640, 480), seed=71+i)
                    depth, intrinsics = raw.depth.copy(), raw.intrinsics.copy()
                    depth[np.random.default_rng(720+i).random(depth.shape) < dropout] = np.nan
                    intrinsics[:2, 2] += .5
                    frames.append(DepthFrame(str(i), (1, .1), depth, intrinsics, raw.world_from_camera, 1.))
                result = MultiViewProcessor(scene).compute(MultiViewInput(tuple(frames), 1.))
                self.assertEqual(len(result.detections), 1)
                fit = result.detections[0].fit
                self.assertEqual(fit.geometry.shape.value, 'sphere')
                self.assertLess(np.linalg.norm(fit.center-[.45, 0., .025]), .001)
                self.assertAlmostEqual(fit.geometry.radius, .025, delta=.001)
                self.assertTrue(fit.geometry.metadata['parameter_observable'])


if __name__ == '__main__':
    unittest.main()
