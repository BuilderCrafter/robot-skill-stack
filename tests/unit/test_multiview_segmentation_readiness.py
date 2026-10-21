"""Measured islands, ambiguous missing data, view evidence and fitting bounds."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit, SurfaceFitter, unknown_fit
from robot_skill_stack.world.perception_multiview.provider import _FitTimer
from robot_skill_stack.world.perception_multiview.segmentation import (
    _metric_components, _partition_view_evidence, cluster_points, refine_cluster, supported_object_cloud,
)
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


def box_fit(center, size):
    return PrimitiveFit(PrimitiveGeometry('cube', 1., box_size=np.asarray(size), yaw=0.),
                        np.asarray(center), np.asarray(size))


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


def sphere_with_islands(center):
    def surface(theta, phi):
        theta, phi = np.meshgrid(theta, phi)
        return np.asarray(center)+.025*np.c_[np.sin(phi.ravel())*np.cos(theta.ravel()),
            np.sin(phi.ravel())*np.sin(theta.ravel()), np.cos(phi.ravel())]
    dense = surface(np.linspace(-1.1, 1.1, 15), np.linspace(.3, 2.8, 18))
    islands = surface(np.linspace(1.7, 4.5, 12), np.linspace(.3, 2.8, 8))
    points = np.r_[dense, islands]
    return points[points[:, 2] > .004]


class SegmentationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.fitter = SurfaceFitter()
        self.parent = box_fit((.45, 0., .025), (.09, .04, .05))
        self.children = [box_fit((x, 0., .025), (.04, .04, .05)) for x in (.425, .475)]

    def test_sparse_current_islands_complete_distinct_sphere_clouds(self):
        points = np.r_[sphere_with_islands((0., 0., .025)), sphere_with_islands((.06, 0., .025))]
        seeds = _metric_components(points)
        self.assertLess(sum(map(len, seeds))/len(points), .85)
        refined = refine_cluster(points, self.fitter)
        self.assertEqual(len(refined), 2)
        self.assertTrue(all(fit.geometry.shape.value == 'sphere' for _, fit in refined))
        self.assertGreater(sum(len(cloud) for cloud, _ in refined)/len(points), .98)
        self.assertLess(np.linalg.norm(refined[0][1].center-[0., 0., .025]), .001)
        self.assertLess(np.linalg.norm(refined[1][1].center-[.06, 0., .025]), .001)

    def test_missing_band_does_not_create_two_known_boxes(self):
        points = box_surface()
        points = points[abs(points[:, 0]-.45) > .004]
        refined = refine_cluster(points, self.fitter)
        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0][1].geometry.shape.value, 'unknown')
        self.assertEqual(refined[0][1].geometry.metadata['reason'], 'ambiguous_single_or_multiple_surfaces')
        np.testing.assert_array_equal(refined[0][0], points)

    def test_measured_table_gap_supports_distinct_solids(self):
        frames = analytic_frames([Solid('cube', center=(x, 0., .025), size=(.04, .04, .05))
                                  for x in (.425, .475)])
        evidence = _partition_view_evidence(self.parent, self.children, frames, 0.)
        self.assertTrue(evidence['free_gap'], evidence)
        self.assertFalse(evidence['occupied_bridge'], evidence)

    def test_valid_occupied_bridge_supports_one_intact_solid(self):
        frames = analytic_frames([Solid('cube', center=(.45, 0., .025), size=(.09, .04, .05))])
        evidence = _partition_view_evidence(self.parent, self.children, frames, 0.)
        self.assertTrue(evidence['occupied_bridge'], evidence)
        self.assertFalse(evidence['free_gap'], evidence)

    def test_invalid_depth_is_neither_a_gap_nor_an_occupied_bridge(self):
        from dataclasses import replace
        frames = [replace(frame, depth=np.full_like(frame.depth, np.nan)) for frame in analytic_frames([])]
        evidence = _partition_view_evidence(self.parent, self.children, frames, 0.)
        self.assertFalse(evidence['free_gap'], evidence)
        self.assertFalse(evidence['occupied_bridge'], evidence)

    def test_insufficient_shared_budget_does_no_partial_numerical_work(self):
        points = np.r_[sphere_with_islands((0., 0., .025)), sphere_with_islands((.06, 0., .025))]
        fitter = Mock(remaining=4)
        refined = refine_cluster(points, fitter)
        self.assertEqual(len(refined), 1)
        self.assertIsNone(refined[0][1])
        fitter.fit.assert_not_called()

    def test_below_fitter_sample_minimum_is_not_an_object_candidate(self):
        self.assertEqual(supported_object_cloud(box_surface()[:26], 0.),
                         (False, 'insufficient_surface_samples'))

    def test_fragmented_elevated_clutter_has_one_global_numerical_budget(self):
        x, z = np.meshgrid(np.linspace(-.007, .007, 7), np.linspace(.025, .043, 6))
        points = np.concatenate([np.c_[x.ravel()+.208+.032*parent,
            np.full(x.size, .007*fragment), z.ravel()] for parent in range(16) for fragment in range(8)])
        numerical = Mock()
        numerical.fit.side_effect = lambda cloud, **kwargs: unknown_fit(cloud)
        bounded = _FitTimer(numerical, max_calls=24)
        broad = cluster_points(points)
        self.assertEqual(len(broad), 16)
        for cloud in broad:
            partitions = refine_cluster(cloud, bounded)
            for child, cached in partitions:
                eligible, reason = supported_object_cloud(child, 0.)
                if eligible or reason == 'no_observed_support_contact':
                    fit = cached or bounded.fit(child)
                    self.assertEqual(fit.geometry.shape.value, 'unknown')
        self.assertEqual(numerical.fit.call_count, 24)
        self.assertEqual(bounded.remaining, 0)
        fresh_cloud = points[:42]+[0., .2, .01]
        exhausted = bounded.fit(fresh_cloud)
        self.assertEqual(numerical.fit.call_count, 24)
        self.assertEqual(exhausted.geometry.shape.value, 'unknown')
        self.assertEqual(exhausted.geometry.metadata['reason'], 'candidate_budget_exhausted')
        np.testing.assert_allclose(exhausted.center, (fresh_cloud.min(0)+fresh_cloud.max(0))/2)

    def test_excessive_dense_fragments_skip_all_child_fits(self):
        x, z = np.meshgrid(np.linspace(-.007, .007, 7), np.linspace(.025, .043, 6))
        points = np.concatenate([np.c_[x.ravel(), np.full(x.size, .007*index), z.ravel()] for index in range(16)])
        fitter = Mock(remaining=24)
        refined = refine_cluster(points, fitter)
        self.assertEqual(len(refined), 1)
        self.assertIs(refined[0][0], points)
        fitter.fit.assert_not_called()

    def test_fresh_good_evidence_restores_graspability_on_the_same_identity(self):
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        from robot_skill_stack.world.model.updater import WorldModelUpdater
        from robot_skill_stack.world.model.world_model import WorldModel
        from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
        from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
        from tests.unit.test_multiview_provider import Camera
        frames = analytic_frames([Solid('sphere')])
        cameras = [Camera(frame) for frame in frames]
        with patch('time.monotonic', return_value=100.) as clock:
            rig = FrameSynchronizer(dict(zip(('A', 'B'), cameras)))
            provider = MultiViewProvider(rig, load_scene_config('config/scenes/multiview_playground.toml').scene)
            self.addCleanup(provider.close)
            good = [False]
            def finalize(fit, points, **kwargs):
                fit.geometry.metadata['parameter_observable'] = good[0]
                return fit
            provider.fitter.finalize = finalize
            model = WorldModel()
            updater = WorldModelUpdater(model, provider)
            for _ in range(3):
                clock.return_value += .2
                for camera in cameras:
                    camera.number += 1
                updater.update()
            self.assertEqual(len(model.objects()), 1)
            original = model.objects()[0]
            self.assertEqual(original.class_name, 'sphere')
            self.assertFalse(original.graspable)
            previous_time = original.last_seen
            good[0] = True
            clock.return_value += .2
            for camera in cameras:
                camera.number += 1
            updater.update()
            refreshed = model.require(original.object_id)
            self.assertTrue(refreshed.graspable)
            self.assertGreater(refreshed.last_seen, previous_time)
            self.assertEqual(len(model.objects()), 1)


if __name__ == '__main__':
    unittest.main()
