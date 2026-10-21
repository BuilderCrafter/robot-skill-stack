from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid


class Camera:
    def __init__(self, frame):
        self.frame, self.number, self.closes = frame, 1, 0
    def get_frame_token(self): return self.number, self.number/10
    def get_depth(self): return self.frame.depth
    def get_rgb(self): return self.frame.rgb
    def get_intrinsics(self): return self.frame.intrinsics
    def get_world_from_camera_transform(self): return self.frame.world_from_camera
    def close(self): self.closes += 1


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.frames = analytic_frames([Solid('sphere')])
        self.cameras = [Camera(f) for f in self.frames]
        self.rig = FrameSynchronizer(dict(zip(('A', 'B'), self.cameras)))
        self.scene = load_scene_config('config/scenes/multiview_playground.toml').scene
        self.provider = MultiViewProvider(self.rig, self.scene)
        self.model = WorldModel()
        self.updater = WorldModelUpdater(self.model, self.provider, stale_object_ttl=2.)
        self.addCleanup(self.provider.close)

    def step(self, frames=None):
        self.now.return_value += .2
        for index, camera in enumerate(self.cameras):
            camera.number += 1
            if frames is not None:
                camera.frame = frames[index]
        return self.updater.update()

    def confirm(self, frames=None):
        for _ in range(3):
            self.step(frames)
        self.assertEqual(len(self.model.objects()), 1, self.provider.diagnostics)
        return self.model.objects()[0].object_id

    def test_fusion_produces_one_observation_and_persistent_identity(self):
        key = self.confirm()
        for _ in range(3):
            self.step()
        self.assertEqual([o.object_id for o in self.model.objects()], [key])
        obj = self.model.require(key)
        self.assertEqual(obj.class_name, 'sphere')
        self.assertEqual(obj.metadata['contributing_views'], ['A', 'B'])
        self.assertLess(np.linalg.norm(obj.pose.position-[.45, 0, .025]), .004)
        self.assertEqual(self.updater.health.status, 'fresh')

    def test_one_view_visible_object_is_not_a_missing_camera(self):
        frames = [analytic_frames([])[0], self.frames[1]]
        key = self.confirm(frames)
        self.assertEqual(self.model.require(key).metadata['contributing_views'], ['B'])
        self.assertEqual(self.model.require(key).metadata['camera_count'], 2)

    def test_separated_nearby_objects_and_outlier_suppression(self):
        frames = analytic_frames([Solid('sphere', center=(.45, -.045, .025)),
                                  Solid('sphere', center=(.45, .045, .025))])
        for _ in range(3):
            self.step(frames)
        self.assertEqual(len(self.model.objects()), 2, self.provider.diagnostics)
        cloud = np.r_[np.random.default_rng(1).normal([.4, 0, .02], .002, (50, 3)), [[.7, .3, .2]]]
        self.assertEqual(len(cluster_points(cloud)), 1)

    def test_frozen_frames_fail_without_refresh_and_recover_only_on_new_pair(self):
        key = self.confirm()
        seen = self.model.require(key).last_seen
        self.now.return_value += .2
        self.cameras[0].number += 1
        self.updater.update()
        self.assertEqual(self.updater.health.status, 'failed')
        self.assertEqual(self.model.require(key).last_seen, seen)
        self.assertFalse(self.model.require(key).visible)
        self.cameras[1].number += 1
        self.updater.update()
        self.assertEqual(self.updater.health.status, 'fresh')
        self.now.return_value += 2.1
        self.updater.update()
        self.assertFalse(self.model.exists(key))
        self.assertNotIn(key, self.provider.tracker.tracks)
        self.assertIsNone(self.provider.get_point_cloud(key))

    def test_held_and_uncertain_ids_survive_missing_observations_and_reacquire_by_hint(self):
        for uncertain in (False, True):
            self.now.return_value += 3.
            self.updater.cleanup()
            key = self.confirm(self.frames)
            self.model.set_held(key, uncertain=uncertain)
            self.now.return_value += 3.
            self.step(analytic_frames([]))
            self.assertTrue(self.model.exists(key))
            self.assertEqual(self.updater.clear_lost(), ())
            target = [.45, .25, .025]
            self.model.expect_object_at(key, target, ttl=10., reason='place')
            self.model.set_held(None)
            self.step(analytic_frames([Solid('sphere', center=tuple(target))]))
            self.assertEqual([o.object_id for o in self.model.objects()], [key])
            self.assertTrue(self.model.require(key).visible)
            self.assertFalse(self.model.association_hints())

    def test_pending_confirmation_needs_consecutive_distinct_samples(self):
        self.step()
        self.now.return_value += .2
        self.updater.update()
        self.assertFalse(self.model.objects())
        self.step(analytic_frames([]))
        self.step(self.frames)
        self.step()
        self.assertFalse(self.model.objects())
        self.step()
        self.assertEqual(len(self.model.objects()), 1)

    def test_supported_unknowns_remain_ungraspable_and_elevated_clutter_is_rejected(self):
        frames = analytic_frames([Solid('ellipsoid', size=(.08, .04, .05)),
            Solid('cylinder', center=(.55, .2, .13), radius=.015, length=.09, axis=(1, 0, 0))])
        key = self.confirm(frames)
        self.assertIsNone(self.model.require(key).class_name)
        self.assertFalse(self.model.require(key).graspable)
        self.assertTrue(any(r['reason'] == 'no_observed_support_contact'
                            for r in self.provider.diagnostics['rejected']))

    def test_updater_and_provider_own_cleanup_exactly_once(self):
        self.updater.stop()
        self.updater.stop()
        self.provider.close()
        self.assertEqual([c.closes for c in self.cameras], [1, 1])
        self.assertIsNone(self.provider.observe().timestamp)


if __name__ == '__main__':
    unittest.main()
