from __future__ import annotations
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.presentation.world_model import WorldModelViewModel
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception.discovery import ObjectCandidate
from robot_skill_stack.world.perception.semantics import SemanticPrediction
from tests.support.primitive_scene import FrameCamera, Solid, render


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sphere = render([Solid('sphere')])
        cls.empty = render([])

    def setUp(self):
        self.clock = patch('time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.camera = FrameCamera(self.sphere)
        cfg = load_scene_config('config/scenes/playground.toml')
        cfg = replace(cfg, world=replace(cfg.world, stale_object_ttl=2.))
        self.provider = build_perception_provider(self.camera, cfg)
        self.model = WorldModel(stale_object_ttl=2.)
        self.updater = WorldModelUpdater(self.model, self.provider, update_hz=5)

    def advance(self, seconds):
        self.now.return_value += seconds

    def step(self, frame=None, seconds=.2):
        self.advance(seconds)
        if frame is not None:
            self.camera.push(frame)
        return self.updater.update()

    def confirmed(self):
        for _ in range(3):
            self.step(self.sphere)
        self.assertEqual(len(self.model.objects()), 1)
        return self.model.objects()[0].object_id

    def assert_forgotten(self, object_id):
        self.assertFalse(self.model.exists(object_id))
        self.assertIsNone(self.provider.tracker.get(object_id))
        self.assertIsNone(self.provider.get_mask(object_id))
        self.assertIsNone(self.provider.get_point_cloud(object_id))
        self.assertNotIn(object_id, self.model._registered_at)
        self.assertNotIn(object_id, self.model._association_hints)

    def test_confirmation_requires_distinct_camera_frames(self):
        self.step(self.sphere)
        self.step()
        self.step()
        self.assertFalse(self.model.objects())
        self.assertEqual(self.provider.tracker.tracks()[0].hits, 1)
        self.step(self.sphere)
        self.assertFalse(self.model.objects())
        self.step(self.sphere)
        self.assertEqual(len(self.model.objects()), 1)

    def test_confirmation_requires_consecutive_hits(self):
        self.step(self.sphere)
        self.step(self.empty)
        self.step(self.sphere)
        self.step(self.sphere)
        self.assertFalse(self.model.objects())
        self.step(self.sphere)
        self.assertEqual(len(self.model.objects()), 1)

    def test_automatic_expiry_removes_tracker_world_and_lazy_geometry(self):
        object_id = self.confirmed()
        self.step(self.empty)
        self.assertFalse(self.model.require(object_id).visible)
        self.advance(2.1)
        self.updater.cleanup()
        self.assert_forgotten(object_id)
        new_id = self.confirmed()
        self.assertNotEqual(new_id, object_id)

    def test_frozen_frame_does_not_keep_objects_alive_or_resurrect_them(self):
        object_id = self.confirmed()
        last_seen = self.model.require(object_id).last_seen
        self.step(seconds=1.1)
        self.assertFalse(self.model.require(object_id).visible)
        self.assertEqual(self.model.require(object_id).last_seen, last_seen)
        self.step(seconds=1.1)
        self.assert_forgotten(object_id)
        for _ in range(4):
            self.step()
        self.assertFalse(self.model.objects())

    def test_missing_depth_does_not_repeat_visible_observations(self):
        object_id = self.confirmed()
        self.camera.get_depth = lambda: None
        self.step(seconds=1.1)
        self.assertFalse(self.model.require(object_id).visible)
        self.step(seconds=1.1)
        self.assert_forgotten(object_id)

    def test_cleanup_without_physics_callbacks(self):
        object_id = self.confirmed()
        self.advance(2.1)
        self.updater.cleanup()
        self.assert_forgotten(object_id)

    def test_held_object_and_track_are_protected(self):
        object_id = self.confirmed()
        self.model.set_held(object_id)
        self.step(self.empty, seconds=10.)
        self.assertTrue(self.model.exists(object_id))
        self.assertIsNotNone(self.provider.tracker.get(object_id))
        self.assertEqual(self.updater.clear_lost(), ())
        self.model.set_held(None)
        self.updater.cleanup()
        self.assert_forgotten(object_id)

    def test_active_place_hint_protects_until_expiry(self):
        object_id = self.confirmed()
        self.model.expect_object_at(object_id, [.45, .25, .025], ttl=10., reason='place')
        self.step(self.empty, seconds=3.)
        self.assertTrue(self.model.exists(object_id))
        self.assertIsNotNone(self.provider.tracker.get(object_id))
        self.assertEqual(self.updater.clear_lost(), ())
        self.advance(8.)
        self.updater.cleanup()
        self.assert_forgotten(object_id)

    def test_long_distance_place_reacquisition_preserves_id(self):
        object_id = self.confirmed()
        self.model.set_held(object_id)
        self.step(self.empty, seconds=3.)
        target = [.45, .25, .025]
        self.model.expect_object_at(object_id, target, ttl=10., reason='place')
        self.model.set_held(None)
        self.step(render([Solid('sphere', center=tuple(target))]))
        self.assertEqual([o.object_id for o in self.model.objects()], [object_id])
        self.assertTrue(self.model.require(object_id).visible)
        self.assertFalse(self.model.association_hints())

    def test_clear_lost_removes_both_stores(self):
        object_id = self.confirmed()
        self.step(self.empty)
        self.assertEqual(self.updater.clear_lost(), (object_id,))
        self.assert_forgotten(object_id)
        self.step()
        self.assertFalse(self.model.objects())

    def test_clear_lost_expires_old_hints_first(self):
        object_id = self.confirmed()
        self.model.expect_object_at(object_id, [.45, 0, .025], ttl=.1)
        self.step(self.empty)
        self.assertEqual(self.updater.clear_lost(), (object_id,))
        self.assert_forgotten(object_id)

    def test_ground_truth_not_forgotten(self):
        self.model.register(WorldObject('configured', visible=False, source='ground_truth'))
        self.advance(100.)
        self.assertFalse(self.updater.cleanup())
        self.assertFalse(self.updater.clear_lost())
        self.assertTrue(self.model.exists('configured'))

    def test_never_seen_record_can_expire(self):
        self.model.register(WorldObject('lost', visible=False))
        self.advance(3.)
        self.updater.cleanup()
        self.assert_forgotten('lost')

    def test_live_frames_keep_same_object(self):
        object_id = self.confirmed()
        for _ in range(12):
            self.step(self.sphere)
        self.assertEqual([o.object_id for o in self.model.objects()], [object_id])
        self.assertTrue(self.model.require(object_id).visible)

    def test_dissimilar_nonspawnable_junk_cannot_refresh_a_track(self):
        object_id = self.confirmed()
        track = self.provider.tracker.get(object_id)
        seen = track.last_seen
        junk = ObjectCandidate(track.position, [.15, .15, .15], np.ones((3, 3), bool), spawnable=False)
        self.provider.tracker.update([junk], [SemanticPrediction(None, 1.)], frame_id=99, timestamp=seen+.2)
        self.assertEqual(track.last_seen, seen)
        self.assertFalse(track.visible)

    def test_geometry_only_changes_refresh_view(self):
        geometry = PrimitiveGeometry(PrimitiveShape.CUBE, 1., box_size=[.05]*3, yaw=0.)
        obj = WorldObject('cube', pose=Pose([.45, 0, .025]), size=np.array([.05]*3), geometry=geometry)
        self.model.register(obj)
        view = WorldModelViewModel(self.model)
        before = view.signature()
        geometry.yaw = np.deg2rad(30)
        self.assertNotEqual(view.signature(), before)
        self.assertIn('30.0', view.rows()[0].primitive_detail)
        obj.geometry = PrimitiveGeometry(PrimitiveShape.CYLINDER, 1., radius=.02, length=.10, axis=[1, 0, 0])
        before = view.signature()
        obj.geometry.axis = np.array([0, 1, 0])
        self.assertNotEqual(view.signature(), before)
        before = view.signature()
        obj.geometry.radius = .025
        self.assertNotEqual(view.signature(), before)

    def test_cube_yaw_updates_through_full_pipeline(self):
        for _ in range(3):
            self.step(render([Solid('cube')]))
        object_id = self.model.objects()[0].object_id
        for yaw in (.52, 1.04, .03):
            self.step(render([Solid('cube', yaw=yaw)]))
            obj = self.model.require(object_id)
            self.assertEqual(obj.geometry.shape.value, 'cube')
            error = abs((obj.geometry.yaw-yaw+np.pi/4) % (np.pi/2)-np.pi/4)
            self.assertLess(error, np.deg2rad(5))

    def test_shape_updates_do_not_inherit_old_cube_label(self):
        for _ in range(3):
            self.step(render([Solid('cube')]))
        object_id = self.model.objects()[0].object_id
        self.step(self.sphere)
        self.assertEqual(self.model.require(object_id).class_name, 'sphere')
        self.step(render([Solid('cylinder', center=(.45, 0, .05), radius=.02)]))
        self.assertEqual(self.model.require(object_id).class_name, 'cylinder')

    def test_multiple_primitives_and_unknown_reach_world_model(self):
        solids = [Solid('cube', center=(.34, -.15, .025), yaw=.52),
                  Solid('sphere'), Solid('cylinder', center=(.52, .18, .05), radius=.02),
                  Solid('ellipsoid', center=(.6, -.15, .025), size=(.08, .05, .05))]
        frame = render(solids)
        for _ in range(3):
            self.step(frame)
        self.assertEqual(len(self.model.objects()), 4)
        self.assertEqual({o.class_name for o in self.model.objects()}, {'cube', 'sphere', 'cylinder', None})
        unknown = next(o for o in self.model.objects() if o.class_name is None)
        self.assertTrue(unknown.visible)
        self.assertEqual(unknown.geometry.shape.value, 'unknown')

    def test_perceived_sphere_pick_uses_async_backend_only(self):
        import asyncio
        from tests.unit.test_async_runtime import FakeAsyncBackend
        from robot_skill_stack.manipulation.grasping import SimplePrimitiveGraspPlanner
        from robot_skill_stack.manipulation.skills.pick import PickSkill
        from robot_skill_stack.runtime.registry import SkillRegistry
        from robot_skill_stack.runtime.runtime import RobotRuntime
        object_id = self.confirmed()
        backend = FakeAsyncBackend()  # Synchronous methods raise AssertionError.
        registry = SkillRegistry()
        registry.register(PickSkill(backend, self.model, SimplePrimitiveGraspPlanner()))
        result = asyncio.run(RobotRuntime(registry).execute_async('pick', object_id=object_id))
        self.assertTrue(result.ok, result.message)
        self.assertEqual(self.model.held_object_id, object_id)


if __name__ == '__main__':
    unittest.main(verbosity=2)
