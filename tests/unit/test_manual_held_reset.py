from __future__ import annotations

import asyncio
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception.factory import build_perception_provider
from tests.support.primitive_scene import FrameCamera, Solid, render
from tests.unit import test_grasp_attachment as attachment_tests
from tests.unit.test_grasp_clearance import solid


class ManualHeldResetTests(unittest.TestCase):
    def setUp(self):
        self.wm = WorldModel()
        self.obj = WorldObject('object_1', pose=Pose([.45, 0, .025]), size=[.05]*3,
                               visible=True, last_seen=123., metadata={'keep': 'data'})
        self.wm.register(self.obj)
        self.wm.register(WorldObject('object_2', pose=Pose([.45, .2, .025])))
        self.wm.set_held('object_1', object_offset_in_ee=[0, 0, .03])

    def test_clears_id_and_attachment(self):
        self.assertEqual(self.wm.clear_held_state(), 'object_1')
        self.assertIsNone(self.wm.held_object_id)
        self.assertIsNone(self.wm.held_object_offset_in_ee)

    def test_idempotent(self):
        self.wm.clear_held_state()
        self.assertIsNone(self.wm.clear_held_state())

    def test_dangling_attachment_is_safe_to_clear(self):
        self.wm.held_object_id = 'no-longer-present'
        self.assertEqual(self.wm.clear_held_state(), 'no-longer-present')
        self.assertIsNone(self.wm.held_object_offset_in_ee)
        self.assertEqual(len(self.wm.objects()), 2)

    def test_orphan_offset_is_cleared_even_without_id(self):
        self.wm.held_object_id = None
        self.wm.clear_held_state()
        self.assertIsNone(self.wm.held_object_offset_in_ee)

    def test_only_held_objects_hint_is_cleared(self):
        self.wm.expect_object_at('object_1', [.45, 0, .1], reason='pick')
        other = self.wm.expect_object_at('object_2', [.45, .25, .025], reason='place')
        self.wm.clear_held_state()
        self.assertEqual(self.wm.association_hints(), (other,))

    def test_observation_context_immediately_drops_protection(self):
        self.wm.expect_object_at('object_1', [.45, 0, .1], ttl=100.)
        self.wm.clear_held_state()
        context = self.wm.observation_context()
        self.assertIsNone(context.held_object_id)
        self.assertEqual(context.association_hints, ())
        self.assertNotIn('object_1', self.wm._protected())

    def test_object_and_observed_geometry_are_not_modified(self):
        pose, size, metadata = self.obj.pose, self.obj.size.copy(), self.obj.metadata.copy()
        self.wm.clear_held_state()
        self.assertIs(self.wm.require('object_1'), self.obj)
        self.assertIs(self.obj.pose, pose)
        np.testing.assert_array_equal(self.obj.size, size)
        self.assertEqual(self.obj.metadata, metadata)
        self.assertEqual(self.obj.last_seen, 123.)
        self.assertTrue(self.obj.visible)

    def test_ground_truth_object_is_not_deleted(self):
        self.obj.source = 'ground_truth'
        self.obj.visible = False
        self.wm.clear_held_state()
        self.assertNotIn('object_1', self.wm.expire_stale(now=99999.))
        self.assertTrue(self.wm.exists('object_1'))

    def perception(self):
        frame = render([Solid('sphere')])
        camera = FrameCamera(frame)
        config = load_scene_config('config/scenes/playground.toml')
        config = replace(config, world=replace(config.world, stale_object_ttl=2.))
        provider = build_perception_provider(camera, config)
        model = WorldModel(stale_object_ttl=2.)
        updater = WorldModelUpdater(model, provider)
        for _ in range(3):
            camera.push(frame)
            updater.update()
        object_id = model.objects()[0].object_id
        model.set_held(object_id, object_offset_in_ee=[0, 0, .03])
        return camera, model, provider, updater, object_id, frame

    def test_tracker_id_and_geometry_survive_reset_until_normal_expiry(self):
        with patch('time.monotonic', return_value=100.):
            camera, model, provider, updater, oid, frame = self.perception()
            track = provider.tracker.get(oid)
            model.clear_held_state()
            self.assertIs(provider.tracker.get(oid), track)
            camera.push(frame)
            updater.update()
            self.assertTrue(model.require(oid).visible)
            self.assertEqual(len(provider.tracker.tracks()), 1)
            self.assertIsNotNone(provider.get_point_cloud(oid))

    def test_unseen_slipped_object_can_now_expire_from_tracker_and_world(self):
        with patch('time.monotonic', return_value=100.) as clock:
            camera, model, provider, updater, oid, _ = self.perception()
            model.expect_object_at(oid, [.45, 0, .2], ttl=100., reason='pick')
            camera.push(render([])); updater.update()
            clock.return_value = 120.
            updater.cleanup()
            self.assertTrue(model.exists(oid))
            model.clear_held_state()
            updater.cleanup()
            self.assertFalse(model.exists(oid))
            self.assertIsNone(provider.tracker.get(oid))
            self.assertIsNone(provider.get_point_cloud(oid))

    def test_reset_does_not_reassert_stale_held_state_on_next_observation(self):
        with patch('time.monotonic', return_value=100.):
            camera, model, _, updater, oid, frame = self.perception()
            model.clear_held_state()
            for _ in range(4):
                camera.push(frame); updater.update()
            self.assertIsNone(model.held_object_id)
            self.assertIsNone(model.held_object_offset_in_ee)
            self.assertTrue(model.exists(oid))

    def test_sync_and_async_pick_another_object_after_manual_reset(self):
        helper = attachment_tests.AttachmentTests()
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                wm, backend, runtime = helper.stack(solid(), asynchronous)
                invoke = lambda name, **kw: helper.invoke(runtime, asynchronous, name, **kw)
                self.assertTrue(invoke('pick', object_id='target').ok)
                backend.closed = False  # Test double: a slip followed by a fresh observed table pose.
                wm.require("target").pose = Pose([.7, -.25, .05])
                next_object = solid('cube'); next_object.object_id = 'next'
                wm.register(next_object)
                backend.obj = next_object
                self.assertFalse(invoke('pick', object_id='next').ok)
                calls = len(backend.calls)
                wm.clear_held_state()
                self.assertEqual(len(backend.calls), calls)
                picked = invoke('pick', object_id='next')
                self.assertTrue(picked.ok, picked)
                self.assertEqual(wm.held_object_id, 'next')

    def test_place_after_reset_refuses_without_any_motion(self):
        helper = attachment_tests.AttachmentTests()
        wm, backend, runtime = helper.stack()
        wm.set_held('target', object_offset_in_ee=[0, 0, .03])
        wm.clear_held_state()
        result = runtime.execute('place', target=Pose([.45, .25, .05]))
        self.assertFalse(result.ok)
        self.assertEqual(backend.calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
