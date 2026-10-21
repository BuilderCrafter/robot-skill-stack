"""Configured N-view geometry through the ordinary synchronizer and updater."""
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from tests.support.multiview_research import analytic_frames
from tests.support.primitive_scene import Solid
from tests.unit.test_multiview_provider import Camera


class VariableCameraProviderTests(unittest.TestCase):
    def setUp(self):
        clock = patch('time.monotonic', return_value=100.)
        self.now = clock.start()
        self.addCleanup(clock.stop)
        self.scene = load_scene_config('config/scenes/multiview_playground.toml').scene
        self.solids = (
            Solid('cube', center=(.47, -.06, .0225), size=(.055, .04, .045), yaw=.25),
            Solid('sphere', center=(.29, .31, .023), radius=.023),
        )
        # Same physical scene in every image. The sphere is outside A/B's
        # actual image frustums, C sees it, and D sees only the valid table.
        self.frames = analytic_frames(self.solids, eyes=(
            (.75, .10, .32), (.75, .25, .32), (.25, .85, .65),
            (.45, 0., .065), (.18, -.75, .60)))

    def source(self, frames, names):
        cameras = {name: Camera(frame) for name, frame in zip(names, frames)}
        provider = MultiViewProvider(FrameSynchronizer(cameras), self.scene)
        model = WorldModel()
        updater = WorldModelUpdater(model, provider)
        self.addCleanup(updater.stop)
        return cameras, provider, model, updater

    def step(self, source):
        cameras, _, _, updater = source
        self.now.return_value += .2
        for camera in cameras.values():
            camera.number += 1
        observations = updater.update()
        self.assertEqual(updater.health.status, 'fresh', updater.last_error)
        return observations

    def confirm(self, source, count):
        for _ in range(2):
            self.assertEqual(self.step(source), [])
            self.assertEqual(source[2].objects(), ())
        self.assertEqual(len(self.step(source)), count)
        self.assertEqual(len(source[2].objects()), count)
        return {obj.class_name: obj for obj in source[2].objects()}

    def cloud(self, frame):
        return crop_workspace(project_depth(frame)[0], self.scene)

    def assert_geometry(self, objects):
        self.assertEqual(set(objects), {'cube', 'sphere'})
        for solid in self.solids:
            obj = objects[solid.shape]
            self.assertTrue(obj.visible)
            self.assertLess(np.linalg.norm(obj.pose.position-solid.center), .003)
            if solid.shape == 'cube':
                np.testing.assert_allclose(obj.geometry.box_size, solid.size, atol=.003, rtol=0)
            else:
                self.assertAlmostEqual(obj.geometry.radius, solid.radius, delta=.002)
                self.assertIsNone(obj.pose.orientation)

    def assert_fusion_and_limits(self, source, frames):
        _, provider, model, _ = source
        expected = fuse_clouds([self.cloud(frame) for frame in frames], voxel=.002, max_points=12000)
        actual = np.concatenate([provider.get_point_cloud(obj.object_id) for obj in model.objects()])
        # This also rejects silently ignored cameras, duplicated measurements,
        # and extra clouds not originating in the calibrated raw depths.
        order = lambda points: points[np.lexsort(points.T[::-1])]
        np.testing.assert_allclose(order(actual), order(expected), atol=1e-12, rtol=0)
        self.assertEqual(provider.diagnostics['workspace_points'], len(expected))
        self.assertEqual(provider.max_points, 12000)
        self.assertLessEqual(len(actual), 12000)
        self.assertEqual(provider.diagnostics['fit_budget'], 24)
        self.assertLessEqual(provider.diagnostics['numerical_fits'], 24)

    def assert_view_evidence(self, objects, names):
        for obj in objects.values():
            self.assertEqual(obj.metadata['camera_count'], len(names))
            self.assertTrue(set(obj.metadata['contributing_views']).issubset(names))
        evidence = objects['cube'].geometry.metadata['geometric_observability']
        for axis in evidence['axes'][:2]:
            for side in ('negative', 'positive'):
                self.assertEqual([view['camera_id'] for view in axis[side]['cameras']], list(names))

    def test_three_views_include_third_only_object_and_preserve_ids_when_reordered(self):
        frames = self.frames[:3]
        for frame in frames[:2]:
            points = self.cloud(frame)
            on_sphere = abs(np.linalg.norm(points-self.solids[1].center, axis=1)-.023) < .002
            self.assertFalse(on_sphere.any())
        self.assertGreater(len(self.cloud(frames[2])), 30)
        source = self.source(frames, ('A', 'B', 'C'))
        objects = self.confirm(source, 2)
        self.assert_geometry(objects)
        self.assertEqual(objects['cube'].metadata['contributing_views'], ['A', 'B', 'C'])
        self.assertEqual(objects['sphere'].metadata['contributing_views'], ['C'])
        self.assert_view_evidence(objects, ('A', 'B', 'C'))
        self.assert_fusion_and_limits(source, frames)
        identities = {shape: obj.object_id for shape, obj in objects.items()}
        rig = source[1].rig
        rig.cameras = {name: rig.cameras[name] for name in ('C', 'A', 'B')}
        self.assertEqual(len(self.step(source)), 2)
        objects = {obj.class_name: obj for obj in source[2].objects()}
        self.assertEqual({shape: obj.object_id for shape, obj in objects.items()}, identities)
        self.assert_geometry(objects)
        self.assert_view_evidence(objects, ('C', 'A', 'B'))
        self.assertEqual(objects['sphere'].metadata['contributing_views'], ['C'])
        self.assert_fusion_and_limits(source, frames)

    def test_valid_empty_camera_preserves_two_camera_geometry_and_cloud(self):
        empty = self.frames[3]
        self.assertTrue(np.isfinite(empty.depth).all())
        self.assertTrue((empty.depth > 0).all())
        self.assertEqual(len(self.cloud(empty)), 0)
        original = self.source(self.frames[:2], ('A', 'B'))
        extended = self.source(self.frames[:2]+[empty], ('A', 'B', 'D'))
        before = self.confirm(original, 1)['cube']
        after = self.confirm(extended, 1)['cube']
        self.assertEqual(before.object_id, after.object_id)
        np.testing.assert_array_equal(before.pose.position, after.pose.position)
        np.testing.assert_array_equal(before.geometry.box_size, after.geometry.box_size)
        np.testing.assert_array_equal(original[1].get_point_cloud(before.object_id),
                                      extended[1].get_point_cloud(after.object_id))
        self.assertEqual(before.metadata['camera_count'], 2)
        self.assertEqual(after.metadata['camera_count'], 3)
        self.assertEqual(after.metadata['contributing_views'], ['A', 'B'])
        self.assertEqual(extended[3].health.status, 'fresh')

    def test_five_views_keep_empty_camera_valid_and_do_not_duplicate_objects(self):
        source = self.source(self.frames, ('A', 'B', 'C', 'D', 'E'))
        objects = self.confirm(source, 2)
        self.assert_geometry(objects)
        self.assertEqual(objects['cube'].metadata['contributing_views'], ['A', 'B', 'C', 'E'])
        self.assertEqual(objects['sphere'].metadata['contributing_views'], ['C', 'E'])
        self.assert_view_evidence(objects, ('A', 'B', 'C', 'D', 'E'))
        self.assert_fusion_and_limits(source, self.frames)
        self.assertEqual(set(source[1].rig.status['tokens']), {'A', 'B', 'C', 'D', 'E'})


if __name__ == '__main__':
    unittest.main()
