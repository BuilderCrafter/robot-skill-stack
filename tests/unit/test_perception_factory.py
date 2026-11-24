"""Public semantic construction with caller-owned initialized camera resources."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from robot_skill_stack.world.perception import MultiViewConfig, build_perception_provider
from robot_skill_stack.world.perception.multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception.v1.provider import PerceptionStateProvider


class PerceptionFactoryTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(support_plane_z=.35,
            workspace_min=np.array([.2, -.4, .35]), workspace_max=np.array([.7, .4, .57]))

    def test_v1_uses_initialized_camera_and_scene(self):
        camera = Mock(get_intrinsics=Mock(return_value=np.eye(3)))
        provider = build_perception_provider('perception', camera, scene=self.scene)
        self.addCleanup(provider.close)
        self.assertIsInstance(provider, PerceptionStateProvider)
        self.assertIs(provider.camera, camera)
        self.assertEqual(provider.discoverer.support_plane_z, .35)
        np.testing.assert_array_equal(provider.discoverer.workspace_min, self.scene.workspace_min)
        camera.get_intrinsics.assert_called_once_with()
        camera.initialize.assert_not_called()
        camera.close.assert_not_called()

    def test_multiview_preserves_rig_and_forwards_default_or_explicit_settings(self):
        for settings in (None, MultiViewConfig(voxel_size=.003, stride=3)):
            with self.subTest(settings=settings):
                rig = Mock()
                provider = build_perception_provider('multi_view', rig, scene=self.scene, settings=settings)
                self.addCleanup(provider.close)
                self.assertIsInstance(provider, MultiViewProvider)
                self.assertIs(provider.rig, rig)
                expected = MultiViewConfig() if settings is None else settings
                self.assertEqual((provider.processor.voxel_size, provider.processor.stride),
                                 (expected.voxel_size, expected.stride))
                self.assertEqual(provider.processor.scene.support_plane_z, .35)
                np.testing.assert_array_equal(provider.processor.scene.workspace_max, self.scene.workspace_max)
                self.assertFalse(rig.mock_calls)

    def test_unsupported_semantic_selection_does_not_touch_source(self):
        source = Mock()
        for kind in ('ground_truth', 'unknown'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Unsupported.*'+kind):
                build_perception_provider(kind, source, scene=self.scene)
        self.assertFalse(source.mock_calls)
