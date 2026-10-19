"""One-file setup parsing and preservation of the playground's effective values."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_perception_provider

PLAYGROUND = Path(__file__).resolve().parents[2]/'config/scenes/playground.toml'


class SetupConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'setup.toml'
            path.write_text(text)
            return load_scene_config(path)

    def test_playground_references_and_world_defaults(self):
        cfg = load_scene_config(PLAYGROUND)
        self.assertEqual(cfg.robot.backend, 'franka')
        self.assertEqual(cfg.robot.prim_path, '/Franka')
        self.assertEqual(cfg.world.provider, 'perception')
        self.assertEqual(cfg.world.stale_object_ttl, 15.)
        self.assertIsNone(cfg.world.update_hz)
        self.assertEqual(cfg.perception.camera_prim_path, '/World/PerceptionCamera')
        self.assertEqual(cfg.perception.resolution, (640, 480))
        self.assertEqual(cfg.ground_truth.objects_root, '/World/Objects')
        np.testing.assert_array_equal(cfg.ground_truth.objects['cube'].size, [.05]*3)
        self.assertTrue(cfg.ground_truth.objects['cube'].graspable)

    def test_effective_v1_parameters_are_preserved(self):
        cfg = load_scene_config(PLAYGROUND)
        source = build_perception_provider(NS(get_intrinsics=lambda: np.eye(3)), cfg)
        expected = {'support_plane_z': 0., 'min_object_height': .008, 'max_object_height': .22,
                    'min_component_pixels': 30, 'max_component_pixels': 30000,
                    'candidate_max_extent': .18, 'spawn_min_extent': .02, 'spawn_max_extent': .12,
                    'spawn_compactness_ratio': 3.5, 'support_contact_tolerance': .015,
                    'component_neighbor_distance': .02}
        for name, value in expected.items():
            self.assertEqual(getattr(source.discoverer, name), value, name)
        np.testing.assert_array_equal(source.discoverer.workspace_min, [.20, -.40, 0.])
        np.testing.assert_array_equal(source.discoverer.workspace_max, [.70, .40, .22])
        self.assertEqual((source.tracker.max_distance, source.tracker.max_size_ratio,
                          source.tracker.min_confirm_hits), (.18, 1.5, 3))
        self.assertEqual(vars(source.primitive_estimator), {
            'threshold': .58, 'margin': .07, 'top_band': .15, 'tolerance': .0008, 'max_points': 1600})

    def test_rgbd_does_not_require_ground_truth_configuration(self):
        text = PLAYGROUND.read_text().split('# Optional oracle')[0]
        cfg = self.load(text)
        self.assertEqual(cfg.ground_truth.objects, {})
        self.assertIsNone(cfg.ground_truth.objects_root)

    def test_ground_truth_does_not_require_camera_configuration(self):
        text = PLAYGROUND.read_text().replace('provider = "perception"', 'provider = "ground_truth"')
        text = text.replace('[perception]\ncamera_prim_path = "/World/PerceptionCamera"', '')
        cfg = self.load(text)
        self.assertIsNone(cfg.perception.camera_prim_path)
        self.assertEqual(cfg.ground_truth.objects['cube'].prim_path, '/World/Objects/Cube')

    def test_scene_properties_reach_perception_independently_of_algorithm_defaults(self):
        text = PLAYGROUND.read_text().replace('support_plane_z = 0.0', 'support_plane_z = 0.35')
        text = text.replace('[0.20, -0.40, 0.0]', '[0.20, -0.40, 0.35]')
        text = text.replace('[0.70, 0.40, 0.22]', '[0.70, 0.40, 0.57]')
        cfg = self.load(text)
        source = build_perception_provider(NS(get_intrinsics=lambda: np.eye(3)), cfg)
        self.assertEqual(source.discoverer.support_plane_z, .35)
        np.testing.assert_array_equal(source.discoverer.workspace_min, [.2, -.4, .35])
        self.assertEqual(source.discoverer.min_component_pixels, 30)

    def test_unknown_selectors_and_removed_options_are_rejected(self):
        text = PLAYGROUND.read_text()
        for invalid, message in (
            (text.replace('backend = "franka"', 'backend = "unsupported"'), 'Unsupported robot backend'),
            (text.replace('provider = "perception"', 'provider = "unsupported"'), 'Unsupported observation source'),
            (text+'\n[grasping]\napproach_height=0.1\n', 'Unsupported configuration sections'),
            (text.replace('[perception]', '[perception]\ntype="yolo"'), 'Invalid scene configuration'),
            (text.replace('[perception]', '[perception]\ntracking={max_distance=0.2}'), 'Invalid scene configuration'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load(invalid)

    def test_bad_scene_geometry_resolution_and_missing_references_are_rejected(self):
        text = PLAYGROUND.read_text()
        for before, after in (
            ('[0.20, -0.40, 0.0]', '[0.2, 0.3]'),
            ('[0.70, 0.40, 0.22]', '[0.1, 0.4, 0.22]'),
            ('support_plane_z = 0.0', 'support_plane_z = nan'),
            ('[perception]', '[perception]\nresolution=[640,0]'),
            ('camera_prim_path = "/World/PerceptionCamera"', ''),
            ('prim_path = "/Franka"', 'prim_path="relative"'),
        ):
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.load(text.replace(before, after))
        with self.assertRaises(ValueError):
            self.load(text.replace('provider = "perception"', 'provider = "ground_truth"').split('# Optional oracle')[0])

    def test_world_tuning_requires_positive_finite_values(self):
        text = PLAYGROUND.read_text()
        for key in ('update_hz', 'stale_object_ttl'):
            for value in ('0', '-1', 'nan', 'inf', '"bad"', 'true'):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'finite'):
                    self.load(text.replace('[world]', f'[world]\n{key}={value}'))
        cfg = self.load(text.replace('[world]', '[world]\nupdate_hz=3.0\nstale_object_ttl=10.0'))
        self.assertEqual((cfg.world.update_hz, cfg.world.stale_object_ttl), (3., 10.))
