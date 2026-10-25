"""Motion selection validates before any SDK/controller construction."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

from robot_skill_stack.integrations.isaac.config import (
    MotionConfig, SceneConfig, load_scene_config,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'config/scenes/playground.toml'


class MotionConfigTests(unittest.TestCase):
    def load(self, section):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scene.toml'
            path.write_text(PROFILE.read_text()+'\n'+section)
            return load_scene_config(path)

    def test_missing_motion_section_defaults_to_direct(self):
        self.assertEqual(self.load('').motion.strategy, 'direct')

    def test_empty_motion_section_defaults_to_direct(self):
        self.assertEqual(self.load('[motion]\n').motion.strategy, 'direct')

    def test_explicit_direct_is_accepted(self):
        config = self.load('[motion]\nstrategy = "direct"\n')
        self.assertEqual(config.motion, MotionConfig())
        self.assertEqual(config.robot.backend, 'franka')
        self.assertEqual(config.world.provider, 'perception')
        self.assertFalse(config.spatial_mapping.enabled)

    def test_every_existing_scene_profile_keeps_implicit_direct(self):
        profiles = sorted((ROOT/'config/scenes').glob('*.toml'))
        self.assertTrue(profiles)
        for path in profiles:
            with self.subTest(profile=path.name):
                self.assertEqual(load_scene_config(path).motion.strategy, 'direct')

    def test_hybrid_fails_explicitly_without_fallback(self):
        with self.assertRaisesRegex(ValueError, r'Unsupported motion strategy.*hybrid.*not implemented'):
            self.load('[motion]\nstrategy = "hybrid"\n')

    def test_global_is_opt_in_without_spatial_mapping(self):
        config = self.load('[motion]\nstrategy = "global"\n')
        self.assertEqual(config.motion.strategy, 'global')
        self.assertFalse(config.spatial_mapping.enabled)
        self.assertEqual((config.motion.rrt_seed, config.motion.rrt_max_iterations), (123456, 10000))

    def test_global_seed_and_search_bound_load_explicitly(self):
        config = self.load('[motion]\nstrategy = "global"\nrrt_seed = 7\nrrt_max_iterations = 2000\n')
        self.assertEqual((config.motion.rrt_seed, config.motion.rrt_max_iterations), (7, 2000))

    def test_global_search_settings_reject_invalid_programmatic_values(self):
        for value in (0, -1, 2**31, True, 1.5, None, '4'):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, 'rrt_seed'):
                MotionConfig(strategy='global', rrt_seed=value)
        for value in (0, -1, 50001, True, 1.5, None, '100'):
            with self.subTest(iterations=value), self.assertRaisesRegex(ValueError, 'rrt_max_iterations'):
                MotionConfig(strategy='global', rrt_max_iterations=value)

    def test_global_search_settings_boundaries_are_accepted(self):
        for seed, iterations in ((1, 1), (2**31-1, 50000)):
            config = MotionConfig(strategy='global', rrt_seed=seed, rrt_max_iterations=iterations)
            self.assertEqual((config.rrt_seed, config.rrt_max_iterations), (seed, iterations))

    def test_unknown_names_are_distinct_from_unimplemented_strategies(self):
        for name in ('rrt', 'Direct', '', 'direct '):
            with self.subTest(strategy=name), self.assertRaisesRegex(
                    ValueError, 'Unknown motion strategy'):
                self.load(f'[motion]\nstrategy = "{name}"\n')

    def test_non_string_selector_has_clear_validation_error(self):
        for value in ('true', '1', '1.5', '[]', '{}'):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, r'motion.strategy must be a string'):
                self.load(f'[motion]\nstrategy = {value}\n')

    def test_unknown_motion_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid scene configuration'):
            self.load('[motion]\nstrategy = "direct"\nobstacles = true\n')

    def test_programmatic_config_validates_and_cannot_be_mutated(self):
        config = MotionConfig()
        with self.assertRaises(FrozenInstanceError):
            config.strategy = 'reactive'
        for value in (None, False, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, r'motion.strategy must be a string'):
                MotionConfig(value)
        self.assertEqual(replace(config, strategy='global').strategy, 'global')

    def test_reactive_requires_mapping_and_accepts_enabled_self_filtered_source(self):
        with self.assertRaisesRegex(ValueError, 'requires enabled Spatial Mapping'):
            self.load('[motion]\nstrategy = "reactive"\n')
        config = self.load('[spatial_mapping]\nenabled = true\nsource = "rgbd"\nautomatic_robot_filter = true\n'
                           '[motion]\nstrategy = "reactive"\nmax_obstacles = 32\n')
        self.assertEqual(config.motion.strategy, 'reactive')
        self.assertEqual(config.motion.max_obstacles, 32)

    def test_reactive_rejects_robot_evidence_and_invalid_proxy_budget(self):
        with self.assertRaisesRegex(ValueError, 'automatic robot self-filtering'):
            self.load('[spatial_mapping]\nenabled = true\nsource = "rgbd"\n'
                'automatic_robot_filter = false\n[motion]\nstrategy = "reactive"\n')
        with self.assertRaisesRegex(ValueError, 'exclude the robot'):
            self.load('[spatial_mapping]\nenabled = true\nsource = "ground_truth"\n'
                'excluded_prim_paths = []\n[motion]\nstrategy = "reactive"\n')
        for value in (0, 129, True, 2.5, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'max_obstacles'):
                MotionConfig(max_obstacles=value)

    def test_programmatic_reactive_scene_validation_matches_loaded_config(self):
        config = self.load('')
        with self.assertRaisesRegex(ValueError, 'requires enabled Spatial Mapping'):
            replace(config, motion=MotionConfig(strategy='reactive'))
        enabled = replace(config.spatial_mapping, enabled=True, source='ground_truth')
        selected = replace(config, motion=MotionConfig(strategy='reactive'), spatial_mapping=enabled)
        self.assertEqual(selected.motion.strategy, 'reactive')

    def test_existing_positional_scene_config_construction_keeps_default(self):
        original = self.load('')
        config = SceneConfig(original.robot, original.world, original.scene,
            original.perception, original.ground_truth, original.multi_view,
            original.spatial_mapping)
        self.assertEqual(config.motion.strategy, 'direct')


if __name__ == '__main__':
    unittest.main()
