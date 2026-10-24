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

    def test_named_future_strategies_fail_explicitly_without_fallback(self):
        for name in ('reactive', 'global', 'hybrid'):
            with self.subTest(strategy=name), self.assertRaisesRegex(
                    ValueError, rf'Unsupported motion strategy.*{name}.*not implemented'):
                self.load(f'[motion]\nstrategy = "{name}"\n')

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
        with self.assertRaisesRegex(ValueError, 'Unsupported motion strategy'):
            replace(config, strategy='reactive')

    def test_existing_positional_scene_config_construction_keeps_default(self):
        original = self.load('')
        config = SceneConfig(original.robot, original.world, original.scene,
            original.perception, original.ground_truth, original.multi_view,
            original.spatial_mapping)
        self.assertEqual(config.motion.strategy, 'direct')


if __name__ == '__main__':
    unittest.main()
