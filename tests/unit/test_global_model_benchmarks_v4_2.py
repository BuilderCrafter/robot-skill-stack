"""Explicit experimental selection must not rewrite historical scene defaults."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robot_skill_stack.integrations.isaac.config import load_scene_config
from scripts import benchmark_global_models_v4_2 as models
from scripts import benchmark_global_lifecycle_v4_2 as lifecycle
from scripts import run_global_validation_v4_2_jobs as driver


class ModelBenchmarkTests(unittest.TestCase):
    def test_generated_selection_keeps_original_movement_and_mapping_settings(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(strategy='global', seed=234567, max_iterations=10000,
                artifacts_dir=folder, enable_mapping=False)
            for selected in ('stock', 'enhanced'):
                path = models.model_profile(args, models.fixture.write_profile, selected)
                config = load_scene_config(path)
                self.assertEqual(config.motion.strategy, 'global')
                self.assertEqual(config.motion.global_collision_model, selected)
                self.assertEqual(config.motion.rrt_seed, 234567)
                self.assertEqual(config.motion.rrt_max_iterations, 10000)
                self.assertFalse(config.spatial_mapping.enabled)

    def test_malformed_generated_profile_cannot_silently_ignore_selection(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'bad.toml'
            path.write_text('[scene]\n')
            with self.assertRaisesRegex(ValueError, 'one generated motion'):
                models.model_profile(None, lambda args: path, 'enhanced')

    def test_lifecycle_wrapper_verifies_real_model_and_restores_observer_hook(self):
        original = lifecycle.acceptance.controlled.make_observer
        bundle = SimpleNamespace(backend=SimpleNamespace(motion_strategy=SimpleNamespace(collision_model='stock')),
            config=SimpleNamespace(motion=SimpleNamespace(global_collision_model='stock')))
        def run(arguments):
            self.assertEqual(arguments, ['--mode', 'cancel'])
            with self.assertRaisesRegex(RuntimeError, 'different collision model'):
                lifecycle.acceptance.controlled.make_observer(bundle)
            return True
        with patch.object(lifecycle.acceptance, 'main', run):
            self.assertTrue(lifecycle.main(['--collision-model', 'enhanced', '--mode', 'cancel']))
        self.assertIs(lifecycle.acceptance.controlled.make_observer, original)

    def test_driver_scope_restores_after_failure(self):
        original = driver.runner.NATIVE, driver.runner.OUTPUT_ROOT
        def failure(arguments):
            self.assertEqual(driver.runner.NATIVE, 'scripts/benchmark_global_models_v4_2.py')
            self.assertEqual(driver.runner.OUTPUT_ROOT, Path('outputs/motion-planning-v4-2'))
            raise RuntimeError('Controlled driver failure')
        with patch.object(driver.runner, 'main', failure):
            with self.assertRaisesRegex(RuntimeError, 'Controlled driver'):
                driver.main(['--suite', 'static', '--jobs', 'unused.json'])
        self.assertEqual((driver.runner.NATIVE, driver.runner.OUTPUT_ROOT), original)
