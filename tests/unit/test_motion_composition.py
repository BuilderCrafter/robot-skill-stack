"""Select strategies before SDK construction and resolve public maps after startup."""
import asyncio
from dataclasses import replace
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from robot_skill_stack.integrations.isaac.config import MotionConfig
from tests.unit.test_runtime_bootstrap import bootstrap_fixture


DIRECT_MODULE = 'robot_skill_stack.integrations.isaac.manipulation.motion.direct'


def direct_double():
    module = ModuleType(DIRECT_MODULE)
    module.DirectRmpflowStrategy = Mock(side_effect=lambda backend, **kwargs:
        NS(backend=backend, **kwargs))
    return module


class MotionCompositionTests(unittest.TestCase):
    def test_factory_injects_direct_with_original_controller_settings(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            direct = direct_double()
            owner = object()
            source = object()
            resolver = lambda: source
            with patch.dict('sys.modules', {DIRECT_MODULE: direct}):
                for motion in (None, MotionConfig(strategy='direct')):
                    bootstrap.create_backend('world', 'robot', config.robot, motion=motion,
                        mapping_source_provider=resolver)
                    options = dict(env.backend_calls[-1])
                    selected = options.pop('strategy_factory')(owner)
                    self.assertEqual(options, dict(world='world', robot='robot',
                        position_tolerance=.01, orientation_tolerance=.05, max_motion_steps=1000))
                    self.assertIs(selected.backend, owner)
                    self.assertIs(selected.mapping_source_provider, resolver)
            self.assertEqual(direct.DirectRmpflowStrategy.call_count, 2)

    def test_factory_rejects_invalid_programmatic_strategy_before_sdk_construction(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            for name in ('reactive', 'global', 'hybrid', 'typo', None, 1):
                with self.subTest(strategy=name), self.assertRaises(ValueError):
                    bootstrap.create_backend(None, None, config.robot, motion=NS(strategy=name))
            self.assertEqual(env.backend_calls, [])

    def test_disabled_mapping_dependency_is_explicitly_absent(self):
        async def run():
            with bootstrap_fixture() as (bootstrap, env):
                bundle = await bootstrap.build_runtime('profile.toml')
                direct = direct_double()
                with patch.dict('sys.modules', {DIRECT_MODULE: direct}):
                    strategy = env.backend_calls[-1]['strategy_factory'](bundle.backend)
                self.assertIsNone(strategy.mapping_source_provider)
                self.assertIsNone(bundle.spatial_mapping)
                bundle.close()
        asyncio.run(run())

    def test_enabled_mapping_resolves_later_service_without_reordering_startup(self):
        async def run():
            with bootstrap_fixture() as (bootstrap, env):
                config = bootstrap.load_scene_config('profile.toml')
                config = replace(config, spatial_mapping=replace(config.spatial_mapping,
                    enabled=True, source='ground_truth'))
                view = object()
                service = NS(snapshot=Mock(return_value=view),
                    health={'fresh': True, 'timestamp': 12., 'generation': 7})
                coordinator = NS(spatial_mapping=service, _semantic_worker=None,
                                 start=Mock(), close=Mock())
                captured = []
                direct = direct_double()

                async def construct_observations(*args):
                    # Backend/Home reference already exists; mapping is not bound yet.
                    selected = env.backend_calls[-1]['strategy_factory'](object())
                    self.assertIsNone(selected.mapping_source_provider())
                    captured.append(selected.mapping_source_provider)
                    return coordinator

                with patch.object(bootstrap, 'load_scene_config', return_value=config), \
                     patch.dict('sys.modules', {DIRECT_MODULE: direct}), \
                     patch('robot_skill_stack.integrations.isaac.observation_runtime.create_observation_runtime',
                           side_effect=construct_observations):
                    bundle = await bootstrap.build_runtime('profile.toml')
                service.snapshot.assert_not_called()
                self.assertIs(captured[0](), bundle.spatial_mapping)
                self.assertIs(captured[0]().health, service.health)
                self.assertIs(captured[0]().snapshot(), view)
                self.assertEqual(captured[0]().health['generation'], 7)
                bundle.close()
                coordinator.close.assert_called_once()
                coordinator.start.assert_called_once()
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
