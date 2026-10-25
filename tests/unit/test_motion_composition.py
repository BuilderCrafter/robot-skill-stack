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
    def test_enhanced_model_is_passed_only_to_selected_global_strategy(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            module_name = DIRECT_MODULE.replace('.direct', '.global_rrt')
            global_motion = ModuleType(module_name)
            global_motion.GlobalRrtStrategy = Mock(return_value=object())
            direct = direct_double()
            with patch.dict('sys.modules', {module_name: global_motion, DIRECT_MODULE: direct}):
                backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                    motion=MotionConfig(strategy='global', global_collision_model='enhanced'))
                env.backend_calls[-1]['strategy_factory'](backend)
                global_motion.GlobalRrtStrategy.assert_called_once_with(backend,
                    seed=123456, max_iterations=10000, collision_model='enhanced')
                direct_backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                    motion=MotionConfig(strategy='direct', global_collision_model='enhanced'))
                env.backend_calls[-1]['strategy_factory'](direct_backend)
                direct.DirectRmpflowStrategy.assert_called_once_with(direct_backend,
                    mapping_source_provider=None)

    def test_global_factory_preserves_seed_and_iteration_bound_without_mapping_access(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            module_name = DIRECT_MODULE.replace('.direct', '.global_rrt')
            global_motion = ModuleType(module_name)
            global_motion.GlobalRrtStrategy = Mock(side_effect=lambda backend, **kwargs:
                NS(backend=backend, **kwargs))
            resolver = Mock(side_effect=AssertionError('Global motion must not request occupancy'))
            with patch.dict('sys.modules', {module_name: global_motion}):
                backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                    motion=MotionConfig(strategy='global', max_obstacles=1,
                        rrt_seed=7, rrt_max_iterations=2500),
                    mapping_source_provider=resolver)
                global_motion.GlobalRrtStrategy.assert_not_called()
                selected = env.backend_calls[-1]['strategy_factory'](backend)
            self.assertIs(selected.backend, backend)
            global_motion.GlobalRrtStrategy.assert_called_once_with(backend,
                seed=7, max_iterations=2500)
            resolver.assert_not_called()

    def test_global_factory_accepts_static_camera_free_scene_and_missing_mapping_provider(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            config = replace(config, motion=MotionConfig(strategy='global'),
                perception=replace(config.perception, camera_prim_path=None))
            self.assertEqual(config.world.provider, 'ground_truth')
            self.assertFalse(config.spatial_mapping.enabled)
            module_name = DIRECT_MODULE.replace('.direct', '.global_rrt')
            global_motion = ModuleType(module_name)
            global_motion.GlobalRrtStrategy = Mock(return_value=object())
            with patch.dict('sys.modules', {module_name: global_motion}):
                backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                    motion=config.motion)
                env.backend_calls[-1]['strategy_factory'](backend)
            global_motion.GlobalRrtStrategy.assert_called_once_with(backend,
                seed=123456, max_iterations=10000)

    def test_global_factory_validates_programmatic_search_settings_before_sdk_construction(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            for options in (dict(rrt_seed=True), dict(rrt_seed=-1),
                            dict(rrt_max_iterations=0), dict(rrt_max_iterations=1.5),
                            dict(global_collision_model='bad')):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    bootstrap.create_backend(None, None, config.robot,
                        motion=NS(strategy='global', **options))
            self.assertEqual(env.backend_calls, [])

    def test_existing_strategies_never_construct_global_resources(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            module_name = DIRECT_MODULE.replace('.direct', '.global_rrt')
            global_motion = ModuleType(module_name)
            global_motion.GlobalRrtStrategy = Mock(side_effect=AssertionError('Unexpected RRT construction'))
            direct = direct_double()
            reactive_name = DIRECT_MODULE.replace('.direct', '.reactive')
            reactive = ModuleType(reactive_name)
            reactive.ReactiveRmpflowStrategy = Mock(return_value=object())
            with patch.dict('sys.modules', {module_name: global_motion, DIRECT_MODULE: direct,
                                          reactive_name: reactive}):
                for motion in (None, MotionConfig(strategy='direct'), MotionConfig(strategy='reactive')):
                    backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                        motion=motion, mapping_source_provider=Mock())
                    env.backend_calls[-1]['strategy_factory'](backend)
            global_motion.GlobalRrtStrategy.assert_not_called()

    def test_reactive_factory_selects_reactive_without_querying_mapping_during_construction(self):
        with bootstrap_fixture() as (bootstrap, env):
            config = bootstrap.load_scene_config('profile.toml')
            module_name = DIRECT_MODULE.replace('.direct', '.reactive')
            reactive = ModuleType(module_name)
            reactive.ReactiveRmpflowStrategy = Mock(side_effect=lambda backend, **kwargs:
                NS(backend=backend, **kwargs))
            resolver = Mock(side_effect=AssertionError('Source must remain late-bound'))
            with patch.dict('sys.modules', {module_name: reactive}):
                backend, _ = bootstrap.create_backend('world', 'robot', config.robot,
                    motion=MotionConfig(strategy='reactive', max_obstacles=32),
                    mapping_source_provider=resolver)
                selected = env.backend_calls[-1]['strategy_factory'](backend)
            self.assertIs(selected.backend, backend)
            self.assertIs(selected.mapping_source_provider, resolver)
            self.assertEqual(selected.max_obstacles, 32)
            reactive.ReactiveRmpflowStrategy.assert_called_once()
            resolver.assert_not_called()

    def test_only_reactive_mapping_enables_worker_obstacle_product(self):
        async def run():
            with bootstrap_fixture() as (bootstrap, env):
                from robot_skill_stack.integrations.isaac.observation_runtime import create_observation_runtime
                from robot_skill_stack.runtime.process_worker import ProcessWorker
                config = bootstrap.load_scene_config('profile.toml')
                mapping = replace(config.spatial_mapping, enabled=True, source='ground_truth')
                world = NS(is_playing=lambda: True)
                updater = NS(provider=NS())
                for name in ('direct', 'reactive', 'global'):
                    selected = replace(config, spatial_mapping=mapping,
                        motion=MotionConfig(strategy=name, max_obstacles=32))
                    service = NS(close=Mock())
                    with patch('robot_skill_stack.world.spatial_mapping.runtime.MappingService',
                               return_value=service) as build:
                        coordinator = await create_observation_runtime(world, selected, updater, None)
                    options = build.call_args.kwargs
                    self.assertIs(options['worker_factory'], ProcessWorker)
                    if name == 'reactive':
                        self.assertEqual(options['obstacle_settings'].max_obstacles, 32)
                        self.assertEqual(options['obstacle_settings'].support_plane_z, config.scene.support_plane_z)
                    else:
                        self.assertNotIn('obstacle_settings', options)
                    coordinator.close()
                    service.close.assert_called_once()
        asyncio.run(run())

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
            for name in ('reactive', 'hybrid', 'typo', None, 1):
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
