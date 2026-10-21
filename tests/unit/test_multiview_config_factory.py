import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from robot_skill_stack.integrations.isaac.config import load_scene_config
from tests.unit.test_runtime_bootstrap import bootstrap_fixture

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'config/scenes/multiview_playground.toml'
CAMERA_SETTING = 'camera_prim_paths = ["/World/PerceptionCamera", "/World/PerceptionCameraSide"]'


class ConfigFactoryTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.toml'
            path.write_text(text)
            return load_scene_config(path)

    def camera_config(self, paths):
        return self.load(PROFILE.read_text().replace(
            CAMERA_SETTING, 'camera_prim_paths = '+json.dumps(paths)))

    def test_independent_selector_needs_no_v1_or_ground_truth_references(self):
        cfg = load_scene_config(PROFILE)
        self.assertEqual(cfg.world.provider, 'multi_view')
        self.assertEqual(len(cfg.multi_view.camera_prim_paths), 2)
        self.assertIsNone(cfg.perception.camera_prim_path)
        self.assertEqual(cfg.ground_truth.objects, {})

    def test_camera_counts_two_three_and_five_preserve_configured_order(self):
        for count in (2, 3, 5):
            paths = [f'/World/Camera_{i}' for i in range(count)]
            with self.subTest(count=count):
                self.assertEqual(self.camera_config(paths).multi_view.camera_prim_paths,
                                 tuple(paths))

    def test_selected_multiview_rejects_zero_and_one_camera(self):
        for paths in ([], ['/World/Camera_A']):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, 'at least two'):
                self.camera_config(paths)

    def test_duplicate_and_invalid_prim_paths_are_rejected(self):
        for paths in (
            ['/World/A', '/World/B', '/World/A'],
            ['/World/A', '/World/B', 'relative'],
            ['/World/A', '/'],
            ['/World/A', '/World//Camera'],
            ['/World/A', '/World/Camera/'],
            ['/World/A', '/World/Camera.focalLength'],
            ['/World/A', '/World/Camera{view=left}'],
            ['/World/A', '/World/Bad Camera'],
            ['/World/A', 12],
        ):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.camera_config(paths)

    def test_v1_and_ground_truth_keep_optional_empty_multiview_configuration(self):
        text = (ROOT/'config/scenes/playground.toml').read_text()
        for provider in ('perception', 'ground_truth'):
            selected = text.replace('provider = "perception"', f'provider = "{provider}"')
            for extra in ('', '\n[multi_view]\ncamera_prim_paths = []\n'):
                with self.subTest(provider=provider, explicit_empty=bool(extra)):
                    cfg = self.load(selected+extra)
                    self.assertEqual(cfg.world.provider, provider)
                    self.assertEqual(cfg.multi_view.camera_prim_paths, ())

    def test_missing_camera_prims_fail_existing_bootstrap_validation_before_construction(self):
        async def run():
            with bootstrap_fixture() as (bootstrap, env):
                for count in (3, 5):
                    paths = [f'/World/Camera_{i}' for i in range(count)]
                    cfg = self.camera_config(paths)
                    for missing in paths:
                        stage = NS(GetPrimAtPath=Mock(side_effect=lambda path:
                            NS(IsValid=lambda: path != missing)))
                        with self.subTest(count=count, missing=missing), \
                             patch.object(bootstrap, 'load_scene_config', return_value=cfg), \
                             patch.object(bootstrap.omni.usd, 'get_context',
                                          return_value=NS(get_stage=lambda: stage)):
                            with self.assertRaisesRegex(RuntimeError, missing):
                                await bootstrap.build_runtime('missing.toml')
                            self.assertEqual({c.args[0] for c in stage.GetPrimAtPath.call_args_list},
                                             {'/Franka', *paths})
                self.assertIsNone(env.world)
                self.assertFalse(env.providers)
        asyncio.run(run())

    def test_factory_forwards_every_camera_and_processing_setting(self):
        for count in (2, 3, 5):
            paths = [f'/World/Camera_{i}' for i in range(count)]
            cfg = self.camera_config(paths)
            cfg = replace(cfg, multi_view=replace(cfg.multi_view, resolution=(320, 240),
                          max_skew=.01, voxel_size=.003, stride=3))
            with self.subTest(count=count), bootstrap_fixture():
                factory = sys.modules['robot_skill_stack.integrations.isaac.world.factory']
                self.assertEqual(factory.required_source_prims(cfg), paths)
                rig = NS(close=Mock(), capture=Mock(), status={})
                module = ModuleType('rig_double')
                module.create_cameras = Mock(return_value=rig)
                with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.world.multiview': module}):
                    provider = asyncio.run(factory.create_source(cfg))
                    module.create_cameras.assert_called_once_with(tuple(paths), (320, 240), max_skew=.01)
                    self.assertEqual((provider.voxel_size, provider.stride), (.003, 3))
                    provider.close()
                rig.close.assert_called_once()

    def test_invalid_cameras_calibration_and_processing_settings_are_rejected(self):
        text = PROFILE.read_text()
        for before, after in (
            ('["/World/PerceptionCamera", "/World/PerceptionCameraSide"]', '[]'),
            ('"/World/PerceptionCameraSide"', '"/World/PerceptionCamera"'),
            ('"/World/PerceptionCameraSide"', '"relative"'),
            ('resolution = [640, 480]', 'resolution = [0, 480]'),
            ('max_skew = 0.025', 'max_skew = nan'),
            ('voxel_size = 0.002', 'voxel_size = -0.1'),
            ('stride = 2', 'stride = true'),
        ):
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.load(text.replace(before, after))

    def test_partial_initialization_closes_every_constructed_camera_even_after_cleanup_error(self):
        created = []
        class FakeCamera:
            def __init__(self, path, **kw):
                self.path, self.closes = path, 0
                self.camera = NS(get_current_frame=lambda: {
                    'rendering_frame': {'referenceTimeNumerator': 111, 'referenceTimeDenominator': 30},
                    'rendering_time': 1.5})
                created.append(self)
            def initialize(self, **kw):
                if self.path == '/B': raise RuntimeError('initialization failed')
            def close(self):
                self.closes += 1
                if self.path == '/B': raise RuntimeError('cleanup failed')
        module = ModuleType('camera_double')
        module.IsaacRgbdCamera = FakeCamera
        with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.sensors.rgbd_camera': module}):
            spec = importlib.util.spec_from_file_location('multiview_camera_under_test',
                ROOT/'robot_skill_stack/integrations/isaac/world/multiview.py')
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
        with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
            adapter.create_cameras(('/A', '/B'))
        self.assertEqual([c.closes for c in created], [1, 1])
        self.assertEqual(created[0].get_frame_token(), (3.7, 1.5))

    def test_source_construction_and_failed_warmup_cleanup(self):
        cfg = load_scene_config(PROFILE)
        with bootstrap_fixture() as (bootstrap, env):
            factory = sys.modules['robot_skill_stack.integrations.isaac.world.factory']
            self.assertEqual(factory.required_source_prims(cfg), list(cfg.multi_view.camera_prim_paths))
            self.assertEqual(bootstrap._update_hz(cfg), 5.)
            rig = NS(close=Mock(), capture=Mock(), status={})
            module = ModuleType('rig_double')
            module.create_cameras = Mock(return_value=rig)
            with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.world.multiview': module}):
                provider = asyncio.run(factory.create_source(cfg))
                provider.close()
                provider.close()
                rig.close.assert_called_once()
                rig.close.reset_mock()
                app = NS(next_update_async=AsyncMock(side_effect=RuntimeError('warmup failed')))
                with patch.object(factory.omni.kit.app, 'get_app', return_value=app):
                    with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
                        asyncio.run(factory.create_source(cfg))
                rig.close.assert_called_once()

    def test_runtime_package_has_no_v1_sdk_or_oracle_imports(self):
        import ast
        for path in (ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(n.startswith(('robot_skill_stack.world.perception.',
                    'tests', 'isaacsim', 'omni', 'pxr')) for n in names), path)


if __name__ == '__main__':
    unittest.main()
