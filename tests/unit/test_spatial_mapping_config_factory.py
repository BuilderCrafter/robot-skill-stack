"""Opt-in configuration, borrowed depth snapshots and explicit resource ownership."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.integrations.isaac.spatial_mapping.factory import MappingSession, create_spatial_mapping
from robot_skill_stack.integrations.isaac.spatial_mapping.frames import CameraSnapshots, SharedFrames
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap
from tests.unit.test_spatial_mapping_rgbd import ray


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def sdk_token_adapter():
    """Load the real ReferenceTime parser with its SDK resource base stubbed."""
    sensor = ModuleType('mapping_camera_sensor_double')
    sensor.IsaacRgbdCamera = object
    with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.sensors.rgbd_camera': sensor}):
        spec = importlib.util.spec_from_file_location('mapping_token_adapter',
            ROOT/'robot_skill_stack/integrations/isaac/world/multiview.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.world.multiview': adapter}):
        yield adapter


def borrowed_camera():
    camera = Mock()
    camera.camera = SimpleNamespace(get_current_frame=Mock(return_value={
        'rendering_frame': {'referenceTimeNumerator': 111, 'referenceTimeDenominator': 30},
        'rendering_time': 1.5}))
    camera.get_depth.return_value = np.ones((2, 2))
    camera.get_intrinsics.return_value = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
    camera.get_world_from_camera_transform.return_value = np.eye(4)
    return camera


class SpatialConfigurationTests(unittest.TestCase):
    def load(self, provider, mapping=''):
        text = f'''[robot]
backend = "franka"
prim_path = "/Franka"
[world]
provider = "{provider}"
[scene]
support_plane_z = 0.0
workspace_min = [0.2, -0.4, 0.0]
workspace_max = [0.7, 0.4, 0.22]
[ground_truth]
objects_root = "/World"
[perception]
camera_prim_path = "/World/Camera"
[multi_view]
camera_prim_paths = ["/World/A", "/World/B"]
'''+mapping
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scene.toml'
            path.write_text(text)
            return load_scene_config(path)

    def test_mapping_defaults_disabled_without_changing_any_semantic_provider(self):
        for provider in ('perception', 'ground_truth', 'multi_view'):
            with self.subTest(provider=provider):
                config = self.load(provider)
                self.assertEqual(config.world.provider, provider)
                self.assertFalse(config.spatial_mapping.enabled)
                self.assertIsNone(create_spatial_mapping(config.spatial_mapping))

    def test_rgbd_mapping_configuration_is_independent_of_semantic_provider_bounds_and_camera_count(self):
        mapping = '''[spatial_mapping]
enabled = true
source = "rgbd"
bounds_min = [-0.1, -0.2, -0.3]
bounds_max = [0.1, 0.2, 0.3]
voxel_size = 0.05
camera_prim_paths = ["/World/OnlyMappingCamera"]
max_rays = 17
max_voxel_visits = 1234
update_hz = 3.0
'''
        for provider in ('perception', 'ground_truth', 'multi_view'):
            with self.subTest(provider=provider):
                config = self.load(provider, mapping)
                settings = config.spatial_mapping
                self.assertEqual(config.world.provider, provider)
                self.assertEqual(settings.camera_prim_paths, ('/World/OnlyMappingCamera',))
                self.assertEqual(settings.bounds_min, (-.1, -.2, -.3))
                self.assertEqual(settings.grid_spec().shape, (4, 8, 12))
                self.assertEqual(settings.max_voxel_visits, 1234)
                self.assertEqual(settings.update_hz, 3.)
                self.assertFalse(np.array_equal(config.scene.workspace_min, settings.bounds_min))
        self.assertEqual(SpatialMappingConfig().max_rays, 1024)

    def test_invalid_mapping_settings_and_excessive_allocation_are_rejected(self):
        for kwargs in (dict(enabled='true'), dict(source='semantic'), dict(voxel_size=0),
                dict(bounds_max=(0., 0., 0.)), dict(max_voxels=1), dict(stride=True),
                dict(max_rays=0), dict(max_voxel_visits=0), dict(max_candidate_voxels=0),
                dict(max_depth=np.nan), dict(max_frame_age=-1), dict(stale_after=0),
                dict(update_hz=0), dict(update_hz=True), dict(evidence_ttl=np.inf),
                dict(camera_prim_paths=('/World/A', '/World/A')),
                dict(camera_prim_paths=('relative',)), dict(camera_prim_paths=('/World/Bad Name',)),
                dict(excluded_prim_paths=('/Franka', '/Franka'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpatialMappingConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.load('ground_truth', '[spatial_mapping]\nunknown_setting = true\n')


class MappingFactoryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.clock = lambda: self.now
        self.settings = SpatialMappingConfig(enabled=True, source='rgbd', bounds_min=(0., 0., 0.),
            bounds_max=(1., 1., 1.), voxel_size=.1, stride=1, camera_prim_paths=('/World/A',))

    def test_disabled_factory_allocates_nothing_imports_no_sdk_and_takes_no_ownership(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('isaacsim', 'omni', 'pxr'):
                raise AssertionError(f'Unexpected SDK import: {name}')
            return original(name, *args, **kwargs)
        frames = Mock()
        with patch('builtins.__import__', side_effect=guarded), \
                patch('robot_skill_stack.integrations.isaac.spatial_mapping.factory.OccupancyMap') as allocate:
            self.assertIsNone(create_spatial_mapping(SpatialMappingConfig(), frame_source=frames,
                                                    own_frame_source=True))
        allocate.assert_not_called()
        frames.capture.assert_not_called()
        frames.close.assert_not_called()

    def test_rgbd_factory_forwards_processing_and_map_lifetime_settings(self):
        settings = replace(self.settings, max_rays=19, max_voxel_visits=567, max_depth=2.,
            max_skew=.02, max_frame_age=.5, stale_after=.6, evidence_ttl=2., update_hz=4.)
        session = create_spatial_mapping(settings, clock=self.clock)
        self.assertEqual(session.source.max_rays, 19)
        self.assertEqual(session.source.max_voxel_visits, 567)
        self.assertEqual(session.source.max_depth, 2.)
        self.assertEqual(session.source.max_skew, .02)
        self.assertEqual(session.source.max_frame_age, .5)
        self.assertEqual(session.source.expected_camera_ids, ('/World/A',))
        self.assertEqual(session.map.stale_after, .6)
        self.assertEqual(session.map.evidence_ttl, 2.)
        self.assertEqual(session.update_hz, 4.)
        self.assertIs(session.map.clock, self.clock)
        session.close()

    def test_default_clock_is_sampled_after_capture_and_mapper_keeps_its_completion_clock(self):
        frame_source = Mock()
        def capture():
            self.now += .1
            return (ray(camera='/World/A', acquired=self.now),)
        frame_source.capture.side_effect = capture
        session = create_spatial_mapping(self.settings, frame_source=frame_source, clock=self.clock)
        with patch.object(session.source, 'update', wraps=session.source.update) as update:
            self.assertTrue(session.update())
        self.assertIsNone(update.call_args.kwargs['now'])
        self.assertEqual(session.map.timestamp, 100.1)
        self.assertTrue(session.map.health['fresh'])
        session.close()
        frame_source.close.assert_not_called()

    def test_explicit_now_remains_deterministic_and_does_not_acquire_from_a_borrowed_source(self):
        frames = (ray(camera='/World/A', acquired=100.),)
        frame_source = Mock()
        session = create_spatial_mapping(self.settings, frame_source=frame_source, clock=self.clock)
        with patch.object(session.source, 'update', wraps=session.source.update) as update:
            self.assertTrue(session.update(frames, now=100.))
        self.assertEqual(update.call_args.kwargs['now'], 100.)
        frame_source.capture.assert_not_called()
        session.close()
        frame_source.close.assert_not_called()

    def test_rate_limit_skips_capture_without_advancing_map_time(self):
        frame_source = Mock(return_value=None)
        frame_source.capture.return_value = (ray(camera='/World/A', acquired=100.),)
        session = create_spatial_mapping(replace(self.settings, update_hz=2.),
            frame_source=frame_source, clock=self.clock)
        self.assertTrue(session.update())
        original = session.map.timestamp
        self.now = 100.2
        self.assertFalse(session.update())
        frame_source.capture.assert_called_once()
        self.assertEqual(session.map.timestamp, original)
        self.now = 100.5
        frame_source.capture.return_value = (ray(camera='/World/A', number=2, acquired=self.now),)
        self.assertTrue(session.update())
        self.assertEqual(frame_source.capture.call_count, 2)
        session.close()

    def test_no_frame_and_acquisition_failure_are_explicit_without_new_map_time(self):
        frame_source = Mock()
        frame_source.capture.return_value = None
        session = create_spatial_mapping(self.settings, frame_source=frame_source, clock=self.clock)
        self.assertFalse(session.update())
        self.assertEqual(session.map.health['status'], 'no_frame')
        self.assertIsNone(session.map.timestamp)
        frame_source.capture.side_effect = RuntimeError('sensor failed')
        with self.assertRaisesRegex(RuntimeError, 'sensor failed'):
            session.update()
        self.assertEqual(session.map.health['status'], 'error')
        self.assertFalse(session.map.health['fresh'])
        session.close()

    def test_ground_truth_session_forwards_stage_and_exclusion_without_camera_capture(self):
        stage, frame_source = object(), Mock()
        settings = replace(self.settings, source='ground_truth', excluded_prim_paths=('/Robot',),
                           max_candidate_voxels=123)
        session = create_spatial_mapping(settings, stage=stage, frame_source=frame_source, clock=self.clock)
        self.assertEqual(session.source.excluded_paths, ('/Robot',))
        self.assertEqual(session.source.max_candidate_voxels, 123)
        with patch.object(session.source, 'update', return_value=True) as update:
            self.assertTrue(session.update(now=100.))
        update.assert_called_once_with(stage=stage, now=100.)
        frame_source.capture.assert_not_called()
        session.close()
        self.assertTrue(session.source._closed)
        frame_source.close.assert_not_called()

    def test_owned_partial_construction_cleanup_preserves_primary_error_and_borrowed_resources(self):
        for owned in (True, False):
            with self.subTest(owned=owned):
                frames = Mock()
                frames.close.side_effect = RuntimeError('cleanup failed')
                with patch('robot_skill_stack.world.spatial_mapping.rgbd.RgbdMapper',
                           side_effect=ValueError('initialization failed')):
                    with self.assertRaisesRegex(ValueError, 'initialization failed') as caught:
                        create_spatial_mapping(self.settings, frame_source=frames,
                            own_frame_source=owned, clock=self.clock)
                self.assertEqual(frames.close.call_count, int(owned))
                self.assertEqual(len(getattr(caught.exception, '__notes__', [])), int(owned))

    def test_close_cleans_source_and_owned_frames_once_despite_multiple_errors(self):
        occupancy = OccupancyMap(GridSpec((0, 0, 0), (1, 1, 1), .1), clock=self.clock)
        source, frames = Mock(), Mock()
        source.close.side_effect = RuntimeError('source cleanup failed')
        frames.close.side_effect = RuntimeError('frame cleanup failed')
        session = MappingSession(occupancy, source, frame_source=frames,
            own_frame_source=True, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'source cleanup failed') as caught:
            session.close()
        session.close()
        source.close.assert_called_once()
        frames.close.assert_called_once()
        self.assertIn('frame cleanup failed', caught.exception.__notes__[0])
        self.assertEqual(occupancy.health['reason'], 'closed')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            session.update()


class SharedFrameTests(unittest.TestCase):
    def test_shared_tap_reuses_owner_snapshots_reports_failure_and_recovers_without_recapture(self):
        first, second = (ray(),), (ray(number=2),)
        rig = SimpleNamespace(capture=Mock(side_effect=[first, None, RuntimeError('camera failed'), second]), close=Mock())
        original = rig.capture
        tap = SharedFrames(rig)
        self.assertIsNone(tap.capture())
        self.assertIs(rig.capture(), first)
        self.assertIs(tap.capture(), first)
        self.assertEqual(original.call_count, 1)
        self.assertIsNone(rig.capture())
        self.assertIs(tap.capture(), first)
        with self.assertRaisesRegex(RuntimeError, 'camera failed'):
            rig.capture()
        with self.assertRaisesRegex(RuntimeError, 'Shared acquisition failed'):
            tap.capture()
        self.assertIs(rig.capture(), second)
        self.assertIs(tap.capture(), second)
        tap.close()
        tap.close()
        self.assertIs(rig.capture, original)
        rig.close.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            tap.capture()

    def test_duplicate_tap_is_rejected_and_close_preserves_a_later_owner_hook(self):
        rig = SimpleNamespace(capture=Mock(return_value=None))
        tap = SharedFrames(rig)
        with self.assertRaisesRegex(RuntimeError, 'already'):
            SharedFrames(rig)
        newer = Mock()
        rig.capture = newer
        tap.close()
        self.assertIs(rig.capture, newer)
        self.assertFalse(hasattr(rig, '_spatial_mapping_frame_tap'))

    def test_context_manager_restores_acquisition_even_on_error(self):
        rig = SimpleNamespace(capture=Mock(return_value=None))
        original = rig.capture
        with self.assertRaisesRegex(ValueError, 'consumer failed'):
            with SharedFrames(rig):
                raise ValueError('consumer failed')
        self.assertIs(rig.capture, original)

    def test_borrowed_camera_snapshots_use_real_reference_time_parser_and_copy_calibration(self):
        cameras = {f'/World/C{index}': borrowed_camera() for index in range(3)}
        with sdk_token_adapter():
            reader = CameraSnapshots(cameras, clock=lambda: 10.)
            frames = reader.capture()
            self.assertEqual([frame.camera_id for frame in frames], list(cameras))
            self.assertEqual([frame.token for frame in frames], [(3.7, 1.5)]*3)
            self.assertEqual([frame.acquired_at for frame in frames], [10.]*3)
            for camera, frame in zip(cameras.values(), frames):
                camera.get_depth.return_value[:] = 99
                camera.get_intrinsics.return_value[0, 0] = 99
                camera.get_world_from_camera_transform.return_value[0, 3] = 99
                self.assertEqual(frame.depth[0, 0], 1.)
                self.assertEqual(frame.intrinsics[0, 0], 2.)
                self.assertEqual(frame.world_from_camera[0, 3], 0.)
                camera.initialize.assert_not_called()
            reader.close()
            reader.close()
            for camera in cameras.values():
                camera.close.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                reader.capture()

    def test_borrowed_camera_missing_and_changing_buffers_never_return_partial_snapshot(self):
        with sdk_token_adapter():
            camera = borrowed_camera()
            camera.get_depth.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                CameraSnapshots({'A': camera}).capture()
            camera = borrowed_camera()
            camera.camera.get_current_frame.side_effect = [dict(rendering_frame=1, rendering_time=.1),
                                                          dict(rendering_frame=2, rendering_time=.2)]
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                CameraSnapshots({'A': camera}).capture()
            with self.assertRaises(ValueError):
                CameraSnapshots({})
            with self.assertRaises(ValueError):
                CameraSnapshots({'A': camera, 'B': camera})


if __name__ == '__main__':
    unittest.main()
