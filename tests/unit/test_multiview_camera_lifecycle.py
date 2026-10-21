"""Exercise the ordinary camera-resource factory with SDK-free camera doubles."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ADAPTER = Path(__file__).resolve().parents[2]/'robot_skill_stack/integrations/isaac/world/multiview.py'


def load_adapter(camera_type):
    sensor = ModuleType('camera_resource_double')
    sensor.IsaacRgbdCamera = camera_type
    with patch.dict(sys.modules, {'robot_skill_stack.integrations.isaac.sensors.rgbd_camera': sensor}):
        spec = importlib.util.spec_from_file_location('variable_camera_lifecycle_adapter', ADAPTER)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    return adapter


def camera_resources(*, initialize_failure=None, constructor_failure=None, cleanup_failures=()):
    created, attempts, closed = [], [], []
    failures = dict(initialize=initialize_failure, constructor=constructor_failure)
    class Camera:
        def __init__(self, path, **kwargs):
            attempts.append(path)
            if path == failures['constructor']:
                raise ValueError(f'Camera prim not found: {path}')
            self.path, self.options, self.initializations, self.closes = path, kwargs, [], 0
            self.camera = SimpleNamespace(get_current_frame=lambda: {
                'rendering_frame': {'referenceTimeNumerator': 111, 'referenceTimeDenominator': 30},
                'rendering_time': 1.5})
            created.append(self)

        def initialize(self, **kwargs):
            self.initializations.append(kwargs)
            if self.path == failures['initialize']:
                raise RuntimeError(f'initialize failed: {self.path}')

        def close(self):
            self.closes += 1
            closed.append(self.path)
            if self.path in cleanup_failures:
                raise RuntimeError(f'cleanup failed: {self.path}')
    return load_adapter(Camera), created, attempts, closed, failures


class CameraLifecycleTests(unittest.TestCase):
    def test_three_and_five_resources_initialize_and_close_once_in_reverse_order(self):
        for count in (3, 5):
            with self.subTest(count=count):
                adapter, created, attempts, closed, _ = camera_resources()
                paths = tuple(f'/World/Camera{index+1}' for index in range(count))
                rig = adapter.create_cameras(paths, resolution=(320, 240), max_skew=.04)
                self.assertEqual(tuple(rig.cameras), paths)
                self.assertEqual(attempts, list(paths))
                self.assertEqual(rig.max_skew, .04)
                for index, camera in enumerate(created):
                    self.assertEqual(camera.options, dict(name=f'multiview_camera_{index}', resolution=(320, 240)))
                    self.assertEqual(camera.initializations, [dict(semantic_segmentation=False)])
                    self.assertEqual(camera.get_frame_token(), (3.7, 1.5))
                rig.close()
                rig.close()
                self.assertEqual(closed, list(reversed(paths)))
                self.assertEqual([camera.closes for camera in created], [1]*count)

    def test_failure_initializing_third_or_fifth_closes_every_constructed_resource(self):
        for failed_index in (3, 5):
            with self.subTest(failed_index=failed_index):
                path = f'/World/Camera{failed_index}'
                adapter, created, attempts, closed, _ = camera_resources(
                    initialize_failure=path, cleanup_failures=(path, '/World/Camera1'))
                paths = tuple(f'/World/Camera{index+1}' for index in range(5))
                with self.assertRaisesRegex(RuntimeError, f'initialize failed: {path}') as caught:
                    adapter.create_cameras(paths)
                self.assertEqual(attempts, list(paths[:failed_index]))
                self.assertEqual(closed, list(reversed(paths[:failed_index])))
                self.assertEqual([camera.closes for camera in created], [1]*failed_index)
                notes = getattr(caught.exception, '__notes__', [])
                self.assertEqual(len(notes), 2)
                self.assertTrue(all('cleanup failed' in note for note in notes))

    def test_missing_third_prim_closes_previous_resources_and_does_not_drop_that_camera(self):
        adapter, created, attempts, closed, _ = camera_resources(constructor_failure='/World/Camera3')
        paths = tuple(f'/World/Camera{index+1}' for index in range(5))
        with self.assertRaisesRegex(ValueError, 'Camera prim not found: /World/Camera3'):
            adapter.create_cameras(paths)
        self.assertEqual(attempts, list(paths[:3]))
        self.assertEqual(closed, list(reversed(paths[:2])))
        self.assertEqual([camera.closes for camera in created], [1, 1])

    def test_reinitialization_after_partial_failure_uses_new_resources(self):
        adapter, created, _, _, failures = camera_resources(initialize_failure='/World/Camera3')
        paths = tuple(f'/World/Camera{index+1}' for index in range(5))
        with self.assertRaisesRegex(RuntimeError, 'initialize failed'):
            adapter.create_cameras(paths)
        failed_resources = list(created)
        failures['initialize'] = None
        rig = adapter.create_cameras(paths)
        self.assertEqual(len(rig.cameras), 5)
        self.assertFalse(set(map(id, failed_resources)) & set(map(id, rig.cameras.values())))
        rig.close()
        rig.close()
        self.assertEqual([camera.closes for camera in created], [1]*8)


if __name__ == '__main__':
    unittest.main()
