"""Comparable camera selections without importing or starting Isaac Sim."""
import hashlib
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.camera_workload_config import (
    MAX_RAYS, MODES, ORIGINAL_CAMERAS, PROFILE, SCENE, THIRD,
    THIRD_RESOLUTIONS, scenario_camera_paths, write_profile)


class CameraWorkloadConfigTests(unittest.TestCase):
    def test_scenarios_keep_semantic_and_mapping_inputs_independent(self):
        two, three = ORIGINAL_CAMERAS, (*ORIGINAL_CAMERAS, THIRD)
        expected = {'A': (two, two), 'B': (two, two),
                    'C': (two, three), 'D': (three, three)}
        for mode, paths in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(scenario_camera_paths(mode), paths)

    def test_all_profiles_preserve_common_workload_and_shipped_assets(self):
        protected = (SCENE, PROFILE)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        original = tomllib.loads(PROFILE.read_text())
        reference_mapping = None
        with tempfile.TemporaryDirectory() as directory:
            for mode in MODES:
                for resolution in THIRD_RESOLUTIONS:
                    with self.subTest(mode=mode, resolution=resolution):
                        folder = Path(directory)/mode/f'{resolution[0]}x{resolution[1]}'
                        profile = write_profile(folder, mode, resolution)
                        self.assertEqual(profile.parent, folder)
                        data = tomllib.loads(profile.read_text())
                        self.assertEqual(data['motion']['strategy'], 'direct')
                        self.assertEqual(data['world']['update_hz'], 5.)
                        self.assertEqual(data['multi_view']['resolution'], [640, 480])
                        self.assertEqual(data['scene'], original['scene'])
                        self.assertEqual(data['robot'], original['robot'])
                        for key in ('voxel_size', 'stride', 'max_skew'):
                            self.assertEqual(data['multi_view'][key], original['multi_view'][key])
                        semantic, mapping = scenario_camera_paths(mode)
                        self.assertEqual(tuple(data['multi_view']['camera_prim_paths']), semantic)
                        settings = dict(data['spatial_mapping'])
                        self.assertEqual(tuple(settings.pop('camera_prim_paths')), mapping)
                        self.assertEqual(settings['update_hz'], 2.)
                        self.assertEqual(settings['max_rays'], MAX_RAYS)
                        self.assertEqual(settings['max_rays'], 1024)
                        self.assertEqual(settings['stride'], 8)
                        self.assertTrue(settings['automatic_robot_filter'])
                        self.assertEqual(settings['max_frame_age'], 1.)
                        self.assertEqual(settings['stale_after'], 1.)
                        if reference_mapping is None:
                            reference_mapping = settings
                        self.assertEqual(settings, reference_mapping,
                                         'Only camera membership may change mapping workload settings')
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        self.assertEqual(before, after)

    def test_unknown_modes_are_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'invalid'
            for mode in ('probe', 'E', 'a', '', None):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    write_profile(output, mode)
                self.assertFalse(output.exists())
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    scenario_camera_paths(mode)

    def test_unsupported_or_noninteger_resolutions_are_rejected(self):
        invalid = ((640, 240), (320, 480), (0, 0), (320., 240),
                   ('320', 240), (True, 240), (320, 240, 3), (320,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'invalid'
            for resolution in invalid:
                with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                    write_profile(output, 'C', resolution)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
