"""Scene mapping configuration validation without legacy synchronous factories."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.spatial_mapping import SpatialMappingConfig


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
        self.assertEqual(SpatialMappingConfig.__module__, 'robot_skill_stack.world.spatial_mapping.config')
        for provider in ('perception', 'ground_truth', 'multi_view'):
            with self.subTest(provider=provider):
                config = self.load(provider)
                self.assertIs(type(config.spatial_mapping), SpatialMappingConfig)
                self.assertEqual(config.world.provider, provider)
                self.assertFalse(config.spatial_mapping.enabled)

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
        for kwargs in (dict(enabled='true'), dict(source='semantic'), dict(max_voxels=1), dict(stride=True),
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



if __name__ == '__main__':
    unittest.main()
