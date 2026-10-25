"""Reproducible source-only profiles and unchanged historical fixtures."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from robot_skill_stack.integrations.isaac.config import load_scene_config
from scripts import global_rgbd_v5_protocol as protocol
from scripts.diagnose_reactive_motion import FIXTURE_ROOT, fixtures as old_fixtures


class GlobalRgbdV5ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_six_historical_scenes_keep_exact_fixture_geometry_and_paths(self):
        for scene in protocol.SCENES[:-1]:
            with self.subTest(scene=scene):
                expected = [dict(path=f'{FIXTURE_ROOT}/{box.name}',center=box.center,
                    dimensions=box.dimensions) for box in old_fixtures(scene)]
                self.assertEqual(protocol.fixtures(scene),expected)

    def test_v43_fixture_and_target_are_predeclared_without_repositioning(self):
        self.assertEqual(protocol.fixtures('v43'),[dict(
            path='/World/ReactiveEvaluation/GlobalRouteDetourV43',
            center=(.55,-.225,.30000000000000004),dimensions=(.06,.06,.12))])
        self.assertEqual(protocol.PROTOCOL['v43_target'],protocol.START)
        self.assertEqual(protocol.PROTOCOL['default_target'],protocol.GOAL)
        self.assertEqual(protocol.PROTOCOL['orientation_wxyz'],(0.,0.,1.,0.))
        self.assertIn('not a cylinder',protocol.PROTOCOL['sparse_fixture_qualification'])

    def test_fixture_callers_cannot_mutate_the_next_trial(self):
        first = protocol.fixtures('partial')
        expected = copy.deepcopy(first)
        first[0]['center']=(100.,100.,100.)
        first.append(dict(path='/Other',center=(0.,0.,0.),dimensions=(1.,1.,1.)))
        self.assertEqual(protocol.fixtures('partial'),expected)

    def test_matched_profiles_differ_only_in_explicit_planning_source(self):
        values = []
        for source in protocol.SOURCES:
            path = protocol.write_profile(self.root/source/'runtime_profile.toml',source)
            values.append(tomllib.loads(path.read_text()))
            config = load_scene_config(path)
            self.assertEqual(config.motion.global_obstacle_source,source)
            self.assertEqual(config.motion.strategy,'global')
            self.assertEqual(config.motion.global_collision_model,'enhanced')
            self.assertEqual(config.motion.max_obstacles,64)
            self.assertEqual(config.spatial_mapping.source,'rgbd')
            self.assertTrue(config.spatial_mapping.enabled)
            self.assertTrue(config.spatial_mapping.automatic_robot_filter)
        values[0]['motion']['global_obstacle_source']=values[1]['motion']['global_obstacle_source']
        self.assertEqual(*values)

    def test_profile_preserves_authored_camera_configuration_and_budget(self):
        before_scene,before_profile = protocol.SCENE.read_bytes(),protocol.PROFILE.read_bytes()
        baseline = tomllib.loads(before_profile.decode())
        path = protocol.write_profile(self.root/'runtime_profile.toml','frozen_rgbd')
        result = tomllib.loads(path.read_text())
        self.assertEqual(result['multi_view'],baseline['multi_view'])
        self.assertEqual(result['robot'],baseline['robot'])
        self.assertEqual(result['world'],dict(provider='multi_view',update_hz=5.))
        self.assertEqual(result['scene']['workspace_min'],list(protocol.WORKSPACE_MIN))
        self.assertEqual(result['scene']['workspace_max'],list(protocol.WORKSPACE_MAX))
        mapping = result['spatial_mapping']
        expected = dict(enabled=True,source='rgbd',bounds_min=list(protocol.MAPPING_MIN),
            bounds_max=list(protocol.MAPPING_MAX),voxel_size=.01,camera_prim_paths=list(protocol.CAMERAS),
            stride=8,max_rays=1024,automatic_robot_filter=True,update_hz=2.,max_frame_age=1.,
            stale_after=1.,evidence_ttl=5.)
        self.assertEqual(mapping,expected)
        self.assertEqual(protocol.SCENE.read_bytes(),before_scene)
        self.assertEqual(protocol.PROFILE.read_bytes(),before_profile)

    def test_diagnostic_budget_is_bounded_and_explicit(self):
        primary = protocol.write_profile(self.root/'64.toml','frozen_rgbd')
        diagnostic = protocol.write_profile(self.root/'128.toml','frozen_rgbd',budget=128)
        first,second = [tomllib.loads(path.read_text()) for path in (primary,diagnostic)]
        self.assertEqual(second['motion']['max_obstacles'],128)
        first['motion']['max_obstacles']=128
        self.assertEqual(first,second)
        for budget in (True,1,63,65,127,129,64.):
            with self.subTest(budget=budget),self.assertRaises(ValueError):
                protocol.write_profile(self.root/'invalid.toml','frozen_rgbd',budget)
        self.assertFalse((self.root/'invalid.toml').exists())

    def test_unknown_inputs_and_shipped_file_overwrites_fail_before_write(self):
        for source in ('ground_truth','direct','hybrid','rgbd'):
            with self.subTest(source=source),self.assertRaises(ValueError):
                protocol.write_profile(self.root/'invalid.toml',source)
        with self.assertRaises(ValueError):protocol.fixtures('cylinder')
        for path in (protocol.SCENE,protocol.PROFILE):
            with self.subTest(path=path),self.assertRaises(ValueError):
                protocol.write_profile(path,'static')
        self.assertFalse((self.root/'invalid.toml').exists())

    def test_authored_camera_drift_is_not_silently_repaired(self):
        changed = self.root/'changed.toml'
        changed.write_text(protocol.PROFILE.read_text().replace('640, 480','320, 240'))
        with patch.object(protocol,'PROFILE',changed),self.assertRaises(ValueError):
            protocol.write_profile(self.root/'runtime_profile.toml','frozen_rgbd')
        self.assertFalse((self.root/'runtime_profile.toml').exists())


if __name__=='__main__':unittest.main()
