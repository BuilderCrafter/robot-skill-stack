"""SDK-free safeguards for matched native V4 measurements and physical scoring."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle
from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh
from scripts.benchmark_global_motion_v4 import (
    GOAL, SCENARIOS, START, contact_summary, fixtures, joint_path_deviation,
    reduce_clearances, reference_match, target_position, write_profile)
from tests.support.reactive_geometry import CollisionMeshReference


def cube_mesh(center):
    vertices = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.1, .1)])
    vertices += np.asarray(center)
    triangles = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return TriangleMesh(vertices, triangles)


class GlobalMotionHarnessTests(unittest.TestCase):
    def test_global_profile_has_no_camera_or_required_mapping_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(SimpleNamespace(strategy='global', seed=7,
                max_iterations=1000, artifacts_dir=directory))
            config = load_scene_config(path)
        self.assertEqual(config.world.provider, 'ground_truth')
        self.assertFalse(config.spatial_mapping.enabled)
        self.assertIsNone(config.perception.camera_prim_path)
        self.assertEqual(config.multi_view.camera_prim_paths, ())
        self.assertEqual((config.motion.rrt_seed, config.motion.rrt_max_iterations), (7, 1000))

    def test_reactive_profile_retains_original_exclusion_and_two_hz_product(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(SimpleNamespace(strategy='reactive', seed=7,
                max_iterations=1000, artifacts_dir=directory))
            config = load_scene_config(path)
        self.assertTrue(config.spatial_mapping.enabled)
        self.assertEqual(config.spatial_mapping.source, 'ground_truth')
        self.assertEqual(config.spatial_mapping.update_hz, 2.)
        self.assertEqual(config.spatial_mapping.excluded_prim_paths, ('/Franka',))

    def test_explicit_background_mapping_does_not_require_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_profile(SimpleNamespace(strategy='global', seed=7,
                max_iterations=1000, artifacts_dir=directory, enable_mapping=True))
            config = load_scene_config(path)
        self.assertTrue(config.spatial_mapping.enabled)
        self.assertIsNone(config.perception.camera_prim_path)

    def test_impossible_goals_do_not_silently_change_regular_fixture_targets(self):
        for name in SCENARIOS:
            self.assertFalse(any(box.moving for box in fixtures(name)))
        np.testing.assert_array_equal(target_position('static'), GOAL)
        self.assertGreater(np.linalg.norm(target_position('unreachable')), 2.)
        blocked = fixtures('blocked')
        np.testing.assert_array_equal(blocked[0].center, GOAL)
        self.assertEqual(fixtures('unreachable'), ())
        with self.assertRaises(ValueError):
            fixtures('moving')

    def test_matched_guard_checks_actual_joints_physics_and_source_before_movement(self):
        configuration = dict(scenario='static', start=START.tolist(), goal=GOAL.tolist(),
            orientation=[0., 0., 1., 0.], objects=[dict(name='Post')],
            position_tolerance_m=.025, orientation_tolerance_rad=.10, max_motion_steps=1000)
        world, hashes = [dict(path='/World/Post', dimensions=[.1, .1, .1])], {'source.py': 'frozen'}
        reference = dict(status='PASS', hashes=hashes, metrics=dict(configuration=configuration,
            initial_joints=[0.]*9, physical_world=world))
        kwargs = dict(hashes=hashes, physical_world=world)
        self.assertTrue(reference_match(reference, configuration, [0.]*9, **kwargs)['matched'])
        self.assertFalse(reference_match(reference, configuration, [.02]+[0.]*8, **kwargs)['matched'])
        self.assertFalse(reference_match(reference, {**configuration, 'goal': [1., 2., 3.]}, [0.]*9, **kwargs)['matched'])
        self.assertFalse(reference_match(reference, configuration, [0.]*9,
            hashes={'source.py': 'changed'}, physical_world=world)['matched'])
        self.assertFalse(reference_match(reference, configuration, [0.]*9,
            hashes=hashes, physical_world=[])['matched'])

    def test_joint_path_distance_keeps_seven_joint_route_instead_of_cartesian_projection(self):
        route = np.zeros((2, 7)); route[1, 0] = 1.
        actual = np.zeros((3, 7)); actual[:, 0] = [0., .5, 1.]; actual[:, 6] = [.1, .2, .3]
        measured = joint_path_deviation(actual, route)
        self.assertAlmostEqual(measured['worst'], .3)
        self.assertAlmostEqual(measured['mean'], .2)
        self.assertEqual(measured['count'], 3)
        self.assertEqual(joint_path_deviation(actual[:, :3], route)['count'], 0)

    def test_positive_separation_reports_are_retained_and_physics_steps_deduplicated(self):
        pair = dict(sensor='/Franka/panda_hand', count=2, minimum_separation_m=.004,
            impulse_magnitude=0.)
        base = dict(sensor='/Franka/panda_link0', count=1, minimum_separation_m=0.,
            impulse_magnitude=0.)
        sample = dict(physics_step=7, simulation_time=.1, contacts=[pair, base])
        measured = contact_summary([sample, dict(sample)])
        self.assertEqual(measured['reported_points'], 3)
        self.assertEqual(measured['penetrating_pairs'], 0)
        self.assertEqual(measured['maximum_normal_pair_impulse_ns'], 0.)
        self.assertEqual(contact_summary([sample], exclude_mounted_base=True)['reported_points'], 2)

    def test_clearance_keeps_support_and_both_fingers_separate_from_anchored_base(self):
        paths = ('/Franka/panda_link0', '/Franka/panda_hand', '/Franka/panda_leftfinger', '/Franka/panda_rightfinger')
        meshes = [CollisionMeshReference(path+'/mesh', path, 'gripper' if index else 'upper_arm_or_base',
            cube_mesh([0., 0., .1 if index == 0 else .5]), np.eye(4), 'physx_cooked:synthetic_unit')
            for index, path in enumerate(paths)]
        plane = StaticObstacle('/Ground', 'cuboid', (0., 0., -25.), dimensions=(50., 50., 50.),
            source_kind='Plane', approximation='plane_halfspace_clipped_to_certified_domain')
        table = StaticObstacle('/Table', 'cuboid', (1., 0., .5), dimensions=(.4, .4, .4), source_kind='Cube')
        measured = reduce_clearances(meshes, paths, [np.repeat(np.eye(4)[None], 4, axis=0)], (plane, table))
        self.assertEqual(measured['minimum_m'], 0.)
        self.assertAlmostEqual(measured['movable_arm_minimum_m'], .4)
        self.assertEqual(set(measured['by_link_m']), set(paths))
        self.assertEqual(set(measured['by_obstacle_m']), {'/Ground', '/Table'})
        self.assertGreater(measured['by_obstacle_m']['/Table'], .6)


if __name__ == '__main__':
    unittest.main()
