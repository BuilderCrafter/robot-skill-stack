"""Independent operating-contract and fixed-base primitive checks, without Kit."""
from dataclasses import replace
import itertools
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import (
    FINGER_NAMES, MODEL_DIRECTORY, MODEL_NAME, load_enhanced_collision_model,
)
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import (
    StaticObstacle, StaticWorldSnapshot, obstacle_from_primitive,
)
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


class CollisionModelTests(unittest.TestCase):
    def setUp(self):
        self.model = load_enhanced_collision_model()

    def robot(self, positions=(.003, .003)):
        names = (FINGER_NAMES[1], 'panda_joint1', FINGER_NAMES[0])
        q = np.array([positions[1], .5, positions[0]])
        return NS(dof_names=names, get_joint_positions=lambda: q.copy())

    def snapshot(self, *obstacles, domains=()):
        return StaticWorldSnapshot(obstacles, 0., 1., '/Franka', plane_domains=domains)

    def test_project_model_loads_hash_and_finite_contract_without_sdk(self):
        self.assertEqual(self.model.finger_range_m, (.001, .005))
        self.assertEqual(self.model.base_link, 'panda_link0')
        self.assertEqual(self.model.descriptor_path, MODEL_DIRECTORY/(MODEL_NAME+'.yaml'))

    def test_changed_descriptor_or_unsupported_metadata_fails_closed(self):
        for mutate in ('descriptor', 'range', 'corners', 'urdf', 'self_pair', 'self_range', 'self_projection',
                'expanded_range', 'fixed', 'padding', 'rules', 'sphere_count'):
            with self.subTest(mutate=mutate), TemporaryDirectory() as folder:
                directory = Path(folder)
                descriptor = directory/(MODEL_NAME+'.yaml')
                descriptor.write_bytes(self.model.descriptor_path.read_bytes())
                metadata = json.loads((MODEL_DIRECTORY/(MODEL_NAME+'.json')).read_text())
                if mutate == 'descriptor': descriptor.write_text('changed')
                if mutate == 'range': metadata['finger_range_m'] = [.005, .001]
                if mutate == 'corners': metadata['base_aabb_corners_m'] = [[0., 0., 0.]]*8
                if mutate == 'urdf': metadata['urdf_sha256'] = 'wrong'
                if mutate == 'self_pair':
                    metadata['certified_self_collision_exclusions'][0]['frames'] = ['panda_leftfinger', 'panda_link6']
                if mutate == 'self_range':
                    metadata['certified_self_collision_exclusions'][0]['finger_range_m'] = [0., .04]
                if mutate == 'self_projection':
                    metadata['certified_self_collision_exclusions'][0]['minimum_separating_projection_m'] = -1e-3
                if mutate == 'expanded_range':
                    metadata['finger_range_m'] = [.001, .04]
                    metadata['descriptor_fixed_finger_m'] = .0205
                    metadata['finger_sweep_padding_m'] = .0195
                    metadata['certified_self_collision_exclusions'][0]['finger_range_m'] = [.001, .04]
                if mutate == 'fixed': metadata['descriptor_fixed_finger_m'] = .004
                if mutate == 'padding': metadata['finger_sweep_padding_m'] = .02
                if mutate == 'sphere_count': metadata['sphere_count'] = 101
                if mutate == 'rules':
                    descriptor.write_text(descriptor.read_text().replace('  value: 0.003', '  value: 0.025'))
                    metadata['descriptor_sha256'] = hashlib.sha256(descriptor.read_bytes()).hexdigest()
                (directory/(MODEL_NAME+'.json')).write_text(json.dumps(metadata))
                with self.assertRaises(PlanningFailure) as caught:
                    load_enhanced_collision_model(directory)
                self.assertEqual(caught.exception.code, 'collision_model_invalid')

    def test_names_preserve_independent_finger_values_and_range_endpoints(self):
        result = self.model.check_gripper(self.robot((.001, .005)))
        self.assertEqual(result['measured_positions_m'], [.001, .005])
        self.assertIsNone(result['native_position_targets_m'])

    def test_bad_or_ambiguous_measured_fingers_reject(self):
        robots = [self.robot((.000999, .003)), self.robot((.003, .005001)),
                  self.robot((float('nan'), .003)), self.robot()]
        robots[-1].dof_names = (FINGER_NAMES[0], FINGER_NAMES[0], FINGER_NAMES[1])
        missing = self.robot(); missing.dof_names = ('other', 'arm', FINGER_NAMES[0])
        robots.append(missing)
        for robot in robots:
            with self.subTest(names=robot.dof_names), self.assertRaises(PlanningFailure) as caught:
                self.model.check_gripper(robot)
            self.assertEqual(caught.exception.code, 'unsupported_gripper_configuration')

    def test_native_latched_opening_target_rejects_before_measured_fingers_move(self):
        robot = self.robot()
        robot.get_applied_action = lambda: NS(joint_positions=[.04, .5, .003], joint_indices=None)
        with self.assertRaisesRegex(PlanningFailure, 'Native finger targets'):
            self.model.check_gripper(robot)
        robot.get_applied_action = lambda: NS(joint_positions=[.004, .5, .002], joint_indices=None)
        self.assertEqual(self.model.check_gripper(robot)['native_position_targets_m'], [.002, .004])

    def test_partial_target_order_is_resolved_and_omitted_fingers_are_explicit(self):
        robot = self.robot()
        robot.get_applied_action = lambda: NS(joint_positions=[.002, .004], joint_indices=[2, 0])
        self.assertEqual(self.model.check_gripper(robot)['native_position_targets_m'], [.002, .004])
        robot.get_applied_action = lambda: NS(joint_positions=[.5], joint_indices=[1])
        self.assertIn('omits fingers', self.model.check_gripper(robot)['target_feedback'])

    def test_native_target_getter_failure_is_not_treated_as_valid_feedback(self):
        robot = self.robot()
        def unavailable(): raise RuntimeError('native target read failed')
        robot.get_applied_action = unavailable
        with self.assertRaisesRegex(PlanningFailure, 'native target read failed'):
            self.model.check_gripper(robot)

    def test_partial_native_action_checks_each_available_finger_target(self):
        for index, name in ((2, FINGER_NAMES[0]), (0, FINGER_NAMES[1])):
            with self.subTest(finger=name):
                robot = self.robot()
                robot.get_applied_action = lambda: NS(joint_positions=[.04], joint_indices=[index])
                with self.assertRaisesRegex(PlanningFailure, 'Native finger targets') as caught:
                    self.model.check_gripper(robot)
                self.assertEqual(caught.exception.details['native_partial_position_targets_m'], {name: .04})
                robot.get_applied_action = lambda: NS(joint_positions=[.004], joint_indices=[index])
                result = self.model.check_gripper(robot)
                self.assertEqual(result['native_partial_position_targets_m'], {name: .004})

    def test_finite_cuboid_check_uses_rotated_box_not_world_aabb(self):
        model = replace(self.model, base_center=(2., 0., 0.), base_radius=.1)
        axis_aligned = StaticObstacle('/Box', 'cuboid', (0, 0, 0), dimensions=(.2, 4., .2))
        model.preflight_base(self.snapshot(axis_aligned), np.zeros(3), np.eye(3))
        turned = replace(axis_aligned, orientation=(np.sqrt(.5), 0., 0., np.sqrt(.5)))
        with self.assertRaises(PlanningFailure) as caught:
            model.preflight_base(self.snapshot(turned), np.zeros(3), np.eye(3))
        self.assertEqual(caught.exception.code, 'base_geometry_rejected')
        self.assertTrue(caught.exception.details['conservative_rejection'])
        self.assertFalse(caught.exception.details['physical_collision_proven'])

    def test_world_base_transform_and_sphere_radius_both_contribute(self):
        model = replace(self.model, base_center=(1., 0., 0.), base_radius=.2)
        obstacle = StaticObstacle('/Ball', 'sphere', (3., 1.25, 0.), radius=.1)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        with self.assertRaises(PlanningFailure):
            model.preflight_base(self.snapshot(obstacle), [3., 0., 0.], rotation)
        obstacle = replace(obstacle, position=(3., 1.4, 0.))
        result = model.preflight_base(self.snapshot(obstacle), [3., 0., 0.], rotation)
        self.assertAlmostEqual(result['checks'][0]['clearance_lower_bound_m'], .1)

    def test_capsule_check_includes_spine_endcaps_and_orientation(self):
        model = replace(self.model, base_center=(1.1, 0., 0.), base_radius=.1)
        capsule = StaticObstacle('/Capsule', 'capsule', (0, 0, 0),
            orientation=(np.sqrt(.5), 0., np.sqrt(.5), 0.), radius=.1, height=2.)
        with self.assertRaises(PlanningFailure):
            model.preflight_base(self.snapshot(capsule), np.zeros(3), np.eye(3))
        capsule = replace(capsule, position=(0., 0., 1.))
        model.preflight_base(self.snapshot(capsule), np.zeros(3), np.eye(3))

    def test_mounted_plane_uses_full_cooked_aabb_not_oversized_ball(self):
        corners = tuple(itertools.product((-.5, .5), (-.5, .5), (0., 1.)))
        model = replace(self.model, base_center=(0., 0., .5), base_radius=1., base_corners=corners)
        slab, domain = obstacle_from_primitive('/Ground', 'Plane', np.eye(4))
        result = model.preflight_base(self.snapshot(slab, domains=(domain,)), np.zeros(3), np.eye(3))
        self.assertEqual(result['checks'][0]['clearance_lower_bound_m'], 0.)
        with self.assertRaises(PlanningFailure):
            model.preflight_base(self.snapshot(slab, domains=(domain,)), [0., 0., -1e-9], np.eye(3))
        with self.assertRaises(PlanningFailure):
            model.preflight_base(self.snapshot(slab, domains=(domain,)), [26., 0., 0.], np.eye(3))

    def test_plane_bypass_requires_matching_certified_slab_and_domain(self):
        slab, domain = obstacle_from_primitive('/Ground', 'Plane', np.eye(4))
        for snapshot in (self.snapshot(slab), self.snapshot(replace(slab, position=(0., 0., -20.)), domains=(domain,)),
                self.snapshot(slab, domains=(domain, domain))):
            with self.assertRaises(PlanningFailure) as caught:
                self.model.preflight_base(snapshot, np.zeros(3), np.eye(3))
            self.assertEqual(caught.exception.code, 'unsupported_geometry')


if __name__ == '__main__':
    unittest.main()
