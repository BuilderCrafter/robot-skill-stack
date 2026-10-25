"""The research target adapter must preserve the real production pipeline."""
from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import LulaGlobalPlanner
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from scripts.global_detour_fixed_target_v4_3 import (
    ARM_NAMES, FixedJointTarget, FixedJointTargetPlanner, fixed_target_planner_factory,
)
from tests.unit.test_lula_global import PlannerFixture


class FixedJointTargetTests(unittest.TestCase):
    def fixture(self, **options):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        fixture = PlannerFixture(folder.name, **options)
        model = replace(load_enhanced_collision_model(),
            urdf_sha256=hashlib.sha256(fixture.urdf.read_bytes()).hexdigest())
        fixture.backend.robot.dof_names = (*ARM_NAMES, 'panda_finger_joint1', 'panda_finger_joint2')
        fixture.backend.robot.get_joint_positions = lambda: np.array([0.]*7+[.003, .003])
        mocked = patch('robot_skill_stack.integrations.isaac.manipulation.motion.lula_global.load_enhanced_collision_model',
                       return_value=model)
        mocked.start()
        self.addCleanup(mocked.stop)
        return fixture

    def fixed(self, **changes):
        values = dict(joint_names=ARM_NAMES, q=[.2, 0., 0., 0., 0., 0., 0.],
            position=[.5, 0., .4], orientation=[1., 0., 0., 0.])
        values.update(changes)
        return FixedJointTarget(**values)

    def planner(self, fixture, fixed=None):
        planner = FixedJointTargetPlanner(fixture.backend, fixed_target=fixed or self.fixed(),
            seed=123456, max_iterations=10000, _sdk=fixture.sdk)
        self.addCleanup(planner.close)
        return planner

    def plan(self, fixture, fixed=None):
        planner = self.planner(fixture, fixed)
        return planner, planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)

    def test_recorded_target_drives_unchanged_real_rrt_route_and_spline_checks(self):
        fixture = self.fixture()
        original_kinematics = fixture.sdk.Kinematics
        _, result = self.plan(fixture)
        self.assertIs(fixture.sdk.Kinematics, original_kinematics)
        self.assertFalse(any(row[0] == 'ik' for row in fixture.events))
        self.assertEqual(sum(row[0] == 'rrt_call' for row in fixture.events), 1)
        np.testing.assert_array_equal(result.route, [np.zeros(7), self.fixed().q])
        goal = next(row[1] for row in fixture.events if row[0] == 'cspace_goal')
        np.testing.assert_array_equal(goal, self.fixed().q)
        self.assertIn(('iterations', 10000), fixture.events)
        self.assertIn(('native_param', 'seed', 123456), fixture.events)
        self.assertGreater(result.diagnostics['configuration_validation_calls'], 0)
        self.assertEqual(result.diagnostics['trajectory_validation']['collision_validation'], 'PASS_SAMPLED_SDK_MODEL')
        research = result.diagnostics['research_fixed_joint_target']
        self.assertEqual(research['candidate_calls'][0]['q'], list(self.fixed().q))
        self.assertEqual(research['frozen_candidate_sha256'], self.fixed().sha256)
        self.assertFalse(research['normal_ik_solver_called'])
        # Search, IK/FK and generator still receive one enhanced description.
        configs = [row[1] for row in fixture.events
                   if row[0] in ('rrt_config', 'kinematics_config', 'generator_config')]
        self.assertEqual(len({config['robot_description_path'] for config in configs}), 1)
        self.assertTrue(configs[0]['robot_description_path'].endswith('franka_enhanced_v4_2.yaml'))

    def test_full_pose_fk_cannot_be_skipped_by_supplied_candidate(self):
        fixture = self.fixture()
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture, self.fixed(q=[.3, 0., 0., 0., 0., 0., 0.]))
        self.assertEqual(caught.exception.code, 'invalid_goal')
        self.assertIn('full-pose', str(caught.exception))
        self.assertFalse(any(row[0] in ('rrt_call', 'trajectory') for row in fixture.events))

    def test_full_orientation_fk_is_still_checked(self):
        fixture = self.fixture(orientation_ok=False)
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'invalid_goal')
        self.assertFalse(any(row[0] == 'rrt_call' for row in fixture.events))

    def test_joint_limits_remain_mandatory(self):
        fixture = self.fixture()
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture, self.fixed(q=[2., 0., 0., 0., 0., 0., 0.]))
        self.assertEqual(caught.exception.code, 'joint_limit_failure')
        self.assertFalse(any(row[0] in ('rrt_call', 'trajectory') for row in fixture.events))

    def test_goal_collision_is_rejected_before_rrt_without_switching_ik_branch(self):
        fixture = self.fixture(collision=lambda q: q[0] > .15)
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'invalid_goal')
        self.assertFalse(any(row[0] in ('rrt_call', 'trajectory', 'ik') for row in fixture.events))
        research = caught.exception.details['research_fixed_joint_target']
        self.assertEqual(len(research['candidate_calls']), 4)
        self.assertTrue(all(row['q'] == list(self.fixed().q) for row in research['candidate_calls']))

    def test_route_collision_still_fails_before_trajectory_generation(self):
        fixture = self.fixture(trajectory_collision=True)
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'route_invalid')
        self.assertEqual(sum(row[0] == 'rrt_call' for row in fixture.events), 1)
        self.assertFalse(any(row[0] == 'trajectory' for row in fixture.events))

    def test_generated_spline_collision_is_rejected_after_clear_route(self):
        fixture = self.fixture(smoothed_collision=True)
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'trajectory_failed')
        self.assertTrue(any(row[0] == 'trajectory' for row in fixture.events))
        self.assertFalse(any(row[0] == 'ik' for row in fixture.events))

    def test_exhausted_search_never_retries_another_goal_or_generates_fallback(self):
        fixture = self.fixture(search_success=False)
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'search_exhausted')
        self.assertEqual(sum(row[0] == 'rrt_call' for row in fixture.events), 1)
        self.assertFalse(any(row[0] in ('trajectory', 'ik') for row in fixture.events))
        attempts = caught.exception.details['ik_attempts']
        self.assertEqual([row['status'] for row in attempts],
            ['search_exhausted', 'duplicate_ik', 'duplicate_ik', 'duplicate_ik'])

    def test_enhanced_finger_guard_cannot_be_bypassed(self):
        fixture = self.fixture()
        fixture.backend.robot.get_joint_positions = lambda: np.array([0.]*7+[.04, .003])
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture)
        self.assertEqual(caught.exception.code, 'unsupported_gripper_configuration')
        self.assertFalse(any(row[0] in ('rrt_call', 'trajectory') for row in fixture.events))

    def test_other_full_pose_cannot_silently_fall_back_to_normal_ik(self):
        fixture = self.fixture()
        planner = self.planner(fixture)
        for target in (Pose([.4, 0., .4], [1., 0., 0., 0.]),
                       Pose([.5, 0., .4], [0., 1., 0., 0.]),
                       Pose([.5, 0., .4], [1., 0., 0., 0.], 'robot')):
            with self.subTest(target=target):
                with self.assertRaises(PlanningFailure) as caught:
                    planner.plan(target, np.zeros(7), fixture.snapshot, .025, .1)
                self.assertEqual(caught.exception.code, 'research_fixed_target_mismatch')
        self.assertFalse(any(row[0] in ('ik', 'rrt_call', 'trajectory') for row in fixture.events))

    def test_quaternion_sign_equivalence_does_not_change_physical_target(self):
        fixture = self.fixture()
        fixture.target = Pose([.5, 0., .4], [-1., 0., 0., 0.])
        _, result = self.plan(fixture)
        self.assertEqual(result.diagnostics['state'], 'planned')

    def test_native_joint_order_must_equal_frozen_candidate_order(self):
        fixture = self.fixture()
        planner = self.planner(fixture, self.fixed(joint_names=ARM_NAMES[::-1]))
        with self.assertRaises(PlanningFailure) as caught:
            planner.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
        self.assertEqual(caught.exception.code, 'research_fixed_joint_order_mismatch')
        self.assertFalse(any(row[0] in ('ik', 'rrt_call') for row in fixture.events))

    def test_pose_shortcut_does_not_masquerade_as_reaching_different_joint_goal(self):
        fixture = self.fixture()
        fixture.target = Pose([.3, 0., .4], [1., 0., 0., 0.])
        fixed = self.fixed(position=fixture.target.position, q=[0., .3, 0., 0., 0., 0., 0.])
        with self.assertRaises(PlanningFailure) as caught:
            self.plan(fixture, fixed)
        self.assertEqual(caught.exception.code, 'research_fixed_target_bypassed')
        self.assertFalse(any(row[0] in ('ik', 'rrt_call', 'trajectory') for row in fixture.events))

    def test_normal_production_ik_still_runs_after_research_planner_closes(self):
        fixture = self.fixture()
        planner, _ = self.plan(fixture)
        planner.close()
        normal = LulaGlobalPlanner(fixture.backend, seed=123456,
            max_iterations=10000, collision_model='enhanced', _sdk=fixture.sdk)
        self.addCleanup(normal.close)
        result = normal.plan(fixture.target, np.zeros(7), fixture.snapshot, .025, .1)
        self.assertEqual(result.diagnostics['state'], 'planned')
        self.assertEqual(sum(row[0] == 'ik' for row in fixture.events), 1)
        self.assertNotIn('research_fixed_joint_target', result.diagnostics)

    def test_frozen_manifest_subset_is_immutable_and_hash_binds_provenance(self):
        q = np.array([.2, 0., 0., 0., 0., 0., 0.])
        raw = dict(schema_version=1, joint_names=list(ARM_NAMES), q=q,
            target=dict(position=[.5, 0., .4], orientation=[1., 0., 0., 0.], frame='world'),
            source_result=dict(path='ignored/result.json', sha256='a'*64), model_sha256='unrelated-root-check')
        fixed = FixedJointTarget.from_dict(raw)
        q[0] = .8
        self.assertEqual(fixed.q[0], .2)
        self.assertEqual(fixed.baseline_artifact_sha256, 'a'*64)
        self.assertEqual(FixedJointTarget.from_dict(fixed.as_dict()).sha256, fixed.sha256)
        changed = replace(fixed, baseline_artifact_sha256='b'*64)
        self.assertNotEqual(changed.sha256, fixed.sha256)
        self.assertEqual(fixed.pose().frame, 'world')

    def test_invalid_manifest_or_unselected_model_fails_before_sdk_calls(self):
        fixture = self.fixture()
        with self.assertRaises(ValueError):
            self.fixed(q=[float('nan')]*7)
        with self.assertRaises(ValueError):
            self.fixed(joint_names=ARM_NAMES[:-1])
        with self.assertRaises(ValueError):
            self.fixed(orientation=[0.]*4)
        with self.assertRaises(ValueError):
            self.fixed(baseline_artifact_sha256='bad')
        with self.assertRaises(ValueError):
            fixed_target_planner_factory(self.fixed())(fixture.backend, collision_model='stock')
        self.assertEqual(fixture.events, [])


if __name__ == '__main__':
    unittest.main()
