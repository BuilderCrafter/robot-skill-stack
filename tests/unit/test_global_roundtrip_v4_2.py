"""SDK-free gates for the separate, independently frozen Home roundtrip."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from scripts import benchmark_global_roundtrip_v4_2 as harness


class FrozenRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.frozen = harness.load_protocol()

    def state(self):
        return dict(feedback=dict(positions=self.frozen['initial_joint_positions']),
            tcp_pose=self.frozen['initial_home'],
            live_drive_feedback=dict(robot_applied_action=dict(available=True,
                value=dict(joint_positions=self.frozen['initial_joint_positions']))))

    def test_new_fixture_keeps_original_home_and_target_separate_from_historical_fixtures(self):
        self.assertEqual(self.frozen['fixture']['position_m'], [.55, -.05, .25])
        self.assertEqual(self.frozen['fixture']['dimensions_m'], [.04, .04, .12])
        self.assertEqual(self.frozen['outward_target']['position'], [.45, -.25, .35])
        np.testing.assert_allclose(self.frozen['initial_home']['position'],
            [.5974912637910632, -.08603929056981253, .3644922301295057], atol=0., rtol=0.)
        self.assertEqual(self.frozen['candidate_RRT_calls_before_freeze'], 0)
        self.assertFalse(self.frozen['fixture_revision_after_candidate_RRT_allowed'])
        self.assertEqual(self.protocol['seeds'], [123456, 234567, 345678])
        self.assertEqual(self.protocol['fresh_primary_repetitions'], 3)
        self.assertEqual(self.protocol['total_roundtrip_processes_per_model'], 5)

    def test_frozen_fixture_edit_is_rejected_before_native_start(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            changed = copy.deepcopy(self.frozen)
            changed['fixture']['position_m'][0] += .001
            (folder/'roundtrip_fixture.json').write_text(json.dumps(changed))
            (folder/'protocol.json').write_text(json.dumps(self.protocol))
            with self.assertRaisesRegex(ValueError, 'Frozen fixture hash changed'):
                harness.load_protocol(folder/'protocol.json')

    def test_changed_independent_design_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'declaration.json').write_text('changed')
            protocol = dict(design_artifact_directory=str(folder), raw_recorded_source_directory=str(folder),
                design_artifact_sha256={'declaration.json':'wrong'}, recorded_source_sha256={})
            with self.assertRaisesRegex(ValueError, 'independent design source hash changed'):
                harness.load_design_evidence(protocol)

    def test_startup_requires_measured_joints_tcp_and_unchanged_runtime_home(self):
        state = self.state()
        self.assertTrue(harness.startup_match(state, self.frozen['initial_home'],
            self.protocol, self.frozen)['matched'])
        changed = copy.deepcopy(state)
        changed['feedback']['positions'][0] += .002
        self.assertFalse(harness.startup_match(changed, self.frozen['initial_home'],
            self.protocol, self.frozen)['matched'])
        changed_home = copy.deepcopy(self.frozen['initial_home'])
        changed_home['position'][0] += .001
        self.assertFalse(harness.startup_match(state, changed_home,
            self.protocol, self.frozen)['matched'])
        changed = copy.deepcopy(state)
        changed['tcp_pose']['orientation'] = [1., 0., 0., 0.]
        self.assertFalse(harness.startup_match(changed, self.frozen['initial_home'],
            self.protocol, self.frozen)['matched'])

    def test_fresh_profiles_select_model_and_keep_original_limits_and_no_mapping(self):
        for mode in harness.MODES:
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(mode=mode, collision_model='enhanced', seed=234567,
                    max_iterations=10000, artifacts_dir=directory)
                config = load_scene_config(harness.write_profile(args))
            self.assertEqual(config.motion.global_collision_model, 'enhanced')
            self.assertEqual(config.motion.rrt_seed, 234567)
            self.assertEqual(config.motion.rrt_max_iterations, 10000)
            self.assertEqual(config.motion.strategy, 'direct' if mode == 'direct_reference' else 'global')
            self.assertFalse(config.spatial_mapping.enabled)
            self.assertIsNone(config.perception.camera_prim_path)

    def parts(self):
        mesh = SimpleNamespace(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
            triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]))
        return [SimpleNamespace(body_path='/Franka/'+name, mesh=mesh,
            body_from_mesh=np.eye(4), approximation='physx_cooked:test') for name in harness.BODY_NAMES]

    def test_cooked_fingerprint_requires_base_hand_and_each_finger_and_detects_geometry_changes(self):
        parts = self.parts()
        first = harness.cooked_fingerprints(parts)
        with self.assertRaisesRegex(ValueError, 'all11 bodies'):
            harness.cooked_fingerprints(parts[1:])
        bad = self.parts()
        bad[0].approximation = 'authored_fallback'
        with self.assertRaisesRegex(ValueError, 'actually cooked'):
            harness.cooked_fingerprints(bad)
        changed = self.parts()
        changed[0].body_from_mesh[0,3] = .001
        self.assertNotEqual(first, harness.cooked_fingerprints(changed))

    def test_zero_motion_home_or_contact_or_intersection_cannot_pass(self):
        row = dict(returned_success=True, measured_full_pose_within_tolerances=True,
            clearance=dict(minimum_m=.01), observed_physical_contact_steps=0,
            maximum_actual_joint_displacement_rad=.3)
        operations = [dict(row, label=label) for label in ('outward','home_return')]
        self.assertTrue(all(harness.physical_checks(operations).values()))
        for change in (dict(maximum_actual_joint_displacement_rad=0.),
                       dict(observed_physical_contact_steps=1), dict(clearance=dict(minimum_m=0.))):
            self.assertFalse(harness.physical_checks([operations[0],
                dict(operations[1], **change)])['home_return'])
        self.assertFalse(harness.physical_checks(operations[:1])['home_return'])

    def test_physics_coverage_exposes_skipped_steps_and_deduplicates_boundaries(self):
        rows = [dict(physics_step=step, simulation_time=step/60, value=index)
                for index,step in enumerate((4,5,5,7))]
        self.assertEqual(harness.physics_step_coverage(rows)['missing_steps'], [6])
        self.assertFalse(harness.physics_step_coverage(rows)['complete'])
        unique = harness.unique_physics_samples(rows)
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique[1]['value'], 2)
        self.assertTrue(harness.physics_step_coverage(rows[:3])['complete'])

    def test_existing_partial_capture_is_preserved_even_without_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(artifacts_dir=directory, result=str(Path(directory)/'result.json'))
            harness.refuse_overwrite(args)
            (Path(directory)/'geometry_reference.npz').write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                harness.refuse_overwrite(args)

    def test_preflight_rejects_geometry_mismatch_and_setup_action_before_any_rrt(self):
        observer = SimpleNamespace(state=self.state, backend=SimpleNamespace(named_poses={
            'home':self.frozen['initial_home']}), metadata=dict(body_paths=[]), samples=[], actions=[])
        observer.sample = lambda: observer.samples.append(dict(poses=[], physics_step=32))
        capture = dict(meshes=[], pose_matrices=lambda poses,units: np.empty((0,4,4)))
        evidence = dict(generation=dict(samples=[dict(poses=[]) for _ in range(339)],
            metadata=dict(body_paths=[])), certificate={}, actual={
                'minimum_continuous_nonadjacent_self_separation_lower_bound_m':.004})
        with patch.object(harness, 'cooked_fingerprints', return_value={'fresh':'different'}), \
                patch.object(harness, 'reference_fingerprints', return_value={'frozen':'reference'}):
            with self.assertRaisesRegex(RuntimeError, 'Fresh PhysX-cooked geometry differs'):
                harness.preflight(observer,capture,self.protocol,self.frozen,evidence)
        with patch.object(harness, 'cooked_fingerprints', return_value={}), \
                patch.object(harness, 'reference_fingerprints', return_value={}), \
                patch.object(harness.baseline, 'reduce_clearances', return_value=dict(minimum_m=.04)):
            observer.actions.append(dict(command='forbidden_setup'))
            with self.assertRaisesRegex(RuntimeError, 'START setup is forbidden'):
                harness.preflight(observer,capture,self.protocol,self.frozen,evidence)
            observer.actions.clear()
            measured = harness.preflight(observer,capture,self.protocol,self.frozen,evidence)
            self.assertTrue(measured['no_setup_arm_actions'])
            self.assertEqual(measured['startup_match']['latched_applied_action'],
                self.state()['live_drive_feedback']['robot_applied_action'])


if __name__ == '__main__':
    unittest.main()
