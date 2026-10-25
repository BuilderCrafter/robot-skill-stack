import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_global_motion_v4 import (
    _load_bound_artifact, declaration, digest, load_attempt, main, summarize,
)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def case(self, name, *, strategy='global', seed=11, status='PASS', goal=True,
             strict=True, reports=0, reference=None, witness=False, lifecycle=False):
        folder = self.folder/name
        folder.mkdir()
        config = dict(strategy=strategy, scenario='static', seed=seed, max_iterations=10000,
            start=[.45, -.25, .35], goal=[.45, .25, .35], orientation=[0, 0, 1, 0],
            objects=[dict(name='Post', dimensions=[.09, .12, .26])],
            position_tolerance_m=.025, orientation_tolerance_rad=.1,
            max_motion_steps=1000, witness=witness)
        diagnostics = dict(ik_candidates_attempted=1, rrt_waypoints=4, rrt_seconds=.1,
            planning_and_validation_wall_seconds=.2, trajectory_generation_seconds=.01,
            execution_wall_seconds=2., execution_sim_seconds=1.5,
            trajectory_duration_seconds=1.5,
            trajectory_validation=dict(collision_validation='PASS_SAMPLED_SDK_MODEL', samples=200))
        metrics = dict(configuration=config, initial_joints=[0.]*9,
            physical_world=[dict(path='/Post', kind='cuboid', dimensions=[.09, .12, .26])],
            motion=dict(result=dict(status='success' if goal else 'failure', details={}), wall_seconds=2.3),
            goal_reached=goal, strict_success=strict, movable_arm_strict_success=strict,
            strategy_diagnostics=diagnostics if strategy == 'global' else {},
            contacts=dict(reported_points=reports, penetrating_pairs=0, links=['/Franka/panda_hand'] if reports else []),
            contact_severity=dict(physical_classification='contact_offset_proximity' if reports else 'no_physical_contact_observed',
                observed_physical_contact_steps=0, depth_m=dict(maximum=0.), normal_pair_impulse_ns=dict(maximum=0.)),
            clearance=dict(minimum_m=.04, movable_arm_minimum_m=.04),
            lifecycle=dict(status='PASS', excluded_from_primary_timing=True) if lifecycle else None)
        payload = dict(status=status, contracts=dict(source_unchanged=True, runtime_closed=True),
            hashes={'example.py': 'a'*64}, metrics=metrics, error=None)
        command = ['python', 'benchmark_global_motion_v4.py', '--strategy', strategy]
        if reference is not None:
            command += ['--reference-result', str(reference)]
            metrics['reference_match'] = dict(matched=True, checks=dict(actual_initial_joints=True,
                source_manifest=True, physical_world=True), maximum_joint_error_rad=0., joint_tolerance_rad=.01)
        if witness:
            command += ['--witness']
            metrics['witness_segments'] = [dict(result={'status': 'success'})]*3
        (folder/'launch.json').write_text(json.dumps(dict(command=command, cwd=str(self.folder))))
        path = folder/'result.json'
        path.write_text(json.dumps(payload))
        return path

    def mutate(self, path, callback):
        data = json.loads(path.read_text())
        callback(data)
        path.write_text(json.dumps(data))

    def test_valid_measurement_is_not_goal_or_avoidance_success(self):
        path = self.case('exhausted', goal=False, strict=False)
        self.mutate(path, lambda raw: raw['metrics']['strategy_diagnostics'].update(
            failure_code='search_exhausted', trajectory_validation=None))
        report = summarize([path])
        self.assertEqual(report['counts']['valid_measurements'], 1)
        self.assertEqual(report['counts']['strict_successes'], 0)
        row = report['cases'][0]
        self.assertEqual(row['failure_code'], 'search_exhausted')
        self.assertFalse(row['global_trajectory_validation_passed'])

    def test_positive_separation_reports_are_not_physical_impacts(self):
        path = self.case('proximity', strict=False, reports=11)
        row, _, _ = load_attempt(path)
        self.assertFalse(row['strict_success'])
        self.assertEqual(row['reported_contact_points'], 11)
        self.assertEqual(row['observed_physical_contact_steps'], 0)
        self.assertEqual(row['physical_contact_classification'], 'contact_offset_proximity')

    def test_false_contract_invalidates_process_pass(self):
        path = self.case('bad-contract')
        self.mutate(path, lambda raw: raw['contracts'].update(runtime_closed=False))
        row, _, _ = load_attempt(path)
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['strict_success'])

    def test_actual_declared_edges_only_no_cartesian_seed_pairs(self):
        baseline1 = self.case('direct-r1', strategy='direct')
        baseline2 = self.case('direct-r2', strategy='direct')
        global1 = self.case('global-seed11', reference=baseline1)
        global2 = self.case('global-seed12', seed=12, reference=baseline1)
        report = summarize([baseline1, baseline2, global1, global2])
        self.assertEqual(report['counts']['declared_reference_edges'], 2)
        self.assertEqual(report['counts']['eligible_reference_edges'], 2)
        self.assertEqual(report['counts']['unique_declared_baselines'], 1)

    def test_joint_and_scene_checks_reject_falsely_claimed_matching(self):
        for field in ('initial_joints', 'physical_world'):
            with self.subTest(field=field):
                baseline = self.case('base-'+field, strategy='direct')
                candidate = self.case('candidate-'+field, reference=baseline)
                if field == 'initial_joints':
                    self.mutate(candidate, lambda raw: raw['metrics']['initial_joints'].__setitem__(0, .5))
                else:
                    self.mutate(candidate, lambda raw: raw['metrics']['physical_world'][0].update(dimensions=[1, 1, 1]))
                edge = summarize([baseline, candidate])['declared_comparisons'][0]
                self.assertFalse(edge['comparison_eligible'])

    def test_failed_seeds_and_missing_artifacts_are_retained(self):
        first = self.case('seed11')
        failed = self.case('seed12', seed=12, status='FAIL', goal=False, strict=False)
        missing = self.folder/'missing/result.json'
        report = summarize([first, failed, missing])
        self.assertEqual(report['counts']['unique_attempts'], 3)
        self.assertEqual(report['counts']['valid_measurements'], 1)
        self.assertTrue(report['cases'][-1]['errors'])

    def test_witness_separate_but_main_lifecycle_trial_stays_primary(self):
        witness = self.case('witness', strategy='direct', witness=True)
        lifecycle = self.case('empty-lifecycle', lifecycle=True)
        report = summarize([witness, lifecycle])
        self.assertEqual(report['modes'], {'witness': 1, 'primary': 1})
        self.assertEqual(report['cohort_counts'], {'witness': 1, 'lifecycle': 1})
        self.assertEqual(len(report['cases'][0]['witness_segments']), 3)
        self.assertEqual(report['cases'][1]['lifecycle']['status'], 'PASS')

    def test_lifecycle_compact_keeps_failures_and_counts_without_raw_trajectories(self):
        path = self.case('failed-lifecycle', lifecycle=True)
        def add_episode(raw):
            raw['metrics']['lifecycle'].update(status='FAIL', checks={'home_completed': False},
                episodes=[dict(name='home', actions=[{'q': ['RAW_TARGET_SENTINEL']}, {'q': [1, 2]}],
                    result=dict(status='failed', failure_code='path_blocked', message='Tracking failed',
                        details=dict(failure_code='tracking_failure', execution_wall_seconds=.2,
                            ik_attempts=[{'warm_start': ['RAW_IK_SENTINEL']}])) )])
        self.mutate(path, add_episode)
        row, _, _ = load_attempt(path)
        lifecycle = row['lifecycle']
        self.assertEqual(lifecycle['status'], 'FAIL')
        self.assertFalse(lifecycle['checks']['home_completed'])
        episode = lifecycle['episodes'][0]
        self.assertEqual(episode['recorded_action_count'], 2)
        self.assertEqual(episode['internal_failure_code'], 'tracking_failure')
        self.assertEqual(episode['timing']['execution_wall_seconds'], .2)
        self.assertEqual(lifecycle['source']['sha256'], digest(path))
        self.assertNotIn('RAW_TARGET_SENTINEL', json.dumps(row))
        self.assertNotIn('RAW_IK_SENTINEL', json.dumps(row))

    def test_duplicate_requested_artifacts_do_not_inflate_counts(self):
        path = self.case('once')
        report = summarize([path, path.parent, path])
        self.assertEqual(report['counts']['unique_attempts'], 1)
        self.assertEqual(len(report['duplicate_requested_artifacts']), 2)

    def test_ambiguous_launch_declaration_is_not_guessed(self):
        path = self.case('ambiguous')
        (path.parent/'launch.json').write_text(json.dumps(dict(command=[
            'python', '--reference-result', 'a', '--reference-result', 'b'])))
        self.assertIsNotNone(declaration(path.parent)['error'])
        self.assertFalse(load_attempt(path)[0]['measurement_valid'])

    def test_no_launch_does_not_invent_a_reference(self):
        path = self.case('manual')
        (path.parent/'launch.json').unlink()
        row, _, _ = load_attempt(path)
        self.assertTrue(row['measurement_valid'])
        self.assertIsNone(row['declared_reference'])

    def test_missing_top_level_rrt_time_uses_recorded_attempt_times_only(self):
        path = self.case('failed-attempts', goal=False, strict=False)
        def change(raw):
            diagnostics = raw['metrics']['strategy_diagnostics']
            diagnostics.pop('rrt_seconds')
            diagnostics['ik_attempts'] = [dict(index=0, rrt_seconds=.1),
                dict(index=1, ik_seconds=.4), dict(index=2, rrt_seconds=.2)]
        self.mutate(path, change)
        self.assertAlmostEqual(load_attempt(path)[0]['rrt_seconds'], .3)

    def test_reduction_does_not_modify_native_inputs(self):
        path = self.case('immutable')
        before = path.read_bytes()
        summarize([path])
        self.assertEqual(before, path.read_bytes())

    def test_oversized_evidence_fails_without_dropping_rows_or_writing_output(self):
        path = self.case('case')
        output = self.folder/'compact.json'
        with self.assertRaisesRegex(ValueError, 'no attempts were removed'):
            main(['--cases', str(path), '--result', str(output), '--max-bytes', '100'])
        self.assertFalse(output.exists())

    def test_plot_artifact_hash_mismatch_is_rejected(self):
        path = self.case('bound')
        trace = path.parent/'trace.json'
        trace.write_text('{}')
        row, _, _ = load_attempt(path)
        trace.write_text('{"tcp":[]}')
        with self.assertRaisesRegex(ValueError, 'changed after summary'):
            _load_bound_artifact(row, 'trace.json')

    def test_nonobject_result_is_retained_as_invalid(self):
        path = self.folder/'nonobject.json'
        path.write_text('[]')
        row, _, _ = load_attempt(path)
        self.assertFalse(row['measurement_valid'])
        self.assertTrue(row['errors'])


if __name__ == '__main__':
    unittest.main()
