import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_global_motion_v4 import digest
from scripts.summarize_global_stabilization_v4_1 import (
    controlled_record, native_count_aggregates, regression_record, static_comparisons, static_record,
)


class StabilizationSummaryTests(unittest.TestCase):
    def write_controlled(self, folder, *, behavior=False, hold=None):
        folder = Path(folder)
        trace = folder/'controlled_trace.json'
        trace.write_text(json.dumps(dict(samples=[
            dict(native_actor_solver_properties=dict(available=True, position=n, velocity=1),
                poses=['RAW_POSE_SENTINEL']) for n in (32, 255, 255, 32)],
            operations=[dict(id=0, label='home', first_sample=0, last_sample=4)])))
        op = dict(id=0, label='home', kind='motion', returned_success=behavior,
            strategy_diagnostics=dict(tracking_abort_rad=.4, start_q=['RAW_POSE_SENTINEL']),
            poses=['RAW_POSE_SENTINEL'], trajectory=['RAW_TRAJECTORY_SENTINEL'])
        if hold is not None:
            op['post_home_hold'] = hold
        payload = dict(status='PASS', controlled=dict(mode='C', strategy='global',
            checks=dict(observer_available=True), acceptance_passed=behavior, operations=[op],
            source_hashes_before={'runtime.py': 'a'*64}, source_hashes_after={'runtime.py': 'a'*64},
            raw_trace=dict(path=str(trace), sha256=digest(trace))))
        result = folder/'result.json';result.write_text(json.dumps(payload))
        return result, trace

    def test_valid_measurement_does_not_imply_behavior_and_raw_arrays_stay_private(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.write_controlled(folder)
            row = controlled_record(result)
        self.assertTrue(row['measurement_valid'])
        self.assertFalse(row['behavior_acceptance_passed'])
        self.assertIsNone(row['post_hold_passed'])
        self.assertFalse(row['raw_trace_verified'])
        self.assertEqual(row['operations'][0]['diagnostics']['tracking_abort_rad'], .4)
        self.assertNotIn('RAW_POSE_SENTINEL', json.dumps(row))
        self.assertNotIn('RAW_TRAJECTORY_SENTINEL', json.dumps(row))

    def test_successful_motion_can_fail_post_hold_and_zero_velocity_is_not_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.write_controlled(folder, behavior=True, hold=dict(
                kinematic_stability_passed=False, final_actual_arm_velocity_rad_s=[0.]*7))
            row = controlled_record(result)
        self.assertTrue(row['behavior_acceptance_passed'])
        self.assertFalse(row['post_hold_passed'])
        self.assertEqual(row['operations'][0]['post_home_hold']['final_measured_arm_velocity_max_rad_s'], 0.)

    def test_native_counts_compress_verified_transitions_and_reject_tampered_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.write_controlled(folder)
            row = controlled_record(result, raw_counts=True)
            operation = row['native_counts']['operations'][0]
            self.assertEqual(operation['available_samples'], 4)
            self.assertEqual(operation['transitions'], [dict(position=n, velocity=1) for n in (32, 255, 32)])
            self.assertTrue(row['raw_trace_verified'])
            reference = json.loads(result.read_text())['controlled']['raw_trace']
            trace.write_text(trace.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                native_count_aggregates(reference)

    def test_independent_analysis_requires_matching_result_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.write_controlled(folder)
            analysis = dict(cases=[dict(artifact_sha256='wrong', operations=[])])
            self.assertNotIn('independent_analysis', controlled_record(result, analysis=analysis))
            analysis['cases'][0]['artifact_sha256'] = digest(result)
            self.assertEqual(controlled_record(result, analysis=analysis)['independent_analysis'], [])

    def test_v4_loader_metrics_are_used_and_source_change_stays_qualified(self):
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for name, seconds, source in (('before', 1., 'a'), ('after', 2., 'b')):
                path = Path(folder)/(name+'.json')
                path.write_text(json.dumps(dict(status='PASS', contracts=dict(source_unchanged=True),
                    hashes={'runtime': source*64}, metrics=dict(configuration=dict(scenario='large', strategy='global', seed=123),
                        physical_world=[{'name': 'same_fixture'}], initial_joints=[0.]*9,
                        motion=dict(wall_seconds=seconds, result=dict(status='succeeded')), strict_success=True))))
                records.append(static_record(path))
            comparison = static_comparisons([records[1]], [records[0]])[0]
            self.assertTrue(comparison['physical_configuration_equal'])
            self.assertFalse(comparison['source_cohorts_equal'])
            self.assertEqual(comparison['actual_initial_joint_max_difference_rad'], 0.)
            self.assertEqual(comparison['descriptive_deltas']['motion_wall_seconds']['percent_change'], 100.)
            ambiguous = static_comparisons([records[1]], [records[0], records[0]])[0]
            self.assertEqual(ambiguous['matching_references'], 2)
            self.assertNotIn('descriptive_deltas', ambiguous)
            self.assertNotIn('initial_joints', records[1][0])

    def test_regression_failures_and_unavailable_inputs_are_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'summary.json'
            path.write_text(json.dumps(dict(unit=dict(status='PASS', return_code=0), mapping=dict(status='FAIL', return_code=1))))
            self.assertFalse(regression_record(path)['all_groups_passed'])
            self.assertEqual(regression_record(Path(folder)/'missing.json')['status'], 'UNAVAILABLE')

    def test_valid_invalid_goal_rejection_is_not_an_execution_tracking_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.write_controlled(folder)
            raw = json.loads(result.read_text())
            raw['controlled']['operations'][0].update(action_count=0,
                strategy_diagnostics=dict(failure_code='invalid_goal'))
            result.write_text(json.dumps(raw))
            row = controlled_record(result)
        self.assertTrue(row['measurement_valid'])
        self.assertFalse(row['behavior_acceptance_passed'])
        self.assertEqual(row['operations'][0]['outcome_class'], 'goal_rejected_before_execution')

    def test_invalid_attempts_and_missing_capture_sections_remain_bound_records(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.write_controlled(folder)
            raw = json.loads(result.read_text());raw['status'] = 'FAIL';raw['error'] = 'PRIVATE_ERROR_ARGUMENTS'
            raw['controlled']['checks'] = dict(callback_complete=False, physical_capture_available=False)
            result.write_text(json.dumps(raw))
            row = controlled_record(result)
            self.assertFalse(row['measurement_valid'])
            self.assertEqual(row['measurement_assessment'], 'invalid_observation_not_an_executor_verdict')
            self.assertEqual(row['failed_observation_checks'], ['callback_complete', 'physical_capture_available'])
            self.assertTrue(row['error_recorded'])
            self.assertEqual(row['operations'][0]['outcome_class'], 'not_assessed_invalid_observation')
            self.assertNotIn('PRIVATE_ERROR_ARGUMENTS', json.dumps(row))
            raw.pop('controlled');result.write_text(json.dumps(raw))
            row = controlled_record(result)
            self.assertEqual(row['measurement_assessment'], 'missing_observation_section')
            self.assertFalse(row['measurement_valid'])
            self.assertEqual(row['artifact']['sha256'], digest(result))

    def test_static_geometry_validation_contracts_and_restoration_scalars_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            path.write_text(json.dumps(dict(status='PASS', contracts=dict(cleanup_complete=True),
                metrics=dict(configuration=dict(scenario='empty', strategy='global', seed=123, max_iterations=10000),
                    goal_reached=True, no_requested_movement_joint_delta_rad=.002,
                    clearance=dict(by_link_m={'hand': .2}, by_group_m={'gripper': .2}, by_obstacle_m={'box': .2}),
                    contacts=dict(minimum_reported_separation_m=.01),
                    contact_severity=dict(physical_classification='no_physical_contact_observed'),
                    strategy_diagnostics=dict(geometric_joint_path_length=.4, rrt_waypoints=2, final_hold=True,
                        trajectory_validation=dict(collision_validation='PASS_SAMPLED_SDK_MODEL',
                            maximum_velocity_rad_s=['PRIVATE_ARRAY']), execution_solver=dict(
                            requested_position_iterations_during=255, original_session_opinion_restored=True))))))
            row, _ = static_record(path)
        self.assertTrue(row['reported_goal_reached'])
        self.assertTrue(row['global_trajectory_validation_passed'])
        self.assertEqual(row['max_iterations'], 10000)
        self.assertEqual(row['geometric_joint_path_length'], .4)
        self.assertEqual(row['clearance_by_link_m'], {'hand': .2})
        self.assertEqual(row['contracts'], {'cleanup_complete': True})
        self.assertTrue(row['execution_solver']['original_session_opinion_restored'])
        self.assertNotIn('PRIVATE_ARRAY', json.dumps(row))

    def test_native_regression_envelopes_keep_acceptance_and_boolean_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'native.json'
            raw = dict(status='PASS', checks=dict(direct_selected=True), native_acceptance=dict(home=True))
            path.write_text(json.dumps(raw));row = regression_record(path)
            self.assertIsNone(row['all_groups_passed'])
            self.assertTrue(row['native_envelope_passed'])
            self.assertTrue(row['regression_passed'])
            self.assertEqual(row['native_checks']['native_acceptance'], {'home': True})
            raw['native_acceptance']['home'] = False
            path.write_text(json.dumps(raw))
            self.assertFalse(regression_record(path)['regression_passed'])
            path.write_text(json.dumps(dict(status='PASS', checks={})))
            self.assertFalse(regression_record(path)['regression_passed'])


if __name__ == '__main__':
    unittest.main()
