"""Compact evidence must distinguish valid observations from successful motion."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_global_validation_v4_2 import (
    build_reports, inventory, lifecycle_record, performance, regression_record, static_record,
)


class ValidationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def write(self, relative, value):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return path

    def static(self, *, goal=True, valid=True, reports=2):
        return dict(status='PASS' if valid else 'FAIL', contracts=dict(source_unchanged=True), error=None,
            hashes={'runtime.py': 'frozen'}, metrics=dict(
                configuration=dict(strategy='global', global_collision_model='enhanced', scenario='static', seed=123456),
                goal_reached=goal, strict_success=goal and reports == 0,
                clearance=dict(minimum_m=.01, movable_arm_minimum_m=.02),
                contacts=dict(reported_points=reports, minimum_reported_separation_m=.015,
                    contact_buffer_saturated=False),
                contact_severity=dict(observed_physical_contact_steps=0,
                    depth_m=dict(maximum=0.), normal_pair_impulse_ns=dict(maximum=0.)),
                closest_robot_link_by_obstacle={'/Post': dict(robot_link='/Franka/panda_hand', clearance_m=.02)},
                memory_before_native=dict(current_rss_bytes=1000), kit_interval_ms=dict(p50=4., worst=300.),
                strategy_diagnostics=dict(global_collision_model='enhanced', failure_code=None if goal else 'invalid_goal',
                    rrt_seconds=.01, route_validation_seconds=.02, trajectory_validation_seconds=.03,
                    planning_and_validation_wall_seconds=.1, configuration_validation_calls=10,
                    configuration_validation_wall_seconds=.001, native_rrt_collision_query_count=None)))

    def test_proximity_reports_preserve_strict_failure_and_physical_completion_separately(self):
        path = self.write('static/enhanced/static/r1/result.json', self.static())
        row = static_record(path)
        self.assertTrue(row['measurement_valid'])
        self.assertFalse(row['metrics']['strict_success'])
        self.assertTrue(row['sampled_physical_contact_free_completion'])
        self.assertEqual(row['contacts']['positive_separation_reported_points'], 2)
        self.assertEqual(row['metrics']['closest_robot_link_by_obstacle']['/Post']['robot_link'], '/Franka/panda_hand')
        self.assertIsNotNone(row['artifact']['sha256'])

    def test_rejection_does_not_establish_physical_infeasibility_or_completion(self):
        row = static_record(self.write('result.json', self.static(goal=False, reports=0)))
        self.assertFalse(row['sampled_physical_contact_free_completion'])
        self.assertEqual(row['rejection_physical_feasibility'], 'NOT_ESTABLISHED_BY_PLANNER_REJECTION')

    def test_zero_reports_without_independent_clearance_cannot_establish_physical_completion(self):
        raw = self.static(reports=0); raw['metrics']['clearance'] = {}
        row = static_record(self.write('result.json', raw))
        self.assertIsNone(row['sampled_physical_contact_free_completion'])
        raw['metrics']['clearance'] = dict(minimum_m=0.)
        row = static_record(self.write('result.json', raw))
        self.assertFalse(row['sampled_physical_contact_free_completion'])

    def test_invalid_or_driver_failed_attempts_are_retained_and_not_pooled(self):
        path = self.write('result.json', self.static())
        self.write('driver_result.json', dict(status='FAIL', return_code=1))
        row = static_record(path)
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['sampled_physical_contact_free_completion'])
        reduced = performance([row])
        self.assertEqual(len(reduced['attempts']), 1)
        self.assertEqual(reduced['descriptive_cohorts'], [])

    def test_every_declared_missing_failed_and_additional_attempt_is_inventoried(self):
        manifest = self.write('protocol.json', dict(jobs=[dict(label='declared',
            result='outputs/motion-planning-v4-2/static/enhanced/empty/r1/result.json')]))
        self.write('static/extra/launch.json', dict(command=[]))
        self.write('static/another/result.json', dict(status='FAIL'))
        rows = inventory(self.root, ('static',), (manifest,))
        self.assertEqual(len(rows), 3)
        states = {static_record(path, declared)['measurement_status'] for path, declared in rows}
        self.assertEqual(states, {'NOT_RUN', 'NO_RESULT_AFTER_LAUNCH', 'FAIL'})

    def test_lifecycle_observation_failure_never_becomes_behavior_pass(self):
        raw = self.static()
        raw['acceptance'] = dict(mode='cancel', acceptance_passed=True,
            observation_checks={'complete': False}, checks={'cancelled': True}, operations=[])
        row = lifecycle_record(self.write('result.json', raw))
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['behavior_acceptance_passed'])

    def test_summary_contains_route_spline_kit_rss_and_separate_native_unknowns(self):
        row = static_record(self.write('result.json', self.static()))
        result = performance([row]); attempt = result['attempts'][0]
        self.assertEqual(attempt['timing']['route_validation_seconds'], .02)
        self.assertEqual(attempt['timing']['trajectory_validation_seconds'], .03)
        self.assertEqual(attempt['kit_interval_ms']['worst'], 300.)
        self.assertEqual(attempt['memory']['memory_before_native']['current_rss_bytes'], 1000)
        self.assertAlmostEqual(attempt['mean_explicit_configuration_check_seconds'], .0001)
        self.assertIsNone(attempt['timing']['native_rrt_collision_query_count'])

    def test_native_regression_and_group_summary_have_independent_check_semantics(self):
        native = regression_record(self.write('native.json', dict(status='PASS', checks={'move': True})))
        self.assertTrue(native['regression_passed'])
        groups = regression_record(self.write('summary.json', dict(unit=dict(status='PASS', return_code=0))))
        self.assertTrue(groups['regression_passed'])
        bad = regression_record(self.write('bad.json', dict(status='PASS', checks={'move': False})))
        self.assertFalse(bad['regression_passed'])

    def test_four_reports_include_missing_declared_results_without_raw_trace_reads(self):
        assets = self.root/'assets'; assets.mkdir()
        for name in ('static_jobs', 'lifecycle_jobs', 'direct_jobs', 'reactive_jobs'):
            (assets/(name+'.json')).write_text(json.dumps(dict(jobs=[])))
        self.write('static/enhanced/empty/r1/result.json', self.static())
        reports = build_reports(self.root, assets)
        self.assertEqual(set(reports), {'static_results.json', 'performance.json', 'lifecycle.json', 'regressions.json'})
        self.assertEqual(len(reports['static_results.json']['attempts']), 1)
        json.dumps(reports, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
