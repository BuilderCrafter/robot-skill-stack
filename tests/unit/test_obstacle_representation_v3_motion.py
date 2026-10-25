"""Native compaction follows declared baselines and preserves failed attempts."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.plot_obstacle_representation_v3_motion import (
    columnar_cases, configure_plot_cache, expand_case_table, label, pose_errors,
    repeat_index, representative_attempts, summarize, world_box_vertices)


def write_attempt(folder, candidate='baseline', reference=None, status='PASS'):
    folder.mkdir(parents=True)
    command = ['launcher', 'script', '--layout-id', 'A_current', '--scenario', 'static', '--strategy', 'reactive']
    if reference is not None:
        command += ['--reference-result', str(reference)]
    (folder/'launch.json').write_text(json.dumps(dict(command=command)))
    payload = dict(status=status, contracts=dict(native=True),
        research=dict(layout_id='A_current', camera_contracts=dict(calibration=True)),
        metrics=dict(configuration=dict(strategy='reactive'), motion=dict(seconds=1., result=dict(status='success')),
            avoidance_success=True, contacts=dict(reported_points=0)),
        representation_v3=dict(candidate_id=candidate, scenario='static', budget=64,
            production_sources_unchanged=True, no_wait_for_mapping_during_motion=True,
            full_capture_excluded_from_timing_claims=False, registry=dict(owner_thread_all=True),
            activation=dict(guard=dict(candidate_active=True), initial_condition_match=None if reference is None else
                dict(matched=True, checks=dict(joint_positions=True), maximum_joint_error_rad=0., joint_tolerance_rad=.01))))
    (folder/'result.json').write_text(json.dumps(payload))
    return folder/'result.json'


class ObstacleRepresentationV3MotionTests(unittest.TestCase):
    def test_one_declared_baseline_edge_does_not_cross_multiply_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = write_attempt(folder/'A-static-baseline-r1')
            second = write_attempt(folder/'A-static-baseline-r2')
            candidate = write_attempt(folder/'A-static-diversity-r2', 'diversity', reference=first)
            result = summarize(dict(primary=[first], repeats=[second, candidate]))
        self.assertEqual(result['counts']['unique_attempts'], 3)
        self.assertEqual(len(result['declared_comparisons']), 1)
        self.assertEqual(result['declared_comparisons'][0]['declared_baseline'], label(first))
        self.assertTrue(result['declared_comparisons'][0]['comparison_eligible'])
        self.assertEqual(result['cases'][2]['repeat'], 2)
        contract = result['contract_sets'][result['cases'][2]['contract_set']]
        self.assertTrue(contract['matching']['joint_positions'])

    def test_failed_attempt_and_missing_latency_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_attempt(Path(temporary)/'failed-r1', status='FAIL')
            result = summarize(dict(primary=[path]))
        self.assertEqual(result['counts']['unique_attempts'], 1)
        row = result['cases'][0]
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['goal_reached'])
        self.assertIsNone(row['strict_contact_free_success'])
        self.assertIsNone(row['observed_worker_ms'])

    def test_duplicate_artifact_never_adds_another_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_attempt(Path(temporary)/'same-r1')
            result = summarize(dict(primary=[path], repeats=[path]))
        self.assertEqual(len(result['cases']), 1)
        self.assertEqual(result['cohort_counts'], dict(primary=1))
        self.assertEqual(result['duplicate_requested_artifacts'][0]['requested_cohort'], 'repeats')

    def test_repeat_suffix_must_be_explicit(self):
        self.assertEqual([repeat_index(Path('attempt-r'+str(index))) for index in (1, 2, 3)], [1, 2, 3])
        self.assertIsNone(repeat_index(Path('old-capture')))

    def test_fixture_outline_uses_actual_translation_and_orientation(self):
        vertices, edges = world_box_vertices([.2, .4, .6], [1., 2., 3., 0., 0., math.sqrt(.5), math.sqrt(.5)])
        self.assertEqual(len(edges), 12)
        self.assertAlmostEqual(max(p[0] for p in vertices)-min(p[0] for p in vertices), .4)
        self.assertAlmostEqual(max(p[1] for p in vertices)-min(p[1] for p in vertices), .2)
        self.assertAlmostEqual(sum(p[2] for p in vertices)/8, 3.)

    def test_representative_plot_cannot_choose_between_two_primary_r1_attempts(self):
        row = dict(cohort='primary', repeat=1, layout_id='A_current', scenario='static',
            strategy='reactive', candidate='baseline')
        with self.assertRaises(ValueError):
            representative_attempts(dict(cases=[dict(row, id='first'), dict(row, id='second')]))

    def test_plot_cache_is_project_local_and_respects_explicit_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {}, clear=True), patch('scripts.plot_obstacle_representation_v3_motion.ROOT', Path(temporary)):
                chosen = configure_plot_cache()
                self.assertEqual(chosen, Path(temporary)/'outputs/obstacle-representation-v3/matplotlib-cache')
                self.assertTrue(chosen.is_dir())
            explicit = Path(temporary)/'explicit'
            with patch.dict(os.environ, dict(MPLCONFIGDIR=str(explicit))):
                self.assertEqual(configure_plot_cache(), explicit)

    def test_report_uses_request_criteria_and_never_invents_backend_tolerances(self):
        metrics = dict(backend_settings=dict(position_tolerance=.01, orientation_tolerance=.05),
            motion=dict(result=dict(details=dict(position_error_m=.0248, position_tolerance_m=.025,
                orientation_error_rad=.0098, orientation_tolerance_rad=.1))))
        errors = pose_errors(metrics)
        self.assertEqual(errors['position_tolerance_m'], .025)
        self.assertEqual(errors['orientation_tolerance_rad'], .1)
        self.assertEqual(errors['position_m'], .0248)
        metrics['motion']['result']['details'] = {}
        self.assertIsNone(pose_errors(metrics)['position_tolerance_m'])
        self.assertIsNone(pose_errors(metrics)['orientation_tolerance_rad'])

    def test_columnar_report_round_trips_null_containers_and_every_attempt(self):
        report = dict(schema_version=1, cases=[dict(id='r1', valid=True, latency=dict(count=1, p95=12.3), error=None),
            dict(id='r2', valid=False, latency=None, error='failed'),
            dict(id='r3', valid=True, latency=dict(count=0, p95=None), error=None)], counts=dict(attempts=3))
        encoded = columnar_cases(report)
        self.assertNotIn('cases', encoded)
        self.assertEqual(len(encoded['case_rows']), 3)
        decoded = expand_case_table(encoded)
        self.assertEqual(decoded['cases'], report['cases'])
        self.assertEqual(decoded['counts'], report['counts'])


if __name__ == '__main__':
    unittest.main()
