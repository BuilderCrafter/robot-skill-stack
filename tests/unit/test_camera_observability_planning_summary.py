"""Physical failures and missing measurements must survive result reduction."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_camera_observability_planning import summarize


def write_case(folder, strategy='reactive', *, success=False, valid=True, contact=False):
    folder.mkdir()
    result = dict(status='PASS' if valid else 'FAIL', contracts=dict(reference=True),
        research=dict(layout_id='A_current', sampling_policy='fixed', global_ray_budget=1024,
            camera_contracts=dict(calibration=True)),
        metrics=dict(configuration=dict(strategy=strategy, mapping_source='rgbd'),
            motion=dict(seconds=2., result=dict(status='success' if success else 'timeout')),
            classification='target_reached_with_contact' if contact else 'blocked_timeout',
            avoidance_success=success and not contact, clearance=dict(minimum_m=0. if contact else .04),
            contacts=dict(reported_points=8 if contact else 0, maximum_pair_impulse=1. if contact else 0.),
            contact_severity=dict(depth_m=dict(maximum=.002) if contact else {},
                normal_pair_impulse_ns=dict(maximum=1.) if contact else {}, severity_pass=None),
            obstacle_updates=dict(maximum_proxy_count=64, extraction=dict(candidate_boxes=100,
                proxy_count=64, occupied_voxels=200, supported_voxels=150, selected_floor_boxes=5))))
    (folder/'result.json').write_text(json.dumps(result))
    (folder/'launch.json').write_text(json.dumps(dict(command=['runner', '--scenario', 'static', '--strategy', strategy])))
    return folder


class CameraObservabilityPlanningSummaryTests(unittest.TestCase):
    def test_valid_contact_measurement_is_not_a_physical_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = write_case(Path(temporary)/'case', success=True, contact=True)
            summary = summarize([case])
        self.assertEqual(summary['measurement_status'], 'PASS')
        self.assertEqual(summary['physical_assessment_status'], 'PARTIAL_FAILURE')
        row = summary['cases'][0]
        self.assertTrue(row['goal_reached'])
        self.assertFalse(row['strict_contact_free_success'])
        self.assertEqual(row['maximum_observed_penetration_m'], .002)
        self.assertIsNone(row['severity_pass'])

    def test_invalid_and_missing_trials_are_retained_without_invented_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = write_case(root/'failed', success=True, valid=False)
            summary = summarize([failed, root/'missing.json'])
        self.assertEqual(summary['counts']['invalid_measurements'], 2)
        self.assertEqual(summary['measurement_status'], 'PARTIAL_FAILURE')
        self.assertTrue(all(row['strict_contact_free_success'] is None for row in summary['cases']))
        self.assertIsNone(summary['cases'][1]['maximum_observed_penetration_m'])

    def test_matched_direct_and_reactive_outcomes_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = write_case(root/'direct', 'direct', success=True, contact=True)
            reactive = write_case(root/'reactive', success=False)
            summary = summarize([direct, reactive])
        pair = summary['paired_comparisons'][0]
        self.assertEqual(pair['status'], 'MATCHED_VALID_MEASUREMENTS')
        self.assertEqual(pair['goal_reached'], dict(direct=True, reactive=False))
        self.assertIsNone(summary['cases'][0]['maximum_registered_proxies'])
        self.assertEqual(summary['cases'][1]['extraction']['unselected_candidate_boxes'], 36.)

    def test_repeated_attempts_and_explicit_preparation_trials_are_not_cherry_picked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = write_case(root/'direct', 'direct', success=True, contact=True)
            failed = write_case(root/'reactive-failed', success=False)
            success = write_case(root/'reactive-success', success=True)
            pilot = write_case(root/'pilot', success=False, valid=False)
            summary = summarize([direct, failed, success, direct], retained_trials=[pilot])
        self.assertEqual(summary['counts']['primary_attempts'], 3)
        self.assertEqual(summary['counts']['retained_trials'], 1)
        self.assertEqual(len(summary['duplicate_requested_artifacts']), 1)
        self.assertEqual(summary['paired_comparisons'][0]['status'], 'MULTIPLE_ATTEMPTS_RETAINED')
        self.assertEqual(summary['counts']['reactive_strict_successes'], 1)
        self.assertEqual(summary['physical_assessment_status'], 'PARTIAL_FAILURE')

    def test_missing_primary_evidence_prevents_a_blanket_physical_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            success = write_case(root/'reactive-success', success=True)
            summary = summarize([success, root/'missing.json'])
        self.assertEqual(summary['counts']['reactive_strict_successes'], 1)
        self.assertEqual(summary['physical_assessment_status'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
