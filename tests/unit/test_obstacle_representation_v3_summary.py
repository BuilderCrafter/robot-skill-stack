"""V3 evidence never turns captures or unmatched/failed cases into live wins."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_obstacle_representation_v3 import (
    distribution, load_case, match_conditions, summarize)


def write_case(path, *, candidate='baseline', captured=False, status='PASS', duration=2.):
    path.mkdir(parents=True)
    payload = dict(status=status, contracts=dict(native=True), hashes=dict(source='same'),
        research=dict(layout_id='A_current', global_ray_budget=1024, sampling_policy='fixed',
            actual_camera_calibration=dict(camera=dict(intrinsics=[[1., 0., 0.]],
                world_from_camera_ros=[[1., 0., 0.]], resolution=[640, 480], first_acquired_at=duration)),
            camera_contracts=dict(pose=True)),
        metrics=dict(configuration=dict(strategy='reactive', mapping_source='rgbd', mapping_hz=2.,
            voxel_size=.01, start=[.45, -.25, .35], goal=[.45, .25, .35], orientation=[0., 0., 1., 0.], objects=[]),
            initial_joints=[0.]*9, backend_settings=dict(max_motion_steps=1000),
            motion=dict(seconds=duration, result=dict(status='success')),
            avoidance_success=True, contacts=dict(reported_points=0),
            path=dict(length_m=.6, straight_distance_m=.5),
            controller=dict(apply_interval_ms=dict(p99=20., worst=40.))),
        representation_v3=dict(candidate_id=candidate, scenario='empty', budget=64, packing='baseline', selection='baseline',
            activation=dict(guard=dict(candidate_active=True, expected_candidate=True, packet_candidate=True, new_original_source=True)),
            registry=dict(owner_thread_all=True), production_sources_unchanged=True,
            no_wait_for_mapping_during_motion=True, original_harness_sha256='same-harness',
            full_capture_excluded_from_timing_claims=captured))
    (path/'result.json').write_text(json.dumps(payload))
    return path/'result.json'


class ObstacleRepresentationV3SummaryTests(unittest.TestCase):
    def test_full_capture_validity_does_not_create_live_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_case(Path(temporary)/'capture', captured=True)
            row = load_case(path)
            result = summarize([path])
        self.assertTrue(row['measurement_valid'])
        self.assertTrue(row['goal_reached'])
        self.assertFalse(row['timing_eligible'])
        self.assertEqual(result['groups'][0]['timing_eligible_runs'], 0)
        self.assertEqual(result['groups'][0]['runtime_metrics']['controller_interval_p99_ms']['count'], 0)

    def test_failed_attempt_remains_and_does_not_count_as_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = write_case(Path(temporary)/'first')
            second = write_case(Path(temporary)/'failed', status='FAIL')
            result = summarize([first, second, first])
        self.assertEqual(result['counts']['primary_attempts'], 2)
        self.assertEqual(result['counts']['invalid_measurements'], 1)
        self.assertEqual(result['groups'][0]['goal_reached'], 1)
        self.assertEqual(len(result['duplicate_requested_artifacts']), 1)
        self.assertIsNone(result['cases'][1]['goal_reached'])

    def test_repeat_baselines_all_retained_and_calibration_mismatch_unproven(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = write_case(Path(temporary)/'baseline1', duration=1.)
            second = write_case(Path(temporary)/'baseline2', duration=3.)
            candidate = write_case(Path(temporary)/'candidate', candidate='coverage')
            result = summarize([first, second, candidate])
            row = load_case(candidate)
            row['source_protocol']['actual_camera_calibration']['camera']['intrinsics'][0][0] = 2.
            mismatch = match_conditions(load_case(first), row)
        self.assertEqual(len(result['paired_comparisons']), 2)
        self.assertTrue(all(row['status'] == 'MATCHED' for row in result['paired_comparisons']))
        self.assertEqual([row['deltas']['motion_seconds'] for row in result['paired_comparisons']], [1., -1.])
        self.assertFalse(mismatch['matched'])
        self.assertFalse(mismatch['checks']['actual_camera_calibration'])

    def test_missing_numbers_not_synthetic_zero(self):
        self.assertEqual(distribution([None, float('nan')]), dict(count=0))
        self.assertEqual(distribution([1., 3.])['median'], 2.)


if __name__ == '__main__':
    unittest.main()
