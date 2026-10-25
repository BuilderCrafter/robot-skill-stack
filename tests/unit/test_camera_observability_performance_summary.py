"""Repeated runs retain their variability and failures rather than pooling frames."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_camera_observability_performance import summarize


def write_case(folder, p99, *, layout='A_current', status='PASS', stall=False, source_failures=0):
    folder.mkdir()
    stat = dict(count=100, mean=p99/2, p50=p99/2, p95=p99*.8, p99=p99, worst=350. if stall else p99+2.)
    summary = {key: dict(stat) for key in ('kit_update_ms', 'physics_interval_ms', 'physics_step_ms',
        'control_interval_ms', 'mapping_source_age_ms', 'mapping_compute_ms', 'mapping_queue_ms', 'mapping_publish_age_ms')}
    channel = dict(source_failures=source_failures, worker_failures=0, rejected=0,
        worker=dict(replaced=0, discarded=0, errors=0))
    summary.update(wall_seconds=30., motion_complete=True, frame_checks=True,
        semantic_hz=5., mapping_hz=2., job_counts=dict(semantic=channel, mapping=channel))
    cameras = [dict(resolution=[640, 480]), dict(resolution=[640, 480])]
    raw = dict(status=status, errors=[], gpu_devices=['NVIDIA test GPU'], summary=summary,
        research=dict(layout_id=layout, mode='baseline', sampling_policy='fixed', global_ray_budget=1024,
            contracts=dict(calibration=True, production_sources_unchanged=True), layout=dict(cameras=cameras)))
    (folder/'result.json').write_text(json.dumps(raw))
    (folder/'native.log').write_text('normal native log\n')
    (folder/'trace.json').write_text(json.dumps(dict(events=dict(motions=[dict(command=str(i), seconds=.5, ok=True)
        for i in range(4)], cpu=[dict(available_ram_bytes=20*1024**3)]))))
    return folder


class CameraObservabilityPerformanceSummaryTests(unittest.TestCase):
    def test_interleaved_repeats_compare_run_medians_and_baseline_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for i, (base, candidate) in enumerate(zip((20., 22., 24.), (30., 32., 34.))):
                paths.extend((write_case(root/f'base{i}', base), write_case(root/f'candidate{i}', candidate, layout='B_two')))
            result = summarize(paths)
        self.assertEqual(result['measurement_status'], 'PASS')
        self.assertEqual(result['baseline']['kit_p99_ms']['median'], 22.)
        self.assertEqual(result['baseline']['kit_p99_ms']['minimum'], 20.)
        candidate = next(g for g in result['groups'] if g['layout_id'] == 'B_two')
        self.assertEqual(candidate['valid_measurements'], 3)
        self.assertEqual(candidate['metrics']['kit_update_ms']['p99']['median'], 32.)
        compare = candidate['baseline_comparison']
        self.assertAlmostEqual(compare['median_kit_p99_relative_cost_percent'], 100*(32/22-1))
        self.assertEqual(compare['cost_flag'], 'MATERIAL_KIT_P99_COST')
        self.assertFalse(compare['cost_flag_is_automatic_veto'])
        self.assertEqual(candidate['instability_status'], 'NO_INSTABILITY_OBSERVED')

    def test_failed_attempts_are_retained_and_prevent_unqualified_group_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = write_case(root/'good', 20.)
            bad = write_case(root/'bad', 80., status='FAIL')
            result = summarize([good, bad])
        self.assertEqual(result['measurement_status'], 'PARTIAL_FAILURE')
        self.assertEqual(result['counts']['attempts'], 2)
        self.assertEqual(result['groups'][0]['invalid_measurements'], 1)
        self.assertEqual(result['groups'][0]['measurement_status'], 'PARTIAL_FAILURE')
        self.assertEqual(result['groups'][0]['metrics']['kit_update_ms']['p99']['count'], 1)

    def test_missing_data_is_not_a_zero_cost_native_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = summarize([Path(temporary)/'missing.json'])
        self.assertEqual(result['measurement_status'], 'NOT_MEASURED')
        row = result['cases'][0]
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['movement_total_seconds'])
        self.assertIsNone(row['metric_distributions']['kit_update_ms'])
        self.assertEqual(result['groups'][0]['baseline_comparison']['cost_flag'], 'NOT_MEASURED')

    def test_source_errors_and_300ms_control_stalls_are_separate_from_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stall = write_case(root/'stall', 20., stall=True)
            failures = write_case(root/'failures', 20., source_failures=1)
            result = summarize([stall, failures])
        self.assertEqual(result['counts']['observed_instability'], 2)
        self.assertTrue(result['cases'][0]['instability']['control_stall_at_least_300ms'])
        self.assertTrue(result['cases'][1]['instability']['repeated_source_failures'])

    def test_cpu_fallback_and_duplicate_paths_cannot_create_repeated_native_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_case(Path(temporary)/'case', 20.)
            result_file = path/'result.json'
            raw = json.loads(result_file.read_text())
            raw['gpu_devices'] = []
            result_file.write_text(json.dumps(raw))
            result = summarize([path, result_file])
        self.assertEqual(result['counts']['attempts'], 1)
        self.assertFalse(result['cases'][0]['measurement_valid'])
        self.assertEqual(len(result['duplicate_requested_artifacts']), 1)


if __name__ == '__main__':
    unittest.main()
