"""Smoke-test decisions must distinguish absent measurements from good runs."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.camera_workload_metrics import (
    compare_cases, counter_deltas, cpu_interval, cpu_snapshot, distribution,
    markdown_table)


def native_result():
    summary = {key: distribution([10., 12., 15.]) for key in
               ('kit_update_ms', 'control_interval_ms', 'physics_interval_ms',
                'physics_step_ms', 'mapping_source_age_ms', 'mapping_compute_ms',
                'mapping_queue_ms', 'mapping_publish_age_ms')}
    summary.update(semantic_hz=5., mapping_hz=2., motion_complete=True,
                   frame_checks=True, job_counts={})
    return dict(label='A', status='PASS', errors=[], summary=summary)


class CameraWorkloadMetricsTests(unittest.TestCase):
    def test_no_native_data_cannot_pass_and_markdown_does_not_invent_zeros(self):
        result = dict(label='C', status='BLOCKED', errors=['No GPU device accessible'], summary={})
        self.assertEqual(compare_cases(result, result)['status'], 'NOT_MEASURED')
        table = markdown_table([result])
        self.assertIn('— / —', table)
        self.assertNotIn('0.00', table)

    def test_healthy_case_and_end_baseline_use_same_decision(self):
        baseline = native_result()
        candidate = copy.deepcopy(baseline)
        candidate['label'] = 'A-repeat'
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'PASS')
        candidate['summary']['mapping_hz'] = 1.59
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')

    def test_baseline_resolution_identifies_group_without_claiming_third_camera(self):
        baseline = native_result()
        baseline['resolution'] = '320x240'
        repeat = copy.deepcopy(baseline)
        repeat.update(label='A_repeat', resolution='640x480')
        table = markdown_table([baseline, repeat])
        self.assertIn('| A / 320×240 group | none (two cameras) |', table)
        self.assertIn('| A_repeat / 640×480 group | none (two cameras) |', table)

    def test_timing_tolerance_has_absolute_floor_and_relative_allowance(self):
        baseline = native_result()
        baseline['summary']['kit_update_ms'] = distribution([40.])
        candidate = copy.deepcopy(baseline)
        candidate['summary']['kit_update_ms'] = distribution([50.])
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'PASS')
        candidate['summary']['kit_update_ms'] = distribution([50.001])
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')
        baseline['summary']['kit_update_ms'] = distribution([4.])
        candidate['summary']['kit_update_ms'] = distribution([9.])
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'PASS')

    def test_failure_dominates_missing_data_and_stale_maps_fail(self):
        baseline = native_result()
        candidate = native_result()
        candidate['summary']['mapping_source_age_ms'] = distribution([1000.001])
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')
        candidate['summary'] = {'frame_checks': False}
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')
        candidate['summary'] = {}
        candidate['errors'] = ['Camera failed']
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')

    def test_weak_baseline_does_not_lower_requested_rate_requirement(self):
        baseline = native_result()
        baseline['summary']['semantic_hz'] = 3.
        candidate = native_result()
        candidate['summary']['semantic_hz'] = 4.5
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')

    def test_missing_physics_or_worker_measurements_cannot_pass(self):
        baseline = native_result()
        candidate = native_result()
        candidate['summary']['physics_step_ms'] = distribution([])
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'NOT_MEASURED')

    def test_runtime_error_summary_cannot_hide_behind_success_status(self):
        baseline = native_result()
        candidate = native_result()
        candidate['summary']['runtime_errors'] = ['Mapping worker exited']
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')
        candidate['summary']['runtime_errors'] = 1
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'FAIL')
        candidate['summary']['runtime_errors'] = []
        self.assertEqual(compare_cases(baseline, candidate)['status'], 'PASS')

    def test_proc_cpu_accounting_excludes_double_counted_guests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'stat').write_text('cpu 100 10 20 300 5 3 2 0 50 5\n')
            (root/'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 600 kB\n')
            before = cpu_snapshot(root)
            self.assertEqual(before['cpu_total'], 440)
            self.assertEqual(before['available_ram_bytes'], 600*1024)
            after = dict(cpu_total=540, cpu_idle=365)
            self.assertAlmostEqual(cpu_interval(before, after), 40.)
            self.assertIsNone(cpu_interval(before, before))
            self.assertIsNone(cpu_interval(after, before))

    def test_denied_missing_proc_and_counter_reset_remain_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = cpu_snapshot(tmp)
        self.assertIsNone(missing['available_ram_bytes'])
        self.assertIsNone(cpu_interval(missing, missing))
        self.assertEqual(counter_deltas({'replaced': 3}, {'replaced': 1}), {'replaced': None})
        self.assertEqual(distribution([None, float('nan')]), {'count': 0})


if __name__ == '__main__':
    unittest.main()
