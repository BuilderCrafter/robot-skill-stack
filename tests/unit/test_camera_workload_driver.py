"""Workflow checks prevent CPU fallback, stale evidence, and ungated workload growth."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_camera_workload as driver


def native_result(label, resolution, mode=None, failed=False):
    metric = dict(count=100, p50=10., p95=12., p99=15., worst=20., mean=10.)
    summary = {key: dict(metric) for key in (
        'kit_update_ms', 'control_interval_ms', 'physics_interval_ms',
        'physics_step_ms', 'mapping_compute_ms', 'mapping_queue_ms',
        'mapping_publish_age_ms')}
    summary.update(semantic_hz=5., mapping_hz=2., motion_complete=True,
                   frame_checks=True, mapping_source_age_ms=dict(count=60,
                   p50=200., p95=300., p99=400., worst=450.))
    return dict(label=label, mode=mode or label, resolution=resolution,
                status='FAIL' if failed else 'PASS', errors=['injected measurement failure'] if failed else [],
                summary=summary)


class CameraWorkloadDriverTests(unittest.TestCase):
    def arguments(self, directory, **changes):
        values = dict(artifacts_dir=Path(directory), seconds=30., warmup=10.,
                      case_timeout=240., include_d=True, try_640=True, headless=True)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_blocked_probe_does_not_launch_timed_or_cpu_fallback_cases(self):
        calls = []

        def runner(args, label, resolution, mode=None):
            calls.append((label, resolution))
            return dict(status='BLOCKED', summary={}, errors=['No GPU'])

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, 'source_manifest', return_value={}), \
                patch.object(driver, 'provenance', return_value={}):
            result = driver.execute(self.arguments(directory), runner)
            self.assertEqual(calls, [('probe', '320x240')])
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertTrue(all(row['status'] == 'NOT_RUN' and not row['summary']
                                for row in result['results']))
            saved = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(saved['status'], 'BLOCKED')

    def test_failed_b_still_measures_c_and_repeat_but_gates_d_and_high_resolution(self):
        calls = []

        def runner(args, label, resolution, mode=None):
            calls.append((label, resolution))
            return native_result(label, resolution, mode, failed=label == 'B')

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, 'source_manifest', return_value={}), \
                patch.object(driver, 'provenance', return_value={}):
            result = driver.execute(self.arguments(directory), runner)
            self.assertEqual(calls, [('probe', '320x240'), ('A', '320x240'),
                ('B', '320x240'), ('C', '320x240'), ('A_repeat', '320x240')])
            self.assertEqual(result['status'], 'FAIL')
            self.assertTrue(all(row['status'] == 'NOT_RUN' for row in result['results']
                                if row['resolution'] == '640x480' or row['label'] == 'D'))

    def test_baseline_drift_prevents_high_resolution_even_when_abc_pass(self):
        calls = []

        def runner(args, label, resolution, mode=None):
            calls.append((label, resolution))
            result = native_result(label, resolution, mode)
            if label == 'A_repeat':
                result['summary']['control_interval_ms']['p99'] = 40.
            return result

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, 'source_manifest', return_value={}), \
                patch.object(driver, 'provenance', return_value={}):
            result = driver.execute(self.arguments(directory), runner)
            self.assertEqual(result['status'], 'FAIL')
            self.assertTrue(all(resolution == '320x240' for _, resolution in calls))
            self.assertEqual(result['comparisons']['320x240']['baseline_drift']['status'], 'FAIL')

    def test_successful_exit_with_missing_gpu_log_is_blocked_and_discards_timings(self):
        def popen(command, *, stdout, **kwargs):
            destination = Path(command[command.index('--result')+1])
            destination.write_text(json.dumps(native_result('A', '320x240')))
            stdout.write('CUDA error: no CUDA-capable device is detected\n')
            stdout.flush()
            return SimpleNamespace(wait=lambda timeout: 0)

        with tempfile.TemporaryDirectory() as directory, patch.object(driver.subprocess,
                'Popen', side_effect=popen):
            result = driver.run_case(self.arguments(directory), 'A', '320x240')
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual(result['summary'], {})
            self.assertFalse(result['measured'])
            self.assertTrue(result['fatal_gpu_messages'])

    def test_native_error_log_invalidates_success_and_retains_bounded_messages(self):
        def popen(command, *, stdout, **kwargs):
            destination = Path(command[command.index('--result')+1])
            destination.write_text(json.dumps(native_result('A', '320x240')))
            for index in range(25):
                stdout.write(f'[Error] [test.plugin] runtime error {index}\n')
            stdout.flush()
            return SimpleNamespace(wait=lambda timeout: 0)

        with tempfile.TemporaryDirectory() as directory, patch.object(driver.subprocess,
                'Popen', side_effect=popen):
            result = driver.run_case(self.arguments(directory), 'A', '320x240')
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['runtime_log_errors']['count'], 25)
            self.assertEqual(len(result['runtime_log_errors']['first_lines']), 20)
            self.assertTrue(result['errors'])
            self.assertTrue(result['summary'])


if __name__ == '__main__':
    unittest.main()
