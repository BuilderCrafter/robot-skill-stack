import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_global_rgbd_v5 as driver


class GlobalRgbdV5DriverTests(unittest.TestCase):
    def test_primary_matrix_is_serial_gt_then_rgbd_and_preserves_launcher(self):
        jobs = driver.jobs_for(driver.OUTPUT_ROOT/'example')
        self.assertEqual(len(jobs),14)
        self.assertEqual([(job['scenario'],job['execution_condition']) for job in jobs],
            [(scene,source) for scene in driver.protocol.SCENES for source in driver.protocol.SOURCES])
        self.assertEqual(len({job['artifacts_dir'] for job in jobs}),14)
        for job in jobs:
            self.assertEqual(job['arguments'][0],driver.NATIVE)
            self.assertEqual(job['timeout'],600)
            self.assertEqual(job['budget'],64)
            self.assertEqual(job['arguments'][job['arguments'].index('--budget')+1],'64')
            self.assertIn('--headless',job['arguments'])
            self.assertNotIn('--no-ros-env',job['arguments'])
            self.assertTrue(job['result'].startswith(job['artifacts_dir']+'/'))

    def test_subset_order_and_diagnostic_bound_are_explicit(self):
        jobs = driver.jobs_for(driver.OUTPUT_ROOT/'subset',('v43','empty'),
            ('frozen_rgbd','static'),budget=128,timeout=900)
        self.assertEqual([job['label'] for job in jobs],
            ['empty-static-b128','empty-frozen_rgbd-b128','v43-static-b128','v43-frozen_rgbd-b128'])
        self.assertTrue(all(job['timeout']==900 for job in jobs))

    def test_invalid_output_selections_and_bounds_are_rejected(self):
        for options in (dict(scenes=()),dict(scenes=('static','static')),dict(scenes=('unknown',)),
                        dict(sources=()),dict(sources=('rgbd',)),dict(budget=65),dict(budget=True),
                        dict(timeout=0),dict(timeout=901),dict(timeout=600.)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                driver.jobs_for(driver.OUTPUT_ROOT/'invalid',**options)
        for path in ('outputs/outside',driver.OUTPUT_ROOT,'/tmp/outside-v5'):
            with self.subTest(path=path),self.assertRaises(ValueError):driver.jobs_for(path)

    def test_default_prepare_writes_protocol_and_jobs_without_native_launch(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'run_job') as run:
            folder = driver.OUTPUT_ROOT/'offline'
            self.assertEqual(driver.main(['--artifacts-dir',str(folder)]),0)
            manifest = json.loads((Path(temporary)/folder/'jobs.json').read_text())
            protocol = json.loads((Path(temporary)/folder/'protocol.json').read_text())
            self.assertEqual(len(manifest['jobs']),14)
            self.assertEqual(manifest['maximum_native_wall_seconds'],8400)
            self.assertTrue(manifest['serial_only'])
            self.assertEqual(protocol['selected_cuboid_budget'],64)
            self.assertFalse(protocol['diagnostic_condition'])
            self.assertEqual(protocol['physical_fixtures']['v43'][0]['center'],[.55,-.225,.30000000000000004])
            run.assert_not_called()
            self.assertFalse((Path(temporary)/folder/'summary.json').exists())

    def test_previous_preparation_or_unrelated_files_are_never_overwritten(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)):
            folder = driver.OUTPUT_ROOT/'old'
            absolute = Path(temporary)/folder
            absolute.mkdir(parents=True)
            previous = absolute/'unrelated.log';previous.write_text('preserve me')
            with self.assertRaises(SystemExit):driver.main(['--artifacts-dir',str(folder),'--run'])
            self.assertEqual(previous.read_text(),'preserve me')
            previous.unlink()
            self.assertEqual(driver.main(['--artifacts-dir',str(folder)]),0)
            old_plan=(absolute/'jobs.json').read_bytes()
            with self.assertRaises(SystemExit):driver.main(['--artifacts-dir',str(folder),'--run'])
            self.assertEqual((absolute/'jobs.json').read_bytes(),old_plan)

    def test_pilot_launch_uses_two_bounded_jobs_and_retains_summary(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver.runner,'run_job',return_value={'status':'PASS'}) as run:
            folder = driver.OUTPUT_ROOT/'pilot'
            code = driver.main(['--artifacts-dir',str(folder),'--case','empty','--run'])
            self.assertEqual(code,0)
            self.assertEqual(run.call_count,2)
            self.assertEqual([call.args[0]['execution_condition'] for call in run.call_args_list],
                ['static','frozen_rgbd'])
            summary = json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertEqual(summary['status'],'PASS')
            self.assertEqual(summary['remaining_unmeasured'],[])

    def test_gpu_failure_stops_even_when_continuing_other_invalid_measurements(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver.runner,'run_job',return_value={'status':'BLOCKED'}) as run:
            folder = driver.OUTPUT_ROOT/'gpu-blocked'
            code = driver.main(['--artifacts-dir',str(folder),'--case','empty','--continue-invalid','--run'])
            self.assertEqual(code,77);self.assertEqual(run.call_count,1)
            summary = json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertEqual(summary['remaining_unmeasured'],['empty-frozen_rgbd-b64'])

    def test_measurement_failure_can_continue_but_is_not_reported_as_pass(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver.runner,'run_job',side_effect=[{'status':'FAIL'},{'status':'PASS'}]) as run:
            folder = driver.OUTPUT_ROOT/'invalid'
            code = driver.main(['--artifacts-dir',str(folder),'--case','empty','--continue-invalid','--run'])
            self.assertEqual(code,1);self.assertEqual(run.call_count,2)
            summary = json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertEqual(summary['status'],'PARTIAL')
            self.assertEqual(summary['remaining_unmeasured'],[])

    def test_driver_exception_leaves_remaining_jobs_explicit(self):
        with TemporaryDirectory() as temporary,patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver.runner,'run_job',side_effect=ValueError('old result')):
            folder = driver.OUTPUT_ROOT/'error'
            self.assertEqual(driver.main(['--artifacts-dir',str(folder),'--case','empty','--run']),1)
            summary = json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertIn('old result',summary['driver_error'])
            self.assertEqual(len(summary['remaining_unmeasured']),2)


if __name__=='__main__':unittest.main()
