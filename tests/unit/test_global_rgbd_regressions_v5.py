import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_global_rgbd_v5_regressions as driver


class GlobalRgbdRegressionDriverTests(unittest.TestCase):
    def test_manifest_retains_original_commands_bounds_and_separate_outputs(self):
        jobs=driver.jobs_for(driver.OUTPUT_ROOT/'example',driver.DEFAULT_CASES)
        self.assertEqual(len(jobs),7)
        self.assertEqual(len({job['artifacts_dir'] for job in jobs}),7)
        self.assertEqual([job['arguments'][0] for job in jobs[:2]],
            ['tests/isaac/test_motion_strategy.py','tests/isaac/test_reactive_motion.py'])
        for job in jobs:
            self.assertEqual(job['timeout'],600)
            self.assertNotIn('--no-ros-env',job['arguments'])
            self.assertTrue(job['result'].startswith(job['artifacts_dir']+'/'))
        obstacle=jobs[-1]['arguments']
        self.assertEqual(obstacle[obstacle.index('--fixture-sha256')+1],driver.FIXTURE_SHA256)

    def test_outside_outputs_unknown_and_duplicate_cases_are_rejected(self):
        for path,cases in (('outputs/outside',('direct',)),
                           (driver.OUTPUT_ROOT/'case',('arbitrary_script',)),
                           (driver.OUTPUT_ROOT/'case',('direct','direct'))):
            with self.subTest(path=path,cases=cases),self.assertRaises(ValueError):
                driver.jobs_for(path,cases)

    def test_default_preparation_never_launches_native_process_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary, patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'run_job') as run:
            folder=driver.OUTPUT_ROOT/'offline'
            self.assertEqual(driver.main(['--artifacts-dir',str(folder)]),0)
            manifest=json.loads((Path(temporary)/folder/'jobs.json').read_text())
            self.assertEqual(len(manifest['jobs']),7)
            self.assertEqual(manifest['maximum_native_wall_seconds'],4200)
            run.assert_not_called()
            with self.assertRaises(SystemExit):
                driver.main(['--artifacts-dir',str(folder)])

    def test_gpu_startup_failure_stops_queue_even_when_continue_invalid_is_selected(self):
        with TemporaryDirectory() as temporary, patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver.runner,'run_job',return_value={'status':'BLOCKED'}) as run:
            folder=driver.OUTPUT_ROOT/'gpu-blocked'
            code=driver.main(['--artifacts-dir',str(folder),'--case','direct','--case','reactive',
                '--continue-invalid','--run'])
            self.assertEqual(code,77)
            self.assertEqual(run.call_count,1)
            summary=json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertEqual(summary['status'],'BLOCKED')
            self.assertEqual(summary['remaining_unmeasured'],['reactive'])

    def test_original_detour_provenance_failure_prevents_all_native_work(self):
        with TemporaryDirectory() as temporary, patch.object(driver,'ROOT',Path(temporary)), \
                patch.object(driver.runner,'source_manifest',return_value={'source':'hash'}), \
                patch.object(driver,'preflight_detour',side_effect=ValueError('Changed witness')), \
                patch.object(driver.runner,'run_job') as run:
            folder=driver.OUTPUT_ROOT/'provenance-failed'
            self.assertEqual(driver.main(['--artifacts-dir',str(folder),'--run']),1)
            run.assert_not_called()
            summary=json.loads((Path(temporary)/folder/'summary.json').read_text())
            self.assertEqual(summary['status'],'PARTIAL')
            self.assertIn('Changed witness',summary['driver_error'])


if __name__=='__main__':unittest.main()
