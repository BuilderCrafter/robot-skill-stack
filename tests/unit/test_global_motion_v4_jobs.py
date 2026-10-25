from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import run_global_motion_v4_jobs as driver


class JobDriverTests(unittest.TestCase):
    def test_outer_driver_restores_original_runner_scope(self):
        original = driver.runner.NATIVE, driver.runner.OUTPUT_ROOT
        def main(arguments):
            self.assertEqual(driver.runner.NATIVE, "scripts/benchmark_global_motion_v4.py")
            self.assertEqual(driver.runner.OUTPUT_ROOT, Path("outputs/motion-planning-v4"))
            return 0
        with patch.object(driver.runner, "main", main):
            self.assertEqual(driver.main([]), 0)
        self.assertEqual((driver.runner.NATIVE, driver.runner.OUTPUT_ROOT), original)

    def test_v4_jobs_accept_only_benchmark_with_own_outputs(self):
        folder = "outputs/motion-planning-v4/test/case"
        job = dict(label="case", artifacts_dir=folder, result=folder+"/result.json",
            arguments=[driver.NATIVE, "--strategy", "global", "--scenario", "empty",
                "--artifacts-dir", folder, "--result", folder+"/result.json"])
        with patch.object(driver.runner, "NATIVE", driver.NATIVE), patch.object(driver.runner, "OUTPUT_ROOT", driver.OUTPUT_ROOT):
            driver.runner.validate_job(job)
            bad = dict(job, arguments=["scripts/unreviewed.py", *job["arguments"][1:]])
            with self.assertRaises(ValueError):
                driver.runner.validate_job(bad)
            with self.assertRaises(ValueError):
                driver.runner.validate_job(dict(job, artifacts_dir="outputs/outside"))


if __name__ == "__main__":
    unittest.main()
