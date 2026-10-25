import copy
from pathlib import Path
import tempfile
import unittest

from scripts.run_obstacle_representation_v3_jobs import NATIVE, validate_job


class RepresentationJobContractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        output = 'outputs/obstacle-representation-v3/test-case'
        self.job = dict(label='test-case', artifacts_dir=output, result=output+'/result.json',
            arguments=[NATIVE, '--artifacts-dir', output, '--result', output+'/result.json'], timeout=120)

    def test_valid_case_paths_are_confined_and_canonical(self):
        paths = validate_job(self.job, root=self.root)
        self.assertEqual(paths['result'], self.root/self.job['result'])

    def test_unreviewed_script_and_non_string_arguments_rejected(self):
        for arguments in (['scripts/diagnose_reactive_motion.py'], ['bash'], [NATIVE, 5]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_job(dict(self.job, arguments=arguments), root=self.root)

    def test_path_escape_absolute_escape_and_output_root_rejected(self):
        for candidate in ('outputs/obstacle-representation-v3/../../../outside',
                          '/tmp/outside-v3', 'outputs/obstacle-representation-v3'):
            job = copy.deepcopy(self.job)
            job['artifacts_dir'] = candidate
            job['arguments'][2] = candidate
            with self.subTest(path=candidate), self.assertRaises(ValueError):
                validate_job(job, root=self.root)

    def test_result_must_belong_to_own_case_and_match_argument(self):
        job = copy.deepcopy(self.job)
        job['result'] = 'outputs/obstacle-representation-v3/other/result.json'
        for update_argument in (False, True):
            if update_argument:
                job['arguments'][-1] = job['result']
            with self.subTest(updated=update_argument), self.assertRaises(ValueError):
                validate_job(job, root=self.root)

    def test_duplicate_flags_and_environment_bypass_rejected(self):
        for suffix in (['--result', self.job['result']], ['--no-ros-env']):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_job(dict(self.job, arguments=self.job['arguments']+suffix), root=self.root)

    def test_invalid_timeouts_rejected(self):
        for timeout in (0, -1, True, float('nan'), float('inf'), 901):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                validate_job(dict(self.job, timeout=timeout), root=self.root)


if __name__ == '__main__':
    unittest.main()
