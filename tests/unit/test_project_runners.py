from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(code=0):
    return subprocess.CompletedProcess([], code)


class ProjectRunnerTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_script('run_primitive_perception_suite')
        self.replay = load_script('replay_perception_capture')
        self.all = load_script('run_all_regressions')
        self.temp = tempfile.TemporaryDirectory(prefix='runner test ')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.capture = self.path / 'capture with spaces.npz'
        self.capture.touch()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.stdout))
        self.stack.enter_context(contextlib.redirect_stderr(self.stderr))

    def test_every_suite_child_uses_wrapper_without_ros_option(self):
        with patch.object(self.suite.subprocess, 'run', return_value=result()) as run:
            self.assertEqual(self.suite.main(), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ['bash', str(self.suite.RUNNER), '-c', self.suite.ENV_CHECK])
        self.assertEqual(commands[1:], [
            ['bash', str(self.suite.RUNNER), '-m', f'tests.unit.{name}']
            for name in self.suite.TESTS
        ])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs, {'cwd': ROOT})
            self.assertNotIn('--no-ros-env', call.args[0])
            self.assertNotIn(sys.executable, call.args[0])

    def test_suite_does_not_change_parent_pythonpath(self):
        with patch.dict(os.environ, {'PYTHONPATH': 'keep-this-path', 'ISAAC_SIM': '/custom/isaac'}):
            before = os.environ.copy()
            with patch.object(self.suite.subprocess, 'run', return_value=result()):
                self.assertEqual(self.suite.main(), 0)
            self.assertEqual(dict(os.environ), before)

    def test_environment_failure_stops_before_tests(self):
        with patch.object(self.suite.subprocess, 'run', return_value=result(7)) as run:
            self.assertEqual(self.suite.main(), 7)
            self.assertEqual(run.call_count, 1)
        self.assertIn('no tests were run', self.stderr.getvalue())
        self.assertNotIn('SUITE PASS', self.stdout.getvalue())

    def test_test_failure_is_propagated_and_stops_suite(self):
        with patch.object(self.suite.subprocess, 'run', side_effect=[result(), result(9)]) as run:
            self.assertEqual(self.suite.main(), 9)
            self.assertEqual(run.call_count, 2)
        self.assertNotIn('SUITE PASS', self.stdout.getvalue())

    def test_missing_wrapper_does_not_fall_back_to_host_python(self):
        with patch.object(self.suite, 'RUNNER', self.path / 'missing.sh'):
            with patch.object(self.suite.subprocess, 'run') as run:
                self.assertEqual(self.suite.main(), 2)
                run.assert_not_called()
        self.assertIn('wrapper not found', self.stderr.getvalue())

    def test_suite_shell_error_is_reported(self):
        with patch.object(self.suite.subprocess, 'run', side_effect=OSError('no bash')):
            self.assertEqual(self.suite.main(), 2)
        self.assertIn('Could not launch project Python', self.stderr.getvalue())

    def test_scripts_import_without_numpy_or_isaac(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'isaacsim', 'robot_skill_stack'}:
                raise AssertionError(f'Parent imported {name}')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            load_script('run_primitive_perception_suite')
            load_script('replay_perception_capture')

    def test_replay_preserves_capture_argument_and_failure_status(self):
        with patch.object(sys, 'argv', ['replay', str(self.capture)]):
            with patch.object(self.replay.subprocess, 'run', return_value=result(13)) as run:
                self.assertEqual(self.replay.main(), 13)
        run.assert_called_once_with(
            ['bash', str(self.replay.RUNNER), '-c', self.replay.REPLAY, str(self.capture)], cwd=ROOT,
        )

    def test_replay_help_needs_no_wrapper(self):
        with patch.object(sys, 'argv', ['replay', '--help']):
            with patch.object(self.replay, 'RUNNER', self.path / 'missing.sh'):
                with patch.object(self.replay.subprocess, 'run') as run:
                    with self.assertRaises(SystemExit) as caught:
                        self.replay.main()
                    self.assertEqual(caught.exception.code, 0)
                    run.assert_not_called()

    def test_replay_missing_capture_is_reported(self):
        with patch.object(sys, 'argv', ['replay', str(self.path / 'missing.npz')]):
            with patch.object(self.replay.subprocess, 'run') as run:
                with self.assertRaises(SystemExit) as caught:
                    self.replay.main()
                self.assertEqual(caught.exception.code, 2)
                run.assert_not_called()

    def test_replay_missing_wrapper_is_reported(self):
        with patch.object(sys, 'argv', ['replay', str(self.capture)]):
            with patch.object(self.replay, 'RUNNER', self.path / 'missing.sh'):
                with self.assertRaises(SystemExit) as caught:
                    self.replay.main()
                self.assertEqual(caught.exception.code, 2)
        self.assertIn('wrapper not found', self.stderr.getvalue())

    def test_replay_shell_error_is_reported(self):
        with patch.object(sys, 'argv', ['replay', str(self.capture)]):
            with patch.object(self.replay.subprocess, 'run', side_effect=OSError('no bash')):
                self.assertEqual(self.replay.main(), 2)

    def orchestrate(self, unit_only, codes):
        argv = ['all'] + (['--unit-only'] if unit_only else [])
        with patch.object(sys, 'argv', argv), patch.object(self.all, 'OUT_ROOT', self.path):
            with patch.object(self.all.subprocess, 'run', side_effect=[result(c) for c in codes]) as run:
                code = self.all.main()
        summary = json.loads(next(self.path.rglob('summary.json')).read_text())
        return code, summary, [call.args[0] for call in run.call_args_list]

    def test_unit_only_preserves_outer_orchestration(self):
        code, summary, calls = self.orchestrate(True, [0, 0])
        self.assertEqual(code, 0)
        self.assertEqual(list(summary['steps']), ['architecture', 'primitives'])
        self.assertEqual(calls, [cmd for name, cmd in self.all.STEPS if name in summary['steps']])

    def test_full_suite_still_dispatches_all_existing_suites(self):
        code, summary, calls = self.orchestrate(False, [0] * len(self.all.STEPS))
        self.assertEqual(code, 0)
        self.assertEqual(list(summary['steps']), [name for name, _ in self.all.STEPS])
        self.assertEqual(calls, [cmd for _, cmd in self.all.STEPS])

    def test_outer_orchestrator_reports_child_failure(self):
        code, summary, _ = self.orchestrate(True, [0, 9])
        self.assertEqual(code, 1)
        self.assertEqual(summary['status'], 'FAIL')
        self.assertEqual(summary['steps']['primitives']['return_code'], 9)

    def sandbox(self):
        repo, isaac = self.path / 'repo with spaces', self.path / 'fake isaac'
        def write(path, text):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name in ('run_primitive_perception_suite', 'replay_perception_capture'):
            write(repo / 'scripts' / f'{name}.py', (ROOT / 'scripts' / f'{name}.py').read_text())
        write(repo / 'run_isaac_python.sh', (ROOT / 'run_isaac_python.sh').read_text())
        (repo / 'run_isaac_python.sh').chmod(0o600)  # ZIP extraction need not retain +x.
        write(isaac / 'python.sh', '#!/bin/bash\nexport VIA_TEST_WRAPPER=yes\n'
              f'exec {shlex.quote(sys.executable)} "$@"\n')
        (isaac / 'python.sh').chmod(0o755)
        write(repo / '.deps' / 'numpy' / '__init__.py',
              "import os\nassert os.environ.get('VIA_TEST_WRAPPER') == 'yes'\n__version__ = 'test-only'\n")
        return repo, {**os.environ, 'ISAAC_SIM': str(isaac)}

    def test_real_shell_dispatch_from_other_cwd_with_spaces(self):
        repo, env = self.sandbox()
        (repo / 'tests' / 'unit').mkdir(parents=True)
        for path in ('tests/__init__.py', 'tests/unit/__init__.py'):
            (repo / path).touch()
        for name in self.suite.TESTS:
            (repo / 'tests' / 'unit' / f'{name}.py').write_text(
                "import numpy as np\nassert np.__version__ == 'test-only'\n")
        process = subprocess.run([sys.executable, str(repo / 'scripts/run_primitive_perception_suite.py')],
                                 cwd=self.path, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('SUITE PASS', process.stdout)
        self.assertIn(str(repo / '.deps/numpy/__init__.py'), process.stdout)

    def test_real_shell_replay_preserves_relative_path_from_other_cwd(self):
        repo, env = self.sandbox()
        package = repo / 'robot_skill_stack/world/perception'
        package.mkdir(parents=True)
        for folder in (package, package.parent, package.parent.parent):
            (folder / '__init__.py').touch()
        (package / 'diagnostics.py').write_text(
            "import numpy as np\ndef replay_capture(path):\n"
            "    return {'capture': path, 'numpy': np.__version__}\n")
        process = subprocess.run([sys.executable, str(repo / 'scripts/replay_perception_capture.py'),
                                  self.capture.name], cwd=self.path, env=env,
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout), {'capture': str(self.capture), 'numpy': 'test-only'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
