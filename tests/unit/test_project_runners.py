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

    def run_unit(self, codes=None, artifacts=False):
        args = ['--unit-only']
        if artifacts:
            args += ['--artifacts-dir', str(self.path/'logs')]
        with patch.object(self.all.subprocess, 'run', side_effect=[result(c) for c in codes] if codes else None,
                          return_value=result()) as run:
            code = self.all.main(args)
        return code, [call.args[0] for call in run.call_args_list]

    def test_unit_only_never_dispatches_through_isaac_or_changes_environment(self):
        before = os.environ.copy()
        code, commands = self.run_unit()
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2+len(self.all.UNIT_SCRIPTS))
        self.assertTrue(all(cmd[0] == sys.executable for cmd in commands))
        self.assertTrue(any('unittest' in cmd for cmd in commands))
        self.assertEqual(dict(os.environ), before)

    def test_child_failure_remains_failure_and_other_checks_run(self):
        code, commands = self.run_unit([0, 7]+[0]*len(self.all.UNIT_SCRIPTS), artifacts=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 2+len(self.all.UNIT_SCRIPTS))
        summary = json.loads(next((self.path/'logs').rglob('summary.json')).read_text())
        self.assertEqual(summary['unit_discovery']['return_code'], 7)
        self.assertEqual(summary['unit_discovery']['status'], 'FAIL')
        self.assertEqual(summary[self.all.UNIT_SCRIPTS[-1]]['status'], 'PASS')

    def test_full_dispatch_covers_all_retained_isaac_and_regression_scenarios(self):
        cases = list(self.all.cases('all', self.path, True))
        for group in ('isaac', 'regression'):
            selected = [(name, cmd, path) for name, cmd, path in cases if name.startswith(group+'_')]
            self.assertEqual(len(selected), len(list((ROOT/'tests'/group).glob('test_*.py'))))
            for name, cmd, path in selected:
                self.assertEqual(cmd[:3], ['bash', str(self.all.RUNNER), str(ROOT/'tests'/group/(name[len(group)+1:]+'.py'))])
                self.assertNotIn('-c', cmd)
                self.assertIn('--result', cmd)
                self.assertIn('--artifacts-dir', cmd)
                self.assertNotIn('--no-ros-env', cmd)

    def test_missing_result_or_reported_failure_cannot_pass_on_exit_zero(self):
        for payload in (None, {'status': 'FAIL'}, {'status': 'PASS'}):
            with self.subTest(payload=payload):
                def child(command, **kwargs):
                    if payload is not None:
                        Path(command[command.index('--result')+1]).write_text(json.dumps(payload))
                    return result()
                with patch.object(self.all.subprocess, 'run', side_effect=child):
                    self.assertEqual(self.all.main(['--group', 'isaac']), 0 if payload == {'status': 'PASS'} else 1)

    def test_failed_launcher_is_reported(self):
        with patch.object(self.all.subprocess, 'run', side_effect=OSError('cannot execute')):
            self.assertEqual(self.all.main(['--group', 'regression']), 1)
        self.assertIn('cannot execute', self.stdout.getvalue())

    def test_native_crash_is_not_hidden_by_pass_json_and_remaining_scenarios_run(self):
        for crash in (-11, 139):  # Native signal vs a shell's encoded exit status.
            with self.subTest(crash=crash):
                codes = iter((crash, 0))
                def child(command, **kwargs):
                    Path(command[command.index('--result')+1]).write_text('{"status":"PASS"}')
                    return result(next(codes))
                logs = self.path/f'crash {crash}'
                with patch.object(self.all.subprocess, 'run', side_effect=child) as run:
                    self.assertEqual(self.all.main(['--group', 'isaac', '--artifacts-dir', str(logs)]), 1)
                self.assertEqual(run.call_count, 2)
                summary = json.loads(next(logs.rglob('summary.json')).read_text())
                first, second = summary.values()
                self.assertEqual(first['status'], 'FAIL')
                self.assertEqual(first['return_code'], crash)
                self.assertEqual(second['status'], 'PASS')

    def test_scripts_import_without_numpy_or_isaac(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'isaacsim', 'robot_skill_stack'}:
                raise AssertionError(f'Parent imported {name}')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            load_script('run_all_regressions')
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

    def sandbox(self):
        repo, isaac = self.path / 'repo with spaces', self.path / 'fake isaac'
        def write(path, text):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name in ('run_all_regressions', 'replay_perception_capture'):
            write(repo / 'scripts' / f'{name}.py', (ROOT / 'scripts' / f'{name}.py').read_text())
        write(repo / 'run_isaac_python.sh', (ROOT / 'run_isaac_python.sh').read_text())
        (repo / 'run_isaac_python.sh').chmod(0o600)  # ZIP extraction need not retain +x.
        write(isaac / 'python.sh', '#!/bin/bash\nexport VIA_TEST_WRAPPER=yes\n'
              f'exec {shlex.quote(sys.executable)} "$@"\n')
        (isaac / 'python.sh').chmod(0o755)
        write(repo / '.deps' / 'numpy' / '__init__.py',
              "import os\nassert os.environ.get('VIA_TEST_WRAPPER') == 'yes'\n__version__ = 'test-only'\n")
        return repo, {**os.environ, 'ISAAC_SIM': str(isaac)}

    def test_real_unit_only_works_with_unavailable_isaac_from_other_cwd(self):
        repo, env = self.sandbox()
        env['ISAAC_SIM'] = str(self.path/'no Isaac installation')
        (repo/'scripts/check_architecture.py').write_text("print('architecture sentinel')\n")
        (repo/'tests/unit').mkdir(parents=True)
        for path in ('tests/__init__.py', 'tests/unit/__init__.py'):
            (repo/path).touch()
        for name in self.all.UNIT_SCRIPTS:
            (repo/'tests/unit'/f'{name}.py').write_text("print('standalone sentinel')\n")
        (repo/'tests/unit/test_sentinel.py').write_text(
            "import os, unittest\nclass Offline(unittest.TestCase):\n"
            "    def test_no_wrapper(self):\n"
            "        self.assertNotIn('VIA_TEST_WRAPPER', os.environ)\n")
        process = subprocess.run([sys.executable, str(repo/'scripts/run_all_regressions.py'), '--unit-only'],
                                 cwd=self.path, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertIn('Ran 1 test', process.stderr)
        self.assertIn('Overall: PASS', process.stdout)

    def test_real_isaac_script_dispatch_preserves_main_arguments_and_fresh_results(self):
        repo, env = self.sandbox()
        numpy = repo/'.deps/numpy/__init__.py'
        numpy.write_text(numpy.read_text().replace('test-only', '1.26.4'))
        (repo/'tests/isaac').mkdir(parents=True)
        for path in ('tests/__init__.py', 'tests/isaac/__init__.py'):
            (repo/path).touch()
        prelude = (ROOT/'tests/isaac/test_perception_discovery.py').read_text().split('from isaacsim import SimulationApp')[0]
        (repo/'tests/isaac/test_sentinel.py').write_text(
            prelude + "import sys, json, argparse\nfrom pathlib import Path\n"
            "assert vars(sys.modules['__main__']) is globals()\n"
            "assert __spec__ is None\n"
            "assert Path(sys.argv[0]) == Path(__file__)\n"
            "assert 'numpy' in sys.modules\n"
            "p=argparse.ArgumentParser(); p.add_argument('--result'); p.add_argument('--artifacts-dir')\n"
            "args=p.parse_args(); assert 'retained logs' in args.result\n"
            "assert Path(args.artifacts_dir).name == 'isaac_test_sentinel'\n"
            "Path(args.result).write_text(json.dumps({'status':'PASS'}))\n")
        args = [sys.executable, str(repo/'scripts/run_all_regressions.py'), '--group', 'isaac',
                '--artifacts-dir', str(self.path/'retained logs')]
        for _ in range(2):
            process = subprocess.run(args, cwd=self.path, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        runs = list((self.path/'retained logs').iterdir())
        self.assertEqual(len(runs), 2)
        for directory in runs:
            self.assertIn('Test NumPy: 1.26.4', (directory/'isaac_test_sentinel.log').read_text())
            self.assertEqual(json.loads((directory/'summary.json').read_text())['isaac_test_sentinel']['status'], 'PASS')

    def test_all_eight_scripts_preload_numpy_before_simulation_app(self):
        repo, env = self.sandbox()
        numpy = repo/'.deps/numpy/__init__.py'
        numpy.write_text(numpy.read_text().replace('test-only', '1.26.4'))
        # Stop at construction; this verifies the real scripts' startup context,
        # not native SDK loading, simulation, or scenario behavior.
        (repo/'.deps/isaacsim.py').write_text(
            "import sys, json\nfrom pathlib import Path\n"
            "assert 'numpy' in sys.modules\n"
            "assert sys.modules['numpy'].__version__ == '1.26.4'\n"
            "class SimulationApp:\n"
            "    def __init__(self, config):\n"
            "        main = sys.modules['__main__']\n"
            "        assert main.__spec__ is None\n"
            "        assert Path(main.__file__) == Path(sys.argv[0])\n"
            "        assert config == {'headless': False}\n"
            "        print(json.dumps(sys.argv))\n"
            "        raise SystemExit(23)\n")
        scripts = [p for folder in ('isaac', 'regression') for p in sorted((ROOT/'tests'/folder).glob('test_*.py'))]
        self.assertEqual(len(scripts), 8)
        for source in scripts:
            with self.subTest(script=source.name):
                target = repo/source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
                args = ['--result', str(self.path/'result with spaces.json'),
                        '--artifacts-dir', str(self.path/'scenario artifacts')]
                process = subprocess.run(['bash', str(repo/'run_isaac_python.sh'), str(target), *args],
                                         cwd=self.path, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(process.returncode, 23, process.stdout+process.stderr)
                self.assertIn('Test NumPy: 1.26.4', process.stdout)
                self.assertEqual(json.loads(process.stdout.splitlines()[-1]), [str(target), *args])

    def test_wrong_numpy_is_rejected_before_importing_sdk(self):
        repo, env = self.sandbox()
        target = repo/'tests/isaac/test_startup.py'
        target.parent.mkdir(parents=True)
        target.write_text((ROOT/'tests/isaac/test_perception_discovery.py').read_text())
        (repo/'.deps/isaacsim.py').write_text("raise AssertionError('SDK imported too early')\n")
        process = subprocess.run(['bash', str(repo/'run_isaac_python.sh'), str(target)],
                                 cwd=self.path, env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Expected NumPy 1.26.4, got test-only', process.stderr)
        self.assertNotIn('SDK imported too early', process.stderr)

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

    def test_gui_shell_dispatch_preserves_arguments_dependencies_and_exit_status(self):
        repo, env = self.sandbox()
        (repo / 'run_isaac_sim.sh').write_text((ROOT / 'run_isaac_sim.sh').read_text())
        (repo / 'run_isaac_python.sh').chmod(0o755)
        # The fake Isaac launcher executes this sentinel, not SimulationApp.
        # An unsupported option before the script makes Python reject dispatch.
        (repo / 'scripts/launch_isaac_gui.py').write_text(
            "import json, sys\nimport numpy as np\n"
            "print(json.dumps({'args': sys.argv[1:], 'numpy': np.__version__}))\n"
            "raise SystemExit(17)\n")
        args = ['--/test/value=two words', '--test-flag']
        process = subprocess.run(['bash', str(repo / 'run_isaac_sim.sh'), *args],
                                 cwd=self.path, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 17, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout), {'args': args, 'numpy': 'test-only'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
