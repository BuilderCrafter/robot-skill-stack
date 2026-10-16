"""One entry point for offline units and workstation-only Isaac scenarios."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'run_isaac_python.sh'
UNIT_SCRIPTS = (
    'test_async_runtime', 'test_grasp_planning', 'test_perception', 'test_placement',
    'test_primitive_perception', 'test_reactive_pick', 'test_world_model_update_rate',
    'test_world_model_view',
)
# The wrapper supplies .deps; preload the pinned NumPy before test modules import
# SimulationApp, matching the GUI launch contract without changing those scripts.
ISAAC_ENTRY = """import sys
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
import runpy
module = sys.argv.pop(1)
sys.argv[0] = module
runpy.run_module(module, run_name='__main__')
"""


def cases(group, run_dir, artifacts):
    if group in ('all', 'unit'):
        yield 'architecture', [sys.executable, str(ROOT/'scripts/check_architecture.py')], None
        yield 'unit_discovery', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/unit', '-t', '.'], None
        for name in UNIT_SCRIPTS:
            yield name, [sys.executable, '-m', f'tests.unit.{name}'], None
    for selected, folder in (('isaac', 'isaac'), ('regression', 'regression')):
        if group not in ('all', selected):
            continue
        for path in sorted((ROOT/'tests'/folder).glob('test_*.py')):
            name = f'{selected}_{path.stem}'
            result = run_dir/f'{name}.json'
            command = ['bash', str(RUNNER), '-c', ISAAC_ENTRY,
                       f'tests.{folder}.{path.stem}', '--result', str(result)]
            if artifacts:
                command += ['--artifacts-dir', str(run_dir/name)]
            yield name, command, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--group', choices=('all', 'unit', 'isaac', 'regression'), default='all')
    selection.add_argument('--unit-only', dest='group', action='store_const', const='unit',
                           help='Run offline tests with this Python; never invoke Isaac')
    parser.add_argument('--artifacts-dir', type=Path, help='Retain logs, result JSON and scenario artifacts in a fresh subdirectory')
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='robot-regressions-') as temporary:
        run_dir = Path(temporary)
        if args.artifacts_dir is not None:
            args.artifacts_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=args.artifacts_dir.expanduser().resolve()))
        summary = {}
        for name, command, result in cases(args.group, run_dir, args.artifacts_dir is not None):
            print(f'== {name} ==', flush=True)
            log_path = run_dir/f'{name}.log'
            error, code = None, 1
            try:
                with (log_path.open('w') if args.artifacts_dir is not None else nullcontext(None)) as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log,
                                             stderr=subprocess.STDOUT if log is not None else None)
                code = process.returncode
                if code == 0 and result is not None:
                    payload = json.loads(result.read_text())
                    if not isinstance(payload, dict) or payload.get('status') != 'PASS':
                        error = f'Scenario did not report PASS: {result}'
            except (OSError, ValueError) as exc:
                error = str(exc)
            status = 'PASS' if code == 0 and error is None else 'FAIL'
            summary[name] = dict(status=status, return_code=code, error=error)
            print(f'{name}: {status} (exit {code})'+(' '+error if error else ''), flush=True)
        passed = bool(summary) and all(row['status'] == 'PASS' for row in summary.values())
        print('Overall:', 'PASS' if passed else 'FAIL', flush=True)
        if args.artifacts_dir is not None:
            (run_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print('Artifacts:', run_dir, flush=True)
        return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
