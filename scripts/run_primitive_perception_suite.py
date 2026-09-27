#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    'test_primitive_depth_pipeline', 'test_perception_lifecycle', 'test_primitive_perception',
    'test_perception', 'test_grasp_planning', 'test_async_runtime', 'test_placement',
    'test_reactive_pick', 'test_world_model_update_rate', 'test_world_model_view',
)


def main():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT/'.deps'), env.get('PYTHONPATH', '')))
    for test in TESTS:
        print(f'== {test} ==', flush=True)
        result = subprocess.run([sys.executable, '-m', f'tests.unit.{test}'], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print(f'PRIMITIVE PERCEPTION SUITE PASS ({len(TESTS)} simulator-independent modules)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
