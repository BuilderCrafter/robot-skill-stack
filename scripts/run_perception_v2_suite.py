#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ('test_perception_v2', 'test_perception_benchmark'):
        result = subprocess.run(['bash', str(root/'run_isaac_python.sh'), '-m', f'tests.unit.{name}'], cwd=root)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
