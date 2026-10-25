#!/usr/bin/env python3
"""Serialize finite V4.2 experiments through the unchanged native launcher."""
import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner

NATIVE = {
    'static': 'scripts/benchmark_global_models_v4_2.py',
    'roundtrip': 'scripts/benchmark_global_roundtrip_v4_2.py',
    'capture': 'scripts/capture_collision_postures_v4_2.py',
    'lifecycle': 'scripts/benchmark_global_lifecycle_v4_2.py',
    'direct': 'tests/isaac/test_motion_strategy.py',
    'reactive': 'tests/isaac/test_reactive_motion.py',
}
OUTPUT_ROOT = Path('outputs/motion-planning-v4-2')


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--suite', choices=tuple(NATIVE), required=True)
    args, remaining = parser.parse_known_args(argv)
    with patch.object(runner, 'NATIVE', NATIVE[args.suite]), patch.object(runner, 'OUTPUT_ROOT', OUTPUT_ROOT):
        return runner.main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
