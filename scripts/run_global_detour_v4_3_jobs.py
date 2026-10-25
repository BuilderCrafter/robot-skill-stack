#!/usr/bin/env python3
"""Serialize finite V4.3 native jobs with the unchanged launcher and runner."""
import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    choices = dict(detour='scripts/benchmark_global_detour_v4_3.py',
        previous_roundtrip='scripts/benchmark_global_roundtrip_v4_2.py',
        previous_static='scripts/benchmark_global_models_v4_2.py',
        previous_lifecycle='scripts/benchmark_global_lifecycle_v4_2.py')
    parser.add_argument('--suite', choices=tuple(choices), default='detour')
    args, remaining = parser.parse_known_args(argv)
    with patch.object(runner, 'NATIVE', choices[args.suite]), \
            patch.object(runner, 'OUTPUT_ROOT', Path('outputs/motion-planning-v4-3')):
        return runner.main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
