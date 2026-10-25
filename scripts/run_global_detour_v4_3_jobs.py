#!/usr/bin/env python3
"""Serialize finite V4.3 native jobs with the unchanged launcher and runner."""
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner


def main(argv=None):
    with patch.object(runner, 'NATIVE', 'scripts/benchmark_global_detour_v4_3.py'), \
            patch.object(runner, 'OUTPUT_ROOT', Path('outputs/motion-planning-v4-3')):
        return runner.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
