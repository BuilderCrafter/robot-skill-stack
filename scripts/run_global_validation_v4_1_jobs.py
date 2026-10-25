#!/usr/bin/env python3
"""Run finite V4.1 validation queues through the unchanged native launcher."""
import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner

NATIVE = {
    "static": "scripts/benchmark_global_motion_v4.py",
    "acceptance": "scripts/benchmark_global_acceptance_v4_1.py",
    "solver": "scripts/diagnose_global_solver_velocity_v4_1.py",
    "controlled": "scripts/validate_global_stabilization_v4_1.py",
}
OUTPUT_ROOT = Path("outputs/motion-planning-v4-1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--suite", choices=tuple(NATIVE), required=True)
    args, remaining = parser.parse_known_args(argv)
    with patch.object(runner, "NATIVE", NATIVE[args.suite]), patch.object(runner, "OUTPUT_ROOT", OUTPUT_ROOT):
        return runner.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
