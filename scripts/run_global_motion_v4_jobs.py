#!/usr/bin/env python3
"""Finite V4 native queue using the existing serialized, bounded job runner.

Only this benchmark's script and ignored output tree are allowed. The original
V3 runner remains unchanged; its launcher, timeout and process-group handling
are reused. No runtime or installed SDK is patched by this outer driver.
"""
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner


NATIVE = "scripts/benchmark_global_motion_v4.py"
OUTPUT_ROOT = Path("outputs/motion-planning-v4")


def main(argv=None):
    with patch.object(runner, "NATIVE", NATIVE), patch.object(runner, "OUTPUT_ROOT", OUTPUT_ROOT):
        return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
