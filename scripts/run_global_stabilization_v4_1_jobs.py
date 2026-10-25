#!/usr/bin/env python3
"""Serialize bounded V4.1 reproductions using the existing native job runner."""
from pathlib import Path
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner

NATIVE = "scripts/diagnose_global_stabilization_v4_1.py"
OUTPUT_ROOT = Path("outputs/motion-planning-v4-1")


def main(argv=None):
    with patch.object(runner, "NATIVE", NATIVE), patch.object(runner, "OUTPUT_ROOT", OUTPUT_ROOT):
        return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
