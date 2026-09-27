#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_isaac_python.sh"
REPLAY = """import json, sys
from robot_skill_stack.world.perception.diagnostics import replay_capture
print(json.dumps(replay_capture(sys.argv[1]), indent=2))
"""


def main():
    parser = argparse.ArgumentParser(description="Replay a GUI Capture using project Python, without starting Isaac Sim.")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    capture = args.capture.expanduser().resolve()
    if not capture.is_file():
        parser.error(f"Capture not found: {capture}")
    if not RUNNER.is_file():
        parser.error(f"Project Python wrapper not found: {RUNNER}")
    try:
        return subprocess.run(
            ["bash", str(RUNNER), "-c", REPLAY, str(capture)], cwd=ROOT,
        ).returncode
    except OSError as exc:
        print(f"Could not launch project Python: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
