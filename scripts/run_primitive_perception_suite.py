#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_isaac_python.sh"
TESTS = (
    "test_manual_held_reset", "test_control_panel_ui",
    "test_project_runners", "test_grasp_clearance", "test_grasp_attachment",
    "test_primitive_depth_pipeline", "test_perception_lifecycle",
    "test_primitive_perception", "test_perception", "test_grasp_planning",
    "test_async_runtime", "test_placement", "test_reactive_pick",
    "test_world_model_update_rate", "test_world_model_view",
)
ENV_CHECK = """import sys
import numpy as np
print("Test Python:", sys.version.split()[0], sys.executable, flush=True)
print("Test NumPy:", np.__version__, np.__file__, flush=True)
"""


def main():
    if not RUNNER.is_file():
        print(f"Project Python wrapper not found: {RUNNER}", file=sys.stderr)
        return 2
    try:
        print("== Project Python environment ==", flush=True)
        result = subprocess.run(["bash", str(RUNNER), "-c", ENV_CHECK], cwd=ROOT)
        if result.returncode:
            print("Project Python check failed; no tests were run.", file=sys.stderr)
            return result.returncode
        for test in TESTS:
            print(f"== {test} ==", flush=True)
            # Match the existing suites: .deps belongs to the wrapper's interpreter.
            result = subprocess.run(
                ["bash", str(RUNNER), "-m", f"tests.unit.{test}"], cwd=ROOT,
            )
            if result.returncode:
                return result.returncode
    except OSError as exc:
        print(f"Could not launch project Python: {exc}", file=sys.stderr)
        return 2
    print(f"PRIMITIVE PERCEPTION SUITE PASS ({len(TESTS)} simulator-independent modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
