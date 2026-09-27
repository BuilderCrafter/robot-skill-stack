from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs"
STEPS = [
    ("architecture", [sys.executable, str(ROOT / "scripts" / "check_architecture.py")]),
    ("primitives", [sys.executable, str(ROOT / "scripts" / "run_primitive_perception_suite.py")]),
    ("perception_v1", [sys.executable, str(ROOT / "scripts" / "run_perception_v1_suite.py")]),
    ("phase_a", [sys.executable, str(ROOT / "scripts" / "run_phase_a_suite.py")]),
    ("phase_b", [sys.executable, str(ROOT / "scripts" / "run_phase_b_suite.py")]),
    ("course_polish", [sys.executable, str(ROOT / "scripts" / "run_course_polish_suite.py")]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true", help="Skip Isaac-only integration/regression suites")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / "lower_level_regressions" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "PASS", "steps": {}}

    for name, command in STEPS:
        if args.unit_only and name not in {"architecture", "primitives"}:
            continue
        print(f"\n===== {name} =====")
        result = subprocess.run(command, cwd=ROOT)
        summary["steps"][name] = {"return_code": result.returncode}
        if result.returncode != 0:
            summary["status"] = "FAIL"

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== ALL LOWER-LEVEL REGRESSIONS =====")
    for name, result in summary["steps"].items():
        print(f"{name}: {'PASS' if result['return_code'] == 0 else 'FAIL'}")
    print("Overall:", summary["status"])
    print("Summary:", summary_path)
    print("=======================================")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
