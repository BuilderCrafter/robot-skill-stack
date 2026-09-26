from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("architecture", [sys.executable, str(ROOT / "scripts" / "check_architecture.py")]),
    ("perception_v1", [sys.executable, str(ROOT / "scripts" / "run_perception_v1_suite.py")]),
    ("phase_a", [sys.executable, str(ROOT / "scripts" / "run_phase_a_suite.py")]),
    ("phase_b", [sys.executable, str(ROOT / "scripts" / "run_phase_b_suite.py")]),
]

rc = 0
for name, command in STEPS:
    print(f"\n===== {name} =====")
    result = subprocess.run(command, cwd=ROOT)
    rc = max(rc, result.returncode)

print("\n===== ALL LOWER-LEVEL REGRESSIONS =====")
print("PASS" if rc == 0 else "FAIL")
raise SystemExit(rc)
