from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_isaac_python.sh"
OUT_ROOT = ROOT / "outputs"

TESTS = [
    ("unit", ROOT / "tests" / "unit" / "test_perception.py", False),
    ("discovery", ROOT / "tests" / "isaac" / "test_perception_discovery.py", True),
    ("manipulation", ROOT / "tests" / "isaac" / "test_perception_manipulation.py", True),
]


def git(cmd):
    try:
        return subprocess.check_output(
            ["git", *cmd],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / "perception_v1_suite" / stamp
    artifacts = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    environment = {
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--short"]),
        "runner_python": sys.version,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "PASS",
        "tests": {},
        "environment": environment,
    }

    for name, script, wants_artifacts in TESTS:
        print(f"\n=== Running {name} ===")
        result_path = run_dir / f"{name}.json"
        log_path = run_dir / f"{name}.log"
        cmd = [
            str(RUNNER),
            str(script),
            "--result",
            str(result_path),
        ]
        if wants_artifacts:
            cmd += [
                "--artifacts-dir",
                str(artifacts / name),
            ]

        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        payload = {
            "status": "FAIL",
            "error": f"process exited with {process.returncode}",
        }
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {
                    "status": "FAIL",
                    "error": f"could not read result json: {exc}",
                }

        payload["return_code"] = process.returncode
        payload["log"] = log_path.name
        summary["tests"][name] = payload
        if process.returncode != 0 or payload.get("status") != "PASS":
            summary["status"] = "FAIL"

        print(f"{name}: {payload.get('status')} (log: {log_path})")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    archive = OUT_ROOT / f"perception_v1_suite_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)

    print("\n=== PERCEPTION V1 SUITE ===")
    for name, result in summary["tests"].items():
        print(f"{name}: {result.get('status')}")
    print("Overall:", summary["status"])
    print("Summary:", summary_path)
    print("Archive:", archive)
    print("===========================")

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
