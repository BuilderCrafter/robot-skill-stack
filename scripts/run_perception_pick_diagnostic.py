from __future__ import annotations

import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_isaac_python.sh"
TEST = ROOT / "scripts" / "test_perception_pick_diagnostic.py"
OUT_ROOT = ROOT / "outputs"


def git(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / "perception_pick_diagnostic" / stamp
    artifacts = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    environment = {
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--short"]),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    (run_dir / "git_diff.patch").write_text(
        git(["diff", "--no-ext-diff"]),
        encoding="utf-8",
    )

    profile = ROOT / "config" / "scenes" / "playground.toml"
    if profile.exists():
        (run_dir / "playground.toml").write_text(
            profile.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    result_path = run_dir / "diagnostic.json"
    log_path = run_dir / "diagnostic.log"
    cmd = [
        str(RUNNER),
        str(TEST),
        "--result",
        str(result_path),
        "--artifacts-dir",
        str(artifacts),
    ]

    print("Running perception pick diagnostic...")
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
                "error": f"could not read diagnostic.json: {exc}",
            }

    summary = {
        "process_return_code": process.returncode,
        "diagnostic": payload,
        "environment": environment,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    archive = OUT_ROOT / f"perception_pick_diagnostic_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)

    diagnosis = payload.get("metrics", {}).get("diagnosis", {}).get("verdict")
    print("\n=== PERCEPTION PICK DIAGNOSTIC ===")
    print("Process return code:", process.returncode)
    print("Result status:", payload.get("status"))
    print("Diagnosis:", diagnosis or "see diagnostic.json/log")
    print("Log:", log_path)
    print("Archive:", archive)
    print("==================================")
    print("\nUpload the .tar.gz archive to ChatGPT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
