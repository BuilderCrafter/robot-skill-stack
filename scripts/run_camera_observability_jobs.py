#!/usr/bin/env python3
"""Run a finite research job list sequentially through the unchanged SDK launcher.

Launch this driver itself with ./run_isaac_python.sh, as in the successful
camera smoke study. A successful process exit never substitutes for a valid
result JSON or working GPU-rendered cameras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.benchmark_camera_workload import FATAL_GPU_PATTERNS, source_manifest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_job(job):
    arguments = job['arguments']
    if (not isinstance(arguments, list) or not arguments or not all(isinstance(v, str) for v in arguments)
            or not arguments[0].startswith('scripts/camera_observability_') or not arguments[0].endswith('.py')):
        raise ValueError('Job must name one isolated camera_observability script and string arguments')
    folder = ROOT/Path(job['artifacts_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'driver_result.json').exists():
        raise ValueError('Refusing to overwrite an existing research job: '+str(folder))
    command = [str(ROOT/'run_isaac_python.sh'), *arguments]
    write(folder/'launch.json', dict(command=command, cwd=str(ROOT), job=job))
    start, code, timeout = time.monotonic(), None, float(job.get('timeout', 240.))
    print('START '+job['label'], flush=True)
    error = None
    with (folder/'native.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            error = f'Own research job exceeded {timeout:g}s; its process group was stopped'
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            code = child.returncode
    result_path = ROOT/Path(job['result'])
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError):
        result = dict(status='FAIL', measured=False, error='No readable result JSON')
    text = (folder/'native.log').read_text(errors='replace')
    fatal = [p for p in FATAL_GPU_PATTERNS if p in text.lower()]
    errors = [line[:1000] for line in text.splitlines() if '[error]' in line.lower() or '[fatal]' in line.lower()]
    status = result.get('status', 'FAIL')
    if fatal:
        status = 'BLOCKED'
    elif error or code != 0:
        status = 'FAIL'
    row = dict(label=job['label'], status=status, return_code=code,
        wall_seconds=time.monotonic()-start, result=str(result_path.relative_to(ROOT)),
        log=str((folder/'native.log').relative_to(ROOT)), fatal_gpu_messages=fatal,
        runtime_log_errors=dict(count=len(errors), first_lines=errors[:20]), driver_error=error,
        script_sha256=hashlib.sha256((ROOT/arguments[0]).read_bytes()).hexdigest())
    write(folder/'driver_result.json', row)
    print('END '+job['label']+': '+status, flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--continue-environment-failure', action='store_true')
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if isinstance(jobs, dict):
        jobs = jobs['jobs']
    before = source_manifest()
    summary = dict(status='RUNNING', jobs=[], production_hashes_before=before,
        job_plan_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest())
    try:
        for job in jobs:
            row = run_job(job)
            summary['jobs'].append(row)
            write(args.result, summary)
            if row['status'] == 'BLOCKED' and not args.continue_environment_failure:
                summary['remaining_unmeasured'] = [v['label'] for v in jobs[len(summary['jobs']):]]
                break
    finally:
        summary['production_hashes_after'] = source_manifest()
        summary['production_unchanged'] = before == summary['production_hashes_after']
        states = [v['status'] for v in summary['jobs']]
        summary['status'] = ('BLOCKED' if 'BLOCKED' in states else 'PASS' if states and all(v == 'PASS' for v in states)
                             and summary['production_unchanged'] else 'PARTIAL')
        write(args.result, summary)
    return 0 if summary['status'] == 'PASS' else 77 if summary['status'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
