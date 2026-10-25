#!/usr/bin/env python3
"""Run a finite V3 simulation queue serially through the unchanged launcher.

Successful measurement and successful contact-free avoidance are different.
Never overwrite a previous attempt, weaken source freshness, or substitute a
CPU-only process for native GPU-backed motion. This driver runs outside Kit.
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
NATIVE = 'scripts/benchmark_obstacle_representation_v3.py'
OUTPUT_ROOT = Path('outputs/obstacle-representation-v3')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_job(job, *, root=ROOT):
    """Limit execution to the one reviewed wrapper and its own output tree."""
    arguments = job.get('arguments')
    if (not isinstance(arguments, list) or not arguments or arguments[0] != NATIVE
            or not all(isinstance(value, str) for value in arguments)):
        raise ValueError('V3 jobs must name the isolated benchmark script and string arguments')
    label = job.get('label')
    if not isinstance(label, str) or not label.strip():
        raise ValueError('Job requires a nonempty label')
    timeout = job.get('timeout', 360.)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 900:
        raise ValueError('Timeout must be finite and within (0,900] seconds')
    allowed = (Path(root)/OUTPUT_ROOT).resolve()
    paths = {}
    for field, flag in (('artifacts_dir', '--artifacts-dir'), ('result', '--result')):
        value = job.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError('Missing '+field)
        candidate = (Path(root)/value).resolve()
        if candidate == allowed or not candidate.is_relative_to(allowed):
            raise ValueError(field+' must be below '+str(OUTPUT_ROOT))
        if arguments.count(flag) != 1:
            raise ValueError('Exactly one '+flag+' is required')
        index = arguments.index(flag)
        if index+1 == len(arguments) or (Path(root)/arguments[index+1]).resolve() != candidate:
            raise ValueError(flag+' must match the declared output path')
        paths[field] = candidate
    if not paths['result'].is_relative_to(paths['artifacts_dir']):
        raise ValueError('Result must belong to its own case directory')
    if '--no-ros-env' in arguments:
        raise ValueError('Launcher environment bypass is not permitted')
    return paths


def run_job(job):
    paths = validate_job(job)
    folder = paths['artifacts_dir']
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/name).exists() for name in ('launch.json', 'native.log', 'driver_result.json')) or paths['result'].exists():
        raise ValueError('Refusing to overwrite a previous V3 attempt: '+str(folder))
    command = [str(ROOT/'run_isaac_python.sh'), *job['arguments']]
    write_json(folder/'launch.json', dict(command=command, cwd=str(ROOT), job=job))
    print('START '+job['label'], flush=True)
    started, error = time.monotonic(), None
    with (folder/'native.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            child.wait(timeout=float(job.get('timeout', 360.)))
        except subprocess.TimeoutExpired:
            error = 'Own V3 subprocess exceeded its declared timeout'
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                # Stop only this fresh process group and its mapping workers.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    try:
        result = json.loads(paths['result'].read_text())
    except (OSError, ValueError):
        result = dict(status='FAIL', measured=False, error='No readable result JSON')
    log_text = (folder/'native.log').read_text(errors='replace')
    gpu_failures = [pattern for pattern in FATAL_GPU_PATTERNS if pattern in log_text.lower()]
    status = result.get('status', 'FAIL')
    if gpu_failures:
        status = 'BLOCKED'
    elif error or child.returncode != 0:
        status = 'FAIL'
    row = dict(label=job['label'], status=status, return_code=child.returncode,
        wall_seconds=time.monotonic()-started, result=str(paths['result'].relative_to(ROOT)),
        log=str((folder/'native.log').relative_to(ROOT)), driver_error=error,
        fatal_gpu_messages=gpu_failures,
        wrapper_sha256=hashlib.sha256((ROOT/NATIVE).read_bytes()).hexdigest())
    write_json(folder/'driver_result.json', row)
    print('END '+job['label']+': '+str(status), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--continue-invalid', action='store_true',
        help='Retain independent invalid cases and continue; GPU failures always stop')
    args = parser.parse_args(argv)
    document = json.loads(args.jobs.read_text())
    jobs = document['jobs'] if isinstance(document, dict) else document
    if not isinstance(jobs, list) or not jobs:
        parser.error('A nonempty finite job list is required')
    for job in jobs:
        validate_job(job)
    folders = [str(validate_job(job)['artifacts_dir']) for job in jobs]
    if len(folders) != len(set(folders)):
        parser.error('Each job requires its own output directory')
    before = source_manifest()
    summary = dict(status='RUNNING', jobs=[], production_hashes_before=before,
        job_plan_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest())
    try:
        for job in jobs:
            row = run_job(job)
            summary['jobs'].append(row)
            write_json(args.result, summary)
            if row['status'] == 'BLOCKED' or (row['status'] != 'PASS' and not args.continue_invalid):
                summary['remaining_unmeasured'] = [item['label'] for item in jobs[len(summary['jobs']):]]
                break
    finally:
        summary['production_hashes_after'] = source_manifest()
        summary['production_unchanged'] = before == summary['production_hashes_after']
        states = [row['status'] for row in summary['jobs']]
        summary['status'] = ('BLOCKED' if 'BLOCKED' in states else 'PASS' if len(states) == len(jobs)
            and all(value == 'PASS' for value in states) and summary['production_unchanged'] else 'PARTIAL')
        write_json(args.result, summary)
    return 0 if summary['status'] == 'PASS' else 77 if summary['status'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
