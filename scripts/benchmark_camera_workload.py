#!/usr/bin/env python3
"""Run the native camera workload smoke test in fresh, sequential Kit processes.

Launch from the workstation terminal through ``./run_isaac_python.sh``. A real
RGB-D rendering probe must pass before any timed cases run. GPU telemetry is
optional; an unavailable GPU never turns into a CPU performance result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_workload_metrics import compare_cases, markdown_table


BASELINE = '8e56b163650b66e940eb39a2ab8f5684bbd08c10'
NATIVE = ROOT/'scripts/camera_workload_native.py'
FATAL_GPU_PATTERNS = (
    'no cuda-capable device is detected',
    'no suitable cuda gpu was found',
    'no usable cuda device present',
    'no valid foundation interface found',
    'failed to create any gpu devices',
    'no supported gpu(s) found',
)
BENCHMARK_FILES = (
    'scripts/benchmark_camera_workload.py', 'scripts/camera_workload_native.py',
    'scripts/camera_workload_config.py', 'scripts/camera_workload_metrics.py',
    'tests/unit/test_camera_workload_driver.py',
    'tests/unit/test_camera_workload_config.py',
    'tests/unit/test_camera_workload_metrics.py',
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_manifest():
    """Hash tracked runtime/config/scene inputs without reading environment data."""
    names = subprocess.check_output(['git', 'ls-files', '-z', '--',
        'robot_skill_stack', 'config', 'scenes', 'run_isaac_python.sh',
        'run_isaac_sim.sh'], cwd=ROOT).decode().split('\0')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in names if name and (ROOT/name).is_file()}


def benchmark_manifest():
    """Include reusable benchmark inputs even before they have been committed."""
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if (ROOT/name).is_file() else None for name in BENCHMARK_FILES}


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT,
                stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=git('rev-parse', 'HEAD'),
                branch=git('branch', '--show-current'), requested_baseline=BASELINE,
                benchmark_file_hashes=benchmark_manifest())


def not_run(label, resolution, reason):
    return dict(label=label, mode='A' if label.startswith('A') else label,
                resolution=resolution, status='NOT_RUN', summary={}, errors=[],
                reason=reason, measured=False)


def run_case(args, label, resolution, mode=None):
    mode = mode or ('A' if label.startswith('A') else label)
    folder = args.artifacts_dir/('probe' if mode == 'probe' else resolution)/label
    folder.mkdir(parents=True, exist_ok=False)
    result_path, log_path = folder/'result.json', folder/'native.log'
    command = [str(ROOT/'run_isaac_python.sh'), str(NATIVE), '--mode', mode,
        '--resolution', resolution, '--seconds', str(args.seconds),
        '--warmup', str(args.warmup), '--result', str(result_path),
        '--artifacts-dir', str(folder)]
    if args.headless:
        command.append('--headless')
    write_json(folder/'launch.json', dict(command=command, cwd=str(ROOT)))
    print(f'{label} {resolution}: starting native {mode}', flush=True)
    started = time.monotonic()
    error, code = None, None
    try:
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=args.case_timeout)
            except subprocess.TimeoutExpired:
                error = f'Native case exceeded {args.case_timeout:g} second timeout'
                # Only the process group created for this native case is stopped;
                # this also bounds its dedicated mapping/perception children.
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
    except OSError as exc:
        error = f'{type(exc).__name__}: {exc}'
    try:
        payload = json.loads(result_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError('Expected result object')
    except (OSError, ValueError) as exc:
        payload = dict(status='FAIL', summary={}, errors=[f'No valid native result: {exc}'])
    text = log_path.read_text(errors='replace') if log_path.exists() else ''
    gpu_failures = [value for value in FATAL_GPU_PATTERNS if value in text.lower()]
    # Retain only bounded diagnostic message lines, not complete logs containing
    # unrelated launch arguments or environment output. Raw logs remain local.
    runtime_errors = [line[:1000] for line in text.splitlines()
                      if re.search(r'\[(?:error|fatal)\]', line, re.IGNORECASE)]
    if code == 77 or gpu_failures or payload.get('status') == 'BLOCKED':
        payload['status'] = 'BLOCKED'
        payload['measured'] = False
        payload['summary'] = {}
        payload.setdefault('errors', []).append(
            'GPU-backed native rendering unavailable; no performance result retained')
    elif code != 0 or error:
        payload['status'] = 'FAIL'
    elif payload.get('status') not in ('PASS', 'FAIL'):
        payload['status'] = 'FAIL'
    if runtime_errors and payload['status'] != 'BLOCKED':
        payload['status'] = 'FAIL'
        payload.setdefault('errors', []).append(
            f'Native Kit log reported {len(runtime_errors)} error/fatal messages; measurement validity failed')
    if error:
        payload.setdefault('errors', []).append(error)
    payload.update(label=label, mode=mode, resolution=resolution,
                   return_code=code, driver_wall_seconds=time.monotonic()-started,
                   result_path=str(result_path), log_path=str(log_path),
                   command=command, fatal_gpu_messages=gpu_failures,
                   runtime_log_errors=dict(count=len(runtime_errors), first_lines=runtime_errors[:20],
                                           retained_limit=20, line_character_limit=1000))
    payload.setdefault('measured', mode != 'probe' and payload.get('status') != 'BLOCKED')
    write_json(folder/'driver_result.json', payload)
    print(f'{label} {resolution}: {payload.get("status", "FAIL")}', flush=True)
    return payload


def run_resolution(args, resolution, runner=run_case):
    """Always retain A/B/C measurements; expensive extensions require PASS gates."""
    results = [runner(args, label, resolution) for label in ('A', 'B', 'C')]
    baseline = results[0]
    baseline['comparison'] = compare_cases(baseline, baseline)
    comparisons = {label: compare_cases(baseline, case)
                   for label, case in zip(('B', 'C'), results[1:])}
    for case in results[1:]:
        case['comparison'] = comparisons[case['label']]
    abc_pass = all(value.get('status') == 'PASS' for value in comparisons.values())
    if args.include_d:
        if abc_pass:
            case = runner(args, 'D', resolution)
            results.append(case)
            comparisons['D'] = compare_cases(baseline, case)
            case['comparison'] = comparisons['D']
        else:
            results.append(not_run('D', resolution, 'A–C stability gates did not pass'))
    repeat = runner(args, 'A_repeat', resolution, mode='A')
    results.append(repeat)
    comparisons['baseline_drift'] = compare_cases(baseline, repeat)
    repeat['comparison'] = comparisons['baseline_drift']
    stable = abc_pass and comparisons['baseline_drift'].get('status') == 'PASS'
    if args.include_d:
        stable = stable and comparisons.get('D', {}).get('status') == 'PASS'
    return results, comparisons, stable


def execute(args, runner=run_case):
    args.artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if (args.artifacts_dir/'summary.json').exists():
        raise ValueError('Output already contains summary.json; choose a fresh artifacts directory')
    before, origin = source_manifest(), provenance()
    if 'benchmark_file_hashes' not in origin:
        origin['benchmark_file_hashes'] = benchmark_manifest()
    summary = dict(schema_version=1, status='RUNNING', provenance=origin,
        protocol=dict(seconds=args.seconds, warmup_seconds=args.warmup,
            semantic_requested_hz=5., mapping_requested_hz=2.,
            total_mapping_ray_budget=1024, direct_motion=True,
            gpu_telemetry_required=False, include_d=args.include_d,
            try_640=args.try_640, headless=args.headless,
            comparison='p99 Kit/control <= max(1.25×baseline, baseline+5ms); '
                'publication rates >=80% baseline and >=80% requested; '
                'mapping source age <=1s; motion/frame/error checks pass',
            timing_scope='Descriptive ~30s smoke windows, not a full benchmark or quality evaluation'),
        results=[], comparisons={}, source_hashes_before=before)
    try:
        probe = runner(args, 'probe', '320x240', mode='probe')
        summary['probe'] = probe
        if probe.get('status') != 'PASS':
            summary['status'] = 'BLOCKED'
            summary['reason'] = 'Native GPU-backed RGB-D probe did not pass; timed scenarios were not run'
            summary['results'] = [not_run(label, '320x240', summary['reason'])
                                  for label in ('A', 'B', 'C', 'A_repeat')]
            if args.include_d:
                summary['results'].insert(3, not_run('D', '320x240', summary['reason']))
        else:
            low, comparisons, stable = run_resolution(args, '320x240', runner)
            summary['results'].extend(low)
            summary['comparisons']['320x240'] = comparisons
            if args.try_640:
                if stable:
                    high, comparisons, high_stable = run_resolution(args, '640x480', runner)
                    summary['results'].extend(high)
                    summary['comparisons']['640x480'] = comparisons
                    stable = stable and high_stable
                else:
                    labels = ('A', 'B', 'C', 'D', 'A_repeat') if args.include_d else ('A', 'B', 'C', 'A_repeat')
                    summary['results'].extend(not_run(label, '640x480',
                        '320×240 stability or baseline-drift gates did not pass') for label in labels)
            summary['status'] = 'PASS' if stable else 'FAIL'
    except Exception as exc:
        summary['status'] = 'FAIL'
        summary.setdefault('errors', []).append(f'{type(exc).__name__}: {exc}')
    finally:
        after = source_manifest()
        benchmark_after = benchmark_manifest()
        summary['source_hashes_after'] = after
        summary['source_unchanged'] = before == after
        summary['benchmark_file_hashes_after'] = benchmark_after
        summary['benchmark_unchanged'] = origin['benchmark_file_hashes'] == benchmark_after
        if not summary['source_unchanged']:
            summary['status'] = 'FAIL'
            summary.setdefault('errors', []).append('Tracked runtime/config/scene inputs changed during experiment')
        if not summary['benchmark_unchanged']:
            summary['status'] = 'FAIL'
            summary.setdefault('errors', []).append('Reusable benchmark inputs changed during experiment')
        write_json(args.artifacts_dir/'summary.json', summary)
        report = markdown_table(summary['results'])
        report += '\n\nStatus: **'+summary['status']+'**. '
        report += summary.get('reason', 'See summary.json for comparison gates, raw measurements and baseline drift.')
        report += '\n\nNo GPU performance estimate is substituted for an unavailable native renderer.\n'
        (args.artifacts_dir/'comparison.md').write_text(report)
    print(f'Overall: {summary["status"]}\nArtifacts: {args.artifacts_dir}', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, default=ROOT/'outputs/camera-workload-smoke'/
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--warmup', type=float, default=10.)
    parser.add_argument('--case-timeout', type=float, default=240.)
    parser.add_argument('--include-d', action='store_true', help='Run semantic+mapping third camera only after A–C pass')
    parser.add_argument('--try-640', action='store_true', help='Try640×480 only after320×240 and drift gates pass')
    parser.add_argument('--headless', action='store_true', help='Use identical headless mode for all native processes')
    args = parser.parse_args(argv)
    if args.seconds < 5 or args.warmup < 1 or args.case_timeout <= args.seconds+args.warmup:
        parser.error('seconds must be >=5, warmup >=1, and timeout longer than the measurement+warmup')
    try:
        summary = execute(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if summary['status'] == 'PASS' else 77 if summary['status'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
