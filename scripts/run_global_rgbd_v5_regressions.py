#!/usr/bin/env python3
"""Prepare or serialize bounded, unchanged native V1/V2/V4 regression fixtures.

Preparation is offline by default. --run launches each selected case through
the normal Isaac launcher and existing bounded process-group runner. Results
are fixture regression outcomes, not a new physical-safety certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_obstacle_representation_v3_jobs as runner


ROOT = runner.ROOT
OUTPUT_ROOT = Path('outputs/motion-planning-v5/regressions')
FIXED_TARGET = 'docs/assets/motion_planning_v4_3/fixed_target.json'
FIXTURE = 'docs/assets/motion_planning_v4_3/frozen_fixture.json'
FIXTURE_SHA256 = 'd5c78c58343f40f559ff174ded89814493ac147d62aad1e00327657efd08680f'
DETOUR_ARGS = ('--seed','123456','--fixed-target',FIXED_TARGET,'--fixture',FIXTURE,
    '--fixture-sha256',FIXTURE_SHA256,'--headless')
MODEL_ARGS = ('--collision-model','enhanced','--seed','123456','--headless')
CASES = {
    'direct': ('tests/isaac/test_motion_strategy.py', ()),
    'reactive': ('tests/isaac/test_reactive_motion.py', ()),
    'global_roundtrip': ('scripts/benchmark_global_roundtrip_v4_2.py', ('--mode','roundtrip',*MODEL_ARGS)),
    'global_large': ('scripts/benchmark_global_models_v4_2.py', ('--scenario','large',*MODEL_ARGS)),
    **{f'global_{mode}': ('scripts/benchmark_global_lifecycle_v4_2.py', ('--mode',mode,*MODEL_ARGS))
        for mode in ('cancel','stop','reset','close')},
    'detour_free': ('scripts/benchmark_global_detour_v4_3.py', ('--condition','free',*DETOUR_ARGS)),
    'detour_obstacle': ('scripts/benchmark_global_detour_v4_3.py', ('--condition','obstacle',*DETOUR_ARGS)),
}
DEFAULT_CASES = ('direct','reactive','global_roundtrip','global_large','global_cancel',
    'detour_free','detour_obstacle')


def output_folder(path):
    path = (ROOT/Path(path)).resolve()
    allowed = (ROOT/OUTPUT_ROOT).resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('Run directory must be below '+str(OUTPUT_ROOT))
    return path


def jobs_for(path, selected):
    """Fixed commands only, with one fresh case directory and 600s bound each."""
    path = output_folder(path)
    selected = tuple(selected)
    if not selected or len(set(selected)) != len(selected) or any(key not in CASES for key in selected):
        raise ValueError('Select nonempty, unique allowlisted regression cases')
    jobs = []
    for label in selected:
        native, options = CASES[label]
        folder = str((path/label).relative_to(ROOT))
        result = folder+'/result.json'
        job = dict(label=label, timeout=600, artifacts_dir=folder, result=result,
            arguments=[native,*options,'--artifacts-dir',folder,'--result',result])
        with patch.object(runner,'NATIVE',native), patch.object(runner,'OUTPUT_ROOT',OUTPUT_ROOT):
            runner.validate_job(job)
        jobs.append(job)
    return jobs


def preflight_detour():
    """Honor the original witness/raw-source hash checks before starting Kit."""
    from scripts.benchmark_global_detour_v4_3 import load_inputs
    load_inputs(SimpleNamespace(fixed_target=str(ROOT/FIXED_TARGET),fixture=str(ROOT/FIXTURE),
        fixture_sha256=FIXTURE_SHA256,condition='obstacle',seed=123456,max_iterations=10000))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir',type=Path,help='Fresh run directory below '+str(OUTPUT_ROOT))
    parser.add_argument('--case',choices=tuple(CASES),action='append',dest='cases',
        help='Repeat to select specific cases; default is seven bounded regressions')
    parser.add_argument('--run',action='store_true',help='Execute serial GPU-backed native fixtures')
    parser.add_argument('--continue-invalid',action='store_true',
        help='Continue independent fixture failures; GPU startup failures always stop')
    args = parser.parse_args(argv)
    if args.artifacts_dir is None:
        base = ROOT/OUTPUT_ROOT
        base.mkdir(parents=True,exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='run-',dir=base))
    else:
        folder = output_folder(args.artifacts_dir)
    if any((folder/name).exists() for name in ('jobs.json','summary.json')):
        parser.error('Refusing to overwrite a prepared or executed regression run')
    jobs = jobs_for(folder,args.cases or DEFAULT_CASES)
    manifest = dict(schema_version=1, jobs=jobs, serial_only=True,
        maximum_native_wall_seconds=sum(job['timeout'] for job in jobs),
        fixture_status_is_not_physical_safety_certification=True,
        offline_command=['./run_isaac_python.sh','scripts/run_all_regressions.py','--unit-only',
            '--artifacts-dir','outputs/motion-planning-v5/unit-final'])
    runner.write_json(folder/'jobs.json',manifest)
    print('Prepared:',folder/'jobs.json',flush=True)
    if not args.run:
        print('No native process was launched. Use --run with a new run directory to execute.',flush=True)
        return 0
    before = runner.source_manifest()
    summary = dict(status='RUNNING',jobs=[],production_hashes_before=before,
        job_plan_sha256=hashlib.sha256((folder/'jobs.json').read_bytes()).hexdigest())
    try:
        if any(job['label'].startswith('detour_') for job in jobs):
            preflight_detour()
        for job in jobs:
            with patch.object(runner,'NATIVE',job['arguments'][0]), patch.object(runner,'OUTPUT_ROOT',OUTPUT_ROOT):
                row = runner.run_job(job)
            summary['jobs'].append(row)
            runner.write_json(folder/'summary.json',summary)
            if row['status']=='BLOCKED' or (row['status']!='PASS' and not args.continue_invalid):
                break
    except Exception as error:
        summary['driver_error'] = f'{type(error).__name__}: {error}'
    finally:
        summary['production_hashes_after'] = runner.source_manifest()
        summary['production_unchanged'] = before==summary['production_hashes_after']
        summary['remaining_unmeasured'] = [job['label'] for job in jobs[len(summary['jobs']):]]
        states = [row['status'] for row in summary['jobs']]
        summary['status'] = ('BLOCKED' if 'BLOCKED' in states else 'PASS'
            if len(states)==len(jobs) and all(state=='PASS' for state in states)
            and summary['production_unchanged'] and 'driver_error' not in summary else 'PARTIAL')
        runner.write_json(folder/'summary.json',summary)
    print('Overall:',summary['status'],'Artifacts:',folder,flush=True)
    return 0 if summary['status']=='PASS' else 77 if summary['status']=='BLOCKED' else 1


if __name__=='__main__':
    raise SystemExit(main())
