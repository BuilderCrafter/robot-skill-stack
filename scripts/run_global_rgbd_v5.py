#!/usr/bin/env python3
"""Prepare or run the finite V5 GT/RGB-D comparison serially.

Preparation launches no native process. --run uses the unchanged Isaac Python
launcher and existing bounded process-group runner, with one fresh case folder
per measurement. Existing plans, attempts and results are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts import global_rgbd_v5_protocol as protocol
from scripts import run_obstacle_representation_v3_jobs as runner


NATIVE = 'scripts/benchmark_global_rgbd_v5.py'
OUTPUT_ROOT = Path('outputs/motion-planning-v5')


def output_folder(path):
    path = (ROOT/Path(path)).resolve()
    allowed = (ROOT/OUTPUT_ROOT).resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('Run directory must be below '+str(OUTPUT_ROOT))
    return path


def _selection(selected, allowed, kind):
    selected = tuple(allowed if selected is None else selected)
    if (not selected or len(set(selected)) != len(selected)
            or any(value not in allowed for value in selected)):
        raise ValueError(f'Select nonempty, unique declared {kind}')
    # The predeclared order is stable even if subset flags arrive reversed.
    return tuple(value for value in allowed if value in selected)


def jobs_for(path, scenes=None, sources=None, *, budget=64, timeout=600):
    """Seven scenes x two sources by default; GT precedes RGB-D per scene."""
    path = output_folder(path)
    scenes = _selection(scenes,protocol.SCENES,'scenes')
    sources = _selection(sources,protocol.SOURCES,'execution conditions')
    if type(budget) is not int or budget not in (64,128):
        raise ValueError('Use primary budget 64 or separately labeled diagnostic 128')
    if type(timeout) is not int or timeout not in (600,900):
        raise ValueError('Each native job must use its declared 600s or 900s limit')
    jobs = []
    for scene in scenes:
        for source in sources:
            label = f'{scene}-{source}-b{budget}'
            folder = str((path/scene/f'{source}-b{budget}').relative_to(ROOT))
            result = folder+'/result.json'
            job = dict(label=label,scenario=scene,execution_condition=source,budget=budget,
                timeout=timeout,artifacts_dir=folder,result=result,
                arguments=[NATIVE,'--scenario',scene,'--execution-condition',source,
                    '--budget',str(budget),'--headless','--artifacts-dir',folder,'--result',result])
            with patch.object(runner,'NATIVE',NATIVE), patch.object(runner,'OUTPUT_ROOT',OUTPUT_ROOT):
                runner.validate_job(job,root=ROOT)
            jobs.append(job)
    return jobs


def _write_new_json(path, value):
    """Claim a new run artifact, preserving every previous attempt."""
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir',type=Path,required=True,
        help='Fresh run directory below '+str(OUTPUT_ROOT))
    parser.add_argument('--case',choices=protocol.SCENES,action='append',dest='scenes',
        help='Repeat to choose a scene subset; default is all seven declared scenes')
    parser.add_argument('--execution-condition',choices=protocol.SOURCES,action='append',dest='sources',
        help='Repeat to choose sources; default is static then frozen_rgbd for each scene')
    parser.add_argument('--budget',type=int,choices=(64,128),default=64,
        help='64 primary; 128 is a separately labeled diagnostic only')
    parser.add_argument('--timeout',type=int,choices=(600,900),default=600,
        help='Wall-time bound for each fresh native subprocess, in seconds')
    parser.add_argument('--run',action='store_true',help='Launch the serial native comparison')
    parser.add_argument('--continue-invalid',action='store_true',
        help='Continue independent measurement failures; GPU startup failures always stop')
    args = parser.parse_args(argv)
    try:
        folder = output_folder(args.artifacts_dir)
        jobs = jobs_for(folder,args.scenes,args.sources,budget=args.budget,timeout=args.timeout)
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise ValueError('Refusing to overwrite a nonempty V5 run directory')
        folder.mkdir(parents=True,exist_ok=True)
    except ValueError as error:
        parser.error(str(error))
    declaration = dict(protocol.PROTOCOL,
        selected_scenes=[scene for scene in protocol.SCENES if any(job['scenario']==scene for job in jobs)],
        selected_execution_conditions=[source for source in protocol.SOURCES
            if any(job['execution_condition']==source for job in jobs)],
        selected_cuboid_budget=args.budget,
        diagnostic_condition=args.budget!=64,
        physical_fixtures={scene:protocol.fixtures(scene) for scene in protocol.SCENES
            if any(job['scenario']==scene for job in jobs)})
    _write_new_json(folder/'protocol.json',declaration)
    manifest = dict(schema_version=1,jobs=jobs,serial_only=True,
        maximum_native_wall_seconds=sum(job['timeout'] for job in jobs),
        protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
        measurement_status_is_separate_from_planner_and_physical_outcomes=True)
    _write_new_json(folder/'jobs.json',manifest)
    print('Prepared:',folder/'jobs.json',flush=True)
    if not args.run:
        print('No native process launched. Use --run with a fresh run directory to execute.',flush=True)
        return 0
    before = runner.source_manifest()
    summary = dict(status='RUNNING',jobs=[],production_hashes_before=before,
        job_plan_sha256=hashlib.sha256((folder/'jobs.json').read_bytes()).hexdigest(),
        protocol_sha256=manifest['protocol_sha256'])
    try:
        for job in jobs:
            with patch.object(runner,'NATIVE',NATIVE), patch.object(runner,'OUTPUT_ROOT',OUTPUT_ROOT):
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
