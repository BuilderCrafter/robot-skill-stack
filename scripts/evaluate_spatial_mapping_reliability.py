#!/usr/bin/env python3
"""Replay captured V4 evidence against V3 and current occupancy policies.

This is an SDK-free comparison on identical rays, masks, original timestamps and
integration times. It measures occupancy policy, not live source freshness or
native update latency. Native fixtures own the independent geometry reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from robot_skill_stack.world.spatial_mapping import occupancy as current
from tests.support.spatial_mapping_reliability import ReferenceShape, score_voxel_centers, score_surface_events


ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'b9afb8e38b285fcace554ea0998e14ddc2e46467'


def baseline_module():
    path = 'robot_skill_stack/world/spatial_mapping/occupancy.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT)
    module = types.ModuleType('v3_reliability_occupancy')
    sys.modules[module.__name__] = module
    exec(compile(source, f'{BASELINE}:{path}', 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def histogram(states):
    return {state.name.lower(): int((states == state).sum()) for state in current.OccupancyState}


def surface_inputs(directory, record, max_depth=5.):
    points, labels, cameras = [], [], []
    for item in record['cameras']:
        with np.load(directory/item['file'], allow_pickle=False) as frame:
            y, x = frame['selected_y'], frame['selected_x']
            retained = ~frame['automatic_mask'][y, x]
            y, x = y[retained], x[retained]
            depth, k, matrix = frame['depth'][y, x], frame['intrinsics'], frame['world_from_camera']
            if not np.isfinite(depth).all() or np.any(depth <= 0) or np.any(depth > max_depth):
                raise ValueError('Captured retained selections contain invalid or over-range depth')
            rays = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))]
            points.append((rays*depth[:, None]) @ matrix[:3, :3].T+matrix[:3, 3])
            paths = json.loads(str(frame['instance_paths_json']))
            labels.extend(paths.get(str(int(value)), 'unlabeled') for value in frame['instance_ids'][y, x])
            cameras.extend([item['camera_id']]*len(x))
    return np.concatenate(points), labels, cameras


def evaluate(directory):
    directory = Path(directory).resolve()
    raw_manifest = (directory/'capture_manifest.json').read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest['status'] != 'PASS':
        raise ValueError('A complete PASS native capture is required')
    records = manifest.get('records', manifest.get('captures'))
    if not records or any('evidence_file' not in row for row in records):
        raise ValueError('This capture predates exact integrate-evidence recording')
    baseline, source_sha = baseline_module()
    settings = manifest['settings']
    clock = [0.]
    maps = {name: module.OccupancyMap(module.GridSpec(settings['bounds_min'], settings['bounds_max'],
        settings['voxel_size']), clock=lambda: clock[0], stale_after=settings['stale_after'],
        evidence_ttl=settings['evidence_ttl']) for name, module in [('baseline', baseline), ('current', current)]}
    rows, native_checks, input_hashes = [], [], {}
    for record in records:
        for filename in [record['evidence_file'], *(camera['file'] for camera in record['cameras']),
                         *([record['snapshot_file']] if record.get('snapshot_file') else [])]:
            input_hashes[filename] = hashlib.sha256((directory/filename).read_bytes()).hexdigest()
        if record.get('clear_map_before'):
            for occupancy in maps.values():
                occupancy.clear()
        with np.load(directory/record['evidence_file'], allow_pickle=False) as saved:
            evidence = {key: saved[key] for key in saved.files}
        clock[0] = float(evidence['now'])
        occupied = evidence['occupied_indices']
        centers = maps['current'].spec.indices_to_centers(occupied)
        before = {key: value.query_many(centers, now=clock[0]) for key, value in maps.items()}
        for occupancy in maps.values():
            assert occupancy.integrate(evidence['free_indices'], occupied,
                timestamp=float(evidence['timestamp']), now=clock[0])
        snapshot_now = float(evidence.get('snapshot_now', record.get('snapshot_now', clock[0])))
        clock[0] = snapshot_now
        snapshots = {key: occupancy.snapshot(now=snapshot_now) for key, occupancy in maps.items()}
        after = {key: snapshot.query_many(centers) for key, snapshot in snapshots.items()}
        cohort = before['baseline'].states == current.OccupancyState.FREE
        still_free = cohort & (after['baseline'].states == current.OccupancyState.FREE)
        row = dict(index=record['index'], episode=record.get('episode'), phase=record.get('phase'),
            original_timestamp=float(evidence['timestamp']), integration_now=float(evidence['now']),
            direct_hit_cells=len(occupied), baseline_previously_free_hit_cells=int(cohort.sum()),
            baseline_still_free_after_hit=int(still_free.sum()),
            current_states_on_baseline_still_free_hits=histogram(after['current'].states[still_free]),
            policies={})
        points, labels, cameras = surface_inputs(directory, record, settings.get('max_depth', 5.))
        for name, snapshot in snapshots.items():
            scored = dict(counts=snapshot.counts(),
                hit_states=histogram(after[name].states),
                selected_retained_surface_events=score_surface_events(snapshot, points, labels, camera_ids=cameras))
            if 'evaluated_indices' in evidence:
                exposures = record.get('exposure_shapes_by_camera', {})
                groups = list(exposures.values()) if exposures else [record.get('shapes', [])]
                groups = [tuple(ReferenceShape.from_payload(shape) for shape in group) for group in groups]
                scored['physical_center_metrics'] = score_voxel_centers(snapshot, groups[0],
                    evidence['evaluated_indices'], other_exposures=groups[1:],
                    excluded_indices=evidence.get('excluded_indices', ()))
            row['policies'][name] = scored
        if record.get('snapshot_file'):
            with np.load(directory/record['snapshot_file'], allow_pickle=False) as native:
                fields = ('states', 'probabilities', 'observed', 'last_observed', 'expired')
                check = {key: bool(np.array_equal(getattr(snapshots['baseline'], key), native[key], equal_nan=True))
                    for key in fields}
                for key, expected in dict(timestamp=snapshots['baseline'].timestamp,
                        bounds_min=snapshots['baseline'].spec.bounds_min,
                        bounds_max=snapshots['baseline'].spec.bounds_max,
                        voxel_size=snapshots['baseline'].spec.voxel_size).items():
                    if key in native:
                        check[key] = bool(np.array_equal(np.asarray(expected), native[key]))
            native_checks.append(dict(index=record['index'], fields_equal=check))
        rows.append(row)
    if native_checks and not all(all(row['fields_equal'].values()) for row in native_checks):
        raise AssertionError(f'V3 replay differs from original native capture: {native_checks}')
    return dict(schema_version=1, baseline=BASELINE, baseline_occupancy_sha256=source_sha,
        current_occupancy_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        native_capture=str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(), settings=settings,
        input_sha256=input_hashes,
        scope='Identical captured HIT/MISS indices and original integration timeline; policy comparison only.',
        timing_claims=False, timestamps_rewritten=False, native_baseline_snapshot_checks=native_checks,
        denominator='Direct-hit comparisons count unique cell/batch events, not obstacle volume or independent cells.',
        baseline_still_free_hit_events=sum(row['baseline_still_free_after_hit'] for row in rows),
        current_still_free_on_same_events=sum(row['current_states_on_baseline_still_free_hits']['free'] for row in rows),
        measurements=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items()
                     if key not in ('measurements', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
