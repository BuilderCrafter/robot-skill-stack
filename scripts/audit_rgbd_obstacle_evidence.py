#!/usr/bin/env python3
"""Reduce diagnostic mapping captures using physical geometry for evaluation.

No simulator imports, camera acquisition, motion commands or algorithm changes.
The independent fixture boxes label evidence only after its runtime processing.
Full worker captures are forensic evidence and cannot establish latency claims.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
import tomllib

import numpy as np

from robot_skill_stack.world.spatial_mapping.obstacles import (
    ObstacleExtractionSettings, ObstacleExtractor, _cuboids, _supported)
from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancySnapshot, OccupancyState


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return dict(count=0)
    return dict(count=len(values), minimum=float(np.min(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def _rays(frame, y, x):
    k, transform = frame.intrinsics, frame.world_from_camera
    directions = np.column_stack(((x+.5-k[0, 2])/k[0, 0],
                                  (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))))
    return directions @ transform[:3, :3].T


def _physical_hits(frame, y, x, box, tolerance=.015):
    directions = _rays(frame, y, x)
    origin = frame.world_from_camera[:3, 3]
    lower = np.asarray(box['center'])-np.asarray(box['dimensions'])/2
    upper = np.asarray(box['center'])+np.asarray(box['dimensions'])/2
    moving = np.abs(directions) > 1e-15
    first, second = np.full_like(directions, -np.inf), np.full_like(directions, np.inf)
    np.divide(lower-origin, directions, out=first, where=moving)
    np.divide(upper-origin, directions, out=second, where=moving)
    near, far = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
    parallel_outside = (~moving & ((origin < lower) | (origin > upper))).any(1)
    potential = ~parallel_outside & (near > 0) & (far >= near)
    measured = frame.depth[y, x]
    valid = np.isfinite(measured) & (measured > 0)
    visible = potential & valid & (np.abs(measured-near) <= tolerance)
    return visible, dict(potential=int(potential.sum()), visible=int(visible.sum()),
        occluded=int(np.count_nonzero(potential & valid & (measured < near-tolerance))),
        behind_expected_surface=int(np.count_nonzero(potential & valid & (measured > near+tolerance))),
        invalid=int(np.count_nonzero(potential & ~valid)))


def _roi(spec, box):
    lower = np.asarray(box['center'])-np.asarray(box['dimensions'])/2
    upper = np.asarray(box['center'])+np.asarray(box['dimensions'])/2
    first = np.maximum(0, np.floor((lower-spec.bounds_min)/spec.voxel_size).astype(int))
    stop = np.minimum(spec.shape, np.ceil((upper-spec.bounds_min)/spec.voxel_size-1e-10).astype(int))
    cells = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(first, stop)), indexing='ij'), -1).reshape(-1, 3)
    return first, stop, cells, np.ravel_multi_index(cells.T, spec.shape)


def _cell_intersection(bounds, first, stop):
    if not len(bounds):
        return np.empty(0, int)
    bounds = np.asarray(bounds)
    lower = np.maximum(bounds[:, [0, 2, 4]], first)
    upper = np.minimum(bounds[:, [1, 3, 5]], stop)
    return np.prod(np.maximum(0, upper-lower), axis=1)


def _neighbor_spacing(cells, voxel_size):
    if len(cells) < 2:
        return dict(voxels=len(cells), nearest_chebyshev_voxels=dict(count=0), nearest_euclidean_m=dict(count=0))
    chebyshev, squared = [], []
    for begin in range(0, len(cells), 128):
        delta = np.abs(cells[begin:begin+128, None, :]-cells[None, :, :]).astype(float)
        infinity = np.arange(begin, min(begin+128, len(cells)))
        delta[np.arange(len(infinity)), infinity] = np.inf
        chebyshev.extend(np.max(delta, axis=2).min(1))
        squared.extend(np.sum(delta*delta, axis=2).min(1))
    return dict(voxels=len(cells), nearest_chebyshev_voxels=_distribution(chebyshev),
        nearest_euclidean_m=_distribution(np.sqrt(squared)*voxel_size),
        nearest_neighbor_beyond_support_radius_2=int(np.count_nonzero(np.asarray(chebyshev) > 2)))


def audit_capture(folder):
    folder = Path(folder)
    payload = json.loads((folder/'result.json').read_text())
    metrics = payload['metrics']
    profile = tomllib.loads((folder/'runtime_profile.toml').read_text())
    cfg = profile['spatial_mapping']
    support_plane = profile.get('scene', {}).get('support_plane_z', 0.)
    spec = GridSpec(tuple(cfg['bounds_min']), tuple(cfg['bounds_max']), cfg['voxel_size'])
    objects = metrics['configuration']['objects']
    if any(box.get('moving') for box in objects):
        raise ValueError('Moving fixtures require exposure-matched physical poses; this reducer handles fixed fixtures')
    placement = metrics.get('fixture_placement_started_at')
    measurement = metrics.get('measurement_started_at')
    audit = folder/'mapping-audit'
    jobs = [json.loads(line) for line in (audit/'jobs.jsonl').read_text().splitlines()]
    rois = {box['name']: _roi(spec, box) for box in objects}
    fixture_mask = np.zeros(spec.shape, bool)
    for first, stop, _, _ in rois.values():
        fixture_mask[tuple(slice(a, b) for a, b in zip(first, stop))] = True
    floor = np.minimum(spec.bounds_min[2]+(np.arange(spec.shape[2])+1)*spec.voxel_size,
        spec.bounds_max[2]) <= support_plane+spec.voxel_size+1e-12
    previous = {name: dict(states=np.zeros(len(region[3]), np.uint8),
        probabilities=np.full(len(region[3]), .5), observed=np.zeros(len(region[3]), bool)) for name, region in rois.items()}
    cumulative = {name: dict(hits=np.zeros(len(region[3]), int), free=np.zeros(len(region[3]), int)) for name, region in rois.items()}
    reduced = []
    for job in jobs:
        if not job.get('input_pickle') or not job.get('arrays_npz'):
            continue
        with (audit/job['input_pickle']).open('rb') as stream:
            inputs = pickle.load(stream)
        with np.load(audit/job['arrays_npz']) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = job['snapshot']
        now = float(meta['timestamp'])+float(meta['health']['age_seconds'])
        snapshot = OccupancySnapshot(spec, arrays['states'], arrays['probabilities'], arrays['observed'],
            arrays['last_observed'], arrays['expired'], meta['timestamp'], meta['revision'], meta['generation'], meta['health'])
        eligible = ((snapshot.states == OccupancyState.OCCUPIED) & snapshot.observed & ~snapshot.expired &
            np.isfinite(snapshot.last_observed) & (snapshot.last_observed <= now) &
            (now-snapshot.last_observed < cfg.get('evidence_ttl', 5.)))
        supported = _supported(eligible)
        bounds, dropped = _cuboids(supported)
        packet = ObstacleExtractor(ObstacleExtractionSettings(64, support_plane)).extract(
            snapshot, now=now, evidence_ttl=cfg.get('evidence_ttl', 5.))
        selected_bounds = np.asarray([list(map(int, box.id.split(':'))) for box in packet.boxes], int).reshape(-1, 6)
        flat_hit = np.ravel_multi_index(arrays.get('integrate_occupied_indices', np.empty((0, 3), int)).T, spec.shape)
        flat_free = np.ravel_multi_index(arrays.get('integrate_free_indices', np.empty((0, 3), int)).T, spec.shape)
        timestamp = job['input_timestamp']
        phase = ('before_actor_insertion' if placement is not None and timestamp < placement else
                 'measured_route' if measurement is not None and timestamp >= measurement else
                 'after_insertion_before_route')
        row = dict(job=job['job'], source_timestamp=timestamp, phase=phase,
            accepted_by_processor=job['accepted'], map_revision=meta['revision'],
            map_state_counts={name: int(np.count_nonzero(snapshot.states == value)) for name, value in
                (('unknown', OccupancyState.UNKNOWN), ('free', OccupancyState.FREE), ('occupied', OccupancyState.OCCUPIED))},
            extractor=dict(packet.diagnostics), objects={}, cameras=[])
        for index, frame in enumerate(inputs.frames):
            h, w = frame.depth.shape
            y, x = np.indices((h, w)); y, x = y.ravel(), x.ravel()
            sy, sx = np.indices(frame.depth[::cfg['stride'], ::cfg['stride']].shape)
            sy, sx = sy.ravel()*cfg['stride'], sx.ravel()*cfg['stride']
            valid = np.isfinite(frame.depth[sy, sx]) & (frame.depth[sy, sx] > 0) & (frame.depth[sy, sx] <= cfg.get('max_depth', 5.))
            manual = (inputs.exclusion_masks or {}).get(frame.camera_id)
            if manual is not None:
                valid &= ~manual[sy, sx]
            cy, cx = sy[valid], sx[valid]
            ay, ax = arrays[f'camera_{index}_selected_y'], arrays[f'camera_{index}_selected_x']
            automatic = arrays[f'camera_{index}_automatic_mask'][ay, ax]
            camera = dict(camera_id=frame.camera_id, source_timestamp=frame.acquired_at,
                valid_depth_pixels=int(np.count_nonzero(np.isfinite(frame.depth) & (frame.depth > 0))),
                stride_pixels=len(sy), candidate_rays=len(cy), allocated_rays=len(ay),
                automatic_masked_rays=int(automatic.sum()), retained_rays=int((~automatic).sum()), objects={})
            for box in objects:
                raw_hits, raw = _physical_hits(frame, y, x, box)
                _, stride = _physical_hits(frame, sy, sx, box)
                _, candidate = _physical_hits(frame, cy, cx, box)
                allocated_hit, allocated = _physical_hits(frame, ay, ax, box)
                camera['objects'][box['name']] = dict(raw_depth=raw, stride_grid=stride,
                    candidates=candidate, allocated=allocated,
                    physically_matching_rays_masked=int(np.count_nonzero(allocated_hit & automatic)),
                    physically_matching_rays_retained=int(np.count_nonzero(allocated_hit & ~automatic)))
            row['cameras'].append(camera)
        for box in objects:
            name = box['name']; first, stop, cells, flat = rois[name]
            inside_hit = np.isin(flat, flat_hit, assume_unique=True)
            inside_free = np.isin(flat, flat_free, assume_unique=True)
            states = snapshot.states.ravel()[flat]
            probabilities = snapshot.probabilities.ravel()[flat]
            observed = snapshot.observed.ravel()[flat]
            if phase != 'before_actor_insertion':
                cumulative[name]['hits'] += inside_hit
                cumulative[name]['free'] += inside_free
            actual_support = supported.ravel()[flat]
            represented = _cell_intersection(selected_bounds, first, stop)
            candidate_volume = _cell_intersection(bounds, first, stop)
            before = previous[name]
            row['objects'][name] = dict(physical_fixture_present=phase != 'before_actor_insertion',
                physical_aabb_intersecting_voxels=len(flat), current_endpoint_hit_cells=int(inside_hit.sum()),
                current_free_traversal_cells=int(inside_free.sum()),
                current_hits_with_previous_free_prior=int(np.count_nonzero(inside_hit & before['observed'] & (before['probabilities'] <= .35))),
                current_hits_previous_free_to_unknown=int(np.count_nonzero(inside_hit & before['observed'] & (before['probabilities'] <= .35) & (states == OccupancyState.UNKNOWN))),
                current_hit_cells_now_occupied=int(np.count_nonzero(inside_hit & (states == OccupancyState.OCCUPIED))),
                occupied_cells=int(np.count_nonzero(eligible.ravel()[flat])),
                supported_cells=int(actual_support.sum()),
                rejected_support_cells=int(np.count_nonzero(eligible.ravel()[flat]))-int(actual_support.sum()),
                candidate_boxes_intersecting_fixture=int(np.count_nonzero(candidate_volume)),
                candidate_represented_cells=int(candidate_volume.sum()),
                selected_boxes_intersecting_fixture=int(np.count_nonzero(represented)),
                selected_represented_cells=int(represented.sum()),
                supported_cells_omitted_by_budget=int(actual_support.sum())-int(represented.sum()),
                observed_probability_distribution=_distribution(probabilities[observed]),
                occupied_probability_distribution=_distribution(probabilities[eligible.ravel()[flat]]),
                cumulative_observed_endpoint_cells=int(np.count_nonzero(cumulative[name]['hits'])),
                cumulative_hit_votes=int(cumulative[name]['hits'].sum()),
                cumulative_free_votes=int(cumulative[name]['free'].sum()),
                nearest_neighbor_spacing=_neighbor_spacing(cells[eligible.ravel()[flat]], spec.voxel_size))
            previous[name] = dict(states=states.copy(), probabilities=probabilities.copy(), observed=observed.copy())
        above = eligible & ~floor[None, None, :]
        outside = ~fixture_mask if phase != 'before_actor_insertion' else np.ones(spec.shape, bool)
        row['evidence_classes'] = dict(occupied_floor_cells=int(np.count_nonzero(eligible & floor[None, None, :])),
            occupied_above_plane_cells=int(np.count_nonzero(above)),
            occupied_above_plane_outside_fixture_aabbs=int(np.count_nonzero(above & outside)),
            supported_above_plane_outside_fixture_aabbs=int(np.count_nonzero(supported & ~floor[None, None, :] & outside)))
        row['confidence_retention_counterfactuals'] = {}
        for threshold in (.9, .95, .98):
            strong = eligible & (snapshot.probabilities >= threshold)
            rescue = strong & ~supported
            trial_bounds, _ = _cuboids(supported | strong)
            if len(trial_bounds):
                sizes = trial_bounds[:, [1, 3, 5]]-trial_bounds[:, [0, 2, 4]]
                volumes = np.prod(sizes, axis=1)
                trial_floor = (spec.bounds_min[2]+trial_bounds[:, 5]*spec.voxel_size <= support_plane+spec.voxel_size+1e-12)
                order = np.lexsort((trial_bounds[:, 4], trial_bounds[:, 2], trial_bounds[:, 0], -volumes, trial_floor))[:64]
                selected = trial_bounds[order]
            else:
                trial_floor = np.empty(0, bool); order = np.empty(0, int); selected = trial_bounds
            trial = dict(rescued_cells=int(rescue.sum()), rescued_floor_cells=int(np.count_nonzero(rescue & floor[None, None, :])),
                rescued_above_plane_cells=int(np.count_nonzero(rescue & ~floor[None, None, :])),
                rescued_above_plane_outside_fixture_aabbs=int(np.count_nonzero(rescue & ~floor[None, None, :] & outside)),
                candidate_boxes=len(trial_bounds), selected_proxies=len(selected),
                selected_floor_proxies=int(trial_floor[order].sum()),
                selected_above_plane_proxies=int(len(order)-trial_floor[order].sum()), objects={})
            for name, (first, stop, _, flat) in rois.items():
                coverage = _cell_intersection(selected, first, stop)
                trial['objects'][name] = dict(rescued_observed_cells=int(rescue.ravel()[flat].sum()),
                    selected_boxes_intersecting_fixture=int(np.count_nonzero(coverage)),
                    selected_represented_cells=int(coverage.sum()))
            row['confidence_retention_counterfactuals'][str(threshold)] = trial
        reduced.append(row)
    return dict(schema_version=1, status='FORENSIC_BASELINE', capture_dir=str(folder),
        artifacts=dict(result_sha256=_sha(folder/'result.json'), jobs_jsonl_sha256=_sha(audit/'jobs.jsonl'),
            reducer_sha256=_sha(__file__)), configuration=metrics['configuration'],
        fixture_placement_started_at=placement, measurement_started_at=measurement,
        capture_jobs=len(jobs), analyzed_jobs=len(reduced),
        method='Original worker inputs, actual selected pixels/masks/integrated indices and exact post-update arrays. Physical fixture boxes label evidence offline only. Existing V2 radius2/exact-cell/64-budget extractor is replayed without algorithm changes.',
        qualifications=[
            'Full capture worker writes perturb publication and queue latency; no latency or availability performance claim uses these jobs.',
            'AABB-intersecting voxel counts quantify source representation near fixture geometry, not complete physical reconstruction or certified clearance.',
            'Raw-depth physical matching uses the unchanged V2 optical-depth tolerance0.015m. Physically matching rays are separated from occluded/missing returns.',
            'Actor insertion starts at a recorded owner time; frames immediately crossing that boundary may precede completion of the physical pose relocation.',
            'Post-update map fields are actual captured state; support/candidate/selection are exact offline replays of the existing adapter, including for Direct whose runtime product is disabled.'],
        jobs=reduced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    if np.__version__ != '1.26.4':
        raise RuntimeError('Run through ./run_isaac_python.sh for NumPy1.26.4')
    started = time.monotonic()
    result = audit_capture(args.capture_dir)
    result['offline_analysis_seconds'] = time.monotonic()-started
    path = Path(args.result); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], jobs=result['analyzed_jobs'],
        analysis_seconds=result['offline_analysis_seconds'], result=str(path))))


if __name__ == '__main__':
    main()
