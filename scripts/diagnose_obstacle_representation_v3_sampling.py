#!/usr/bin/env python3
"""Read-only actual-ray sparsity and quantization diagnostic, without Isaac.

Recorded selected pixels and masks are mandatory: this does not rerun a sampler,
filter, mapper, extractor or controller. Physical references score the already
selected rays only. Current-input clouds are never cumulative occupancy maps.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import sys
import time
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_obstacle_representation_v3 import (
    PROTECTED, load_snapshot, reference_matches, sha, write_json)
from scripts.obstacle_representation_v3 import prepare_evidence


PHASES = tuple(product((0., .5), repeat=3))
PARSER_SOURCES = ('robot_skill_stack/integrations/isaac/config.py',
    'robot_skill_stack/integrations/isaac/spatial_mapping/config.py')
BASELINE_COMMIT = '1a6d4901c52c6fec4e1eeb40d2e44eb16be5b3a8'


def _absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def effective_mapping_settings(profile_path, case_result, *, proof_directory=None):
    """Recover defaults only from the case-bound, hash-proven actual parser.

    A runtime profile can intentionally omit defaults. The capture driver's
    original before/after source manifests and the protected starting commit
    bind both the top-level parser and the leaf SpatialMappingConfig class.
    No local numeric depth default is supplied by this diagnostic.
    """
    profile_path, case_result = Path(profile_path), _absolute(case_result)
    raw = tomllib.loads(profile_path.read_text())['spatial_mapping']
    required = ('max_depth', 'max_rays', 'stride')
    if all(key in raw for key in required):
        return {key:raw[key] for key in required}, dict(mode='EXPLICIT_RECORDED_PROFILE',
            inherited_fields=[], case_result=str(case_result))
    proof_directory = Path(proof_directory or ROOT/'outputs/obstacle-representation-v3')
    baseline_path = proof_directory/'protected-baseline.json'
    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, ValueError) as exc:
        raise LookupError('Cannot prove original parser defaults: protected baseline unavailable') from exc
    if baseline.get('commit') != BASELINE_COMMIT:
        raise LookupError('Cannot prove original parser defaults: protected baseline commit mismatch')
    current = {name:sha(ROOT/name) for name in PARSER_SOURCES}
    if any(baseline.get('sha256', {}).get(name) != value for name,value in current.items()):
        raise LookupError('Cannot prove original parser defaults: current parser or leaf configuration differs from protected baseline')
    matching = []
    for name in ('pilot-driver.json', 'baseline-capture-driver.json'):
        path = proof_directory/name
        if not path.exists():
            continue
        driver = json.loads(path.read_text())
        for job in driver.get('jobs', []):
            if job.get('result') and _absolute(job['result']) == case_result:
                matching.append((path, driver, job))
    if len(matching) != 1:
        raise LookupError('Cannot prove original parser defaults: case is not uniquely bound to a recorded capture driver')
    path, driver, job = matching[0]
    if driver.get('status') != 'PASS' or job.get('status') != 'PASS' or driver.get('production_unchanged') is not True:
        raise LookupError('Cannot prove original parser defaults: capture driver/job did not pass unchanged-source guard')
    for name,value in current.items():
        if (driver.get('production_hashes_before', {}).get(name) != value or
                driver.get('production_hashes_after', {}).get(name) != value):
            raise LookupError('Cannot prove original parser defaults: actual capture parser or leaf source hash mismatch')
    # Import only after the source provenance checks. This is the unchanged
    # production TOML parser, including its real leaf-class validation/defaults.
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    actual = load_scene_config(profile_path).spatial_mapping
    values = {key:getattr(actual, key) for key in required}
    return values, dict(mode='VERIFIED_ORIGINAL_PRODUCTION_PARSER_DEFAULTS',
        inherited_fields=[key for key in required if key not in raw],
        baseline_commit=baseline['commit'], protected_baseline_sha256=sha(baseline_path),
        parser_and_leaf_sha256=current, capture_driver=str(path), capture_driver_sha256=sha(path),
        case_result=str(case_result))


def endpoint_points(frame, y, x):
    depth, k, transform = frame.depth[y, x], frame.intrinsics, frame.world_from_camera
    local = np.column_stack(((x+.5-k[0, 2])/k[0, 0]*depth,
        (y+.5-k[1, 2])/k[1, 1]*depth, depth))
    return local@transform[:3, :3].T+transform[:3, 3]


def component_counts(cells):
    """Three evidence graphs, no cell fill or geometric joining."""
    cells = np.unique(np.asarray(cells, np.int64).reshape(-1, 3), axis=0)
    count = len(cells)
    parents = np.tile(np.arange(count), (3, 1))
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    def find(graph, item):
        while parents[graph, item] != item:
            parents[graph, item] = parents[graph, parents[graph, item]]
            item = parents[graph, item]
        return item
    offsets = [v for v in product(range(-2, 3), repeat=3) if v > (0, 0, 0)]
    for index, cell in enumerate(cells):
        x, y, z = map(int, cell)
        for dx, dy, dz in offsets:
            other = lookup.get((x+dx, y+dy, z+dz))
            if other is None:
                continue
            graphs = [2]
            if max(abs(dx), abs(dy), abs(dz)) <= 1:
                graphs.append(1)
                if abs(dx)+abs(dy)+abs(dz) == 1:
                    graphs.append(0)
            for graph in graphs:
                first, last = find(graph, index), find(graph, other)
                if first != last:
                    parents[graph, last] = first
    return dict(unique_cells=count, face6=len({find(0, i) for i in range(count)}),
        face26=len({find(1, i) for i in range(count)}),
        radius2_chebyshev=len({find(2, i) for i in range(count)}))


def nearest_distances(points):
    """Self excluded; duplicate returns are legitimate zero-distance neighbors."""
    points = np.asarray(points, float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('NN diagnostic requires finite continuous endpoints')
    if len(points) < 2:
        return dict(status='UNMEASURED_FEWER_THAN_TWO_ENDPOINTS', endpoints=len(points),
            exact_unique_coordinates=len(np.unique(points, axis=0)))
    euclidean, linf = np.full(len(points), np.inf), np.full(len(points), np.inf)
    for start in range(0, len(points), 128):
        delta = abs(points[start:start+128, None, :]-points[None, :, :])
        row = np.arange(len(delta))
        delta[row, start+row, :] = np.inf
        euclidean[start:start+len(delta)] = np.sqrt(np.square(delta).sum(2).min(1))
        linf[start:start+len(delta)] = delta.max(2).min(1)
    def stats(values):
        return dict(min=float(values.min()), p50=float(np.quantile(values, .5)),
            p95=float(np.quantile(values, .95)), worst=float(values.max()))
    return dict(status='MEASURED_CONTINUOUS_CURRENT_INPUT', endpoints=len(points),
        exact_unique_coordinates=len(np.unique(points, axis=0)), euclidean_m=stats(euclidean), linf_m=stats(linf),
        fraction_nearest_linf_gt_010_m=float(np.mean(linf > .01)),
        fraction_nearest_linf_gt_020_m=float(np.mean(linf > .02)))


def phase_counts(points, spec):
    """Same fixed in-original-ROI points, eight evaluation-only infinite lattices.

    Negative phase-cell indices are allowed. No shifted ROI drops points and no
    shifted lattice is supplied to an occupancy map, packet or controller.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    result = {name: [] for name in ('unique_cells', 'face6', 'face26', 'radius2_chebyshev')}
    for phase in PHASES:
        cells = np.floor((points-np.asarray(spec.bounds_min))/spec.voxel_size-np.asarray(phase)).astype(np.int64)
        counts = component_counts(cells)
        for name in result:
            result[name].append(counts[name])
    return result


def actual_selection(archive, index, frame, explicit, max_depth, spec):
    """Reconstruct only captured ray endpoints, retaining each rejection stage."""
    keys = [f'camera_{index}_'+name for name in ('selected_y', 'selected_x', 'automatic_mask')]
    if any(key not in archive for key in keys):
        raise LookupError('Actual selected pixels or automatic mask were not captured')
    y, x, automatic = (np.asarray(archive[key]) for key in keys)
    if (y.ndim != 1 or x.shape != y.shape or y.dtype.kind not in 'iu' or x.dtype.kind not in 'iu'
            or automatic.dtype != np.bool_ or automatic.shape != frame.depth.shape
            or np.any(y < 0) or np.any(y >= frame.depth.shape[0])
            or np.any(x < 0) or np.any(x >= frame.depth.shape[1])):
        raise ValueError('Invalid captured actual selected-pixel/mask contract')
    if len(set(zip(y.tolist(), x.tolist()))) != len(y):
        raise ValueError('Duplicate selected pixel in one actual camera capture')
    if explicit is not None and (np.asarray(explicit).dtype != np.bool_ or np.shape(explicit) != frame.depth.shape):
        raise ValueError('Original explicit mask shape/type changed')
    depth = frame.depth[y, x]
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    preexcluded = np.zeros(len(y), bool) if explicit is None else np.asarray(explicit)[y, x]
    if np.any(~valid | preexcluded):
        raise ValueError('Captured selected pixel violates original pre-sampling depth/explicit-mask gate')
    retained = ~automatic[y, x]
    yy, xx = y[retained], x[retained]
    points = endpoint_points(frame, yy, xx)
    cells = spec.world_to_indices(points)
    inside = (cells >= 0).all(1)
    return dict(y=yy, x=xx, recorded_y=y, recorded_x=x,
        points=points[inside], cells=cells[inside], inside=inside,
        recorded_selected_pixels=len(y), automatic_masked_selected=int((~retained).sum()),
        selected_unmasked_valid_rays=int(retained.sum()), selected_in_grid_endpoint_rays=int(inside.sum()),
        selected_out_of_grid_endpoint_rays=int((~inside).sum()))


def frame_references(record, frame):
    references = record.get('frame_evaluation_objects', [])
    if references:
        matching = [value for value in references if value['camera_id'] == frame.camera_id and tuple(value['frame_token']) == tuple(frame.token)]
        return matching[0]['objects'] if len(matching) == 1 else []
    if record.get('physical_attribution_status', '').startswith('UNMEASURED'):
        return []
    return record.get('evaluation_objects', [])


def diagnose_record(record, *, phases=False, artifacts=None):
    source = record['source_artifacts']
    array_path, payload_path = Path(record['snapshot_npz']), Path(source['input_pickle'])
    metadata = json.loads(Path(source['metadata_json']).read_text())
    configuration_path = array_path.parent.parent/'runtime_profile.toml'
    cfg, settings_proof = effective_mapping_settings(configuration_path, array_path.parent.parent/'result.json')
    if sha(array_path) != source['arrays_sha256'] or sha(payload_path) != source['input_sha256']:
        raise ValueError('Immutable captured arrays or input changed since primary manifest')
    payload = pickle.loads(payload_path.read_bytes())
    if [frame.camera_id for frame in payload.frames] != metadata['camera_ids']:
        raise ValueError('Captured selected-mask camera order does not match original payload')
    snapshot = load_snapshot(record)
    evidence = prepare_evidence(snapshot, now=record['snapshot']['evaluated_at'], evidence_ttl=record['snapshot']['evidence_ttl'])
    clouds, selected_frames, counters = [], [], {}
    with np.load(array_path, allow_pickle=False) as archive:
        for index, frame in enumerate(payload.frames):
            explicit = (payload.exclusion_masks or {}).get(frame.camera_id)
            selected = actual_selection(archive, index, frame, explicit, cfg['max_depth'], snapshot.spec)
            selected_frames.append(selected)
            clouds.append(selected['points'])
            for name in ('recorded_selected_pixels', 'automatic_masked_selected', 'selected_unmasked_valid_rays',
                    'selected_in_grid_endpoint_rays', 'selected_out_of_grid_endpoint_rays'):
                counters[name] = counters.get(name, 0)+selected[name]
        current_points = np.concatenate(clouds) if clouds else np.empty((0, 3))
        current_cells = np.unique(snapshot.spec.world_to_indices(current_points), axis=0)
        if 'integrate_occupied_indices' not in archive:
            raise LookupError('Actual occupied integration indices unavailable to verify endpoint reconstruction')
        actual_cells = np.unique(archive['integrate_occupied_indices'], axis=0)
        if not np.array_equal(current_cells, actual_cells):
            raise ValueError('Captured selected endpoint cell set does not equal actual occupied integration input')
    result = dict(case_id=record['case_id'], sequence_id=record['sequence_id'], layout_id=record['layout_id'], scenario=record['scenario'],
        status='MEASURED_ACTUAL_SELECTED_RAYS', source_input_sha256=source['input_sha256'],
        source_array_sha256=source['arrays_sha256'], profile_sha256=sha(configuration_path),
        settings=dict(max_depth_m=cfg['max_depth'], ray_budget=cfg['max_rays'], stride=cfg['stride'], voxel_size_m=snapshot.spec.voxel_size),
        settings_source_binding=settings_proof,
        actual_selected=counters, current_input_cells=component_counts(current_cells),
        current_input_continuous_nn=nearest_distances(current_points),
        canonical_eligible=component_counts(np.argwhere(evidence.eligible)), canonical_supported=component_counts(evidence.cells),
        actual_integration_cell_set_matches=True, physical_objects={})
    if phases:
        result['evaluation_only_current_cloud_origin_phases'] = phase_counts(current_points, snapshot.spec)
    # All actual-sampling and continuous geometry measurements above precede
    # physical scoring. No reference reaches a mapper or a selection function.
    objects = {}
    for frame, selected in zip(payload.frames, selected_frames):
        explicit = (payload.exclusion_masks or {}).get(frame.camera_id)
        valid = np.isfinite(frame.depth) & (frame.depth > 0) & (frame.depth <= cfg['max_depth'])
        if explicit is not None:
            valid &= ~np.asarray(explicit)
        yy, xx = np.nonzero(valid)
        for obj in frame_references(record, frame):
            row = objects.setdefault(obj['name'], dict(raw_valid_depth_matching_pixels=0,
                raw_valid_in_grid_matching_pixels=0, raw_stride_grid_valid_matching_pixels=0,
                recorded_selected_physical_hit_rays_before_automatic_mask=0,
                selected_unmasked_physical_hit_rays=0, points=[]))
            with np.errstate(invalid='ignore'):
                matches = reference_matches(frame, yy, xx, obj)
                before_mask_matches = reference_matches(frame, selected['recorded_y'], selected['recorded_x'], obj)
                selected_matches = reference_matches(frame, selected['y'], selected['x'], obj)
            raw_points = endpoint_points(frame, yy[matches], xx[matches])
            raw_cells = snapshot.spec.world_to_indices(raw_points)
            row['raw_valid_depth_matching_pixels'] += int(matches.sum())
            row['raw_valid_in_grid_matching_pixels'] += int((raw_cells >= 0).all(1).sum())
            row['raw_stride_grid_valid_matching_pixels'] += int(np.count_nonzero(matches & (yy % cfg['stride'] == 0) & (xx % cfg['stride'] == 0)))
            row['recorded_selected_physical_hit_rays_before_automatic_mask'] += int(before_mask_matches.sum())
            selected_matches = selected_matches[selected['inside']]
            row['selected_unmasked_physical_hit_rays'] += int(selected_matches.sum())
            row['points'].append(selected['points'][selected_matches])
    for name, value in objects.items():
        points = np.concatenate(value.pop('points'))
        value['current_selected_endpoint_cells'] = component_counts(snapshot.spec.world_to_indices(points))
        value['continuous_selected_endpoint_nn'] = nearest_distances(points)
        if phases:
            value['evaluation_only_origin_phases'] = phase_counts(points, snapshot.spec)
        result['physical_objects'][name] = value
    result['physical_reference_status'] = record.get('physical_attribution_status')
    if artifacts is not None:
        path = Path(artifacts)/(record['sequence_id']+'-current-endpoints.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, current_selected_endpoints_world_m=current_points, current_endpoint_cells=current_cells)
    return result


def run(manifest, *, phases=False, artifacts=None):
    before = {str(path.relative_to(ROOT)):sha(path) for path in PROTECTED}
    document = json.loads(Path(manifest).read_text())
    final = {row['sequence_id']:row for row in document['records']}
    started = time.perf_counter()
    rows = []
    for record in final.values():
        try:
            rows.append(diagnose_record(record, phases=phases, artifacts=artifacts))
        except LookupError as exc:
            rows.append(dict(case_id=record['case_id'], status='UNMEASURED_ACTUAL_CAPTURE_UNAVAILABLE', reason=str(exc)))
        except (ValueError, OSError, KeyError) as exc:
            rows.append(dict(case_id=record['case_id'], status='REJECTED_INCONSISTENT_CAPTURE', reason=type(exc).__name__+': '+str(exc)))
    if before != {str(path.relative_to(ROOT)):sha(path) for path in PROTECTED}:
        raise RuntimeError('Protected source changed during read-only diagnostic')
    proofs = []
    for row in rows:
        if 'settings_source_binding' not in row:
            continue
        proof = dict(row.pop('settings_source_binding'))
        row['settings_case_result'] = proof.pop('case_result')
        if proof not in proofs:
            proofs.append(proof)
        row['settings_source_proof_id'] = proofs.index(proof)
    return dict(schema_version=1, scope='Actual current selected-ray sparsity versus evaluation-only voxel phase sensitivity',
        manifest_sha256=sha(manifest), method_sha256=sha(__file__),
        reference_helper_sha256=sha(ROOT/'scripts/evaluate_obstacle_representation_v3.py'),
        representation_helper_sha256=sha(ROOT/'scripts/obstacle_representation_v3.py'),
        protected_production_sha256=before, production_sources_unchanged=True,
        configuration_source_proofs=proofs,
        evaluation_only_phase_offsets_voxels=PHASES if phases else [],
        qualifications=[
            'Recorded selected pixels and automatic masks are mandatory; no policy, mask or map is recomputed.',
            'Raw physical counts use original finite positive depth, original max-depth and explicit masks; automatic masks were evaluated only at selected pixels, so raw dense counts are not densely robot-filtered.',
            'Current-input endpoint cells and continuous distances are separate from cumulative canonical eligible/supported states at their original extraction clock.',
            'Nearest L-infinity and Euclidean distances are independently minimized, exclude self and retain duplicate returns as zero-distance neighbors.',
            'A nearest L-infinity distance above20mm rules out another current endpoint in a26-neighbor10mm cell for any uniform origin; above10mm only rules out sharing one10mm cell. This says nothing about unseen surfaces or temporal map support.',
            'Eight fixed half-voxel phases keep identical points in the original physical ROI and change only evaluation lattice indices; no phase is selected, no ROI point is dropped, no packet/map/controller is altered.',
            'Face6,face26 and Chebyshev radius2 are graph counts; no intervening cell or unknown solid is filled.',
            'Physical first-hit depth tolerance15mm is scoring only; no physical reference enters actual sampling or the canonical map. Full-solid IoU remains unmeasured.'],
        diagnostic_wall_seconds=time.perf_counter()-started, rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--origin-phases', action='store_true')
    parser.add_argument('--artifacts-dir')
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, phases=args.origin_phases, artifacts=args.artifacts_dir)
    write_json(args.result, result)
    print(f"Measured {sum(row['status']=='MEASURED_ACTUAL_SELECTED_RAYS' for row in result['rows'])}/{len(result['rows'])} actual final inputs", flush=True)
    return 0 if all(row['status']=='MEASURED_ACTUAL_SELECTED_RAYS' for row in result['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
