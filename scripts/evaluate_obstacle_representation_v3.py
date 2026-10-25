#!/usr/bin/env python3
"""Offline immutable-snapshot evaluation for obstacle representation V3.

Native capture arrays and canonical replay arrays are kept distinct. Cumulative
hit/free masks are never accepted as canonical occupancy. Independent physical
labels are consulted only after extraction. No simulator or GPU is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import pickle
import sys
import time
import tomllib
from types import MappingProxyType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancySnapshot
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings, ObstacleExtractor
from scripts.camera_workload_metrics import distribution


ARRAYS = ('states', 'probabilities', 'observed', 'last_observed', 'expired')
PROTECTED = tuple(ROOT/name for name in (
    'robot_skill_stack/world/spatial_mapping/obstacles.py',
    'robot_skill_stack/world/spatial_mapping/occupancy.py',
    'robot_skill_stack/world/spatial_mapping/rgbd.py',
    'robot_skill_stack/world/spatial_mapping/runtime.py',
    'robot_skill_stack/integrations/isaac/manipulation/motion/obstacles.py'))
DEFAULT_VARIANTS = (
    ('baseline', 'baseline', 'baseline'), ('axis_best', 'axis_best', 'baseline'),
    ('greedy', 'greedy', 'baseline'), ('component', 'component', 'baseline'),
    ('coverage', 'baseline', 'coverage'), ('diversity', 'baseline', 'diversity'),
    ('motion', 'baseline', 'motion'))


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def immutable(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_digest(snapshot):
    result = hashlib.sha256()
    for name in ARRAYS:
        value = getattr(snapshot, name)
        result.update(name.encode())
        result.update(str((value.dtype.str, value.shape)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def load_snapshot(record, base=ROOT):
    """Reject evidence-only archives and make all canonical arrays irreversible."""
    path = Path(record['snapshot_npz'])
    if not path.is_absolute():
        path = Path(base)/path
    meta = record['snapshot']
    spec = GridSpec(tuple(meta['bounds_min']), tuple(meta['bounds_max']), meta['voxel_size'])
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ARRAYS)-set(archive.files)
        if missing:
            raise ValueError('Not a canonical snapshot; missing arrays: '+','.join(sorted(missing)))
        arrays = {name: immutable(archive[name]) for name in ARRAYS}
    if any(value.shape != spec.shape for value in arrays.values()):
        raise ValueError('Canonical arrays do not match the recorded grid')
    if not np.isin(arrays['states'], (0, 1, 2)).all():
        raise ValueError('Invalid in-grid occupancy classification')
    return OccupancySnapshot(spec, *(arrays[name] for name in ARRAYS), meta['timestamp'],
        int(meta['revision']), int(meta['generation']), MappingProxyType(dict(meta.get('health', {}))))


def snapshot_metadata(snapshot, now, ttl):
    return dict(bounds_min=snapshot.spec.bounds_min, bounds_max=snapshot.spec.bounds_max,
        voxel_size=snapshot.spec.voxel_size, dimensions=snapshot.spec.shape,
        timestamp=snapshot.timestamp, revision=snapshot.revision, generation=snapshot.generation,
        evaluated_at=now, evidence_ttl=ttl, health=dict(snapshot.health))


def inventory():
    rows = []
    for folder in sorted((ROOT/'outputs').glob('motion-planning-v2b-baseline-rgbd-*-direct-capture')):
        audit = folder/'mapping-audit'
        jobs = [json.loads(line) for line in (audit/'jobs.jsonl').read_text().splitlines()]
        rows.append(dict(path=str(folder.relative_to(ROOT)),
            canonical_snapshot_files=sum(bool(row.get('arrays_npz')) for row in jobs),
            raw_mapping_inputs=sum(bool(row.get('input_pickle')) for row in jobs),
            clock_scope='Actual recorded processor phase clocks',
            layout_scope='Historical V2b diagnostic EYES, not authored A_current'))
    for path in sorted((ROOT/'outputs/camera-observability-feasibility').glob('sampling-home-*/sampling_capture.json')):
        document = json.loads(path.read_text())
        rows.append(dict(path=str(path.relative_to(ROOT)), layout=document['capture_layout'],
            raw_mapping_inputs=len(document['entries']), canonical_snapshot_files=0,
            clock_scope='No worker clock recorded; explicit canonical offline replay required',
            physical_guard=document.get('robot_fixture_clearance')))
    native_scenarios = set()
    for audit in sorted((ROOT/'outputs/obstacle-representation-v3').glob('*/*/mapping-audit')):
        path = audit/'jobs.jsonl'
        result = audit.parent/'result.json'
        if not path.exists() or not result.exists():
            continue
        document = json.loads(result.read_text())
        scenario = document.get('representation_v3', {}).get('scenario')
        jobs = [json.loads(line) for line in path.read_text().splitlines()]
        count = sum(bool(row.get('arrays_npz')) and bool(row.get('snapshot')) for row in jobs)
        if count:
            native_scenarios.add(scenario)
        rows.append(dict(path=str(audit.parent.relative_to(ROOT)), scenario=scenario,
            layout=document.get('research', {}).get('layout_id'), canonical_snapshot_files=count,
            raw_mapping_inputs=sum(bool(row.get('input_pickle')) for row in jobs),
            clock_scope='Original recorded snapshot and packet/extraction clocks are distinct and retained',
            full_capture_excluded_from_timing_claims=True))
    return dict(status='READ_ONLY_INVENTORY', sources=rows,
        initially_missing_historical_canonical_scenarios=['two_separated', 'narrow', 'moving', 'native_true_cylinder'],
        missing_current_native_scenarios=sorted(set(('empty', 'static', 'sparse', 'cylinder', 'partial', 'separated', 'narrow', 'moving'))-native_scenarios),
        limitation='Physical-motion geometry_reference.npz and cumulative unique_hit/unique_free archives are not canonical maps.')


def _motion_reference(payload, configuration):
    from scripts.obstacle_representation_v3 import public_motion_context
    goal = getattr(payload, 'public_goal', configuration.get('goal'))
    start = configuration.get('start')
    corridor = getattr(payload, 'public_corridor', [] if start is None or goal is None else [start, goal])
    snapshots = getattr(payload, 'robot_snapshots', None)
    values = (dict(robot_points=(), goal=goal, corridor=corridor, relevance_radius_m=.20)
        if not snapshots and goal is None and not corridor else
        asdict(public_motion_context(snapshots, goal=goal, corridor=corridor)))
    return dict(values,
        source='All sorted exposure-camera mesh origins, exact duplicate points removed; payload public goal/corridor or recorded START/GOAL. Identical native policy contract, no fixture labels.')


def recorded_clocks(item):
    """Preserve the distinct snapshot and actual original extraction times."""
    meta = item['snapshot']
    snapshot_now = meta['timestamp']+meta['health']['age_seconds']
    clocks = [float(r['value']) for r in item.get('clock_reads', []) if r['phase'] == 'processor']
    if clocks and not any(abs(value-snapshot_now) <= 1e-7 for value in clocks):
        raise ValueError('Recorded snapshot clock and its age disagree')
    packet_now = item.get('packet', {}).get('extracted_at')
    extraction_now = float(packet_now if packet_now is not None else clocks[-1] if clocks else snapshot_now)
    if clocks and abs(extraction_now-clocks[-1]) > 1e-7:
        raise ValueError('Recorded packet and extraction clock disagree')
    if extraction_now < snapshot_now-1e-7:
        raise ValueError('Extraction predates its snapshot')
    return snapshot_now, extraction_now


def pose_matrix(pose):
    """Recorded Isaac XYZ, quaternion XYZW, normalized for float32 pose reads."""
    position, q = np.asarray(pose, float)[:3], np.asarray(pose, float)[3:]
    q = q/np.linalg.norm(q)
    x, y, z, w = q
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = position
    return result


def physical_reference(folder, metrics, scenario, configuration, payload):
    """Evaluation-only actual poses. A dynamic pose is never guessed."""
    objects = [dict(obj) for obj in configuration.get('objects', [])]
    if not objects:
        return objects, [], dict(status='NO_EXTERNAL_FIXTURE_OBJECTS')
    paths = metrics.get('capture', {}).get('body_paths', [])
    trace_path = folder/'trace.json'
    samples = json.loads(trace_path.read_text()).get('samples', []) if trace_path.exists() else []
    setup = metrics.get('fixture_setup', {}).get('actual_actor_poses', [])
    is_native = bool(setup and len(setup) == len(objects) and paths)
    if not is_native:
        return objects, [], dict(status='NOMINAL_STATIC_REFERENCE_ONLY' if not any(o.get('moving') for o in objects)
            else 'UNMEASURED_MOVING_POSES_NOT_CAPTURED', qualification='Historical configured primitive pose; no independent per-exposure pose record.')
    object_indices = []
    for obj in objects:
        matching = [i for i, path in enumerate(paths) if path.endswith('/'+obj['name'])]
        if len(matching) != 1:
            return [], [], dict(status='UNMEASURED_FIXTURE_BODY_PATH_AMBIGUOUS')
        object_indices.append(matching[0])
    cylinder = folder/'cylinder_cooked_reference.npz'
    def at_pose(obj, pose):
        value = dict(obj, world_from_body=pose_matrix(pose).tolist(), center=list(pose[:3]))
        if scenario == 'cylinder':
            value.update(kind='cooked_convex', cooked_npz=str(cylinder),
                cooked_sha256=sha(cylinder) if cylinder.exists() else None)
        return value
    if scenario == 'cylinder' and not cylinder.exists():
        return [], [], dict(status='UNMEASURED_ACTUAL_COOKED_CYLINDER_MISSING')
    if any(obj.get('moving') for obj in objects):
        frames, matched = [], 0
        times = np.array([sample['simulation_time'] for sample in samples], float)
        for frame in payload.frames if payload is not None else ():
            exposure = float(frame.token[1])
            valid = np.flatnonzero(np.abs(times-exposure) <= 1e-6)
            if not len(valid):
                frames.append(dict(camera_id=frame.camera_id, frame_token=frame.token, status='UNMATCHED_EXPOSURE_CLOCK', objects=[]))
                continue
            # Same simulated step can appear twice; identical body poses are
            # required before accepting the timestamp join.
            poses = [np.asarray(samples[int(i)]['poses'])[object_indices] for i in valid]
            if any(not np.array_equal(poses[0], pose) for pose in poses[1:]):
                frames.append(dict(camera_id=frame.camera_id, frame_token=frame.token, status='AMBIGUOUS_EXPOSURE_POSE', objects=[]))
                continue
            matched += 1
            frames.append(dict(camera_id=frame.camera_id, frame_token=frame.token, status='EXACT_SIMULATION_STEP_MATCH',
                reference_simulation_time=float(times[valid[0]]), rendering_simulation_time=exposure,
                absolute_time_error_seconds=float(abs(times[valid[0]]-exposure)),
                objects=[at_pose(obj, pose) for obj, pose in zip(objects, poses[0])]))
        return objects, frames, dict(status='EXPOSURE_MATCHED_DYNAMIC_REFERENCE' if matched else 'UNMEASURED_NO_EXPOSURE_POSE_MATCHES',
            matched_frames=matched, total_frames=len(frames), simulation_time_tolerance_seconds=1e-6,
            qualification='Only actual same-rendering/physics-step poses qualify; no interpolation, initial-pose substitution or monotonic-clock matching.')
    matrices = [pose_matrix(pose) for pose in setup]
    drift = max((float(np.max(np.abs(pose_matrix(sample['poses'][idx])-matrix)))
        for sample in samples for idx, matrix in zip(object_indices, matrices)), default=0.)
    if drift > 1e-7:
        return [], [], dict(status='UNMEASURED_STATIC_POSE_DRIFT', maximum_matrix_component_drift=drift)
    values = [at_pose(obj, pose) for obj, pose in zip(objects, setup)]
    if scenario == 'cylinder':
        final = np.asarray(metrics['cylinder_reference']['sampled_actual_world_from_body'])
        if not np.allclose(final, matrices[0], rtol=0, atol=1e-7):
            return [], [], dict(status='UNMEASURED_COOKED_REFERENCE_POSE_MISMATCH')
    return values, [], dict(status='ACTUAL_STATIC_KINEMATIC_REFERENCE',
        maximum_matrix_component_drift=drift, trace_samples_checked=len(samples),
        qualification='Actual post-insertion kinematic fixture pose; verified unchanged through recorded motion trace. Pre-motion exposures rely on the static kinematic-body invariant, not a trace timestamp join.')


def export_capture(folder, maximum, phase, layout_override=None):
    folder = Path(folder).resolve()
    if folder.name == 'mapping-audit':
        folder = folder.parent
    result_path = folder/'result.json'
    document = json.loads(result_path.read_text()) if result_path.exists() else {}
    metrics = document.get('metrics', document)
    configuration = metrics.get('configuration', {})
    profile = tomllib.loads((folder/'runtime_profile.toml').read_text())
    cfg = profile['spatial_mapping']
    audit = folder/'mapping-audit' if (folder/'mapping-audit').exists() else folder
    jobs = [json.loads(line) for line in (audit/'jobs.jsonl').read_text().splitlines()]
    placement, measurement = metrics.get('fixture_placement_started_at'), metrics.get('measurement_started_at')
    eligible = [row for row in jobs if row.get('arrays_npz') and row.get('snapshot')
        and row['snapshot'].get('timestamp') is not None and row.get('accepted', False)]
    if phase in ('before-motion', 'post-insertion') and placement is not None:
        eligible = [row for row in eligible if row['input_timestamp'] >= placement]
    if phase == 'before-motion' and measurement is not None:
        eligible = [row for row in eligible if row['input_timestamp'] <= measurement]
    chosen = eligible[-maximum:] if maximum else eligible
    scenario = document.get('representation_v3', {}).get('scenario', configuration.get('scenario', folder.name))
    for name in ('empty', 'static', 'sparse', 'cylinder', 'partial', 'separated', 'narrow', 'moving'):
        if '-'+name+'-' in folder.name or folder.name.endswith('-'+name):
            scenario = name
            break
    if scenario == 'sparse':
        scenario = 'thin_box_reference'
    layout = layout_override or document.get('research', {}).get('layout_id') or metrics.get('camera_layout', {}).get('layout_id') or (
        'V2b_reference_EYES' if 'motion-planning-v2b-baseline' in folder.name else 'UNSPECIFIED_NATIVE_LAYOUT')
    rows = []
    for item in chosen:
        meta = item['snapshot']
        snapshot_now, now = recorded_clocks(item)
        payload_path = audit/item['input_pickle'] if item.get('input_pickle') else None
        payload = pickle.loads(payload_path.read_bytes()) if payload_path is not None else None
        objects, frame_objects, reference = physical_reference(folder, metrics, scenario, configuration, payload)
        record = dict(case_id=f'{folder.name}/job-{item["job"]:05d}', sequence_id=folder.name,
            scenario=scenario, layout_id=layout, snapshot_npz=str(audit/item['arrays_npz']),
            snapshot=dict(meta, snapshot_evaluated_at=snapshot_now, evaluated_at=now, evidence_ttl=cfg.get('evidence_ttl', 5.)),
            source_artifacts=dict(kind='actual_native_worker_snapshot', arrays_sha256=sha(audit/item['arrays_npz']),
                metadata_json=str(audit/item['metadata_json']), metadata_sha256=sha(audit/item['metadata_json']),
                input_pickle=None if payload_path is None else str(payload_path),
                input_sha256=None if payload_path is None else sha(payload_path),
                clock_reads=item.get('clock_reads'), original_input_timestamp=item['input_timestamp']),
            evaluation_objects=item.get('evaluation_objects', objects), frame_evaluation_objects=frame_objects,
            physical_reference=reference,
            motion_context=_motion_reference(payload, configuration),
            robot_state_reference=dict(initial_joints=metrics.get('initial_joints'),
                source=str(result_path), exposure_match=False,
                qualification='Recorded initial fixture joint pose, not necessarily this snapshot exposure.'),
            support_plane_z=profile.get('scene', {}).get('support_plane_z', 0.))
        # Dynamic truth must be independent and exposure matched, never inferred
        # from the object's initial center or future collision outcome.
        if item.get('evaluation_objects') is not None:
            record['physical_attribution_status'] = 'EXPOSURE_MATCHED_REFERENCE'
        else:
            record['physical_attribution_status'] = reference['status']
        rows.append(record)
    return rows


class LogicalReplayClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


def replay_capture(folder, maximum, output):
    """Create real canonical maps, preserving source times and logical spacing."""
    from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor
    from scripts.camera_observability_sampling import load_case
    case = load_case(folder, maximum)
    clock = LogicalReplayClock()
    processor = MappingProcessor(case['settings'], clock=clock)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, item in enumerate(case['rows']):
        source = case['base']/item['input_pickle']
        payload = pickle.loads(source.read_bytes())
        clock.now = max(frame.acquired_at for frame in payload.frames)+.001
        completed = processor.compute(payload)
        if not completed.accepted:
            raise RuntimeError('Canonical input replay rejected: '+str(dict(completed.diagnostics)))
        snapshot = completed.snapshot
        path = output/f'snapshot-{i:03d}.npz'
        np.savez_compressed(path, **{name: getattr(snapshot, name) for name in ARRAYS})
        record = dict(case_id=f'{case["label"]}/canonical-replay-{i:03d}',
            sequence_id=case['folder'].name, scenario='thin_cylinder', layout_id=case['label'],
            snapshot_npz=str(path.resolve()), snapshot=snapshot_metadata(snapshot, clock.now, case['settings'].evidence_ttl),
            source_artifacts=dict(kind='canonical_replay_of_native_depth', arrays_sha256=sha(path),
                input_pickle=str(source), input_sha256=sha(source), original_input_timestamp=completed.input_timestamp,
                clock_scope='Explicit logical max(original acquired_at)+1ms, preserving original inter-input spacing. Not recorded worker completion/publication time.'),
            evaluation_objects=case['objects'], motion_context=_motion_reference(payload, {}),
            physical_attribution_status='STATIC_REFERENCE_ONLY', robot_state_reference=dict(exposure_match=False),
            support_plane_z=0.)
        records.append(record)
    return records


def bounds_mask(bounds, shape):
    values = np.asarray(bounds)
    if values.size == 0:
        values = np.empty((0, 6), np.int64)
    if values.ndim != 2 or values.shape[1] != 6 or values.dtype.kind not in 'iu':
        raise ValueError('Cell bounds must be integer Nx6')
    covered = np.zeros(shape, bool)
    overlap = 0
    for box in values:
        lo, hi = box[[0, 2, 4]], box[[1, 3, 5]]
        if np.any(lo < 0) or np.any(hi > shape) or np.any(hi <= lo):
            raise ValueError('Box extends outside grid or has nonpositive size')
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        overlap += int(covered[slices].sum())
        covered[slices] = True
    return covered, overlap


def packet_bounds(packet, spec):
    result = []
    lower, upper = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max)
    for box in packet.boxes:
        lo = np.asarray(box.center)-np.asarray(box.dimensions)/2
        hi = np.asarray(box.center)+np.asarray(box.dimensions)/2
        first = (lo-lower)/spec.voxel_size
        stop = (hi-lower)/spec.voxel_size
        at_upper = np.isclose(hi, upper, rtol=0, atol=1e-10)
        stop[at_upper] = np.asarray(spec.shape)[at_upper]
        if not np.allclose(first, np.rint(first), rtol=0, atol=1e-7) or not np.allclose(stop, np.rint(stop), rtol=0, atol=1e-7):
            raise ValueError('Packet geometry does not align exactly with voxel cell faces')
        result.append(np.column_stack((np.rint(first), np.rint(stop))).ravel())
    return np.asarray(result, np.int64).reshape(-1, 6)


def audit_packet(snapshot, packet, supported, eligible, now, ttl):
    bounds = packet_bounds(packet, snapshot.spec)
    covered, overlap = bounds_mask(bounds, snapshot.spec.shape)
    errors = dict(free_cells=int(np.count_nonzero(covered & (snapshot.states == 1))),
        unknown_cells=int(np.count_nonzero(covered & (snapshot.states == 0))),
        ineligible_cells=int(np.count_nonzero(covered & ~eligible)),
        unsupported_cells=int(np.count_nonzero(covered & ~supported)), overlap_cells=overlap,
        wrong_expiry_boxes=0, expired_boxes=0, wrong_voxel_count_boxes=0)
    errors['wrong_extracted_at'] = int(abs(packet.extracted_at-now) > 1e-9)
    for box, bound in zip(packet.boxes, bounds):
        slices = tuple(slice(int(bound[2*a]), int(bound[2*a+1])) for a in range(3))
        expected = float(snapshot.last_observed[slices].min())+ttl
        errors['wrong_expiry_boxes'] += int(not math.isfinite(expected) or abs(box.expires_at-expected) > 1e-8)
        errors['expired_boxes'] += int(box.expires_at <= now)
        errors['wrong_voxel_count_boxes'] += int(box.voxel_count != math.prod(s.stop-s.start for s in slices))
    if (packet.generation, packet.revision, packet.timestamp) != (snapshot.generation, snapshot.revision, snapshot.timestamp):
        raise ValueError('Representation changed source generation/revision/timestamp')
    return bounds, covered, errors


def fragmentation(mask):
    from scripts.obstacle_representation_v3 import connected_components, pack_cells
    cells = np.argwhere(mask)
    groups = connected_components(cells, adjacency=6)
    sizes = [len(group) for group in groups]
    nonrectangular = sum(math.prod((group.max(0)-group.min(0)+1).tolist()) != len(group) for group in groups)
    axis_counts = {order: len(pack_cells(mask, 'axis_'+order).bounds)
                   for order in ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')}
    return dict(supported_cells=len(cells), face6_components=len(groups),
        face6_singletons=sum(size == 1 for size in sizes), face6_component_sizes=distribution(sizes),
        nonrectangular_face6_components=int(nonrectangular),
        exact_box_count_lower_bound=len(groups)+int(nonrectangular), axis_order_candidate_counts=axis_counts,
        proof='Each face-connected component needs at least one exact cuboid; a nonrectangular component cannot fit one and needs at least two. This is a lower bound, not a global optimal-packing solution.')


def radius2_components(cells):
    remaining = set(map(tuple, np.asarray(cells, int).tolist()))
    offsets = [v for v in product(range(-2, 3), repeat=3) if v != (0, 0, 0)]
    sizes = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        pending, size = [seed], 0
        while pending:
            cell = pending.pop()
            size += 1
            for delta in offsets:
                neighbor = tuple(a+b for a, b in zip(cell, delta))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        sizes.append(size)
    return dict(adjacency='Chebyshev radius2 (124 neighbors), evidence graph only; gaps are never filled',
        count=len(sizes), singleton_components=sum(n == 1 for n in sizes), sizes=distribution(sizes))


class CountingNumericInterface:
    """Only count calls requested by the real registry; no SDK cost claim."""
    def add_cuboid(self, proxy, **kwargs):
        return True

    def remove_obstacle(self, proxy):
        return True

    def update_world(self, values=None):
        return None


def convex_depth(directions, origin, planes):
    """Actual cooked face halfspaces, bounded chunks; optical ray parameter."""
    first = np.full(len(directions), np.inf)
    for begin in range(0, len(directions), 4096):
        rays = directions[begin:begin+4096]
        nearest = np.full(len(rays), np.inf)
        for normals, offsets in planes:
            denominator = rays@normals.T
            numerator = offsets-origin@normals.T
            parallel = np.abs(denominator) < 1e-15
            valid = ~np.any(parallel & (numerator < -1e-12), axis=1)
            ratios = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=~parallel)
            near = np.where(denominator < -1e-15, ratios, -np.inf).max(1)
            far = np.where(denominator > 1e-15, ratios, np.inf).min(1)
            valid &= (far >= near) & (near > 0)
            nearest = np.minimum(nearest, np.where(valid, near, np.inf))
        first[begin:begin+len(rays)] = nearest
    return first


def reference_matches(frame, y, x, obj):
    """Reference geometry never enters mapper, packing, or selection."""
    from scripts.camera_observability_sampling import normalize_object, physical_matches
    from scripts.camera_observability_geometry import shape_depth
    if 'world_from_body' not in obj:
        return physical_matches(frame, y, x, normalize_object(obj))
    body_from_camera = np.linalg.inv(np.asarray(obj['world_from_body']))@frame.world_from_camera
    k = frame.intrinsics
    rays = np.column_stack(((x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))))
    directions = rays@body_from_camera[:3, :3].T
    origin = body_from_camera[:3, 3]
    if obj.get('kind') == 'cooked_convex':
        from scripts.obstacle_representation_v3_cylinder import _planes
        path = Path(obj['cooked_npz'])
        if sha(path) != obj['cooked_sha256']:
            raise ValueError('Cooked physical reference changed after manifest export')
        with np.load(path, allow_pickle=False) as archive:
            indices = sorted(int(name.removeprefix('vertices_')) for name in archive.files if name.startswith('vertices_'))
            planes = [_planes((archive[f'vertices_{i}'], archive[f'face_counts_{i}'], archive[f'face_indices_{i}'])) for i in indices]
        if not planes:
            raise ValueError('Actual cooked physical reference has no hulls')
        near = convex_depth(directions, origin, planes)
    else:
        shape = normalize_object(dict(obj, center=[0., 0., 0.], yaw=0.))
        near = shape_depth(shape, origin, directions)
    measured = frame.depth[y, x]
    return np.isfinite(near) & (near > 0) & np.isfinite(measured) & (measured > 0) & (abs(measured-near) <= .015)


def physical_attribution(record, snapshot):
    source = record['source_artifacts'].get('input_pickle')
    if not source or record.get('physical_attribution_status', '').startswith('UNMEASURED'):
        return {}, dict(status='UNMEASURED', reason=record.get('physical_attribution_status', 'No native input'))
    payload = pickle.loads(Path(source).read_bytes())
    objects = record.get('evaluation_objects', [])
    masks = {obj['name']: np.zeros(snapshot.spec.shape, bool) for obj in objects}
    matched_frames = 0
    # Score every valid native pixel, not just production-selected rays. The
    # canonical state later limits the denominator to actually occupied cells.
    for frame in payload.frames:
        references = record.get('frame_evaluation_objects', [])
        if references:
            match = [value for value in references if value['camera_id'] == frame.camera_id and tuple(value['frame_token']) == tuple(frame.token)]
            frame_objects = match[0]['objects'] if len(match) == 1 else []
        else:
            frame_objects = objects
        matched_frames += int(bool(frame_objects))
        y, x = np.indices(frame.depth.shape)
        y, x = y.ravel(), x.ravel()
        k, transform = frame.intrinsics, frame.world_from_camera
        for obj in frame_objects:
            matched = reference_matches(frame, y, x, obj)
            yy, xx = y[matched], x[matched]
            depth = frame.depth[yy, xx]
            points = np.column_stack(((xx+.5-k[0, 2])/k[0, 0]*depth,
                (yy+.5-k[1, 2])/k[1, 1]*depth, depth))@transform[:3, :3].T+transform[:3, 3]
            cells = snapshot.spec.world_to_indices(points)
            valid = (cells >= 0).all(1)
            masks[obj['name']][tuple(cells[valid].T)] = True
    return masks, dict(status='MEASURED_NATIVE_DEPTH_ATTRIBUTION', optical_depth_tolerance_m=.015,
        matched_reference_frames=matched_frames, total_frames=len(payload.frames),
        reference=record.get('physical_reference', {}),
        scope='Evaluation-only primitive or actual cooked-convex face-halfspace first-hit intersections. Cooked-plane queries are floating numeric face representations, not a vertex-hull distance certificate. Counts are current eligible occupied cells also supported by a matching current native depth endpoint; historical-only occupied surfaces and full-solid IoU are unmeasured.')


def _component_coverage(groups, selected):
    counts = [int(selected[tuple(group.T)].sum()) for group in groups]
    return dict(face6_components_with_any_selected=sum(n > 0 for n in counts),
        face6_components_completely_omitted=sum(n == 0 for n in counts),
        largest_completely_omitted_component=max((len(group) for group, n in zip(groups, counts) if not n), default=0))


def gap_audit(record, snapshot, selected):
    """Whole voxel cells between disjoint fixture AABBs; UNKNOWN stays separate."""
    from scripts.camera_observability_sampling import normalize_object, aabb_reference
    if record.get('physical_attribution_status', '').startswith('UNMEASURED'):
        return dict(status='UNMEASURED_DYNAMIC_REFERENCE', regions=[])
    objects = [aabb_reference(normalize_object(obj)) for obj in record.get('evaluation_objects', [])]
    if any(obj.get('moving') for obj in objects):
        return dict(status='UNMEASURED_DYNAMIC_GAP_REFERENCE', regions=[])
    rows = []
    for i, first in enumerate(objects):
        for second in objects[i+1:]:
            a0 = np.array(first['center'])-np.array(first['dimensions'])/2
            a1 = np.array(first['center'])+np.array(first['dimensions'])/2
            b0 = np.array(second['center'])-np.array(second['dimensions'])/2
            b1 = np.array(second['center'])+np.array(second['dimensions'])/2
            for axis in range(3):
                lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
                if a1[axis] < b0[axis]:
                    lo[axis], hi[axis] = a1[axis], b0[axis]
                elif b1[axis] < a0[axis]:
                    lo[axis], hi[axis] = b1[axis], a0[axis]
                else:
                    continue
                if np.any(hi <= lo):
                    continue
                start = np.maximum(0, np.ceil((lo-snapshot.spec.bounds_min)/snapshot.spec.voxel_size-1e-8).astype(int))
                stop = np.minimum(snapshot.spec.shape, np.floor((hi-snapshot.spec.bounds_min)/snapshot.spec.voxel_size+1e-8).astype(int))
                if np.any(stop <= start):
                    continue
                slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
                states = snapshot.states[slices]
                represented = selected[slices]
                rows.append(dict(objects=[first['name'], second['name']], gap_axis=axis,
                    physical_gap_width_m=float(hi[axis]-lo[axis]), full_voxel_cells=int(states.size),
                    known_free_cells=int((states == 1).sum()), unknown_cells=int((states == 0).sum()),
                    occupied_cells=int((states == 2).sum()), represented_cells=int(represented.sum()),
                    known_free_cells_covered=int((represented & (states == 1)).sum()),
                    unknown_cells_covered=int((represented & (states == 0)).sum())))
    return dict(status='MEASURED_REFERENCE_GAP_CELLS' if rows else 'NO_WHOLE_VOXEL_REFERENCE_GAP',
        scope='Full cells between pairs of fixture enclosing boxes, not full robot-passage or swept-path certification. Other scene geometry is not erased. UNKNOWN is never scored as a physical collision.', regions=rows)


def evaluate(manifests, variants, budgets, output, figures=None):
    from scripts.obstacle_representation_v3 import ExperimentalObstacleExtractor, MotionContext, connected_components, prepare_evidence, pack_cells, select_bounds
    from robot_skill_stack.integrations.isaac.manipulation.motion.obstacles import RmpflowObstacleRegistry
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_records = []
    for manifest in manifests:
        value = json.loads(Path(manifest).read_text())
        all_records.extend(value['records'])
    report = dict(schema_version=1, status='OFFLINE_RESEARCH_MEASUREMENT',
        snapshot_manifests=[dict(path=str(p), sha256=sha(p)) for p in manifests],
        implementation_sha256=sha(ROOT/'scripts/obstacle_representation_v3.py'),
        evaluator_sha256=sha(__file__), variants=[dict(id=v[0], packing=v[1], selection=v[2]) for v in variants],
        budgets=budgets, cases=[], qualifications=[
            'All variants receive the exact same immutable canonical snapshot and original recorded evaluation/TTL clocks.',
            'Canonical replays of raw native inputs retain source timestamps but declare their logical completion clock; they are not native freshness measurements.',
            'Physical object metadata is evaluation-only. Motion ranking receives only exposure robot points and a recorded public target/corridor when available.',
            'Registry call counts use the unchanged registry with a counting numeric interface. They predict orchestration operations, not installed-SDK registration latency.',
            'Full-capture IO perturbs native cadence; extraction wall times are offline CPU measurements, not control or GPU performance.',
            'Endpoint surface support is not full-solid reconstruction or a certified free-space/swept-path guarantee.'])
    registries, previous, bundles = {}, {}, {}
    for index, record in enumerate(all_records):
        snapshot = load_snapshot(record)
        now, ttl = record['snapshot']['evaluated_at'], record['snapshot']['evidence_ttl']
        digest = array_digest(snapshot)
        stage_started = time.perf_counter()
        evidence = prepare_evidence(snapshot, now=now, evidence_ttl=ttl)
        evidence_seconds = time.perf_counter()-stage_started
        frag = fragmentation(evidence.supported)
        frag['eligible_radius2_components'] = radius2_components(np.argwhere(evidence.eligible))
        truth = None
        groups = connected_components(evidence.cells)
        row = dict(case_id=record['case_id'], sequence_id=record['sequence_id'], scenario=record['scenario'],
            layout_id=record['layout_id'], snapshot=record['snapshot'], source_artifacts=record['source_artifacts'],
            snapshot_arrays_sha256=digest, evidence=dict(evidence.counts), fragmentation=frag,
            physical_attribution=dict(status='UNMEASURED_NO_SUCCESSFUL_EXTRACTION'), offline_shared_evidence_prepare_ms=evidence_seconds*1000, variants={})
        pack_profiles = {}
        packet_bundle = {}
        for budget in budgets:
            for name, packing, selection in variants:
                key = name if budget == 64 else name+'_'+str(budget)
                context = None
                if selection == 'motion':
                    values = record.get('motion_context', {})
                    if not values.get('robot_points') and not values.get('goal') and not values.get('corridor'):
                        row['variants'][key] = dict(status='UNMEASURED_NO_PUBLIC_MOTION_CONTEXT')
                        continue
                    context = MotionContext(**{k: values[k] for k in ('robot_points', 'goal', 'corridor', 'relevance_radius_m') if k in values})
                settings = ObstacleExtractionSettings(budget, record.get('support_plane_z', 0.))
                try:
                    if packing not in pack_profiles:
                        stage_started = time.perf_counter()
                        packed = pack_cells(evidence.supported, packing)
                        pack_profiles[packing] = (packed, (time.perf_counter()-stage_started)*1000)
                    packed, packing_ms = pack_profiles[packing]
                    stage_started = time.perf_counter()
                    selected_indices, selection_diagnostics = select_bounds(snapshot, packed.bounds,
                        settings=settings, selection=selection, motion_context=context)
                    selection_ms = (time.perf_counter()-stage_started)*1000
                    extractor = (ObstacleExtractor(settings, temporal_probabilities=True) if name == 'baseline'
                        else ExperimentalObstacleExtractor(settings, packing=packing, selection=selection, motion_context=context))
                    started = time.perf_counter()
                    packet = extractor.extract(snapshot, now=now, evidence_ttl=ttl)
                    elapsed = time.perf_counter()-started
                    bounds, selected, audit = audit_packet(snapshot, packet, evidence.supported, evidence.eligible, now, ttl)
                    candidates = (packed.bounds if name == 'baseline'
                                  else extractor.last_candidate_bounds)
                    candidate_mask, candidate_overlap = bounds_mask(candidates, snapshot.spec.shape)
                    audit.update(candidate_incorrect_cells=int((candidate_mask & ~evidence.supported).sum()),
                        candidate_overlap_cells=candidate_overlap,
                        candidate_missing_supported_cells=int((evidence.supported & ~candidate_mask).sum()),
                        stage_selection_disagrees=int(not np.array_equal(bounds, packed.bounds[selected_indices])))
                except Exception as exc:
                    row['variants'][key] = dict(status='REJECTED_EXTRACTION_EXCEPTION', error=type(exc).__name__+': '+str(exc))
                    continue
                valid = not any(audit.values())
                if truth is None:
                    truth, row['physical_attribution'] = physical_attribution(record, snapshot)
                physical = {}
                for object_id, mask in truth.items():
                    observed = mask & evidence.eligible
                    physical[object_id] = dict(eligible_native_depth_supported_cells=int(observed.sum()),
                        supported_cells=int((observed & evidence.supported).sum()),
                        candidate_covered_cells=int((observed & candidate_mask).sum()),
                        selected_cells=int((observed & selected).sum()),
                        candidate_cuboids_with_attributed_cells=sum(bool(observed[tuple(slice(int(b[2*a]), int(b[2*a+1])) for a in range(3))].any()) for b in candidates),
                        selected_cuboids_with_attributed_cells=sum(bool(observed[tuple(slice(int(b[2*a]), int(b[2*a+1])) for a in range(3))].any()) for b in bounds))
                state_key = (record['sequence_id'], key)
                registry = registries.setdefault(state_key, RmpflowObstacleRegistry(CountingNumericInterface(), max_obstacles=budget, meters_per_unit=1.))
                before = registry.diagnostics
                registry.sync(packet, now=now)
                after = registry.diagnostics
                operations = {k: after[k]-before[k] for k in ('additions', 'removals', 'pose_updates', 'dimension_updates', 'identity_reuses', 'world_updates')}
                old = previous.get(state_key)
                stable = dict(first_packet=old is None)
                if old is not None:
                    old_ids, old_selected = old
                    ids = {box.id for box in packet.boxes}
                    stable.update(id_jaccard=len(ids & old_ids)/max(1, len(ids | old_ids)),
                        selected_cell_jaccard=int((selected & old_selected).sum())/max(1, int((selected | old_selected).sum())))
                previous[state_key] = ({box.id for box in packet.boxes}, selected)
                expiry_registry = RmpflowObstacleRegistry(CountingNumericInterface(), max_obstacles=budget, meters_per_unit=1.)
                expiry_registry.sync(packet, now=now)
                late = max((box.expires_at for box in packet.boxes), default=now)+1e-6
                expiry_registry.sync(packet, now=late)
                row['variants'][key] = dict(status='PASS_EXACT_AUDIT' if valid else 'REJECT_INVALID_GEOMETRY',
                    packing=packing, selection=selection, budget=budget, extraction_wall_ms=elapsed*1000,
                    offline_stage_ms=dict(evidence_prepare=evidence_seconds*1000, packing=packing_ms, selection=selection_ms,
                        scope='Independent SDK-free helper calls on identical inputs, separate from the full extraction timer. Packing timing reused for identical packings within this snapshot; these are not native pipeline timings.'),
                    diagnostics=dict(packet.diagnostics), audit=audit, physical_objects=physical,
                    known_free_cells_covered=audit['free_cells'], unknown_cells_covered=audit['unknown_cells'],
                    gap_preservation=gap_audit(record, snapshot, selected),
                    registry_operations=operations, temporal_stability=stable,
                    quiet_packet_expiry_removed_all=expiry_registry.diagnostics['registered'] == 0,
                    expiry_remaining_seconds=distribution(box.expires_at-now for box in packet.boxes),
                    component_coverage=_component_coverage(groups, selected))
                packet_bundle[key] = dict(boxes=[asdict(box) for box in packet.boxes], diagnostics=dict(packet.diagnostics))
                # A representative voxel/cuboid view for later figures and
                # static Lula analysis; never supply evaluation labels to it.
                artifact_path = output/(hashlib.sha256((record['case_id']+key).encode()).hexdigest()[:16]+'.npz')
                np.savez_compressed(artifact_path,
                    eligible=evidence.eligible, supported=evidence.supported, selected=selected,
                    selected_bounds=bounds, candidate_bounds=candidates)
                row['variants'][key]['cell_audit_npz'] = str(artifact_path.resolve())
        if digest != array_digest(snapshot):
            raise RuntimeError('A representation mutated the canonical snapshot')
        row['snapshot_unchanged'] = True
        report['cases'].append(row)
        bundles[record['sequence_id']] = dict(case_id=record['case_id'], layout_id=record['layout_id'], scenario=record['scenario'],
            snapshot=record['snapshot'], packets=packet_bundle, source_artifacts=record['source_artifacts'],
            robot_state_reference=record.get('robot_state_reference', {}))
        print(f"Evaluated {index+1}/{len(all_records)}: {record['case_id']}", flush=True)
    report['all_geometry_audits_pass'] = all(value['status'] in ('PASS_EXACT_AUDIT', 'UNMEASURED_NO_PUBLIC_MOTION_CONTEXT')
        for row in report['cases'] for value in row['variants'].values())
    report['summary'] = summarize(report['cases'])
    write_json(output/'packet_bundle.json', dict(schema_version=1, cases=list(bundles.values())))
    if figures:
        make_figures(report, output, figures)
    return report


def summarize(cases):
    result = []
    sequences = sorted({row['sequence_id'] for row in cases})
    for sequence in sequences:
        rows = [row for row in cases if row['sequence_id'] == sequence]
        last = rows[-1]
        for name, final in last['variants'].items():
            values = [row['variants'][name] for row in rows if row['variants'].get(name, {}).get('status') == 'PASS_EXACT_AUDIT']
            if not values:
                continue
            result.append(dict(sequence_id=sequence, layout_id=last['layout_id'], scenario=last['scenario'], variant=name,
                snapshots=len(values), final=final, fragmentation=last['fragmentation'],
                extraction_wall_ms=distribution(v['extraction_wall_ms'] for v in values),
                registry_operations={k: sum(v['registry_operations'][k] for v in values) for k in values[0]['registry_operations']},
                steady_registry_operations={k: sum(v['registry_operations'][k] for v in values[1:]) for k in values[0]['registry_operations']}))
    return result


def compact_report(report, source):
    """Commit scalar evidence and hashes; keep verbose capture diagnostics raw."""
    result = {key: report[key] for key in ('schema_version', 'status', 'snapshot_manifests',
        'implementation_sha256', 'evaluator_sha256', 'variants', 'budgets', 'qualifications',
        'all_geometry_audits_pass', 'production_sources_unchanged') if key in report}
    result['full_report'] = dict(path=str(Path(source)), sha256=sha(source),
        scope='Ignored full scalar diagnostics and exact numeric source/cell-audit references; no cumulative mask replaces canonical state.')
    result['cases'] = []
    audit_totals = Counter()
    for row in report['cases']:
        frag = row['fragmentation']
        value = dict(case_id=row['case_id'], snapshot_arrays_sha256=row['snapshot_arrays_sha256'],
            clock={key:row['snapshot'][key] for key in ('timestamp', 'revision', 'generation', 'snapshot_evaluated_at', 'evaluated_at', 'evidence_ttl') if key in row['snapshot']},
            evidence={key:row['evidence'][key] for key in ('occupied_voxels', 'supported_voxels', 'spatial_supported_voxels', 'strong_sparse_retained', 'rejected_sparse_voxels')},
            face6_components=frag['face6_components'], face6_singletons=frag['face6_singletons'],
            exact_box_count_lower_bound=frag['exact_box_count_lower_bound'],
            axis_order_candidate_counts=frag['axis_order_candidate_counts'],
            radius2_components=frag['eligible_radius2_components']['count'], packing_counts={})
        for name, variant in row['variants'].items():
            audit_totals.update(variant.get('audit', {}))
            if name in ('baseline', 'axis_best', 'greedy', 'component'):
                value['packing_counts'][name] = variant.get('diagnostics', {}).get('candidate_boxes')
        result['cases'].append(value)
    result['audit_error_totals'] = dict(audit_totals)
    result['all_input_snapshots_unchanged'] = all(row['snapshot_unchanged'] for row in report['cases'])
    result['all_quiet_packet_expiry_checks_pass'] = all(value.get('quiet_packet_expiry_removed_all') is True
        for row in report['cases'] for value in row['variants'].values())
    def quantiles(values):
        return {key: (round(value, 4) if isinstance(value, float) else value)
            for key,value in distribution(values).items() if key in ('count', 'p50', 'p95', 'p99', 'worst')}
    result['summary'] = []
    for row in report['summary']:
        matched = [case['variants'][row['variant']] for case in report['cases'] if case['sequence_id']==row['sequence_id']]
        final = row['final']
        value = {key: row[key] for key in ('sequence_id', 'layout_id', 'scenario', 'variant', 'snapshots', 'steady_registry_operations')}
        value['initial_registry_additions'] = matched[0]['registry_operations']['additions']
        value['final'] = dict(diagnostics={key:item for key,item in final['diagnostics'].items() if key in (
            'candidate_boxes', 'proxy_count', 'represented_voxels', 'omitted_voxels', 'selected_floor_boxes', 'selected_hazard_boxes')},
            physical_objects={name:{key:item for key,item in obj.items() if key in (
                'eligible_native_depth_supported_cells', 'supported_cells', 'candidate_covered_cells', 'selected_cells')}
                for name,obj in final['physical_objects'].items()})
        value['extraction_wall_ms'] = quantiles(v['extraction_wall_ms'] for v in matched)
        value['offline_stage_ms'] = {stage:quantiles(v['offline_stage_ms'][stage] for v in matched)
            for stage in ('evidence_prepare', 'packing', 'selection')}
        value['consecutive_id_jaccard'] = quantiles(v['temporal_stability'].get('id_jaccard') for v in matched)
        value['consecutive_selected_cell_jaccard'] = quantiles(v['temporal_stability'].get('selected_cell_jaccard') for v in matched)
        result['summary'].append(value)
    final_cases = {}
    for row in report['cases']:
        final_cases[row['sequence_id']] = row
    result['sequences'] = [dict(sequence_id=sequence, layout_id=row['layout_id'], scenario=row['scenario'],
        final_case_id=row['case_id'], snapshot_count=sum(case['sequence_id']==sequence for case in report['cases']),
        physical_attribution=row['physical_attribution'],
        baseline_gap_preservation=row['variants'].get('baseline', {}).get('gap_preservation', {}))
        for sequence,row in final_cases.items()]
    result['snapshot_count'] = len(report['cases'])
    result['exact_audit_count'] = sum(len(row['variants']) for row in report['cases'])
    result['baseline_at_proven_lower_bound_snapshots'] = sum(
        row['variants'].get('baseline', {}).get('diagnostics', {}).get('candidate_boxes') ==
        row['fragmentation']['exact_box_count_lower_bound'] for row in report['cases'])
    result['supported_face_singleton_cell_fraction'] = distribution(
        row['fragmentation']['face6_singletons']/max(1, row['fragmentation']['supported_cells']) for row in report['cases'])
    if len(report['budgets']) > 1:
        # A single snapshot per scenario has no cadence or churn estimate.
        # Keep only the actual diagnostic counts, with explicit column names.
        result['diagnostic_scope'] = 'One exact final snapshot per sequence. Budget sensitivity only; no temporal/native performance inference.'
        result['count_columns'] = ['candidate_boxes', 'proxy_count', 'represented_voxels', 'omitted_voxels',
            'selected_floor_boxes', 'selected_hazard_boxes']
        result['physical_columns'] = ['eligible_native_depth_supported_cells', 'supported_cells', 'candidate_covered_cells', 'selected_cells']
        result['summary'] = [dict(sequence_id=row['sequence_id'], variant=row['variant'],
            counts=[row['final']['diagnostics'].get(key) for key in result['count_columns']],
            physical_objects={name:[obj[key] for key in result['physical_columns']] for name,obj in row['final']['physical_objects'].items()})
            for row in report['summary']]
    return result


def make_figures(report, output, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for row in report['summary']:
        final = row['final']
        observed = sum(o['eligible_native_depth_supported_cells'] for o in final['physical_objects'].values())
        retained = sum(o['selected_cells'] for o in final['physical_objects'].values())
        if observed:
            ax.scatter(final['diagnostics']['proxy_count'], retained/observed*100, label=row['variant'], alpha=.65)
    ax.set_xlabel('Selected cuboids');ax.set_ylabel('Native-depth-attributed eligible cells retained (%)')
    ax.set_title('Measured snapshot support versus bounded representation')
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.savefig(directory/'proxy_count_vs_observed_coverage.png', dpi=160)
    plt.close(fig)
    # Genuine final snapshots, with omitted support distinguished from cells
    # rejected by the unchanged support rule. UNKNOWN/FREE are not plotted.
    final_cases = {}
    for row in report['cases']:
        final_cases[row['sequence_id']] = row
    preferred = [row for row in final_cases.values() if row['scenario'] in ('partial', 'separated', 'narrow', 'cylinder')]
    if not preferred:
        preferred = list(final_cases.values())
    preferred = preferred[:4]
    if preferred:
        fig = plt.figure(figsize=(13, 4*len(preferred)), layout='constrained')
        for row_index, row in enumerate(preferred):
            available = row['variants']
            alternative = next((name for name in ('motion', 'diversity', 'coverage', 'component', 'greedy', 'axis_best')
                if available.get(name, {}).get('status') == 'PASS_EXACT_AUDIT'), 'baseline')
            for column, name in enumerate(('baseline', alternative)):
                value = available.get(name, {})
                if value.get('status') != 'PASS_EXACT_AUDIT':
                    continue
                with np.load(value['cell_audit_npz'], allow_pickle=False) as archive:
                    eligible, supported, selected = archive['eligible'], archive['supported'], archive['selected']
                    bounds = archive['selected_bounds']
                ax = fig.add_subplot(len(preferred), 2, row_index*2+column+1, projection='3d')
                spec = row['snapshot']
                lower, voxel = np.asarray(spec['bounds_min']), spec['voxel_size']
                for mask, color, label in ((eligible & ~supported, '#909090', 'Support rejected'),
                    (supported & ~selected, '#df8422', 'Supported, omitted'), (selected, '#2073ac', 'Selected')):
                    cells = np.argwhere(mask)
                    if len(cells):
                        points = lower+(cells+.5)*voxel
                        ax.scatter(*points.T, s=4, color=color, alpha=.7, label=label, rasterized=True)
                for bound in bounds:
                    lo = lower+bound[[0, 2, 4]]*voxel
                    hi = np.minimum(lower+bound[[1, 3, 5]]*voxel, spec['bounds_max'])
                    for axis in range(3):
                        other = [a for a in range(3) if a != axis]
                        for ends in product((0, 1), repeat=2):
                            first, last = lo.copy(), lo.copy()
                            for a, end in zip(other, ends):
                                first[a] = last[a] = hi[a] if end else lo[a]
                            last[axis] = hi[axis]
                            ax.plot(*np.column_stack((first, last)), color='#194b7c', alpha=.4, linewidth=.45)
                diag = value['diagnostics']
                ax.set_title(f"{row['layout_id']} / {row['scenario']} / {name}\n{diag['proxy_count']} boxes, {diag['represented_voxels']}/{int(supported.sum())} supported cells")
                ax.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
                ax.view_init(elev=24, azim=-58)
                if row_index == 0 and column == 0:
                    ax.legend(fontsize=7)
        fig.suptitle('Actual canonical snapshot cells and selected cuboid edges; UNKNOWN/FREE omitted from view', fontsize=12)
        fig.savefig(directory/'actual_voxel_cuboid_views.png', dpi=160)
        plt.close(fig)
    rows = [row for row in final_cases.values() if 'baseline' in row['variants']]
    if rows:
        labels = [row['layout_id']+' / '+row['scenario'] for row in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(9, len(rows)*.75), 5), layout='constrained')
        values = [row['fragmentation']['exact_box_count_lower_bound'] for row in rows]
        ax.bar(x-.25, values, width=.16, label='Proven lower bound')
        for column, name in enumerate(('baseline', 'axis_best', 'greedy', 'component')):
            values = [row['variants'].get(name, {}).get('diagnostics', {}).get('candidate_boxes', np.nan) for row in rows]
            ax.bar(x-.09+column*.16, values, width=.16, label=name)
        ax.axhline(64, color='black', linestyle='--', linewidth=.9, label='64-box budget')
        ax.set_xticks(x, labels, rotation=55, ha='right', fontsize=8)
        ax.set_ylabel('Exact candidate cuboids (count)')
        ax.set_title('Measured fragmentation versus exact packing; all candidates preserve the same supported cells')
        ax.legend(fontsize=8)
        fig.savefig(directory/'fragmentation_and_packing.png', dpi=160)
        plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory');inv.add_argument('--result', required=True)
    export = commands.add_parser('export')
    export.add_argument('--capture-dir', action='append', required=True)
    export.add_argument('--max-snapshots', type=int, default=8)
    export.add_argument('--phase', choices=('before-motion', 'post-insertion', 'all'), default='before-motion')
    export.add_argument('--layout-id')
    export.add_argument('--manifest', required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('--capture-dir', required=True)
    replay.add_argument('--max-snapshots', type=int, default=8)
    replay.add_argument('--artifacts-dir', required=True)
    replay.add_argument('--manifest', required=True)
    evaluate_parser = commands.add_parser('evaluate')
    evaluate_parser.add_argument('--manifest', action='append', required=True)
    evaluate_parser.add_argument('--variants', nargs='+', choices=[row[0] for row in DEFAULT_VARIANTS], default=[row[0] for row in DEFAULT_VARIANTS])
    evaluate_parser.add_argument('--budgets', nargs='+', type=int, choices=(16, 32, 64, 128), default=[64])
    evaluate_parser.add_argument('--artifacts-dir', required=True)
    evaluate_parser.add_argument('--figures-dir')
    evaluate_parser.add_argument('--result', required=True)
    figure_parser = commands.add_parser('figures')
    figure_parser.add_argument('--report', required=True)
    figure_parser.add_argument('--artifacts-dir', required=True)
    figure_parser.add_argument('--figures-dir', required=True)
    compact_parser = commands.add_parser('compact')
    compact_parser.add_argument('--report', required=True)
    compact_parser.add_argument('--result', required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    before = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
    if args.command == 'inventory':
        write_json(args.result, inventory())
    elif args.command == 'export':
        records = []
        for folder in args.capture_dir:
            records.extend(export_capture(folder, args.max_snapshots, args.phase, args.layout_id))
        write_json(args.manifest, dict(schema_version=1, records=records,
            scope='Original actual worker canonical arrays and recorded snapshot clocks; no replay or cumulative-mask substitution.'))
        print(f'Exported {len(records)} canonical snapshot references', flush=True)
    elif args.command == 'replay':
        records = replay_capture(args.capture_dir, args.max_snapshots, args.artifacts_dir)
        write_json(args.manifest, dict(schema_version=1, records=records,
            scope='Canonical mapper replay of actual native frames with explicitly declared logical completion clock.'))
    elif args.command == 'evaluate':
        variants = [row for row in DEFAULT_VARIANTS if row[0] in args.variants]
        report = evaluate(args.manifest, variants, args.budgets, args.artifacts_dir, args.figures_dir)
        report['production_sources_unchanged'] = before == {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}
        write_json(args.result, report)
    elif args.command == 'figures':
        make_figures(json.loads(Path(args.report).read_text()), args.artifacts_dir, args.figures_dir)
    else:
        write_json(args.result, compact_report(json.loads(Path(args.report).read_text()), args.report))
    if before != {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED}:
        raise RuntimeError('Protected production source changed during evaluation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
