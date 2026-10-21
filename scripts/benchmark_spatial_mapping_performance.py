#!/usr/bin/env python3
"""Capture the unchanged V2 fixture, then compare mapping on its exact inputs.

Capture launches Kit and must be run sequentially. Recorded CPU replay uses the
original acquisition timeline; it cannot establish live wall-clock freshness.
The capture hook writes uncompressed NPZs and is excluded from performance claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'd260377'
NAMESPACE = 'baseline_spatial_mapping'
PREFIXES = ('robot_skill_stack.world.spatial_mapping',
            'robot_skill_stack.integrations.isaac.spatial_mapping')
FIXTURE = 'tests/spatial_mapping/test_robot_self_filter.py'
CONFIGURATIONS = ('home_closed', 'extended_open', 'rotated_closed', 'extended_side_open',
                  'moving_4', 'moving_8', 'moving_12')


def _git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _rewrite(source):
    for prefix in PREFIXES:
        source = source.replace(prefix, NAMESPACE+prefix.removeprefix('robot_skill_stack'))
    return source


def export_baseline(output, reference=BASELINE):
    """Export mapping packages only; canonical DepthFrame stays shared."""
    output = Path(output)
    commit = _git('rev-parse', reference).decode().strip()
    files = _git('ls-tree', '-r', '--name-only', commit, '--',
                 *(prefix.replace('.', '/') for prefix in PREFIXES)).decode().splitlines()
    manifest = dict(commit=commit, namespace=NAMESPACE, files={})
    for name in files:
        if not name.endswith('.py'):
            continue
        original = _git('show', f'{commit}:{name}')
        exported = _rewrite(original.decode()).encode()
        destination = output/NAMESPACE/Path(name).relative_to('robot_skill_stack')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(exported)
        manifest['files'][name] = dict(original_sha256=_sha(original), exported_sha256=_sha(exported))
        parent = destination.parent
        while parent != output:
            initializer = parent/'__init__.py'
            if not initializer.exists():
                initializer.write_text('"""Isolated benchmark baseline namespace."""\n')
            parent = parent.parent
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'manifest.json', manifest)
    sys.path.insert(0, str(output.resolve()))
    return manifest


class CaptureRecorder:
    """Save the first fifteen production subset calls, not dense diagnostics."""
    def __init__(self, output, baseline):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.mesh_ids = {}
        self.manifest = dict(schema_version=1, status='RUNNING', baseline=baseline,
            input_scope='Unchanged V2 native fixture; first 15 production selected-pixel calls.',
            timing_scope='Capture hook I/O invalidates timings from this capture; use dedicated native profiles.',
            replay_scope='Original acquired_at and tokens, with explicit offline logical time; not a live freshness test.',
            settings=dict(bounds_min=[-.3, -.6, -.02], bounds_max=[.85, .6, 1.12],
                voxel_size=.01, stride=8, max_rays=1024, max_voxel_visits=500000,
                max_frame_age=5., stale_after=5., evidence_ttl=30.),
            representative_record_indices=[2, 5, 8, 11, 12, 13, 14], meshes=[], records=[])
        self.flush()

    def flush(self):
        _write_json(self.output/'capture_manifest.json', self.manifest)

    def record(self, frames, snapshots, selections, masks):
        if selections is None or len(self.manifest['records']) >= 15:
            return
        index = len(self.manifest['records'])
        configuration = CONFIGURATIONS[index//3] if index < 12 else CONFIGURATIONS[index-8]
        record = dict(index=index, configuration=configuration,
                      clear_map_before=index in (0, 3, 6, 9, 12), cameras=[])
        for camera_index, frame in enumerate(frames):
            snapshot = snapshots[frame.camera_id]
            mesh_ids = []
            for part in snapshot.parts:
                key = id(part.mesh)
                if key not in self.mesh_ids:
                    mesh_id = len(self.mesh_ids)
                    self.mesh_ids[key] = mesh_id
                    filename = f'mesh_{mesh_id:02d}.npz'
                    np.savez(self.output/filename, vertices=part.mesh.vertices, triangles=part.mesh.triangles)
                    self.manifest['meshes'].append(dict(index=mesh_id, file=filename,
                        vertices=len(part.mesh.vertices), triangles=len(part.mesh.triangles),
                        sha256=_sha((self.output/filename).read_bytes())))
                mesh_ids.append(self.mesh_ids[key])
            filename = f'record_{index:02d}_camera_{camera_index}.npz'
            ys, xs = selections[frame.camera_id]
            payload = dict(depth=frame.depth, intrinsics=frame.intrinsics,
                world_from_camera=frame.world_from_camera, token=np.asarray(frame.token),
                acquired_at=np.asarray(frame.acquired_at), part_mesh_ids=np.asarray(mesh_ids),
                part_matrices=np.stack([part.world_from_local for part in snapshot.parts]),
                simulation_time=np.asarray(snapshot.simulation_time),
                snapshot_frame_token=np.asarray(snapshot.frame_token), selected_y=ys, selected_x=xs,
                production_mask=masks[frame.camera_id])
            if frame.rgb is not None:
                payload['rgb'] = frame.rgb
            np.savez(self.output/filename, **payload)
            record['cameras'].append(dict(camera_id=frame.camera_id, file=filename,
                token=list(frame.token), acquired_at=float(frame.acquired_at),
                simulation_time=float(snapshot.simulation_time), selected_pixels=len(ys),
                masked_selected_pixels=int(masks[frame.camera_id][ys, xs].sum()),
                sha256=_sha((self.output/filename).read_bytes())))
        self.manifest['records'].append(record)
        self.flush()
        print(f'Captured mapping input {index+1}/15: {configuration}', flush=True)


def capture(args):
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = export_baseline(output/'baseline_snapshot', args.baseline_ref)
    fixture_original = _git('show', f'{baseline["commit"]}:{FIXTURE}')
    fixture_source = _rewrite(fixture_original.decode())
    (output/'native_fixture_original.py').write_bytes(fixture_original)
    (output/'native_fixture_isolated.py').write_text(fixture_source)
    recorder = CaptureRecorder(output, baseline)
    recorder.manifest['provenance'] = dict(fixture_commit=baseline['commit'],
        fixture_original_sha256=_sha(fixture_original),
        fixture_isolated_sha256=_sha(fixture_source.encode()),
        capture_script_sha256=_sha(Path(__file__).read_bytes()),
        working_commit=_git('rev-parse', 'HEAD').decode().strip())
    recorder.flush()
    generic = importlib.import_module(NAMESPACE+'.world.spatial_mapping.robot_filter')
    from tests.support import results
    original_masks = generic.RobotSelfFilter.masks
    original_write = results.write_result

    def masks(self, frames, snapshots_by_camera, *, pixels_by_camera=None):
        result = original_masks(self, frames, snapshots_by_camera, pixels_by_camera=pixels_by_camera)
        recorder.record(frames, snapshots_by_camera, pixels_by_camera, result)
        return result

    def write_result(path, *, status, metrics=None, error=None):
        complete = len(recorder.manifest['records']) == 15
        script_unchanged = recorder.manifest['provenance']['capture_script_sha256'] == _sha(Path(__file__).read_bytes())
        recorder.manifest.update(status='PASS' if status == 'PASS' and complete and script_unchanged else 'FAIL',
            native_status=status, error=error, complete=complete,
            capture_script_unchanged=script_unchanged)
        recorder.flush()
        metrics = dict(metrics or {})
        metrics['performance_capture'] = dict(records=len(recorder.manifest['records']),
            meshes=len(recorder.manifest['meshes']), complete=complete,
            manifest=str(output/'capture_manifest.json'), timings_usable=False)
        original_write(path, status=recorder.manifest['status'], metrics=metrics, error=error)

    sys.argv = [str(ROOT/FIXTURE), '--result', str(Path(args.result).resolve()),
                '--artifacts-dir', str(output/'native_artifacts')]
    with patch.object(generic.RobotSelfFilter, 'masks', masks), patch.object(results, 'write_result', write_result):
        exec(compile(fixture_source, str(ROOT/FIXTURE), 'exec'),
             {'__name__': '__main__', '__file__': str(ROOT/FIXTURE)})


def load_capture(directory):
    """Load owned inputs once; retain exact frame buffers, tokens and stamps."""
    from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'capture_manifest.json').read_text())
    if manifest['status'] != 'PASS' or len(manifest['records']) != 15:
        raise ValueError('Replay requires a complete PASS fifteen-record native capture')
    geometries, records = [], []
    for item in manifest['meshes']:
        path = directory/item['file']
        if _sha(path.read_bytes()) != item['sha256']:
            raise ValueError(f'Changed mesh input: {path}')
        with np.load(path, allow_pickle=False) as values:
            geometries.append((values['vertices'], values['triangles']))
    for item in manifest['records']:
        frames, poses, selected, masks = [], {}, {}, {}
        for camera in item['cameras']:
            path = directory/camera['file']
            if _sha(path.read_bytes()) != camera['sha256']:
                raise ValueError(f'Changed frame input: {path}')
            with np.load(path, allow_pickle=False) as values:
                frame = DepthFrame(camera['camera_id'], tuple(values['token']), values['depth'],
                    values['intrinsics'], values['world_from_camera'], float(values['acquired_at']),
                    values['rgb'] if 'rgb' in values else None)
                frames.append(frame)
                poses[frame.camera_id] = dict(mesh_ids=values['part_mesh_ids'],
                    matrices=values['part_matrices'], simulation_time=float(values['simulation_time']),
                    frame_token=tuple(values['snapshot_frame_token']))
                selected[frame.camera_id] = values['selected_y'], values['selected_x']
                masks[frame.camera_id] = values['production_mask']
        records.append(dict(index=item['index'], configuration=item['configuration'], frames=tuple(frames),
            poses=poses, selected=selected, production_masks=masks,
            clear_map_before=item['index'] in (0, 3, 6, 9, 12)))
    return manifest, geometries, records


def _statistics(values):
    values = np.asarray(values, float)
    return dict(samples=len(values), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(values.max()),
                mean_ms=float(values.mean())) if len(values) else dict(samples=0)


class ReplayClock:
    def __init__(self, value=0.):
        self.value = value

    def __call__(self):
        return self.value


class RecordedRobotFilter:
    """Cached exact meshes with captured exposure poses, without SDK resources."""
    def __init__(self, module, meshes):
        self.module, self.meshes = module, meshes
        self.filter = module.RobotSelfFilter(stride=8, depth_tolerance=.005, pose_tolerance=.01)
        self.record, self.diagnostics, self.last_masks = None, {}, {}
        self.selection_matches_capture = True

    def snapshots(self):
        return {camera: self.module.RobotSnapshot(tuple(self.module.MeshInstance(self.meshes[int(mesh)], matrix)
            for mesh, matrix in zip(pose['mesh_ids'], pose['matrices'])),
            pose['simulation_time'], pose['frame_token']) for camera, pose in self.record['poses'].items()}

    def masks(self, frames, *, pixels_by_camera=None):
        started = time.perf_counter()
        snapshots = self.snapshots()
        self.last_masks = self.filter.masks(frames, snapshots, pixels_by_camera=pixels_by_camera)
        if pixels_by_camera is not None:
            self.selection_matches_capture = self.selection_matches_capture and all(
                np.array_equal(pixels_by_camera[key][axis], self.record['selected'][key][axis])
                for key in pixels_by_camera for axis in (0, 1))
        self.diagnostics = dict(self.filter.diagnostics)
        self.diagnostics['elapsed_seconds_with_recorded_pose_assembly'] = time.perf_counter()-started
        return self.last_masks

    def close(self):
        pass


def _array_signature(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_sha(value.tobytes()))


def snapshot_signature(snapshot, queries):
    fields = ('states', 'probabilities', 'observed', 'last_observed', 'expired')
    return dict(arrays={name: _array_signature(getattr(snapshot, name)) for name in fields},
        query_arrays={name: _array_signature(getattr(queries, name)) for name in (*fields, 'indices')},
        timestamp=snapshot.timestamp, revision=snapshot.revision, generation=snapshot.generation,
        health=dict(snapshot.health), counts=snapshot.counts())


def _new_session(variant, manifest, meshes, clock, filtered):
    prefix = NAMESPACE if variant == 'baseline' else 'robot_skill_stack'
    integration = importlib.import_module(prefix+'.integrations.isaac.spatial_mapping')
    robot = importlib.import_module(prefix+'.world.spatial_mapping.robot_filter')
    settings = integration.SpatialMappingConfig(enabled=True, source='rgbd',
        automatic_robot_filter=filtered, **{key: value for key, value in manifest['settings'].items()})
    adapter = RecordedRobotFilter(robot, meshes) if filtered else None
    return integration.create_spatial_mapping(settings, clock=clock, robot_prim_path='/Franka',
                                              robot_filter=adapter), adapter


def _source_hashes():
    paths = [path for prefix in PREFIXES for path in (ROOT/prefix.replace('.', '/')).glob('*.py')]
    paths += list((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))
    return {str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in sorted(paths)}


def replay_mapping(args, manifest, geometries, records, output):
    variants = ('baseline', 'current') if args.variants == 'both' else (args.variants,)
    mesh_cache, initialization = {}, {}
    for variant in variants:
        prefix = NAMESPACE if variant == 'baseline' else 'robot_skill_stack'
        module = importlib.import_module(prefix+'.world.spatial_mapping.robot_filter')
        started = time.perf_counter()
        mesh_cache[variant] = [module.TriangleMesh(*geometry) for geometry in geometries]
        initialization[variant] = dict(mesh_bvh_seconds=time.perf_counter()-started,
            mesh_count=len(geometries), triangles=sum(len(triangles) for _, triangles in geometries))
    samples, signatures, checks, errors = [], {}, [], []
    representatives = set(manifest['representative_record_indices'])
    rng = np.random.default_rng(20261022)
    points = rng.uniform(manifest['settings']['bounds_min'], manifest['settings']['bounds_max'], (1000, 3))
    for repetition in range(-args.warmup, args.repeats):
        order = variants if repetition % 2 == 0 else variants[::-1]
        for variant in order:
            clock = ReplayClock()
            sessions = {filtered: _new_session(variant, manifest, mesh_cache[variant], clock, filtered)
                        for filtered in (False, True)}
            try:
                for record in records:
                    clock.value = max(frame.acquired_at for frame in record['frames'])
                    for filtered, (session, adapter) in sessions.items():
                        if record['clear_map_before']:
                            session.map.clear()
                        if adapter:
                            adapter.record = record
                        started = time.perf_counter()
                        updated = session.update(record['frames'], now=clock.value)
                        elapsed = (time.perf_counter()-started)*1000
                        if not updated:
                            raise RuntimeError(f'{variant}/{record["index"]}/{filtered}: {dict(session.map.health)}')
                        if repetition < 0:
                            continue
                        started = time.perf_counter()
                        snapshot = session.map.snapshot(now=clock.value)
                        snapshot_ms = (time.perf_counter()-started)*1000
                        started = time.perf_counter()
                        queried = session.map.query_many(points, now=clock.value)
                        query_ms = (time.perf_counter()-started)*1000
                        signature = snapshot_signature(snapshot, queried)
                        key = variant, record['index'], filtered
                        prior = signatures.setdefault(key, signature)
                        checks.append(dict(kind='deterministic_repeat', variant=variant,
                            record=record['index'], filtered=filtered, repetition=repetition, equal=prior == signature))
                        captured_mask_equal = None
                        if adapter:
                            captured_mask_equal = all(np.array_equal(adapter.last_masks[key], expected)
                                for key, expected in record['production_masks'].items())
                            checks.append(dict(kind='captured_production_masks', variant=variant,
                                record=record['index'], repetition=repetition, equal=captured_mask_equal,
                                selected_pixels_equal=adapter.selection_matches_capture))
                        row = dict(variant=variant, record=record['index'],
                            configuration=record['configuration'], filtered=filtered,
                            repetition=repetition, representative=record['index'] in representatives,
                            update_ms=elapsed, snapshot_ms=snapshot_ms, query_1000_ms=query_ms,
                            filter_ms=None if adapter is None else
                                adapter.diagnostics['elapsed_seconds_with_recorded_pose_assembly']*1000,
                            counts=snapshot.counts(), timestamp=snapshot.timestamp,
                            acquired_at=min(frame.acquired_at for frame in record['frames']),
                            diagnostics=dict(session.source.diagnostics),
                            captured_mask_equal=captured_mask_equal)
                        samples.append(row)
                print(f'Replay pass {repetition+1}/{args.repeats} {variant} complete', flush=True)
            except Exception as exc:
                errors.append(dict(variant=variant, repetition=repetition,
                    error=f'{type(exc).__name__}: {exc}'))
            finally:
                for session, _ in sessions.values():
                    session.close()
            _write_json(output/'mapping_samples.json', samples)
            _write_json(output/'mapping_errors.json', errors)
    for record in records:
        for filtered in (False, True):
            old, new = signatures.get(('baseline', record['index'], filtered)), signatures.get(('current', record['index'], filtered))
            if len(variants) == 2:
                checks.append(dict(kind='baseline_current_full_arrays_and_queries',
                    record=record['index'], filtered=filtered,
                    equal=old is not None and new is not None and old == new))
    _write_json(output/'equality_checks.json', checks)
    _write_json(output/'snapshot_signatures.json', [dict(variant=key[0], record=key[1], filtered=key[2], **value)
        for key, value in signatures.items()])
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for filtered in (False, True):
            selected = [row for row in samples if row['variant'] == variant and row['filtered'] == filtered and row['representative']]
            summary[variant]['filtered' if filtered else 'unfiltered'] = dict(
                update=_statistics([row['update_ms'] for row in selected]),
                snapshot=_statistics([row['snapshot_ms'] for row in selected]),
                query_1000=_statistics([row['query_1000_ms'] for row in selected]),
                configurations={name: _statistics([row['update_ms'] for row in selected if row['configuration'] == name])
                    for name in CONFIGURATIONS},
                filter=_statistics([row['filter_ms'] for row in selected if row['filter_ms'] is not None]))
    return dict(summary=summary, initialization=initialization,
        all_checks_passed=bool(checks) and all(check['equal'] and check.get('selected_pixels_equal', True) for check in checks),
        runtime_errors=errors, measured_representative_updates=sum(row['representative'] for row in samples),
        processed_updates=len(samples), warmup_passes=args.warmup, measured_passes=args.repeats,
        baseline_current_order='Alternated by full-pass parity; each variant sees unchanged original order.',
        clock_scope='Explicit original acquisition timeline; excludes live acquisition, pose-history callbacks and real completion-age acceptance.')


class OneShotFrames:
    def __init__(self):
        self.frames = None
        self.status = dict(state='waiting')

    def capture(self):
        result, self.frames = self.frames, None
        self.status = dict(state='captured' if result is not None else 'unchanged')
        return result

    def close(self):
        pass


def _semantic_signature(batch):
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value
    observations = []
    for observation in batch.observations:
        geometry = observation.geometry
        observations.append(dict(object_id=observation.object_id, class_name=observation.class_name,
            timestamp=observation.timestamp, graspable=observation.graspable,
            confidence=observation.confidence, position=observation.pose.position,
            orientation=observation.pose.orientation, size=observation.size,
            geometry=dict(shape=geometry.shape.value, box_size=geometry.box_size, yaw=geometry.yaw,
                radius=geometry.radius, length=geometry.length, axis=geometry.axis,
                confidence=geometry.confidence, metadata=geometry.metadata), metadata=observation.metadata))
    payload = plain(dict(timestamp=batch.timestamp, observations=observations))
    return _sha(json.dumps(payload, sort_keys=True).encode())


def semantic_and_combined(args, manifest, geometries, records, output):
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
    config = load_scene_config(ROOT/'config/scenes/multiview_perception_final.toml')
    mesh_cache = {}
    for variant in ('baseline', 'current'):
        prefix = NAMESPACE if variant == 'baseline' else 'robot_skill_stack'
        module = importlib.import_module(prefix+'.world.spatial_mapping.robot_filter')
        mesh_cache[variant] = [module.TriangleMesh(*geometry) for geometry in geometries]
    samples, contracts, checks, signatures = [], [], [], {}
    representatives = set(manifest['representative_record_indices'])
    variants = ('semantic_only', 'baseline', 'current')
    for repetition in range(-args.warmup, args.repeats):
        order = variants if repetition % 2 == 0 else variants[::-1]
        for variant in order:
            rig, clock = OneShotFrames(), ReplayClock()
            semantic = MultiViewProvider(rig, config.scene,
                voxel_size=config.multi_view.voxel_size, stride=config.multi_view.stride)
            mapping, adapter = (_new_session(variant, manifest, mesh_cache[variant], clock, True)
                                if variant != 'semantic_only' else (None, None))
            try:
                for record in records:
                    clock.value = max(frame.acquired_at for frame in record['frames'])
                    rig.frames = record['frames']
                    if mapping and record['clear_map_before']:
                        mapping.map.clear()
                    if adapter:
                        adapter.record = record
                    started = time.perf_counter()
                    batch = semantic.observe()
                    semantic_ms = (time.perf_counter()-started)*1000
                    mapping_ms = 0.
                    if mapping:
                        before = time.perf_counter()
                        if not mapping.update(record['frames'], now=clock.value):
                            raise RuntimeError(str(dict(mapping.map.health)))
                        mapping_ms = (time.perf_counter()-before)*1000
                    combined_ms = (time.perf_counter()-started)*1000
                    if repetition >= 0:
                        signature = _semantic_signature(batch)
                        key = variant, record['index']
                        prior = signatures.setdefault(key, signature)
                        checks.append(dict(kind='deterministic_semantic_repeat', variant=variant,
                            record=record['index'], repetition=repetition, equal=prior == signature))
                        contracts.append(dict(variant=variant, record=record['index'], repetition=repetition,
                            timestamp_preserved=batch.timestamp == min(frame.acquired_at for frame in record['frames']),
                            observation_timestamps_preserved=all(observation.timestamp == batch.timestamp for observation in batch.observations)))
                        samples.append(dict(variant=variant, record=record['index'],
                            configuration=record['configuration'], repetition=repetition,
                            representative=record['index'] in representatives, semantic_ms=semantic_ms,
                            mapping_ms=mapping_ms, combined_ms=combined_ms,
                            observation_classes=[observation.class_name for observation in batch.observations],
                            object_ids=[observation.object_id for observation in batch.observations],
                            semantic_output_sha256=signature, diagnostics=dict(semantic.diagnostics)))
                print(f'Semantic/combined pass {repetition+1}/{args.repeats} {variant} complete', flush=True)
            finally:
                semantic.close()
                if mapping:
                    mapping.close()
            _write_json(output/'semantic_samples.json', samples)
            _write_json(output/'semantic_contracts.json', contracts)
    for record in records:
        values = [signatures.get((variant, record['index'])) for variant in variants]
        checks.append(dict(kind='semantic_only_baseline_combined_current_combined', record=record['index'],
            equal=None not in values and len(set(values)) == 1))
    _write_json(output/'semantic_equality_checks.json', checks)
    return dict(scope='Measured sequential semantic then filtered mapping on identical full-arm frames; frozen semantic code throughout. This is not generalized tabletop latency or asynchronous control latency.',
        configurations={variant: {field: _statistics([sample[field] for sample in samples
            if sample['variant'] == variant and sample['representative']])
            for field in ('semantic_ms', 'mapping_ms', 'combined_ms')} for variant in variants},
        timestamps_preserved=all(row['timestamp_preserved'] and row['observation_timestamps_preserved'] for row in contracts),
        semantic_outputs_equal=all(row['equal'] for row in checks),
        semantic_source='Unchanged working-tree implementation for all three scheduling modes.')


class SourcePhaseProfiler:
    """Benchmark-only line attribution; instrumented times are never headlines.

    A line interval includes its nested calls. Scalar traversal also includes
    per-visit set insertion, unlike the batched implementation's later dedup.
    Core expiration inside integrate is nested evidence, reported separately.
    """
    def __init__(self, source, occupancy_map):
        import inspect
        lines, first = inspect.getsourcelines(source.update)
        self.code, self.phases = source.update.__func__.__code__, {}
        self.elapsed, self.last, self.expire_seconds = {}, None, 0.
        phase, evidence = 'acquisition_validation', False
        for offset, text in enumerate(lines):
            stripped = text.strip()
            if stripped.startswith('masks = {}'):
                phase = 'candidate_selection'
            elif stripped.startswith('if mask_provider is not None and'):
                phase = 'robot_geometry_masks'
            elif stripped.startswith('retained = sum'):
                phase = 'accumulation_dedup'
                evidence = True
            elif evidence and stripped.startswith('frame = by_id[key]'):
                phase = 'world_projection'
            elif evidence and stripped.startswith('view_occupied ='):
                phase = 'accumulation_dedup'
            elif evidence and (stripped.startswith('for endpoint in endpoints:') or
                              stripped.startswith('visited, count_visits = _traverse_rays')):
                phase = 'ray_traversal'
            elif evidence and (stripped.startswith('free.update(view_free)') or
                              stripped.startswith('view_free = np.unique')):
                phase = 'accumulation_dedup'
            elif stripped.startswith('changed = self.map.integrate'):
                phase = 'probabilistic_integration'
            self.phases[first+offset] = phase
        self.map, self.original_expire = occupancy_map, occupancy_map.expire

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        now = time.perf_counter()
        if self.last is not None:
            phase, before = self.last
            self.elapsed[phase] = self.elapsed.get(phase, 0.)+now-before
        self.last = (self.phases.get(frame.f_lineno, 'acquisition_validation'), now) if event != 'return' else None
        return self.trace

    def expire(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.original_expire(*args, **kwargs)
        finally:
            self.expire_seconds += time.perf_counter()-started

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        self.patch = patch.object(self.map, 'expire', self.expire)
        self.patch.__enter__()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
        self.patch.__exit__(*exc)


def profile_mapping(args, manifest, geometries, records, output):
    import cProfile
    import pstats
    profile_rows = []
    # Home's final batch is a concrete representative hotspot, not all-scene timing.
    target = next(record for record in records if record['index'] == args.profile_record)
    for variant in ('baseline', 'current'):
        prefix = NAMESPACE if variant == 'baseline' else 'robot_skill_stack'
        module = importlib.import_module(prefix+'.world.spatial_mapping.robot_filter')
        meshes = [module.TriangleMesh(*geometry) for geometry in geometries]
        for filtered in (False, True):
            clock = ReplayClock(max(frame.acquired_at for frame in target['frames']))
            session, adapter = _new_session(variant, manifest, meshes, clock, filtered)
            if adapter:
                adapter.record = target
            try:
                with SourcePhaseProfiler(session.source, session.map) as phases:
                    started = time.perf_counter()
                    assert session.update(target['frames'], now=clock.value)
                    instrumented_ms = (time.perf_counter()-started)*1000
                profile_rows.append(dict(variant=variant, filtered=filtered,
                    configuration=target['configuration'], record=target['index'],
                    instrumented_total_ms=instrumented_ms,
                    source_line_attribution_ms={key: value*1000 for key, value in phases.elapsed.items()},
                    expiry_nested_and_session_ms=phases.expire_seconds*1000,
                    scope='Single representative fresh-state update with line tracing. Nested expiry overlaps integration; scalar traversal includes per-visit insertion. Excluded from headline latency.'))
            finally:
                session.close()
            # Independent cProfile pass supplies callable/cumulative evidence.
            session, adapter = _new_session(variant, manifest, meshes, clock, filtered)
            if adapter:
                adapter.record = target
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                assert session.update(target['frames'], now=clock.value)
                profiler.disable()
                filename = f'{variant}_{"filtered" if filtered else "unfiltered"}.pstats'
                profiler.dump_stats(str(output/filename))
                statistics = pstats.Stats(profiler)
                rows = []
                for key, value in sorted(statistics.stats.items(), key=lambda item: item[1][3], reverse=True)[:35]:
                    primitive, calls, total, cumulative, _ = value
                    rows.append(dict(file=key[0], line=key[1], function=key[2], calls=calls,
                        primitive_calls=primitive, self_ms=total*1000, cumulative_ms=cumulative*1000))
                _write_json(output/filename.replace('.pstats', '_profile.json'), rows)
            finally:
                profiler.disable()
                session.close()
    _write_json(output/'phase_profiles.json', profile_rows)
    return profile_rows


def memory_worker(args):
    """Separate process: report process RSS and traced update allocation peak."""
    import resource
    import tracemalloc
    capture_path = Path(args.replay).resolve()
    manifest, geometries, records = load_capture(capture_path)
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_baseline(output/'baseline_snapshot', args.baseline_ref)
    variant = args.memory_worker
    prefix = NAMESPACE if variant == 'baseline' else 'robot_skill_stack'
    module = importlib.import_module(prefix+'.world.spatial_mapping.robot_filter')
    meshes = [module.TriangleMesh(*geometry) for geometry in geometries]
    target = records[args.profile_record]
    clock = ReplayClock(max(frame.acquired_at for frame in target['frames']))
    session, adapter = _new_session(variant, manifest, meshes, clock, True)
    adapter.record = target
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    def resident_bytes():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
        return None
    steady_before = resident_bytes()
    try:
        tracemalloc.start()
        assert session.update(target['frames'], now=clock.value)
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = dict(variant=variant, record=target['index'], filtered=True,
            process_peak_rss_before_update_bytes=before_rss,
            process_peak_rss_after_update_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            resident_before_update_bytes=steady_before, resident_after_update_bytes=resident_bytes(),
            update_traced_current_bytes=traced_current, update_traced_peak_bytes=traced_peak,
            owned_frame_buffer_bytes=sum(array.nbytes for record in records for frame in record['frames']
                for array in (frame.depth, frame.rgb, frame.intrinsics, frame.world_from_camera) if array is not None),
            owned_pose_array_bytes=sum(pose['mesh_ids'].nbytes+pose['matrices'].nbytes
                for record in records for pose in record['poses'].values()),
            scope='Fresh SDK-free process per variant; RSS includes all captured inputs, cached geometry/BVH and map. Tracemalloc update is separately instrumented and not timed.')
        _write_json(Path(args.result), result)
    finally:
        session.close()


def replay(args):
    import resource
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError('warmup must be >=0 and repeats >=1')
    capture_path = Path(args.replay).resolve()
    manifest, geometries, records = load_capture(capture_path)
    baseline = export_baseline(output/'baseline_snapshot', args.baseline_ref)
    if baseline['commit'] != manifest['baseline']['commit']:
        raise ValueError('Replay baseline must equal the captured fixture baseline')
    if args.limit:
        records = records[:args.limit]
    start_hashes = _source_hashes()
    provenance = dict(baseline=baseline, source_sha256=start_hashes,
        script_sha256=_sha(Path(__file__).read_bytes()),
        capture_manifest_sha256=_sha((capture_path/'capture_manifest.json').read_bytes()),
        capture_path=str(capture_path),
        capture_annotation_correction='Replay clears map at index12 before moving loop, matching unchanged V2 fixture; initial manifest omitted this annotation. Original manifest retained.',
        reset_indices=[0, 3, 6, 9, 12], numpy_version=np.__version__)
    _write_json(output/'provenance_start.json', provenance)
    result = dict(status='RUNNING', metrics={}, error=None)
    _write_json(Path(args.result), result)
    try:
        mapping = replay_mapping(args, manifest, geometries, records, output)
        result['metrics']['mapping'] = mapping
        _write_json(Path(args.result), result)
        if args.semantic:
            result['metrics']['semantic'] = semantic_and_combined(args, manifest, geometries, records, output)
        if args.profile:
            result['metrics']['phase_profiles'] = profile_mapping(args, manifest, geometries, records, output)
        if args.memory:
            memories = []
            for variant in ('baseline', 'current'):
                path = output/f'memory_{variant}.json'
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--replay', str(capture_path),
                    '--memory-worker', variant, '--profile-record', str(args.profile_record),
                    '--baseline-ref', args.baseline_ref, '--artifacts-dir', str(output/'memory_workers'/variant),
                    '--result', str(path)], cwd=ROOT, check=True)
                memories.append(json.loads(path.read_text()))
            result['metrics']['memory'] = memories
        result['metrics']['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['metrics']['source_unchanged'] = start_hashes == _source_hashes()
        result['metrics']['script_unchanged'] = provenance['script_sha256'] == _sha(Path(__file__).read_bytes())
        result['status'] = 'PASS' if (mapping['all_checks_passed'] and not mapping['runtime_errors'] and
            result['metrics']['source_unchanged'] and result['metrics']['script_unchanged'] and
            result['metrics'].get('semantic', {}).get('timestamps_preserved', True) and
            result['metrics'].get('semantic', {}).get('semantic_outputs_equal', True)) else 'FAIL'
    except Exception as exc:
        result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json(output/'provenance_end.json', dict(source_sha256=_source_hashes(),
            script_sha256=_sha(Path(__file__).read_bytes())))
        _write_json(Path(args.result), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--variants', choices=('both', 'baseline', 'current'), default='both')
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-record', type=int, default=2)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--memory-worker', choices=('baseline', 'current'), help=argparse.SUPPRESS)
    parser.add_argument('--baseline-ref', default=BASELINE)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.capture_only == bool(args.replay):
        parser.error('Specify exactly one of --capture-only or --replay.')
    if args.capture_only:
        capture(args)
    elif args.memory_worker:
        memory_worker(args)
    else:
        result = replay(args)
        if result['status'] != 'PASS':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
