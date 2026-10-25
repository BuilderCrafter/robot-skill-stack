#!/usr/bin/env python3
"""SDK-free sampling research on recorded native RGB-D/robot snapshots.

Production code is never written. The research mapper retains the canonical
validation, masking, traversal, confidence and TTL body, replacing only candidate
pixel locations and selection. Truth labels are consulted after computation.
Offline replay cannot establish live publication age or effective native rate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import textwrap
import time
import tomllib
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.world.spatial_mapping import rgbd, runtime
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractor, ObstacleExtractionSettings
from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyState
from scripts.audit_rgbd_obstacle_evidence import _rays, _roi
from scripts.camera_workload_metrics import distribution


POLICIES = ('fixed1024', 'phase1024', 'cycle1024', 'edge1024', 'fixed2048', 'fixed4096')
PHASES = (.5, .25, .75, 0.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def candidate_pixels(frame, stride, policy, iteration):
    """Cycle four stride offsets only for cycle1024; fixed remains identical."""
    oy = ox = 0
    if policy == 'cycle1024':
        half = stride//2
        oy, ox = ((0, 0), (half, half), (0, half), (half, 0))[iteration % 4]
    y, x = np.indices(frame.depth[oy::stride, ox::stride].shape)
    return y.ravel()*stride+oy, x.ravel()*stride+ox


def selection_indices(frame, y, x, depth, count, policy, iteration):
    """No semantic labels, physical geometry or map state enter this selector."""
    if not count:
        return np.empty(0, int)
    phase = PHASES[iteration % 4] if policy in ('phase1024', 'cycle1024') else .5
    midpoint = ((np.arange(count)+phase)*len(x)/count).astype(int)
    if policy != 'edge1024' or count == len(x):
        return midpoint
    # Retain half the evenly spread samples. The remaining quota selects depth
    # discontinuities deterministically; valid returns alone supply scores.
    uniform = midpoint[::2]
    score = np.zeros(len(x), float)
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (0, -8), (0, 8), (-8, 0), (8, 0)):
        yy, xx = np.clip(y+dy, 0, frame.depth.shape[0]-1), np.clip(x+dx, 0, frame.depth.shape[1]-1)
        neighbor = frame.depth[yy, xx]
        valid = np.isfinite(neighbor) & (neighbor > 0)
        delta = np.where(valid, abs(neighbor-depth)/np.maximum(depth, 1e-6), 0.)
        score = np.maximum(score, delta)
    order = np.lexsort((x, y, -score))
    chosen = np.zeros(len(x), bool)
    chosen[uniform] = True
    extra = order[~chosen[order]][:count-len(uniform)]
    # Canonical integration does not depend on order; sorting keeps mask calls
    # and diagnostic output deterministic for equivalent candidate evidence.
    return np.sort(np.r_[uniform, extra])


def _research_update():
    source = textwrap.dedent(inspect.getsource(rgbd.RgbdMapper.update))
    grid = ('y, x = np.indices(frame.depth[::self.stride, ::self.stride].shape)\n'
            '            y, x = y.ravel()*self.stride, x.ravel()*self.stride')
    selected = 'selected = ((np.arange(count)+.5)*len(x)/count).astype(int) if count else np.empty(0, int)'
    if source.count(grid) != 1 or source.count(selected) != 1:
        raise RuntimeError('Canonical sampler body changed; review research hooks before replay')
    changed = source.replace(grid,
        'y, x = candidate_pixels(frame, self.stride, self.policy, self.iteration)').replace(selected,
        'selected = selection_indices(by_id[key], y, x, depth, count, self.policy, self.iteration)')
    namespace = dict(vars(rgbd), candidate_pixels=candidate_pixels, selection_indices=selection_indices)
    exec(compile(changed, '<isolated camera sampling research>', 'exec'), namespace)
    return namespace['update']


RESEARCH_UPDATE = _research_update()


class ResearchMapper(rgbd.RgbdMapper):
    def __init__(self, *args, policy='fixed1024', **kwargs):
        if policy not in POLICIES:
            raise ValueError('Unknown sampling policy: '+policy)
        super().__init__(*args, **kwargs)
        self.policy, self.iteration = policy, -1

    def update(self, *args, **kwargs):
        self.iteration += 1
        prior = getattr(self.clock, 'phase', None)
        if prior is not None:
            self.clock.phase = 'rgbd_update'
        try:
            return RESEARCH_UPDATE(self, *args, **kwargs)
        finally:
            if prior is not None:
                self.clock.phase = prior


class ResearchMappingProcessor(runtime.MappingProcessor):
    """Explicit SDK-free prototype usable by an isolated native ProcessWorker.

    Construct this only through research scripts. The ordinary MappingService
    continues constructing its unchanged production MappingProcessor.
    """
    def __init__(self, settings, *, policy='depth_edge', budget=None,
                 clock=time.monotonic, obstacle_settings=None, allow_cycle_offsets=False):
        if budget is not None:
            settings = replace(settings, max_rays=budget)
        aliases = dict(depth_edge='edge1024', fixed='fixed'+str(settings.max_rays),
                       phase_cycle='phase1024', cycle_stride='cycle1024')
        policy = aliases.get(policy, policy)
        if policy not in POLICIES:
            raise ValueError('Unknown research sampling policy: '+policy)
        expected = int(policy.removeprefix('fixed')) if policy.startswith('fixed') else 1024
        if settings.max_rays != expected:
            raise ValueError('Sampling policy label must match its configured total ray budget')
        self.policy, self.allow_cycle_offsets = policy, bool(allow_cycle_offsets)
        super().__init__(settings, clock=clock, obstacle_settings=obstacle_settings)

    def _reset(self, generation):
        with patch.object(runtime, 'RgbdMapper', partial(ResearchMapper, policy=self.policy)):
            super()._reset(generation)
        if self.policy == 'cycle1024' and self.allow_cycle_offsets:
            # Explicit pixels are still supplied on every mask call. Stride1
            # permits those coordinates; it does not request a dense mask cast.
            self.filter.stride = 1


def normalize_object(value):
    value = dict(value)
    value['name'] = value.get('name', value.get('id'))
    if not value['name']:
        raise ValueError('Evaluation shape requires a name/id')
    kind = value.get('kind', value.get('shape', 'box'))
    value['kind'] = 'box' if kind == 'cube' else kind
    if 'dimensions' not in value:
        if kind == 'sphere':
            value['dimensions'] = [2*value['radius']]*3
        elif kind == 'cylinder':
            value['dimensions'] = [2*value['radius'], 2*value['radius'], value['height']]
        else:
            raise ValueError('Evaluation box requires dimensions')
    return value


def aabb_reference(value):
    """AABB ROI counts are distinct from exact primitive depth matching."""
    result = dict(value)
    angle = float(value.get('yaw', 0.))
    if value['kind'] == 'box' and angle:
        x, y, z = value['dimensions']
        result['dimensions'] = [abs(np.cos(angle))*x+abs(np.sin(angle))*y,
                                abs(np.sin(angle))*x+abs(np.cos(angle))*y, z]
    return result


def physical_matches(frame, y, x, shape, tolerance=.015):
    from scripts.camera_observability_geometry import shape_depth
    near = shape_depth(shape, frame.world_from_camera[:3, 3], _rays(frame, y, x))
    measured = frame.depth[y, x]
    return (np.isfinite(near) & (near > 0) & np.isfinite(measured) & (measured > 0)
            & (abs(measured-near) <= tolerance))


class ReplayClock:
    """Retain recorded phase clocks, or use a declared logical input age."""
    def __init__(self):
        self.phase, self._values, self._reads = 'processor', {}, defaultdict(int)

    def load(self, row, frames):
        self._reads = defaultdict(int)
        self._values = defaultdict(list)
        for record in row.get('clock_reads') or []:
            self._values[record['phase']].append(float(record['value']))
        if not self._values:
            now = max(frame.acquired_at for frame in frames)+.001
            self._values = {'rgbd_update': [now, now], 'processor': [now]}

    def __call__(self):
        at = self._reads[self.phase]
        self._reads[self.phase] += 1
        values = self._values[self.phase]
        if at >= len(values):
            raise RuntimeError('Replay exceeds captured clock reads in '+self.phase)
        return values[at]

    @property
    def evaluated_at(self):
        return self._values['processor'][-1]


def load_case(folder, maximum):
    """Load a legacy audit directory or a new research capture manifest."""
    folder = Path(folder)
    manifest_path = folder if folder.is_file() else folder/'sampling_manifest.json'
    if not manifest_path.exists() and (folder/'sampling_capture.json').exists():
        manifest_path = folder/'sampling_capture.json'
    if manifest_path.exists():
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        cfg = manifest.get('spatial_mapping', manifest.get('config', {}).get('spatial_mapping'))
        if cfg is None:
            raise ValueError('Sampling manifest needs spatial_mapping settings')
        objects = manifest.get('evaluation_objects', [])
        rows = manifest.get('records', manifest.get('entries'))
        if rows is None:
            raise ValueError('Sampling manifest needs records/entries')
        base = folder
        label = manifest['capture_layout']
        source_kind = 'research_native_capture'
    else:
        metrics = json.loads((folder/'result.json').read_text())['metrics']
        cfg = tomllib.loads((folder/'runtime_profile.toml').read_text())['spatial_mapping']
        objects = metrics['configuration']['objects']
        base = folder/'mapping-audit'
        rows = [json.loads(line) for line in (base/'jobs.jsonl').read_text().splitlines()]
        placement = metrics.get('fixture_placement_started_at')
        measurement = metrics.get('measurement_started_at')
        rows = [row for row in rows if (placement is None or row['input_timestamp'] >= placement)
                and (measurement is None or row['input_timestamp'] <= measurement)]
        label = 'V2b_reference_fixedposes/'+metrics['configuration'].get('scenario', folder.name)
        source_kind = 'historical_v2b_diagnostic_capture'
    rows = [row for row in rows if row.get('input_pickle')]
    rows = rows[-maximum:] if maximum else rows
    if not rows:
        raise ValueError('No captured inputs in '+str(folder))
    if any(obj.get('moving') for obj in objects):
        raise ValueError('Moving references need exposure-matched geometry per record')
    return dict(folder=folder, base=base, label=label, source_kind=source_kind,
                settings=SpatialMappingConfig(**cfg), objects=[normalize_object(obj) for obj in objects], rows=rows)


def _represented(packet, spec):
    result = np.zeros(spec.shape, bool)
    for box in packet.boxes:
        b = tuple(map(int, box.id.split(':')))
        result[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = True
    return result


def _stage_snapshot(snapshot, packet, objects, physical_cells, spec):
    states = snapshot.states
    eligible = ((states == OccupancyState.OCCUPIED) & snapshot.observed & ~snapshot.expired)
    selected = _represented(packet, spec)
    if np.any(selected & ~eligible):
        raise AssertionError('Planning adapter invented FREE/UNKNOWN/expired geometry')
    object_rows = {}
    for obj in objects:
        name = obj['name']
        _, _, _, flat = _roi(spec, aabb_reference(obj))
        represented = selected.ravel()[flat]
        occupied = eligible.ravel()[flat]
        ids = np.asarray(sorted(physical_cells[name]), int)
        object_rows[name] = dict(aabb_intersecting_voxels=len(flat),
            occupied_roi_voxels=int(occupied.sum()), represented_roi_voxels=int(represented.sum()),
            current_or_previous_physical_hit_cells=len(ids),
            hit_cells_currently_occupied=int(eligible.ravel()[ids].sum()),
            hit_cells_represented=int(selected.ravel()[ids].sum()),
            occupied_probability=distribution(snapshot.probabilities.ravel()[flat][occupied]))
        current_hits = object_rows[name]['hit_cells_currently_occupied']
        represented_hits = object_rows[name]['hit_cells_represented']
        object_rows[name]['occupied_physical_hit_discarded_fraction'] = (
            None if not current_hits else 1-represented_hits/current_hits)
    occupied_count = int(eligible.sum())
    represented_count = int(selected.sum())
    return dict(states={name: int(np.count_nonzero(states == value)) for name, value in
        (('unknown', 0), ('free', 1), ('occupied', 2))}, objects=object_rows,
        extraction={k: v for k, v in packet.diagnostics.items() if k != 'extract_seconds'},
        occupied_evidence_discarded_fraction=None if not occupied_count else
            1-represented_count/occupied_count,
        false_free_or_unknown_or_expired_cells=0)


def replay(case, policies, output, *, allow_cycle_offsets=False):
    settings, spec = case['settings'], case['settings'].grid_spec()
    processors, clocks, stats = {}, {}, {}
    for policy in policies:
        budget = int(policy.removeprefix('fixed')) if policy.startswith('fixed') else 1024
        cfg = replace(settings, max_rays=budget)
        clock = ReplayClock()
        processors[policy] = ResearchMappingProcessor(cfg, clock=clock,
            policy=policy, allow_cycle_offsets=allow_cycle_offsets)
        clocks[policy] = clock
        stats[policy] = dict(rows=[], selected={}, free=np.zeros(spec.shape, bool),
            occupied=np.zeros(spec.shape, bool), physical_cells=defaultdict(set),
            physical_pixels=defaultdict(set), selected_counts=defaultdict(int),
            final=None, stages=defaultdict(list), total_retained=0, total_allocated=0)
    provenance = []
    calibration = {}
    for index, row in enumerate(case['rows']):
        path = case['base']/row['input_pickle']
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        if payload.reset:
            raise ValueError('Replay requires non-reset captured inputs; start condition is explicit UNKNOWN')
        provenance.append(dict(input_timestamp=min(f.acquired_at for f in payload.frames),
            path=str(path), sha256=sha(path), recorded_clock_reads=row.get('clock_reads')))
        truth = {}
        for frame in payload.frames:
            key = frame.camera_id
            current = dict(intrinsics=frame.intrinsics.tolist(), world_from_camera=frame.world_from_camera.tolist(),
                           resolution=[frame.depth.shape[1], frame.depth.shape[0]])
            if key in calibration and calibration[key] != current:
                raise ValueError('Camera pose/calibration changed within fixed-layout sampling capture')
            calibration[key] = current
            y, x = np.indices(frame.depth.shape)
            truth[key] = {obj['name']: physical_matches(frame, y.ravel(), x.ravel(), obj).reshape(frame.depth.shape)
                          for obj in case['objects']}
        for policy in policies:
            processor, clock, record = processors[policy], clocks[policy], stats[policy]
            clock.load(row, payload.frames)
            if processor.map is None:
                processor._reset(payload.generation)
            if processor.generation != payload.generation:
                raise ValueError('Mixed map generations in sampling capture')
            selection = {}
            integration = {}
            original_masks = processor.filter.masks
            original_integrate = processor.map.integrate
            original_traverse = rgbd._traverse_rays
            times = defaultdict(float)

            def masks(frames, snapshots, *, pixels_by_camera=None):
                before = time.perf_counter()
                result = original_masks(frames, snapshots, pixels_by_camera=pixels_by_camera)
                times['robot_filter'] += time.perf_counter()-before
                selection.update({f.camera_id: (pixels_by_camera[f.camera_id], result[f.camera_id]) for f in frames})
                return result

            def integrate(free, occupied, **kwargs):
                before = time.perf_counter()
                result = original_integrate(free, occupied, **kwargs)
                times['occupancy_integrate'] += time.perf_counter()-before
                integration.update(free=free, occupied=occupied)
                return result

            def traverse(*args, **kwargs):
                before = time.perf_counter()
                result = original_traverse(*args, **kwargs)
                times['ray_traversal'] += time.perf_counter()-before
                return result

            with patch.object(processor.filter, 'masks', masks), patch.object(processor.map, 'integrate', integrate), \
                    patch.dict(RESEARCH_UPDATE.__globals__, {'_traverse_rays': traverse}):
                before = time.perf_counter()
                completed = processor.compute(payload)
                times['mapping_worker'] = time.perf_counter()-before
            before = time.perf_counter()
            packet = ObstacleExtractor(ObstacleExtractionSettings(64, 0.), temporal_probabilities=True).extract(
                completed.snapshot, now=clock.evaluated_at, evidence_ttl=settings.evidence_ttl)
            times['obstacle_extract'] = time.perf_counter()-before
            times['worker_plus_proxy'] = times['mapping_worker']+times['obstacle_extract']
            for key, value in integration.items():
                if len(value):
                    record[key][tuple(value.T)] = True
            object_hits = defaultdict(lambda: dict(raw=0, selected=0, retained=0, masked=0))
            for frame in payload.frames:
                key = frame.camera_id
                if key not in record['selected']:
                    record['selected'][key] = np.zeros(frame.depth.shape, np.uint16)
                if key not in selection:
                    continue
                (y, x), mask = selection[key]
                keep = ~mask[y, x]
                record['selected'][key][y, x] += 1
                record['total_allocated'] += len(y)
                record['total_retained'] += int(keep.sum())
                points = _rays(frame, y, x)*frame.depth[y, x, None]+frame.world_from_camera[:3, 3]
                cells = spec.world_to_indices(points)
                valid_cell = (cells >= 0).all(1)
                for obj in case['objects']:
                    name = obj['name']
                    hit = truth[key][name][y, x]
                    object_hits[name]['raw'] += int(truth[key][name].sum())
                    object_hits[name]['selected'] += int(hit.sum())
                    object_hits[name]['retained'] += int((hit & keep).sum())
                    object_hits[name]['masked'] += int((hit & ~keep).sum())
                    record['physical_pixels'][name].update((key, int(yy), int(xx)) for yy, xx in zip(y[hit & keep], x[hit & keep]))
                    valid = hit & keep & valid_cell
                    if valid.any():
                        record['physical_cells'][name].update(np.ravel_multi_index(cells[valid].T, spec.shape).tolist())
            final = _stage_snapshot(completed.snapshot, packet, case['objects'], record['physical_cells'], spec)
            record['final'] = final
            for key, value in times.items():
                record['stages'][key].append(value*1000)
            record['rows'].append(dict(index=index, original_source_timestamp=completed.input_timestamp,
                accepted=completed.accepted, diagnostics=plain(completed.diagnostics),
                logical_evaluated_at=clock.evaluated_at, object_ray_hits=dict(object_hits),
                extraction=final['extraction'], stages_ms={k: v*1000 for k, v in times.items()}))
        print(f"{case['label']}: replayed{index+1}/{len(case['rows'])} inputs", flush=True)
    summaries = {}
    output.mkdir(parents=True, exist_ok=True)
    for policy, record in stats.items():
        summary = dict(final=record['final'],
            accepted_updates=sum(row['accepted'] for row in record['rows']), attempted_updates=len(record['rows']),
            unique_depth_pixels=sum(int(np.count_nonzero(values)) for values in record['selected'].values()),
            depth_pixel_revisit_counts=distribution(np.concatenate([values[values > 0] for values in record['selected'].values()])),
            unique_free_voxel_evidence=int(record['free'].sum()),
            unique_occupied_voxel_evidence=int(record['occupied'].sum()),
            allocated_ray_observations=record['total_allocated'], retained_ray_observations=record['total_retained'],
            discarded_self_filter_fraction=None if not record['total_allocated'] else
                1-record['total_retained']/record['total_allocated'],
            unique_physically_matching_pixels={k: len(v) for k, v in record['physical_pixels'].items()},
            stages_ms={key: distribution(values) for key, values in record['stages'].items()},
            raw_depth_hits_last_input=record['rows'][-1]['object_ray_hits'],
            publication_age='NOT_MEASURED_OFFLINE', effective_native_update_hz='NOT_MEASURED_OFFLINE')
        summaries[policy] = summary
        (output/(policy+'.json')).write_text(json.dumps(record['rows'], indent=2, allow_nan=False)+'\n')
        np.savez_compressed(output/(policy+'-evidence.npz'), unique_free=record['free'],
                            unique_hit=record['occupied'])
    return dict(capture_layout=case['label'], source_kind=case['source_kind'],
        input_count=len(case['rows']), input_files=provenance, actual_camera_calibration=calibration,
        initial_condition='All policies start with UNKNOWN in independent fresh maps at the first selected original input; omitted earlier capture history is not recreated.',
        policy_results=summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', action='append', required=True)
    parser.add_argument('--max-jobs', type=int, default=12)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    parser.add_argument('--allow-cycle-offsets', action='store_true',
        help='Explicit research-only stride1 pixel validation for off-grid cycle samples; ray budget unchanged')
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    if args.max_jobs < 0:
        parser.error('max-jobs must be nonnegative; zero means all eligible inputs')
    protected = [ROOT/p for p in ('robot_skill_stack/world/spatial_mapping/rgbd.py',
        'robot_skill_stack/world/spatial_mapping/occupancy.py', 'robot_skill_stack/world/spatial_mapping/robot_filter.py',
        'robot_skill_stack/world/spatial_mapping/obstacles.py', 'robot_skill_stack/world/spatial_mapping/runtime.py')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    result = dict(schema_version=1, status='RESEARCH_OFFLINE_ONLY', sampler_uses_evaluation_truth=False,
        source_hashes=before, script_sha256=sha(__file__),
        protocol=dict(policies=args.policies, fixed_budget_strategy_comparison=1024,
            separate_budget_experiments=[2048, 4096], production_stride=8, cycle_offsets=[[0, 0], [4, 4], [0, 4], [4, 0]],
            selection_phases=PHASES, adaptive='Half uniform midpoint samples plus deterministically ranked valid-depth discontinuities; no truth labels or semantic regions.',
            cycle_filter_validation_stride=1 if args.allow_cycle_offsets else 'production_stride',
            timing='Measured offline CPU computation using current exact self-filter and canonical mapper; small timing hooks included. Input loading, evaluation truth rasterization and output I/O excluded.',
            geometry='Exact evaluation box/sphere/Z-cylinder intersections label matched native optical depth and endpoint cells after sampling/masking only; AABBs label ROI counts separately. No truth enters selection or integration.',
            source_age='Original captured phase clocks retained where present. Otherwise explicit logical replay age1ms; no native publication/freshness/rate inference.',
            unknown='UNKNOWN is neither obstruction nor collision. Adapter output is verified to contain only original observed OCCUPIED, unexpired cells.'), cases=[])
    for folder in args.capture_dir:
        case = load_case(folder, args.max_jobs)
        name = case['folder'].name
        result['cases'].append(replay(case, args.policies, Path(args.artifacts_dir)/name,
            allow_cycle_offsets=args.allow_cycle_offsets))
    result['production_sources_unchanged'] = before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    if not result['production_sources_unchanged']:
        raise RuntimeError('Production sources changed during sampling research')
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], result=str(path), cases=len(result['cases']))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
