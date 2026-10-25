#!/usr/bin/env python3
"""Isolated V3 representations in the unchanged native camera motion fixture.

Run one case per fresh launcher process. Ordinary mapping/self-filtering stays
in its subprocess; numeric registration remains on the Kit owner. Candidates
activate only after the common production-extractor setup pose is recorded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from functools import partial
import hashlib
import importlib
import json
import math
from pathlib import Path
import threading
import time
import traceback
from unittest.mock import patch

import numpy as np

from robot_skill_stack.world.spatial_mapping.geometry import freeze
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractor, ObstaclePacket
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor
from tests.support.mapping_evidence_audit import AuditedMappingProcessor


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'scripts.benchmark_obstacle_representation_v3'
START = (.45, -.25, .35)
GOAL = (.45, .25, .35)
PACKINGS = ('baseline', 'axis_xyz', 'axis_xzy', 'axis_yxz', 'axis_yzx', 'axis_zxy',
            'axis_zyx', 'axis_best', 'greedy', 'component')
SELECTIONS = ('baseline', 'coverage', 'diversity', 'motion')


@dataclass(frozen=True)
class RepresentationSpec:
    packing: str = 'baseline'
    selection: str = 'baseline'
    budget: int = 64
    candidate_id: str = 'baseline'

    def __post_init__(self):
        if self.packing not in PACKINGS or self.selection not in SELECTIONS:
            raise ValueError('Unknown isolated packing/selection policy')
        if self.budget not in (16, 32, 64, 128):
            raise ValueError('Research budgets are16,32,64or128')
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError('An explicit representation ID is required')


@dataclass(frozen=True)
class RepresentationInput(MappingInput):
    activate_representation: bool = False
    capture_selected_phase: bool = False
    public_goal: tuple = GOAL
    public_corridor: tuple = (START, GOAL)


def shared_input(payload, *, active, capture=False):
    """Rewrap already-owned buffers; never scan/copy images on the owner."""
    cls = importlib.import_module(MODULE).RepresentationInput
    values = {field.name: getattr(payload, field.name) for field in fields(MappingInput)}
    return cls(**values, activate_representation=bool(active), capture_selected_phase=bool(capture))


def policy_metadata(processor):
    spec = processor.representation_spec
    return dict(candidate_id=spec.candidate_id, active=processor._representation_active,
        packing=spec.packing if processor._representation_active else 'baseline',
        selection=spec.selection if processor._representation_active else 'baseline',
        budget=spec.budget if processor._representation_active else 64,
        context='Exposure-matched robot snapshot origins plus public goal/START→GOAL corridor; no fixture labels')


class _TaggedExtractor:
    def __init__(self, extractor, metadata):
        self.extractor, self.metadata = extractor, freeze(metadata)

    def extract(self, snapshot, *, now, evidence_ttl):
        packet = self.extractor.extract(snapshot, now=now, evidence_ttl=evidence_ttl)
        return replace(packet, diagnostics={**dict(packet.diagnostics), 'representation_v3': self.metadata})


def configure_extractor(processor, payload):
    if processor._obstacle_settings is None:
        return
    spec = processor.representation_spec
    active = bool(getattr(payload, 'activate_representation', False))
    processor._representation_active = active
    settings = replace(processor._obstacle_settings, max_obstacles=spec.budget if active else 64)
    context = None
    if active and spec.selection == 'motion':
        from scripts.obstacle_representation_v3 import public_motion_context
        context = public_motion_context(getattr(payload, 'robot_snapshots', None),
            goal=tuple(payload.public_goal), corridor=tuple(payload.public_corridor))
    key = (active, settings, spec, context)
    if key == processor._representation_key:
        return
    if not active or spec.packing == 'baseline' and spec.selection == 'baseline':
        extractor = ObstacleExtractor(settings, temporal_probabilities=processor.settings.source == 'rgbd')
    else:
        from scripts.obstacle_representation_v3 import ExperimentalObstacleExtractor
        extractor = ExperimentalObstacleExtractor(settings, temporal_probabilities=processor.settings.source == 'rgbd',
            packing=spec.packing, selection=spec.selection, motion_context=context)
    processor._extractor = _TaggedExtractor(extractor, policy_metadata(processor))
    processor._representation_key = key


def tag_completed(processor, completed):
    return replace(completed, diagnostics=freeze({**dict(completed.diagnostics),
        'representation_v3': policy_metadata(processor)}))


class RepresentationMappingProcessor(MappingProcessor):
    def __init__(self, settings, *, representation_spec=RepresentationSpec(), clock=time.monotonic,
                 obstacle_settings=None):
        self.representation_spec = representation_spec
        self._representation_active, self._representation_key, self._representation_payload = False, None, None
        super().__init__(settings, clock=clock, obstacle_settings=obstacle_settings)

    def _reset(self, generation):
        super()._reset(generation)
        self._representation_key = None
        configure_extractor(self, self._representation_payload)

    def compute(self, payload):
        self._representation_payload = payload
        self._representation_active = bool(getattr(payload, 'activate_representation', False))
        try:
            if self.map is not None and self.generation == payload.generation and not payload.reset:
                configure_extractor(self, payload)
            return tag_completed(self, super().compute(payload))
        finally:
            self._representation_payload = None


class AuditedRepresentationMappingProcessor(AuditedMappingProcessor):
    """Full forensic captures retain the existing exact-input audit schema."""
    def __init__(self, settings, *, representation_spec=RepresentationSpec(), **kwargs):
        self.representation_spec = representation_spec
        self._representation_active, self._representation_key, self._representation_payload = False, None, None
        self._capture_enabled, self._capture_count = False, 0
        super().__init__(settings, **kwargs)

    def _reset(self, generation):
        def reset():
            MappingProcessor._reset(self, generation)
            self._representation_key = None
            configure_extractor(self, self._representation_payload)
        self._time('processor_reset', reset)
        self._install_hooks()

    def _full_capture(self):
        return self.capture_inputs and self._capture_enabled and self._capture_count < self.max_captures

    def _persist(self, payload, completed):
        captured = self._full_capture()
        super()._persist(payload, completed)
        self._capture_count += int(captured)

    def compute(self, payload):
        self._representation_payload = payload
        self._representation_active = bool(getattr(payload, 'activate_representation', False))
        self._capture_enabled = bool(getattr(payload, 'capture_selected_phase', False))
        try:
            if self.map is not None and self.generation == payload.generation and not payload.reset:
                configure_extractor(self, payload)
            return tag_completed(self, super().compute(payload))
        finally:
            self._representation_payload = None


def progress_windows(samples, *, window_seconds=1., minimum_progress_m=.005):
    """Descriptive stationary/goal-progress windows, not a local-minimum proof."""
    if len(samples) < 2:
        return dict(samples=len(samples), qualified_windows=0, longest_low_progress_seconds=None)
    times = np.asarray([row['at'] for row in samples], float)
    points = np.asarray([row['position'] for row in samples], float)
    errors = np.linalg.norm(points-np.asarray(GOAL), axis=1)
    first, qualified, low, stationary, longest, run_start = 0, 0, 0, 0, 0., None
    for last in range(1, len(samples)):
        while first+1 < last and times[first+1] <= times[last]-window_seconds:
            first += 1
        if times[last]-times[first] < window_seconds:
            continue
        qualified += 1
        is_low = errors[first]-errors[last] < minimum_progress_m
        low += int(is_low)
        stationary += int(np.linalg.norm(points[last]-points[first]) < minimum_progress_m)
        run_start = times[last] if is_low and run_start is None else run_start if is_low else None
        if run_start is not None:
            longest = max(longest, times[last]-run_start)
    return dict(samples=len(samples), qualified_windows=qualified, low_goal_progress_windows=low,
        stationary_windows=stationary, longest_low_progress_seconds=longest,
        window_seconds=window_seconds, minimum_progress_m=minimum_progress_m,
        meaning='Overlapping trailing windows; detours and orientation correction can also have low goal-distance progress.')


def calibration_signature(calibration):
    if not isinstance(calibration, dict):
        return None
    required = ('intrinsics', 'world_from_camera_ros', 'resolution')
    if any(not isinstance(row, dict) or any(key not in row for key in required) for row in calibration.values()):
        return None
    return {camera: {key: row[key] for key in required} for camera, row in calibration.items()}


def reference_match(reference, metrics, actual_q, *, tolerance, runtime_protocol=None):
    old = reference.get('metrics') or {}
    expected = np.asarray(old.get('initial_joints', []), float)
    actual = np.asarray(actual_q, float)
    equal_shape = expected.shape == actual.shape and expected.ndim == 1 and len(expected) >= 7
    maximum = float(np.max(np.abs(expected-actual))) if equal_shape else None
    first, second = old.get('configuration') or {}, metrics.get('configuration') or {}
    checks = dict(joint_positions=maximum is not None and maximum <= tolerance,
        start=first.get('start') == second.get('start'), goal=first.get('goal') == second.get('goal'),
        orientation=first.get('orientation') == second.get('orientation'),
        physical_objects=first.get('objects') == second.get('objects'),
        mapping_hz=first.get('mapping_hz') == second.get('mapping_hz'),
        voxel_size=first.get('voxel_size') == second.get('voxel_size'))
    old_pose = (old.get('fixture_setup') or {}).get('actual_actor_poses', [])
    new_pose = (metrics.get('fixture_setup') or {}).get('actual_actor_poses', [])
    old_pose, new_pose = np.asarray(old_pose, float), np.asarray(new_pose, float)
    checks['physical_fixture_poses'] = old_pose.shape == new_pose.shape and bool(
        not old_pose.size or np.max(np.abs(old_pose-new_pose)) <= 1e-4)
    if runtime_protocol is not None:
        research, representation = reference.get('research') or {}, reference.get('representation_v3') or {}
        checks.update(layout_id=research.get('layout_id') == runtime_protocol['layout_id'],
            global_ray_budget=research.get('global_ray_budget') == runtime_protocol['global_ray_budget'],
            sampling_policy=research.get('sampling_policy') == runtime_protocol['sampling_policy'],
            backend_settings=old.get('backend_settings') == runtime_protocol['backend_settings'],
            production_source_manifest=bool(reference.get('hashes')) and reference['hashes'] == runtime_protocol['source_hashes'],
            original_harness_sha256=representation.get('original_harness_sha256') == runtime_protocol['original_harness_sha256'],
            wrapper_sha256=representation.get('wrapper_sha256') == runtime_protocol['wrapper_sha256'])
        expected_calibration = calibration_signature(research.get('actual_camera_calibration'))
        actual_calibration = calibration_signature(runtime_protocol['actual_camera_calibration'])
        equal_cameras = bool(expected_calibration) and bool(actual_calibration) and expected_calibration.keys() == actual_calibration.keys()
        checks['actual_camera_calibration'] = bool(equal_cameras and all(
            np.asarray(expected_calibration[camera][key]).shape == np.asarray(actual_calibration[camera][key]).shape
            and np.allclose(expected_calibration[camera][key], actual_calibration[camera][key], atol=1e-5, rtol=0.)
            for camera in expected_calibration for key in ('intrinsics', 'world_from_camera_ros', 'resolution')))
    return dict(checks=checks, matched=all(checks.values()), joint_tolerance_rad=tolerance,
        maximum_joint_error_rad=maximum, reference_q=expected.tolist(), actual_q=actual.tolist())


def run(args, app, layout):
    from scripts import camera_observability_planner as camera
    from scripts import diagnose_reactive_motion as diagnostic
    from robot_skill_stack.integrations.isaac import bootstrap
    from robot_skill_stack.integrations.isaac.manipulation.franka_backend import IsaacFrankaBackend as FrankaBackend
    from robot_skill_stack.integrations.isaac.manipulation.motion.obstacles import RmpflowObstacleRegistry
    from robot_skill_stack.world.spatial_mapping import runtime
    from scripts.camera_workload_config import _set_table_values

    module = importlib.import_module(MODULE)  # Stable child-pickle module, not __main__.
    spec = module.RepresentationSpec(args.packing, args.selection, args.max_obstacles, args.candidate_id)
    state = dict(result=None, bundle=None, activation_at=None, initial_checked=False)
    rows = []
    original_submit, original_native = runtime.MappingService.submit, diagnostic.run_native
    original_profile, original_finish, original_fixtures = diagnostic.profile, diagnostic.finish, diagnostic.fixtures
    original_build, original_move = bootstrap.build_runtime, FrankaBackend.move_to_pose_async
    original_sync = RmpflowObstacleRegistry.sync
    reference = None if not args.reference_result else json.loads(Path(args.reference_result).read_text())
    source_before = diagnostic.source_hashes()

    def fixtures(name):
        return (diagnostic.Box('ThinCylinder', (.45, 0., .35), (.035, .035, .20)),) if name == 'cylinder' else original_fixtures(name)

    def profile(value):
        path = original_profile(value)
        path.write_text(_set_table_values(path.read_text(), 'motion', dict(max_obstacles=max(64, args.max_obstacles))))
        if args.scenario == 'cylinder':
            import omni.usd
            from pxr import Gf, UsdGeom, UsdPhysics
            from scripts.obstacle_representation_v3_cylinder import unit_prism
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(diagnostic.FIXTURE_ROOT+'/ThinCylinder')
            prim.SetTypeName('Mesh')
            mesh = UsdGeom.Mesh(prim)
            vertices, counts, indices = unit_prism(32)
            vertices *= np.array([.035, .035, .20])
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
            mesh.CreateFaceVertexCountsAttr(counts.tolist())
            mesh.CreateFaceVertexIndicesAttr(indices.tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(1., 1., 1.))
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        return path

    def native(value, application, result):
        state['result'] = result
        return original_native(value, application, result)

    async def build(path):
        state['bundle'] = await original_build(path)
        return state['bundle']

    def submit(service, payload):
        result = state['result'] or {}
        active = 'setup_joints_before_obstacles' in result
        if active and state['activation_at'] is None:
            state['activation_at'] = time.monotonic()
        capture = active and (args.capture_phase == 'post_setup' or
            'measurement_started_at' in result and 'motion' not in result)
        return original_submit(service, module.shared_input(payload, active=active, capture=capture))

    async def move(backend, target, *positional, **keywords):
        result = state['result'] or {}
        if 'initial_joints' in result and not state['initial_checked']:
            service = state['bundle'].spatial_mapping
            metadata = dict(service.diagnostics).get('representation_v3') or {}
            packet = service.snapshot().obstacles
            packet_meta = {} if packet is None else dict(packet.diagnostics).get('representation_v3', {})
            guard = dict(candidate_active=metadata.get('active') is True,
                expected_candidate=metadata.get('candidate_id') == args.candidate_id,
                packet_candidate=(args.strategy != 'reactive' or packet_meta.get('candidate_id') == args.candidate_id
                    and packet_meta.get('active') is True),
                new_original_source=service.health.get('input_timestamp', 0) >= (state['activation_at'] or math.inf))
            result['representation_activation_guard'] = guard
            if not all(guard.values()):
                raise RuntimeError('Fresh candidate packet was not ready before motion: '+str(guard))
            if reference is not None:
                cached = state['bundle'].observation_runtime.pool._cache
                frames = [value[1] for value in cached.values() if value[1] is not None]
                protocol = dict(layout_id=layout['layout_id'], global_ray_budget=args.max_rays,
                    sampling_policy=args.sampling_policy, source_hashes=source_before,
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    original_harness_sha256=hashlib.sha256((ROOT/'scripts/diagnose_reactive_motion.py').read_bytes()).hexdigest(),
                    backend_settings=dict(position_tolerance=backend.position_tolerance,
                        orientation_tolerance=backend.orientation_tolerance, max_motion_steps=backend.max_motion_steps),
                    actual_camera_calibration={frame.camera_id:dict(intrinsics=frame.intrinsics,
                        world_from_camera_ros=frame.world_from_camera,
                        resolution=[frame.depth.shape[1], frame.depth.shape[0]]) for frame in frames})
                check = reference_match(reference, result, backend.robot.get_joint_positions(),
                    tolerance=args.initial_joint_tolerance, runtime_protocol=protocol)
                result['initial_condition_match'] = check
                if not check['matched']:
                    raise RuntimeError('Initial configuration/fixture differs from declared reference')
            state['initial_checked'] = True
        return await original_move(backend, target, *positional, **keywords)

    def sync(registry, packet, *, now=None):
        keys = ('additions', 'removals', 'pose_updates', 'dimension_updates', 'identity_reuses', 'world_updates', 'skipped')
        before = dict(registry._counts)
        started = time.perf_counter()
        changed = original_sync(registry, packet, now=now)
        seconds = time.perf_counter()-started
        result = state['result'] or {}
        rows.append(dict(at=time.monotonic(), phase='post_motion' if 'motion' in result else
            'motion' if 'measurement_started_at' in result else
            'settling' if 'setup_joints_before_obstacles' in result else 'setup', seconds=seconds,
            generation=packet.generation, revision=packet.revision, original_timestamp=packet.timestamp,
            source_age_seconds=None if now is None else now-packet.timestamp,
            extracted_at=packet.extracted_at, packet_proxy_count=len(packet.boxes),
            registered_proxy_count=len(registry._proxies), changed=bool(changed),
            owner_thread=threading.get_ident() == registry._owner,
            operations={key: registry._counts[key]-before[key] for key in keys}))
        return changed

    def finish(value, result, capture):
        if args.scenario != 'cylinder' or capture is None:
            return original_finish(value, result, capture)
        import omni.usd
        from pxr import UsdGeom
        from scripts.obstacle_representation_v3_cylinder import (
            reduce_cylinder_clearances, depth_reference_scores_convex)
        path = diagnostic.FIXTURE_ROOT+'/ThinCylinder'
        stage = omni.usd.get_context().get_stage()
        before = tuple(str(prim.GetPath()) for prim in stage.Traverse())
        copied = diagnostic.cooked_reference(stage, stage.GetPrimAtPath(path))
        if not copied['hulls'] or not copied['owner_thread']:
            raise RuntimeError('Actual cooked cylinder reference unavailable; no bounding-box fallback permitted')
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        hulls = [(vertices*units, counts, indices) for vertices, counts, indices in copied['hulls']]
        actual = capture['pose_matrices'](capture['raw']['samples'][-1]['poses'], capture['units'])[
            capture['body_paths'].index(path)]
        def reduce(meshes, body_paths, poses, boxes):
            return reduce_cylinder_clearances(meshes, body_paths, poses, boxes, hulls, path)
        with patch.object(diagnostic, 'reduce_clearances', reduce), patch.object(diagnostic,
                'depth_reference_scores', lambda frames, objects: depth_reference_scores_convex(frames, hulls, actual)):
            original_finish(value, result, capture)
        result['cylinder_reference'] = dict(kind='Actual PhysX-cooked32-sided convex prism',
            analytic_diameter_m=.035, analytic_height_m=.20, sides=32,
            cooked_status=copied['status'], cooked_hulls=len(hulls), callback_owner_thread=copied['owner_thread'],
            sampled_actual_world_from_body=diagnostic.serial(actual),
            no_scene_prims_added=before == tuple(str(prim.GetPath()) for prim in stage.Traverse()),
            clearance_scope='Actual cooked robot and cylinder convexes; no enclosing-box substitute')
        arrays = {}
        for index, (vertices, counts, indices) in enumerate(hulls):
            arrays.update({f'vertices_{index}': vertices, f'face_counts_{index}': counts, f'face_indices_{index}': indices})
        np.savez_compressed(Path(args.artifacts_dir)/'cylinder_cooked_reference.npz', **arrays)

    processor = module.RepresentationMappingProcessor
    worker_options = dict(representation_spec=spec)
    if args.capture_dir:
        processor = module.AuditedRepresentationMappingProcessor
        worker_options.update(audit_dir=args.capture_dir, capture_inputs=True,
            max_captures=args.capture_limit, capture_format='uncompressed')
    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime, 'MappingProcessor', partial(processor, **worker_options)))
        stack.enter_context(patch.object(runtime.MappingService, 'submit', submit))
        stack.enter_context(patch.object(diagnostic, 'run_native', native))
        stack.enter_context(patch.object(diagnostic, 'profile', profile))
        stack.enter_context(patch.object(diagnostic, 'finish', finish))
        stack.enter_context(patch.object(diagnostic, 'fixtures', fixtures))
        stack.enter_context(patch.object(bootstrap, 'build_runtime', build))
        stack.enter_context(patch.object(FrankaBackend, 'move_to_pose_async', move))
        stack.enter_context(patch.object(RmpflowObstacleRegistry, 'sync', sync))
        camera.run(args, app, layout)
    payload = json.loads(Path(args.result).read_text())
    result = payload.get('metrics') or {}
    trace_path = Path(args.artifacts_dir)/'trace.json'
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    motion_rows = [row for row in rows if row['phase'] == 'motion']
    operations = {key: sum(row['operations'][key] for row in motion_rows)
        for key in ('additions', 'removals', 'pose_updates', 'dimension_updates', 'identity_reuses', 'world_updates', 'skipped')}
    actual_updates = [row for row in motion_rows if any(row['operations'][key] for key in
        ('additions', 'removals', 'pose_updates', 'dimension_updates', 'world_updates'))]
    payload['representation_v3'] = dict(candidate_id=args.candidate_id, packing=args.packing,
        selection=args.selection, budget=args.max_obstacles, scenario=args.scenario,
        fixture_kind='32-sided cooked convex cylinder' if args.scenario == 'cylinder' else
            'rectangular thin box' if args.scenario == 'sparse' else args.scenario,
        activation=dict(owner_first_active_submit_at=state['activation_at'],
            phase='After recorded production-extractor setup q; before external-fixture warm publications',
            guard=result.get('representation_activation_guard'), initial_condition_match=result.get('initial_condition_match')),
        registry=dict(samples=len(motion_rows), owner_thread_all=all(row['owner_thread'] for row in rows),
            sync_ms=diagnostic.distribution([row['seconds'] for row in motion_rows], 1000),
            actual_update_ms=diagnostic.distribution([row['seconds'] for row in actual_updates], 1000),
            operations=operations, changed_calls=len(actual_updates),
            lifetime_registry=(result.get('closed_strategy') or {}).get('registry')),
        progress=progress_windows(trace.get('tcp') or []),
        full_capture_excluded_from_timing_claims=bool(args.capture_dir), capture_directory=args.capture_dir,
        capture_limit=args.capture_limit, capture_phase=args.capture_phase,
        original_harness_sha256=hashlib.sha256((ROOT/'scripts/diagnose_reactive_motion.py').read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        extractor_sha256=hashlib.sha256((ROOT/'scripts/obstacle_representation_v3.py').read_bytes()).hexdigest(),
        production_sources_unchanged=source_before == diagnostic.source_hashes(),
        no_dense_kit_processing=True, no_wait_for_mapping_during_motion=True)
    if not state['initial_checked'] or not payload['representation_v3']['production_sources_unchanged']:
        payload['status'] = 'FAIL'
    Path(args.artifacts_dir,'representation_trace.json').write_text(json.dumps(diagnostic.serial(rows), indent=2)+'\n')
    Path(args.result).write_text(json.dumps(diagnostic.serial(payload), indent=2, allow_nan=False)+'\n')
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layouts', default=str(ROOT/'docs/assets/camera_observability_feasibility/camera_layouts.json'))
    parser.add_argument('--layout-id', required=True)
    parser.add_argument('--strategy', choices=('direct', 'reactive'), default='reactive')
    parser.add_argument('--scenario', choices=('empty', 'static', 'sparse', 'cylinder', 'partial', 'narrow', 'moving', 'separated'), required=True)
    parser.add_argument('--packing', choices=PACKINGS, default='baseline')
    parser.add_argument('--selection', choices=SELECTIONS, default='baseline')
    parser.add_argument('--max-obstacles', type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument('--candidate-id')
    parser.add_argument('--reference-result')
    parser.add_argument('--initial-joint-tolerance', type=float, default=.01)
    parser.add_argument('--capture-dir')
    parser.add_argument('--capture-limit', type=int, default=8)
    parser.add_argument('--capture-phase', choices=('post_setup', 'motion'))
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)
    args.candidate_id = args.candidate_id or ('baseline' if (args.packing, args.selection, args.max_obstacles) == ('baseline', 'baseline', 64)
        else f'{args.packing}__{args.selection}__{args.max_obstacles}')
    if args.capture_limit < 1 or not math.isfinite(args.initial_joint_tolerance) or args.initial_joint_tolerance <= 0:
        parser.error('Capture limit and initial-q tolerance must be positive')
    args.capture_phase = args.capture_phase or ('motion' if args.scenario == 'moving' else 'post_setup')
    args.source, args.mapping_hz, args.max_rays, args.sampling_policy = 'rgbd', 2., 1024, 'fixed'
    args.mapping_audit_dir, args.audit_capture_inputs, args.audit_max_captures = None, False, 8
    args.lifecycle, args.contact_metrics = False, True
    from scripts.camera_observability_planner import select_layout
    layout = select_layout(json.loads(Path(args.layouts).read_text()), args.layout_id)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    app = None
    try:
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4 before SDK import, got {np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import warp as wp
        if not any(device.is_cuda for device in wp.get_devices()):
            Path(args.result).write_text(json.dumps(dict(status='BLOCKED', measured=False,
                error='GPU-backed native execution unavailable; no CPU fallback result'))+'\n')
            return False
        return run(args, app, layout)
    except Exception:
        Path(args.result).write_text(json.dumps(dict(status='FAIL', measured=False,
            error=traceback.format_exc()), indent=2)+'\n')
        return False
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
