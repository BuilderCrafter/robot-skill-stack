#!/usr/bin/env python3
"""One native V4.3 fixed-goal comparison, with read-only physical observations.

The initial baseline uses ordinary production IK. Subsequent conditions use a
research-only frozen IK candidate and unchanged production search/validation.
Measurement validity, physical completion and obstacle-induced detouring are
separate outcomes. No production source, joint state or controller is edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np

from scripts import benchmark_global_roundtrip_v4_2 as shared
from scripts import benchmark_global_acceptance_v4_1 as acceptance
from scripts import diagnose_global_stabilization_v4_1 as controlled
from scripts import benchmark_global_motion_v4 as baseline
from tests.support.reactive_contact_metrics import analyze_joint_feedback


ROOT = baseline.ROOT
TARGET = dict(position=[.45, -.25, .35], orientation=[0., 0., 1., 0.], frame='world')
CONDITIONS = ('baseline_free', 'free', 'obstacle', 'near', 'goal_blocked')
MODEL_SHA = '6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d'


def source_hashes(args):
    result = shared.source_hashes()
    paths = [Path(__file__)]
    if args.condition != 'baseline_free':
        paths.extend((ROOT/'scripts/global_detour_fixed_target_v4_3.py', Path(args.fixed_target)))
    if args.fixture:
        paths.append(Path(args.fixture))
    for path in paths:
        result[str(path.resolve().relative_to(ROOT))] = shared.digest(path)
    return result


def verify_frozen_design(fixture, *, expected_sha256, path):
    """Fail closed on changed pre-RRT geometry, witness or source artifacts."""
    if not expected_sha256 or shared.digest(path) != expected_sha256:
        raise ValueError('Frozen fixture differs from its predeclared job-plan SHA256')
    witness = fixture['witness']
    if (witness['status'] != 'INDEPENDENT_NUMERIC_CONTINUOUS_WITNESS_CERTIFIED'
            or witness['independent_of_candidate_rrt'] is not True
            or witness['planning_model_sampled_self_and_original_world_clear'] is not True):
        raise ValueError('Independent geometric/model witness was not established')
    bounds = [witness['numericcontinuousexternal_lower_bound_m'], witness['self_lower_bound_m'],
        witness['planning_model_continuous_external_lower_bound_m']]
    if not np.isfinite(bounds).all() or bounds[0] < .01 or min(bounds[1:]) <= 0.:
        raise ValueError('Frozen witness does not meet its declared positive clearance bounds')
    alignment = witness['maximum_native_sphere_alignment_error_m']
    if not np.isfinite(alignment) or not 0. <= alignment <= 1e-9:
        raise ValueError('Native sphere alignment exceeds the witness numeric allowance')
    sources = {**fixture['baseline']['source_sha256'], **witness['source_sha256']}
    for name, expected in sources.items():
        if shared.digest(name) != expected:
            raise ValueError('Frozen independent-design source changed: '+name)
    for name in ('raw_artifact', 'proof_artifact'):
        ref = witness[name]
        if shared.digest(ref['path']) != ref['sha256']:
            raise ValueError('Frozen independent witness artifact changed: '+name)
    if fixture.get('algorithm_revision', 0) >= 2:
        components = {
            'scripts/design_global_detour_exact_v4_3.py':fixture['original_design_producer_sha256'],
            'docs/assets/motion_planning_v4_3/design_exact_protocol.json':fixture['original_design_protocol_sha256'],
            'scripts/finalize_global_detour_v4_3.py':fixture['finalization_producer_sha256'],
        }
        for name, expected in components.items():
            if shared.digest(ROOT/name) != expected:
                raise ValueError('Frozen design component changed: '+name)
        for name in ('main_design_lock', 'main_design_proof', 'finalization_protocol', 'control_finalization_proof'):
            ref = fixture[name]
            if shared.digest(ref['path']) != ref['sha256']:
                raise ValueError('Frozen design/finalization reference changed: '+name)
        lock = json.loads(Path(fixture['main_design_lock']['path']).read_text())['locked_main_fixture']
        if any(fixture['fixture'][key] != lock[key] for key in ('position_m','dimensions_m','orientation_wxyz')):
            raise ValueError('Main obstacle changed during control-only finalization')
        control = json.loads(Path(fixture['finalization_protocol']['path']).read_text())
        for name, expected in control['original_design_source_sha256'].items():
            if shared.digest(name) != expected:
                raise ValueError('Control-finalization source changed: '+name)
        if set(fixture['protocol_overrides']) != {'negative_controls.goal_blocked.dimensions_m'}:
            raise ValueError('Control-only finalization changed another declared setting')


def load_inputs(args):
    target = None if args.fixed_target is None else json.loads(Path(args.fixed_target).read_text())
    fixture = None if args.fixture is None else json.loads(Path(args.fixture).read_text())
    if args.seed not in (123456, 234567, 345678) or args.max_iterations != 10000:
        raise ValueError('V4.3 uses only the three declared seeds and original10000 iteration limit')
    if args.condition != 'baseline_free':
        if target is None or target['target'] != TARGET or target['model_sha256'] != MODEL_SHA:
            raise ValueError('Fixed target/model differs from the declared baseline')
        for name in ('source_result', 'source_trace', 'source_geometry'):
            if shared.digest(target[name]['path']) != target[name]['sha256']:
                raise ValueError('Fixed-goal baseline source changed: '+name)
    if fixture is not None:
        verify_frozen_design(fixture, expected_sha256=getattr(args, 'fixture_sha256', None), path=args.fixture)
        if (fixture['candidate_RRT_calls_before_freeze'] != 0
                or fixture['fixture_revision_after_candidate_RRT_allowed'] is not False):
            raise ValueError('Candidate fixture must be frozen before candidate RRT evaluation')
        if target is not None and fixture['source_baseline_result_sha256'] != target['source_result']['sha256']:
            raise ValueError('Fixture and fixed target use different obstacle-free baselines')
        if target is not None and (not np.array_equal(fixture['baseline']['goal_q'], target['q'])
                or not np.array_equal(fixture['baseline']['start_q'], target['initial_state']['feedback']['positions'][:7])):
            raise ValueError('Fixture and comparison joint endpoints differ')
    if args.condition in ('obstacle', 'near', 'goal_blocked') and fixture is None:
        raise ValueError('Obstacle conditions require an independently frozen fixture')
    if args.condition == 'free' and fixture is None:
        raise ValueError('Matched free condition requires the same frozen design source cohort')
    item = None
    if args.condition in ('obstacle', 'near', 'goal_blocked'):
        item = fixture['fixture'] if args.condition == 'obstacle' else fixture['controls'][args.condition]
        if item['orientation_wxyz'] != [1., 0., 0., 0.]:
            raise ValueError('This bounded fixture supports only its declared axis-aligned cuboid')
        for key in ('position_m', 'dimensions_m'):
            value = np.asarray(item[key], float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Invalid physical cuboid '+key)
        if np.any(np.asarray(item['dimensions_m']) <= 0):
            raise ValueError('Cuboid dimensions must be positive')
    return target, fixture, item


def match_initial(actual, target, names, named_home):
    expected = target['initial_state']
    first, second = np.asarray(expected['feedback']['positions']), np.asarray(actual['feedback']['positions'])
    checks = dict(exact_named_dof_order=list(names) == target['robot_dof_names'],
        exact_actual_nine_joint_positions=first.shape == second.shape == (9,) and np.array_equal(first, second),
        same_named_home=baseline.serial(named_home) == target['named_home'],
        same_measured_base_pose=actual.get('robot_base_pose') == expected.get('robot_base_pose'),
        same_native_position_targets=(actual['live_drive_feedback']['robot_applied_action'] ==
            expected['live_drive_feedback']['robot_applied_action']))
    return dict(checks=checks, matched=all(checks.values()),
        maximum_actual_joint_difference=float(np.max(abs(first-second))) if first.shape == second.shape else None)


def observe_planner(factory, state):
    """Capture limits and result metadata without changing a planning call."""
    def create(backend, **kwargs):
        planner = factory(backend, **kwargs)
        original = planner.plan
        def plan(*args, **options):
            record = dict(target=baseline.serial(args[0]), start_q=baseline.serial(args[1]),
                snapshot=baseline.serial(args[2]),
                position_tolerance_m=options.get('position_tolerance', args[3] if len(args) > 3 else None),
                orientation_tolerance_rad=options.get('orientation_tolerance', args[4] if len(args) > 4 else None))
            state['planning_records'].append(record)
            try:
                result = original(*args, **options)
                record.update(joint_names=list(result.joint_names), route=result.route.tolist(),
                    duration=result.duration, diagnostics=baseline.serial(result.diagnostics),
                    limits={name:getattr(result, name).tolist() for name in
                        ('lower_limits', 'upper_limits', 'velocity_limits', 'acceleration_limits', 'jerk_limits')})
                return result
            finally:
                record.setdefault('diagnostics', baseline.serial(planner._last_diagnostics))
        planner.plan = plan
        return planner
    return create


def run_native(args, app, state, target_document, item):
    import omni.kit.app
    from pxr import UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from tests.support.isaac_async import run_kit_coroutine

    stage = shared.prepare_stage(app, obstacle=item)
    async def evaluate():
        bundle = observer = subscription = None
        callback = False
        try:
            profile_args = SimpleNamespace(mode='roundtrip', collision_model='enhanced', seed=args.seed,
                max_iterations=args.max_iterations, artifacts_dir=args.artifacts_dir)
            bundle = await build_runtime(shared.write_profile(profile_args))
            observer = controlled.make_observer(bundle)
            state['observer'] = observer
            capture = acceptance.fresh_physical_capture(bundle)
            state['capture'] = capture
            capture['body_paths'] = tuple(observer.metadata['body_paths'])
            state['initial_state'] = observer.state()
            state['named_home'] = baseline.serial(bundle.backend.named_poses['home'])
            state['fingerprints'] = shared.cooked_fingerprints(capture['meshes'])
            observer.metadata['robot_dof_names'] = list(bundle.backend.robot.dof_names)
            observer.metadata['meters_per_unit'] = capture['snapshot'].meters_per_unit
            state['initial_match'] = (dict(matched=True, meaning='Initial ordinary-IK baseline; no prior candidate')
                if target_document is None else match_initial(state['initial_state'], target_document,
                    bundle.backend.robot.dof_names, bundle.backend.named_poses['home']))
            if not state['initial_match']['matched'] or observer.actions:
                raise RuntimeError('Measured starting state differs from frozen baseline or unexpected setup action occurred')
            if target_document is not None and state['fingerprints'] != target_document['cooked_fingerprints']:
                raise RuntimeError('Fresh cooked body geometry differs from fixed-goal baseline')
            state['world_before_snapshot'], state['world_before_sha256'] = shared._world_signature(stage)
            initial_prims = tuple(str(prim.GetPath()) for prim in stage.Traverse())
            obstacle_view = None if item is None else SimulationManager.get_physics_sim_view().create_rigid_body_view([item['prim_path']])
            def signature():
                if item is None:
                    return dict(physically_present=False)
                prim = stage.GetPrimAtPath(item['prim_path'])
                return dict(physically_present=bool(prim and prim.IsActive()),
                    actor_pose=np.asarray(obstacle_view.get_transforms()).copy(),
                    collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                    size=prim.GetAttribute('size').Get(), translate=tuple(prim.GetAttribute('xformOp:translate').Get()),
                    scale=tuple(prim.GetAttribute('xformOp:scale').Get()))
            state['fixture_before'] = baseline.serial(signature())
            if item is not None:
                expected = np.array([*item['position_m'], 0., 0., 0., 1.])
                if np.max(abs(np.asarray(state['fixture_before']['actor_pose'])[0]-expected)) > 1e-6:
                    raise RuntimeError('Physical actor pose differs from frozen cuboid')
                boxes = [x for x in state['world_before_snapshot'].obstacles if x.path == item['prim_path']]
                if len(boxes) != 1 or not np.allclose(boxes[0].dimensions, item['dimensions_m'], atol=1e-7, rtol=0):
                    raise RuntimeError('Frozen obstacle is missing from captured planner world')
            sample = observer.sample
            def observe():
                sample()
                observer.samples[-1]['frozen_fixture'] = baseline.serial(signature())
            observer.sample = observe
            observer.sample()
            initial_transforms = [capture['pose_matrices'](observer.samples[-1]['poses'], capture['snapshot'].meters_per_unit)]
            state['initial_physical_clearance'] = baseline.reduce_clearances(capture['meshes'], capture['body_paths'],
                initial_transforms, capture['snapshot'].obstacles)
            if state['initial_physical_clearance']['minimum_m'] is None or state['initial_physical_clearance']['minimum_m'] <= 0.:
                raise RuntimeError('Actual initial/Home physical robot is not independently clear')
            bundle.world.add_physics_callback('v4_3_detour_capture', lambda dt: observer.sample())
            callback = True
            observer.metadata.update(kit_times=[], controller_waits_seconds=[],
                sampling='Physics callback plus update and operation boundaries; missing steps reported')
            update = bundle.backend._next_update
            async def wait():
                started = time.monotonic()
                try:
                    return await update()
                finally:
                    observer.metadata['controller_waits_seconds'].append(time.monotonic()-started)
            bundle.backend._next_update = wait
            subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                lambda event:observer.metadata['kit_times'].append(time.monotonic()), name='v4_3_detour_kit')
            strategy = bundle.backend.motion_strategy
            original_factory = strategy._planner_factory
            outward_factory = original_factory
            if target_document is not None:
                from scripts.global_detour_fixed_target_v4_3 import FixedJointTarget, FixedJointTargetPlanner
                fixed = FixedJointTarget.from_dict(target_document)
                outward_factory = lambda backend, **kwargs:FixedJointTargetPlanner(backend, fixed_target=fixed, **kwargs)
            strategy._planner_factory = observe_planner(outward_factory, state)
            state['rss_before_motion'] = shared.rss_bytes()
            try:
                result = await acceptance.motion(observer, 'outward', 'move_to_pose',
                    target=Pose(TARGET['position'], TARGET['orientation']))
            finally:
                strategy._planner_factory = observe_planner(original_factory, state)
            state['world_after_outward_snapshot'], state['world_after_outward_sha256'] = shared._world_signature(stage)
            if result.ok and observer.operations[-1]['measured_full_pose_within_tolerances'] and args.condition != 'goal_blocked':
                await acceptance.motion(observer, 'home_return', 'home')
            else:
                state['return_not_attempted'] = 'Outward did not complete or negative goal-blocked control'
            strategy._planner_factory = original_factory
            state['rss_after_motion'] = shared.rss_bytes()
            state['fixture_after'] = baseline.serial(signature())
            state['world_after_snapshot'], state['world_after_sha256'] = shared._world_signature(stage)
            state['same_stage_prims'] = initial_prims == tuple(str(prim.GetPath()) for prim in stage.Traverse())
        finally:
            subscription = None
            if bundle is not None:
                if callback:
                    bundle.world.remove_physics_callback('v4_3_detour_capture')
                if observer is not None:
                    observer.restore()
                bundle.close()
                state['runtime_closed'] = True
                state['strategy_after_close'] = baseline.serial(bundle.backend.motion_strategy.diagnostics)
    run_kit_coroutine(evaluate(), app)


def reduce_capture(observer, capture, item):
    if item is not None:
        return shared.reduce_operations(observer, capture, {'fixture':item})
    operations = controlled.reduce_observations(observer, capture)
    for summary, operation in zip(operations, observer.operations):
        samples = observer.samples[operation['first_sample']:operation['last_sample']]
        summary.update(physics_step_coverage=shared.physics_step_coverage(samples),
            joint_feedback=analyze_joint_feedback(samples, arm_joint_indices=observer.indices),
            maximum_actual_joint_displacement_rad=acceptance.maximum_observed_displacement(
                operation['before']['feedback']['positions'], samples, observer.indices),
            simulation_duration_seconds=operation['after']['simulation_time']-operation['before']['simulation_time'])
    return operations


def execute(args, app, target_document, frozen, item, before):
    output = Path(args.artifacts_dir)
    state = dict(planning_records=[])
    error = None
    operations = []
    try:
        run_native(args, app, state, target_document, item)
    except BaseException:
        error = traceback.format_exc()
    observer, capture = state.get('observer'), state.get('capture')
    if observer is not None and capture is not None:
        try:
            operations = reduce_capture(observer, capture, item)
        except BaseException:
            error = (error or '')+'\nOffline reduction:\n'+traceback.format_exc()
        arrays = {}
        for index, part in enumerate(capture['meshes']):
            arrays.update({f'vertices_{index}':part.mesh.vertices, f'triangles_{index}':part.mesh.triangles,
                f'body_from_mesh_{index}':part.body_from_mesh})
        arrays['body_poses'] = np.asarray([r['poses'] for r in observer.samples])
        arrays['simulation_times'] = np.asarray([r['simulation_time'] for r in observer.samples])
        np.savez_compressed(output/'geometry_reference.npz', **arrays)
        controlled._json(output/'roundtrip_trace.json', dict(metadata=observer.metadata, samples=observer.samples,
            actions=observer.actions, operations=observer.operations,
            clearance_curves={r['label']:r.pop('fixture_clearance_curve', []) for r in operations}))
    after = source_hashes(args)
    samples = [] if observer is None else observer.samples
    signatures = [r['frozen_fixture'] for r in samples if 'frozen_fixture' in r]
    checks = dict(source_unchanged=before == after, runtime_closed=state.get('runtime_closed', False),
        measured_start_matches_baseline=state.get('initial_match', {}).get('matched', False),
        all11_cooked_bodies=bool(state.get('fingerprints')),
        same_stage_prims=state.get('same_stage_prims', False),
        fixture_stationary_every_observed_step=bool(signatures) and all(shared.canonical_hash(x) ==
            shared.canonical_hash(state['fixture_before']) for x in signatures),
        world_geometry_unchanged=bool(state.get('world_before_sha256')) and state.get('world_before_sha256') ==
            state.get('world_after_outward_sha256') == state.get('world_after_sha256'),
        owner_thread=bool(samples) and all(r['owner_thread'] for r in samples),
        contact_buffer_unsaturated=bool(samples) and not any(r['contact_buffer_full'] for r in samples),
        complete_physics_steps=bool(operations) and all(r['physics_step_coverage']['complete'] for r in operations))
    physical = shared.physical_checks(operations)
    metrics = dict(condition=args.condition, mode='roundtrip', collision_model='enhanced', seed=args.seed,
        max_iterations=args.max_iterations, configuration=dict(target=TARGET, model_sha256=MODEL_SHA,
            rrt_step_size_rad=.05, position_tolerance_m=.025, orientation_tolerance_rad=.10),
        fixed_target=target_document, initial_state=state.get('initial_state'), named_home=state.get('named_home'),
        initial_match=state.get('initial_match'), cooked_fingerprints=state.get('fingerprints'),
        initial_physical_clearance=state.get('initial_physical_clearance'),
        planning_records=state['planning_records'], snapshot=baseline.serial(capture['snapshot']) if capture else None,
        robot_dof_names=observer.metadata['robot_dof_names'] if observer else [],
        meters_per_unit=capture['snapshot'].meters_per_unit if capture else None, trace_artifact='roundtrip_trace.json',
        geometry_reference=dict(cooking=capture['cooking'], meshes=len(capture['meshes']), cooked_only=True,
            hand_and_both_fingers=True, all11_bodies=True) if capture else None,
        contact_reference=dict(sensor_paths=observer.metadata['body_paths'], filter_paths=observer.metadata['contact_filter_paths']) if observer else None,
        operations=operations, acceptance_checks=physical,
        roundtrip_acceptance=bool(physical) and all(physical.values()) and all(checks.values()) and error is None,
        strict_zero_report_roundtrip=bool(physical) and all(physical.values()) and all(r['strict_success'] for r in operations),
        return_not_attempted=state.get('return_not_attempted'),
        fixture=item, fixture_document_sha256=shared.digest(args.fixture) if args.fixture else None,
        fixture_before=state.get('fixture_before'), fixture_after=state.get('fixture_after'),
        kit_interval_ms=baseline.distribution(np.diff(observer.metadata['kit_times'])*1000) if observer else {},
        controller_wait_ms=baseline.distribution(np.asarray(observer.metadata['controller_waits_seconds'])*1000) if observer else {},
        rss={k:v for k,v in state.items() if k.startswith('rss_')}, strategy_after_close=state.get('strategy_after_close'))
    payload = dict(schema_version=1, status='PASS' if error is None and all(checks.values()) else 'FAIL',
        contracts=checks, hashes=before, hashes_after=after, metrics=metrics, error=error,
        meaning='PASS means valid retained observations. Physical roundtrip completion and obstacle-induced route change are assessed separately. Sampled clearance is not continuous swept-volume certification.')
    controlled._json(args.result, payload)
    if args.condition == 'baseline_free' and metrics['roundtrip_acceptance']:
        outward = observer.operations[0]
        route = outward['plan_trace']['route']
        selected = metrics['planning_records'][0]
        diag = selected['diagnostics']
        goal_q = diag['ik_attempts'][diag['selected_ik_index']]['ik_q']
        if np.max(abs(np.asarray(goal_q)-np.asarray(route[-1]))) > 1e-6:
            raise RuntimeError('Baseline route endpoint differs from validated IK goal')
        if diag['sdk_configuration_sha256']['robot_description_path'] != MODEL_SHA:
            raise RuntimeError('Baseline did not use frozen enhanced collision description')
        fixed = dict(schema_version=1, joint_names=outward['plan_trace']['joint_names'], q=goal_q, target=TARGET,
            baseline_route_goal_q=route[-1], selected_ik_index=diag['selected_ik_index'],
            end_effector_frame=observer.backend.end_effector_frame, initial_state=state['initial_state'],
            named_home=state['named_home'], robot_dof_names=observer.metadata['robot_dof_names'],
            cooked_fingerprints=state['fingerprints'], model_sha256=MODEL_SHA, planning_limits=selected['limits'],
            rrt_max_iterations=10000, source_result=dict(path=str(args.result), sha256=shared.digest(args.result)),
            source_trace=dict(path=str(output/'roundtrip_trace.json'), sha256=shared.digest(output/'roundtrip_trace.json')),
            source_geometry=dict(path=str(output/'geometry_reference.npz'), sha256=shared.digest(output/'geometry_reference.npz')))
        controlled._json(output/'frozen_target.json', fixed)
    print(json.dumps(dict(status=payload['status'], condition=args.condition,
        physical_roundtrip=metrics['roundtrip_acceptance'], error=error)), flush=True)
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--fixed-target')
    parser.add_argument('--fixture')
    parser.add_argument('--fixture-sha256')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    shared.refuse_overwrite(args)
    target, frozen, item = load_inputs(args)
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use unchanged native launcher with NumPy1.26.4')
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    before = source_hashes(args)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':args.headless})
    try:
        return 0 if execute(args, app, target, frozen, item, before) else 1
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
