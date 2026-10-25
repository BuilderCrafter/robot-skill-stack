#!/usr/bin/env python3
"""One fresh frozen V4.2 Home/target/Home or counterfactual direct reference.

The new fixture is independent of all historical fixtures. Its declaration and
numeric witness precede candidate RRT evaluation. Observers never command joint
states or modify gains, iteration scopes, tracking, settling or tolerances.
Large physics captures stay under ignored outputs; PASS means valid observation,
while roundtrip_acceptance records the separate physical execution outcome.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from scripts import benchmark_global_motion_v4 as baseline
from scripts import benchmark_global_acceptance_v4_1 as acceptance
from scripts import diagnose_global_stabilization_v4_1 as controlled
from tests.support.reactive_contact_metrics import analyze_contact_samples, analyze_joint_feedback
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle


ROOT = baseline.ROOT
PROTOCOL = ROOT/'docs/assets/motion_planning_v4_2/roundtrip_protocol.json'
FROZEN_SHA256 = '79270a3d691451d631830bc9d130945d58d1ebc44a14ec7630d6a86ea7b0f54b'
BODY_NAMES = ('panda_hand', 'panda_rightfinger', 'panda_leftfinger',
              *(f'panda_link{i}' for i in range(8)))
MODES = ('roundtrip', 'direct_reference')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(baseline.serial(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_protocol(path=PROTOCOL):
    path = Path(path)
    protocol = json.loads(path.read_text())
    frozen_path = path.parent/protocol['fixture_file']
    if protocol['fixture_sha256'] != FROZEN_SHA256 or digest(frozen_path) != FROZEN_SHA256:
        raise ValueError('Frozen fixture hash changed; candidate outcomes cannot revise this fixture')
    frozen = json.loads(frozen_path.read_text())
    if frozen['candidate_RRT_calls_before_freeze'] != 0 or frozen['fixture_revision_after_candidate_RRT_allowed']:
        raise ValueError('Fixture was not frozen independently before candidate RRT')
    return protocol, frozen


def load_design_evidence(protocol):
    folder = ROOT/protocol['design_artifact_directory']
    reference = ROOT/protocol['raw_recorded_source_directory']
    for directory, manifest in ((folder, protocol['design_artifact_sha256']),
                                (reference, protocol['recorded_source_sha256'])):
        for name, expected in manifest.items():
            if digest(directory/name) != expected:
                raise ValueError(f'Frozen independent design source hash changed: {directory/name}')
    search = json.loads((folder/'search-result.json').read_text())
    for name, expected in search['source_hashes'].items():
        if digest(name) != expected:
            raise ValueError(f'Independent witness geometry/FK source changed: {name}')
    certificate = json.loads((folder/'witness-certificate.json').read_text())
    actual = json.loads((folder/'actual-and-interval-audit.json').read_text())
    if (not certificate['continuous_external_obstacle_clearance_proven_for_declared_piecewise_joint_linear_numeric_witness']
            or certificate['minimum_continuous_external_obstacle_separation_lower_bound_m'] <= 0
            or actual['minimum_continuous_nonadjacent_self_separation_lower_bound_m'] <= 0):
        raise ValueError('Independent declared witness is not geometrically feasible')
    generation = json.loads((reference/'acceptance_trace.json').read_text())[0]
    return dict(folder=folder, reference=reference, search=search,
                certificate=certificate, actual=actual, generation=generation)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return canonical_hash(dict(shape=value.shape, dtype=value.dtype.str,
                               bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest()))


def cooked_fingerprints(meshes):
    result = {}
    for part in meshes:
        if part.body_path in result or not part.approximation.startswith('physx_cooked:'):
            raise ValueError('Exactly one actually cooked hull per physical body is required')
        result[part.body_path] = dict(vertices=_array_hash(part.mesh.vertices),
            triangles=_array_hash(part.mesh.triangles), transform=_array_hash(part.body_from_mesh))
    expected = {'/Franka/'+name for name in BODY_NAMES}
    if set(result) != expected:
        raise ValueError('Independent reference must retain all11 bodies including base and both fingers')
    return result


def reference_fingerprints(evidence):
    result = {}
    with np.load(evidence['reference']/'acceptance_geometry_0.npz', allow_pickle=False) as archive:
        for index, row in enumerate(evidence['generation']['cooking']):
            if row['cooked_hulls'] != 1 or row['body'] in result:
                raise ValueError('Frozen reference has ambiguous cooked geometry')
            result[row['body']] = dict(vertices=_array_hash(archive[f'vertices_{index}']),
                triangles=_array_hash(archive[f'triangles_{index}']),
                transform=_array_hash(archive[f'body_from_mesh_{index}']))
    return result


def frozen_obstacle(frozen):
    item = frozen['fixture']
    return StaticObstacle(item['prim_path'], 'cuboid', tuple(item['position_m']),
        orientation=tuple(item['orientation_wxyz']), dimensions=tuple(item['dimensions_m']),
        source_kind='Cube', approximation='exact')


def startup_match(state, named_home, protocol, frozen):
    """Measured state and the runtime's actual named Home must both match design."""
    settings = protocol['startup_match']
    q = np.asarray(state['feedback']['positions'], float)
    expected = np.asarray(frozen['initial_joint_positions'], float)
    finite = q.shape == expected.shape == (9,) and np.isfinite(q).all()
    joint_error = float(np.max(abs(q-expected))) if finite else None
    pose, target = state['tcp_pose'], frozen['initial_home']
    position = float(np.linalg.norm(np.asarray(pose['position'])-target['position']))
    from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import orientation_error, quaternion_matrix
    orientation = orientation_error(quaternion_matrix(pose['orientation']), quaternion_matrix(target['orientation']))
    named = baseline.serial(named_home)
    named_position = float(np.linalg.norm(np.asarray(named['position'])-target['position']))
    named_orientation = orientation_error(quaternion_matrix(named['orientation']), quaternion_matrix(target['orientation']))
    checks = dict(actual_joints=joint_error is not None and joint_error <= settings['maximum_joint_difference_rad_or_m'],
        actual_tcp_position=position <= settings['maximum_tcp_position_difference_m'],
        actual_tcp_orientation=orientation <= settings['maximum_tcp_orientation_difference_rad'],
        named_home_position=named_position <= settings['maximum_tcp_position_difference_m'],
        named_home_orientation=named_orientation <= settings['maximum_tcp_orientation_difference_rad'])
    return dict(checks=checks, matched=all(checks.values()), maximum_joint_difference_rad_or_m=joint_error,
        tcp_position_difference_m=position, tcp_orientation_difference_rad=orientation,
        named_home_position_difference_m=named_position, named_home_orientation_difference_rad=named_orientation,
        measured_finger_positions_m=q[7:9].tolist() if finite else None,
        latched_applied_action=state['live_drive_feedback']['robot_applied_action'])


def source_hashes():
    result = controlled.source_hashes()
    for path in (Path(__file__), Path(acceptance.__file__), PROTOCOL, PROTOCOL.parent/'roundtrip_fixture.json'):
        result[str(path.relative_to(ROOT))] = digest(path)
    # Include project-owned planning resources without assuming their filenames.
    for folder in ('robot_skill_stack/integrations/isaac/manipulation/motion', 'config/motion'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.yaml', '.yml', '.urdf', '.json'):
                result[str(path.relative_to(ROOT))] = digest(path)
    return result


def write_profile(args):
    # Reuse the production-parsed established profile, with a new explicit model.
    from types import SimpleNamespace
    options = SimpleNamespace(strategy='direct' if args.mode == 'direct_reference' else 'global',
        seed=args.seed, max_iterations=args.max_iterations, artifacts_dir=args.artifacts_dir,
        enable_mapping=False)
    path = baseline.write_profile(options)
    text = path.read_text().replace('[motion]\n', f'[motion]\nglobal_collision_model = "{args.collision_model}"\n')
    path.write_text(text)
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    load_scene_config(path)
    return path


def rss_bytes():
    current = None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                current = int(line.split()[1])*1024
    except OSError:
        pass
    return dict(current_rss_bytes=current, process_maximum_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        interpretation='Process RSS includes Isaac, resources and retained observations; differences alone do not establish a leak')


def physical_checks(operations):
    by_label = {row['label']: row for row in operations}
    return {label: acceptance.physical_motion_passed(by_label.get(label, {}), moving=True)
            for label in ('outward', 'home_return')}


def physics_step_coverage(samples):
    unique = sorted({int(row['physics_step']) for row in samples})
    missing = [step for first, last in zip(unique[:-1], unique[1:]) for step in range(first+1, last)]
    return dict(unique_steps=len(unique), first=unique[0] if unique else None,
        last=unique[-1] if unique else None, missing_steps=missing,
        complete=bool(unique) and not missing)


def unique_physics_samples(samples):
    return list({(row['physics_step'], row['simulation_time']): row for row in samples}.values())


def refuse_overwrite(args):
    if Path(args.result).exists() or any((Path(args.artifacts_dir)/name).exists() for name in (
            'roundtrip_trace.json', 'geometry_reference.npz', 'runtime_profile.toml')):
        raise ValueError('Refusing to overwrite an existing native attempt')


def preflight(observer, capture, protocol, frozen, evidence):
    actual = cooked_fingerprints(capture['meshes'])
    expected = reference_fingerprints(evidence)
    if actual != expected:
        raise RuntimeError('Fresh PhysX-cooked geometry differs from frozen independent witness reference')
    match = startup_match(observer.state(), observer.backend.named_poses['home'], protocol, frozen)
    if not match['matched']:
        raise RuntimeError('Fresh startup/named Home differs from recorded design Home: '+str(match))
    obstacle = frozen_obstacle(frozen)
    observer.sample()
    samples = evidence['generation']['samples']
    paths = tuple(observer.metadata['body_paths'])
    old_paths = evidence['generation']['metadata']['body_paths']
    reorder = [old_paths.index(path) for path in paths]
    to_pose = capture['pose_matrices']
    def clearance(rows, old=False):
        poses = [np.asarray(row['poses'])[reorder] if old else row['poses'] for row in rows]
        return baseline.reduce_clearances(capture['meshes'], paths,
            [to_pose(pose, 1.) for pose in poses], (obstacle,))
    home = clearance(observer.samples[-1:])
    target = clearance(samples[116:117], old=True)
    witness = clearance([*reversed(samples[204:339]), *reversed(samples[116:204])], old=True)
    if min(home['minimum_m'], target['minimum_m'], witness['minimum_m']) <= 0.:
        raise RuntimeError('Frozen fixture preflight has a cooked-solid Home/target/witness intersection')
    if observer.actions:
        raise RuntimeError('An arm action preceded preflight; START setup is forbidden')
    return dict(passed=True, startup_match=match, fresh_cooked_fingerprints=actual,
        no_setup_arm_actions=True, initial_physics_step=observer.samples[-1]['physics_step'],
        predeclared_direct_overlap=frozen['ordinary_direct_obstruction'],
        initial_home_clearance=home, retained_target_clearance=target, retained_witness_sampled_clearance=witness,
        independent_joint_linear_witness_certificate=evidence['certificate'],
        independent_witness_continuous_nonadjacent_self_lower_bound_m=evidence['actual']['minimum_continuous_nonadjacent_self_separation_lower_bound_m'],
        qualification='The independent constructed witness predates candidate RRT. Its continuous numeric bounds do not certify a future executed trajectory. Initial/Home clearance uses actual fresh physical poses.')


def _world_signature(stage):
    from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import capture_static_world
    snapshot = capture_static_world(stage, '/Franka')
    return snapshot, canonical_hash([asdict(item) for item in snapshot.obstacles])


def prepare_stage(app, *, obstacle=None):
    """Session-only scene and report setup shared by the new V4.2 captures."""
    import omni.usd
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import is_stage_loading, open_stage

    if not open_stage(str(baseline.SCENE)):
        raise RuntimeError('Could not open established native scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
    item = obstacle
    UsdGeom.Xform.Define(stage, str(Path(item['prim_path']).parent) if item else baseline.FIXTURE_ROOT)
    if item is not None:
        cube = UsdGeom.Cube.Define(stage, item['prim_path'])
        cube.GetSizeAttr().Set(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*item['position_m']))
        cube.AddScaleOp().Set(Gf.Vec3f(*item['dimensions_m']))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(.75, .18, .08)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr().Set(True)
    # Same report ownership as the existing V4 reference: rigid robot bodies
    # and external actor owners (or the static collider when it has no actor).
    robot_root = stage.GetPrimAtPath('/Franka')
    report_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(robot_root,
        Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith('/Franka/') or path == '/Franka' or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        report_paths.append(str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else path)
    for path in dict.fromkeys(report_paths):
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
    return stage


def run_native(args, app, state, protocol, frozen, evidence):
    import omni.kit.app
    from pxr import UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from tests.support.isaac_async import run_kit_coroutine
    item = frozen['fixture']
    stage = prepare_stage(app, obstacle=item if args.mode == 'roundtrip' else None)

    async def evaluate():
        bundle, observer, subscription, callback_added = None, None, None, False
        try:
            bundle = await build_runtime(write_profile(args))
            state['bundle'] = bundle
            observer = controlled.make_observer(bundle)
            state['observer'] = observer
            capture = acceptance.fresh_physical_capture(bundle)
            state['capture'] = capture
            capture['body_paths'] = tuple(observer.metadata['body_paths'])
            state['preflight'] = preflight(observer, capture, protocol, frozen, evidence)
            state['world_before_snapshot'], state['world_before_sha256'] = _world_signature(stage)
            if args.mode == 'roundtrip':
                captured_boxes = [box for box in state['world_before_snapshot'].obstacles if box.path == item['prim_path']]
                if len(captured_boxes) != 1 or not np.allclose(captured_boxes[0].dimensions,
                        item['dimensions_m'], rtol=0., atol=1e-7):
                    raise RuntimeError('Frozen physical obstacle dimensions differ from declaration')
            prims = tuple(str(prim.GetPath()) for prim in stage.Traverse())
            obstacle_view = None
            if args.mode == 'roundtrip':
                obstacle_view = SimulationManager.get_physics_sim_view().create_rigid_body_view([item['prim_path']])
                if tuple(obstacle_view.prim_paths) != (item['prim_path'],):
                    raise RuntimeError('Frozen physical obstacle actor is absent')
            def obstacle_signature():
                prim = stage.GetPrimAtPath(item['prim_path'])
                if args.mode == 'direct_reference':
                    return dict(physically_present=bool(prim and prim.IsActive()))
                if not prim or not prim.IsActive():
                    return dict(physically_present=False)
                return dict(physically_present=True,
                    actor_pose=np.asarray(obstacle_view.get_transforms()).copy(),
                    collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                    size=prim.GetAttribute('size').Get(),
                    translate=tuple(prim.GetAttribute('xformOp:translate').Get()),
                    scale=tuple(prim.GetAttribute('xformOp:scale').Get()))
            state['fixture_before'] = baseline.serial(obstacle_signature())
            if args.mode == 'roundtrip':
                measured = np.asarray(state['fixture_before']['actor_pose'])[0]
                if np.max(abs(measured-np.array([*item['position_m'], 0., 0., 0., 1.]))) > 1e-6:
                    raise RuntimeError('Frozen actor physical pose differs from declared fixture')
            sample = observer.sample
            def observe():
                sample()
                observer.samples[-1]['frozen_fixture'] = baseline.serial(obstacle_signature())
            observer.sample = observe
            # A physics callback retains every observed simulation step even
            # when one Kit update advances several physics steps.
            bundle.world.add_physics_callback('v4_2_roundtrip_capture', lambda dt: observer.sample())
            callback_added = True
            observer.metadata['sampling'] = 'Read-only physics callback plus operation/update boundaries; coverage reported from physics_step values'
            observer.metadata['kit_times'] = []
            observer.metadata['controller_waits_seconds'] = []
            update = bundle.backend._next_update
            async def wait():
                began = time.monotonic()
                try:
                    return await update()
                finally:
                    observer.metadata['controller_waits_seconds'].append(time.monotonic()-began)
            bundle.backend._next_update = wait
            subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                lambda event: observer.metadata['kit_times'].append(time.monotonic()), name='v4_2_roundtrip_kit')
            state['rss_before_motion'] = rss_bytes()
            target = frozen['outward_target']
            await acceptance.motion(observer, 'outward', 'move_to_pose',
                target=Pose(target['position'], target['orientation']))
            state['world_after_outward_snapshot'], state['world_after_outward_sha256'] = _world_signature(stage)
            state['rss_after_outward'] = rss_bytes()
            outward = observer.operations[-1]
            if args.mode == 'roundtrip' and outward['returned_success'] and outward['measured_full_pose_within_tolerances']:
                await acceptance.motion(observer, 'home_return', 'home')
            elif args.mode == 'roundtrip':
                state['return_not_attempted'] = 'Outward failed; retain failure without changing fixture or attempting an unsafe follow-up'
            state['rss_after_motion'] = rss_bytes()
            state['fixture_after'] = baseline.serial(obstacle_signature())
            state['world_after_snapshot'], state['world_after_sha256'] = _world_signature(stage)
            state['same_stage_prims_after_motion'] = prims == tuple(str(prim.GetPath()) for prim in stage.Traverse())
        finally:
            subscription = None
            if bundle is not None:
                if callback_added:
                    bundle.world.remove_physics_callback('v4_2_roundtrip_capture')
                if observer is not None:
                    observer.restore()
                bundle.close()
                state['runtime_closed'] = True
                state['strategy_after_close'] = baseline.serial(getattr(bundle.backend.motion_strategy, 'diagnostics', {}))
    run_kit_coroutine(evaluate(), app)


def reduce_operations(observer, capture, frozen):
    summaries = controlled.reduce_observations(observer, capture)
    obstacle = frozen_obstacle(frozen)
    for summary, operation in zip(summaries, observer.operations):
        samples = observer.samples[operation['first_sample']:operation['last_sample']]
        transforms = [capture['pose_matrices'](row['poses'], 1.) for row in samples]
        summary['fixture_clearance'] = baseline.reduce_clearances(capture['meshes'],
            capture['body_paths'], transforms, (obstacle,))
        summary['fixture_clearance_interpretation'] = 'Counterfactual frozen box' if not any(
            row.get('frozen_fixture', {}).get('physically_present') for row in samples) else 'Unchanged physical frozen box'
        summary['joint_feedback'] = analyze_joint_feedback(samples, arm_joint_indices=observer.indices)
        summary['physics_step_coverage'] = physics_step_coverage(samples)
        summary['simulation_duration_seconds'] = operation['after']['simulation_time']-operation['before']['simulation_time']
        summary['final_measured_arm_velocity_max_rad_s'] = float(np.max(np.abs(
            np.asarray(operation['after']['feedback']['velocities'])[list(observer.indices)])))
        solver_before = samples[0]['dynamics']['get_solver_position_iteration_counts']
        solver_after = samples[-1]['dynamics']['get_solver_position_iteration_counts']
        summary['solver_position_iterations_before_after'] = dict(before=solver_before, after=solver_after,
            restored=solver_before == solver_after)
        # Record actual displacement across the entire operation, not only endpoints.
        summary['maximum_actual_joint_displacement_rad'] = acceptance.maximum_observed_displacement(
            operation['before']['feedback']['positions'], samples, observer.indices)
        distinct = unique_physics_samples(samples)
        summary['fixture_clearance_curve'] = [dict(physics_step=row['physics_step'], simulation_time=row['simulation_time'],
            **baseline.reduce_clearances(capture['meshes'], capture['body_paths'], [pose], (obstacle,)))
            for row in distinct for pose in [capture['pose_matrices'](row['poses'], 1.)]]
        pairs = [dict(row, contacts=[pair for pair in row['contacts'] if pair['obstacle'] == obstacle.path]) for row in distinct]
        summary['fixture_contacts'] = baseline.contact_summary(pairs)
        summary['fixture_contact_severity'] = analyze_contact_samples(pairs, capture['body_paths'],
            np.asarray(observer.metadata['local_coms']), minimum_clearance_m=summary['fixture_clearance']['minimum_m'])
        summary['positive_separation_reports'] = sum(int(np.count_nonzero(np.asarray(pair['separations_m']) > 0.))
            for row in distinct for pair in row['contacts'])
        # Curves belong in ignored captures, compact metrics retain their path.
    return summaries


def execute_case(args, app, protocol, frozen, evidence):
    output, result_path = Path(args.artifacts_dir), Path(args.result)
    refuse_overwrite(args)
    output.mkdir(parents=True, exist_ok=True)
    before, state, error = source_hashes(), {}, None
    operations = []
    try:
        run_native(args, app, state, protocol, frozen, evidence)
    except BaseException:
        error = traceback.format_exc()
    observer, capture = state.get('observer'), state.get('capture')
    if observer is not None and capture is not None:
        try:
            operations = reduce_operations(observer, capture, frozen)
        except BaseException:
            error = (error or '')+'\nOffline reduction:\n'+traceback.format_exc()
        arrays = {}
        for index, part in enumerate(capture['meshes']):
            arrays.update({f'vertices_{index}':part.mesh.vertices, f'triangles_{index}':part.mesh.triangles,
                           f'body_from_mesh_{index}':part.body_from_mesh})
        arrays['body_poses'] = np.asarray([row['poses'] for row in observer.samples])
        arrays['simulation_times'] = np.asarray([row['simulation_time'] for row in observer.samples])
        np.savez_compressed(output/'geometry_reference.npz', **arrays)
        controlled._json(output/'roundtrip_trace.json', dict(metadata=observer.metadata,
            samples=observer.samples, actions=observer.actions, operations=observer.operations,
            clearance_curves={row['label']:row.pop('fixture_clearance_curve', []) for row in operations}))
    after = source_hashes()
    samples = [] if observer is None else observer.samples
    observed_signatures = [row['frozen_fixture'] for row in samples if 'frozen_fixture' in row]
    fixture_stable = bool(observed_signatures) and all(
        canonical_hash(value) == canonical_hash(state.get('fixture_before')) for value in observed_signatures)
    checks = dict(source_unchanged=before == after, independently_valid_preflight=state.get('preflight', {}).get('passed', False),
        runtime_closed=state.get('runtime_closed', False), same_stage_prims=state.get('same_stage_prims_after_motion', False),
        frozen_fixture_every_observed_step=fixture_stable,
        world_geometry_unchanged=bool(state.get('world_before_sha256')) and state.get('world_before_sha256') == state.get('world_after_outward_sha256') == state.get('world_after_sha256'),
        owner_thread=bool(samples) and all(row['owner_thread'] for row in samples),
        contact_buffer_unsaturated=bool(samples) and not any(row['contact_buffer_full'] for row in samples),
        complete_physics_steps=bool(operations) and all(row['physics_step_coverage']['complete'] for row in operations))
    physical = physical_checks(operations) if args.mode == 'roundtrip' else {}
    summary = dict(mode=args.mode, collision_model=args.collision_model, seed=args.seed,
        frozen_fixture_sha256=FROZEN_SHA256, candidate_evaluation_did_not_revise_fixture=True,
        snapshot=baseline.serial(capture['snapshot']) if capture is not None else None,
        robot_dof_names=observer.metadata['robot_dof_names'] if observer is not None else [],
        meters_per_unit=capture['snapshot'].meters_per_unit if capture is not None else None,
        trace_artifact='roundtrip_trace.json',
        geometry_reference=dict(cooking=capture['cooking'], meshes=len(capture['meshes']),
            cooked_only=True, hand_and_both_fingers=True, all11_bodies=True) if capture is not None else None,
        contact_reference=dict(sensor_paths=observer.metadata['body_paths'],
            filter_paths=observer.metadata['contact_filter_paths']) if observer is not None else None,
        preflight=state.get('preflight'), operations=operations, acceptance_checks=physical,
        roundtrip_acceptance=bool(physical) and all(physical.values()) and all(checks.values()) and error is None,
        strict_zero_report_roundtrip=bool(physical) and all(physical.values()) and all(row['strict_success'] for row in operations),
        direct_reference_intersects_frozen_box=(any(row['fixture_clearance']['minimum_m'] == 0. for row in operations)
            if args.mode == 'direct_reference' else None),
        return_not_attempted=state.get('return_not_attempted'),
        kit_interval_ms=baseline.distribution(np.diff(observer.metadata.get('kit_times', []))*1000) if observer is not None else {},
        controller_wait_ms=baseline.distribution(np.asarray(observer.metadata['controller_waits_seconds'])*1000) if observer is not None and 'controller_waits_seconds' in observer.metadata else {},
        rss={key:value for key,value in state.items() if key.startswith('rss_')},
        world_hashes={key:value for key,value in state.items() if key.endswith('_sha256')},
        fixture_before=state.get('fixture_before'), fixture_after=state.get('fixture_after'),
        strategy_after_close=state.get('strategy_after_close'))
    payload = dict(schema_version=1, status='PASS' if error is None and all(checks.values()) else 'FAIL',
        contracts=checks, hashes=before, hashes_after=after, metrics=summary, error=error,
        meaning='PASS means a valid retained measurement. Physical acceptance separately requires both genuinely moving directions, original full-pose success, positive sampled all-body cooked-solid clearance and zero observed physical-contact steps. Strict zero-report success remains separate. The predeclared continuous witness is not a future execution certificate.')
    controlled._json(result_path, payload)
    print(json.dumps(dict(status=payload['status'], mode=args.mode, model=args.collision_model,
        roundtrip_acceptance=summary['roundtrip_acceptance'], error=error)), flush=True)
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='roundtrip')
    parser.add_argument('--collision-model', choices=('stock', 'enhanced'), default='enhanced')
    parser.add_argument('--seed', type=int, choices=(123456, 234567, 345678), default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    try:
        refuse_overwrite(args)
    except ValueError as error:
        parser.error(str(error))
    protocol, frozen = load_protocol()
    evidence = load_design_evidence(protocol)
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy='direct' if args.mode == 'direct_reference' else 'global',
        global_collision_model=args.collision_model, rrt_seed=args.seed, rrt_max_iterations=args.max_iterations)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':args.headless})
    try:
        return execute_case(args, app, protocol, frozen, evidence)
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
