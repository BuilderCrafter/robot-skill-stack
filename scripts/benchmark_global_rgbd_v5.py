#!/usr/bin/env python3
"""One native V5 source comparison, using existing cameras and static execution.

Large exposure/map/cooked-geometry captures are written only after runtime
cleanup. GT data belong exclusively to the control and independent evaluator.
No camera, shipped profile, planner threshold or robot joint state is edited.
"""
from __future__ import annotations

import argparse
from collections import deque
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np

from scripts import global_rgbd_v5_protocol as protocol
from scripts import benchmark_global_motion_v4 as baseline
from scripts import benchmark_global_roundtrip_v4_2 as shared
from scripts import benchmark_global_acceptance_v4_1 as acceptance
from scripts import diagnose_global_stabilization_v4_1 as controlled
from scripts.global_rgbd_v5_pair import MatchedSourcePlanner


def source_hashes():
    paths = [p for root in ('robot_skill_stack','scripts','tests/support')
             for p in (baseline.ROOT/root).rglob('*.py')]
    return {str(p.relative_to(baseline.ROOT)):shared.digest(p) for p in paths}


def prepare_stage(app, objects):
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    stage = shared.prepare_stage(app)
    for index, item in enumerate(objects):
        cube = UsdGeom.Cube.Define(stage, item['path'])
        cube.GetSizeAttr().Set(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(3.+index,3.,3.))
        cube.AddScaleOp().Set(Gf.Vec3f(*item['dimensions']))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(.75,.18,.08)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr().Set(True)
        PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr().Set(0.)
    return stage


def install_mapping_audit(state, budget):
    """Same existing worker/extractor for both sources; bounded borrowed inputs."""
    from robot_skill_stack.world.spatial_mapping import runtime
    from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
    original = runtime.MappingService
    def create(*args, **kwargs):
        kwargs['obstacle_settings'] = ObstacleExtractionSettings(max_obstacles=budget, support_plane_z=0.)
        service = original(*args, **kwargs)
        submit, snapshot = service.submit, service.snapshot
        state['mapping_service'] = service
        def retained_submit(payload):
            accepted = submit(payload)
            if accepted:
                state['mapping_inputs'].append(payload)
                state['mapping_submissions'].append(dict(at=time.monotonic(), generation=payload.generation,
                    frames=[dict(camera_id=f.camera_id, token=f.token, acquired_at=f.acquired_at,
                        depth_shape=f.depth.shape) for f in payload.frames]))
            return accepted
        def retained_snapshot(*values, **options):
            view = snapshot(*values, **options)
            if state.get('capturing_rgbd'):
                state['capture_view'] = view
            return view
        service.submit, service.snapshot = retained_submit, retained_snapshot
        return service
    runtime.MappingService = create
    return lambda:setattr(runtime, 'MappingService', original)


def retain_mapping_publication(state, service):
    """Published count identifies acceptance; health deliberately has no revision."""
    health = dict(service.health)
    marker = (health.get('generation'), health.get('published',0))
    if marker[1] and marker != state.get('last_mapping_publication'):
        view = service.snapshot()
        state['last_mapping_publication'] = marker
        state['mapping_publications'].append(dict(at=time.monotonic(),health=health,
            generation=view.generation,revision=view.revision,timestamp=view.timestamp,
            diagnostics=baseline.serial(service.diagnostics)))
    return health


def serialize_mapping_input(payload):
    """Use the existing trusted-local transport's immutable-value reducers."""
    from robot_skill_stack.runtime.process_worker import _TransportPickler
    stream = io.BytesIO()
    _TransportPickler(stream,protocol=5).dump(payload)
    return stream.getvalue()


def install_camera_timing(pool,state):
    """Observe existing batch calls without reading or copying sensor buffers."""
    if pool is None:
        return lambda:None
    original = pool.batch
    last_tokens = {}
    def batch(consumer,paths,**options):
        began,frames,error = time.perf_counter(),None,None
        try:
            frames = original(consumer,paths,**options)
            return frames
        except Exception as caught:
            error = type(caught).__name__
            raise
        finally:
            tokens = {} if frames is None else {frame.camera_id:tuple(frame.token) for frame in frames}
            previous = last_tokens.get(consumer,{})
            advanced = bool(tokens) and all(previous.get(key)!=token for key,token in tokens.items())
            if tokens:
                last_tokens[consumer] = tokens
            state.setdefault('camera_acquisition',[]).append(dict(consumer=consumer,
                phase=state.get('acquisition_phase','setup'),seconds=time.perf_counter()-began,
                at=time.monotonic(),frame_count=len(tokens),frames_returned=frames is not None,
                tokens_advanced=advanced,error=error))
    pool.batch = batch
    return lambda:setattr(pool,'batch',original)


def camera_timing_summary(rows):
    return {consumer:dict(calls=len(selected),successful_batches=sum(row['frames_returned'] for row in selected),
        advancing_batches=sum(row['tokens_advanced'] for row in selected),
        elapsed_ms=baseline.distribution([row['seconds']*1000 for row in selected]))
        for consumer in sorted({row['consumer'] for row in rows})
        for selected in ([row for row in rows if row['consumer']==consumer],)}


def run_native(args, app, state):
    import omni.kit.app
    from pxr import Gf, UsdGeom
    from isaacsim.core.simulation_manager import SimulationManager
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.manipulation.motion.direct import DirectRmpflowStrategy
    from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import capture_frozen_rgbd_world
    from scripts.evaluate_global_rgbd_v5 import robot_parts_from_capture, physical_obstacles_from_static
    from tests.support.isaac_async import run_kit_coroutine

    objects = protocol.fixtures(args.scenario)
    stage = prepare_stage(app, objects)
    restore_mapping = install_mapping_audit(state, args.budget)
    async def evaluate():
        bundle = observer = subscription = None
        restore_camera_timing = lambda:None
        callback = False
        try:
            path = protocol.write_profile(Path(args.artifacts_dir)/'runtime_profile.toml', args.execution_condition, args.budget)
            bundle = await build_runtime(path)
            restore_camera_timing = install_camera_timing(bundle.observation_runtime.pool,state)
            backend, service = bundle.backend, bundle.spatial_mapping
            strategy = backend.motion_strategy
            state['bundle'] = bundle
            if args.scenario != 'v43':
                backend._motion_strategy = DirectRmpflowStrategy(backend)
                try:
                    setup = await bundle.runtime.execute_async('move_to_pose',
                        target=Pose(protocol.START,protocol.ORIENTATION), position_tolerance=.025,orientation_tolerance=.10)
                    state['setup_result'] = baseline.serial(setup)
                    if not setup.ok:
                        raise RuntimeError('Ordinary direct START setup did not complete')
                finally:
                    backend._motion_strategy = strategy
            # Move only independently known external actors after common setup.
            if objects:
                view = SimulationManager.get_physics_sim_view().create_rigid_body_view([x['path'] for x in objects])
                ordered = {item['path']:item for item in objects}
                poses = np.array([[*ordered[path]['center'],0.,0.,0.,1.] for path in view.prim_paths],np.float32)
                for item in objects:
                    stage.GetPrimAtPath(item['path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*item['center']))
                view.set_transforms(poses, np.arange(len(poses),dtype=np.int32))
                state['fixture_actual_poses_before'] = np.asarray(view.get_transforms()).copy()
            # >TTL removes setup/parked-scene evidence; keep identical bounded warmup.
            warm_start = time.monotonic()
            state['acquisition_phase'] = 'warmup'
            first_published = service.health['published']
            for _ in range(2400):
                await omni.kit.app.get_app().next_update_async()
                health = retain_mapping_publication(state,service)
                if (time.monotonic()-warm_start >= 8. and health.get('published',0)-first_published >= 8
                        and health.get('fresh') is True):
                    break
                if time.monotonic()-warm_start > 30.:
                    break
            state['warmup'] = dict(seconds=time.monotonic()-warm_start, health=dict(service.health))
            observer = controlled.make_observer(bundle)
            state['observer'] = observer
            capture = acceptance.fresh_physical_capture(bundle)
            capture['body_paths'] = tuple(observer.metadata['body_paths'])
            state['capture'] = capture
            state['robot_parts'] = robot_parts_from_capture(capture['meshes'])
            state['physical_obstacles'] = physical_obstacles_from_static(capture['snapshot'])
            state['fingerprints'] = shared.cooked_fingerprints(capture['meshes'])
            observer.metadata.update(kit_times=[],controller_waits_seconds=[])
            original_end = observer.end_operation
            def serializable_end(*values, **options):
                # Keep the live fixed-time view outside JSON; save its numeric archive offline.
                frozen = strategy.last_plan_trace.pop('frozen_rgbd',None)
                try:
                    return original_end(*values,**options)
                finally:
                    if frozen is not None:
                        strategy.last_plan_trace['frozen_rgbd'] = frozen
            observer.end_operation = serializable_end
            observer.sample()
            matrices = capture['pose_matrices'](observer.samples[-1]['poses'],capture['snapshot'].meters_per_unit)
            actual_poses = dict(zip(capture['body_paths'],matrices))
            q = np.asarray(backend.robot.get_joint_positions()).copy()
            names = list(backend.robot.dof_names)
            physical_context = dict(joint_names=names,actual_start_q=q.tolist(),
                actual_start_body_poses={k:v.tolist() for k,v in actual_poses.items()},
                base_transform=actual_poses['/Franka/panda_link0'].tolist(),
                finger_positions=[float(q[names.index(name)]) for name in ('panda_finger_joint1','panda_finger_joint2')])
            state['initial_state'] = observer.state()
            state['runtime_before'] = baseline.serial(bundle.observation_runtime.diagnostics)
            def capture_rgbd():
                state['capturing_rgbd'] = True
                try:
                    world, frozen = capture_frozen_rgbd_world(backend,lambda:service,max_obstacles=args.budget)
                    state['frozen_world'] = frozen
                    state['selected_mapping_input'] = next((p for p in reversed(state['mapping_inputs'])
                        if p.generation==frozen.generation and p.frames and
                        min(f.acquired_at for f in p.frames)==frozen.timestamp),None)
                    return world
                except Exception as error:
                    from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
                    if isinstance(error,PlanningFailure):
                        state['paired_plans']['frozen_rgbd'] = dict(planner_success=False,
                            failure_code=error.code,failure_message=str(error),stage='SOURCE',
                            diagnostics=baseline.serial(error.details))
                    raise
                finally:
                    state['capturing_rgbd'] = False
            # Production RGB-D capture is still called once by the strategy.
            original_rgbd_capture = strategy._rgbd_world_capture
            def audited_capture(*values,**options):
                world = capture_rgbd()
                return world,state['frozen_world']
            strategy._rgbd_world_capture = audited_capture
            strategy._planner_factory = lambda backend,**options:MatchedSourcePlanner(backend,state=state,
                selected_source=args.execution_condition,capture_rgbd=capture_rgbd,
                gt_snapshot=capture['snapshot'],physical_context=physical_context,**options)
            update = backend._next_update
            async def wait():
                started=time.monotonic()
                try:
                    return await update()
                finally:
                    observer.metadata['controller_waits_seconds'].append(time.monotonic()-started)
            backend._next_update=wait
            bundle.world.add_physics_callback('v5_motion_capture',lambda dt:observer.sample())
            callback=True
            def on_update(event):
                observer.metadata['kit_times'].append(time.monotonic())
                retain_mapping_publication(state,service)
            subscription=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update,name='v5_measurement')
            state['world_before_snapshot'],state['world_before_sha256']=shared._world_signature(stage)
            state['request_started']=time.monotonic()
            state['acquisition_phase'] = 'measurement'
            observer.metadata['kit_times'].append(state['request_started'])
            result=await acceptance.motion(observer,'outward','move_to_pose',
                target=Pose(protocol.START if args.scenario=='v43' else protocol.GOAL,protocol.ORIENTATION))
            state['selected_result']=baseline.serial(result)
            state['runtime_after']=baseline.serial(bundle.observation_runtime.diagnostics)
            state['strategy_after_motion']=baseline.serial(strategy.diagnostics)
            state['world_after_snapshot'],state['world_after_sha256']=shared._world_signature(stage)
            if objects:
                state['fixture_actual_poses_after']=np.asarray(view.get_transforms()).copy()
            state['final_state']=observer.state()
            commands=[row for row in observer.actions if row.get('at',-1.)>=state['request_started']]
            state['request_to_first_command_seconds']=(commands[0]['at']-state['request_started']
                if commands else None)
            strategy._rgbd_world_capture=original_rgbd_capture
        finally:
            subscription=None
            restore_camera_timing()
            if bundle is not None:
                if callback:
                    bundle.world.remove_physics_callback('v5_motion_capture')
                if observer is not None:
                    observer.restore()
                bundle.close()
                state['runtime_closed']=True
                state['strategy_after_close']=baseline.serial(bundle.backend.motion_strategy.diagnostics)
    try:
        run_kit_coroutine(evaluate(),app)
    finally:
        restore_mapping()


def save_capture(args,state):
    from scripts.evaluate_global_rgbd_v5 import write_frozen_archive, write_read_view_archive
    output=Path(args.artifacts_dir)
    observer,capture=state.get('observer'),state.get('capture')
    operations=[]
    if observer is not None and capture is not None:
        operations=controlled.reduce_observations(observer,capture)
        for summary,operation in zip(operations,observer.operations):
            rows=observer.samples[operation['first_sample']:operation['last_sample']]
            summary.update(physics_step_coverage=shared.physics_step_coverage(rows),
                maximum_actual_joint_displacement_rad=acceptance.maximum_observed_displacement(
                    operation['before']['feedback']['positions'],rows,observer.indices))
        arrays={}
        for i,part in enumerate(capture['meshes']):
            arrays.update({f'vertices_{i}':part.mesh.vertices,f'triangles_{i}':part.mesh.triangles,
                f'body_from_mesh_{i}':part.body_from_mesh})
        arrays['body_poses']=np.asarray([r['poses'] for r in observer.samples])
        arrays['simulation_times']=np.asarray([r['simulation_time'] for r in observer.samples])
        np.savez_compressed(output/'geometry_reference.npz',**arrays)
        controlled._json(output/'motion_trace.json',dict(metadata=observer.metadata,samples=observer.samples,
            actions=observer.actions,operations=observer.operations))
        controlled._json(output/'geometry_reference.json',dict(
            robot_parts=baseline.serial(state.get('robot_parts',())),
            physical_obstacles=baseline.serial(state.get('physical_obstacles',())),
            urdf_path=state.get('urdf_path')))
    archive=None
    if state.get('frozen_world') is not None:
        archive=write_frozen_archive(state['frozen_world'],output,evidence_ttl=5.)
    elif state.get('capture_view') is not None:
        try:
            archive=write_read_view_archive(state['capture_view'],output,evidence_ttl=5.,
                metadata=dict(source_accepted_for_planning=False,
                    source_failure=state['paired_plans'].get('frozen_rgbd')))
        except Exception as error:
            archive=dict(source_accepted_for_planning=False,
                archive_error=f'{type(error).__name__}: {error}')
    selected=state.get('selected_mapping_input')
    if selected is None and state.get('capture_view') is not None:
        view=state['capture_view']
        selected=next((p for p in reversed(state['mapping_inputs']) if p.frames and
            p.generation==view.generation and min(f.acquired_at for f in p.frames)==view.timestamp),None)
    if selected is not None:
        (output/'accepted_mapping_input.pickle').write_bytes(serialize_mapping_input(selected))
        arrays={}
        cameras=[]
        for i,frame in enumerate(selected.frames):
            arrays.update({f'depth_{i}':frame.depth,f'K_{i}':frame.intrinsics,f'world_from_camera_{i}':frame.world_from_camera})
            if frame.rgb is not None:
                arrays[f'rgb_{i}']=frame.rgb
            cameras.append(dict(camera_id=frame.camera_id,token=frame.token,acquired_at=frame.acquired_at,
                intrinsics=frame.intrinsics,world_from_camera=frame.world_from_camera,depth_shape=frame.depth.shape))
        np.savez_compressed(output/'camera_frames.npz',**arrays)
        controlled._json(output/'camera_metadata.json',cameras)
    controlled._json(output/'mapping_publications.json',state['mapping_publications'])
    controlled._json(output/'mapping_submissions.json',state['mapping_submissions'])
    controlled._json(output/'motion_payloads.json',state['motion_payloads'])
    controlled._json(output/'physical_evaluation.json',state['physical_evaluation'])
    controlled._json(output/'camera_acquisition.json',state.get('camera_acquisition',[]))
    return operations,archive


def execute(args,app):
    before=source_hashes()
    state=dict(mapping_inputs=deque(maxlen=16),mapping_submissions=[],mapping_publications=[],
        paired_plans={},motion_payloads={},physical_evaluation={})
    error=None
    try:
        run_native(args,app,state)
    except BaseException:
        error=traceback.format_exc()
    operations,archive=[],None
    try:
        operations,archive=save_capture(args,state)
    except BaseException:
        error=(error or '')+'\nOffline save/reduction:\n'+traceback.format_exc()
    observer=state.get('observer')
    summaries={key:baseline.serial(state.get(key)) for key in ('setup_result','warmup','initial_state','final_state',
        'selected_result','fixed_joint_target','matched_joint_goal','matched_start_and_physics_instant',
        'runtime_before','runtime_after','strategy_after_motion','strategy_after_close','fingerprints',
        'fixture_actual_poses_before','fixture_actual_poses_after','world_before_sha256','world_after_sha256',
        'request_to_first_command_seconds')}
    # Per-sample physical curves remain raw; compact output retains outcomes.
    physical={key:{k:v for k,v in value.items() if k!='per_sample'} for key,value in state['physical_evaluation'].items()}
    checks=dict(runtime_closed=state.get('runtime_closed',False),source_unchanged=before==source_hashes(),
        no_internal_runtime_error=(state.get('selected_result') or {}).get('failure_code')!='internal_error',
        all11_cooked_bodies=bool(state.get('fingerprints')),
        frozen_scene=bool(state.get('world_before_sha256')) and state.get('world_before_sha256')==state.get('world_after_sha256'))
    payload=dict(schema_version=1,scenario=args.scenario,execution_condition=args.execution_condition,
        budget=args.budget,protocol=protocol.PROTOCOL,status='PASS' if error is None and all(checks.values()) else 'FAIL',
        contracts=checks,source_sha256=before,paired_plans=state['paired_plans'],physical_evaluation=physical,
        frozen_archive=archive,operations=operations,metrics=summaries,
        camera_acquisition=camera_timing_summary(state.get('camera_acquisition',[])),
        kit_interval_ms=baseline.distribution(np.diff(observer.metadata['kit_times'])*1000) if observer else {},
        controller_wait_ms=baseline.distribution(np.asarray(observer.metadata['controller_waits_seconds'])*1000) if observer else {},
        error=error,qualification='PASS means retained valid measurement; planner success, independent veto and native completion are separate outcomes. Experimental paired planning and physical-veto waits block Kit before the first command and are reported separately.')
    controlled._json(args.result,payload)
    print(json.dumps(dict(status=payload['status'],scenario=args.scenario,execution_condition=args.execution_condition,
        selected_result=summaries['selected_result'],error=error)),flush=True)
    return payload['status']=='PASS'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=protocol.SCENES,required=True)
    parser.add_argument('--execution-condition',choices=('static','frozen_rgbd'),required=True)
    parser.add_argument('--budget',type=int,choices=(64,128),default=64)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--result',required=True)
    parser.add_argument('--artifacts-dir',required=True)
    args=parser.parse_args(argv)
    if Path(args.result).exists() or (Path(args.artifacts_dir)/'runtime_profile.toml').exists():
        raise ValueError('Refusing to overwrite a native attempt')
    if np.__version__!='1.26.4':
        raise RuntimeError('Use unchanged native Isaac launcher with NumPy1.26.4')
    Path(args.artifacts_dir).mkdir(parents=True,exist_ok=True)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless})
    try:
        return 0 if execute(args,app) else 1
    finally:
        app.close()


if __name__=='__main__':
    raise SystemExit(main())
