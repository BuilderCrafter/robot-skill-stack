"""Matched direct/reactive free-space motion with independent physical references.

One case per fresh Isaac process. Fixtures and contact-report APIs are authored
in the session layer before bootstrap. The robot uses ordinary async commands.
During motion only bounded telemetry, current body poses and PhysX contacts are
captured; triangle/cuboid clearance reduction runs after runtime closure, with
no further Kit stepping, before SimulationApp's process-terminating close.
"""
import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import dataclass,fields,is_dataclass
from enum import Enum
import hashlib
from functools import partial
import json
import math
from pathlib import Path
import time
import threading
import traceback
from unittest.mock import patch

import numpy as np

from scripts.diagnose_isaac_runtime import SCENE, PROFILE, EYES, distribution
from tests.support.reactive_geometry import (
    CollisionMeshReference,link_group,reduce_clearances,convex_polygon_triangles)
from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = '/World/ReactiveEvaluation'
START = np.array([.45,-.25,.35])
GOAL = np.array([.45,.25,.35])
ORIENTATION = np.array([0.,0.,1.,0.])
FIXTURE_SETTLE_SECONDS = 5.
SCENARIOS = ('empty','static','separated','narrow','irregular','large','near','moving','partial','sparse')


def serial(value):
    # Unlike dataclasses.asdict, this also preserves immutable mapping metadata.
    if is_dataclass(value):
        return {item.name:serial(getattr(value,item.name)) for item in fields(value)}
    if isinstance(value,np.ndarray):
        return serial(value.tolist())
    if isinstance(value,np.generic):
        return serial(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,Enum):
        return value.value
    if hasattr(value,'items'):
        return {str(key):serial(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [serial(item) for item in value]
    return value


def cooked_reference(stage,prim):
    """Read the installed SDK's cooked hulls once, before the measured window."""
    import omni.physx
    from pxr import UsdUtils,PhysicsSchemaTools
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    owner = threading.get_ident()
    captured = dict(status='no synchronous result',owner_thread=True,hulls=[])

    def received(status,convexes):
        captured.update(status=str(status),owner_thread=threading.get_ident() == owner)
        if status != PhysxCollisionRepresentationResult.RESULT_VALID:
            return
        for convex in convexes:
            vertices = np.array([[v.x,v.y,v.z] for v in convex.vertices],dtype=float)
            indices,counts = [],[]
            for polygon in convex.polygons:
                start,count = int(polygon.index_base),int(polygon.num_vertices)
                counts.append(count)
                indices.extend(convex.indices[start:start+count])
            captured['hulls'].append((vertices,np.asarray(counts,dtype=np.int64),
                                      np.asarray(indices,dtype=np.int64)))
    try:
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
            run_asynchronously=False,on_result=received)
    except Exception as exc:
        captured['status'] = f'{type(exc).__name__}: {exc}'
    return captured


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple
    dimensions: tuple
    moving: bool = False


def fixtures(name):
    if name == 'empty':
        return ()
    if name == 'sparse':
        return (Box('ThinPost',(.45,0.,.35),(.035,.06,.20)),)
    if name in ('static','partial'):
        boxes = [Box('Post',(.45,0.,.35),(.09,.12,.26))]
        if name == 'partial':
            boxes.append(Box('Occluder',(.79,-.30,.53),(.08,.40,.40)))
        return tuple(boxes)
    if name in ('separated','narrow'):
        gap = .30 if name == 'separated' else .22
        return tuple(Box(label,(.45+sign*(gap+.10)/2,0.,.35),(.10,.14,.26))
                     for sign,label in ((-1,'Left'),(1,'Right')))
    if name == 'irregular':
        return (Box('LStem',(.45,0.,.35),(.08,.15,.26)),
                Box('LFoot',(.52,.10,.35),(.22,.05,.26)))
    if name == 'large':
        # Keep the broad wall on the measured crossing, beyond the START
        # forearm/elbow envelope rather than intersecting it on insertion.
        return (Box('Wall',(.55,.05,.35),(.50,.12,.60)),)
    if name == 'near':
        return (Box('OffRoute',(.72,-.25,.38),(.09,.14,.25)),)
    if name == 'moving':
        return (Box('MovingPost',(.50,0.,.35),(.08,.12,.26),True),)
    raise ValueError(name)


def source_hashes():
    paths = [SCENE,PROFILE,Path(__file__),ROOT/'tests/support/reactive_geometry.py',
             ROOT/'tests/isaac/test_reactive_motion.py',
             ROOT/'tests/support/mapping_evidence_audit.py',
             ROOT/'tests/support/reactive_contact_metrics.py']
    paths += [ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/config.py',
        'robot_skill_stack/integrations/isaac/bootstrap.py',
        'robot_skill_stack/integrations/isaac/manipulation/factory.py',
        'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py',
        'robot_skill_stack/integrations/isaac/manipulation/franka_feedback.py',
        'robot_skill_stack/integrations/isaac/manipulation/motion/direct.py',
        'robot_skill_stack/integrations/isaac/manipulation/motion/reactive.py',
        'robot_skill_stack/integrations/isaac/manipulation/motion/obstacles.py',
        'robot_skill_stack/world/spatial_mapping/obstacles.py',
        'robot_skill_stack/world/spatial_mapping/runtime.py',
        'robot_skill_stack/world/spatial_mapping/occupancy.py',
        'robot_skill_stack/world/spatial_mapping/robot_filter.py',
        'robot_skill_stack/runtime/process_worker.py',
        'robot_skill_stack/runtime/worker_process.py',
        'robot_skill_stack/integrations/isaac/observation_runtime.py')]
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def profile(args):
    path = Path(args.artifacts_dir)/'runtime_profile.toml'
    path.write_text(PROFILE.read_text()+f'''\n[motion]
strategy = "{args.strategy}"

[spatial_mapping]
enabled = true
source = "{args.source}"
bounds_min = [-0.3, -0.6, -0.02]
bounds_max = [0.85, 0.6, 1.12]
voxel_size = 0.01
camera_prim_paths = ["/World/PerceptionCamera", "/World/PerceptionCameraSide"]
stride = 8
max_rays = 1024
automatic_robot_filter = true
update_hz = {getattr(args,'mapping_hz',2.0)}
max_frame_age = 1.0
stale_after = 1.0
evidence_ttl = 5.0
''')
    return path


def depth_reference_scores(frames,objects):
    """Score partial visibility from already-acquired frames, not extra reads."""
    rows = []
    for frame in frames:
        y,x = np.indices(frame.depth[::8,::8].shape)
        y,x = y.ravel()*8,x.ravel()*8
        k = frame.intrinsics
        directions = np.column_stack(((x+.5-k[0,2])/k[0,0],
            (y+.5-k[1,2])/k[1,1],np.ones(len(x))))@frame.world_from_camera[:3,:3].T
        origin = frame.world_from_camera[:3,3]
        measured = frame.depth[y,x]
        for box in objects:
            if box.moving:
                continue  # No unmatched current pose substituted for an exposure.
            half = np.asarray(box.dimensions)/2
            lower,upper = np.asarray(box.center)-half,np.asarray(box.center)+half
            first = np.divide(lower-origin,directions,out=np.full_like(directions,-np.inf),
                where=np.abs(directions)>1e-15)
            second = np.divide(upper-origin,directions,out=np.full_like(directions,np.inf),
                where=np.abs(directions)>1e-15)
            near,far = np.minimum(first,second).max(1),np.maximum(first,second).min(1)
            parallel_outside = ((np.abs(directions)<=1e-15)
                & ((origin<lower)|(origin>upper))).any(1)
            potential = ~parallel_outside & (near>0) & (far>=near)
            valid = np.isfinite(measured)&(measured>0)
            rows.append(dict(camera_id=frame.camera_id,obstacle=box.name,
                original_timestamp=frame.acquired_at,frame_token=frame.token,
                potential_stride_grid_rays=int(potential.sum()),
                visible_stride_grid_returns=int(np.count_nonzero(potential&valid&(np.abs(measured-near)<=.015))),
                occluded_stride_grid_returns=int(np.count_nonzero(potential&valid&(measured<near-.015))),
                invalid_stride_grid_returns=int(np.count_nonzero(potential&~valid))))
    return dict(rows=rows,stride=8,depth_match_tolerance_m=.015,
        meaning='Camera stride-grid visibility only; not the mapper ray allocation, a completed volume, or a clearance proof.')


def run_native(args, app, result):
    import omni.kit.app
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import open_stage, is_stage_loading
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import (
        _triangles, _world_meters, _rigid_part, _pose_matrices)
    from tests.support.isaac_async import run_kit_coroutine
    from tests.support.multiview_native import set_viewpoint

    output = Path(args.artifacts_dir)
    objects = fixtures(args.scenario)
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open validated scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
    root_prim = UsdGeom.Xform.Define(stage,FIXTURE_ROOT).GetPrim()
    obstacle_paths = []
    for index,box in enumerate(objects):
        path = FIXTURE_ROOT+'/'+box.name
        obstacle_paths.append(path)
        cube = UsdGeom.Cube.Define(stage,path)
        cube.GetSizeAttr().Set(1.)
        # Topology exists before bootstrap; actors stay outside the map/route
        # while both selected strategies reach the same ordinary setup goal.
        cube.AddTranslateOp().Set(Gf.Vec3d(3.+index,3.,3.))
        cube.AddScaleOp().Set(Gf.Vec3f(*box.dimensions))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(.75,.18,.08)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        body.CreateKinematicEnabledAttr().Set(not box.moving)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr().Set(True)
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(100.)

    # Contact reporters are prepared before physics views/model snapshots exist.
    robot_root = stage.GetPrimAtPath('/Franka')
    rigid_paths = []
    for prim in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(str(prim.GetPath()))
    for path in rigid_paths+obstacle_paths:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)

    async def evaluate():
        bundle = None
        state = dict(active=False,phase='setup')
        raw = dict(samples=[],apply=[],tcp=[],kit=[],wait=[],runtime=[],source_frames=[],
                   sdk_updates=[],world_updates=[],forward=[],owner_callbacks=[])
        active_tasks = []
        meshes, body_paths, view, contact, subscription = [],(),None,None,None
        try:
            bundle = await build_runtime(profile(args))
            for camera,eye in zip(bundle.world_updater.provider.rig.cameras.values(),EYES):
                set_viewpoint(camera,eye,target=(.15,0.,.45))

            async def until(predicate,timeout=15.):
                started = time.monotonic()
                while not predicate():
                    if time.monotonic()-started > timeout:
                        raise AssertionError('Runtime did not become ready: '+str(bundle.spatial_mapping.health))
                    await omni.kit.app.get_app().next_update_async()

            for _ in range(60):
                await omni.kit.app.get_app().next_update_async()
            settled = time.monotonic()

            def ready():
                health = bundle.spatial_mapping.health
                snapshot = bundle.spatial_mapping.snapshot()
                packet = getattr(snapshot,'obstacles',None)
                return (health['fresh'] and health.get('timestamp',0.) > settled
                    and bundle.world_updater.health.status == 'fresh'
                    and (args.strategy != 'reactive' or packet is not None
                         and packet.generation == snapshot.generation
                         and packet.revision == snapshot.revision))

            async def warm_publications():
                threshold = settled
                seen = set()
                def current():
                    if ready():
                        health = bundle.spatial_mapping.health
                        if health['timestamp'] > threshold:
                            seen.add((health['generation'],health['timestamp']))
                    return len(seen)>=2
                await until(current)
                return dict(minimum_new_publications=2,cache_completed_at=threshold,
                    accepted_original_timestamps=sorted(stamp for _,stamp in seen),
                    health=serial(bundle.spatial_mapping.health))

            result['pre_setup_warmup'] = await warm_publications()
            result['configuration'] = dict(strategy=bundle.config.motion.strategy,
                mapping_source=args.source,mapping_hz=getattr(args,'mapping_hz',2.),
                voxel_size=.01,fixture_settle_seconds=FIXTURE_SETTLE_SECONDS,objects=serial(objects),
                start=START.tolist(),goal=GOAL.tolist(),orientation=ORIENTATION.tolist())
            setup_apply,setup_wait,setup_forward,setup_world,setup_sync = [],[],[],[],[]
            setup_robot = bundle.backend.robot
            setup_controller = bundle.backend.motion_controller
            setup_policy = setup_controller.get_motion_policy()
            saved_apply,saved_next,saved_forward,saved_world = (setup_robot.apply_action,
                bundle.backend._next_update,setup_controller.forward,setup_policy.update_world)
            setup_strategy = bundle.backend.motion_strategy
            saved_sync = getattr(setup_strategy,'_sync',None)
            setup_last_key = None
            setup_proxy_count = 0

            def setup_action(action,*positional,**keywords):
                setup_apply.append(time.monotonic())
                return saved_apply(action,*positional,**keywords)

            async def setup_update():
                started = time.perf_counter()
                try:
                    return await saved_next()
                finally:
                    setup_wait.append(time.perf_counter()-started)

            def setup_forward_action(*positional,**keywords):
                started = time.perf_counter()
                try:
                    return saved_forward(*positional,**keywords)
                finally:
                    setup_forward.append(time.perf_counter()-started)

            def setup_world_update(*positional,**keywords):
                started = time.perf_counter()
                try:
                    return saved_world(*positional,**keywords)
                finally:
                    setup_world.append(time.perf_counter()-started)

            def setup_obstacle_sync(view,packet,now):
                nonlocal setup_last_key,setup_proxy_count
                started = time.perf_counter()
                value = saved_sync(view,packet,now)
                seconds = time.perf_counter()-started
                key = (packet.generation,packet.revision)
                if key != setup_last_key:
                    setup_proxy_count = setup_strategy.diagnostics['proxy_count']
                    setup_last_key = key
                setup_sync.append(dict(seconds=seconds,generation=packet.generation,
                    revision=packet.revision,source_age_seconds=now-packet.timestamp,
                    proxy_count=setup_proxy_count))
                return value

            setup_robot.apply_action = setup_action
            bundle.backend._next_update = setup_update
            setup_controller.forward = setup_forward_action
            setup_policy.update_world = setup_world_update
            if callable(saved_sync):
                setup_strategy._sync = setup_obstacle_sync
            setup_started = time.monotonic()
            try:
                setup = await bundle.runtime.execute_async('move_to_pose',target=Pose(START,ORIENTATION))
            finally:
                setup_robot.apply_action = saved_apply
                bundle.backend._next_update = saved_next
                setup_controller.forward = saved_forward
                setup_policy.update_world = saved_world
                if callable(saved_sync):
                    setup_strategy._sync = saved_sync
            result['setup'] = serial(setup)
            result['setup_seconds'] = time.monotonic()-setup_started
            result['setup_controller'] = dict(apply_interval_ms=distribution(np.diff(setup_apply),1000),
                wait_duration_ms=distribution(setup_wait,1000),forward_ms=distribution(setup_forward,1000),
                sdk_world_update_ms=distribution(setup_world,1000),
                obstacle_sync_ms=distribution([row['seconds'] for row in setup_sync],1000),
                map_age_ms=distribution([row['source_age_seconds'] for row in setup_sync],1000),
                maximum_proxy_count=max((row['proxy_count'] for row in setup_sync),default=0))
            if not setup.ok:
                result['classification'] = 'setup_failed'
                result['setup_failure_context'] = dict(mapping=serial(bundle.spatial_mapping.health),
                    mapping_diagnostics=serial(bundle.spatial_mapping.diagnostics),
                    runtime=serial(bundle.observation_runtime.diagnostics),
                    obstacle_packet=serial(getattr(bundle.spatial_mapping.snapshot(),'obstacles',None)),
                    strategy=serial(getattr(bundle.backend.motion_strategy,'diagnostics',{})),
                    backend_ready=bundle.backend.is_ready)
                return None
            await until(ready)
            result['setup_joints_before_obstacles'] = serial(bundle.backend.robot.get_joint_positions())
            sim = SimulationManager.get_physics_sim_view()
            view = sim.create_rigid_body_view(rigid_paths+obstacle_paths)
            body_paths = tuple(view.prim_paths)
            assert set(body_paths) == set(rigid_paths+obstacle_paths)
            if objects:
                result['fixture_placement_started_at'] = time.monotonic()
                placement_view = sim.create_rigid_body_view(obstacle_paths)
                by_path = dict(zip(obstacle_paths,objects))
                desired = np.array([[*by_path[path].center,0.,0.,0.,1.]
                    for path in placement_view.prim_paths],dtype=np.float32)
                for path,box in zip(obstacle_paths,objects):
                    stage.GetPrimAtPath(path).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*box.center))
                placement_view.set_transforms(desired,np.arange(len(objects),dtype=np.uint32))
                for _ in range(60):
                    await omni.kit.app.get_app().next_update_async()
                settled = time.monotonic()
                await until(ready)
                actual_poses = np.asarray(placement_view.get_transforms()).copy()
                errors = np.linalg.norm(actual_poses[:,:3]-desired[:,:3],axis=1)
                result['fixture_setup'] = dict(
                    actors_created_before_bootstrap=True,
                    only_external_bodies_relocated_before_measurement=True,
                    robot_joint_positions_never_set=True,
                    actual_actor_poses=serial(actual_poses),
                    maximum_actor_placement_error_m=float(errors.max()),
                    fresh_map_after_relocation=serial(bundle.spatial_mapping.health))
                assert np.isfinite(actual_poses).all() and np.all(errors<=1e-4), (
                    'Physical test actors did not reach declared setup poses: '+str(errors.tolist()))
            if obstacle_paths:
                # PhysX wildcard filters pair matches one-to-one. Explicit
                # paths and one complete filter list per rigid sensor provide
                # the intended all-links by all-obstacles reference instead.
                contact = sim.create_rigid_contact_view(rigid_paths,
                    filter_patterns=[list(obstacle_paths) for _ in rigid_paths],
                    max_contact_data_count=1024)
                assert contact.sensor_count > 0 and contact.filter_count > 0
                assert set(rigid_paths) <= set(map(str,contact.sensor_paths)), 'Contact sensors omitted a robot rigid link'
                assert all(set(obstacle_paths) <= set(map(str,group)) for group in contact.filter_paths), (
                    'A robot contact sensor omitted a physical test obstacle filter')
                result['contact_reference'] = dict(sensor_paths=serial(contact.sensor_paths),
                    filter_paths=serial(contact.filter_paths),max_contact_data_count=1024,
                    contact_reporting_threshold=0.,physics_stabilization_unchanged=True,
                    layout='Every explicit robot rigid body paired with every explicit physical test obstacle')
                # Populate the new read-only tensor sensor before classifying
                # the fixture's initial state; topology was prepared earlier.
                for _ in range(2):
                    await omni.kit.app.get_app().next_update_async()

            contact_metrics_enabled = bool(getattr(args,'contact_metrics',False))
            local_com_poses = None
            if contact_metrics_enabled:
                local_com_poses = np.asarray(view.get_coms()).copy()
                result['contact_measurement'] = dict(
                    enabled=True,local_com_poses=serial(local_com_poses),
                    physics_dt_s=float(bundle.world.get_physics_dt()),
                    policy='Observational only; strict V2 success is unchanged; no acceptable-force threshold or pushing.')

            units = float(UsdGeom.GetStageMetersPerUnit(stage))
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            skipped,cooking_records = [],[]
            prims_before_cooking = tuple(str(p.GetPath()) for p in stage.Traverse())
            for prim in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
                if not prim.IsA(UsdGeom.Gprim) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                ancestor = prim
                while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    ancestor = ancestor.GetParent()
                if not ancestor:
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    skipped.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName()))
                    continue
                mesh = UsdGeom.Mesh(prim)
                rigid_world = _rigid_part(_world_meters(cache.GetLocalToWorldTransform(ancestor),units))
                mesh_world = _world_meters(cache.GetLocalToWorldTransform(prim),units)
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                cooked = cooked_reference(stage,prim)
                cooking_records.append(dict(path=str(prim.GetPath()),status=cooked['status'],
                    callback_on_owner_thread=cooked['owner_thread'],hulls=len(cooked['hulls']),
                    fallback_to_authored_triangles=not bool(cooked['hulls'])))
                if cooked['hulls']:
                    assert cooked['owner_thread'],'Synchronous cooking callback left the SDK owner thread'
                    for index,(vertices,counts,indices) in enumerate(cooked['hulls']):
                        vertices = vertices*units
                        triangles = convex_polygon_triangles(vertices,counts,indices)
                        meshes.append(CollisionMeshReference(str(prim.GetPath())+f'#convex{index}',
                            str(ancestor.GetPath()),link_group(str(ancestor.GetPath())),
                            TriangleMesh(vertices,triangles),np.linalg.inv(rigid_world)@mesh_world,
                            'physx_cooked:'+str(approximation)))
                else:
                    vertices = np.asarray(mesh.GetPointsAttr().Get(),float)*units
                    triangles = _triangles(vertices,np.asarray(mesh.GetFaceVertexCountsAttr().Get()),
                        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),np.asarray(mesh.GetHoleIndicesAttr().Get()))
                    meshes.append(CollisionMeshReference(str(prim.GetPath()),str(ancestor.GetPath()),
                        link_group(str(ancestor.GetPath())),TriangleMesh(vertices,triangles),
                        np.linalg.inv(rigid_world)@mesh_world,'usd_authored_fallback:'+str(approximation)))
            result['geometry_reference'] = dict(meshes=[dict(path=m.path,body=m.body_path,
                group=m.group,triangles=len(m.mesh.triangles),approximation=m.approximation)
                for m in meshes],unsupported_primitives=skipped,
                cooking_requests=cooking_records,
                cooked_triangulation='Checked fan of SDK-guaranteed ordered convex polygons; raw float32 coordinates retained',
                cooking_did_not_add_scene_prims=prims_before_cooking == tuple(str(p.GetPath()) for p in stage.Traverse()),
                reference_kind=('physx_cooked_convexes' if cooking_records and all(not row['fallback_to_authored_triangles']
                    for row in cooking_records) else 'mixed_or_authored_fallback'),
                limitation='Cooked hull surface distances exclude contact/rest offsets; any fallback to authored triangles is explicit. PhysX contacts are independently measured.')
            assert meshes, 'No independent robot collision mesh reference'
            required_groups = {'elbow','forearm','wrist','gripper'}
            reference_groups = {part.group for part in meshes}
            result['geometry_reference']['covered_link_groups'] = sorted(reference_groups)
            result['geometry_reference']['missing_required_link_groups'] = sorted(required_groups-reference_groups)
            assert required_groups <= reference_groups, 'Independent robot reference omitted a required arm/gripper group'
            original_apply,original_pose,original_next = (
                bundle.backend.robot.apply_action,bundle.backend.get_end_effector_pose,bundle.backend._next_update)
            last_runtime = None

            def current_runtime():
                service = bundle.spatial_mapping
                return dict(at=time.monotonic(),mapping=serial(service.health),
                    mapping_diagnostics=serial(service.diagnostics),
                    semantic=serial(bundle.observation_runtime.diagnostics['semantic']),
                    strategy=serial(getattr(bundle.backend.motion_strategy,'diagnostics',{})))

            def sample():
                nonlocal last_runtime
                if not state['active']:
                    return
                started = time.monotonic()
                item = dict(at=started,phase=state['phase'],simulation_time=float(bundle.world.current_time),
                    poses=np.asarray(view.get_transforms()).copy(),contacts=[])
                if contact_metrics_enabled:
                    item.update(body_velocities=np.asarray(view.get_velocities()).copy(),
                        physics_step=int(bundle.world.current_time_step_index),
                        physics_dt_s=float(bundle.world.get_physics_dt()))
                if contact is not None:
                    forces,points,normals,separations,counts,starts = contact.get_contact_data(1.)
                    counts,starts = np.asarray(counts),np.asarray(starts)
                    for sensor,filtered in np.argwhere(counts > 0):
                        begin,count = int(starts[sensor,filtered]),int(counts[sensor,filtered])
                        pair = dict(sensor=str(contact.sensor_paths[sensor]),
                            filter_index=int(filtered),count=count,
                            minimum_separation_m=float(np.min(np.asarray(separations)[begin:begin+count])),
                            impulse_magnitude=float(np.abs(np.asarray(forces)[begin:begin+count]).sum()))
                        if contact_metrics_enabled:
                            pair.update(obstacle=str(contact.filter_paths[sensor][filtered]),
                                normal_impulses_ns=np.asarray(forces)[begin:begin+count].copy().reshape(-1),
                                points_world_m=np.asarray(points)[begin:begin+count].copy()*units,
                                normals_world=np.asarray(normals)[begin:begin+count].copy(),
                                separations_m=np.asarray(separations)[begin:begin+count].copy().reshape(-1)*units)
                        item['contacts'].append(pair)
                    item['contact_buffer_full'] = int(counts.sum()) >= 1024
                if contact_metrics_enabled:
                    from tests.support.reactive_contact_metrics import capture_joint_feedback
                    item['joint_feedback'] = capture_joint_feedback(
                        bundle.backend.robot,state.get('last_action'))
                item['observer_seconds'] = time.monotonic()-started
                raw['samples'].append(item)
                health = bundle.spatial_mapping.health
                marker = (health['generation'],health['published'],health['failures'])
                if marker != last_runtime:
                    last_runtime = marker
                    raw['runtime'].append(current_runtime())

            def apply(action,*positional,**keywords):
                state['last_action'] = action
                row = dict(at=time.monotonic(),phase=state['phase'],measured=state['active'],
                    indices=serial(getattr(action,'joint_indices',None)),
                    positions=serial(getattr(action,'joint_positions',None)),
                    velocities=serial(getattr(action,'joint_velocities',None)))
                raw['apply'].append(row)
                try:
                    value = original_apply(action,*positional,**keywords)
                except BaseException as exc:
                    row.update(sdk_returned=False,error=f'{type(exc).__name__}: {exc}')
                    raise
                row['sdk_returned'] = True
                return value

            def pose():
                current = original_pose()
                if state['active']:
                    row = dict(at=time.monotonic(),position=current.position.copy(),
                               orientation=current.orientation.copy())
                    if contact_metrics_enabled:
                        map_view = bundle.spatial_mapping.snapshot()
                        row.update(occupancy_state=int(map_view.query_many([current.position]).states[0]),
                            map_timestamp=map_view.timestamp,map_fresh=map_view.health['fresh'])
                    raw['tcp'].append(row)
                return current

            async def next_update():
                started = time.monotonic()
                await original_next()
                if state['active']:
                    raw['wait'].append(time.monotonic()-started)
                    sample()

            def kit_tick(event):
                if state['active']:
                    raw['kit'].append(time.monotonic())

            bundle.backend.robot.apply_action = apply
            bundle.backend.get_end_effector_pose = pose
            bundle.backend._next_update = next_update
            controller = bundle.backend.motion_controller
            policy = controller.get_motion_policy()
            original_forward,original_world = controller.forward,policy.update_world
            original_tick = bundle.observation_runtime.tick

            def world_update(*positional,**keywords):
                started = time.perf_counter()
                try:
                    return original_world(*positional,**keywords)
                finally:
                    if state['active']:
                        raw['world_updates'].append(dict(at=time.monotonic(),
                            seconds=time.perf_counter()-started,origin=state.get('sdk_phase','other')))

            def forward(*positional,**keywords):
                started = time.perf_counter()
                previous = state.get('sdk_phase')
                state['sdk_phase'] = 'controller_forward'
                try:
                    return original_forward(*positional,**keywords)
                finally:
                    state['sdk_phase'] = previous
                    if state['active']:
                        raw['forward'].append(time.perf_counter()-started)

            def tick(*positional,**keywords):
                started = time.perf_counter()
                try:
                    return original_tick(*positional,**keywords)
                finally:
                    if state['active']:
                        raw['owner_callbacks'].append(time.perf_counter()-started)

            policy.update_world = world_update
            controller.forward = forward
            bundle.observation_runtime.tick = tick
            strategy = bundle.backend.motion_strategy
            if callable(getattr(strategy,'_sync',None)):
                original_sync = strategy._sync
                last_key = None
                last_proxy_count = 0
                def sync(view,packet,now):
                    nonlocal last_key,last_proxy_count
                    started = time.perf_counter()
                    previous = state.get('sdk_phase')
                    state['sdk_phase'] = 'obstacle_sync'
                    try:
                        value = original_sync(view,packet,now)
                    finally:
                        state['sdk_phase'] = previous
                    seconds = time.perf_counter()-started
                    if state['active']:
                        key = (packet.generation,packet.revision)
                        changed = key != last_key
                        diagnostics = None
                        if changed:
                            diagnostics = serial(strategy.diagnostics)
                            last_proxy_count = diagnostics['proxy_count']
                            last_key = key
                        raw['sdk_updates'].append(dict(at=time.monotonic(),seconds=seconds,
                            generation=packet.generation,revision=packet.revision,
                            original_timestamp=packet.timestamp,map_age_seconds=now-packet.timestamp,
                            proxy_count=last_proxy_count,new_revision=changed,diagnostics=diagnostics))
                    return value
                strategy._sync = sync
            subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                kit_tick,name='reactive_motion_evaluation')
            prim_paths_before = sorted(str(p.GetPath()) for p in stage.Traverse())
            # Use the same fixed post-insertion settling time for every source,
            # strategy and scenario. This permits normal evidence accumulation
            # without conditioning the experiment on occupancy or motion success.
            settle_started = time.monotonic()
            settle_updates = 0
            while time.monotonic()-settle_started < FIXTURE_SETTLE_SECONDS:
                await omni.kit.app.get_app().next_update_async()
                settle_updates += 1
            result['fixture_settling'] = dict(requested_seconds=FIXTURE_SETTLE_SECONDS,
                actual_seconds=time.monotonic()-settle_started,kit_updates=settle_updates,
                applied_after='external actor insertion and reference caching',
                mapping_health=serial(bundle.spatial_mapping.health))
            # Static reference construction can block Kit before timing. Drain
            # that startup disturbance using genuinely NEW original timestamps,
            # without extending freshness budgets or changing source cadence.
            settled = time.monotonic()
            result['post_reference_warmup'] = await warm_publications()
            initial = current_runtime()
            result['initial'] = initial
            result['initial_joints'] = serial(bundle.backend.robot.get_joint_positions())
            if contact_metrics_enabled:
                route = bundle.spatial_mapping.snapshot().query_many(np.linspace(START,GOAL,51))
                result['initial_straight_route_occupancy'] = dict(
                    samples=51,unknown=int(np.count_nonzero(route.states==0)),
                    free=int(np.count_nonzero(route.states==1)),
                    occupied=int(np.count_nonzero(route.states==2)),
                    meaning='Diagnostic point queries only; UNKNOWN is not a planning obstacle or clearance certificate.')
            state.update(active=True,phase='move')
            sample()
            moving = [i for i,box in enumerate(objects) if box.moving]
            if moving:
                moving_view = sim.create_rigid_body_view([obstacle_paths[i] for i in moving])
                moving_view.set_velocities(np.array([[-.04,0.,0.,0.,0.,0.]],dtype=np.float32),
                    np.arange(len(moving),dtype=np.uint32))
            started = time.monotonic()
            result['measurement_started_at'] = started
            movement = await bundle.runtime.execute_async('move_to_pose',target=Pose(GOAL,ORIENTATION))
            result['motion'] = dict(result=serial(movement),seconds=time.monotonic()-started)
            sample()
            state['active'] = False
            if moving:
                moving_view.set_velocities(np.zeros((len(moving),6),dtype=np.float32),
                    np.arange(len(moving),dtype=np.uint32))
            result['final'] = current_runtime()
            result['final_joints'] = serial(bundle.backend.robot.get_joint_positions())
            result['final_tcp'] = serial(original_pose())
            result['same_stage_prims_after_motion'] = prim_paths_before == sorted(
                str(p.GetPath()) for p in stage.Traverse())
            result['backend_settings'] = dict(position_tolerance=bundle.backend.position_tolerance,
                orientation_tolerance=bundle.backend.orientation_tolerance,
                max_motion_steps=bundle.backend.max_motion_steps)
            if args.scenario == 'empty' and movement.ok:
                state['phase'] = 'home'
                home = await bundle.runtime.execute_async('home')
                result['home'] = serial(home)
            cached = bundle.observation_runtime.pool._cache
            raw['source_frames'] = [value[1] for value in cached.values() if value[1] is not None]
            result['final_mapping_view'] = bundle.spatial_mapping.snapshot()
            result['capture'] = dict(body_paths=body_paths,objects=serial(objects))
            measured_actions = [row for row in raw['apply'] if row['measured']]
            result['controller'] = dict(
                apply_interval_ms=distribution(np.diff([x['at'] for x in measured_actions]),1000),
                wait_duration_ms=distribution(raw['wait'],1000),
                kit_interval_ms=distribution(np.diff(raw['kit']),1000),
                forward_ms=distribution(raw['forward'],1000),
                owner_callback_ms=distribution(raw['owner_callbacks'],1000),
                sdk_world_update_ms=distribution([row['seconds'] for row in raw['world_updates']],1000),
                reference_observer_ms=distribution([s['observer_seconds'] for s in raw['samples']],1000),
                waits_over_100_ms=sum(value > .1 for value in raw['wait']))

            if getattr(args,'lifecycle',False):
                assert args.strategy == 'reactive' and args.scenario == 'empty'
                checks,episodes = {},[]
                result['lifecycle'] = dict(checks=checks,episodes=episodes)
                backend,service = bundle.backend,bundle.spatial_mapping

                def held_at_end(before):
                    actions = raw['apply'][before:]
                    if len(actions)<2:
                        return False
                    previous,last = actions[-2:]
                    return (last['sdk_returned'] and last['positions'] == previous['positions']
                        and last['indices'] == previous['indices'] and last['velocities'] is not None
                        and all(value == 0. for value in last['velocities']))

                async def begin(label):
                    await until(ready)
                    state['phase'] = label
                    before = len(raw['apply'])
                    task = asyncio.create_task(bundle.runtime.execute_async(
                        'move_to_pose',target=Pose(START,ORIENTATION)))
                    active_tasks.append(task)
                    await until(lambda: len(raw['apply'])>before or task.done())
                    assert len(raw['apply'])>before and not task.done(), 'Fault was not injected during active movement'
                    return task,before

                async def recover(label):
                    await until(ready)
                    response = await bundle.runtime.execute_async('home')
                    checks[label+'_recovery_home'] = response.ok
                    episodes.append(dict(name=label+'_recovery',result=serial(response),
                        backend_ready=backend.is_ready,gripper_feedback=serial(backend.gripper_feedback())))
                    assert response.ok,response.message

                # Intentional contact is rejected before arm or gripper commands.
                before = len(raw['apply'])
                rejected_pick = await bundle.runtime.execute_async('pick',object_id='not_required_for_capability_check')
                rejected_place = await bundle.runtime.execute_async('place',target=Pose([.45,.2,.025]))
                checks['pick_place_explicitly_unsupported_without_actuation'] = (
                    not rejected_pick.ok and not rejected_place.ok
                    and rejected_pick.failure_code.value == 'motion_unsupported'
                    and rejected_place.failure_code.value == 'motion_unsupported'
                    and len(raw['apply']) == before)
                assert checks['pick_place_explicitly_unsupported_without_actuation']

                task,before = await begin('cancel')
                count = len(raw['apply'])
                competing = await backend.move_to_pose_async(Pose(GOAL,ORIENTATION))
                checks['competing_motion_rejected_without_action'] = (
                    not competing.ok and 'already in progress' in competing.message
                    and len(raw['apply']) == count)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    checks['cancelled_error_propagated'] = True
                else:
                    raise AssertionError('Cancellation did not propagate')
                checks['cancellation_final_hold_and_owner_release'] = (
                    held_at_end(before) and not backend._motion_active and not bundle.runtime.busy)
                assert checks['cancellation_final_hold_and_owner_release']
                await recover('cancellation')

                async def fault(label,inject):
                    task,before = await begin(label)
                    generation = service.generation
                    await inject()
                    response = await task
                    check = (not response.ok and response.details.get('world_state_unavailable') is True
                        and held_at_end(before) and not backend._motion_active and not bundle.runtime.busy)
                    checks[label+'_stopped_with_hold'] = check
                    episodes.append(dict(name=label,result=serial(response),generation_before=generation,
                        generation_after=service.generation,strategy=serial(backend.motion_strategy.diagnostics)))
                    assert check,label+': '+response.message

                async def source_failure():
                    service.note_source_failure('native injected source failure')
                await fault('source_failure',source_failure)
                await recover('source_failure')

                async def map_reset():
                    service.reset()
                generation = service.generation
                await fault('map_reset',map_reset)
                await recover('map_reset')
                checks['new_generation_observed_after_reset'] = (
                    service.generation > generation
                    and backend.motion_strategy.diagnostics['map_generation'] == service.generation)

                worker = service._worker
                previous_failures = service.health['worker_failures']
                with patch.object(worker,'_process',side_effect=RuntimeError('native injected mapping worker failure')):
                    async def worker_failure():
                        # A deterministic lifecycle fault, not a production rate
                        # change: make the next admitted map attempt occur now.
                        bundle.observation_runtime._mapping_due = 0.
                        await until(lambda: service.health['worker_failures'] > previous_failures)
                    await fault('worker_failure',worker_failure)
                await recover('worker_failure')

                task,before = await begin('stop_play')
                old_generation = service.generation
                await bundle.world.stop_async()
                response = await task
                checks['stop_did_not_continue_motion'] = not response.ok and not backend._motion_active
                episodes.append(dict(name='stop',result=serial(response),hold_requested=held_at_end(before),
                    strategy=serial(backend.motion_strategy.diagnostics)))
                assert checks['stop_did_not_continue_motion']
                await bundle.world.play_async()
                await until(ready)
                checks['play_map_advanced_generation'] = service.generation > old_generation
                post_play = await bundle.runtime.execute_async('home')
                episodes.append(dict(name='post_play_before_reinitialization',result=serial(post_play),
                    backend_ready=backend.is_ready,gripper_feedback=serial(backend.gripper_feedback()),
                    mapping_generation=service.generation,
                    limitation='Isaac5.1 Stop invalidates articulation/gripper views; fresh mapping alone does not reinitialize arm SDK handles.'))
                checks['invalid_sdk_after_play_rejected_explicitly'] = (
                    not backend.is_ready and not post_play.ok
                    and post_play.details.get('world_state_unavailable') is True)
                assert checks['invalid_sdk_after_play_rejected_explicitly'],post_play.message
                old_bundle,old_backend,old_service = bundle,backend,service
                bundle = await build_runtime(profile(args))
                backend,service = bundle.backend,bundle.spatial_mapping
                # Preserve the unmeasured lifecycle action trace on the new
                # backend. The SDK articulation may be reused, but its handles
                # are initialized through the normal composition root.
                if backend.robot is not old_backend.robot:
                    original_apply = backend.robot.apply_action
                backend.robot.apply_action = apply
                checks['clean_reinit_has_distinct_strategy_and_service'] = (
                    backend.motion_strategy is not old_backend.motion_strategy and service is not old_service)
                checks['old_registry_and_source_closed_after_reinit'] = (
                    old_backend.motion_strategy.diagnostics['closed']
                    and old_backend.motion_strategy.diagnostics['proxy_count'] == 0
                    and old_service.health['closed'])
                await recover('clean_reinitialization')
                checks['new_runtime_uses_own_mapping_generation'] = (
                    backend.motion_strategy.diagnostics['map_generation'] == service.generation)

                task,before = await begin('shutdown')
                publication_count = service.health['published']
                bundle.close()
                response = await task
                checks['shutdown_stopped_motion_with_hold'] = (
                    not response.ok and held_at_end(before) and not backend._motion_active)
                checks['shutdown_closed_registry_without_proxies'] = (
                    backend.motion_strategy.diagnostics['closed']
                    and backend.motion_strategy.diagnostics['proxy_count'] == 0)
                checks['shutdown_no_late_mapping_publication'] = service.health['published'] == publication_count
                checks['all_lifecycle_steps_completed'] = True
                assert all(checks.values()),checks
            return dict(raw=raw,meshes=meshes,body_paths=body_paths,units=units,
                        objects=objects,pose_matrices=_pose_matrices)
        finally:
            state['active'] = False
            subscription = None
            if bundle is not None:
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc:
                            result.setdefault('cleanup_errors',[]).append(str(exc))
                bundle.close()
                result['mapping_closed'] = bundle.spatial_mapping.health['closed']
                result['closed_strategy'] = serial(getattr(bundle.backend.motion_strategy,'diagnostics',{}))

    return run_kit_coroutine(evaluate(),app)


def finish(args,result,capture):
    if capture is None:
        result['avoidance_success'] = False
        return
    raw,meshes,paths = capture['raw'],capture['meshes'],capture['body_paths']
    matrices = [capture['pose_matrices'](sample['poses'],capture['units']) for sample in raw['samples']]
    indices = {path:i for i,path in enumerate(paths)}
    boxes = [[(FIXTURE_ROOT+'/'+box.name,poses[indices[FIXTURE_ROOT+'/'+box.name]],box.dimensions)
              for box in capture['objects']] for poses in matrices]
    started = time.monotonic()
    result['clearance'] = reduce_clearances(meshes,paths,matrices,boxes)
    result['initial_clearance'] = reduce_clearances(meshes,paths,matrices[:1],boxes[:1])
    result['offline_geometry_seconds'] = time.monotonic()-started
    arrays = dict(body_poses=np.asarray([sample['poses'] for sample in raw['samples']]),
                  simulation_times=np.asarray([sample['simulation_time'] for sample in raw['samples']]))
    for index,mesh in enumerate(meshes):
        arrays[f'vertices_{index}'] = mesh.mesh.vertices
        arrays[f'triangles_{index}'] = mesh.mesh.triangles
        arrays[f'body_from_mesh_{index}'] = mesh.body_from_mesh
    result['camera_visibility'] = depth_reference_scores(raw['source_frames'],capture['objects'])
    for index,frame in enumerate(raw['source_frames']):
        arrays[f'depth_{index}'] = frame.depth
        arrays[f'intrinsics_{index}'] = frame.intrinsics
        arrays[f'world_from_camera_{index}'] = frame.world_from_camera
    np.savez_compressed(Path(args.artifacts_dir,'geometry_reference.npz'),**arrays)
    # The explicit final hold sample can repeat a physical step. Keep raw traces
    # intact while contact aggregates count each simulation step exactly once.
    unique_samples = list({sample['simulation_time']:sample for sample in raw['samples']}.values())
    contacts = [pair for sample in unique_samples for pair in sample['contacts']]
    result['contacts'] = dict(frames=sum(bool(s['contacts']) for s in unique_samples),
        initial_reported_points=sum(pair['count'] for pair in raw['samples'][0]['contacts']),
        reported_points=sum(pair['count'] for pair in contacts),
        penetrating_pairs=sum(pair['minimum_separation_m'] < -1e-5 for pair in contacts),
        minimum_reported_separation_m=min((pair['minimum_separation_m'] for pair in contacts),default=None),
        maximum_pair_impulse=max((pair['impulse_magnitude'] for pair in contacts),default=0.),
        links=sorted({pair['sensor'] for pair in contacts}),
        buffer_saturated=any(s.get('contact_buffer_full',False) for s in raw['samples']))
    positions = np.asarray([row['position'] for row in raw['tcp']])
    steps = np.linalg.norm(np.diff(positions,axis=0),axis=1) if len(positions)>1 else np.array([])
    errors = np.linalg.norm(positions-GOAL,axis=1) if len(positions) else np.array([])
    result['path'] = dict(length_m=float(steps.sum()),
        straight_distance_m=float(np.linalg.norm(GOAL-START)),
        maximum_x_detour_m=None if not len(positions) else float(np.max(np.abs(positions[:,0]-.45))),
        maximum_z_detour_m=None if not len(positions) else float(np.max(np.abs(positions[:,2]-.35))),
        progress_reversals=int(np.count_nonzero(np.diff(errors) > .001)))
    observed_path_states = [row['occupancy_state'] for row in raw['tcp'] if 'occupancy_state' in row]
    if observed_path_states:
        result['path_occupancy'] = dict(samples=len(observed_path_states),
            unknown=observed_path_states.count(0),free=observed_path_states.count(1),
            occupied=observed_path_states.count(2),
            unknown_fraction=observed_path_states.count(0)/len(observed_path_states),
            meaning='Sparse TCP point queries at observation time; these are not swept-volume clearance tests.')
    initial,final = result['initial'],result['final']
    seconds = result['motion']['seconds']
    result['observation_rates'] = dict(
        semantic_hz=(final['semantic']['published']-initial['semantic']['published'])/seconds,
        mapping_hz=(final['mapping']['published']-initial['mapping']['published'])/seconds,
        meaning='Publications during this command; short windows have phase-boundary count variation.')
    result['obstacle_updates'] = dict(
        maximum_proxy_count=max((row['proxy_count'] for row in raw['sdk_updates']),default=0),
        accepted_revisions=len({(row['generation'],row['revision']) for row in raw['sdk_updates']}),
        kit_sync_ms=distribution([row['seconds'] for row in raw['sdk_updates']],1000),
        map_age_ms=distribution([row['map_age_seconds'] for row in raw['sdk_updates']],1000),
        extraction=final['mapping_diagnostics'].get('obstacle_extraction',{}))
    motion = result['motion']['result']
    contact = result['contacts']['reported_points'] > 0
    success = motion['status'] == 'success'
    reference_intersection = (bool(capture['objects']) and result['clearance']['minimum_m'] == 0.)
    initial_reference_intersection = (bool(capture['objects']) and result['initial_clearance']['minimum_m'] == 0.)
    result['reference_contact_disagreement'] = reference_intersection and not contact
    if result['contacts']['initial_reported_points']:
        result['classification'] = 'initial_state_contact'
    elif initial_reference_intersection:
        result['classification'] = 'initial_state_reference_intersection'
    elif success:
        result['classification'] = ('target_reached_with_contact' if contact else
            'target_reached_with_reference_intersection' if reference_intersection else 'target_reached_clear')
    elif contact:
        result['classification'] = 'contact_and_target_not_reached'
    elif reference_intersection:
        result['classification'] = 'reference_intersection_and_target_not_reached'
    elif motion.get('status') == 'timeout':
        result['classification'] = 'blocked_timeout'
    else:
        result['classification'] = 'motion_failed'
    result['avoidance_success'] = success and not contact and not reference_intersection
    if getattr(args,'contact_metrics',False):
        from tests.support.reactive_contact_metrics import analyze_contact_samples
        result['contact_severity'] = analyze_contact_samples(raw['samples'],paths,
            result['contact_measurement']['local_com_poses'],meters_per_unit=capture['units'],
            goal_status=motion['status'],failure_code=motion.get('failure_code'),
            minimum_clearance_m=result['clearance']['minimum_m'],
            strict_v2_success=result['avoidance_success'])
    if result.get('final_mapping_view') is not None:
        view = result.pop('final_mapping_view')
        states = view.states
        result['map_states'] = dict(zip(('free','occupied','unknown'),
            [int(np.count_nonzero(states == key)) for key in (1,2,0)]))
    # Retain large per-step evidence outside the compact result/committed assets.
    raw['source_frames'] = [dict(camera_id=f.camera_id,token=f.token,acquired_at=f.acquired_at,
        shape=f.depth.shape,valid_depth_pixels=int(np.count_nonzero(np.isfinite(f.depth)&(f.depth>0))))
        for f in raw['source_frames']]
    Path(args.artifacts_dir,'trace.json').write_text(json.dumps(serial(raw),indent=2,allow_nan=False)+'\n')


def execute_case(args,app):
    """Measure, close the runtime, reduce geometry and persist before app close."""
    output = Path(args.artifacts_dir);output.mkdir(parents=True,exist_ok=True)
    before = source_hashes()
    result,capture,error = {},None,None
    try:
        with ExitStack() as diagnostic_context:
            audit_dir = getattr(args,'mapping_audit_dir',None)
            if audit_dir:
                from tests.support.mapping_evidence_audit import AuditedMappingProcessor
                diagnostic_context.enter_context(patch(
                    'robot_skill_stack.world.spatial_mapping.runtime.MappingProcessor',
                    partial(AuditedMappingProcessor,audit_dir=str(Path(audit_dir).resolve()),
                        capture_inputs=bool(getattr(args,'audit_capture_inputs',False)),
                        max_captures=getattr(args,'audit_max_captures',80))))
                result['mapping_audit'] = dict(directory=str(Path(audit_dir).resolve()),
                    captures_enabled=bool(getattr(args,'audit_capture_inputs',False)),
                    full_capture_excluded_from_performance_claims=bool(getattr(args,'audit_capture_inputs',False)))
            capture = run_native(args,app,result)
    except Exception:
        error = traceback.format_exc()
    if error is None:
        try:
            finish(args,result,capture)
        except Exception:
            error = traceback.format_exc()
    result.pop('final_mapping_view',None)
    contracts = dict(source_unchanged=before == source_hashes(),
        ordinary_async_motion=True,
        numeric_proxies_did_not_add_scene_prims=result.get('same_stage_prims_after_motion',False),
        mapping_closed=result.get('mapping_closed',False),
        physical_reference_captured=capture is not None,
        cooking_did_not_add_scene_prims=result.get('geometry_reference',{}).get('cooking_did_not_add_scene_prims',False),
        contact_buffer_not_saturated=not result.get('contacts',{}).get('buffer_saturated',False))
    payload = dict(status='PASS' if error is None and all(contracts.values()) else 'FAIL',
        contracts=contracts,metrics=result,error=error,hashes=before,
        meaning='PASS means valid measurement, not collision avoidance. avoidance_success requires goal success, no PhysX contact and no cooked-reference intersection; blocked/timeout is never avoidance success.')
    Path(args.result).parent.mkdir(parents=True,exist_ok=True)
    Path(args.result).write_text(json.dumps(serial(payload),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=payload['status'],classification=result.get('classification'),
        avoidance_success=result.get('avoidance_success'),error=error)),flush=True)
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',choices=('direct','reactive'),required=True)
    parser.add_argument('--source',choices=('ground_truth','rgbd'),default='ground_truth')
    parser.add_argument('--mapping-hz',type=float,default=2.,help='Explicit experiment cadence; library/default remains 2 Hz')
    parser.add_argument('--scenario',choices=SCENARIOS,default='static')
    parser.add_argument('--lifecycle',action='store_true',help='Inject controlled reactive free-space lifecycle faults')
    parser.add_argument('--mapping-audit-dir',help='Diagnostic-only worker timing/capture directory')
    parser.add_argument('--audit-capture-inputs',action='store_true',help='Bounded full worker captures; excluded from performance claims')
    parser.add_argument('--audit-max-captures',type=int,default=80)
    parser.add_argument('--contact-metrics',action='store_true',help='Observe contact speed, impulse and joint feedback without changing control')
    parser.add_argument('--result',required=True)
    parser.add_argument('--artifacts-dir',required=True)
    args = parser.parse_args(argv)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':False})
    try:
        return execute_case(args,app)
    finally:
        app.close()


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
