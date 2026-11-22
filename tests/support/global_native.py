"""Native regression fixtures and independent capture; never a planning source."""
from dataclasses import fields,is_dataclass
from enum import Enum
import math,threading,time
from pathlib import Path
import numpy as np
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from robot_skill_stack.runtime.worker import BoundedWorker
ROOT=Path(__file__).resolve().parents[2]
SCENE=ROOT/'scenes/multiview_perception_final.usda'
FIXTURE_ROOT='/World/ReactiveEvaluation'  # Frozen historical physical fixture path.
START=(.45,-.25,.35)
GOAL=(.45,.25,.35)
ORIENTATION=(0.,0.,1.,0.)


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


def _capture_robot_meshes(stage, robot_root):
    from pxr import Usd, UsdGeom, UsdPhysics
    from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _rigid_part, _world_meters
    from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh
    from tests.support.physical_geometry import CollisionMeshReference, convex_polygon_triangles, link_group
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache, meshes, records = UsdGeom.XformCache(Usd.TimeCode.Default()), [], []
    for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f'Uncovered physical robot collision primitive: {prim.GetPath()}')
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if not ancestor:
            raise ValueError(f'Robot collider has no physical body: {prim.GetPath()}')
        captured = cooked_reference(stage, prim)
        if not captured['hulls'] or not captured['owner_thread']:
            raise ValueError(f'Actual owner-thread cooked hull unavailable: {prim.GetPath()}: {captured["status"]}')
        body = _rigid_part(_world_meters(cache.GetLocalToWorldTransform(ancestor), units))
        mesh_world = _world_meters(cache.GetLocalToWorldTransform(prim), units)
        records.append(dict(path=str(prim.GetPath()), body=str(ancestor.GetPath()),
            status=captured['status'], cooked_hulls=len(captured['hulls'])))
        for index, (vertices, counts, indices) in enumerate(captured['hulls']):
            vertices = vertices*units
            triangles = convex_polygon_triangles(vertices, counts, indices)
            meshes.append(CollisionMeshReference(str(prim.GetPath())+f'#convex{index}',
                str(ancestor.GetPath()), link_group(str(ancestor.GetPath())),
                TriangleMesh(vertices, triangles), np.linalg.inv(body)@mesh_world,
                'physx_cooked:actual_convex_solid'))
    body_names = {part.body_path.rsplit('/', 1)[-1] for part in meshes}
    required = {'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
        'panda_hand', 'panda_leftfinger', 'panda_rightfinger'}
    if not required <= body_names:
        raise ValueError(f'Cooked physical reference omitted arm/hand/fingers: {sorted(required-body_names)}')
    return meshes, records


def prepare_stage(app, *, obstacle=None):
    """Session-only scene and report setup shared by the new V4.2 captures."""
    import omni.usd
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import is_stage_loading, open_stage

    if not open_stage(str(SCENE)):
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
    UsdGeom.Xform.Define(stage, str(Path(item['prim_path']).parent) if item else FIXTURE_ROOT)
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


def sample_trajectory(plan):
    duration = float(plan.duration)
    velocity = np.asarray(plan.velocity_limits,float)
    if (not np.isfinite(duration) or duration<0. or velocity.shape!=(7,)
            or not np.isfinite(velocity).all() or np.any(velocity<=0.)):
        raise PlanningFailure('research_veto_budget','Invalid trajectory sampling limits')
    step = min(1/240., .01/float(np.max(velocity)))
    count = max(2,int(np.ceil(duration/step))+1)
    if count > 12000:
        raise PlanningFailure('research_veto_budget', 'Independent trajectory sample budget exceeded')
    times = np.linspace(0., duration, count)
    positions = np.asarray([plan.trajectory.get_joint_targets(float(t))[0] for t in times])
    return times, positions


def bounded_evaluate(evaluator, payload, *, timeout=120.):
    """Existing bounded worker, with explicitly measured pre-command Kit wait.

    No simulation update is allowed here: the unchanged executor requires its
    actual starting state and physics time to survive synchronous planning.
    This experimental wait is not production source/planner latency.
    """
    started = time.perf_counter()
    worker = BoundedWorker(evaluator, name='v5-independent-physical-veto')
    try:
        if not worker.submit(payload, tag='physical-veto'):
            raise RuntimeError('Physical worker refused its only job')
        event = threading.Event()
        while time.perf_counter()-started < timeout:
            result = worker.poll()
            if result is not None:
                if result.error:
                    return dict(execution_allowed=False, status='VETO_EVALUATION_ERROR', error=result.error,
                        experimental_owner_wait_seconds=time.perf_counter()-started)
                return dict(result.value, experimental_owner_wait_seconds=time.perf_counter()-started,
                    worker_seconds=result.duration, queue_seconds=result.started_at-result.queued_at)
            event.wait(.005)
        return dict(execution_allowed=False, status='VETO_EVALUATION_TIMEOUT',
            experimental_owner_wait_seconds=time.perf_counter()-started)
    finally:
        worker.close()


def contact_rows(view, units):
    impulses, points, normals, separations, counts, starts = view.get_contact_data(1.)
    counts, starts = np.asarray(counts), np.asarray(starts)
    rows = []
    for sensor, filtered in np.argwhere(counts > 0):
        first, count = int(starts[sensor, filtered]), int(counts[sensor, filtered])
        signed = np.asarray(separations)[first:first+count].copy().reshape(-1)*units
        normal = np.asarray(impulses)[first:first+count].copy().reshape(-1)
        other = str(view.filter_paths[sensor][filtered])
        rows.append(dict(sensor=str(view.sensor_paths[sensor]), obstacle=other,
            kind='self' if other == '/Franka' or other.startswith('/Franka/') else 'external',
            count=count, minimum_separation_m=float(signed.min()), impulse_magnitude=float(np.abs(normal).sum()),
            normal_impulses_ns=normal, separations_m=signed,
            points_world_m=np.asarray(points)[first:first+count].copy()*units,
            normals_world=np.asarray(normals)[first:first+count].copy()))
    return serial(rows), int(counts.sum()) >= 4096
