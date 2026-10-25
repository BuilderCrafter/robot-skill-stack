#!/usr/bin/env python3
"""One fresh native static global-planning case, with independent physics evidence.

The shipped scene and controller defaults are never written. Ordinary direct
commands reach START before external fixtures enter the route. Dense cooked
geometry reduction runs only after the runtime has closed, outside live timing.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.diagnose_isaac_runtime import SCENE, distribution
from scripts.diagnose_reactive_motion import (
    Box, FIXTURE_ROOT, GOAL, ORIENTATION, START, cooked_reference,
    fixtures as previous_fixtures, serial)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('empty', 'near', 'static', 'narrow', 'large', 'blocked', 'unreachable')
STRATEGIES = ('direct', 'reactive', 'global')
POSITION_TOLERANCE = .025
ORIENTATION_TOLERANCE = .10


def fixtures(name):
    if name == 'blocked':
        return (Box('BlockedGoal', tuple(GOAL), (.18, .18, .18)),)
    if name == 'unreachable':
        return ()
    if name not in SCENARIOS:
        raise ValueError(f'Unknown V4 static scenario {name!r}')
    return previous_fixtures(name)


def target_position(name):
    if name not in SCENARIOS:
        raise ValueError(f'Unknown V4 static scenario {name!r}')
    return np.array([2., .25, .35]) if name == 'unreachable' else GOAL.copy()


def write_profile(args):
    """Static observations need neither cameras nor an RGB-D mapping worker."""
    if args.strategy not in STRATEGIES:
        raise ValueError('Unknown motion strategy')
    mapping = args.strategy == 'reactive' or bool(getattr(args, 'enable_mapping', False))
    text = f'''# Generated V4 case; shipped profiles remain unchanged.
[robot]
backend = "franka"
prim_path = "/Franka"
[world]
provider = "ground_truth"
update_hz = 5.0
[scene]
support_plane_z = 0.0
workspace_min = [-0.3, -0.6, -0.02]
workspace_max = [1.0, 0.6, 1.12]
[ground_truth]
objects_root = "{FIXTURE_ROOT}"
[motion]
strategy = "{args.strategy}"
rrt_seed = {args.seed}
rrt_max_iterations = {args.max_iterations}
[spatial_mapping]
enabled = {str(mapping).lower()}
source = "ground_truth"
bounds_min = [-0.3, -0.6, -0.02]
bounds_max = [1.0, 0.6, 1.12]
voxel_size = 0.01
update_hz = 2.0
excluded_prim_paths = ["/Franka"]
stale_after = 1.0
evidence_ttl = 5.0
'''
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'runtime_profile.toml'
    path.write_text(text)
    # Use the production parser; CLI and programmatic settings share validation.
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    load_scene_config(path)
    return path


def source_hashes():
    names = [SCENE, Path(__file__), ROOT/'tests/support/reactive_geometry.py',
        ROOT/'tests/support/reactive_contact_metrics.py', ROOT/'scripts/diagnose_reactive_motion.py',
        ROOT/'scripts/obstacle_representation_v3_cylinder.py']
    for prefix in ('robot_skill_stack/integrations/isaac/manipulation',
                   'robot_skill_stack/manipulation/motion'):
        names.extend((ROOT/prefix).rglob('*.py'))
    names.extend(ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/config.py',
        'robot_skill_stack/integrations/isaac/bootstrap.py',
        'robot_skill_stack/integrations/isaac/observation_runtime.py',
        'robot_skill_stack/world/spatial_mapping/occupancy.py',
        'robot_skill_stack/world/spatial_mapping/runtime.py',
        'robot_skill_stack/world/spatial_mapping/obstacles.py',
        'run_isaac_python.sh', 'run_isaac_sim.sh'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(names)) if p.is_file()}


def reference_match(reference, configuration, initial_joints, *, hashes, physical_world,
                    joint_tolerance_rad=.01):
    """Compare actual states and physics conditions, allowing strategy/seed changes."""
    old = reference.get('metrics') or {}
    old_q, q = np.asarray(old.get('initial_joints', []), float), np.asarray(initial_joints, float)
    shape = old_q.shape == q.shape and q.ndim == 1 and len(q) >= 7
    error = float(np.max(np.abs(old_q-q))) if shape and np.isfinite(old_q).all() and np.isfinite(q).all() else None
    first = old.get('configuration') or {}
    fields = ('scenario', 'start', 'goal', 'orientation', 'objects',
              'position_tolerance_m', 'orientation_tolerance_rad', 'max_motion_steps')
    checks = {name: first.get(name) == configuration.get(name) for name in fields}
    checks.update(actual_initial_joints=error is not None and error <= joint_tolerance_rad,
        source_manifest=bool(reference.get('hashes')) and reference['hashes'] == hashes,
        physical_world=old.get('physical_world') == physical_world,
        reference_measurement_valid=reference.get('status') == 'PASS')
    return dict(matched=all(checks.values()), checks=checks,
        maximum_joint_error_rad=error, joint_tolerance_rad=joint_tolerance_rad)


def joint_path_deviation(actual, route):
    """Nearest sampled q-to-joint-polyline distance; no timing/tracking claim."""
    points, path = np.asarray(actual, float), np.asarray(route, float)
    if (points.ndim != 2 or path.ndim != 2 or points.shape[1:] != (7,)
            or path.shape[1:] != (7,) or not len(points) or not len(path)
            or not np.isfinite(points).all() or not np.isfinite(path).all()):
        return dict(count=0, reason='No compatible measured seven-joint points and RRT route')
    if len(path) == 1:
        distances = np.linalg.norm(points-path[0], axis=1)
    else:
        starts, directions = path[:-1], np.diff(path, axis=0)
        denominator = np.sum(directions*directions, axis=1)
        delta = points[:, None]-starts[None]
        t = np.divide(np.einsum('nsi,si->ns', delta, directions), denominator[None],
            out=np.zeros((len(points), len(starts))), where=denominator[None] > 0)
        distances = np.linalg.norm(delta-np.clip(t, 0., 1.)[:, :, None]*directions[None], axis=2).min(axis=1)
    return dict(distribution(distances), units='joint-space Euclidean radians',
        meaning='Distance to the geometric joint polyline; distinct from time-matched command tracking and Cartesian swept clearance.')


def _is_base(path):
    return path.rstrip('/').rsplit('/', 1)[-1] in ('Franka', 'panda_link0')


def contact_summary(samples, *, exclude_mounted_base=False):
    unique = list({(row['physics_step'], row['simulation_time']): row for row in samples}.values())
    pairs = [pair for row in unique for pair in row['contacts']
        if not exclude_mounted_base or not _is_base(pair['sensor'])]
    return dict(reported_points=sum(pair['count'] for pair in pairs),
        reported_pairs=len(pairs),
        penetrating_pairs=sum(pair['minimum_separation_m'] < -1e-5 for pair in pairs),
        minimum_reported_separation_m=min((pair['minimum_separation_m'] for pair in pairs), default=None),
        maximum_normal_pair_impulse_ns=max((pair['impulse_magnitude'] for pair in pairs), default=0.),
        links=sorted({pair['sensor'] for pair in pairs}),
        contact_buffer_saturated=any(row.get('contact_buffer_full', False) for row in unique),
        mounted_base_excluded=exclude_mounted_base,
        meaning='Reported points include positive-separation contact-offset proximity; not unique physical impacts.')


def reduce_clearances(meshes, body_paths, transforms, obstacles):
    """Actual cooked robot solids against exact physical boxes and plane halfspaces."""
    from robot_skill_stack.common.rotations import quat_matrix
    from scripts.obstacle_representation_v3_cylinder import convex_distance_details
    indices = {path: i for i, path in enumerate(body_paths)}
    boxes, planes = [], []
    for item in obstacles:
        rotation = quat_matrix(item.orientation)
        if item.source_kind == 'Plane':
            top = np.asarray(item.position)+rotation[:, 2]*item.dimensions[2]/2
            planes.append((item.path, top, rotation[:, 2]))
        elif item.kind == 'cuboid' and item.approximation == 'exact':
            half = np.asarray(item.dimensions)/2
            corners = np.array([[x, y, z] for x in (-half[0], half[0])
                for y in (-half[1], half[1]) for z in (-half[2], half[2])])
            corners = corners@rotation.T+np.asarray(item.position)
            boxes.append((item.path, corners, corners.min(0), corners.max(0)))
        else:
            raise ValueError(f'Native V4 independent reference does not support {item.kind}: {item.path}')
    minima = {part.body_path: math.inf for part in meshes}
    groups = {part.group: math.inf for part in meshes}
    by_obstacle = {item.path: math.inf for item in obstacles}
    closest, minimum = None, math.inf
    seen, queries, skipped, certificate = {}, 0, 0, 0.
    for sample, poses in enumerate(transforms):
        for part in meshes:
            transform = poses[indices[part.body_path]]@part.body_from_mesh
            previous = seen.setdefault(part.path, set())
            key = transform.tobytes()
            if key in previous:
                skipped += 1
                continue
            previous.add(key)
            vertices = part.mesh.vertices@transform[:3, :3].T+transform[:3, 3]
            distances = [(path, max(0., float(((vertices-point)@normal).min())))
                for path, point, normal in planes]
            low, high = vertices.min(0), vertices.max(0)
            for path, corners, lower, upper in boxes:
                bound = float(np.linalg.norm(np.maximum(np.maximum(low-upper, lower-high), 0.)))
                # A query may improve the per-link OR per-obstacle minimum.
                if bound >= max(minima[part.body_path], by_obstacle[path]):
                    continue
                measured = convex_distance_details(vertices, corners)
                queries += 1
                certificate = max(certificate, measured['upper_bound_m']-measured['lower_bound_m'])
                distances.append((path, measured['distance_m']))
            for path, distance in distances:
                minima[part.body_path] = min(minima[part.body_path], distance)
                groups[part.group] = min(groups[part.group], distance)
                by_obstacle[path] = min(by_obstacle[path], distance)
                if distance < minimum:
                    minimum = distance
                    closest = dict(sample=sample, robot_link=part.body_path, obstacle=path, clearance_m=distance)
    finite = lambda value: float(value) if math.isfinite(value) else None
    arm = [value for path, value in minima.items() if not _is_base(path)]
    return dict(minimum_m=finite(minimum), movable_arm_minimum_m=finite(min(arm, default=math.inf)),
        by_link_m={path: finite(value) for path, value in minima.items()},
        by_group_m={name: finite(value) for name, value in groups.items()},
        by_obstacle_m={path: finite(value) for path, value in by_obstacle.items()},
        closest_pair=closest, sampled_poses=len(transforms), convex_distance_queries=queries,
        duplicate_transforms_skipped=skipped, maximum_numerical_certificate_width_m=certificate,
        reference='Copied actual PhysX cooked robot convex solids versus exact authored physical cuboids and analytic plane halfspaces',
        limitation='Sampled geometry excludes contact/rest offsets and does not certify continuous swept clearance; all arm, hand, fingers and mounted base are retained.')


def _capture_robot_meshes(stage, robot_root):
    from pxr import Usd, UsdGeom, UsdPhysics
    from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _rigid_part, _world_meters
    from robot_skill_stack.world.spatial_mapping.robot_filter import TriangleMesh
    from tests.support.reactive_geometry import CollisionMeshReference, convex_polygon_triangles, link_group
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


def run_native(args, app, result):
    import omni.kit.app
    import omni.usd
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.manipulation.motion.direct import DirectRmpflowStrategy
    from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import capture_static_world
    from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _pose_matrices
    from tests.support.isaac_async import run_kit_coroutine
    from tests.support.reactive_contact_metrics import capture_joint_feedback

    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open validated native scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
    UsdGeom.Xform.Define(stage, FIXTURE_ROOT)
    objects = fixtures(args.scenario)
    obstacle_paths = []
    for index, box in enumerate(objects):
        path = FIXTURE_ROOT+'/'+box.name
        obstacle_paths.append(path)
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(3.+index, 3., 3.))
        cube.AddScaleOp().Set(Gf.Vec3f(*box.dimensions))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(.75, .18, .08)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr().Set(True)
    robot_root = stage.GetPrimAtPath('/Franka')
    rigid_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(robot_root,
        Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    external_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not (path == '/Franka' or path.startswith('/Franka/')):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                ancestor = prim
                while ancestor and not ancestor.IsPseudoRoot() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    ancestor = ancestor.GetParent()
                external_paths.append(str(ancestor.GetPath()) if ancestor and not ancestor.IsPseudoRoot() else path)
    external_paths = list(dict.fromkeys(external_paths))
    for path in rigid_paths+external_paths:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)

    async def evaluate():
        bundle, subscription, contact = None, None, None
        active = False
        last_action = None
        raw = dict(samples=[], actions=[], tcp=[], kit=[], waits=[])
        try:
            profile = write_profile(args)
            bundle = await build_runtime(profile)
            backend = bundle.backend
            selected = backend.motion_strategy
            backend._motion_strategy = DirectRmpflowStrategy(backend)
            try:
                setup = await bundle.runtime.execute_async('move_to_pose', target=Pose(START, ORIENTATION),
                    position_tolerance=POSITION_TOLERANCE, orientation_tolerance=ORIENTATION_TOLERANCE)
            finally:
                backend._motion_strategy = selected
            result['setup'] = serial(setup)
            if not setup.ok:
                raise RuntimeError('Ordinary direct START setup failed: '+setup.message)
            if obstacle_paths:
                placement = SimulationManager.get_physics_sim_view().create_rigid_body_view(obstacle_paths)
                by_path = dict(zip(obstacle_paths, objects))
                desired = np.array([[*by_path[path].center, 0., 0., 0., 1.]
                    for path in placement.prim_paths], dtype=np.float32)
                for path, box in zip(obstacle_paths, objects):
                    stage.GetPrimAtPath(path).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*box.center))
                placement.set_transforms(desired, np.arange(len(objects), dtype=np.uint32))
                for _ in range(30):
                    await backend._next_update()
                actual = np.asarray(placement.get_transforms()).copy()
                if not np.isfinite(actual).all() or np.max(np.abs(actual-desired)) > 1e-4:
                    raise RuntimeError('Session fixture physical transforms differ from declared conditions')
                result['fixture_setup'] = dict(actual_actor_poses=serial(actual), robot_joint_positions_never_set=True)
            snapshot = capture_static_world(stage, '/Franka')
            # Capture metadata once without timestamp for matched-condition guards.
            physical_world = serial([asdict(item) for item in snapshot.obstacles])
            result['physical_world'] = physical_world
            before_cooking = tuple(str(prim.GetPath()) for prim in stage.Traverse())
            meshes, cooking = _capture_robot_meshes(stage, robot_root)
            result['geometry_reference'] = dict(cooking=cooking, meshes=len(meshes),
                cooked_only=True, hand_and_both_fingers=True,
                cooking_did_not_add_scene_prims=before_cooking == tuple(str(prim.GetPath()) for prim in stage.Traverse()))
            sim = SimulationManager.get_physics_sim_view()
            body_view = sim.create_rigid_body_view(rigid_paths+obstacle_paths)
            body_paths = tuple(body_view.prim_paths)
            if not set(rigid_paths) <= set(body_paths):
                raise RuntimeError('Physical body view omitted robot links')
            coms = np.asarray(body_view.get_coms()).copy()
            if external_paths:
                contact = sim.create_rigid_contact_view(rigid_paths,
                    filter_patterns=[list(external_paths) for _ in rigid_paths], max_contact_data_count=4096)
                if not set(rigid_paths) <= set(map(str, contact.sensor_paths)):
                    raise RuntimeError('Contact view omitted robot rigid links')
                if any(not set(external_paths) <= set(map(str, group)) for group in contact.filter_paths):
                    raise RuntimeError('Contact view omitted table/support/external collider filters')
                result['contact_reference'] = dict(sensor_paths=serial(contact.sensor_paths),
                    filter_paths=serial(contact.filter_paths), capacity=4096,
                    includes_all_external_colliders=True, positive_separation_is_not_physical_penetration=True)
            for _ in range(30):
                await backend._next_update()
            if bundle.spatial_mapping is not None:
                started = time.monotonic()
                seen = set()
                while len(seen) < 2:
                    health = bundle.spatial_mapping.health
                    if health.get('fresh') and health.get('timestamp', 0.) > started:
                        view = bundle.spatial_mapping.snapshot()
                        if args.strategy != 'reactive' or view.obstacles is not None:
                            seen.add(health['timestamp'])
                    if time.monotonic()-started > 20:
                        raise RuntimeError('GT mapping failed to provide two fresh post-reference publications')
                    await backend._next_update()
            result['configuration'] = dict(strategy=args.strategy, scenario=args.scenario,
                seed=args.seed, max_iterations=args.max_iterations, start=START.tolist(),
                goal=target_position(args.scenario).tolist(), orientation=ORIENTATION.tolist(),
                objects=serial(objects), position_tolerance_m=POSITION_TOLERANCE,
                orientation_tolerance_rad=ORIENTATION_TOLERANCE,
                max_motion_steps=backend.max_motion_steps,
                mapping_enabled=bundle.spatial_mapping is not None, mapping_source='ground_truth',
                mapping_hz=2. if bundle.spatial_mapping is not None else None,
                camera_required=False, ordinary_direct_setup=True, stationary_scene_assumption=True,
                witness=bool(args.witness))
            if args.witness:
                protocol = ROOT/'docs/assets/motion_planning_v4/feasibility_protocol.json'
                result['configuration']['witness_protocol_sha256'] = hashlib.sha256(protocol.read_bytes()).hexdigest()
            result['initial_joints'] = serial(backend.robot.get_joint_positions())
            result['initial_finger_feedback'] = serial(backend.gripper_feedback())
            if args.reference_result:
                reference = json.loads(Path(args.reference_result).read_text())
                match = reference_match(reference, result['configuration'], result['initial_joints'],
                    hashes=result['_source_before'], physical_world=physical_world)
                result['reference_match'] = match
                if not match['matched']:
                    raise RuntimeError('Matched-case guard failed before requested motion: '+str(match['checks']))
            original_apply, original_next, original_pose = backend.robot.apply_action, backend._next_update, backend.get_end_effector_pose

            def sample():
                if not active:
                    return
                item = dict(at=time.monotonic(), simulation_time=float(bundle.world.current_time),
                    physics_step=int(bundle.world.current_time_step_index), physics_dt_s=float(bundle.world.get_physics_dt()),
                    poses=np.asarray(body_view.get_transforms()).copy(),
                    body_velocities=np.asarray(body_view.get_velocities()).copy(), contacts=[],
                    joint_feedback=capture_joint_feedback(backend.robot, last_action))
                if contact is not None:
                    impulses, points, normals, separations, counts, starts = contact.get_contact_data(1.)
                    counts, starts = np.asarray(counts), np.asarray(starts)
                    for sensor, filtered in np.argwhere(counts > 0):
                        begin, count = int(starts[sensor, filtered]), int(counts[sensor, filtered])
                        separation = np.asarray(separations)[begin:begin+count].copy().reshape(-1)*snapshot.meters_per_unit
                        force = np.asarray(impulses)[begin:begin+count].copy().reshape(-1)
                        item['contacts'].append(dict(sensor=str(contact.sensor_paths[sensor]),
                            obstacle=str(contact.filter_paths[sensor][filtered]), count=count,
                            minimum_separation_m=float(separation.min()), impulse_magnitude=float(np.abs(force).sum()),
                            normal_impulses_ns=force, separations_m=separation,
                            points_world_m=np.asarray(points)[begin:begin+count].copy()*snapshot.meters_per_unit,
                            normals_world=np.asarray(normals)[begin:begin+count].copy()))
                    item['contact_buffer_full'] = int(counts.sum()) >= 4096
                raw['samples'].append(item)

            def apply(action, *positional, **keywords):
                nonlocal last_action
                last_action = action
                if active:
                    raw['actions'].append(dict(at=time.monotonic(), sim_time=float(bundle.world.current_time),
                        indices=serial(getattr(action, 'joint_indices', None)),
                        q=serial(getattr(action, 'joint_positions', None)),
                        qd=serial(getattr(action, 'joint_velocities', None))))
                return original_apply(action, *positional, **keywords)

            async def update():
                started = time.monotonic()
                try:
                    return await original_next()
                finally:
                    if active:
                        raw['waits'].append(time.monotonic()-started)
                        sample()

            def pose():
                value = original_pose()
                if active:
                    raw['tcp'].append(dict(at=time.monotonic(), sim_time=float(bundle.world.current_time),
                        position=value.position.copy(), orientation=value.orientation.copy()))
                return value

            def tick(event):
                if active:
                    raw['kit'].append(time.monotonic())

            backend.robot.apply_action, backend._next_update, backend.get_end_effector_pose = apply, update, pose
            subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(tick,
                name='global_motion_v4_measurement')
            prims_before = tuple(str(prim.GetPath()) for prim in stage.Traverse())
            active = True
            raw['kit'].append(time.monotonic())
            sample()
            initial_tcp = pose()
            started = time.monotonic()
            if args.witness:
                result['witness_segments'] = []
                for point in ((.45, -.25, .85), (.45, .25, .85), tuple(GOAL)):
                    segment_started = time.monotonic()
                    movement = await bundle.runtime.execute_async('move_to_pose', target=Pose(point, ORIENTATION),
                        position_tolerance=POSITION_TOLERANCE, orientation_tolerance=ORIENTATION_TOLERANCE)
                    result['witness_segments'].append(dict(target=point, result=serial(movement),
                        wall_seconds=time.monotonic()-segment_started))
                    if not movement.ok:
                        break
            else:
                movement = await bundle.runtime.execute_async('move_to_pose',
                    target=Pose(target_position(args.scenario), ORIENTATION),
                    position_tolerance=POSITION_TOLERANCE, orientation_tolerance=ORIENTATION_TOLERANCE)
            result['motion'] = dict(result=serial(movement), wall_seconds=time.monotonic()-started)
            sample()
            final_tcp = pose()
            active = False
            result['final_joints'] = serial(backend.robot.get_joint_positions())
            result['final_finger_feedback'] = serial(backend.gripper_feedback())
            result['final_pose_errors'] = dict(position_m=float(np.linalg.norm(final_tcp.position-target_position(args.scenario))),
                orientation_rad=backend._quat_error(final_tcp.orientation, ORIENTATION))
            result['same_stage_prims_after_motion'] = prims_before == tuple(str(prim.GetPath()) for prim in stage.Traverse())
            result['strategy_diagnostics'] = serial(getattr(selected, 'diagnostics', {}))
            plan_trace = serial(getattr(selected, 'last_plan_trace', None))
            result['initial_tcp'] = serial(initial_tcp)
            result['final_tcp'] = serial(final_tcp)
            result['no_requested_movement_joint_delta_rad'] = float(np.max(np.abs(
                np.asarray(result['initial_joints'])-np.asarray(result['final_joints']))))
            if args.lifecycle:
                result['lifecycle'] = await exercise_lifecycle(args, bundle, raw, Pose)
            backend.robot.apply_action, backend._next_update, backend.get_end_effector_pose = original_apply, original_next, original_pose
            return dict(raw=raw, meshes=meshes, body_paths=body_paths, pose_matrices=_pose_matrices,
                snapshot=snapshot, local_coms=coms, plan_trace=plan_trace)
        finally:
            active = False
            subscription = None
            if bundle is not None:
                bundle.close()
                result['runtime_closed'] = True
                result['strategy_after_close'] = serial(getattr(bundle.backend.motion_strategy, 'diagnostics', {}))
                if bundle.spatial_mapping is not None:
                    result['mapping_closed'] = bool(bundle.spatial_mapping.health['closed'])

    return run_kit_coroutine(evaluate(), app)


async def exercise_lifecycle(args, bundle, raw, Pose):
    """Unmeasured ownership/cancel/Home/Stop/Play/rebuild/shutdown episodes."""
    backend = bundle.backend
    checks, episodes = {}, []
    actions = []
    original_apply = backend.robot.apply_action

    def apply(action, *positional, **keywords):
        actions.append(dict(sim_time=float(bundle.world.current_time),
            q=serial(getattr(action, 'joint_positions', None)),
            qd=serial(getattr(action, 'joint_velocities', None)),
            indices=serial(getattr(action, 'joint_indices', None))))
        return original_apply(action, *positional, **keywords)

    backend.robot.apply_action = apply

    async def begin(target):
        before = len(actions)
        task = asyncio.create_task(backend.move_to_pose_async(target,
            position_tolerance=POSITION_TOLERANCE, orientation_tolerance=ORIENTATION_TOLERANCE))
        started = time.monotonic()
        while not task.done() and not backend._motion_active:
            if time.monotonic()-started > 30:
                task.cancel()
                raise RuntimeError('Lifecycle command did not start within bounded wait')
            await backend._next_update()
        return task, before

    rebuilt = None
    try:
        # The primary measurement is over; these are never pooled into timing.
        task, before = await begin(Pose(START, ORIENTATION))
        overlap = await backend.move_to_pose_async(Pose(GOAL, ORIENTATION))
        checks['overlap_rejected'] = not overlap.ok and 'already in progress' in overlap.message
        if task.done():
            episodes.append(dict(name='cancel_not_exercised', result=serial(await task)))
            checks['cancellation_exercised'] = False
        else:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                checks['cancellation_exercised'] = True
            checks['cancel_released_arm_owner'] = not backend._motion_active
            checks['cancel_final_arm_hold'] = (len(actions) > before and actions[-1]['qd'] is not None
                and np.all(np.asarray(actions[-1]['qd'], float) == 0.))
            episodes.append(dict(name='cancellation', actions=serial(actions[before:]),
                strategy=serial(getattr(backend.motion_strategy, 'diagnostics', {}))))
        home = await bundle.runtime.execute_async('home')
        episodes.append(dict(name='home', result=serial(home)))
        checks['home_completed'] = home.ok
        task, before = await begin(Pose(GOAL, ORIENTATION))
        checks['stop_interrupted_active_command'] = not task.done() and backend._motion_active
        await bundle.world.stop_async()
        stopped = await asyncio.wait_for(task, timeout=20.)
        checks['stop_aborted_active_command'] = not stopped.ok and not backend._motion_active
        after_stop = len(actions)
        for _ in range(3):
            await backend._next_update()
        checks['no_stale_commands_after_stop'] = len(actions) == after_stop
        episodes.append(dict(name='stop', result=serial(stopped), actions=serial(actions[before:]),
            ready=backend.is_ready))
        await bundle.world.play_async()
        if not backend.is_ready:
            before = len(actions)
            stale = await backend.move_to_pose_async(Pose(GOAL, ORIENTATION))
            checks['invalid_handles_rejected_after_play'] = not stale.ok and len(actions) == before
            episodes.append(dict(name='play_invalid_handles', result=serial(stale), ready=backend.is_ready))
        else:
            checks['invalid_handles_rejected_after_play'] = True
            episodes.append(dict(name='play_handles_available', ready=True,
                meaning='The ended trajectory stays aborted; no automatic execution restart is requested.'))
        from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
        old_strategy = backend.motion_strategy
        backend.robot.apply_action = original_apply
        rebuilt = await build_runtime(write_profile(args))
        checks['rebuild_owns_new_strategy'] = rebuilt.backend.motion_strategy is not old_strategy
        checks['old_strategy_closed'] = bool(getattr(old_strategy, 'diagnostics', {}).get('closed'))
        bundle, backend = rebuilt, rebuilt.backend
        original_apply = backend.robot.apply_action
        backend.robot.apply_action = apply
        home = await bundle.runtime.execute_async('home')
        checks['home_after_reset_rebuild'] = home.ok
        episodes.append(dict(name='reset_rebuild_home', result=serial(home), ready=backend.is_ready,
            reset_is_unmeasured_lifecycle_only=True))
        task, before = await begin(Pose(GOAL, ORIENTATION))
        checks['reset_interrupted_active_command'] = not task.done() and backend._motion_active
        await bundle.world.reset_async()
        reset_response = await asyncio.wait_for(task, timeout=20.)
        checks['reset_aborted_active_command'] = not reset_response.ok and not backend._motion_active
        after_reset = len(actions)
        for _ in range(3):
            await backend._next_update()
        checks['no_stale_commands_after_reset'] = len(actions) == after_reset
        episodes.append(dict(name='reset_during_command', result=serial(reset_response),
            actions=serial(actions[before:]), ready=backend.is_ready,
            reset_is_unmeasured_lifecycle_only=True))
        # Reset may invalidate articulation handles; rebuild through the normal
        # composition root rather than repairing SDK handles in the harness.
        backend.robot.apply_action = original_apply
        previous = backend.motion_strategy
        rebuilt = await build_runtime(write_profile(args))
        checks['second_rebuild_closes_reset_strategy'] = bool(getattr(previous, 'diagnostics', {}).get('closed'))
        bundle, backend = rebuilt, rebuilt.backend
        original_apply = backend.robot.apply_action
        backend.robot.apply_action = apply
        task, before = await begin(Pose(GOAL, ORIENTATION))
        checks['close_interrupted_active_command'] = not task.done() and backend._motion_active
        bundle.close()
        response = await asyncio.wait_for(task, timeout=20.)
        episodes.append(dict(name='close_during_command', result=serial(response), actions=serial(actions[before:])))
        checks['close_stopped_command'] = not response.ok and not backend._motion_active
        checks['strategy_closed'] = bool(getattr(backend.motion_strategy, 'diagnostics', {}).get('closed'))
    finally:
        backend.robot.apply_action = original_apply
        if rebuilt is not None:
            rebuilt.close()
    return dict(status='PASS' if checks and all(checks.values()) else 'FAIL', checks=checks,
        episodes=episodes, excluded_from_primary_timing=True,
        limitation='Explicit simulated lifecycle faults and normal runtime rebuild; no old trajectory is resumed, and lifecycle trajectories are excluded from primary motion/clearance timing.')


def finish(args, result, capture):
    from tests.support.reactive_contact_metrics import analyze_contact_samples, analyze_joint_feedback
    raw, snapshot = capture['raw'], capture['snapshot']
    transforms = [capture['pose_matrices'](row['poses'], snapshot.meters_per_unit) for row in raw['samples']]
    began = time.monotonic()
    result['clearance'] = reduce_clearances(capture['meshes'], capture['body_paths'], transforms, snapshot.obstacles)
    result['initial_clearance'] = reduce_clearances(capture['meshes'], capture['body_paths'], transforms[:1], snapshot.obstacles)
    result['offline_clearance_seconds'] = time.monotonic()-began
    result['contacts'] = contact_summary(raw['samples'])
    result['movable_arm_contacts'] = contact_summary(raw['samples'], exclude_mounted_base=True)
    result['control_interval_ms'] = distribution(np.diff([row['at'] for row in raw['actions']]), 1000)
    result['kit_interval_ms'] = distribution(np.diff(raw['kit']), 1000)
    result['controller_wait_ms'] = distribution(raw['waits'], 1000)
    result['controller_waits_over_100ms'] = int(np.count_nonzero(np.asarray(raw['waits']) > .1))
    motion = result['motion']['result']
    points = np.asarray([row['position'] for row in raw['tcp']], float)
    result['tcp_path'] = dict(samples=len(points),
        length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) else None,
        straight_distance_m=float(np.linalg.norm(target_position(args.scenario)-START)))
    success = (motion['status'] == 'success'
        and result['final_pose_errors']['position_m'] <= POSITION_TOLERANCE
        and result['final_pose_errors']['orientation_rad'] <= ORIENTATION_TOLERANCE)
    if args.witness:
        success = success and len(result.get('witness_segments', [])) == 3
    reported = result['contacts']['reported_points']
    intersection = result['clearance']['minimum_m'] == 0.
    result['strict_success'] = bool(success and reported == 0 and not intersection)
    result['movable_arm_strict_success'] = bool(success
        and result['movable_arm_contacts']['reported_points'] == 0
        and result['clearance']['movable_arm_minimum_m'] is not None
        and result['clearance']['movable_arm_minimum_m'] > 0.)
    result['mounted_base_qualification'] = dict(
        reported_points=reported-result['movable_arm_contacts']['reported_points'],
        by_link_clearance_m={path: value for path, value in result['clearance']['by_link_m'].items() if _is_base(path)},
        meaning='All-scene strict_success retains mounted-base reports and reference intersection; movable_arm_strict_success explicitly excludes only the fixed Franka/panda_link0 mount, while still checking every movable arm, hand and finger against table/ground and all external obstacles.')
    result['goal_reached'] = success
    result['classification'] = ('target_reached_clear' if result['strict_success'] else
        'target_reached_with_reports_or_reference_intersection' if success else
        'motion_failed_with_reports_or_reference_intersection' if reported or intersection else 'motion_failed_clear')
    result['contact_severity'] = analyze_contact_samples(raw['samples'], capture['body_paths'],
        capture['local_coms'], meters_per_unit=snapshot.meters_per_unit,
        goal_status=motion['status'], failure_code=motion.get('failure_code'),
        minimum_clearance_m=result['clearance']['minimum_m'], strict_v2_success=result['strict_success'])
    result['joint_feedback'] = analyze_joint_feedback(raw['samples'])
    trace = capture['plan_trace'] or {}
    indices = trace.get('joint_indices')
    if indices is not None and trace.get('route') is not None:
        actual = [np.asarray(row['joint_feedback']['positions'])[indices] for row in raw['samples']
            if row['joint_feedback'].get('positions') is not None]
        result['joint_path_deviation'] = joint_path_deviation(actual, trace['route'])
    arrays = dict(body_poses=np.asarray([row['poses'] for row in raw['samples']]),
        simulation_times=np.asarray([row['simulation_time'] for row in raw['samples']]))
    for index, mesh in enumerate(capture['meshes']):
        arrays[f'vertices_{index}'], arrays[f'triangles_{index}'] = mesh.mesh.vertices, mesh.mesh.triangles
        arrays[f'body_from_mesh_{index}'] = mesh.body_from_mesh
    output = Path(args.artifacts_dir)
    np.savez_compressed(output/'geometry_reference.npz', **arrays)
    (output/'trace.json').write_text(json.dumps(serial(raw), allow_nan=False)+'\n')
    (output/'plan_trace.json').write_text(json.dumps(serial(trace), allow_nan=False)+'\n')


def execute_case(args, app):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    if Path(args.result).exists() or (output/'trace.json').exists():
        raise ValueError('Refusing to overwrite a recorded native attempt')
    before, result, error, capture = source_hashes(), {}, None, None
    result['_source_before'] = before
    try:
        capture = run_native(args, app, result)
        finish(args, result, capture)
    except BaseException:
        error = traceback.format_exc()
    result.pop('_source_before', None)
    checks = dict(source_unchanged=before == source_hashes(),
        physical_reference_captured=capture is not None,
        runtime_closed=result.get('runtime_closed', False),
        no_added_scene_prims=result.get('same_stage_prims_after_motion', False),
        owner_thread_cooked_reference=result.get('geometry_reference', {}).get('cooked_only', False),
        cooking_no_added_prims=result.get('geometry_reference', {}).get('cooking_did_not_add_scene_prims', False),
        contact_buffer_unsaturated=not result.get('contacts', {}).get('contact_buffer_saturated', False),
        lifecycle_pass=not args.lifecycle or result.get('lifecycle', {}).get('status') == 'PASS')
    payload = dict(schema_version=1, status='PASS' if error is None and all(checks.values()) else 'FAIL',
        contracts=checks, hashes=before, metrics=result, error=error,
        meaning='PASS is a valid measurement, not successful avoidance. Strict success additionally requires target reach, zero reports and no sampled cooked-reference intersection; all contacts and mounted-base evidence remain separately visible.')
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(payload), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=payload['status'], goal_reached=result.get('goal_reached'),
        strict_success=result.get('strict_success'), error=error)), flush=True)
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=STRATEGIES, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--enable-mapping', action='store_true', help='Optional GT mapping background workload; global never consumes it')
    parser.add_argument('--reference-result')
    parser.add_argument('--witness', action='store_true', help='Predeclared direct lift/cross/descend feasibility witness; not an RRT candidate')
    parser.add_argument('--lifecycle', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy=args.strategy, rrt_seed=args.seed, rrt_max_iterations=args.max_iterations)
    if args.lifecycle and (args.scenario != 'empty' or args.strategy != 'global'):
        parser.error('--lifecycle requires an empty global case')
    if args.witness and (args.strategy != 'direct' or args.scenario not in ('narrow', 'large') or args.lifecycle):
        parser.error('--witness requires direct narrow/large without lifecycle')
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    try:
        return execute_case(args, app)
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
