"""Native reliability fixtures; exact observations are retained for SDK-free scoring.

Fixtures and oracle labels never enter production mapping. No planner, skill or
runtime integration is constructed. Run this native test sequentially.
"""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping import SpatialMappingConfig, create_spatial_mapping
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.spatial_mapping.robot_filter import RobotSelfFilter
from tests.support.multiview_native import set_viewpoint
from tests.support.results import fail_result, save_ppm, write_result


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
FIXTURE = '/World/SpatialMappingV4'
EYES = ((1.2, -.85, .95), (-.65, .85, .85))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def transform(center, scale=(1., 1., 1.), yaw=0., normal=None):
    angle = np.deg2rad(yaw)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    if normal is not None:
        z = np.asarray(normal, float)/np.linalg.norm(normal)
        x = np.cross([0., 0., 1.], z)
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1., 0., 0.])
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(z, x), z))
    result = np.eye(4)
    result[:3, :3] = rotation*np.asarray(scale)
    result[:3, 3] = center
    return result


def primitive(stage, episode, name, kind, center, size=None, radius=None, height=None,
              axis='Z', yaw=0., normal=None):
    path = f'{FIXTURE}/{episode}/{name}'
    if kind == 'box':
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.)
    elif kind == 'sphere':
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(radius)
    elif kind == 'cylinder':
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.CreateAxisAttr(axis)
    else:
        raise ValueError(kind)
    matrix = transform(center, size if kind == 'box' else (1., 1., 1.), yaw, normal)
    UsdGeom.Xformable(shape).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    shape.CreateDisplayColorAttr([Gf.Vec3f(.82, .25, .12)])
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    payload = dict(name=path, kind=kind, world_from_local=matrix.tolist(),
                   reference_role='physical_and_visual')
    if kind == 'box':
        payload.update(dimensions=[1., 1., 1.], authored_dimensions_m=list(size))
    elif kind == 'sphere':
        payload['radius'] = radius
    else:
        payload.update(radius=radius, height=height, axis=axis)
    return payload


def fixtures(stage):
    """Bounded separate scenes; no search or adaptive geometry tuning."""
    groups = {}
    def add(episode, *args, **kwargs):
        item = primitive(stage, episode, *args, **kwargs)
        groups.setdefault(episode, []).append(item)
        return item
    add('mixed', 'LargeUnclassified', 'box', (.30, .24, .14), (.18, .18, .28))
    add('mixed', 'Sphere', 'sphere', (.58, -.23, .045), radius=.045)
    add('mixed', 'Cylinder', 'cylinder', (.57, .12, .065), radius=.035, height=.13)
    add('mixed', 'RotatedBox', 'box', (.36, -.20, .065), (.11, .065, .13), yaw=37.)
    add('resolution', 'ThinUpright', 'box', (.48, -.16, .11), (.006, .14, .22), yaw=19.)
    add('resolution', 'Small12mm', 'box', (.57, .12, .006), (.012, .012, .012))
    add('resolution', 'Boundary', 'cylinder', (.839, .29, .065), radius=.028, height=.13)
    add('irregular', 'LPart1', 'box', (.38, -.16, .045), (.18, .05, .09))
    add('irregular', 'LPart2', 'box', (.315, -.10, .045), (.05, .17, .09))
    # Closed convex triangular prism: its empty upper corner is not its GT AABB.
    vertices = np.array([[-.06, -.05, -.06], [.06, -.05, -.06], [-.06, -.05, .06],
                         [-.06, .05, -.06], [.06, .05, -.06], [-.06, .05, .06]])
    triangles = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                          [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    mesh = UsdGeom.Mesh.Define(stage, FIXTURE+'/irregular/ConvexWedge')
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    matrix = transform((.60, .18, .06), yaw=-23.)
    UsdGeom.Xformable(mesh).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    groups['irregular'].append(dict(name=str(mesh.GetPath()), kind='convex_mesh',
        world_from_local=matrix.tolist(), vertices=vertices.tolist(), triangles=triangles.tolist(),
        reference_role='physical_and_visual', note='Exact convex visual/collision hull; mapper GT world AABB is approximate.'))
    for suffix, left, gap in (('20mm', .387, .020), ('30mm', .637, .030)):
        add('clutter', 'Left'+suffix, 'box', (left-.04, .04, .10), (.08, .20, .20))
        add('clutter', 'Right'+suffix, 'box', (left+gap+.04, .04, .10), (.08, .20, .20))
    target = np.array([.55, -.16, .055])
    for episode, covered in (('both_visible', ()), ('one_view', (0,)), ('both_hidden', (0, 1))):
        add(episode, 'OccludedTarget', 'box', target, (.10, .10, .11))
        for index in covered:
            eye = np.asarray(EYES[index])
            add(episode, 'OpaquePanel'+str(index), 'box', (eye+target)/2,
                (.18, .18, .018), normal=eye-target)
    for name in groups:
        stage.GetPrimAtPath(FIXTURE+'/'+name).SetActive(False)
    return groups


def projected(frame, ys, xs):
    depth, k, matrix = frame.depth[ys, xs], frame.intrinsics, frame.world_from_camera
    rays = np.c_[(xs+.5-k[0, 2])/k[0, 0], (ys+.5-k[1, 2])/k[1, 1], np.ones(len(xs))]
    return (rays*depth[:, None]) @ matrix[:3, :3].T+matrix[:3, 3]


def local_indices(spec, shapes):
    """Bound scoring to fixture neighborhoods, above the ground contact band."""
    boxes = []
    for shape in shapes:
        center = np.asarray(shape['world_from_local'])[:3, 3]
        # All fixtures fit this bounded neighborhood; physical free scoring below
        # z=15mm is excluded because the imported scene's ground is not empty.
        low = np.maximum(center-[.22, .22, .20], spec.bounds_min)
        high = np.minimum(center+[.22, .22, .20], spec.bounds_max)
        low[2] = max(low[2], .015)
        if np.any(low >= high):
            continue
        first = np.maximum(np.floor((low-spec.bounds_min)/spec.voxel_size).astype(int), 0)
        last = np.minimum(np.ceil((high-spec.bounds_min)/spec.voxel_size).astype(int), spec.shape)
        indices = np.column_stack(np.unravel_index(np.arange(np.prod(last-first)), last-first))+first
        boxes.append(indices)
    return np.unique(np.concatenate(boxes), axis=0) if boxes else np.empty((0, 3), int)


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'reliability'
    output.mkdir(parents=True, exist_ok=True)
    protected = (SCENE, PROFILE, ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open unchanged reference scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        simulation_app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
    groups = fixtures(stage)
    cfg = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.)
    robot = world.scene.add(Franka(prim_path=cfg.robot.prim_path, name='reliability_fixture_robot'))
    world.reset()
    world.play()
    rig = mapping = truth = None
    rows, captures = [], []
    manifest = dict(schema_version=1, status='RUNNING', captures=captures,
        capture_only=args.capture_only, settings=dict(bounds_min=[-.3, -.6, -.02],
        bounds_max=[.85, .6, 1.12], voxel_size=.01, stride=8, max_rays=1024,
        max_frame_age=1., stale_after=1., evidence_ttl=5.),
        camera_eyes=EYES, camera_target=(.15, 0., .45),
        scoring_scope='Fixture-local centers; all physical occupied centers retained. Otherwise-free ground-contact-band and possible robot-bound centers explicitly omitted. GT mapper approximations reported separately.',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        rig = create_cameras(cfg.multi_view.camera_prim_paths)
        for camera, eye in zip(rig.cameras.values(), EYES):
            set_viewpoint(camera, eye, target=(.15, 0., .45))
            camera.camera.add_instance_id_segmentation_to_frame()  # Oracle only.
        settings = SpatialMappingConfig(enabled=True, source='rgbd', automatic_robot_filter=True,
            camera_prim_paths=cfg.multi_view.camera_prim_paths, **manifest['settings'])
        callbacks_before = len(SimulationManager._callbacks)
        mapping = create_spatial_mapping(settings, stage=stage, robot_prim_path=cfg.robot.prim_path)
        truth = create_spatial_mapping(replace(settings, source='ground_truth'), stage=stage)
        original_masks, original_generic = mapping.robot_filter.masks, RobotSelfFilter.masks
        original_integrate = mapping.map.integrate
        saved_masks, saved_pixels, saved_snapshots, evidence = {}, {}, {}, {}
        def generic(self, frames, snapshots_by_camera, *, pixels_by_camera=None):
            saved_snapshots.update(snapshots_by_camera)
            return original_generic(self, frames, snapshots_by_camera, pixels_by_camera=pixels_by_camera)
        def integrate(free_indices, occupied_indices, *, timestamp, now=None):
            evidence.update(free_indices=free_indices, occupied_indices=occupied_indices,
                            timestamp=timestamp, now=now)
            return original_integrate(free_indices, occupied_indices, timestamp=timestamp, now=now)
        def masks(frames, *, pixels_by_camera=None):
            result = original_masks(frames, pixels_by_camera=pixels_by_camera)
            saved_masks.update(result)
            saved_pixels.update(pixels_by_camera)
            return result

        def step(count=8):
            for _ in range(count):
                world.step(render=True)

        step(24)
        first_camera = next(iter(rig.cameras.values()))
        camera_transform = first_camera.get_world_from_camera_transform()
        side, eye = camera_transform[:3, 0], camera_transform[:3, 3]
        near = []
        for name, link in (('Forearm', 'panda_link5'), ('Wrist', 'panda_link7'),
                ('Gripper', 'panda_hand'), ('Foreground', 'panda_link4'), ('BehindRobot', 'panda_link3')):
            prim = stage.GetPrimAtPath(cfg.robot.prim_path+'/'+link+'/geometry/'+link)
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            vertices = vertices @ matrix[:3, :3].T+matrix[:3, 3]
            direction = side if name not in ('Foreground', 'BehindRobot') else eye-vertices.mean(0)
            direction = direction/np.linalg.norm(direction)
            if name == 'BehindRobot':
                direction = -direction
            size = .05 if name == 'Foreground' else .035
            gap = .04 if name in ('Foreground', 'BehindRobot') else .01
            center = vertices[np.argmax(vertices @ direction)]+direction*(size/2*abs(direction).sum()+gap)
            near.append(primitive(stage, 'near_robot', name, 'box', center, (size,)*3))
        groups['near_robot'] = near
        stage.GetPrimAtPath(FIXTURE+'/near_robot').SetActive(False)

        def observe(episode, shapes, batch_index, final=False):
            step()
            frames = rig.capture()
            assert frames is not None
            started = time.perf_counter()
            with patch.object(RobotSelfFilter, 'masks', generic), \
                    patch.object(mapping.robot_filter, 'masks', side_effect=masks), \
                    patch.object(mapping.map, 'integrate', side_effect=integrate):
                updated = mapping.update(frames)
            completed = time.monotonic()
            elapsed = (time.perf_counter()-started)*1000
            health = dict(mapping.map.health)
            assert updated and health['fresh'], (episode, updated, health, mapping.source.diagnostics)
            assert mapping.map.timestamp == min(frame.acquired_at for frame in frames)
            snapshot = mapping.map.snapshot(now=completed)
            row = dict(index=len(rows), episode=episode, batch_index=batch_index,
                update_ms=elapsed, original_acquired_at=mapping.map.timestamp,
                completion_age_ms=(completed-mapping.map.timestamp)*1000, health=health,
                counts=snapshot.counts(), camera_visibility={}, diagnostics=mapping.source.diagnostics)
            payloads = [{key: value for key, value in shape.items() if key != 'authored_dimensions_m'}
                        for shape in shapes]
            capture = dict(index=row['index'], episode=episode, batch_index=batch_index,
                shapes=payloads, cameras=[], clear_map_before=batch_index == 0,
                exposure_shapes_by_camera={}, snapshot_now=completed,
                evidence_file=f'evidence_{row["index"]:03d}.npz', robot_bounds_by_camera={})
            for camera_id, robot_snapshot in saved_snapshots.items():
                bounds = []
                for part, model in zip(robot_snapshot.parts, mapping.robot_filter._parts):
                    low, high = part.mesh.vertices.min(0), part.mesh.vertices.max(0)
                    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
                    world_points = corners @ part.world_from_local[:3, :3].T+part.world_from_local[:3, 3]
                    bounds.append(dict(path=model.path, minimum=world_points.min(0).tolist(),
                                       maximum=world_points.max(0).tolist()))
                capture['robot_bounds_by_camera'][camera_id] = bounds
            for camera_index, frame in enumerate(frames):
                packet = rig.cameras[frame.camera_id].camera.get_current_frame()['instance_id_segmentation']
                assert tuple(rig.cameras[frame.camera_id].get_frame_token()) == tuple(frame.token)
                labels = np.asarray(packet['data']).squeeze().copy()
                paths = {str(key): value for key, value in packet['info']['idToLabels'].items()}
                ys, xs = saved_pixels[frame.camera_id]
                row['camera_visibility'][frame.camera_id] = {
                    shape['name']: dict(raw_pixels=int(np.isin(labels, [int(key) for key, path in paths.items()
                        if path == shape['name'] or path.startswith(shape['name']+'/')]).sum())) for shape in shapes}
                filename = f'frame_{row["index"]:03d}_camera_{camera_index}.npz'
                np.savez(output/filename, depth=frame.depth, intrinsics=frame.intrinsics,
                    world_from_camera=frame.world_from_camera, token=np.asarray(frame.token),
                    acquired_at=np.asarray(frame.acquired_at), rgb=frame.rgb,
                    selected_y=ys, selected_x=xs, automatic_mask=saved_masks[frame.camera_id],
                    instance_ids=labels, instance_paths_json=np.asarray(json.dumps(paths)))
                capture['cameras'].append(dict(camera_id=frame.camera_id, file=filename,
                    token=frame.token, acquired_at=frame.acquired_at))
                capture['exposure_shapes_by_camera'][frame.camera_id] = payloads
            evaluated_indices, excluded_indices = np.empty((0, 3), int), np.empty((0, 3), int)
            if not args.capture_only:
                from tests.support.spatial_mapping_reliability import (
                    ReferenceShape, score_voxel_centers, neighborhood_indices, center_membership, score_clear_region)
                references = tuple(ReferenceShape.from_payload(shape) for shape in payloads)
                evaluated_indices = neighborhood_indices(snapshot.spec, references)
                centers = snapshot.spec.indices_to_centers(evaluated_indices)
                occupied = center_membership(centers, references)
                omitted = centers[:, 2] <= .015
                for bounds in capture['robot_bounds_by_camera'].values():
                    for bound in bounds:
                        omitted |= ((centers >= bound['minimum']) & (centers <= bound['maximum'])).all(1)
                excluded_indices = evaluated_indices[omitted & ~occupied]
                row['physical_center_metrics'] = score_voxel_centers(snapshot, references, evaluated_indices,
                    excluded_indices=excluded_indices)
                if episode == 'clutter':
                    row['clear_corridors'] = {}
                    capture['clear_regions'] = {}
                    for label, left, gap in (('gap20mm', .387, .020), ('gap30mm', .637, .030)):
                        xs = np.linspace(left+.002, left+gap-.002, 3)
                        points = np.array([[x, y, z] for x in xs for y in np.arange(-.04, .131, .01)
                                           for z in np.arange(.025, .186, .01)])
                        row['clear_corridors'][label] = score_clear_region(snapshot, points, references,
                            excluded_indices=excluded_indices)
                        assert row['clear_corridors'][label]['valid'], row['clear_corridors'][label]
                        capture['clear_regions'][label] = points.tolist()
            np.savez(output/capture['evidence_file'], **evidence, snapshot_now=completed,
                     evaluated_indices=evaluated_indices, excluded_indices=excluded_indices)
            if final:
                assert truth.update()
                row['approximate_mapper_gt'] = dict(counts=truth.map.counts(), diagnostics=truth.source.diagnostics)
                filename = f'snapshot_{episode}_{batch_index}.npz'
                np.savez_compressed(output/filename, states=snapshot.states,
                    probabilities=snapshot.probabilities, observed=snapshot.observed,
                    last_observed=snapshot.last_observed, expired=snapshot.expired,
                    bounds_min=snapshot.spec.bounds_min, bounds_max=snapshot.spec.bounds_max,
                    voxel_size=snapshot.spec.voxel_size, timestamp=np.asarray(snapshot.timestamp))
                capture['snapshot_file'] = filename
                for camera_index, frame in enumerate(frames):
                    save_ppm(output/f'{episode}_camera_{camera_index}.ppm', frame.rgb)
            rows.append(row)
            captures.append(capture)
            write_json(output/'samples.json', rows)
            write_json(output/'capture_manifest.json', manifest)
            print(f'Reliability {episode} batch{batch_index}: {elapsed:.1f}ms age{row["completion_age_ms"]:.1f}ms', flush=True)

        for episode, shapes in groups.items():
            stage.GetPrimAtPath(FIXTURE+'/'+episode).SetActive(True)
            mapping.map.clear()
            step(16)
            for batch in range(4):
                observe(episode, shapes, batch, final=batch == 3)
            stage.GetPrimAtPath(FIXTURE+'/'+episode).SetActive(False)
        for row in rows:
            if row['episode'] in ('both_visible', 'one_view', 'both_hidden'):
                values = [view[FIXTURE+'/'+row['episode']+'/OccludedTarget']['raw_pixels']
                          for view in row['camera_visibility'].values()]
                assert ((min(values) >= 8 if row['episode'] == 'both_visible' else
                    (values[0] == 0 and values[1] >= 8) if row['episode'] == 'one_view' else
                    max(values) == 0)), (row['episode'], values)
            if row['episode'] == 'near_robot':
                for name in ('Forearm', 'Wrist', 'Gripper', 'Foreground'):
                    assert max(view[FIXTURE+'/near_robot/'+name]['raw_pixels']
                               for view in row['camera_visibility'].values()) >= 8
        assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        manifest.update(status='PASS', saved_scenes_and_config_unchanged=True,
            episodes=list(groups), batches=len(rows), actual_robot_commands='Fixture initialization only')
        write_json(output/'capture_manifest.json', manifest)
    finally:
        for resource in (truth, mapping, rig):
            if resource is not None:
                resource.close()
        if 'callbacks_before' in locals():
            assert len(SimulationManager._callbacks) == callbacks_before
    metrics = dict(episodes=list(groups), batches=len(rows), capture_only=args.capture_only,
        accuracy_claims=not args.capture_only, measurements=rows,
        saved_scenes_and_config_unchanged=True, cleaned_up=True,
        actual_robot_commands='Fixture initialization only', settings=manifest['settings'])
    write_result(args.result, status='PASS', metrics=metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir')
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--static-only', action='store_true')
    arguments = parser.parse_args()
    code = 0
    try:
        main(arguments)
    except Exception as error:
        fail_result(arguments.result, error)
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
