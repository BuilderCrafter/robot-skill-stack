#!/usr/bin/env python3
"""Native research captures: legitimate robot postures and rendered visibility.

All images/meshes are evaluation artifacts. Capture I/O and oracle scoring are
outside performance measurements and never supply truth to the runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from scripts.camera_observability_planner import (
    author_camera_layout, apply_camera_pose, copied_profile, select_layout)
from scripts.camera_workload_config import SCENE, write_profile
from scripts.camera_workload_native import serial, write_json


POSTURES = dict(home=None, hover=(.45, 0., .25), extended=(.60, .10, .25),
                raised=(.38, -.10, .50), low=(.45, -.08, .16))


def run(args, app, layout):
    import numpy as np
    import omni.kit.app
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.utils.stage import open_stage, is_stage_loading
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.world.multiview import IsaacMultiViewCamera
    from robot_skill_stack.runtime.process_worker import _TransportPickler
    from robot_skill_stack.world.spatial_mapping.runtime import MappingInput
    from scripts.camera_observability_geometry import (
        fixtures, shape_depth, shape_surface, visibility, background_shapes, fixture_robot_clearance)
    from tests.support.isaac_async import run_kit_coroutine

    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    document = dict(schema_version=1, status='RUNNING', task=args.task,
        layout_id=layout['layout_id'], layout=layout, snapshots=[], records=[], motions=[], errors=[],
        scope='Native GPU-rendered RGB-D and actual exposure-matched robot meshes. Capture I/O/scoring excludes runtime timing claims.',
        physical_truth_scope='Render-only fixture surfaces are used to measure visibility without moving the robot through contacting test objects. They never augment RGB-D mapping. Independent planner experiments retain real physical colliders.',
        camera_housing='USD cameras have no physical housings/mounts; geometric mount-envelope screening is reported separately.')
    bundle = None
    original_init = IsaacMultiViewCamera.__init__
    by_id = {c['camera_id']: c for c in layout['cameras']}
    fixture_prims = []

    def constructor(camera, prim_path='/World/PerceptionCamera', name='perception_camera', resolution=(640, 480)):
        return original_init(camera, prim_path=prim_path, name=name,
            resolution=tuple(by_id.get(prim_path, {}).get('resolution', resolution)))

    async def updates(count):
        for _ in range(count):
            await omni.kit.app.get_app().next_update_async()

    async def elapsed(seconds):
        start = time.monotonic()
        while time.monotonic()-start < seconds:
            await updates(1)

    def save_input(path, payload):
        with path.open('wb') as stream:
            _TransportPickler(stream, protocol=5).dump(payload)

    async def acquire():
        start = time.monotonic()
        while time.monotonic()-start < 12.:
            frames = bundle.observation_runtime.pool.batch('observability-capture', tuple(by_id),
                max_skew=.025, simulation_time=float(bundle.world.current_time))
            if frames is not None:
                snapshots = bundle.observation_runtime.robot_filter.prepare(frames)
                return MappingInput(generation=bundle.spatial_mapping.generation,
                    frames=frames, robot_snapshots=snapshots)
            await updates(1)
        raise RuntimeError('No advancing synchronized research capture within12s')

    def calibration(frame):
        matrix = np.asarray(frame.world_from_camera)
        forward = matrix[:3, 2]
        width, height = frame.depth.shape[1], frame.depth.shape[0]
        k = np.asarray(frame.intrinsics)
        return dict(camera_id=frame.camera_id, position_world_m=matrix[:3, 3],
            world_from_camera_ros=matrix, intrinsics=k, resolution=[width, height],
            actual_downward_pitch_deg=np.degrees(np.arctan2(-forward[2], np.linalg.norm(forward[:2]))),
            actual_azimuth_deg=np.degrees(np.arctan2(forward[1], forward[0])),
            horizontal_fov_deg=np.degrees(2*np.arctan(width/(2*k[0, 0]))),
            vertical_fov_deg=np.degrees(2*np.arctan(height/(2*k[1, 1]))),
            requested_transform_max_error=float(np.max(np.abs(matrix-np.asarray(by_id[frame.camera_id]['world_from_camera_ros'])))),
            token=frame.token, acquired_at=frame.acquired_at)

    def export_pose(name, payload):
        snap = payload.robot_snapshots[payload.frames[0].camera_id]
        arrays = {}
        for index, part in enumerate(snap.parts):
            arrays[f'vertices_{index}'] = part.mesh.vertices
            arrays[f'triangles_{index}'] = part.mesh.triangles
            arrays[f'world_from_local_{index}'] = part.world_from_local
        mesh_path = output/f'robot-{name}.npz'
        np.savez(mesh_path, **arrays)
        input_path = output/f'pose-{name}.pkl'
        save_input(input_path, payload)
        row = dict(id=name, name=name, posture=name, geometry_npz=mesh_path.name,
            mapping_input_pickle=input_path.name,
            part_names=[p.path for p in bundle.observation_runtime.robot_filter._parts],
            tcp_position=bundle.backend.get_end_effector_pose().position,
            joints=bundle.backend.robot.get_joint_positions(), simulation_time=snap.simulation_time,
            cameras=[calibration(frame) for frame in payload.frames])
        document['snapshots'].append(serial(row))
        return row

    def hide_fixtures():
        for prim in fixture_prims:
            UsdGeom.Imageable(prim).MakeInvisible()
            UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(4., 4., 4.))

    def show_scenario(scenario):
        hide_fixtures()
        for shape in scenario['objects']:
            prim = next(p for p in fixture_prims if p.GetName() == scenario['scenario_id']+'_'+shape['id'])
            UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(*shape['center']))
            UsdGeom.Imageable(prim).MakeVisible()

    def score_record(payload, scenario, posture):
        arrays, rows = {}, []
        for index, frame in enumerate(payload.frames):
            arrays[f'depth_{index}'], arrays[f'rgb_{index}'] = frame.depth, frame.rgb
            arrays[f'K_{index}'], arrays[f'X_{index}'] = frame.intrinsics, frame.world_from_camera
            spec = dict(by_id[frame.camera_id], intrinsics=frame.intrinsics,
                        world_from_camera_ros=frame.world_from_camera)
            h, w = frame.depth.shape
            y, x = np.indices((h, w))
            k, matrix = frame.intrinsics, frame.world_from_camera
            directions = np.column_stack(((x.ravel()+.5-k[0, 2])/k[0, 0],
                (y.ravel()+.5-k[1, 2])/k[1, 1], np.ones(h*w))) @ matrix[:3, :3].T
            depth = frame.depth.ravel()
            valid = np.isfinite(depth) & (depth > 0)
            snapshot = payload.robot_snapshots[frame.camera_id]
            for shape in scenario['objects']:
                reference = shape_depth(shape, matrix[:3, 3], directions)
                possible = np.isfinite(reference) & (reference > 0)
                matched = possible & valid & (np.abs(depth-reference) <= .005)
                points, normals, weights = shape_surface(shape, density=7)
                others = background_shapes()+[s for s in scenario['objects'] if s['id'] != shape['id']]
                prediction, uv = visibility(spec, points, snapshot.parts, others, normals)
                point_camera = (points-matrix[:3, 3]) @ matrix[:3, :3]
                inside = (point_camera[:, 2] > 0) & np.isfinite(uv).all(1)
                px = np.clip(np.nan_to_num(uv[:, 0], nan=-1, posinf=-1, neginf=-1).astype(int), 0, w-1)
                py = np.clip(np.nan_to_num(uv[:, 1], nan=-1, posinf=-1, neginf=-1).astype(int), 0, h-1)
                inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                native = inside & (np.sum(normals*(matrix[:3, 3]-points), axis=1) > 0)
                sampled = frame.depth[py, px]
                native &= np.isfinite(sampled) & (sampled > 0) & (np.abs(sampled-point_camera[:, 2]) <= .008)
                rows.append(dict(camera_id=frame.camera_id, object_id=shape['id'],
                    raw_projected_object_pixels=int(possible.sum()), valid_depth_pixels=int((possible & valid).sum()),
                    native_surface_return_pixels=int(matched.sum()), occluded_pixels=int((possible & valid & (depth < reference-.005)).sum()),
                    invalid_depth_pixels=int((possible & ~valid).sum()),
                    predicted_visible_surface_fraction=float(weights[prediction].sum()/weights.sum()),
                    native_visible_surface_fraction=float(weights[native].sum()/weights.sum()),
                    sampled_visibility_agreement=float(weights[prediction == native].sum()/weights.sum()),
                    surface_points=points, predicted_mask=prediction, native_mask=native, surface_weights=weights))
        filename = f"view-{posture}-{scenario['scenario_id']}.npz"
        # Dense masks/surface samples stay in ignored captures, not compact tables.
        for index, row in enumerate(rows):
            for key in ('surface_points', 'predicted_mask', 'native_mask', 'surface_weights'):
                arrays[f'score_{index}_{key}'] = row.pop(key)
        np.savez(output/filename, **arrays)
        return dict(posture=posture, scenario_id=scenario['scenario_id'], image_npz=filename,
            objects=scenario['objects'], cameras=[calibration(frame) for frame in payload.frames], scores=rows)

    async def evaluate():
        nonlocal bundle
        if not open_stage(str(SCENE)):
            raise RuntimeError('Could not open validated scene')
        for _ in range(600):
            if not is_stage_loading():
                break
            await updates(1)
        else:
            raise RuntimeError('Scene load timed out')
        stage = omni.usd.get_context().get_stage()
        author_camera_layout(layout, stage)
        scenarios = fixtures()
        for scenario in scenarios:
            for shape in scenario['objects']:
                path = '/World/CameraObservabilityFixtures/'+scenario['scenario_id']+'_'+shape['id']
                if shape['kind'] == 'box':
                    schema = UsdGeom.Cube.Define(stage, path)
                    schema.CreateSizeAttr(1.)
                elif shape['kind'] == 'sphere':
                    schema = UsdGeom.Sphere.Define(stage, path)
                    schema.CreateRadiusAttr(shape['radius'])
                else:
                    schema = UsdGeom.Cylinder.Define(stage, path)
                    schema.CreateRadiusAttr(shape['radius'])
                    schema.CreateHeightAttr(shape['height'])
                    schema.CreateAxisAttr('Z')
                prim = schema.GetPrim()
                xform = UsdGeom.Xformable(prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(4., 4., 4.))
                if shape.get('yaw'):
                    xform.AddRotateZOp().Set(float(np.degrees(shape['yaw'])))
                if shape['kind'] == 'box':
                    xform.AddScaleOp().Set(Gf.Vec3f(*shape['dimensions']))
                schema.GetDisplayColorAttr().Set([Gf.Vec3f(.85, .25, .12)])
                fixture_prims.append(prim)
        hide_fixtures()
        profile = write_profile(output, 'C' if len(layout['cameras']) == 3 else 'A')
        profile.write_text(copied_profile(profile.read_text(), layout))
        bundle = await build_runtime(profile)
        for name, camera in bundle.observation_runtime.pool.cameras.items():
            apply_camera_pose(camera, by_id[name])
        await elapsed(3.)
        document['settings'] = serial(asdict(bundle.config.spatial_mapping))
        for posture in args.postures.split(','):
            hide_fixtures()
            await updates(4)
            target = POSTURES[posture]
            moved = await bundle.runtime.execute_async('home' if target is None else 'move_to_pose',
                **({} if target is None else dict(target=Pose(target, [0., 0., 1., 0.]))))
            document['motions'].append(dict(posture=posture, result=serial(moved), ok=moved.ok))
            if not moved.ok:
                document['errors'].append('Legitimate posture move failed: '+posture)
                continue
            await updates(10)
            payload = await acquire()
            export_pose(posture, payload)
            if args.task == 'visibility':
                for scenario in scenarios:
                    show_scenario(scenario)
                    await updates(10)
                    payload = await acquire()
                    document['records'].append(serial(score_record(payload, scenario, posture)))
                    write_json(output/'capture.json', document)
                    print(f"captured {layout['layout_id']} / {posture} / {scenario['scenario_id']}", flush=True)
        if args.burst_count:
            hide_fixtures()
            await updates(4)
            burst_target = POSTURES[args.burst_posture]
            response = await bundle.runtime.execute_async('home' if burst_target is None else 'move_to_pose',
                **({} if burst_target is None else dict(target=Pose(burst_target, [0., 0., 1., 0.]))))
            if not response.ok:
                raise RuntimeError('Burst posture failed: '+response.message)
            scenario = next(s for s in scenarios if s['scenario_id'] == 'thin_post')
            show_scenario(scenario)
            await elapsed(1.)
            check_payload = await acquire()
            parts = next(iter(check_payload.robot_snapshots.values())).parts
            clearances = {s['id']: fixture_robot_clearance(SimpleNamespace(parts=parts), s) for s in scenario['objects']}
            if any(value['potential_overlap'] for value in clearances.values()):
                raise RuntimeError('Sampling fixture may overlap the arm; use a clear posture instead of scoring it as a physical fixture')
            entries = []
            for index in range(args.burst_count):
                await elapsed(.5)
                payload = await acquire()
                name = f'burst-{index:03d}.pkl'
                save_input(output/name, payload)
                entries.append(dict(input_pickle=name, input_timestamp=min(f.acquired_at for f in payload.frames)))
            write_json(output/'sampling_capture.json', dict(capture_layout=layout['layout_id'], entries=entries,
                config=dict(spatial_mapping=document['settings']),
                burst_posture=args.burst_posture, robot_fixture_clearance=clearances,
                capture_valid_for_physical_fixture_scoring=True,
                evaluation_objects=[dict(name=s['id'], center=s['center'], dimensions=[2*s['radius'],2*s['radius'],s['height']],
                    kind=s['kind'], radius=s['radius'], height=s['height']) for s in scenario['objects']],
                scoring_shape_scope='Cylinder exact shape available; bounding dimensions are an explicitly conservative evaluation ROI, never sampler input.'))
        hide_fixtures()
        document['status'] = 'PASS' if not document['errors'] else 'PARTIAL'

    try:
        with patch.object(IsaacMultiViewCamera, '__init__', constructor):
            run_kit_coroutine(evaluate(), app)
    except Exception:
        document['status'] = 'FAIL'
        document['errors'].append(traceback.format_exc())
    finally:
        if bundle is not None:
            try:
                bundle.close()
            except Exception:
                document['errors'].append(traceback.format_exc())
                document['status'] = 'FAIL'
        write_json(output/'capture.json', document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layouts', required=True)
    parser.add_argument('--layout-id', default='A_current')
    parser.add_argument('--task', choices=('postures', 'visibility'), default='postures')
    parser.add_argument('--postures', default='home,hover,extended,raised,low')
    parser.add_argument('--burst-count', type=int, default=0)
    parser.add_argument('--burst-posture', choices=tuple(POSTURES), default='home')
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if set(args.postures.split(','))-POSTURES.keys() or args.burst_count < 0:
        parser.error('Unknown posture or invalid burst count')
    layout = select_layout(json.loads(Path(args.layouts).read_text()), args.layout_id)
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    app = None
    try:
        import numpy as np
        if np.__version__ != '1.26.4':
            raise RuntimeError('Expected NumPy1.26.4')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import warp as wp
        if not any(device.is_cuda for device in wp.get_devices()):
            raise RuntimeError('No CUDA GPU; native captures remain unmeasured')
        return run(args, app, layout)['status'] == 'PASS'
    except Exception:
        write_json(output/'capture.json', dict(status='FAIL', measured=False, error=traceback.format_exc()))
        return False
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
