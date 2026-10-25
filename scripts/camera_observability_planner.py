#!/usr/bin/env python3
"""Replay the unchanged V2b physical motion fixture with a research camera layout.

Use one fresh process per direct/reactive case through run_isaac_python.sh.
Only a copied profile and the anonymous USD session layer are changed. Camera
calibration is collected from ordinary admitted frames, without extra reads.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import traceback
from unittest.mock import patch

from scripts.camera_workload_config import (
    ORIGINAL_CAMERAS, ROOT, _set_table_values)


THIRD = '/World/CameraObservabilityThird'
SAMPLING_POLICIES = ('fixed', 'depth_edge', 'phase_cycle', 'cycle_stride')


def validate_sampling_policy(policy, max_rays, *, audit_dir=None):
    """Reject ambiguous research combinations before any simulator starts."""
    if policy not in SAMPLING_POLICIES:
        raise ValueError('Unknown research sampling policy: '+str(policy))
    if policy != 'fixed' and max_rays != 1024:
        raise ValueError('Nonfixed strategies compare at the unchanged1024-ray total budget')
    if policy != 'fixed' and audit_dir:
        raise ValueError('Nonfixed sampling cannot be combined with mapping audit processor replacement')
    return policy


def enter_sampling_research(stack, policy, max_rays, *, audit_dir=None):
    """Patch only an explicitly selected research process's processor factory."""
    validate_sampling_policy(policy, max_rays, audit_dir=audit_dir)
    if policy == 'fixed':
        return
    from robot_skill_stack.world.spatial_mapping import runtime
    from scripts.camera_observability_sampling import ResearchMappingProcessor
    stack.enter_context(patch.object(runtime, 'MappingProcessor', partial(
        ResearchMappingProcessor, policy=policy, allow_cycle_offsets=policy == 'cycle_stride')))


def author_camera_layout(layout, stage=None):
    """Author exact ROS-optical poses in the anonymous USD session layer.

    Call only after SimulationApp starts and a scene is open. This helper is
    shared by the observation capture and performance wrappers; importing the
    module itself requires neither Isaac nor USD.
    """
    import numpy as np
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    from pxr import Gf, Usd, UsdGeom
    stage.SetEditTarget(stage.GetSessionLayer())
    for value in layout['cameras']:
        camera_id = value['camera_id']
        prim = stage.GetPrimAtPath(camera_id)
        if not prim.IsValid():
            prim = UsdGeom.Camera.Define(stage, camera_id).GetPrim()
            reference = stage.GetPrimAtPath(ORIGINAL_CAMERAS[1])
            for name in ('projection', 'focalLength', 'horizontalAperture',
                         'verticalAperture', 'clippingRange'):
                prim.GetAttribute(name).Set(reference.GetAttribute(name).Get())
        transform = np.asarray(value['world_from_camera_ros'], float)
        # ROS optical right/down/forward -> USD camera right/up/backward.
        world_from_usd = transform @ np.diag([1., -1., -1., 1.])
        parent_world = np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).
            GetLocalToWorldTransform(prim.GetParent()), float)
        local_row = world_from_usd.T @ np.linalg.inv(parent_world)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*local_row.ravel().tolist()))
    return tuple(value['camera_id'] for value in layout['cameras'])


def apply_camera_pose(camera, specification):
    """Apply the same optical pose through the public Camera API after init."""
    import numpy as np
    from pxr import Gf
    transform = np.asarray(specification['world_from_camera_ros'], float)
    rotation = Gf.Matrix3d(*transform[:3, :3].T.ravel().tolist()).ExtractRotation().GetQuat()
    quaternion = np.r_[rotation.GetReal(), np.asarray(rotation.GetImaginary())]
    camera.camera.set_world_pose(position=transform[:3, 3], orientation=quaternion,
                                 camera_axes='ros')


def select_layout(document, layout_id):
    """Validate the geometric-screening contract before importing Isaac."""
    matches = [value for value in document.get('layouts', [])
               if value.get('layout_id') == layout_id]
    if len(matches) != 1:
        raise ValueError(f'Expected one layout named {layout_id!r}, found {len(matches)}')
    layout = matches[0]
    if str(layout.get('family', '')).startswith('J'):
        raise ValueError('Wrist layouts remain geometric-only until attachment and synchronization are validated')
    cameras = layout.get('cameras', [])
    paths = tuple(value.get('camera_id') for value in cameras)
    if len(cameras) not in (2, 3) or paths[:2] != ORIGINAL_CAMERAS:
        raise ValueError('Native layouts require the canonical two cameras, plus at most one fixed camera')
    if len(cameras) == 3 and paths[2] != THIRD:
        raise ValueError(f'Third research camera must be {THIRD}')
    if len(set(paths)) != len(paths):
        raise ValueError('Camera IDs must be distinct')
    for camera in cameras:
        transform = camera.get('world_from_camera_ros')
        if (not isinstance(transform, list) or len(transform) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in transform)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                       for row in transform for v in row)):
            raise ValueError('Every camera needs a finite 4x4 world_from_camera_ros')
        if any(abs(transform[3][i]-float(i == 3)) > 1e-8 for i in range(4)):
            raise ValueError('Camera transform must be homogeneous')
        rotation = [row[:3] for row in transform[:3]]
        if any(abs(sum(rotation[k][i]*rotation[k][j] for k in range(3))-float(i == j)) > 1e-5
               for i in range(3) for j in range(3)):
            raise ValueError('Camera optical rotation must be orthonormal')
        determinant = sum(rotation[0][i]*(rotation[1][(i+1)%3]*rotation[2][(i+2)%3]
            - rotation[1][(i+2)%3]*rotation[2][(i+1)%3]) for i in range(3))
        if determinant < .99999:
            raise ValueError('Camera optical rotation must be right handed')
        resolution = tuple(camera.get('resolution', (640, 480)))
        if resolution not in ((320, 240), (640, 480)):
            raise ValueError('Only previously tested RGB-D resolutions are supported')
    return layout


def copied_profile(text, layout, *, max_rays=1024):
    """Keep semantic perception on the validated pair and change mapping views."""
    if max_rays not in (1024, 2048, 4096):
        raise ValueError('Research ray budget must be1024,2048,or4096')
    text = _set_table_values(text, 'multi_view', dict(camera_prim_paths=ORIGINAL_CAMERAS))
    return _set_table_values(text, 'spatial_mapping', dict(
        camera_prim_paths=[c['camera_id'] for c in layout['cameras']],
        max_rays=max_rays))


def run(args, app, layout):
    import numpy as np
    import omni.usd
    from scripts import diagnose_reactive_motion as diagnostic
    from robot_skill_stack.integrations.isaac.world.multiview import IsaacMultiViewCamera
    from robot_skill_stack.world.perception_multiview.shared_frames import SharedFramePool
    from tests.support import multiview_native

    by_id = {camera['camera_id']: camera for camera in layout['cameras']}
    actual = {}
    original_profile = diagnostic.profile
    original_init = IsaacMultiViewCamera.__init__
    original_batch = SharedFramePool.batch
    protected_before = diagnostic.source_hashes()

    def profile(value):
        author_camera_layout(layout)
        path = original_profile(value)
        path.write_text(copied_profile(path.read_text(), layout, max_rays=args.max_rays))
        return path

    def camera_init(camera, prim_path='/World/PerceptionCamera', name='perception_camera',
                    resolution=(640, 480)):
        return original_init(camera, prim_path=prim_path, name=name,
            resolution=tuple(by_id[prim_path].get('resolution', resolution))
                if prim_path in by_id else resolution)

    def fixed_viewpoint(camera, *_positional, **_keywords):
        # The delegated fixture calls its historical hardcoded EYES after
        # bootstrap. Preserve this layout's full authored optical transform.
        camera_id = str(camera.camera.prim_path)
        if camera_id not in by_id:
            raise ValueError('Unexpected camera in fixture: '+camera_id)
        apply_camera_pose(camera, by_id[camera_id])

    def batch(pool, *positional, **keywords):
        frames = original_batch(pool, *positional, **keywords)
        for frame in frames or ():
            camera = by_id[frame.camera_id]
            row = actual.setdefault(frame.camera_id, dict(camera_id=frame.camera_id,
                first_token=diagnostic.serial(frame.token), first_acquired_at=frame.acquired_at,
                intrinsics=diagnostic.serial(frame.intrinsics),
                world_from_camera_ros=diagnostic.serial(frame.world_from_camera),
                resolution=[frame.depth.shape[1], frame.depth.shape[0]],
                advances=0, regressions=0, last_token=None))
            token = tuple(frame.token)
            previous = row['last_token']
            if previous is not None and previous != token:
                row['advances'] += 1
                row['regressions'] += int(token[0] <= previous[0] or token[1] <= previous[1])
            row['last_token'] = token
            row['last_acquired_at'] = frame.acquired_at
            row['transform_max_error'] = float(np.max(np.abs(np.asarray(
                row['world_from_camera_ros'])-np.asarray(camera['world_from_camera_ros']))))
            row['resolution_matches'] = row['resolution'] == camera.get('resolution', [640, 480])
            width, height = row['resolution']
            fx, fy = row['intrinsics'][0][0], row['intrinsics'][1][1]
            row['horizontal_fov_degrees'] = math.degrees(2*math.atan(width/(2*fx)))
            row['vertical_fov_degrees'] = math.degrees(2*math.atan(height/(2*fy)))
        return frames

    with ExitStack() as stack:
        enter_sampling_research(stack, args.sampling_policy, args.max_rays,
                                audit_dir=args.mapping_audit_dir)
        stack.enter_context(patch.object(diagnostic, 'profile', profile))
        stack.enter_context(patch.object(IsaacMultiViewCamera, '__init__', camera_init))
        stack.enter_context(patch.object(multiview_native, 'set_viewpoint', fixed_viewpoint))
        stack.enter_context(patch.object(SharedFramePool, 'batch', batch))
        valid = diagnostic.execute_case(args, app)
    payload = json.loads(Path(args.result).read_text())
    expected = set(by_id) if args.source == 'rgbd' else set(ORIGINAL_CAMERAS)
    camera_contracts = dict(all_selected_cameras_observed=set(actual) == expected,
        calibrated_transforms_match=bool(actual) and all(row['transform_max_error'] <= 1e-5 for row in actual.values()),
        resolutions_match=bool(actual) and all(row['resolution_matches'] for row in actual.values()),
        tokens_advance=bool(actual) and all(row['advances'] >= 2 and row['regressions'] == 0 for row in actual.values()),
        production_sources_unchanged=protected_before == diagnostic.source_hashes())
    payload['research'] = dict(layout_id=layout['layout_id'], layout=layout,
        actual_camera_calibration=actual, camera_contracts=camera_contracts,
        global_ray_budget=args.max_rays, sampling_policy=args.sampling_policy,
        sampling=('Unchanged production fixed sampler' if args.sampling_policy == 'fixed' else
            'Explicit research-only RGB-D candidate/selection prototype; original clocks and occupancy model retained'),
        sampling_script_sha256=hashlib.sha256((ROOT/'scripts/camera_observability_sampling.py').read_bytes()).hexdigest(),
        layout_file_sha256=hashlib.sha256(Path(args.layouts).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_truth_scope='Existing independent collision/reference and visibility scoring only; never an RGB-D runtime input.',
        camera_resource_scope='Anonymous session-layer cameras; no physical mount or camera-body collision geometry is introduced.',
        semantic_camera_paths=ORIGINAL_CAMERAS, mapping_camera_paths=tuple(by_id))
    if not all(camera_contracts.values()):
        payload['status'] = 'FAIL'
        payload['research']['error'] = 'Camera calibration/frame contract failed; physical metrics retained for diagnosis'
    Path(args.result).write_text(json.dumps(diagnostic.serial(payload), indent=2, allow_nan=False)+'\n')
    return valid and payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layouts', required=True)
    parser.add_argument('--layout-id', required=True)
    parser.add_argument('--strategy', choices=('direct', 'reactive'), required=True)
    parser.add_argument('--source', choices=('rgbd', 'ground_truth'), default='rgbd')
    parser.add_argument('--scenario', choices=('empty', 'static', 'sparse', 'partial', 'narrow',
        'separated', 'irregular', 'large', 'near', 'moving'), default='static')
    parser.add_argument('--mapping-hz', type=float, default=2.)
    parser.add_argument('--max-rays', type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument('--sampling-policy', choices=SAMPLING_POLICIES, default='fixed')
    parser.add_argument('--mapping-audit-dir')
    parser.add_argument('--audit-capture-inputs', action='store_true')
    parser.add_argument('--audit-max-captures', type=int, default=24)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)
    args.lifecycle = False
    args.contact_metrics = True
    try:
        validate_sampling_policy(args.sampling_policy, args.max_rays, audit_dir=args.mapping_audit_dir)
        if args.sampling_policy != 'fixed' and args.source != 'rgbd':
            raise ValueError('Research depth selection applies only to explicit RGB-D inputs')
    except ValueError as exc:
        parser.error(str(exc))
    layout = select_layout(json.loads(Path(args.layouts).read_text()), args.layout_id)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    app = None
    try:
        import numpy as np
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import warp as wp
        if not any(device.is_cuda for device in wp.get_devices()):
            raise RuntimeError('No CUDA GPU; native camera/planner results are unmeasured')
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
