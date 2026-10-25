#!/usr/bin/env python3
"""Run the established native smoke measurement with a screened camera layout.

All observation, physics, controller and resource measurement hooks come from
camera_workload_native.run. This wrapper changes only a copied profile and
anonymous session-layer camera poses; it does not change production defaults.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback
from unittest.mock import patch

from scripts.camera_observability_planner import (
    THIRD, SAMPLING_POLICIES, apply_camera_pose, author_camera_layout,
    enter_sampling_research, select_layout, validate_sampling_policy)
from scripts.camera_workload_config import ORIGINAL_CAMERAS, ROOT, _set_table_values, write_profile


def camera_selections(layout, mode):
    paths = tuple(camera['camera_id'] for camera in layout['cameras'])
    if mode == 'baseline' and len(paths) == 2:
        return paths, paths
    if mode in ('mapping', 'full') and len(paths) == 3:
        return (paths if mode == 'full' else paths[:2]), paths
    raise ValueError('baseline requires two cameras; mapping/full require three fixed cameras')


def write_research_profile(output, layout, mode, *, max_rays=1024):
    semantic, mapping = camera_selections(layout, mode)
    if max_rays not in (1024, 2048, 4096):
        raise ValueError('Research ray budget must be1024,2048,or4096')
    resolution = tuple(layout['cameras'][-1].get('resolution', [640, 480]))
    path = write_profile(output, 'A', resolution)
    text = _set_table_values(path.read_text(), 'multi_view', dict(camera_prim_paths=semantic))
    text = _set_table_values(text, 'spatial_mapping', dict(camera_prim_paths=mapping, max_rays=max_rays))
    path.write_text('# Research layout '+layout['layout_id']+' / '+mode+'\n'+text)
    return path


def run(args, app, layout, gpu_names):
    import numpy as np
    from scripts import camera_workload_native as native
    from scripts.benchmark_camera_workload import source_manifest
    from robot_skill_stack.integrations.isaac import bootstrap
    from robot_skill_stack.integrations.isaac.world.multiview import IsaacMultiViewCamera
    from robot_skill_stack.world.perception_multiview.shared_frames import SharedFramePool
    from tests.support import multiview_native

    semantic, mapping = camera_selections(layout, args.mode)
    internal_mode = 'A' if len(mapping) == 2 else 'D' if args.mode == 'full' else 'C'
    by_id = {camera['camera_id']: camera for camera in layout['cameras']}
    resolution = tuple(layout['cameras'][-1].get('resolution', [640, 480]))
    native_args = SimpleNamespace(mode=internal_mode, resolution='x'.join(map(str, resolution)),
        seconds=args.seconds, warmup=args.warmup, result=args.result,
        artifacts_dir=args.artifacts_dir, headless=args.headless)
    before = source_manifest()
    actual = {}
    original_init = IsaacMultiViewCamera.__init__
    original_batch = SharedFramePool.batch
    original_build = bootstrap.build_runtime

    def profile(output, _internal_mode, _third_resolution):
        author_camera_layout(layout)
        return write_research_profile(output, layout, args.mode, max_rays=args.max_rays)

    def camera_init(camera, prim_path='/World/PerceptionCamera', name='perception_camera',
                    resolution=(640, 480)):
        return original_init(camera, prim_path=prim_path, name=name,
            resolution=tuple(by_id[prim_path].get('resolution', resolution))
                if prim_path in by_id else resolution)

    async def build_runtime(path):
        bundle = await original_build(path)
        for camera_id, camera in bundle.observation_runtime.pool.cameras.items():
            apply_camera_pose(camera, by_id[camera_id])
        return bundle

    def viewpoint(camera, *_positional, **_keywords):
        camera_id = str(camera.camera.prim_path)
        apply_camera_pose(camera, by_id[camera_id])

    def batch(pool, *positional, **keywords):
        frames = original_batch(pool, *positional, **keywords)
        for frame in frames or ():
            if frame.camera_id in actual:
                continue
            specification = by_id[frame.camera_id]
            height, width = frame.depth.shape
            expected = specification['world_from_camera_ros']
            intrinsics = np.asarray(frame.intrinsics)
            transform = np.asarray(frame.world_from_camera)
            actual[frame.camera_id] = dict(camera_id=frame.camera_id,
                original_acquired_at=frame.acquired_at, token=native.serial(frame.token),
                intrinsics=native.serial(intrinsics), world_from_camera_ros=native.serial(transform),
                position_world_m=transform[:3, 3].tolist(), resolution=[width, height],
                transform_max_error=float(np.max(np.abs(transform-np.asarray(expected)))),
                resolution_matches=[width, height] == specification.get('resolution', [640, 480]),
                downward_pitch_degrees=math.degrees(math.atan2(-transform[2, 2],
                    math.hypot(transform[0, 2], transform[1, 2]))),
                azimuth_degrees=math.degrees(math.atan2(transform[1, 2], transform[0, 2])),
                horizontal_fov_degrees=math.degrees(2*math.atan(width/(2*intrinsics[0, 0]))),
                vertical_fov_degrees=math.degrees(2*math.atan(height/(2*intrinsics[1, 1]))))
        return frames

    third = layout['cameras'][-1]
    with ExitStack() as stack:
        enter_sampling_research(stack, args.sampling_policy, args.max_rays)
        stack.enter_context(patch.object(native, 'write_profile', profile))
        stack.enter_context(patch.object(native, 'THIRD', THIRD))
        stack.enter_context(patch.object(native, 'THIRD_EYE', tuple(third['position_world_m'])))
        stack.enter_context(patch.object(native, 'THIRD_TARGET', tuple(third.get('aim_world_m', [.45, 0., .35]))))
        stack.enter_context(patch.object(native, 'MAX_RAYS', args.max_rays))
        stack.enter_context(patch.object(native, 'scenario_camera_paths', lambda _mode: (semantic, mapping)))
        stack.enter_context(patch.object(bootstrap, 'build_runtime', build_runtime))
        stack.enter_context(patch.object(IsaacMultiViewCamera, '__init__', camera_init))
        stack.enter_context(patch.object(SharedFramePool, 'batch', batch))
        stack.enter_context(patch.object(multiview_native, 'set_viewpoint', viewpoint))
        result = native.run(native_args, app, gpu_names)
    contracts = dict(all_selected_cameras_calibrated=set(actual) == set(mapping),
        actual_optical_transforms_match=bool(actual) and all(row['transform_max_error'] <= 1e-5 for row in actual.values()),
        actual_resolutions_match=bool(actual) and all(row['resolution_matches'] for row in actual.values()),
        production_sources_unchanged=before == source_manifest())
    result['research'] = dict(layout_id=layout['layout_id'], mode=args.mode, layout=layout,
        actual_camera_calibration=actual, contracts=contracts,
        semantic_camera_paths=semantic, mapping_camera_paths=mapping,
        global_ray_budget=args.max_rays, sampling_policy=args.sampling_policy,
        sampler=('Unchanged production fixed selection' if args.sampling_policy == 'fixed' else
            'Explicit research-only RGB-D candidate/selection prototype; original clocks and occupancy model retained'),
        sampling_script_sha256=hashlib.sha256((ROOT/'scripts/camera_observability_sampling.py').read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout_file_sha256=hashlib.sha256(Path(args.layouts).read_bytes()).hexdigest(),
        source_hashes_before=before, source_hashes_after=source_manifest(),
        measurement='Unchanged camera_workload_native.run normal Kit/physics/direct-motion measurement; camera selection and poses are isolated research inputs.')
    result['protocol']['shared_budget'] = f'{args.max_rays} total mapping rays, fairly redistributed across selected cameras.'
    if len(mapping) == 2:
        result['protocol']['third_eye'] = None
        result['protocol']['third_target'] = None
    if not all(contracts.values()):
        result['status'] = 'FAIL'
        result.setdefault('errors', []).append('Research layout calibration/source contract failed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layouts', required=True)
    parser.add_argument('--layout-id', required=True)
    parser.add_argument('--mode', choices=('baseline', 'mapping', 'full'), required=True)
    parser.add_argument('--max-rays', type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument('--sampling-policy', choices=SAMPLING_POLICIES, default='fixed')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--warmup', type=float, default=10.)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.warmup < 0:
        parser.error('seconds must be positive and warmup nonnegative')
    try:
        validate_sampling_policy(args.sampling_policy, args.max_rays)
    except ValueError as exc:
        parser.error(str(exc))
    layout = select_layout(json.loads(Path(args.layouts).read_text()), args.layout_id)
    camera_selections(layout, args.mode)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    app, result = None, None
    try:
        import numpy as np
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import warp as wp
        names = [device.name for device in wp.get_devices() if device.is_cuda]
        if not names:
            result = dict(status='BLOCKED', measured=False,
                errors=['No CUDA GPU; native performance is unmeasured'])
        else:
            result = run(args, app, layout, names)
    except Exception:
        result = dict(status='FAIL', measured=False, errors=[traceback.format_exc()])
    finally:
        if result is not None:
            Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(status=result['status'], layout_id=args.layout_id,
                mode=args.mode, result=args.result)), flush=True)
        if app is not None:
            app.close()
    return 0 if result['status'] == 'PASS' else 77 if result['status'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
