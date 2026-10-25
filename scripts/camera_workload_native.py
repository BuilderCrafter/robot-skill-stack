#!/usr/bin/env python3
"""One fresh, native camera-workload case; normally use benchmark_camera_workload.

No performance result is produced without a CUDA device and advancing, valid
rendered RGB-D buffers. All USD/profile changes are local to this process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

from scripts.camera_workload_config import (
    MAX_RAYS, ORIGINAL_CAMERAS, SCENE, THIRD, THIRD_EYE, THIRD_TARGET,
    scenario_camera_paths, write_profile)
from scripts.camera_workload_metrics import (
    counter_deltas, cpu_interval, cpu_snapshot, distribution)


class NativeUnavailable(RuntimeError):
    """Native GPU-backed execution was not demonstrated, never a CPU fallback."""


def serial(value):
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, 'items'):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if hasattr(value, 'tolist'):
        return serial(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')


def validate_buffers(camera, resolution):
    """Read images outside the measurement window; tokens are sampled in it."""
    import numpy as np
    depth, rgb = camera.get_depth(), camera.get_rgb()
    shape = (resolution[1], resolution[0])
    return dict(depth_shape=None if depth is None else list(depth.shape),
        rgb_shape=None if rgb is None else list(rgb.shape), expected_shape=list(shape),
        valid=bool(depth is not None and rgb is not None and depth.shape == shape
            and rgb.shape[:2] == shape and rgb.ndim == 3 and rgb.shape[2] >= 3
            and np.any(np.isfinite(depth) & (depth > 0))))


def numeric_token(camera):
    token = camera.get_frame_token()
    if token is None:
        return None
    number, stamp = map(float, token)
    return (number, stamp) if (math.isfinite(number) and math.isfinite(stamp)
                               and number > 0 and stamp >= 0) else None


def gpu_snapshot():
    """Optional snapshot outside timed work; never gate a run on NVIDIA-SMI."""
    try:
        completed = subprocess.run(['nvidia-smi',
            '--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw',
            '--format=csv'], capture_output=True, text=True, timeout=3.)
        return dict(available=completed.returncode == 0, return_code=completed.returncode,
            csv=completed.stdout.strip() if completed.returncode == 0 else None,
            limitation=None if completed.returncode == 0 else 'NVIDIA-SMI query failed',
            error=None if completed.returncode == 0 else completed.stderr.strip()[:1000],
            scope='Optional snapshot outside measurement, not an interval GPU profile')
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(available=False, limitation=type(exc).__name__)


def run(args, app, gpu_names):
    import numpy as np
    import omni.kit.app
    import omni.physx
    import omni.usd
    from omni.kit.async_engine import run_coroutine
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from pxr import UsdGeom
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.runtime_lifecycle import construct_world_deferred
    from robot_skill_stack.integrations.isaac.world.multiview import IsaacMultiViewCamera
    from tests.support.multiview_native import set_viewpoint

    output = Path(args.artifacts_dir)
    resolution = tuple(map(int, args.resolution.split('x')))
    state = dict(active=False, command=None, physics_start=None)
    events = {key: [] for key in ('kit', 'kit_update', 'physics', 'physics_step',
        'physics_dt', 'control', 'control_wait', 'apply', 'motions', 'runtime',
        'semantic_completions', 'mapping_completions', 'mapping_age', 'acquisitions', 'cpu', 'frames')}
    cameras, buffers_before, buffers_after, frame_stats = {}, {}, {}, {}
    bundle = None
    extra = None
    subscriptions = []
    errors, cleanup = [], {}
    last_completion = {}
    last_health_marker = None
    started = ended = None
    runtime_before = runtime_after = None
    probe_result = None
    optional_gpu_before = optional_gpu_after = None

    def health():
        return dict(semantic=serial(bundle.observation_runtime.diagnostics['semantic']),
                    mapping=serial(bundle.spatial_mapping.health))

    def sample_frames(now):
        for name, camera in cameras.items():
            token = numeric_token(camera)
            row = frame_stats.setdefault(name, dict(first=None, last=None, advances=0,
                regressions=0, missing=0, maximum_stall_seconds=0., last_change_at=now))
            if token is None:
                row['missing'] += 1
                continue
            previous = row['last']
            if previous != token:
                if previous is not None:
                    row['regressions'] += int(token[0] <= previous[0] or token[1] <= previous[1])
                    row['advances'] += 1
                row['first'] = token if row['first'] is None else row['first']
                row['last'] = token
                row['maximum_stall_seconds'] = max(row['maximum_stall_seconds'], now-row['last_change_at'])
                row['last_change_at'] = now
                events['frames'].append(dict(at=now, camera=name, token=token))

    def poll(now):
        nonlocal last_health_marker
        current = health()
        events['mapping_age'].append(dict(at=now, timestamp=current['mapping'].get('timestamp'),
            age_seconds=current['mapping'].get('age_seconds'),
            fresh=current['mapping'].get('fresh'), status=current['mapping'].get('status')))
        marker = []
        for label in ('semantic', 'mapping'):
            counters = current[label]
            last = counters.get('last') or {}
            completed = last.get('completed_at')
            marker.append((counters.get('published'), counters.get('rejected'),
                counters.get('source_failures'), counters.get('worker_failures'), completed))
            if completed is not None and completed != last_completion.get(label):
                last_completion[label] = completed
                row = dict(at=now, **last)
                if label == 'mapping':
                    row['diagnostics'] = serial(bundle.spatial_mapping.diagnostics)
                events[label+'_completions'].append(row)
        if marker != last_health_marker:
            events['runtime'].append(dict(at=now, **current))
            last_health_marker = marker
        if not events['cpu'] or now-events['cpu'][-1]['at'] >= 1.:
            events['cpu'].append(cpu_snapshot())
        sample_frames(now)

    def drive(coroutine):
        task = run_coroutine(coroutine)
        while not task.done():
            before = time.monotonic()
            app.update()
            after = time.monotonic()
            if state['active']:
                events['kit'].append(after)
                events['kit_update'].append((after-before)*1000)
                poll(after)
        return task.result()

    async def wait_for(predicate, timeout, reason):
        before = time.monotonic()
        while not predicate():
            if time.monotonic()-before > timeout:
                raise NativeUnavailable(reason) if args.mode == 'probe' else RuntimeError(reason)
            await omni.kit.app.get_app().next_update_async()

    async def delay(seconds):
        before = time.monotonic()
        while time.monotonic()-before < seconds:
            await omni.kit.app.get_app().next_update_async()

    original_init = IsaacMultiViewCamera.__init__

    def camera_init(camera, prim_path='/World/PerceptionCamera', name='perception_camera',
                    resolution=(640, 480)):
        return original_init(camera, prim_path=prim_path, name=name,
            resolution=tuple(map(int, args.resolution.split('x'))) if prim_path == THIRD else resolution)

    async def setup():
        nonlocal bundle, extra, optional_gpu_before
        if not open_stage(str(SCENE)):
            raise NativeUnavailable('Validated two-camera scene could not be opened')
        await wait_for(lambda: not is_stage_loading(), 40., 'Scene loading timed out')
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        if args.mode not in ('A', 'probe'):
            third = UsdGeom.Camera.Define(stage, THIRD)
            reference = stage.GetPrimAtPath(ORIGINAL_CAMERAS[1])
            for name in ('projection', 'focalLength', 'horizontalAperture',
                         'verticalAperture', 'clippingRange'):
                third.GetPrim().GetAttribute(name).Set(reference.GetAttribute(name).Get())
        if args.mode == 'probe':
            world = World.instance()
            if world is None:
                world = construct_world_deferred(World, stage_units_in_meters=1.)
            if world.get_physics_context() is None:
                await world.initialize_simulation_context_async()
            await world.reset_async()
            await world.play_async()
            extra = IsaacMultiViewCamera(ORIGINAL_CAMERAS[0], name='camera_workload_probe')
            extra.initialize(semantic_segmentation=False)
            cameras[ORIGINAL_CAMERAS[0]] = extra
        else:
            bundle = await build_runtime(write_profile(output, args.mode, resolution))
            if bundle.config.motion.strategy != 'direct':
                raise RuntimeError('Camera benchmark requires the existing direct strategy')
            cameras.update(bundle.observation_runtime.pool.cameras)
            if args.mode == 'B':
                extra = IsaacMultiViewCamera(THIRD, name='camera_workload_render_only')
                extra.initialize(semantic_segmentation=False)
                cameras[THIRD] = extra
            if THIRD in cameras:
                set_viewpoint(cameras[THIRD], THIRD_EYE, THIRD_TARGET)
        # Original camera poses stay exactly as authored in the existing scene.
        await wait_for(lambda: all(numeric_token(c) is not None and
            validate_buffers(c, resolution if name == THIRD else (640, 480))['valid']
            for name, c in cameras.items()), 25., 'No valid GPU-rendered RGB-D buffers')
        if args.mode == 'probe':
            first = {name: numeric_token(c) for name, c in cameras.items()}
            await delay(1.)
            last = {name: numeric_token(c) for name, c in cameras.items()}
            if not all(last[name] is not None and last[name][0] > first[name][0] and
                       last[name][1] > first[name][1] for name in cameras):
                raise NativeUnavailable('RGB-D camera render tokens did not advance')
            return
        optional_gpu_before = gpu_snapshot()
        await delay(args.warmup)
        # Warm-up must yield real publications, not just elapsed sleep time.
        await wait_for(lambda: bundle.world_updater.health.status == 'fresh'
            and bundle.spatial_mapping.health['fresh'], 20., 'Observation workers did not become fresh')
        for name, camera in cameras.items():
            buffers_before[name] = validate_buffers(camera, resolution if name == THIRD else (640, 480))

    def before_physics(dt):
        if state['active']:
            now = time.monotonic()
            state['physics_start'] = now
            events['physics'].append(now)
            events['physics_dt'].append(float(dt)*1000)

    def after_physics(_dt):
        if state['active'] and state['physics_start'] is not None:
            events['physics_step'].append((time.monotonic()-state['physics_start'])*1000)
            state['physics_start'] = None

    async def measure():
        nonlocal started, ended, runtime_before, runtime_after, optional_gpu_before, optional_gpu_after
        backend = bundle.backend
        apply, next_update = backend.robot.apply_action, backend._next_update
        joints_before = np.asarray(backend.robot.get_joint_positions(), float).copy()
        joint_maximum_displacement = 0.

        def measured_apply(action, *positional, **keywords):
            nonlocal joint_maximum_displacement
            before = time.monotonic()
            result = apply(action, *positional, **keywords)
            actual = np.asarray(backend.robot.get_joint_positions(), float)
            joint_maximum_displacement = max(joint_maximum_displacement,
                float(np.linalg.norm(actual-joints_before)))
            events['apply'].append(dict(at=before, command=state['command'],
                milliseconds=(time.monotonic()-before)*1000))
            return result

        async def measured_next_update():
            before = time.monotonic()
            await next_update()
            now = time.monotonic()
            events['control'].append(dict(at=now, command=state['command']))
            events['control_wait'].append((now-before)*1000)

        runtime_before = health()
        for label in ('semantic', 'mapping'):
            last_completion[label] = (runtime_before[label].get('last') or {}).get('completed_at')
        backend.robot.apply_action, backend._next_update = measured_apply, measured_next_update
        pool = bundle.observation_runtime.pool
        original_batch = pool.batch

        def measured_batch(*positional, **keywords):
            before = time.monotonic()
            frames = original_batch(*positional, **keywords)
            if state['active']:
                events['acquisitions'].append(dict(at=before, milliseconds=(time.monotonic()-before)*1000,
                    consumer=positional[0] if positional else keywords.get('consumer'),
                    frames=[] if frames is None else [dict(camera=f.camera_id, token=f.token,
                        acquired_at=f.acquired_at) for f in frames]))
            return frames

        pool.batch = measured_batch
        try:
            started = time.monotonic()
            state['active'] = True
            poll(started)
            sequence = [('hover', [.45, 0., .25]), ('side', [.45, .08, .25]),
                        ('return', [.45, 0., .25]), ('home', None)]
            for label, position in sequence:
                state['command'] = label
                before = time.monotonic()
                result = await bundle.runtime.execute_async('home' if position is None else 'move_to_pose',
                    **({} if position is None else dict(target=Pose(position, [0., 0., 1., 0.]))))
                events['motions'].append(dict(command=label, seconds=time.monotonic()-before,
                    ok=result.ok, result=serial(result)))
                if not result.ok:
                    errors.append(f'Direct motion {label} failed: {result.message}')
                    break
            state['command'] = None
            while time.monotonic()-started < args.seconds:
                await omni.kit.app.get_app().next_update_async()
            ended = time.monotonic()
            poll(ended)
            runtime_after = health()
            state['joint_maximum_displacement_rad'] = joint_maximum_displacement
        finally:
            state['active'] = False
            backend.robot.apply_action, backend._next_update = apply, next_update
            pool.batch = original_batch
        optional_gpu_after = gpu_snapshot()
        for name, camera in cameras.items():
            buffers_after[name] = validate_buffers(camera, resolution if name == THIRD else (640, 480))

    try:
        with patch.object(IsaacMultiViewCamera, '__init__', camera_init):
            drive(setup())
            if args.mode == 'probe':
                probe_result = dict(status='PASS', mode='probe', gpu_devices=gpu_names,
                    frame_checks={name: dict(token=numeric_token(c), buffers=validate_buffers(c, (640, 480)))
                                  for name, c in cameras.items()}, errors=errors)
            else:
                physx = omni.physx.get_physx_interface()
                subscriptions.extend((physx.subscribe_physics_on_step_events(before_physics, True, 0),
                                      physx.subscribe_physics_on_step_events(after_physics, False, 0)))
                drive(measure())
    except NativeUnavailable:
        raise
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        state['active'] = False
        # Outside onUpdate: never change physics subscriptions inside a step.
        subscriptions.clear()
        for name, resource in [('render_only_camera', extra), ('runtime', bundle)]:
            if resource is not None:
                try:
                    resource.close()
                    cleanup[name] = 'closed'
                except Exception:
                    errors.append(traceback.format_exc())

    if args.mode == 'probe':
        if errors or probe_result is None:
            raise NativeUnavailable('GPU-backed RGB-D probe failed: '+'; '.join(errors))
        return probe_result

    def intervals(values):
        return [(last-first)*1000 for first, last in zip(values, values[1:])]

    def timings(label, key):
        return distribution(row[key]*1000 for row in events[label+'_completions']
                            if row.get(key) is not None)

    counters = {}
    if runtime_before is not None and runtime_after is not None:
        for label in ('semantic', 'mapping'):
            selected = ('attempts', 'submitted', 'published', 'no_frame', 'rejected',
                        'failures', 'source_failures', 'worker_failures', 'processing_failures')
            counters[label] = counter_deltas(
                {k: v for k, v in runtime_before[label].items() if k in selected},
                {k: v for k, v in runtime_after[label].items() if k in selected})
            worker_keys = ('submitted', 'started', 'completed', 'replaced', 'discarded', 'errors', 'delivered')
            counters[label]['worker'] = counter_deltas(
                {k: runtime_before[label]['worker'].get(k) for k in worker_keys},
                {k: runtime_after[label]['worker'].get(k) for k in worker_keys})
    wall = None if started is None or ended is None else ended-started
    for row in frame_stats.values():
        if ended is not None:
            row['maximum_stall_seconds'] = max(row['maximum_stall_seconds'], ended-row['last_change_at'])
    frame_checks = bool(frame_stats) and set(buffers_before) == set(cameras) == set(buffers_after) and all(
        buffers_before[name]['valid'] and buffers_after[name]['valid'] and row['advances'] >= 2
        and row['regressions'] == 0 and row['missing'] == 0 and row['maximum_stall_seconds'] <= 1.
        for name, row in frame_stats.items())
    if not frame_checks:
        errors.append('RGB-D shape, validity, token advancement or one-second stall check failed')
    control_intervals = []
    for command in ('hover', 'side', 'return', 'home'):
        control_intervals.extend(intervals([row['at'] for row in events['control'] if row['command'] == command]))
    source_ages = [row['age_seconds'] for row in events['mapping_age']]
    acquisition_checks = {}
    for label, expected in zip(('semantic', 'mapping'), scenario_camera_paths(args.mode)):
        batches = [row['frames'] for row in events['acquisitions']
                   if row['consumer'] == label and row['frames']]
        acquisition_checks[label] = bool(batches) and all(
            set(f['camera'] for f in batch) == set(expected) for batch in batches)
    mapping_diagnostics = [row['diagnostics'] for row in events['mapping_completions']
                           if row.get('diagnostics', {}).get('allocated_rays') is not None]
    ray_checks = bool(mapping_diagnostics) and all(0 < row['allocated_rays'] <= MAX_RAYS
        and set(row['cameras']) == set(scenario_camera_paths(args.mode)[1])
        and sum(camera['allocated_rays'] for camera in row['cameras'].values()) == row['allocated_rays']
        and all(camera['allocated_rays'] > 0 and camera['retained_rays'] > 0
                for camera in row['cameras'].values()) for row in mapping_diagnostics)
    if not all(acquisition_checks.values()) or not ray_checks:
        errors.append('Observed consumer camera selection or global mapping ray-budget check failed')
    def publication_rate(label):
        value = counters.get(label, {}).get('published')
        return None if wall is None or value is None else value/wall
    summary = dict(wall_seconds=wall, kit_update_ms=distribution(events['kit_update']),
        kit_interval_ms=distribution(intervals(events['kit'])),
        physics_interval_ms=distribution(intervals(events['physics'])),
        physics_step_ms=distribution(events['physics_step']), physics_dt_ms=distribution(events['physics_dt']),
        control_interval_ms=distribution(control_intervals), control_wait_ms=distribution(events['control_wait']),
        control_apply_ms=distribution(row['milliseconds'] for row in events['apply']),
        semantic_hz=publication_rate('semantic'), mapping_hz=publication_rate('mapping'),
        mapping_source_age_ms=distribution(v*1000 for v in source_ages if v is not None),
        mapping_compute_ms=timings('mapping', 'compute_seconds'),
        mapping_queue_ms=timings('mapping', 'queue_seconds'),
        mapping_worker_roundtrip_ms=timings('mapping', 'worker_seconds'),
        mapping_transport_envelope_ms=timings('mapping', 'transport_seconds'),
        mapping_worker_to_publication_ms=timings('mapping', 'worker_to_publication_seconds'),
        mapping_publish_age_ms=timings('mapping', 'publish_age_seconds'),
        semantic_compute_ms=timings('semantic', 'compute_seconds'),
        semantic_publish_age_ms=timings('semantic', 'publish_age_seconds'),
        cpu_utilization_percent=distribution(cpu_interval(a, b) for a, b in zip(events['cpu'], events['cpu'][1:])),
        available_ram_bytes=distribution(row.get('available_ram_bytes') for row in events['cpu']),
        job_counts=counters, motion_complete=len(events['motions']) == 4 and all(r['ok'] for r in events['motions']),
        frame_checks=frame_checks, acquisition_checks=acquisition_checks, global_ray_budget_check=ray_checks,
        allocated_mapping_rays=distribution(row['allocated_rays'] for row in mapping_diagnostics),
        mapping_rays_per_camera={name: {key: distribution(row['cameras'][name][key]
            for row in mapping_diagnostics) for key in ('allocated_rays', 'retained_rays')}
            for name in scenario_camera_paths(args.mode)[1]} if mapping_diagnostics else {},
        mapping_fresh_sample_fraction=None if not events['mapping_age'] else
            sum(row['fresh'] is True for row in events['mapping_age'])/len(events['mapping_age']),
        joint_maximum_displacement_rad=state.get('joint_maximum_displacement_rad'))
    for key in ('kit_update_ms', 'physics_step_ms', 'physics_interval_ms', 'control_interval_ms',
                'mapping_compute_ms', 'mapping_queue_ms', 'mapping_publish_age_ms'):
        if summary[key]['count'] == 0:
            errors.append('Required native timing was not measured: '+key)
    for key in ('semantic_hz', 'mapping_hz'):
        if summary[key] is None or summary[key] <= 0:
            errors.append('No measured publications for '+key)
    if summary['joint_maximum_displacement_rad'] is None or summary['joint_maximum_displacement_rad'] <= .05:
        errors.append('Actual robot joint movement was not demonstrated')
    if not summary['motion_complete']:
        errors.append('The unchanged four-command direct movement sequence did not complete')
    for label, counts in counters.items():
        if any((counts.get(k) or 0) > 0 for k in ('source_failures', 'worker_failures', 'failures', 'processing_failures')):
            errors.append(label+' source/worker errors occurred in the measurement window')
        if (counts.get('worker', {}).get('errors') or 0) > 0:
            errors.append(label+' worker reported computation errors')
    write_json(output/'trace.json', dict(events=events, runtime_before=runtime_before, runtime_after=runtime_after,
        frame_stats=frame_stats, buffers_before=buffers_before, buffers_after=buffers_after))
    return dict(status='FAIL' if errors else 'PASS', mode=args.mode, resolution=resolution,
        summary=summary, gpu_devices=gpu_names, errors=errors, cleanup=cleanup,
        optional_gpu_snapshots=dict(before=optional_gpu_before, after=optional_gpu_after),
        protocol=dict(scene=str(SCENE), semantic_cameras=scenario_camera_paths(args.mode)[0],
            mapping_cameras=scenario_camera_paths(args.mode)[1], third_eye=THIRD_EYE, third_target=THIRD_TARGET,
            original_resolution=(640, 480), semantic_configured_hz=5., mapping_configured_hz=2.,
            global_mapping_ray_budget=MAX_RAYS, warmup_seconds=args.warmup, measurement_seconds=args.seconds,
            motion='Existing direct hover, 80mm side, return, home; one identical sequence per window.',
            physics='Public PhysX pre/post step wall time; not GPU kernel time. Physics dt and wall cadence are separate.',
            source_age='Original monotonic first owner acquisition timestamp; not measured GPU exposure time.',
            transport='Round-trip minus reported worker compute is an envelope, not isolated IPC time.',
            token_probe='All cameras sampled each Kit update; image validity read before/after measured window.',
            gpu_telemetry='Not required; CUDA device and real RGB-D rendering required.',
            timing_scope='Fresh process per case. Warm-up/startup/image validation/log serialization excluded. Instrumentation is identical.',
            shared_budget='1024 total mapping rays fairly redistributed over selected cameras; no per-camera budget increase.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'A', 'B', 'C', 'D'), required=True)
    parser.add_argument('--resolution', choices=('320x240', '640x480'), default='320x240')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--warmup', type=float, default=10.)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup < 0:
        parser.error('seconds must be positive and warmup nonnegative')
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    result, app = None, None
    try:
        import numpy as np
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import warp as wp
        gpu_names = [device.name for device in wp.get_devices() if device.is_cuda]
        if not gpu_names:
            raise NativeUnavailable('Isaac started without a CUDA GPU. CPU-only Kit startup is not a camera performance measurement.')
        result = run(args, app, gpu_names)
    except NativeUnavailable as exc:
        result = dict(status='BLOCKED', mode=args.mode, errors=[str(exc)], measured=False)
    except Exception:
        result = dict(status='FAIL', mode=args.mode, errors=[traceback.format_exc()], measured=False)
    finally:
        # SimulationApp.close may terminate Python; persist before calling it.
        if result is not None:
            write_json(args.result, result)
            print(json.dumps(dict(status=result['status'], mode=args.mode, result=args.result)), flush=True)
        if app is not None:
            app.close()
    return 0 if result['status'] == 'PASS' else 77 if result['status'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
