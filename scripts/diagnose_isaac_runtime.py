"""Measure ordinary async Franka control with synchronous or worker observations.

One mode per fresh Kit process. This drives Kit's normal update loop, never a
research world.step loop. Only --pipeline sync injects a diagnostic mapping
callback; --pipeline runtime uses automatic bootstrap services. No trajectory
consumes occupancy.
"""
import argparse
from dataclasses import asdict, is_dataclass
from contextlib import ExitStack
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
EYES = ((1.2, -.85, .95), (-.65, .85, .85))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return serial(asdict(value))
    if hasattr(value, 'items'):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def distribution(values, scale=1.):
    data = np.asarray(values, float)*scale
    data = data[np.isfinite(data)]
    if not len(data):
        return dict(count=0)
    return dict(count=len(data), mean=float(data.mean()),
                **{key: float(np.percentile(data, percentile))
                   for key, percentile in [('p50', 50), ('p95', 95), ('p99', 99)]},
                worst=float(data.max()))


def hashes():
    files = [SCENE, PROFILE, Path(__file__),
        ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd']
    files += [ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/bootstrap.py',
        'robot_skill_stack/world/model/updater.py',
        'robot_skill_stack/world/perception_multiview/provider.py',
        'robot_skill_stack/integrations/isaac/manipulation/franka_backend.py',
        'robot_skill_stack/world/spatial_mapping/rgbd.py',
        'robot_skill_stack/world/spatial_mapping/occupancy.py',
        'robot_skill_stack/integrations/isaac/spatial_mapping/robot_filter.py')]
    files += [ROOT/path for path in (
        'robot_skill_stack/integrations/isaac/observation_runtime.py',
        'robot_skill_stack/runtime/worker.py',
        'robot_skill_stack/runtime/process_worker.py',
        'robot_skill_stack/runtime/worker_process.py',
        'robot_skill_stack/world/spatial_mapping/runtime.py',
        'robot_skill_stack/world/spatial_mapping/geometry.py',
        'robot_skill_stack/world/perception_multiview/shared_frames.py') if (ROOT/path).exists()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def runtime_profile(output):
    """Session-only diagnostic configuration; never rewrite the shipped TOML."""
    path = Path(output)/'runtime_profile.toml'
    path.write_text(PROFILE.read_text()+'''\n[spatial_mapping]
enabled = true
source = "rgbd"
bounds_min = [-0.3, -0.6, -0.02]
bounds_max = [0.85, 0.6, 1.12]
voxel_size = 0.01
camera_prim_paths = ["/World/PerceptionCamera", "/World/PerceptionCameraSide"]
stride = 8
max_rays = 1024
automatic_robot_filter = true
update_hz = 2.0
max_frame_age = 1.0
stale_after = 1.0
evidence_ttl = 5.0
''')
    return path


def run(args, app):
    import omni.kit.app
    import omni.usd
    from omni.kit.async_engine import run_coroutine
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.stage import open_stage, is_stage_loading
    from pxr import PhysxSchema
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from robot_skill_stack.integrations.isaac.runtime_lifecycle import (
        PhysicsCallbackScope, construct_world_deferred)
    from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
    from robot_skill_stack.integrations.isaac.spatial_mapping.factory import create_spatial_mapping
    from robot_skill_stack.integrations.isaac.spatial_mapping.frames import SharedFrames
    from robot_skill_stack.world.model.provider import ObservationBatch
    from robot_skill_stack.world.model.updater import WorldModelUpdater
    from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
    from tests.support.multiview_native import set_viewpoint

    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial_hashes = hashes()
    events = {key: [] for key in ('physics', 'kit', 'app_update', 'updater_callback',
        'callbacks', 'semantic', 'mapping', 'runtime', 'acquisition', 'association',
        'forward', 'apply', 'control_wait', 'tcp', 'motions', 'windows')}
    state = dict(active=False, phase='startup', command=None, goal=None,
                 disable_semantics=False, bundle=None, actor_direction=1., cameras={},
                 acquisition_by_timestamp={}, semantic_published=0, mapping_published=0)
    bundle = mapping = tap = scope = subscription = actor = None
    errors = []
    cleanup = {}

    def mark():
        return dict(at=time.monotonic(), phase=state['phase'], command=state['command'])

    def drive(coroutine):
        task = run_coroutine(coroutine)
        while not task.done():
            started = time.monotonic()
            app.update()
            if state['active']:
                events['app_update'].append(dict(**mark(), seconds=time.monotonic()-started))
        return task.result()

    original_observe = MultiViewProvider.observe
    original_callback = WorldModelUpdater._on_physics
    original_add_callback = PhysicsCallbackScope.add_physics_callback

    def add_callback(callback_scope, name, callback_fn):
        def measured(dt):
            started = time.monotonic()
            try:
                return callback_fn(dt)
            finally:
                if state['active']:
                    events['callbacks'].append(dict(**mark(), name=name,
                        seconds=time.monotonic()-started))
        return original_add_callback(callback_scope, name, measured)

    def retain_frames(frames):
        if frames:
            rows = [dict(camera_id=f.camera_id, token=serial(f.token), acquired_at=f.acquired_at)
                    for f in frames]
            stamp = min(f.acquired_at for f in frames)
            state['acquisition_by_timestamp'][stamp] = rows
            while len(state['acquisition_by_timestamp']) > 128:
                state['acquisition_by_timestamp'].pop(next(iter(state['acquisition_by_timestamp'])))
            return rows
        return []

    def poll_runtime():
        coordinator = bundle.observation_runtime
        diagnostics = serial(coordinator.diagnostics)
        semantic = diagnostics.get('semantic', {})
        map_health = serial(bundle.spatial_mapping.health)
        map_diagnostics = serial(bundle.spatial_mapping.diagnostics)
        for label, counters in [('semantic', semantic), ('mapping', map_health)]:
            published = int(counters.get('published', 0))
            previous = state[label+'_published']
            if published <= previous:
                continue
            state[label+'_published'] = published
            last = (counters.get('last') or {}) if label == 'semantic' else {
                key: counters[key] for key in ('timestamp', 'input_timestamp', 'completed_at',
                    'worker_seconds', 'queue_seconds', 'publish_age_seconds', 'status') if key in counters}
            timestamp = last.get('timestamp')
            frames = state['acquisition_by_timestamp'].get(timestamp, [])
            input_timestamp = last.get('input_timestamp',
                min((row['acquired_at'] for row in frames), default=None))
            row = dict(**mark(), seconds=last.get('worker_seconds', 0.),
                timestamp=timestamp, published=published-previous, error=None,
                input_timestamp=input_timestamp, worker=last,
                timestamp_matches_acquisition=timestamp is not None and timestamp == input_timestamp)
            if label == 'semantic':
                row.update(source_frames=frames, observations=[dict(id=obj.object_id,
                    class_name=obj.class_name, timestamp=obj.last_seen)
                    for obj in bundle.world_model.objects() if obj.last_seen == timestamp],
                    diagnostics=serial(bundle.world_updater.provider.diagnostics))
            else:
                accepted = bool(bundle.spatial_mapping.last_result.accepted)
                row.update(accepted=accepted, health=map_health,
                    frames=frames, acquisition_seconds=last.get('acquisition_seconds'),
                    update_seconds=last.get('worker_seconds'),
                    diagnostics=map_diagnostics,
                    filter=map_diagnostics.get('robot_filter', {}))
            events[label].append(row)
        marker = (semantic.get('published'), semantic.get('rejected'), semantic.get('worker_failures'),
                  map_health.get('published'), map_health.get('rejected'), map_health.get('failures'))
        if state.get('runtime_marker') != marker:
            state['runtime_marker'] = marker
            events['runtime'].append(dict(**mark(), coordinator=diagnostics,
                                          mapping=map_health))

    def observe(provider, *positional, **keywords):
        if state['disable_semantics']:
            return ObservationBatch()
        started = time.monotonic()
        error = None
        batch = None
        try:
            batch = original_observe(provider, *positional, **keywords)
            return batch
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if state['active']:
                frames = () if tap is None or tap.latest is None else tap.latest
                events['semantic'].append(dict(**mark(), seconds=time.monotonic()-started,
                    timestamp=None if batch is None else batch.timestamp, error=error,
                    source_frames=[dict(camera_id=f.camera_id, token=serial(f.token),
                                        acquired_at=f.acquired_at) for f in frames],
                    timestamp_matches_acquisition=None if batch is None or batch.timestamp is None
                        else bool(frames and batch.timestamp == min(f.acquired_at for f in frames)),
                    observations=[] if batch is None else [dict(id=obj.object_id,
                        class_name=obj.class_name, timestamp=obj.timestamp)
                        for obj in batch.observations], diagnostics=serial(provider.diagnostics)))

    def updater_callback(updater, dt):
        started = time.monotonic()
        count = updater.update_count
        try:
            return original_callback(updater, dt)
        finally:
            if state['active']:
                events['updater_callback'].append(dict(**mark(), seconds=time.monotonic()-started,
                    dt=float(dt), published=updater.update_count-count,
                    source_status=updater.health.status, error=updater.last_error))

    async def setup():
        nonlocal bundle, actor, mapping, tap, scope, subscription
        world = World.instance()
        if world is None:
            world = construct_world_deferred(World, stage_units_in_meters=1.)
        if world.get_physics_context() is None:
            await world.initialize_simulation_context_async()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        actor = world.scene.add(DynamicCuboid(prim_path='/World/V5RuntimeDiagnostic/MovingCube',
            name='v5_runtime_moving_cube', position=np.array([.46, -.28, .021]),
            size=.04, mass=.1, color=np.array([.15, .65, .25])))
        PhysxSchema.PhysxRigidBodyAPI.Apply(actor.prim).CreateDisableGravityAttr(True)
        bundle = await build_runtime(runtime_profile(output) if args.pipeline == 'runtime' else PROFILE)
        state['bundle'] = bundle
        if args.pipeline == 'sync' and getattr(bundle, 'observation_runtime', None) is not None:
            coordinator = bundle.observation_runtime
            worker = coordinator._semantic_worker
            health = {} if worker is None else worker.health
            if health.get('started', 0) or health.get('running') or health.get('pending'):
                raise RuntimeError('Cannot reconstruct synchronous baseline after semantic work started')
            # No await between bootstrap return and closure: a new coordinator
            # has not acquired/submitted anything. Its borrowed cameras remain
            # provider-owned, available to the original synchronous callback.
            coordinator.close()
            bundle.observation_runtime = bundle.spatial_mapping = None
            bundle.world_updater.update()
            bundle.world_updater.start(PhysicsCallbackScope(world))
        state['disable_semantics'] = args.pipeline == 'sync' and args.mode in ('control', 'mapping')
        if args.pipeline == 'runtime':
            bundle.observation_runtime.set_enabled(semantic=args.mode in ('semantic', 'combined'),
                mapping=args.mode in ('mapping', 'combined'))
        for camera, eye in zip(bundle.world_updater.provider.rig.cameras.values(), EYES):
            set_viewpoint(camera, eye, target=(.15, 0., .45))
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
        state['cameras'] = {key: dict(intrinsics=serial(camera.get_intrinsics()),
            world_from_camera=serial(camera.get_world_from_camera_transform()))
            for key, camera in bundle.world_updater.provider.rig.cameras.items()}
        rig = bundle.world_updater.provider.rig
        if args.pipeline == 'sync':
            tap = SharedFrames(rig)
        else:
            pool = bundle.observation_runtime.pool
            original_batch = pool.batch

            def capture(*positional, **keywords):
                started = time.monotonic()
                frames = original_batch(*positional, **keywords)
                retained = retain_frames(frames)
                if state['active']:
                    events['acquisition'].append(dict(**mark(), seconds=time.monotonic()-started,
                        consumer=positional[0] if positional else keywords.get('consumer'), frames=retained))
                return frames
            pool.batch = capture
            if hasattr(bundle.world_updater.provider, 'associate'):
                associate = bundle.world_updater.provider.associate

                def measured_associate(*positional, **keywords):
                    started = time.monotonic()
                    batch = associate(*positional, **keywords)
                    if state['active']:
                        events['association'].append(dict(**mark(), seconds=time.monotonic()-started,
                            timestamp=batch.timestamp))
                    return batch
                bundle.world_updater.provider.associate = measured_associate
            state['semantic_published'] = int(bundle.observation_runtime.diagnostics.get(
                'semantic', {}).get('published', 0))
            state['mapping_published'] = int(bundle.spatial_mapping.health.get('published', 0))
        scope = PhysicsCallbackScope(world)

        actor_elapsed = 0.

        def physics(dt):
            nonlocal actor_elapsed
            if state['active']:
                started = time.monotonic()
                if state['phase'] == 'moving':
                    actor_elapsed += float(dt)
                    if actor_elapsed >= .1:
                        actor_elapsed = 0.
                        x = float(actor.get_world_pose()[0][0])
                        state['actor_path_m'] += abs(x-state['actor_last_x'])
                        state['actor_last_x'] = x
                        state['actor_x_range'][0] = min(state['actor_x_range'][0], x)
                        state['actor_x_range'][1] = max(state['actor_x_range'][1], x)
                        direction = -1. if x >= .62 else 1. if x <= .42 else state['actor_direction']
                        if direction != state['actor_direction']:
                            state['actor_direction'] = direction
                            actor.set_linear_velocity(np.array([direction*.018, 0., 0.]))
                events['physics'].append(dict(**mark(), dt=float(dt),
                    simulation_time=float(world.current_time), observer_seconds=time.monotonic()-started))
                if args.pipeline == 'runtime':
                    poll_runtime()
        scope.add_physics_callback('v5_diagnostic_physics', physics)

        def kit(_event):
            if state['active']:
                events['kit'].append(mark())
        subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            kit, name='v5_runtime_diagnostic')

        if args.pipeline == 'sync' and args.mode in ('mapping', 'combined'):
            settings = SpatialMappingConfig(enabled=True, source='rgbd', automatic_robot_filter=True,
                bounds_min=(-.3, -.6, -.02), bounds_max=(.85, .6, 1.12), voxel_size=.01,
                camera_prim_paths=bundle.config.multi_view.camera_prim_paths,
                stride=8, max_rays=1024)
            mapping = create_spatial_mapping(settings, stage=stage,
                robot_prim_path=bundle.config.robot.prim_path)
            elapsed = 0.

            def mapping_callback(dt):
                nonlocal elapsed
                if not state['active']:
                    return
                elapsed += float(dt)
                if elapsed < .5:
                    return
                elapsed = max(0., elapsed-.5)
                started, frames, changed, error = time.monotonic(), None, False, None
                acquired, update_started = started, None
                try:
                    frames = (tap.capture() if args.mode == 'combined'
                              else bundle.world_updater.provider.rig.capture())
                    acquired = time.monotonic()
                    if frames is None:
                        mapping.map.note_no_frame()
                    else:
                        update_started = time.monotonic()
                        changed = mapping.update(frames)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                events['mapping'].append(dict(**mark(), seconds=time.monotonic()-started,
                    acquisition_seconds=acquired-started,
                    update_seconds=None if update_started is None else time.monotonic()-update_started,
                    accepted=bool(changed), error=error, health=serial(mapping.map.health),
                    frames=[] if frames is None else [dict(camera_id=f.camera_id,
                        token=serial(f.token), acquired_at=f.acquired_at) for f in frames],
                    diagnostics=serial(mapping.source.diagnostics),
                    filter=serial(mapping.robot_filter.diagnostics)))
            scope.add_physics_callback('v5_diagnostic_sync_mapping', mapping_callback)
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

        backend = bundle.backend
        forward, apply = backend.motion_controller.forward, backend.robot.apply_action
        tcp, next_update = backend.get_end_effector_pose, backend._next_update

        def traced_forward(*positional, **keywords):
            started = time.monotonic()
            result = forward(*positional, **keywords)
            if state['active']:
                events['forward'].append(dict(**mark(), seconds=time.monotonic()-started))
            return result

        def traced_apply(action, *positional, **keywords):
            started = time.monotonic()
            result = apply(action, *positional, **keywords)
            if state['active']:
                actual = np.asarray(backend.robot.get_joint_positions(), float)
                desired = getattr(action, 'joint_positions', None)
                tracking = None
                if desired is not None:
                    desired = np.asarray([np.nan if value is None else value for value in desired], float)
                    indices = getattr(action, 'joint_indices', None)
                    measured = actual[:len(desired)] if indices is None else actual[np.asarray(indices, int)]
                    good = np.isfinite(desired) & np.isfinite(measured)
                    if np.any(good):
                        tracking = float(np.sqrt(np.mean((desired[good]-measured[good])**2)))
                events['apply'].append(dict(**mark(), seconds=time.monotonic()-started,
                    joint_target_error_rms_rad=tracking, joints=actual.tolist()))
            return result

        def traced_tcp():
            pose = tcp()
            if state['active']:
                goal = state['goal']
                events['tcp'].append(dict(**mark(), simulation_time=float(world.current_time),
                    position=pose.position.tolist(), orientation=serial(pose.orientation),
                    goal_distance_m=None if goal is None else float(np.linalg.norm(pose.position-goal))))
            return pose

        async def traced_next_update():
            started = time.monotonic()
            await next_update()
            if state['active']:
                events['control_wait'].append(dict(**mark(), seconds=time.monotonic()-started))

        backend.motion_controller.forward, backend.robot.apply_action = traced_forward, traced_apply
        backend.get_end_effector_pose, backend._next_update = traced_tcp, traced_next_update

    async def movements():
        phases = ('stationary', 'moving') if args.objects == 'both' else (args.objects,)
        for phase in phases:
            state.update(phase=phase, command=None, goal=None)
            state['actor_direction'] = 1.
            actor.set_linear_velocity(np.array([.018, 0., 0.]) if phase == 'moving' else np.zeros(3))
            actor.set_angular_velocity(np.zeros(3))
            initial_position = np.asarray(actor.get_world_pose()[0]).copy()
            state.update(actor_path_m=0., actor_last_x=float(initial_position[0]),
                         actor_x_range=[float(initial_position[0])]*2)
            started = time.monotonic()
            runtime_start = (serial(bundle.observation_runtime.diagnostics)
                             if args.pipeline == 'runtime' else None)
            state['active'] = True
            for label, position in [('hover', [.45, 0., .25]), ('side', [.45, .08, .25]),
                                    ('return', [.45, 0., .25]), ('home', None)]:
                state['command'] = label
                target = (bundle.backend.named_poses['home'] if position is None
                          else Pose(position, [0., 0., 1., 0.]))
                state['goal'] = target.position.copy()
                before = time.monotonic()
                result = await bundle.runtime.execute_async(
                    'home' if position is None else 'move_to_pose',
                    **({} if position is None else dict(target=target)))
                events['motions'].append(dict(**mark(), seconds=time.monotonic()-before,
                    target=serial(target), result=serial(result), ok=result.ok))
                print(f"[v5 diagnostic] {args.mode}/{phase}/{label}: {result.status.value}", flush=True)
            state.update(command='post_motion_observation', goal=None)
            while time.monotonic()-started < args.seconds:
                await omni.kit.app.get_app().next_update_async()
            state['active'] = False
            final_position = np.asarray(actor.get_world_pose()[0]).copy()
            events['windows'].append(dict(phase=phase, started=started, ended=time.monotonic(),
                runtime_start=runtime_start,
                runtime_end=serial(bundle.observation_runtime.diagnostics) if args.pipeline == 'runtime' else None,
                actor_start=initial_position.tolist(), actor_end=final_position.tolist(),
                actor_observed_path_m=state['actor_path_m'], actor_x_range=state['actor_x_range'],
                actor_displacement_m=float(np.linalg.norm(final_position-initial_position))))
            actor.set_linear_velocity(np.zeros(3))

    try:
        if not open_stage(str(SCENE)):
            raise RuntimeError('Could not open existing final-perception scene')
        for _ in range(600):
            if not is_stage_loading():
                break
            app.update()
        else:
            raise RuntimeError('Stage loading timed out')
        with ExitStack() as stack:
            stack.enter_context(patch.object(MultiViewProvider, 'observe', observe))
            stack.enter_context(patch.object(WorldModelUpdater, '_on_physics', updater_callback))
            stack.enter_context(patch.object(PhysicsCallbackScope, 'add_physics_callback', add_callback))
            if args.pipeline == 'runtime':
                from robot_skill_stack.integrations.isaac.observation_runtime import ObservationRuntime
                original_on_update = ObservationRuntime._on_update

                def on_update(coordinator, *positional, **keywords):
                    started = time.monotonic()
                    try:
                        return original_on_update(coordinator, *positional, **keywords)
                    finally:
                        if state['active']:
                            events['callbacks'].append(dict(**mark(), name='observation_runtime',
                                seconds=time.monotonic()-started))
                stack.enter_context(patch.object(ObservationRuntime, '_on_update', on_update))
            drive(setup())
            drive(movements())
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        state['active'] = False
        if scope is not None:
            for name in tuple(scope.owned):
                try:
                    scope.remove_physics_callback(name)
                except Exception:
                    errors.append(traceback.format_exc())
            cleanup['diagnostic_callbacks_removed'] = not scope.owned
        subscription = None
        for name, resource in [('mapping', mapping), ('shared_frames', tap), ('bundle', bundle)]:
            if resource is not None:
                try:
                    resource.close()
                    cleanup[name] = 'closed'
                except Exception:
                    errors.append(traceback.format_exc())

    summaries = {}
    for phase in ('stationary', 'moving'):
        windows = [row for row in events['windows'] if row['phase'] == phase]
        if not windows:
            continue
        rows = {key: [row for row in value if row['phase'] == phase]
                for key, value in events.items()}
        summary = dict(wall_seconds=sum(row['ended']-row['started'] for row in windows))
        for key in ('physics', 'kit', 'apply', 'control_wait', 'tcp'):
            summary[key+'_interval_ms'] = distribution(np.diff([row['at'] for row in rows[key]]), 1000)
        for key in ('app_update', 'updater_callback', 'semantic', 'mapping', 'forward', 'apply', 'control_wait'):
            summary[key+'_duration_ms'] = distribution([row['seconds'] for row in rows[key]], 1000)
        summary['callbacks_ms'] = {name: distribution([row['seconds'] for row in rows['callbacks']
            if row['name'] == name], 1000) for name in sorted({row['name'] for row in rows['callbacks']})}
        for key in ('acquisition', 'association'):
            summary[key+'_duration_ms'] = distribution([row['seconds'] for row in rows[key]], 1000)
        summary['observer_duration_ms'] = distribution([row['observer_seconds'] for row in rows['physics']], 1000)
        summary['simulation_dt_ms'] = distribution([row['dt'] for row in rows['physics']], 1000)
        summary['joint_target_error_rms_rad'] = distribution([row['joint_target_error_rms_rad']
            for row in rows['apply'] if row['joint_target_error_rms_rad'] is not None])
        summary['semantic'] = dict(attempts=len(rows['semantic']),
            fresh_batches=sum(row['timestamp'] is not None for row in rows['semantic']),
            published_batches=sum(row.get('published', 1) for row in rows['semantic'])
                if args.pipeline == 'runtime' else sum(row['published'] for row in rows['updater_callback']),
            errors=sum(row['error'] is not None for row in rows['semantic']))
        summary['semantic']['published_hz_wall'] = summary['semantic']['published_batches']/summary['wall_seconds']
        summary['semantic']['no_new_frame'] = sum(row['timestamp'] is None and row['error'] is None
            for row in rows['semantic'])
        summary['semantic']['source_age_ms'] = distribution([row['at']-row['timestamp']
            for row in rows['semantic'] if row['timestamp'] is not None], 1000)
        summary['semantic']['publish_age_ms'] = distribution([row['worker']['publish_age_seconds']
            for row in rows['semantic'] if row.get('worker', {}).get('publish_age_seconds') is not None], 1000)
        stages = sorted({key for row in rows['semantic'] if row['timestamp'] is not None
                         for key in row['diagnostics'] if key.endswith('_ms')})
        summary['semantic']['stages_ms'] = {key: distribution([row['diagnostics'][key]
            for row in rows['semantic'] if row['timestamp'] is not None and key in row['diagnostics']])
            for key in stages}
        summary['mapping'] = dict(attempts=len(rows['mapping']),
            accepted=sum(row['accepted'] for row in rows['mapping']),
            errors=sum(row['error'] is not None for row in rows['mapping']))
        summary['mapping']['accepted_hz_wall'] = summary['mapping']['accepted']/summary['wall_seconds']
        summary['mapping']['not_accepted'] = sum(not row['accepted'] for row in rows['mapping'])
        summary['mapping']['completion_age_ms'] = distribution([row['at']-min(
            frame['acquired_at'] for frame in row['frames']) for row in rows['mapping'] if row['frames']], 1000)
        summary['mapping']['acquisition_ms'] = distribution([row['acquisition_seconds']
            for row in rows['mapping'] if row['acquisition_seconds'] is not None], 1000)
        summary['mapping']['update_ms'] = distribution([row['update_seconds']
            for row in rows['mapping'] if row['update_seconds'] is not None], 1000)
        summary['mapping']['filter_ms'] = distribution([row['filter']['elapsed_seconds']
            for row in rows['mapping'] if 'elapsed_seconds' in row['filter']], 1000)
        summary['mapping']['publish_age_ms'] = distribution([row['worker']['publish_age_seconds']
            for row in rows['mapping'] if row.get('worker', {}).get('publish_age_seconds') is not None], 1000)
        summary['mapping']['source_states'] = {key: sum(row['diagnostics'].get('state') == key
            for row in rows['mapping']) for key in sorted({row['diagnostics'].get('state', 'missing')
                                                          for row in rows['mapping']})}
        if args.pipeline == 'runtime':
            summary['runtime_windows'] = [dict(start=window['runtime_start'], end=window['runtime_end'])
                                          for window in windows]
            for label, fields in [('semantic', ('attempts', 'published', 'no_frame', 'rejected',
                    'source_failures', 'worker_failures')), ('mapping', ('submitted', 'published',
                    'rejected', 'failures', 'source_failures', 'worker_failures', 'resets'))]:
                counts = {key: sum(window['runtime_end'][label].get(key, 0)-
                    window['runtime_start'][label].get(key, 0) for window in windows) for key in fields}
                summary[label]['admission_counts'] = counts
                summary[label]['worker_counts'] = {key: sum(window['runtime_end'][label]['worker'].get(key, 0)-
                    window['runtime_start'][label]['worker'].get(key, 0) for window in windows)
                    for key in ('submitted', 'started', 'completed', 'replaced', 'discarded', 'errors', 'delivered')}
                summary[label]['publication_samples'] = len(rows[label])
                summary[label]['attempts'] = counts.get('attempts', counts.get('submitted', 0))
                completions = {}
                for row in rows['runtime']:
                    metrics = row['coordinator']['semantic'] if label == 'semantic' else row['mapping']
                    last = metrics.get('last') or {}
                    completed_at = last.get('completed_at')
                    if completed_at is not None and any(
                            window['started'] <= completed_at <= window['ended'] for window in windows):
                        completions[last['completed_at']] = last
                for name in ('worker_seconds', 'queue_seconds', 'publish_age_seconds'):
                    summary[label]['all_completion_'+name.replace('_seconds', '_ms')] = distribution([
                        item[name] for item in completions.values() if item.get(name) is not None], 1000)
        summary['motion_successes'] = sum(row['ok'] for row in rows['motions'])
        summary['motion_attempts'] = len(rows['motions'])
        summaries[phase] = summary
    final_hashes = hashes()
    contracts = dict(source_unchanged=initial_hashes == final_hashes,
        completed_windows=set(summaries) == ({'stationary', 'moving'} if args.objects == 'both'
                                            else {args.objects}),
        ordinary_async_motion_only=True,
        robot_joint_motion_observed=any(np.linalg.norm(np.asarray(row['joints'])-
            np.asarray(events['apply'][0]['joints'])) > .05 for row in events['apply'])
            if events['apply'] else False,
        all_motion_succeeded=bool(events['motions']) and all(row['ok'] for row in events['motions']),
        moving_actor_moved=all(row['actor_observed_path_m'] > .005 for row in events['windows']
                              if row['phase'] == 'moving'),
        semantic_published_each_phase=all(row['semantic']['published_batches'] > 0
            for row in summaries.values()) if args.mode in ('semantic', 'combined') else True,
        mapping_accepted_each_phase=all(row['mapping']['accepted'] > 0
            for row in summaries.values()) if args.mode in ('mapping', 'combined') else True,
        semantic_original_timestamps=all(row['timestamp_matches_acquisition'] is True
            and all(item['timestamp'] == row['timestamp'] for item in row['observations'])
            for row in events['semantic'] if row['timestamp'] is not None),
        mapping_original_timestamps=all(row['timestamp_matches_acquisition'] is True
            for row in events['mapping'] if row['accepted']) if args.pipeline == 'runtime' else all(
            row['health']['timestamp'] == min(frame['acquired_at'] for frame in row['frames'])
            for row in events['mapping'] if row['accepted']))
    result = dict(status='PASS' if not errors and all(contracts.values()) else 'FAIL',
        mode=args.mode, pipeline=args.pipeline, objects=args.objects, summaries=summaries, contracts=contracts,
        cleanup=cleanup, errors=errors, hashes_before=initial_hashes, hashes_after=final_hashes,
        protocol=dict(scene=str(SCENE), profile=str(PROFILE), camera_eyes=EYES,
            camera_target=(.15, 0., .45), actual_cameras=state['cameras'],
            semantic_configured_hz=5., mapping_configured_hz=2.,
            execution='Normal Kit update loop plus ordinary async MoveToPose/Home skills; no measured world.step calls.',
            mapping_scope=('Explicit diagnostic synchronous physics callback; mapping is absent from unchanged V4 normal bootstrap.'
                if args.pipeline == 'sync' else 'Real automatic runtime mapping service; no injected mapping callback.'),
            control_scope=('Same MV cameras/render products; semantic observe suppressed after startup.'
                if args.pipeline == 'sync' else 'Same runtime profile/resources; coordinator.set_enabled selects workloads.'),
            interval_scope='Wall-clock intervals, including compute and rendering; physics dt is separately retained.',
            tracking_scope='Joint command-to-measured error and sampled TCP distance-to-goal, not a time-parameterized trajectory error.',
            p99_scope='Sample counts are explicit; tail estimates from short runs are descriptive.',
            moving_object='Actual gravity-disabled PhysX cuboid, .018m/s; owner-context velocity reversals at X=.42/.62, checked each .1 simulated seconds; no pose teleports.',
            freshness_scope=('Original monotonic stamps at completed synchronous call.' if args.pipeline == 'sync'
                else 'publish_age_ms is exact owner publication age; source_age_ms/completion_age_ms include the later diagnostic physics poll. All completion timings include rejected jobs; attempts are admission counter deltas.')+
                ' Rejected/stale/no-frame attempts remain in raw rows; accepted rates use wall time.',
            initialization_and_io_excluded=True))
    (output/'trace.json').write_text(json.dumps(serial(events), indent=2)+'\n')
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(serial(result), indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], summaries=summaries)), flush=True)
    return result['status'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'semantic', 'mapping', 'combined'), required=True)
    parser.add_argument('--pipeline', choices=('sync', 'runtime'), default='runtime')
    parser.add_argument('--objects', choices=('stationary', 'moving', 'both'), default='both')
    parser.add_argument('--seconds', type=float, default=20., help='Minimum real seconds per object condition')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__} from {np.__file__}')
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({'headless': False})
    try:
        passed = run(args, simulation_app)
    finally:
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
