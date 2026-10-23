"""Automatic observation workers during ordinary Franka Pick/Place and recovery.

The existing physical playground cube is enabled in the session layer. Oracle
poses score the outcome only. No joint teleport, world.step loop, or occupancy
driven trajectory is used. Run this standalone through run_isaac_python.sh.
"""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import json
from pathlib import Path
import threading
import time
import traceback
from unittest.mock import patch

import omni.kit.app
import omni.usd
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from scripts.diagnose_isaac_runtime import SCENE, EYES, hashes, runtime_profile, serial
from tests.support.isaac_async import run_kit_coroutine
from tests.support.isaac_helpers import playground_cube
from tests.support.multiview_native import set_viewpoint


async def until(predicate, timeout=12.):
    started = time.monotonic()
    while not predicate():
        if time.monotonic()-started >= timeout:
            raise AssertionError(f'Condition did not recover in {timeout}s')
        await omni.kit.app.get_app().next_update_async()


async def elapsed(seconds):
    started = time.monotonic()
    await until(lambda: time.monotonic()-started >= seconds, timeout=seconds+2.)


def state(bundle):
    return dict(runtime=serial(bundle.observation_runtime.diagnostics),
        mapping=serial(bundle.spatial_mapping.health),
        semantic_status=bundle.world_updater.health.status,
        semantic_timestamp=bundle.world_updater.health.last_frame_at,
        held=bundle.world_model.held_object_id,
        held_uncertain=bundle.world_model.held_object_uncertain,
        objects=[dict(id=obj.object_id, class_name=obj.class_name, visible=obj.visible,
            last_seen=obj.last_seen, graspable=obj.graspable,
            position=None if obj.pose is None else obj.pose.position.tolist())
            for obj in bundle.world_model.objects()])


def recovery_diagnostics(bundle):
    coordinator = bundle.observation_runtime
    robot_filter = coordinator.robot_filter
    details = dict(state=state(bundle), world_playing=bundle.world.is_playing(),
                   world_simulation_time=float(bundle.world.current_time),
                   pool_status={} if coordinator.pool is None else serial(coordinator.pool.status))
    if robot_filter is not None:
        history = tuple(robot_filter._history)
        details['robot_filter'] = dict(diagnostics=serial(robot_filter.diagnostics),
            history_count=len(history), history_generation=robot_filter._generation,
            capture_error=robot_filter._capture_error,
            history_range=[] if not history else [history[0][0], history[-1][0]],
            callback_id=robot_filter._callback_id,
            callback_retained=robot_filter._callback_id in robot_filter._manager._callbacks,
            manager_simulation_time=float(robot_filter._manager.get_simulation_time()))
    return details


async def fresh(bundle):
    await until(lambda: bundle.world_updater.health.status == 'fresh'
        and bundle.spatial_mapping.health['fresh'])


async def evaluate(args, metrics, checks):
    bundle = None
    profile = runtime_profile(args.artifacts_dir)
    try:
        bundle = await build_runtime(profile)
        assert bundle.observation_runtime is not None and bundle.spatial_mapping is not None
        for camera, eye in zip(bundle.world_updater.provider.rig.cameras.values(), EYES):
            set_viewpoint(camera, eye, target=(.15, 0., .45))
        await fresh(bundle)
        metrics['initial'] = state(bundle)
        cube = playground_cube()

        if not args.lifecycle_only:
            def candidate():
                objects = [obj for obj in bundle.world_model.visible_objects()
                           if obj.class_name == 'cube' and obj.graspable is True]
                return objects[0] if len(objects) == 1 else None
            await until(lambda: candidate() is not None, timeout=20.)
            obj = candidate()
            object_id = obj.object_id
            initial_position = np.asarray(cube.get_world_pose()[0]).copy()
            metrics['target'] = dict(id=object_id, source_position=obj.pose.position.tolist(),
                actual_position=initial_position.tolist(),
                initial_center_error_mm=float(np.linalg.norm(obj.pose.position-initial_position)*1000))
            started = time.monotonic()
            pick = await bundle.runtime.execute_async('pick', object_id=object_id)
            metrics['pick'] = dict(result=serial(pick), seconds=time.monotonic()-started,
                actual_position=serial(cube.get_world_pose()[0]), state=state(bundle))
            checks['pick_succeeded'] = pick.ok
            checks['actual_lift'] = float(cube.get_world_pose()[0][2]-initial_position[2]) > .06
            checks['held_identity'] = (bundle.world_model.held_object_id == object_id
                                      and not bundle.world_model.held_object_uncertain)
            assert pick.ok, pick.message
            home = await bundle.runtime.execute_async('home')
            metrics['held_home'] = serial(home)
            checks['ordinary_held_home_succeeded'] = home.ok
            assert home.ok, home.message
            target = Pose([.4465, .25, initial_position[2]])
            started = time.monotonic()
            place = await bundle.runtime.execute_async('place', target=target, mode='stable')
            metrics['place'] = dict(result=serial(place), seconds=time.monotonic()-started,
                actual_position=serial(cube.get_world_pose()[0]), state=state(bundle))
            checks['place_succeeded'] = place.ok
            checks['release_cleared_held'] = bundle.world_model.held_object_id is None
            assert place.ok, place.message
            await until(lambda: bundle.world_model.get(object_id) is not None
                and bundle.world_model.get(object_id).visible)
            checks['placed_identity_reacquired'] = True
            metrics['place']['actual_target_error_mm'] = float(np.linalg.norm(
                np.asarray(cube.get_world_pose()[0])-target.position)*1000)
            home = await bundle.runtime.execute_async('home')
            metrics['released_home'] = serial(home)
            checks['ordinary_released_home_succeeded'] = home.ok
            assert home.ok, home.message

        if not args.skip_lifecycle:
            coordinator, service = bundle.observation_runtime, bundle.spatial_mapping
            pool = coordinator.pool
            frames = pool.batch('native_freeze_control', tuple(pool.cameras), max_skew=.025)
            assert frames is not None
            with patch.object(pool, 'batch', return_value=frames):
                # Drain any pre-freeze job past the normal one-second freshness
                # limit; an old completion can never become a fresh publication.
                await elapsed(1.1)
                before = (service.snapshot().timestamp, bundle.world_updater.health.last_frame_at)
                await elapsed(.4)
                after = (service.snapshot().timestamp, bundle.world_updater.health.last_frame_at)
                metrics['frozen'] = state(bundle)
                checks['frozen_no_timestamp_advance'] = before == after
                checks['frozen_not_fresh'] = (not service.health['fresh']
                    and bundle.world_updater.health.status != 'fresh')
            await fresh(bundle)
            metrics['frozen_recovery'] = state(bundle)
            checks['frozen_recovered'] = True

            with patch.object(pool, 'batch', side_effect=RuntimeError('native injected camera failure')):
                await until(lambda: bundle.world_updater.last_error is not None
                    and service.health['last_error'] is not None)
                metrics['camera_failure'] = state(bundle)
                checks['source_failure_not_fresh'] = not service.health['fresh']
            await fresh(bundle)
            checks['source_failure_recovered'] = True

            semantic_worker, mapping_worker = coordinator._semantic_worker, service._worker
            with patch.object(semantic_worker, '_process', side_effect=RuntimeError('native semantic worker failure')), \
                    patch.object(mapping_worker, '_process', side_effect=RuntimeError('native mapping worker failure')):
                prior = coordinator.diagnostics['semantic']['worker_failures']
                mapping_prior = service.health['worker_failures']
                await until(lambda: coordinator.diagnostics['semantic']['worker_failures'] > prior
                    and service.health['worker_failures'] > mapping_prior)
                metrics['worker_failure'] = state(bundle)
                checks['worker_failure_observed'] = not service.health['fresh']
            await fresh(bundle)
            checks['worker_failure_recovered'] = True

            previous_generation = coordinator.diagnostics['generation']
            await bundle.world.stop_async()
            await elapsed(.1)
            metrics['stopped'] = state(bundle)
            checks['stop_not_fresh'] = (not service.health['fresh']
                and bundle.world_updater.health.status != 'fresh')
            await bundle.world.play_async()
            metrics['after_play'] = recovery_diagnostics(bundle)
            try:
                await fresh(bundle)
            except Exception:
                metrics['stop_play_failure'] = recovery_diagnostics(bundle)
                raise
            metrics['restarted'] = state(bundle)
            checks['play_new_generation'] = coordinator.diagnostics['generation'] > previous_generation

            # A deliberately blocked SDK-free computation proves close does not
            # join it or allow its eventual result to publish after shutdown.
            entered, release, exited = threading.Event(), threading.Event(), threading.Event()
            original = mapping_worker._process

            def blocked(payload):
                entered.set()
                release.wait(timeout=10.)
                try:
                    return original(payload)
                finally:
                    exited.set()
            with patch.object(mapping_worker, '_process', blocked):
                await until(entered.is_set)
                before = service.health['published']
                started = time.monotonic()
                bundle.close()
                metrics['close_seconds'] = time.monotonic()-started
                checks['close_did_not_wait_for_computation'] = not exited.is_set()
                release.set()
                await until(lambda: not mapping_worker.health['running'])
                checks['closed_no_late_publication'] = service.health['published'] == before
                checks['closed_workers'] = semantic_worker.health['closed'] and mapping_worker.health['closed']
                checks['closed_service_not_fresh'] = not service.health['fresh']
            old_bundle = bundle
            bundle = await build_runtime(profile)
            await fresh(bundle)
            metrics['reinitialized'] = state(bundle)
            checks['reinitialized_distinct_owners'] = (bundle is not old_bundle
                and bundle.observation_runtime is not coordinator and bundle.spatial_mapping is not service)
        checks['all_requested_steps_completed'] = True
    except Exception:
        if bundle is not None:
            metrics['failure_state'] = recovery_diagnostics(bundle)
        raise
    finally:
        if bundle is not None:
            bundle.close()
            checks['final_service_closed'] = bundle.spatial_mapping.health['closed']


def main(args):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    before = hashes()
    metrics, checks, error = {}, {}, None
    try:
        if not open_stage(str(SCENE)):
            raise RuntimeError('Could not open existing scene')
        for _ in range(600):
            if not is_stage_loading():
                break
            simulation_app.update()
        else:
            raise RuntimeError('Stage loading timed out')
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
        stage.GetPrimAtPath('/World/Objects/Cube').SetActive(True)
        run_kit_coroutine(evaluate(args, metrics, checks), simulation_app)
    except Exception:
        error = traceback.format_exc()
    checks['source_and_saved_scenes_unchanged'] = before == hashes()
    result = dict(status='PASS' if error is None and checks and all(checks.values()) else 'FAIL',
        checks=checks, metrics=metrics, error=error, hashes=before,
        scope='Normal asynchronous backend/skills and automatic observation runtime; fixed session-only physical cube; all oracle poses are outcome scoring only.')
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(serial(result), indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], checks=checks, error=error)), flush=True)
    return result['status'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--lifecycle-only', action='store_true')
    parser.add_argument('--skip-lifecycle', action='store_true')
    args = parser.parse_args()
    try:
        passed = main(args)
    finally:
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
