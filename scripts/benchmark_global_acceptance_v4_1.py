#!/usr/bin/env python3
"""Bounded native Home/cycle/lifecycle acceptance through the unchanged V4 fixture.

One mode per fresh process. Native execution is intentionally separate from the
controlled A–E reproduction harness. Stop/reset/close end the old tensor capture
at the last valid pre-fault state; an action spy then checks quiescence without
reading invalid tensor views. Ordinary bootstrap creates fresh views on rebuild.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import benchmark_global_motion_v4 as fixture
from scripts import diagnose_global_stabilization_v4_1 as controlled
from tests.support.reactive_contact_metrics import capture_joint_feedback


MODES = ('obstacle_home', 'cycles', 'cancel', 'stop', 'reset', 'close')
MINIMUM_MOVEMENT_RAD = .02


def source_hashes():
    rows = controlled.source_hashes()
    rows[str(Path(__file__).relative_to(fixture.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return rows


def maximum_observed_displacement(before, samples, indices):
    """Use finite measured joint states, never requested targets, to gate faults."""
    start = np.asarray(before, dtype=float)
    if start.ndim != 1 or not np.isfinite(start).all():
        raise ValueError('Initial measured joint state is unavailable')
    maximum = 0.
    for sample in samples:
        current = np.asarray(sample['feedback']['positions'], dtype=float)
        if current.shape != start.shape or not np.isfinite(current).all():
            raise ValueError('Measured joint state is unavailable before fault')
        maximum = max(maximum, float(np.max(np.abs((current-start)[list(indices)]))))
    return maximum


async def wait_for_moving_home(observer, row, task, *, clock=time.monotonic, timeout=30.):
    """Yield to Kit until Home has actually moved; refuse a first-command no-op."""
    began = clock()
    while not task.done():
        displacement = maximum_observed_displacement(row['before']['feedback']['positions'],
            observer.samples[row['first_sample']:], observer.indices)
        commanded = len(observer.actions) > row['first_action']
        if observer.backend._motion_active and commanded and displacement > MINIMUM_MOVEMENT_RAD:
            return dict(actual_joint_displacement_rad=displacement, commanded=True,
                active=True, task_complete=False)
        if clock()-began > timeout:
            raise RuntimeError('Home did not begin measured nonzero movement within bounded wait')
        # Isaac's Kit scheduler requires an actual frame await. A repeatedly
        # ready sleep(0) task can starve native frame delivery indefinitely.
        # This is the existing backend Kit await, never a direct world.step.
        await observer.backend._next_update()
    return dict(actual_joint_displacement_rad=maximum_observed_displacement(
        row['before']['feedback']['positions'], observer.samples[row['first_sample']:], observer.indices),
        commanded=len(observer.actions) > row['first_action'], active=False, task_complete=True)


def physical_motion_passed(row, *, moving=False):
    """Keep historical strict-success separately; proximity reports remain visible."""
    return bool(row.get('returned_success') and row.get('measured_full_pose_within_tolerances')
        and row.get('clearance', {}).get('minimum_m') is not None
        and row['clearance']['minimum_m'] > 0.
        and row.get('observed_physical_contact_steps') == 0
        and (not moving or row.get('maximum_actual_joint_displacement_rad', 0.) > MINIMUM_MOVEMENT_RAD))


def post_cancel_hold_metrics(samples, indices, *, extra_actions, runtime_busy_seen):
    """Read actual/live-target behavior under the original tracking bounds.

    Final measured velocity is evidence, not a new cancellation settling gate.
    No command, hold refresh, solver change or controller update is introduced.
    """
    gaps, velocities, stamps = [], [], []
    for sample in samples:
        gap = np.asarray(sample.get('actual_minus_applied_target_arm_rad'), dtype=float)
        velocity = np.asarray(sample['feedback'].get('velocities'), dtype=float)
        if gap.shape != (len(indices),) or not np.isfinite(gap).all() or velocity.ndim != 1 or not np.isfinite(velocity).all():
            return dict(readings_available=False)
        gaps.append(gap)
        velocities.append(velocity[list(indices)])
        stamps.append(float(sample['simulation_time']))
    if not stamps:
        return dict(readings_available=False)
    maximum_gap = np.max(np.abs(gaps), axis=1)
    longest_bad, began_bad = 0., None
    for stamp, gap in zip(stamps, maximum_gap):
        if gap > .15:
            began_bad = stamp if began_bad is None else began_bad
            longest_bad = max(longest_bad, stamp-began_bad)
        else:
            began_bad = None
    return dict(readings_available=True, duration_sim_seconds=stamps[-1]-stamps[0],
        maximum_actual_target_gap_rad=float(np.max(maximum_gap)),
        final_actual_target_gap_arm_rad=gaps[-1].tolist(),
        final_actual_target_gap_max_rad=float(maximum_gap[-1]),
        longest_sampled_above_sustained_limit_seconds=longest_bad,
        tracking_within_original_bounds=bool(np.max(maximum_gap) <= .40 and longest_bad < .25),
        original_tracking_limits=dict(immediate_rad=.40, sustained_rad=.15, grace_seconds=.25),
        final_measured_arm_velocity_rad_s=velocities[-1].tolist(),
        final_measured_arm_velocity_max_rad_s=float(np.max(np.abs(velocities[-1]))),
        maximum_measured_arm_velocity_rad_s=float(np.max(np.abs(velocities))),
        additional_arm_commands=int(extra_actions), no_additional_arm_commands=extra_actions == 0,
        ownership_released_throughout=not runtime_busy_seen and not any(row['arm_motion_active'] for row in samples),
        limitation='Finite physics samples of the unchanged final hold; measured velocity is reported without adding a cancellation velocity-settling criterion.')


async def observe_post_cancel_hold(observer):
    row = observer.begin_operation('post_cancel_hold', None, kind='hold_observation')
    began_wall, updates = time.monotonic(), 0
    runtime_busy_seen = observer.bundle.runtime.busy
    try:
        while float(observer.bundle.world.current_time)-row['before']['simulation_time'] < 1.:
            if time.monotonic()-began_wall > 15. or updates >= 600:
                break
            await observer.backend._next_update()
            runtime_busy_seen |= observer.bundle.runtime.busy
            updates += 1
    finally:
        observer.end_operation(row)
    row['hold_observation'] = post_cancel_hold_metrics(
        observer.samples[row['first_sample']:row['last_sample']], observer.indices,
        extra_actions=row['last_action']-row['first_action'], runtime_busy_seen=runtime_busy_seen)
    return row['hold_observation']


def acceptance_checks(mode, operations, episode, cycles):
    by_label = {row['label']: row for row in operations}
    checks = {label: physical_motion_passed(by_label.get(label, {}))
        for label in ('setup_start', 'move_to_goal')}
    homes = [f'home_cycle_{index}' for index in range(1, cycles+1)] if mode == 'cycles' else (
        ['obstacle_home'] if mode == 'obstacle_home' else ['home_after_cancel'] if mode == 'cancel'
        else [] if mode == 'close' else ['home_after_rebuild'])
    for label in homes:
        checks[label] = physical_motion_passed(by_label.get(label, {}), moving=True)
    if mode == 'cycles':
        for index in range(2, cycles+1):
            label = f'goal_cycle_{index}'
            checks[label] = physical_motion_passed(by_label.get(label, {}), moving=True)
    if mode in ('cancel', 'stop', 'reset', 'close'):
        checks.update({key: bool(episode.get(key)) for key in (
            'genuinely_moving_home', 'competing_request_rejected', 'runtime_competing_request_rejected', 'ownership_released',
            'request_interrupted', 'no_stale_commands_after_completion')})
        partial = by_label.get('interrupted_home', {})
        checks['pre_fault_geometry_clear'] = bool((partial.get('clearance', {}).get('minimum_m') or 0.) > 0.
            and partial.get('observed_physical_contact_steps') == 0)
        if mode == 'cancel':
            checks['cancel_final_velocity_zero'] = bool(partial.get('final_applied_arm_velocity_zero'))
            hold = by_label.get('post_cancel_hold', {})
            readings = hold.get('hold_observation', {})
            checks['cancel_hold_one_second_observed'] = bool(readings.get('readings_available')
                and readings.get('duration_sim_seconds', 0.) >= 1.)
            for key in ('tracking_within_original_bounds', 'no_additional_arm_commands', 'ownership_released_throughout'):
                checks[f'cancel_hold_{key}'] = bool(readings.get(key))
            checks['cancel_hold_zero_velocity_target'] = bool(hold.get('final_applied_arm_velocity_zero'))
            checks['cancel_hold_geometry_clear'] = bool((hold.get('clearance', {}).get('minimum_m') or 0.) > 0.
                and hold.get('observed_physical_contact_steps') == 0)
        else:
            checks['lifecycle_failure_code'] = episode.get('strategy_failure_code') == 'lifecycle_invalidated'
            checks['no_trajectory_restart_after_play'] = bool(episode.get('no_trajectory_restart_after_play'))
        if mode in ('stop', 'reset'):
            checks['rebuild_new_strategy'] = bool(episode.get('rebuild_new_strategy'))
            checks['old_strategy_closed'] = bool(episode.get('old_strategy_closed'))
            checks['move_away_after_rebuild'] = physical_motion_passed(
                by_label.get('move_away_after_rebuild', {}), moving=True)
        if mode == 'close':
            checks['strategy_closed'] = bool(episode.get('strategy_closed'))
    return checks


def fresh_physical_capture(bundle):
    """Read fresh cooked robot geometry and support after every normal bootstrap."""
    import omni.usd
    from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import capture_static_world
    from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _pose_matrices
    stage = omni.usd.get_context().get_stage()
    prims = tuple(str(prim.GetPath()) for prim in stage.Traverse())
    meshes, cooking = fixture._capture_robot_meshes(stage, stage.GetPrimAtPath(bundle.backend.robot.prim_path))
    if prims != tuple(str(prim.GetPath()) for prim in stage.Traverse()):
        raise RuntimeError('Independent geometry capture unexpectedly changed scene prims')
    return dict(meshes=meshes, snapshot=capture_static_world(stage, bundle.backend.robot.prim_path),
        pose_matrices=_pose_matrices, cooking=fixture.serial(cooking))


async def motion(observer, label, name, *, target=None):
    """Retain the ordinary skill/runtime path with a useful acceptance label."""
    target = target if name == 'move_to_pose' else observer.backend.named_poses['home']
    row = observer.begin_operation(label, target)
    result, cancelled = None, False
    try:
        kwargs = dict(target=target, position_tolerance=fixture.POSITION_TOLERANCE,
            orientation_tolerance=fixture.ORIENTATION_TOLERANCE) if name == 'move_to_pose' else {}
        result = await observer._original_execute(name, **kwargs)
        return result
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        observer.end_operation(row, result, cancelled=cancelled)


async def interrupt_home(mode, observer, episode):
    """Measure Home before a fault, then inspect ownership and command quiescence."""
    backend, bundle = observer.backend, observer.bundle
    row = observer.begin_operation('interrupted_home', backend.named_poses['home'], kind='interruption')
    task = asyncio.create_task(observer._original_execute('home'))
    detached, result, cancelled = False, None, False
    original_apply = None
    post_fault_actions = []
    try:
        condition = await wait_for_moving_home(observer, row, task)
        episode['interruption_condition'] = condition
        episode['genuinely_moving_home'] = (condition['active'] and condition['commanded']
            and condition['actual_joint_displacement_rad'] > MINIMUM_MOVEMENT_RAD and not task.done())
        if not episode['genuinely_moving_home']:
            result = await task
            return
        # Exercise both the backend arm owner and the ordinary runtime busy guard.
        competing = await backend.move_to_pose_async(backend.named_poses['home'])
        episode['competing_result'] = fixture.serial(competing)
        episode['competing_request_rejected'] = not competing.ok and 'already in progress' in competing.message
        runtime_competing = await observer._original_execute('home')
        episode['runtime_competing_result'] = fixture.serial(runtime_competing)
        episode['runtime_competing_request_rejected'] = (
            not runtime_competing.ok and fixture.serial(runtime_competing).get('failure_code') == 'busy')
        episode['last_valid_pre_fault_state'] = observer.state()
        if mode != 'cancel':
            # Preserve a bounded valid physical interval. Lifecycle invalidates
            # tensor handles, so ending this capture is an observation boundary.
            observer.end_operation(row)
            observer.metadata['tensor_capture_boundary'] = dict(
                reason=f'{mode} invalidates old physical views; no post-fault tensor reads',
                last_sample=len(observer.samples)-1)
            # An already awaited update may still resume its finally callback
            # after restoration; disable that final read before invalidation.
            observer.sample = lambda: None
            observer.restore()
            detached = True
            original_apply = backend.robot.apply_action

            def apply(action, *args, **kwargs):
                post_fault_actions.append(fixture.serial({name: getattr(action, name, None)
                    for name in ('joint_positions', 'joint_velocities', 'joint_indices')}))
                return original_apply(action, *args, **kwargs)

            backend.robot.apply_action = apply
        if mode == 'cancel':
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                cancelled = True
            episode['request_interrupted'] = cancelled
        else:
            if mode == 'stop':
                await bundle.world.stop_async()
            elif mode == 'reset':
                await bundle.world.reset_async()
            else:
                bundle.close()
            result = await asyncio.wait_for(asyncio.shield(task), timeout=20.)
            episode['request_interrupted'] = not result.ok
            episode['strategy_failure_code'] = (getattr(result, 'details', {}) or {}).get('failure_code')
        episode['result'] = fixture.serial(result)
        episode['ownership_released'] = not backend._motion_active
        episode['nullable_post_fault_joint_feedback'] = capture_joint_feedback(backend.robot)
        count = len(observer.actions) if mode == 'cancel' else len(post_fault_actions)
        for _ in range(3):
            await backend._next_update()
        episode['no_stale_commands_after_completion'] = count == (
            len(observer.actions) if mode == 'cancel' else len(post_fault_actions))
        if mode != 'cancel':
            if mode == 'stop':
                await bundle.world.play_async()
            # Play/reset can make valid handles available; neither may revive
            # the interrupted trajectory. No old-handle repair is attempted.
            count = len(post_fault_actions)
            for _ in range(3):
                await backend._next_update()
            episode['no_trajectory_restart_after_play'] = len(post_fault_actions) == count and not backend._motion_active
            episode['post_fault_actions'] = post_fault_actions
            episode['strategy_closed'] = bool(getattr(backend.motion_strategy, 'diagnostics', {}).get('closed'))
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        if original_apply is not None:
            backend.robot.apply_action = original_apply
        if not detached:
            observer.end_operation(row, result, cancelled=cancelled)


def callback_for(mode, state, cycles):
    async def callback(args, bundle, raw, Pose):
        observer = state['observers'][0]
        primary = next((row for row in observer.operations if row['label'] == 'move_to_goal'), {})
        if not primary.get('returned_success') or not primary.get('measured_full_pose_within_tolerances'):
            state['followup_precondition_failed'] = True
        elif mode == 'obstacle_home':
            await motion(observer, 'obstacle_home', 'home')
        elif mode == 'cycles':
            for index in range(1, cycles+1):
                if index > 1 and not (await motion(observer, f'goal_cycle_{index}', 'move_to_pose',
                    target=Pose(fixture.GOAL, fixture.ORIENTATION))).ok:
                    break
                if not (await motion(observer, f'home_cycle_{index}', 'home')).ok:
                    break
        else:
            episode = state.setdefault('episode', {})
            await interrupt_home(mode, observer, episode)
            if mode == 'cancel' and episode.get('request_interrupted'):
                await observe_post_cancel_hold(observer)
                await motion(observer, 'home_after_cancel', 'home')
            elif mode in ('stop', 'reset') and episode.get('request_interrupted'):
                from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
                old_strategy = bundle.backend.motion_strategy
                rebuilt = await build_runtime(fixture.write_profile(args))
                state['rebuilt_bundle'] = rebuilt
                new_observer = state['observers'][-1]
                episode['rebuild_new_strategy'] = rebuilt.backend.motion_strategy is not old_strategy
                episode['old_strategy_closed'] = bool(getattr(old_strategy, 'diagnostics', {}).get('closed'))
                if (await motion(new_observer, 'move_away_after_rebuild', 'move_to_pose',
                    target=Pose(fixture.GOAL, fixture.ORIENTATION))).ok:
                    await motion(new_observer, 'home_after_rebuild', 'home')
        state['callback_complete'] = True
        return dict(status='PASS', checks=dict(observation_callback_complete=True),
            meaning='Callback/capture completion only; measured acceptance is separate.', excluded_from_primary_timing=True)
    return callback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--obstacle-scenario', choices=('static', 'near'),
        help='Existing fixture for obstacle_home only; default static. Near checks obstacle presence, not a forced detour.')
    parser.add_argument('--cycles', type=int, choices=(2, 3), default=3)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    if args.mode != 'obstacle_home' and args.obstacle_scenario is not None:
        parser.error('--obstacle-scenario applies only to --mode obstacle_home')
    seed = args.seed if args.seed is not None else (345678 if args.mode == 'obstacle_home' else 123456)
    options = SimpleNamespace(**{**vars(args), 'seed': seed}, strategy='global',
        scenario=(args.obstacle_scenario or 'static') if args.mode == 'obstacle_home' else 'empty', enable_mapping=False,
        lifecycle=True, witness=False, reference_result=None)
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy='global', rrt_seed=seed, rrt_max_iterations=args.max_iterations)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    output = Path(args.artifacts_dir)
    if Path(args.result).exists() or (output/'acceptance_trace.json').exists():
        raise ValueError('Refusing to overwrite a recorded acceptance attempt')
    output.mkdir(parents=True, exist_ok=True)
    before, state = source_hashes(), dict(observers=[], startup_captures=[])
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    from robot_skill_stack.integrations.isaac import bootstrap
    original_build, original_native = bootstrap.build_runtime, fixture.run_native

    async def build(profile):
        bundle = await original_build(profile)
        # Fresh geometry/support and observer views for every physical reset.
        state['startup_captures'].append(fresh_physical_capture(bundle))
        state['observers'].append(controlled.make_observer(bundle))
        return bundle

    def native(options, application, result):
        capture = original_native(options, application, result)
        state['physical_capture'] = capture
        return capture

    try:
        with patch.object(bootstrap, 'build_runtime', build), patch.object(fixture, 'run_native', native), \
                patch.object(fixture, 'exercise_lifecycle', callback_for(args.mode, state, args.cycles)):
            fixture.execute_case(options, app)
        payload = json.loads(Path(args.result).read_text())
        operations, raw = [], []
        for index, (observer, startup) in enumerate(zip(state['observers'], state['startup_captures'])):
            capture = state.get('physical_capture') if index == 0 else startup
            # Preserve observations even when the fixture fails before returning
            # its final world capture. Missing reference data must not erase the
            # actual states/actions that explain an invalid attempt.
            arrays = {}
            for mesh_index, mesh in enumerate(startup['meshes']):
                arrays[f'vertices_{mesh_index}'] = mesh.mesh.vertices
                arrays[f'triangles_{mesh_index}'] = mesh.mesh.triangles
                arrays[f'body_from_mesh_{mesh_index}'] = mesh.body_from_mesh
            geometry_path = output/f'acceptance_geometry_{index}.npz'
            np.savez_compressed(geometry_path, **arrays)
            raw.append(dict(runtime_generation=index, metadata=observer.metadata,
                samples=observer.samples, actions=observer.actions, operations=observer.operations,
                startup_world=fixture.serial(startup['snapshot']), cooking=startup['cooking'],
                cooked_artifact=dict(path=str(geometry_path),
                    sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest())))
            if capture is None:
                continue
            # Original setup occurs with fixture actors parked far away, before
            # their final placement. Its physical reference is the startup world.
            # Subsequent primary/callback motion uses the final unchanged world.
            for row in observer.operations:
                one = SimpleNamespace(operations=[row], samples=observer.samples, metadata=observer.metadata,
                    indices=observer.indices)
                selected_capture = startup if index == 0 and row['label'] == 'setup_start' else capture
                reduced = controlled.reduce_observations(one, selected_capture)[0]
                reduced['runtime_generation'] = index
                if 'hold_observation' in row:
                    reduced['hold_observation'] = row['hold_observation']
                operations.append(reduced)
        controlled._json(output/'acceptance_trace.json', raw)
        after = source_hashes()
        valid = dict(source_unchanged=before == after, callback_complete=state.get('callback_complete', False),
            physical_capture_available=state.get('physical_capture') is not None,
            observers_available=bool(state['observers']),
            no_contact_overflow=all(not sample['contact_buffer_full'] for observer in state['observers'] for sample in observer.samples),
            owner_thread=all(sample['owner_thread'] for observer in state['observers'] for sample in observer.samples))
        checks = acceptance_checks(args.mode, operations, state.get('episode', {}), args.cycles)
        payload['acceptance'] = dict(mode=args.mode, seed=seed, checks=checks, observation_checks=valid,
            acceptance_passed=bool(all(valid.values()) and all(checks.values())), operations=operations,
            episode=state.get('episode'), source_hashes_before=before, source_hashes_after=after,
            followup_precondition_failed=state.get('followup_precondition_failed', False),
            raw_trace=dict(path=str(output/'acceptance_trace.json'),
                sha256=hashlib.sha256((output/'acceptance_trace.json').read_bytes()).hexdigest()),
            meaning='All historical strict-success flags/reports retained. Physical acceptance requires measured full pose, positive sampled cooked-solid clearance and no physical-contact steps. Fault boundaries end old tensor sampling; post-fault action quiescence and fresh bootstrap captures are checked separately.')
        payload['status'] = 'PASS' if payload['status'] == 'PASS' and all(valid.values()) else 'FAIL'
        controlled._json(args.result, payload)
        print(json.dumps(dict(status=payload['status'], mode=args.mode,
            acceptance_passed=payload['acceptance']['acceptance_passed'], checks=checks)), flush=True)
        return payload['status'] == 'PASS'
    finally:
        for observer in reversed(state['observers']):
            observer.restore()
        if state.get('rebuilt_bundle') is not None:
            state['rebuilt_bundle'].close()
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
