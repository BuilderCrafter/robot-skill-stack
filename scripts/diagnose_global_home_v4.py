#!/usr/bin/env python3
"""Bounded observational replay of V4 GOAL → cancel START → moving Home.

Only the benchmark's lifecycle callback is substituted. The native fixture,
primary GOAL command, global implementation, limits, physical drives and
controller behavior remain unchanged. The failed Home is retained as failure.
All detailed telemetry stays in the supplied ignored artifact directory.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from scripts import benchmark_global_motion_v4 as benchmark
from tests.support.reactive_contact_metrics import capture_joint_feedback


def _read(call):
    try:
        value = call()
        if hasattr(value, 'joint_positions'):
            value = {field: getattr(value, field, None) for field in (
                'joint_positions', 'joint_velocities', 'joint_efforts', 'joint_indices')}
        return dict(available=True, value=benchmark.serial(value))
    except Exception as error:
        return dict(available=False, error=f'{type(error).__name__}: {error}')


def public_drive_feedback(robot):
    """Public SDK readings only; USD authored targets are separately qualified."""
    values = {'robot_applied_action': _read(robot.get_applied_action)}
    for name in ('get_joint_position_targets', 'get_joint_velocity_targets',
                 'get_max_efforts', 'get_effort_modes'):
        method = getattr(robot, name, None)
        values[name] = _read(method) if callable(method) else dict(available=False, error='Public method unavailable')
    controller = robot.get_articulation_controller()
    for name in ('get_applied_action', 'get_gains', 'get_effort_modes'):
        method = getattr(controller, name, None)
        values['controller_'+name] = _read(method) if callable(method) else dict(available=False, error='Public method unavailable')
    return values


def _usd_number(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    try:
        return [_usd_number(item) for item in value]
    except TypeError:
        return str(value)


def authored_drive_properties(stage, root):
    """Read USD drive attributes without changing authored or PhysX targets."""
    from pxr import Usd, UsdPhysics
    rows = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        attributes = {str(attribute.GetName()): _usd_number(attribute.Get()) for attribute in prim.GetAttributes()
            if (str(attribute.GetName()).startswith('drive:') or str(attribute.GetName()) in (
                'physics:lowerLimit', 'physics:upperLimit', 'physics:jointEnabled', 'physics:axis'))}
        rows.append(dict(path=str(prim.GetPath()), joint_type=str(prim.GetTypeName()), attributes=attributes))
    return dict(joints=rows,
        units='USD angular drive/limit position values use degrees; SDK articulation actions use radians. Linear values use stage distance units.',
        limitation='Authored USD drive target attributes need not reflect live PhysX articulation targets; public get_applied_action readings are recorded separately.')


def lifecycle_probe(capture):
    async def exercise(args, bundle, raw, Pose):
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics

        backend, robot = bundle.backend, bundle.backend.robot
        owner = threading.get_ident()
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(robot.prim_path)
        body_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(root,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(body_paths,
            filter_patterns=[list(body_paths) for _ in body_paths], max_contact_data_count=4096)
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        capture.update(owner_thread=owner, robot_joint_names=list(robot.dof_names),
            requested_home=benchmark.serial(backend.named_poses['home']),
            current_before_cancel=benchmark.serial(backend.get_end_effector_pose()),
            initial_drive_feedback=public_drive_feedback(robot),
            authored_drives=authored_drive_properties(stage, root),
            robot_dof_properties=dict(columns=list(robot.dof_properties.dtype.names),
                values=benchmark.serial(robot.dof_properties)),
            self_contact_reference=dict(sensor_paths=benchmark.serial(view.sensor_paths),
                filter_paths=benchmark.serial(view.filter_paths), capacity=4096,
                meaning='All robot rigid bodies paired with all robot rigid bodies. Existing PhysX collision filtering remains unchanged.'),
            actions=[], samples=[], episodes=[])
        original_apply, original_next = robot.apply_action, backend._next_update
        last_action, phase = None, 'cancel_start'

        def sample():
            row = dict(phase=phase, wall_at=time.monotonic(), owner_thread=threading.get_ident() == owner,
                sim_time=float(bundle.world.current_time), physics_step=int(bundle.world.current_time_step_index),
                physics_dt_s=float(bundle.world.get_physics_dt()), feedback=capture_joint_feedback(robot, last_action),
                live_drive_feedback=public_drive_feedback(robot), self_contacts=[])
            row['robot_base_pose'] = benchmark.serial(robot.get_world_pose())
            impulses, points, normals, separations, counts, starts = view.get_contact_data(1.)
            counts, starts = np.asarray(counts), np.asarray(starts)
            for sensor, filtered in np.argwhere(counts > 0):
                first, number = int(starts[sensor, filtered]), int(counts[sensor, filtered])
                signed = np.asarray(separations)[first:first+number].copy().reshape(-1)*units
                normal = np.asarray(impulses)[first:first+number].copy().reshape(-1)
                row['self_contacts'].append(dict(sensor=str(view.sensor_paths[sensor]),
                    obstacle=str(view.filter_paths[sensor][filtered]), reported_points=number,
                    separations_m=signed, normal_impulses_ns=normal,
                    points_world_m=np.asarray(points)[first:first+number].copy()*units,
                    normals_world=np.asarray(normals)[first:first+number].copy(),
                    minimum_separation_m=float(signed.min()), normal_pair_impulse_ns=float(np.abs(normal).sum())))
            row['self_contact_buffer_saturated'] = int(counts.sum()) >= 4096
            capture['samples'].append(benchmark.serial(row))

        def apply(action, *positional, **keywords):
            nonlocal last_action
            last_action = action
            row = dict(phase=phase, sim_time=float(bundle.world.current_time),
                physics_step=int(bundle.world.current_time_step_index),
                owner_thread=threading.get_ident() == owner,
                q=benchmark.serial(getattr(action, 'joint_positions', None)),
                qd=benchmark.serial(getattr(action, 'joint_velocities', None)),
                indices=benchmark.serial(getattr(action, 'joint_indices', None)))
            value = original_apply(action, *positional, **keywords)
            row['drive_feedback_after_apply'] = public_drive_feedback(robot)
            capture['actions'].append(row)
            return value

        async def update():
            value = await original_next()
            sample()
            return value

        robot.apply_action, backend._next_update = apply, update
        checks = {}
        task = None
        try:
            sample()
            task = asyncio.create_task(backend.move_to_pose_async(Pose(benchmark.START, benchmark.ORIENTATION),
                position_tolerance=benchmark.POSITION_TOLERANCE,
                orientation_tolerance=benchmark.ORIENTATION_TOLERANCE))
            began = time.monotonic()
            while not task.done() and not backend._motion_active:
                if time.monotonic()-began > 30:
                    raise RuntimeError('Diagnostic cancellation request did not start')
                await backend._next_update()
            overlap = await backend.move_to_pose_async(Pose(benchmark.GOAL, benchmark.ORIENTATION))
            checks['overlap_rejected'] = not overlap.ok and 'already in progress' in overlap.message
            checks['cancellation_exercised'] = not task.done()
            if task.done():
                capture['episodes'].append(dict(name='cancel_not_exercised', result=benchmark.serial(await task)))
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    checks['cancellation_propagated'] = True
            checks['cancel_released_arm_owner'] = not backend._motion_active
            capture['cancel_trace'] = benchmark.serial(backend.motion_strategy.last_plan_trace)
            capture['episodes'].append(dict(name='cancel', diagnostics=benchmark.serial(backend.motion_strategy.diagnostics)))
            phase = 'home'
            capture['home_initial_drive_feedback'] = public_drive_feedback(robot)
            sample()
            strategy = backend.motion_strategy
            original_factory = strategy._planner_factory
            if args.time_scale != 1.:
                strategy._planner_factory = time_scaled_planner_factory(original_factory,
                    args.time_scale, capture)
            began = time.monotonic()
            try:
                home = await bundle.runtime.execute_async('home')
            finally:
                strategy._planner_factory = original_factory
            capture['home_wall_seconds'] = time.monotonic()-began
            capture['home_result'] = benchmark.serial(home)
            capture['home_plan_trace'] = benchmark.serial(backend.motion_strategy.last_plan_trace)
            capture['home_final_drive_feedback'] = public_drive_feedback(robot)
            sample()
            checks['home_completed'] = home.ok
            capture['episodes'].append(dict(name='home', result=benchmark.serial(home)))
            capture['checks'] = checks
            capture['diagnostic_capture_complete'] = True
            return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                episodes=capture['episodes'], excluded_from_primary_timing=True,
                limitation='Bounded diagnostic replays only cancellation and moving Home; detailed feedback observation is not a performance benchmark or a substitute for the complete native lifecycle case.')
        finally:
            robot.apply_action, backend._next_update = original_apply, original_next
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
    return exercise


def time_scaled_planner_factory(original_factory, factor, capture):
    """Diagnostic-only exact time warp, followed by original collision checks.

    The existing planner chooses IK and RRT once. No registered object, joint
    limit, robot model, route, or original generator parameter is changed.
    """
    from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import (
        adjacent_self_collision_pairs, quaternion_matrix)
    from robot_skill_stack.manipulation.motion.trajectory import TimeScaledTrajectory

    def factory(backend, **options):
        planner = original_factory(backend, **options)
        original_plan = planner.plan

        def plan(target, start_q, snapshot, position_tolerance, orientation_tolerance):
            result = original_plan(target, start_q, snapshot, position_tolerance, orientation_tolerance)
            if result.duration <= 0.:
                raise RuntimeError('The causal time-warp probe requires a moving Home trajectory')
            wrapped = TimeScaledTrajectory(result.trajectory, factor=factor)
            base_position, base_orientation = backend.robot.get_world_pose()
            base_position = np.asarray(base_position, float)*float(snapshot.meters_per_unit)
            base_rotation = quaternion_matrix(base_orientation)
            excluded_pairs = adjacent_self_collision_pairs(planner._config['urdf_path'])
            began = time.monotonic()
            validation = planner._validate_trajectory(wrapped, snapshot, base_position, base_rotation,
                excluded_pairs, result.lower_limits, result.upper_limits, result.velocity_limits,
                result.acceleration_limits, result.jerk_limits)
            validation_seconds = time.monotonic()-began
            errors = np.zeros(4)
            for fraction in np.linspace(0., 1., 41):
                before = result.trajectory.start_time+fraction*result.duration
                after = wrapped.start_time+fraction*result.duration*factor
                for order in range(4):
                    errors[order] = max(errors[order], float(np.max(np.abs(
                        np.asarray(wrapped.trajectory.eval(float(after), order))
                        -np.asarray(result.trajectory.trajectory.eval(float(before), order))/factor**order))))
            if np.any(errors > 1e-11):
                raise RuntimeError('Exact geometric/derivative time-warp identity failed')
            proof = dict(factor=factor, applies_only_to='moving Home after the nominal cancellation',
                time_helper_sha256=hashlib.sha256((benchmark.ROOT/'robot_skill_stack/manipulation/motion/trajectory.py').read_bytes()).hexdigest(),
                nominal_duration_seconds=result.duration, scaled_duration_seconds=result.duration*factor,
                nominal_start_time=result.trajectory.start_time,
                scaled_start_time=wrapped.start_time,
                maximum_identity_errors_for_orders0_through3=errors.tolist(),
                identity_conditions=41, nominal_trajectory_validation=result.diagnostics.get('trajectory_validation'),
                scaled_trajectory_validation=validation, scaled_validation_seconds=validation_seconds,
                hard_position_limits=[result.lower_limits.tolist(), result.upper_limits.tolist()],
                hard_velocity_limits=result.velocity_limits.tolist(),
                hard_acceleration_limits=result.acceleration_limits.tolist(), hard_jerk_limits=result.jerk_limits.tolist(),
                collision_world_unchanged=True, robot_model_unchanged=True, route_unchanged=True,
                requested_pose_unchanged=True, hard_limits_unchanged=True, drive_gains_unchanged=True,
                method='q_scaled(t0+s)=q_base(t0+s/factor); derivative n divided by factor**n. Original inspector, static snapshot, base frame, self-collision exclusions and hard limits validate the entire scaled curve again.')
            capture['home_time_scaling'] = benchmark.serial(proof)
            updates = dict(trajectory_time_scale=factor,
                nominal_trajectory_duration=result.duration,
                trajectory_duration=result.duration*factor,
                nominal_trajectory_validation=result.diagnostics.get('trajectory_validation'),
                trajectory_validation=validation,
                additional_time_scaled_validation_seconds=validation_seconds)
            planner._last_diagnostics.update(updates)
            diagnostics = {**result.diagnostics, **updates}
            return replace(result, trajectory=wrapped, duration=result.duration*factor,
                diagnostics=diagnostics)
        planner.plan = plan
        return planner
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--time-scale', type=float, choices=(1., 2.), default=1.,
        help='One bounded causal comparison: exact 2x moving-Home timing only, with all original limits and dense collision checks')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    options = SimpleNamespace(**vars(args), strategy='global', scenario='empty',
        enable_mapping=False, reference_result=None, lifecycle=True, witness=False)
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy='global', rrt_seed=args.seed, rrt_max_iterations=args.max_iterations)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    previous, capture = benchmark.exercise_lifecycle, {}
    before = benchmark.source_hashes()
    wrapper_path = Path(__file__)
    wrapper_sha = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    benchmark.exercise_lifecycle = lifecycle_probe(capture)
    try:
        outcome = benchmark.execute_case(options, app)
        capture.update(schema_version=1, wrapper_sha256=wrapper_sha,
            native_result=str(Path(args.result)),
            core_source_hashes_before=before, core_source_hashes_after=benchmark.source_hashes(),
            core_sources_unchanged=before == benchmark.source_hashes(),
            wrapper_unchanged=wrapper_sha == hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
            purpose='Read-only causal diagnosis of moving Home tracking failure; no tuning, drive edits or tolerance changes.')
        output = Path(args.artifacts_dir)
        plan = capture.pop('home_plan_trace', {})
        (output/'home_plan_trace.json').write_text(json.dumps(benchmark.serial(plan), allow_nan=False)+'\n')
        (output/'home_diagnostic.json').write_text(json.dumps(benchmark.serial(capture), allow_nan=False)+'\n')
        print(json.dumps(dict(diagnostic_capture_complete=capture.get('diagnostic_capture_complete', False),
            home_status=capture.get('home_result', {}).get('status'), samples=len(capture.get('samples', [])),
            actions=len(capture.get('actions', [])), core_sources_unchanged=capture['core_sources_unchanged'])), flush=True)
        return outcome
    finally:
        benchmark.exercise_lifecycle = previous
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
