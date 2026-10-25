#!/usr/bin/env python3
"""Capture actual fixed Home/Extended/Raised/Low Franka geometry with direct motion.

No box, teleport, gripper command, candidate RRT or target adaptation is used.
The fixed goals are declarations, not claimed achieved postures. Every outcome
and actual physical sample is retained; measurement validity and goal success
are separately recorded, and a failed goal makes the overall outcome FAIL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np

from scripts import benchmark_global_roundtrip_v4_2 as shared


POSTURES = (
    ('Extended', (.65, 0., .35)),
    ('Raised', (.45, -.1, .65)),
    ('Low', (.45, -.1, .20)),
)
DOWN = (0., 0., 1., 0.)
FINGER_RANGE_M = (.001, .005)


def source_hashes():
    return {**shared.source_hashes(), str(Path(__file__).relative_to(shared.ROOT)):shared.digest(__file__)}


def gripper_scope(samples, names):
    """Classify actual named joints and exposed latched targets observationally."""
    fingers = ('panda_finger_joint1', 'panda_finger_joint2')
    if len(names) != len(set(names)) or not set(fingers) <= set(names):
        return dict(classification='INVALID_GRIPPER_FEEDBACK', reason='Missing or ambiguous named fingers')
    indices = [names.index(name) for name in fingers]
    measured, targets, unavailable = [], [], 0
    try:
        for row in samples:
            q = np.asarray(row['joint_feedback']['positions'], float)
            if q.shape != (len(names),) or not np.isfinite(q).all():
                raise ValueError('Invalid actual joint positions')
            measured.append(q[indices])
            feedback = row.get('live_drive_feedback', {}).get('robot_applied_action', {})
            action = feedback.get('value') if feedback.get('available') else None
            if not action or action.get('joint_positions') is None:
                unavailable += 1
                continue
            values = np.asarray(action['joint_positions'], float)
            selected = action.get('joint_indices')
            if selected is None:
                if values.shape != (len(names),) or not np.isfinite(values).all():
                    raise ValueError('Invalid full native position targets')
                targets.append(values[indices])
            else:
                if len(values) != len(selected) or len(set(selected)) != len(selected):
                    raise ValueError('Invalid partial native position targets')
                mapping = dict(zip(selected, values))
                if not set(indices) <= set(mapping):
                    unavailable += 1
                    continue
                pair = np.array([mapping[index] for index in indices])
                if not np.isfinite(pair).all():
                    raise ValueError('Nonfinite native finger targets')
                targets.append(pair)
        if not measured:
            raise ValueError('No actual finger measurements')
        measured, targets = np.asarray(measured), np.asarray(targets).reshape(-1, 2)
        low, high = FINGER_RANGE_M
        outside = bool(np.any(measured < low) or np.any(measured > high)
                       or np.any(targets < low) or np.any(targets > high))
        classification = ('OUTSIDE_ENHANCED_GRIPPER_CONTRACT' if outside else
            'WITHIN_ENHANCED_GRIPPER_CONTRACT' if not unavailable else
            'MEASURED_WITHIN_CONTRACT_TARGETS_UNAVAILABLE')
        return dict(classification=classification, enhanced_contract_range_m=list(FINGER_RANGE_M),
            actual_minimum_m=measured.min(0).tolist(), actual_maximum_m=measured.max(0).tolist(),
            latched_target_minimum_m=targets.min(0).tolist() if len(targets) else None,
            latched_target_maximum_m=targets.max(0).tolist() if len(targets) else None,
            target_unavailable_samples=unavailable,
            interpretation='Direct capture is retained outside this range; it is not mislabeled as validated enhanced-model operation')
    except (TypeError, ValueError, KeyError) as error:
        return dict(classification='INVALID_GRIPPER_FEEDBACK', reason=str(error))


def goal_checks(operations):
    by_label = {row['label']:row for row in operations}
    return {label:shared.acceptance.physical_motion_passed(by_label.get(label, {}), moving=True)
            for label in (*[name for name, _ in POSTURES], 'Home_return')}


def outcome(valid_measurement, goals):
    return dict(status='PASS' if valid_measurement and goals and all(goals.values()) else 'FAIL',
        measurement_status='PASS' if valid_measurement else 'FAIL',
        posture_acceptance=bool(valid_measurement and goals and all(goals.values())))


def run_native(args, app, state, protocol, frozen):
    import omni.kit.app
    from robot_skill_stack.common.types import Pose
    from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
    from tests.support.isaac_async import run_kit_coroutine

    stage = shared.prepare_stage(app)

    async def evaluate():
        bundle, observer, subscription, callback_added = None, None, None, False
        try:
            options = SimpleNamespace(mode='direct_reference', collision_model='stock', seed=123456,
                max_iterations=10000, artifacts_dir=args.artifacts_dir)
            bundle = await build_runtime(shared.write_profile(options))
            observer = shared.controlled.make_observer(bundle)
            state['observer'] = observer
            capture = shared.acceptance.fresh_physical_capture(bundle)
            state['capture'] = capture
            capture['body_paths'] = tuple(observer.metadata['body_paths'])
            state['cooked_fingerprints'] = shared.cooked_fingerprints(capture['meshes'])
            state['startup_match'] = shared.startup_match(observer.state(), bundle.backend.named_poses['home'], protocol, frozen)
            if not state['startup_match']['matched'] or observer.actions:
                raise RuntimeError('Actual startup Home differs from declaration or an unexpected setup action occurred')
            state['world_before_snapshot'], state['world_before_sha256'] = shared._world_signature(stage)
            if any(obstacle.source_kind != 'Plane' for obstacle in state['world_before_snapshot'].obstacles):
                raise RuntimeError('Posture capture requires the declared empty workspace without a box')
            prims = tuple(str(prim.GetPath()) for prim in stage.Traverse())
            observer.metadata.update(robot_dof_names=list(bundle.backend.robot.dof_names),
                meters_per_unit=capture['snapshot'].meters_per_unit,
                declared_posture_sequence=[dict(label='Home', target=frozen['initial_home']),
                    *[dict(label=label, target=dict(position=list(point), orientation=list(DOWN))) for label,point in POSTURES],
                    dict(label='Home_return', target=frozen['initial_home'])],
                sampling='Actual read-only physics callbacks plus operation/update boundaries; no hypothetical FK postures',
                kit_times=[], controller_waits_seconds=[])
            bundle.world.add_physics_callback('v4_2_posture_capture', lambda dt:observer.sample())
            callback_added = True
            update = bundle.backend._next_update
            async def wait():
                began = time.monotonic()
                try:
                    return await update()
                finally:
                    observer.metadata['controller_waits_seconds'].append(time.monotonic()-began)
            bundle.backend._next_update = wait
            subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                lambda event:observer.metadata['kit_times'].append(time.monotonic()), name='v4_2_posture_kit')
            home = observer.begin_operation('Home', bundle.backend.named_poses['home'], kind='observation')
            observer.end_operation(home)
            state['rss_before_motion'] = shared.rss_bytes()
            for label, point in POSTURES:
                await shared.acceptance.motion(observer, label, 'move_to_pose', target=Pose(point, DOWN))
            await shared.acceptance.motion(observer, 'Home_return', 'home')
            state['rss_after_motion'] = shared.rss_bytes()
            state['world_after_snapshot'], state['world_after_sha256'] = shared._world_signature(stage)
            state['same_stage_prims_after_motion'] = prims == tuple(str(prim.GetPath()) for prim in stage.Traverse())
        finally:
            subscription = None
            if bundle is not None:
                if callback_added:
                    bundle.world.remove_physics_callback('v4_2_posture_capture')
                if observer is not None:
                    observer.restore()
                bundle.close()
                state['runtime_closed'] = True
    run_kit_coroutine(evaluate(), app)


def execute_case(args, app, protocol, frozen):
    output = Path(args.artifacts_dir)
    shared.refuse_overwrite(args)
    if (output/'trace.json').exists():
        raise ValueError('Refusing to overwrite an existing posture capture')
    output.mkdir(parents=True, exist_ok=True)
    before, state, error, operations = source_hashes(), {}, None, []
    began = time.monotonic()
    try:
        run_native(args, app, state, protocol, frozen)
    except BaseException:
        error = traceback.format_exc()
    observer, capture = state.get('observer'), state.get('capture')
    if observer is not None and capture is not None:
        try:
            operations = shared.controlled.reduce_observations(observer, capture)
            for summary, operation in zip(operations, observer.operations):
                samples = observer.samples[operation['first_sample']:operation['last_sample']]
                summary['maximum_actual_joint_displacement_rad'] = shared.acceptance.maximum_observed_displacement(
                    operation['before']['feedback']['positions'], samples, observer.indices)
                summary['physics_step_coverage'] = shared.physics_step_coverage(samples)
                summary['gripper_scope'] = gripper_scope(samples, observer.metadata['robot_dof_names'])
                summary['simulation_duration_seconds'] = operation['after']['simulation_time']-operation['before']['simulation_time']
        except BaseException:
            error = (error or '')+'\nOffline reduction:\n'+traceback.format_exc()
        arrays = dict(body_poses=np.asarray([row['poses'] for row in observer.samples]),
            simulation_times=np.asarray([row['simulation_time'] for row in observer.samples]))
        for index, part in enumerate(capture['meshes']):
            arrays.update({f'vertices_{index}':part.mesh.vertices, f'triangles_{index}':part.mesh.triangles,
                           f'body_from_mesh_{index}':part.body_from_mesh})
        np.savez_compressed(output/'geometry_reference.npz', **arrays)
        shared.controlled._json(output/'trace.json', dict(metadata=observer.metadata,
            samples=observer.samples, actions=observer.actions, operations=observer.operations))
    after = source_hashes()
    samples = observer.samples if observer is not None else []
    checks = dict(source_unchanged=before == after, runtime_closed=state.get('runtime_closed', False),
        actual_startup_home=state.get('startup_match', {}).get('matched', False),
        all11_cooked_bodies=bool(state.get('cooked_fingerprints')),
        world_geometry_unchanged=bool(state.get('world_before_sha256')) and state.get('world_before_sha256') == state.get('world_after_sha256'),
        same_stage_prims=state.get('same_stage_prims_after_motion', False),
        owner_thread=bool(samples) and all(row['owner_thread'] for row in samples),
        contact_buffer_unsaturated=bool(samples) and not any(row['contact_buffer_full'] for row in samples),
        complete_physics_steps=bool(operations) and all(row['physics_step_coverage']['complete'] for row in operations))
    goals = goal_checks(operations)
    valid = error is None and all(checks.values())
    classified = outcome(valid, goals)
    payload = dict(schema_version=1, status=classified['status'],
        measurement_status=classified['measurement_status'], contracts=checks, hashes=before, hashes_after=after,
        metrics=dict(configuration=dict(strategy='direct', scenario='v4_2_fixed_postures',
                targets=[dict(label=label, position=list(point), orientation=list(DOWN)) for label,point in POSTURES],
                position_tolerance_m=shared.baseline.POSITION_TOLERANCE,
                orientation_tolerance_rad=shared.baseline.ORIENTATION_TOLERANCE,
                teleport=False, gripper_commands=False, targets_adapted_after_outcomes=False),
            startup_match=state.get('startup_match'), robot_dof_names=observer.metadata['robot_dof_names'] if observer else [],
            meters_per_unit=capture['snapshot'].meters_per_unit if capture else None,
            snapshot=shared.baseline.serial(capture['snapshot']) if capture else None,
            geometry_reference=dict(cooking=capture['cooking'], meshes=len(capture['meshes']), cooked_only=True,
                hand_and_both_fingers=True, all11_bodies=True) if capture else None,
            contact_reference=dict(sensor_paths=observer.metadata['body_paths'],
                filter_paths=observer.metadata['contact_filter_paths']) if observer else None,
            cooked_fingerprints=state.get('cooked_fingerprints'), operations=operations,
            posture_goal_checks=goals, posture_acceptance=classified['posture_acceptance'],
            gripper_scope=gripper_scope(samples, observer.metadata['robot_dof_names']) if observer else None,
            wall_seconds=time.monotonic()-began,
            kit_interval_ms=shared.baseline.distribution(np.diff(observer.metadata.get('kit_times', []))*1000) if observer else {},
            controller_wait_ms=shared.baseline.distribution(np.asarray(observer.metadata.get('controller_waits_seconds', []))*1000) if observer else {},
            rss={key:value for key,value in state.items() if key.startswith('rss_')}),
        error=error, meaning='Measurement validity and posture-goal success are separate. Overall FAIL retains failed/unreachable/contacting fixed goals and every actual observed state. Phase labels declare requested goals; actual measured TCP and returned_success determine whether those goals were reached. No low-reaching state is inferred from older captures.')
    shared.controlled._json(args.result, payload)
    print(json.dumps(dict(status=payload['status'], measurement_status=payload['measurement_status'],
        posture_goal_checks=goals, error=error)), flush=True)
    return payload['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    try:
        shared.refuse_overwrite(args)
        if (Path(args.artifacts_dir)/'trace.json').exists():
            raise ValueError('Refusing to overwrite an existing posture capture')
    except ValueError as error:
        parser.error(str(error))
    protocol, frozen = shared.load_protocol()
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':args.headless})
    try:
        return execute_case(args, app, protocol, frozen)
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
