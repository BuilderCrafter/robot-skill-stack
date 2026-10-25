#!/usr/bin/env python3
"""Controlled V4.1 execution reproductions through the unchanged V4 fixture.

One mode per fresh Isaac process. Observers read actual articulation targets,
body poses and contacts; they never edit gains, drives, joint state or the
planner. Failed motion is a recorded outcome, not an invalid observation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import benchmark_global_motion_v4 as fixture
from scripts.diagnose_global_home_v4 import _read, authored_drive_properties, public_drive_feedback
from robot_skill_stack.manipulation.motion.joint_route import map_arm_joints
from tests.support.reactive_contact_metrics import analyze_contact_samples, capture_joint_feedback


MODES = {
    'A': dict(strategy='global', followup=None, cancel=False),
    'B': dict(strategy='global', followup='home', cancel=False),
    'C': dict(strategy='global', followup='home', cancel=True),
    'D': dict(strategy='global', followup='different_goal', cancel=True),
    'E': dict(strategy='direct', followup='home', cancel=False),
    'E_cancel': dict(strategy='direct', followup='home', cancel=True),
}
DIFFERENT_GOAL = (.50, .10, .45)
ARM = tuple(f'panda_joint{i}' for i in range(1, 8))


def source_hashes():
    rows = fixture.source_hashes()
    for path in (Path(__file__), fixture.ROOT/'scripts/diagnose_global_home_v4.py'):
        rows[str(path.relative_to(fixture.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture.serial(value), allow_nan=False, indent=2)+'\n')


def arm_target_error(positions, applied_action, indices):
    """Actual target-buffer error, not a time-matched trajectory error."""
    if positions is None:
        return None
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 1 or not np.isfinite(positions).all():
        return None
    targets = applied_action.get('joint_positions')
    if targets is None:
        return None
    targets = np.asarray(targets, dtype=float)
    selected = applied_action.get('joint_indices')
    if selected is None and targets.shape == positions.shape:
        return (positions[list(indices)]-targets[list(indices)]).tolist()
    if selected is not None and len(selected) == len(targets):
        lookup = dict(zip(map(int, selected), targets))
        if all(index in lookup for index in indices):
            return [float(positions[index]-lookup[index]) for index in indices]
    return None


def dynamics_feedback(robot):
    """Read-only SDK dynamics; signals are not calibrated motor torques."""
    view = getattr(robot, '_articulation_view', None)
    fields = ('get_mass_matrices', 'get_coriolis_and_centrifugal_forces',
        'get_generalized_gravity_forces', 'get_solver_position_iteration_counts',
        'get_solver_velocity_iteration_counts', 'get_enabled_self_collisions')
    result = {name: (_read(getattr(view, name)) if callable(getattr(view, name, None))
                   else dict(available=False, error='Installed public view getter unavailable'))
            for name in fields}
    native = getattr(view, '_physics_view', None)
    drive_getter = getattr(view, 'get_drive_types', None)
    drive_source = 'articulation_view.get_drive_types'
    if not callable(drive_getter):
        drive_getter = getattr(native, 'get_drive_types', None)
        drive_source = 'public_tensor_view.get_drive_types'
    result['get_drive_types'] = (_read(drive_getter) if callable(drive_getter)
        else dict(available=False, error='Installed drive-type getter unavailable'))
    result['get_drive_types']['source'] = drive_source
    getter = getattr(native, 'get_dof_drive_model_properties', None)
    result['get_dof_drive_model_properties'] = (_read(getter) if callable(getter)
        else dict(available=False, error='Installed public tensor getter unavailable'))
    return result


def contact_rows(view, units):
    impulses, points, normals, separations, counts, starts = view.get_contact_data(1.)
    counts, starts = np.asarray(counts), np.asarray(starts)
    rows = []
    for sensor, filtered in np.argwhere(counts > 0):
        first, count = int(starts[sensor, filtered]), int(counts[sensor, filtered])
        signed = np.asarray(separations)[first:first+count].copy().reshape(-1)*units
        normal = np.asarray(impulses)[first:first+count].copy().reshape(-1)
        other = str(view.filter_paths[sensor][filtered])
        rows.append(dict(sensor=str(view.sensor_paths[sensor]), obstacle=other,
            kind='self' if other == '/Franka' or other.startswith('/Franka/') else 'external',
            count=count, minimum_separation_m=float(signed.min()), impulse_magnitude=float(np.abs(normal).sum()),
            normal_impulses_ns=normal, separations_m=signed,
            points_world_m=np.asarray(points)[first:first+count].copy()*units,
            normals_world=np.asarray(normals)[first:first+count].copy()))
    return fixture.serial(rows), int(counts.sum()) >= 4096


def _lula_fk(backend, q, base_pose):
    """Model FK is kept distinct from independent PhysX rigid-body poses."""
    try:
        solver = backend.lula_solver
        names = tuple(solver.get_joint_names())
        indices = map_arm_joints(names, backend.robot.dof_names)
        solver.set_robot_base_pose(*[np.asarray(value, dtype=float) for value in base_pose])
        arm = np.asarray(q, dtype=float)[list(indices)]
        frames = {name: _read(lambda name=name: solver.compute_forward_kinematics(name, arm))
                  for name in ('panda_link1', 'panda_hand', backend.end_effector_frame)}
        return dict(available=True, active_names=list(names), frames=frames,
            meaning='Existing Lula robot-model FK in world stage units; not physical body observations. Fingers use descriptor assumptions.')
    except Exception as error:
        return dict(available=False, error=f'{type(error).__name__}: {error}')


class Observer:
    """Bounded observations attached before START setup and primary GOAL."""
    def __init__(self, bundle, *, contact_view, body_view, units, metadata):
        self.bundle, self.backend, self.robot = bundle, bundle.backend, bundle.backend.robot
        self.contact_view, self.body_view, self.units = contact_view, body_view, units
        self.metadata, self.owner = metadata, threading.get_ident()
        self.indices = map_arm_joints(ARM, self.robot.dof_names)
        self.samples, self.actions, self.operations, self.kit_times = [], [], [], []
        self.phase, self.operation_id, self.last_action = 'idle', None, None
        self._original_apply = self.robot.apply_action
        self._original_update = self.backend._next_update
        self._original_execute = bundle.runtime.execute_async
        self._runtime_calls = 0

    def state(self):
        feedback = capture_joint_feedback(self.robot)
        base = self.robot.get_world_pose()
        pose = self.backend.get_end_effector_pose()
        return fixture.serial(dict(simulation_time=float(self.bundle.world.current_time),
            physics_step=int(self.bundle.world.current_time_step_index),
            feedback=feedback, live_drive_feedback=public_drive_feedback(self.robot),
            tcp_pose=pose, gripper=self.backend.gripper_feedback(), robot_base_pose=base,
            arm_motion_active=bool(self.backend._motion_active), articulation_ready=self.backend.is_ready,
            playing=bool(self.bundle.world.is_playing()),
            actual_minus_applied_target_arm_rad=arm_target_error(feedback['positions'],
                feedback['action'], self.indices)))

    def sample(self):
        if len(self.samples) >= 20000:
            raise RuntimeError('Bounded observational sample budget exhausted')
        state = self.state()
        contacts, saturated = contact_rows(self.contact_view, self.units)
        state.update(phase=self.phase, operation_id=self.operation_id,
            at=time.monotonic(), owner_thread=threading.get_ident() == self.owner,
            physics_dt_s=float(self.bundle.world.get_physics_dt()),
            joint_feedback=state['feedback'], contacts=contacts,
            contact_buffer_full=saturated,
            poses=np.asarray(self.body_view.get_transforms()).copy(),
            body_velocities=np.asarray(self.body_view.get_velocities()).copy(),
            requested_action=None if self.last_action is None else {
                name: fixture.serial(getattr(self.last_action, name, None)) for name in
                ('joint_positions', 'joint_velocities', 'joint_efforts', 'joint_indices')},
            dynamics=dynamics_feedback(self.robot))
        state['lula_model_fk'] = _lula_fk(self.backend, state['feedback']['positions'], state['robot_base_pose'])
        self.samples.append(fixture.serial(state))

    def begin_operation(self, label, target, *, kind='motion'):
        self.operation_id = len(self.operations)
        self.phase = label
        row = dict(id=self.operation_id, label=label, kind=kind,
            target=fixture.serial(target), before=self.state(), first_sample=len(self.samples),
            first_action=len(self.actions), wall_started=time.monotonic())
        self.operations.append(row)
        self.sample()
        return row

    def end_operation(self, row, result=None, *, cancelled=False):
        self.sample()
        row.update(after=self.state(), result=fixture.serial(result), cancelled=cancelled,
            last_sample=len(self.samples), last_action=len(self.actions),
            wall_seconds=time.monotonic()-row['wall_started'],
            strategy_diagnostics=fixture.serial(getattr(self.backend.motion_strategy, 'diagnostics', {})),
            plan_trace=fixture.serial(getattr(self.backend.motion_strategy, 'last_plan_trace', None)))
        before = np.asarray(row['before']['feedback']['positions'], dtype=float)[list(self.indices)]
        after = np.asarray(row['after']['feedback']['positions'], dtype=float)[list(self.indices)]
        row['maximum_actual_joint_displacement_rad'] = float(np.max(np.abs(after-before)))
        if row['target'] is not None:
            actual = row['after']['tcp_pose']
            row['position_error_m'] = float(np.linalg.norm(np.asarray(actual['position'])-row['target']['position']))
            row['orientation_error_rad'] = self.backend._quat_error(
                np.asarray(actual['orientation']), np.asarray(row['target']['orientation']))
            row['measured_full_pose_within_tolerances'] = (
                row['position_error_m'] <= fixture.POSITION_TOLERANCE
                and row['orientation_error_rad'] <= fixture.ORIENTATION_TOLERANCE)
        row['returned_success'] = bool(getattr(result, 'ok', False)) if result is not None else False
        self.operation_id, self.phase = None, 'idle'

    def install(self):
        def apply(action, *args, **kwargs):
            self.last_action = action
            result = self._original_apply(action, *args, **kwargs)
            self.actions.append(fixture.serial(dict(phase=self.phase, operation_id=self.operation_id,
                at=time.monotonic(), simulation_time=float(self.bundle.world.current_time),
                physics_step=int(self.bundle.world.current_time_step_index),
                requested_positions=getattr(action, 'joint_positions', None),
                requested_velocities=getattr(action, 'joint_velocities', None),
                requested_indices=getattr(action, 'joint_indices', None),
                actual_physx_targets=public_drive_feedback(self.robot)['robot_applied_action'])))
            return result

        async def update():
            try:
                return await self._original_update()
            finally:
                self.sample()

        async def execute(name, *args, **kwargs):
            self._runtime_calls += 1
            label = ('setup_start' if self._runtime_calls == 1 else
                     'move_to_goal' if self._runtime_calls == 2 else
                     'different_goal' if name == 'move_to_pose' else name)
            target = kwargs.get('target') if name == 'move_to_pose' else self.backend.named_poses.get(kwargs.get('name', 'home'))
            row = self.begin_operation(label, target)
            result = None
            cancelled = False
            try:
                result = await self._original_execute(name, *args, **kwargs)
                return result
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                self.end_operation(row, result, cancelled=cancelled)

        self.robot.apply_action, self.backend._next_update = apply, update
        self.bundle.runtime.execute_async = execute

    def restore(self):
        self.robot.apply_action = self._original_apply
        self.backend._next_update = self._original_update
        self.bundle.runtime.execute_async = self._original_execute


def make_observer(bundle):
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager
    stage = omni.usd.get_context().get_stage()
    robot_root = stage.GetPrimAtPath(bundle.backend.robot.prim_path)
    robot_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(robot_root,
        Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    external = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == '/Franka' or path.startswith('/Franka/') or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        external.append(str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else path)
    filters = list(dict.fromkeys([*robot_paths, *external]))
    sim = SimulationManager.get_physics_sim_view()
    contacts = sim.create_rigid_contact_view(robot_paths,
        filter_patterns=[filters for _ in robot_paths], max_contact_data_count=4096)
    bodies = sim.create_rigid_body_view(robot_paths)
    if not set(robot_paths) <= set(bodies.prim_paths):
        raise RuntimeError('Independent body view omitted physical robot links')
    if not set(robot_paths) <= set(contacts.sensor_paths):
        raise RuntimeError('Controlled contact view omitted physical robot sensors')
    if any(not set(filters) <= set(group) for group in contacts.filter_paths):
        raise RuntimeError('Controlled contact view omitted external or self filters')
    metadata = dict(body_paths=list(bodies.prim_paths), contact_sensor_paths=fixture.serial(contacts.sensor_paths),
        contact_filter_paths=fixture.serial(contacts.filter_paths), contact_capacity=4096,
        self_pair_directions_may_duplicate=True,
        local_coms=fixture.serial(bodies.get_coms()),
        drive_type_encoding={'0': 'none', '1': 'force', '2': 'acceleration'},
        drive_model_property_order=['speed_effort_gradient', 'max_actuator_velocity', 'velocity_dependent_resistance'],
        authored_drive_properties=authored_drive_properties(stage, robot_root),
        robot_dof_names=list(bundle.backend.robot.dof_names),
        robot_dof_properties=dict(columns=list(bundle.backend.robot.dof_properties.dtype.names),
            values=fixture.serial(bundle.backend.robot.dof_properties)),
        actual_body_pose_convention='PhysX world positions plus quaternion xyzw; meters_per_unit recorded separately',
        initial_dynamics=dynamics_feedback(bundle.backend.robot))
    observer = Observer(bundle, contact_view=contacts, body_view=bodies,
        units=float(UsdGeom.GetStageMetersPerUnit(stage)), metadata=metadata)
    observer.metadata['initial_state'] = observer.state()
    observer.install()
    return observer


async def cancel_start(observer, Pose):
    backend = observer.backend
    row = observer.begin_operation('cancel_start', Pose(fixture.START, fixture.ORIENTATION), kind='cancellation')
    before = len(observer.actions)
    task = asyncio.create_task(backend.move_to_pose_async(Pose(fixture.START, fixture.ORIENTATION),
        position_tolerance=fixture.POSITION_TOLERANCE, orientation_tolerance=fixture.ORIENTATION_TOLERANCE))
    cancelled = False
    result = None
    try:
        began = time.monotonic()
        while not task.done() and not backend._motion_active:
            if time.monotonic()-began > 30:
                raise RuntimeError('Controlled cancellation request did not begin within 30 seconds')
            await backend._next_update()
        overlap = await backend.move_to_pose_async(Pose(fixture.GOAL, fixture.ORIENTATION))
        row['overlap_result'] = fixture.serial(overlap)
        row['overlap_rejected'] = not overlap.ok and 'already in progress' in overlap.message
        row['first_command_observed'] = len(observer.actions) > before
        row['active_before_cancel'] = bool(backend._motion_active and not task.done())
        if not row['active_before_cancel'] or not row['first_command_observed']:
            row['not_exercised'] = True
            result = await task
        else:
            row['at_cancel'] = observer.state()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                cancelled = True
        row['ownership_released'] = not backend._motion_active
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        observer.end_operation(row, result, cancelled=cancelled)
    return row


def callback_for(mode, state):
    spec = MODES[mode]
    async def callback(args, bundle, raw, Pose):
        observer = state['observer']
        primary = next((row for row in observer.operations if row['label'] == 'move_to_goal'), None)
        if primary is None or not primary['returned_success'] or not primary['measured_full_pose_within_tolerances']:
            state['followup_precondition_failed'] = True
        else:
            if spec['cancel']:
                await cancel_start(observer, Pose)
            if spec['followup'] == 'home':
                await bundle.runtime.execute_async('home')
            elif spec['followup'] == 'different_goal':
                await bundle.runtime.execute_async('move_to_pose', target=Pose(DIFFERENT_GOAL, fixture.ORIENTATION),
                    position_tolerance=fixture.POSITION_TOLERANCE,
                    orientation_tolerance=fixture.ORIENTATION_TOLERANCE)
        state['callback_complete'] = True
        return dict(status='PASS', checks=dict(observation_callback_complete=True),
            meaning='Capture validity only; operation outcomes and actual pose/tracking determine acceptance separately.',
            excluded_from_primary_timing=True)
    return callback


def reduce_observations(observer, capture):
    summaries = []
    for row in observer.operations:
        samples = observer.samples[row['first_sample']:row['last_sample']]
        transforms = [capture['pose_matrices'](sample['poses'], capture['snapshot'].meters_per_unit)
                      for sample in samples]
        clearance = fixture.reduce_clearances(capture['meshes'], tuple(observer.metadata['body_paths']),
            transforms, capture['snapshot'].obstacles)
        contacts = fixture.contact_summary(samples)
        severity = analyze_contact_samples(samples, tuple(observer.metadata['body_paths']),
            np.asarray(observer.metadata['local_coms']), meters_per_unit=capture['snapshot'].meters_per_unit,
            goal_status=(row.get('result') or {}).get('status', 'cancelled' if row['cancelled'] else 'unknown'),
            failure_code=(row.get('result') or {}).get('failure_code'),
            minimum_clearance_m=clearance['minimum_m'])
        applied = row['after']['live_drive_feedback']['robot_applied_action']
        velocity = (applied.get('value') or {}).get('joint_velocities') if applied.get('available') else None
        zero_velocity = (velocity is not None and np.asarray(velocity).ndim == 1
            and np.max(np.abs(np.asarray(velocity)[list(observer.indices)])) <= 1e-12)
        scalar = {key: row.get(key) for key in ('id', 'label', 'kind', 'cancelled', 'returned_success',
            'position_error_m', 'orientation_error_rad', 'measured_full_pose_within_tolerances',
            'maximum_actual_joint_displacement_rad', 'wall_seconds', 'first_command_observed',
            'active_before_cancel', 'ownership_released', 'overlap_rejected', 'not_exercised')}
        scalar.update(clearance=clearance, contacts=contacts, contact_severity=severity,
            observed_physical_contact_steps=severity['observed_physical_contact_steps'],
            strict_success=(row['returned_success'] and row.get('measured_full_pose_within_tolerances', False)
                and contacts['reported_points'] == 0 and clearance['minimum_m'] is not None and clearance['minimum_m'] > 0.)
                if row['kind'] != 'cancellation' else None,
            final_applied_arm_velocity_zero=bool(zero_velocity),
            sample_count=len(samples), action_count=row['last_action']-row['first_action'],
            before_simulation_time=row['before']['simulation_time'], after_simulation_time=row['after']['simulation_time'],
            before_physics_step=row['before']['physics_step'], after_physics_step=row['after']['physics_step'],
            before_target_gap_arm_rad=row['before']['actual_minus_applied_target_arm_rad'],
            after_target_gap_arm_rad=row['after']['actual_minus_applied_target_arm_rad'],
            strategy_diagnostics=row['strategy_diagnostics'])
        if 'overlap_result' in row:
            scalar['overlap_result'] = row['overlap_result']
        summaries.append(scalar)
    return summaries


def controlled_acceptance(mode, operations):
    by_label = {row['label']: row for row in operations}
    required = ['setup_start', 'move_to_goal']
    if MODES[mode]['followup'] is not None:
        required.append('home' if MODES[mode]['followup'] == 'home' else 'different_goal')
    if not all(name in by_label and by_label[name]['strict_success'] for name in required):
        return False
    if MODES[mode]['cancel']:
        row = by_label.get('cancel_start', {})
        return bool(row.get('cancelled') and row.get('ownership_released') and row.get('overlap_rejected')
            and row.get('final_applied_arm_velocity_zero') and row.get('contacts', {}).get('reported_points') == 0
            and row.get('clearance', {}).get('minimum_m', 0.) > 0.)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(MODES), required=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--reference-result')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    options = SimpleNamespace(**vars(args), strategy=MODES[args.mode]['strategy'], scenario='empty',
        enable_mapping=False, lifecycle=args.mode != 'A', witness=False)
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy=options.strategy, rrt_seed=args.seed, rrt_max_iterations=args.max_iterations)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    output = Path(args.artifacts_dir)
    if (output/'controlled_trace.json').exists() or Path(args.result).exists():
        raise ValueError('Refusing to overwrite a recorded V4.1 attempt')
    output.mkdir(parents=True, exist_ok=True)
    before, state = source_hashes(), {}
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    from robot_skill_stack.integrations.isaac import bootstrap
    original_build, original_native = bootstrap.build_runtime, fixture.run_native

    async def build(profile):
        bundle = await original_build(profile)
        state['observer'] = make_observer(bundle)
        return bundle

    def native(options, application, result):
        capture = original_native(options, application, result)
        state['physical_capture'] = capture
        return capture

    try:
        with patch.object(bootstrap, 'build_runtime', build), patch.object(fixture, 'run_native', native), \
                patch.object(fixture, 'exercise_lifecycle', callback_for(args.mode, state)):
            fixture.execute_case(options, app)
        payload = json.loads(Path(args.result).read_text())
        observer, capture = state.get('observer'), state.get('physical_capture')
        operations = reduce_observations(observer, capture) if observer is not None and capture is not None else []
        after = source_hashes()
        controlled_checks = dict(source_unchanged=before == after,
            observer_available=observer is not None, physical_capture_available=capture is not None,
            callback_complete=args.mode == 'A' or state.get('callback_complete', False),
            no_contact_overflow=observer is not None and not any(sample['contact_buffer_full'] for sample in observer.samples),
            owner_thread=observer is not None and all(sample['owner_thread'] for sample in observer.samples))
        if observer is not None:
            _json(output/'controlled_trace.json', dict(metadata=observer.metadata,
                samples=observer.samples, actions=observer.actions, operations=observer.operations))
            trace_sha = hashlib.sha256((output/'controlled_trace.json').read_bytes()).hexdigest()
        else:
            trace_sha = None
        payload['controlled'] = dict(mode=args.mode, strategy=options.strategy,
            checks=controlled_checks, operations=operations,
            acceptance_passed=controlled_acceptance(args.mode, operations),
            source_hashes_before=before, source_hashes_after=after,
            raw_trace=dict(path=str(output/'controlled_trace.json'), sha256=trace_sha),
            followup_precondition_failed=state.get('followup_precondition_failed', False),
            meaning='Observation validity and successful execution are separate. No retiming, IK preference, drive/gain/limit edits or production patch is enabled.')
        payload['status'] = ('PASS' if payload['status'] == 'PASS' and all(controlled_checks.values()) else 'FAIL')
        _json(args.result, payload)
        print(json.dumps(dict(status=payload['status'], mode=args.mode,
            acceptance_passed=payload['controlled']['acceptance_passed'],
            operations=[dict(label=row['label'], returned_success=row['returned_success'],
                cancelled=row['cancelled']) for row in operations])), flush=True)
        return payload['status'] == 'PASS'
    finally:
        if state.get('observer') is not None:
            state['observer'].restore()
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
