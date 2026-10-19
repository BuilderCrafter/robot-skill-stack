"""Temporary workstation telemetry shared by the empty and loaded diagnostics."""
from contextlib import ExitStack, contextmanager
import json
import time
from unittest.mock import patch

import numpy as np


class TraceJournal:
    """Keep complete samples on disk without depending on Python shutdown."""
    def __init__(self, samples, stream):
        self.samples, self.stream = samples, stream

    def _write(self, row):
        self.stream.write(json.dumps(row)+'\n')
        self.stream.flush()

    def append(self, row):
        self._write(row)
        self.samples.append(row)

    def mark(self, stage):
        self._write({'event': 'stage', 'stage': stage, 'monotonic_s': time.monotonic()})


def action_snapshot(action):
    if action is None:
        return None
    return {name: None if getattr(action, name) is None else np.asarray(getattr(action, name)).tolist()
            for name in ('joint_positions', 'joint_velocities', 'joint_indices')}


@contextmanager
def record_control(backend, scheduler, samples, cube=None):
    """Observe existing actions/waits and a transient physics callback; no control changes."""
    stage = 'setup'
    callback_errors = []

    def set_stage(name):
        nonlocal stage
        stage = name

    def record(event, action=None, physics_dt=None, **details):
        tcp = backend.get_end_effector_pose()
        row = {
            'stage': stage, 'event': event, 'monotonic_s': time.monotonic(),
            'physics_step': int(backend.world.current_time_step_index),
            'simulation_time_s': float(backend.world.current_time), 'physics_dt_s': physics_dt,
            'joint_positions': np.asarray(backend.robot.get_joint_positions()).tolist(),
            'joint_velocities': np.asarray(backend.robot.get_joint_velocities()).tolist(),
            'tcp_position': tcp.position.tolist(),
            'tcp_orientation': None if tcp.orientation is None else tcp.orientation.tolist(),
            'applied_action': action_snapshot(backend.robot.get_applied_action()),
            'action': action_snapshot(action),
        }
        if cube is not None:
            position, orientation = cube.get_world_pose()
            row.update(cube_position=np.asarray(position).tolist(),
                       cube_orientation=np.asarray(orientation).tolist())
        row.update(details)
        samples.append(row)

    def on_physics(dt):
        # SDK callback exceptions need to surface through the coroutine as well.
        if not callback_errors:
            try:
                record('physics_callback', physics_dt=float(dt))
            except Exception as exc:
                callback_errors.append(exc)

    original_forward = backend.motion_controller.forward
    original_apply = backend.robot.apply_action
    original_wait = backend._next_update

    def forward(*args, **kwargs):
        record('before_forward')
        action = original_forward(*args, **kwargs)
        position = kwargs.get('target_end_effector_position', args[0] if args else None)
        orientation = kwargs.get('target_end_effector_orientation', args[1] if len(args) > 1 else None)
        feedback = backend.motion_controller.get_articulation_motion_policy().joint_velocity_feedback
        record('after_forward', action, target_position=np.asarray(position).tolist(),
               target_orientation=None if orientation is None else np.asarray(orientation).tolist(),
               controller_joint_velocities=None if feedback is None else np.asarray(feedback).tolist())
        return action

    def apply(action, *args, **kwargs):
        record('before_apply', action)
        return original_apply(action, *args, **kwargs)

    async def wait():
        await original_wait()
        if callback_errors:
            raise callback_errors[0]
        record('after_update')

    name = 'control_diagnostic'
    scheduler.add_physics_callback(name, callback_fn=on_physics)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(backend.motion_controller, 'forward', new=forward))
            stack.enter_context(patch.object(backend.robot, 'apply_action', new=apply))
            stack.enter_context(patch.object(backend, '_next_update', new=wait))
            yield set_stage
        if callback_errors:
            raise callback_errors[0]
    finally:
        scheduler.remove_physics_callback(name)


