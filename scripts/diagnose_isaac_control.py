"""Workstation-only, empty-gripper control trace; not a regression pass/fail gate."""
# Preserve the repository's standalone NumPy/Isaac startup contract.
import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Diagnostic NumPy: {np.__version__} from {np.__file__}', flush=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope
from tests.support.isaac_async import run_kit_coroutine
from tests.support.isaac_helpers import build_ground_truth_runtime, open_playground
from tests.support.results import write_result


def action_snapshot(action):
    if action is None:
        return None
    return {name: None if getattr(action, name) is None else np.asarray(getattr(action, name)).tolist()
            for name in ('joint_positions', 'joint_velocities', 'joint_indices')}


@contextmanager
def record_control(backend, scheduler, samples):
    """Observe existing actions/waits and a transient physics callback; no control changes."""
    stage = 'setup'
    callback_errors = []

    def set_stage(name):
        nonlocal stage
        stage = name

    def record(event, action=None, physics_dt=None):
        tcp = backend.get_end_effector_pose()
        samples.append({
            'stage': stage, 'event': event, 'monotonic_s': time.monotonic(),
            'physics_step': int(backend.world.current_time_step_index),
            'simulation_time_s': float(backend.world.current_time), 'physics_dt_s': physics_dt,
            'joint_positions': np.asarray(backend.robot.get_joint_positions()).tolist(),
            'joint_velocities': np.asarray(backend.robot.get_joint_velocities()).tolist(),
            'tcp_position': tcp.position.tolist(),
            'tcp_orientation': None if tcp.orientation is None else tcp.orientation.tolist(),
            'applied_action': action_snapshot(backend.robot.get_applied_action()),
            'action': action_snapshot(action),
        })

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
        record('after_forward', action)
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


async def run_diagnostic(backend, scheduler, metrics):
    hover = Pose([.45, 0, .25], [0, 0, 1, 0])
    side = hover.translated([0, .08, 0])
    with record_control(backend, scheduler, metrics['samples']) as set_stage:
        for name, target in (('hover', hover), ('open_empty', None), ('close_empty', None),
                             ('side', side), ('idle_after_side', None), ('back', hover),
                             ('idle_after_back', None)):
            set_stage(name)
            print('Diagnostic stage:', name, flush=True)
            row = {'stage': name}
            metrics['stages'].append(row)
            if name.startswith('idle_'):
                # Deliberately observe retained targets without issuing another action.
                for _ in range(60):
                    await backend._next_update()
                row['updates'] = 60
                continue
            if target is not None:
                row['target_position'] = target.position.tolist()
                row['target_orientation'] = target.orientation.tolist()
                result = await backend.move_to_pose_async(target, position_tolerance=.02,
                                                          orientation_tolerance=.10)
            elif name == 'open_empty':
                result = await backend.open_gripper_async()
            else:
                result = await backend.close_gripper_async()
            row['result'] = {'ok': result.ok, 'message': result.message,
                             'timed_out': result.timed_out, 'details': dict(result.details)}
            if not result.ok:
                raise RuntimeError(f'{name}: {result.message}')


def main(result_path, artifacts_dir):
    metrics = {'samples': [], 'stages': []}
    try:
        open_playground(simulation_app)
        bundle = build_ground_truth_runtime(simulation_app, Path(artifacts_dir))
        policy = bundle.backend.motion_controller.get_articulation_motion_policy()
        metrics['timesteps_s'] = {
            'world_physics': float(bundle.world.get_physics_dt()),
            'world_rendering': float(bundle.world.get_rendering_dt()),
            'controller': float(policy.get_default_physics_dt()),
        }
        run_kit_coroutine(run_diagnostic(bundle.backend, PhysicsCallbackScope(bundle.world), metrics),
                          simulation_app)
    except BaseException as exc:
        write_result(result_path, status='FAIL', metrics=metrics, error=f'{type(exc).__name__}: {exc}')
        raise
    write_result(result_path, status='RECORDED', metrics=metrics)
    print('Trace recorded; this does not certify stable or collision-free motion.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir)
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        simulation_app.close()
