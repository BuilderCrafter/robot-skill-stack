"""Workstation regression: remove a carried cube and verify position-based loss detection."""
import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': False})

import argparse
from pathlib import Path
import traceback
from unittest.mock import AsyncMock, patch

from robot_skill_stack.common.types import GripperState
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope
from robot_skill_stack.runtime.skill import FailureCode
from tests.support.control_trace import TraceJournal, record_control
from tests.support.isaac_async import run_kit_coroutine
from tests.support.isaac_helpers import build_ground_truth_runtime, choose_object, open_playground, settle
from tests.support.results import write_result


def run_check(bundle, metrics, journal):
    model, backend, cube = bundle.world_model, bundle.backend, bundle.objects['cube']
    object_id = choose_object(model).object_id
    initial_z = float(cube.get_world_pose()[0][2])
    metrics['injected'] = False

    def remove_cube(dt):
        try:
            if (not metrics['injected'] and model.gripper_state == GripperState.HELD and
                    cube.get_world_pose()[0][2] > initial_z+.04):
                # Deliberate test-only object removal; no robot/gripper command.
                _, orientation = cube.get_world_pose()
                cube.set_world_pose(position=np.array([.35, -.25, .025]), orientation=orientation)
                metrics['injected'] = True
                journal.mark('cube_removed_during_lift')
        except Exception as exc:
            metrics['injection_error'] = str(exc)

    scheduler = PhysicsCallbackScope(bundle.world)
    scheduler.add_physics_callback('held_state_remove_cube', callback_fn=remove_cube)
    try:
        with record_control(backend, scheduler, journal, cube=cube) as stage:
            stage('pick_with_object_removal')
            with patch.object(backend, 'open_gripper_async', new=AsyncMock(wraps=backend.open_gripper_async)) as opened, \
                 patch.object(backend, 'close_gripper_async', new=AsyncMock(wraps=backend.close_gripper_async)) as closed, \
                 patch.object(backend, 'home_async', new=AsyncMock(wraps=backend.home_async)) as homed:
                result = run_kit_coroutine(bundle.runtime.execute_async('pick', object_id=object_id), simulation_app)
                metrics['commands'] = {'open': opened.await_count, 'close': closed.await_count,
                                       'home': homed.await_count}
        metrics.update(status=result.status.value,
                       failure_code=None if result.failure_code is None else result.failure_code.value,
                       message=result.message, details=result.details,
                       gripper_state=model.gripper_state.value, held_object_id=model.held_object_id,
                       attachment_cleared=model.held_object_offset_in_ee is None,
                       gripper_feedback=backend.gripper_feedback().details)
        assert 'injection_error' not in metrics, metrics.get('injection_error')
        assert metrics['injected'], 'Cube removal never triggered; this is not a loss-detection pass'
        assert result.failure_code == FailureCode.GRASP_LOST, metrics
        assert model.gripper_state == GripperState.FREE, metrics
        assert model.held_object_id is None and metrics['attachment_cleared'], metrics
        assert metrics['commands'] == {'open': 1, 'close': 1, 'home': 0}, metrics
    finally:
        scheduler.remove_physics_callback('held_state_remove_cube')


def main(result_path, artifacts_dir):
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'control_trace_file': str(out/'control_trace.jsonl')}
    try:
        with (out/'control_trace.jsonl').open('w', encoding='utf-8') as stream:
            journal = TraceJournal(stream)
            journal.mark('startup')
            open_playground(simulation_app)
            bundle = build_ground_truth_runtime(simulation_app, out)
            settle(bundle.world, 90)
            run_check(bundle, metrics, journal)
        write_result(result_path, status='PASS', metrics=metrics)
        print('PASS: object loss detected; state free; no recovery/release command.', flush=True)
    except BaseException as exc:
        write_result(result_path, status='FAIL', metrics=metrics, error=f'{type(exc).__name__}: {exc}')
        raise


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
