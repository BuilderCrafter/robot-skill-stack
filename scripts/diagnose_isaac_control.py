"""Workstation-only, empty-gripper control trace; not a regression pass/fail gate."""
# Preserve the repository's standalone NumPy/Isaac startup contract.
import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Diagnostic NumPy: {np.__version__} from {np.__file__}', flush=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
from pathlib import Path
import traceback

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope
from tests.support.isaac_async import run_kit_coroutine
from tests.support.isaac_helpers import build_ground_truth_runtime, open_playground
from tests.support.results import write_result
from tests.support.control_trace import record_control


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
