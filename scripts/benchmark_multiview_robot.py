"""Actual Franka perception research; documented motion requires supervision."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from tests.support.multiview_robot_native import run_robot_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--supervised-motion', action='store_true',
                        help='Operator is supervising the documented hover/side/return sequence')
    parser.add_argument('--occlusion-fixtures', action='store_true',
                        help='Supported tall targets in the actual arm viewing shadows')
    parser.add_argument('--workspace-max-z', type=float,
                        help='Explicit research-only workspace-height ablation, in meters')
    parser.add_argument('--temporal-viewpoints', action='store_true',
                        help='Separate overlapping-shadow camera arrangement for occlusion/recovery')
    args = parser.parse_args()
    try:
        run_robot_evaluation(simulation_app, args.result, args.artifacts_dir,
                             supervised_motion=args.supervised_motion,
                             occlusion_fixtures=args.occlusion_fixtures,
                             workspace_max_z=args.workspace_max_z,
                             temporal_viewpoints=args.temporal_viewpoints)
    finally:
        simulation_app.close()
