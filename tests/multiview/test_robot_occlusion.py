"""Native real-Franka source contract; documented motion is explicitly opt-in.

The default regression run uses a stationary articulation and supported targets.
The supervised command additionally measures the documented hover/side/return
sequence. Hidden-object losses remain in the report rather than being published
as current geometry. This gate checks source/identity and false-object safety.
"""
import argparse
from pathlib import Path

import numpy as np

if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4 before SimulationApp, got {np.__version__}')

from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

from tests.support.multiview_robot_native import run_robot_evaluation
from tests.support.results import fail_result, write_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir')
    parser.add_argument('--supervised-motion', action='store_true')
    args = parser.parse_args()
    artifacts = args.artifacts_dir or str(Path(args.result).parent/'real_franka')
    try:
        metrics = run_robot_evaluation(simulation_app, args.result, artifacts,
                                      supervised_motion=args.supervised_motion,
                                      occlusion_fixtures=True)
        safe = not any(metrics[key] for key in ('source_failures', 'false_positives',
                                               'identity_switches', 'false_splits', 'duplicate_ids'))
        safe = safe and metrics['final_health'] == 'fresh' and metrics['samples'] >= 18
        metrics['gate_scope'] = 'Source freshness, stable identity, no false objects; visibility and geometry accuracy remain separately reported'
        write_result(args.result, status='PASS' if safe else 'FAIL', metrics=metrics)
    except Exception as error:
        fail_result(args.result, error)
    finally:
        simulation_app.close()
