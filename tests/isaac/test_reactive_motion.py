"""Reactive free-space motion, source barriers, cancellation and SDK cleanup."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__}')
print(f'Test NumPy: {np.__version__}',flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless':False})

import argparse
import json
from pathlib import Path

from scripts.diagnose_reactive_motion import execute_case


def main(args):
    args.strategy,args.source,args.scenario,args.lifecycle = 'reactive','ground_truth','empty',True
    measured = execute_case(args,simulation_app)
    path = Path(args.result)
    payload = json.loads(path.read_text())
    metrics = payload['metrics']
    checks = dict(valid_measurement=measured,
        ordinary_move_reached_without_contact=metrics.get('avoidance_success') is True,
        ordinary_home_succeeded=metrics.get('home',{}).get('status') == 'success',
        lifecycle_complete=metrics.get('lifecycle',{}).get('checks',{}).get('all_lifecycle_steps_completed') is True,
        lifecycle_all_checks_pass=bool(metrics.get('lifecycle',{}).get('checks')) and
            all(metrics['lifecycle']['checks'].values()))
    payload['native_acceptance'] = checks
    payload['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=payload['status'],checks=checks)),flush=True)
    return payload['status'] == 'PASS'


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True)
    parser.add_argument('--artifacts-dir',default='outputs/motion-planning-v2-native')
    args = parser.parse_args()
    try:
        passed = main(args)
    finally:
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
