#!/usr/bin/env python3
"""Production B/C validation with a bounded read-only post-Home observation.

The frozen controlled runner supplies all motion operations. No solver probe is
installed: the production strategy owns its settings and terminal settling.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

from scripts import diagnose_global_stabilization_v4_1 as controlled
from scripts import diagnose_global_solver_velocity_v4_1 as observations


def callback_for(mode, state, original_factory, post_hold_seconds=1.):
    original = original_factory(mode, state)

    async def callback(*args):
        result = await original(*args)
        await observations.observe_post_home_hold(state['observer'], post_hold_seconds)
        return result

    return callback


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=('B', 'C'), required=True)
    parser.add_argument('--post-hold-seconds', type=float, choices=(1., 5.), default=1.)
    known, _ = parser.parse_known_args(arguments)
    for index, argument in enumerate(arguments):
        if argument == '--post-hold-seconds':
            del arguments[index:index+2]
            break
        if argument.startswith('--post-hold-seconds='):
            del arguments[index]
            break
    original_hashes = controlled.source_hashes
    original_reduce = controlled.reduce_observations
    original_callback_for = controlled.callback_for
    base = controlled.Observer

    class ProductionObserver(base):
        def sample(self):
            super().sample()
            self.samples[-1]['native_actor_solver_properties'] = observations.native_properties(self.robot)

    def hashes():
        rows = original_hashes()
        for path in (Path(__file__), Path(observations.__file__)):
            rows[str(path.relative_to(controlled.fixture.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return rows

    def reduce(observer, capture):
        rows = original_reduce(observer, capture)
        for compact, full in zip(rows, observer.operations):
            if 'post_home_hold' in full:
                report = full['post_home_hold']
                report['validation_passed'] = (report['kinematic_stability_passed']
                    and compact['contacts']['reported_points'] == 0
                    and compact['clearance']['minimum_m'] is not None and compact['clearance']['minimum_m'] > 0.)
                compact['post_home_hold'] = report
                compact['strict_success'] = None
        return rows

    with patch.object(controlled, 'Observer', ProductionObserver), \
            patch.object(controlled, 'source_hashes', hashes), patch.object(controlled, 'reduce_observations', reduce), \
            patch.object(controlled, 'callback_for', lambda mode, state: callback_for(
                mode, state, original_callback_for, known.post_hold_seconds)):
        return controlled.main(arguments)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
