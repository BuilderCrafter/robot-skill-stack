#!/usr/bin/env python3
"""Compare stock/enhanced global models using unchanged historical V4 scenes.

Only an ignored generated profile selects the model. Original fixtures, targets,
seeds, iteration budgets and physical contact/clearance definitions are reused.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import benchmark_global_motion_v4 as fixture
from scripts.benchmark_global_roundtrip_v4_2 import rss_bytes


ROOT = fixture.ROOT


def model_profile(args, original, selected):
    path = original(args)
    text = path.read_text()
    if text.count('[motion]\n') != 1:
        raise ValueError('Expected one generated motion configuration')
    path.write_text(text.replace('[motion]\n',
        f'[motion]\nglobal_collision_model = "{selected}"\n'))
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    if load_scene_config(path).motion.global_collision_model != selected:
        raise ValueError('Generated profile did not select the requested model')
    return path


def project_sources(original):
    hashes = original()
    paths = [Path(__file__), ROOT/'scripts/benchmark_global_roundtrip_v4_2.py']
    paths.extend((ROOT/'config/motion').glob('franka_enhanced_v4_2.*'))
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def obstacle_closest_links(capture):
    """Offline exact same cooked-solid metric, separately per frozen obstacle."""
    snapshot = capture['snapshot']
    transforms = [capture['pose_matrices'](row['poses'], snapshot.meters_per_unit)
                  for row in capture['raw']['samples']]
    return {obstacle.path: fixture.reduce_clearances(capture['meshes'], capture['body_paths'],
                transforms, (obstacle,))['closest_pair'] for obstacle in snapshot.obstacles}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collision-model', choices=('stock', 'enhanced'), required=True)
    parser.add_argument('--scenario', choices=fixture.SCENARIOS, required=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--reference-result')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args(argv)
    args.strategy, args.lifecycle, args.witness, args.enable_mapping = 'global', False, False, False
    from robot_skill_stack.integrations.isaac.config import MotionConfig
    MotionConfig(strategy='global', rrt_seed=args.seed, rrt_max_iterations=args.max_iterations,
        global_collision_model=args.collision_model)
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
    original_profile, original_hashes = fixture.write_profile, fixture.source_hashes
    original_native, original_finish = fixture.run_native, fixture.finish

    def native(options, app, result):
        result['memory_before_native'] = rss_bytes()
        capture = original_native(options, app, result)
        result['memory_after_runtime_close'] = rss_bytes()
        actual = result.get('strategy_diagnostics', {}).get('global_collision_model')
        result['configuration']['global_collision_model'] = args.collision_model
        result['actual_model_selection_verified'] = actual == args.collision_model
        if not result['actual_model_selection_verified']:
            raise RuntimeError('Requested collision model was not the actual selected global strategy')
        return capture

    def finish(options, result, capture):
        original_finish(options, result, capture)
        result['closest_robot_link_by_obstacle'] = obstacle_closest_links(capture)
        result['memory_after_offline_reduction'] = rss_bytes()

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    try:
        with patch.object(fixture, 'write_profile', lambda options: model_profile(options, original_profile, args.collision_model)), \
                patch.object(fixture, 'source_hashes', lambda: project_sources(original_hashes)), \
                patch.object(fixture, 'run_native', native), patch.object(fixture, 'finish', finish):
            return fixture.execute_case(args, app)
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
