"""Measure bounded numerical work on fragmented clutter outside Isaac.

This is an adversarial segmentation/fitting CPU microbenchmark, not native
observation throughput. Run without other simulations or numerical workloads.
No truth, gripper constants or historical geometry enters either algorithm.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from robot_skill_stack.world.perception_multiview import fitting, segmentation
from robot_skill_stack.world.perception_multiview.provider import _FitTimer
from tests.support.multiview_iteration2 import distribution


def fragmented_cloud(planes):
    x, z = np.meshgrid(np.linspace(-.007, .007, 7), np.linspace(.025, .043, 6))
    return np.concatenate([np.c_[x.ravel()+.208+.032*parent,
        np.full(x.size, .007*fragment), z.ravel()] for parent in range(16) for fragment in range(planes)])


def measure(points, grouping, fitter, cap):
    started = time.perf_counter()
    clusters = grouping.cluster_points(points)
    timer = _FitTimer(fitter, cap)
    outputs = []
    for cloud in clusters:
        for child, cached in grouping.refine_cluster(cloud, timer):
            supported, reason = grouping.supported_object_cloud(child, 0.)
            if not supported and reason != 'no_observed_support_contact':
                continue
            fit = cached or timer.fit(child)
            if not supported:
                bottom = fit.center[2]-fit.size[2]/2
                if fit.geometry.shape.value not in ('sphere', 'cylinder') or abs(bottom) > .004:
                    continue
            outputs.append(fit.geometry.shape.value)
    return dict(elapsed_ms=1000*(time.perf_counter()-started), fitting_ms=1000*timer.seconds,
                numerical_fits=timer.calls, cap=timer.max_calls,
                cheap_unknown_fallbacks=timer.exhaustions, points=len(points), broad_groups=len(clusters),
                accepted_shapes=outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-pythonpath', type=Path,
                        default=Path('outputs/multiview-readiness/baseline_snapshot'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    old_group = importlib.import_module('baseline_multiview.segmentation')
    old_fit = importlib.import_module('baseline_multiview.fitting')
    rows = []
    for count in (8, 16):
        points = fragmented_cloud(count)
        for method, group, fitter, cap in (('baseline', old_group, old_fit.SurfaceFitter(), 10000),
                                           ('improved', segmentation, fitting.SurfaceFitter(), 24)):
            for repeat in range(args.repeats):
                row = dict(method=method, fine_planes_per_parent=count, repeat=repeat,
                           **measure(points, group, fitter, cap))
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {f'{method}/{planes}': {metric: distribution([r[metric] for r in rows
        if r['method'] == method and r['fine_planes_per_parent'] == planes])
        for metric in ('elapsed_ms', 'fitting_ms', 'numerical_fits')}
        for method in ('baseline', 'improved') for planes in (8, 16)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path('robot_skill_stack/world/perception_multiview').glob('*.py'))}
    args.output.write_text(json.dumps(dict(kind='offline_fragmented_clutter_cpu_profile',
        rows=rows, summary=summary, sha256=hashes,
        note='Deterministic repeated geometry measures latency variation, not independent statistical evidence.'), indent=2)+'\n')
