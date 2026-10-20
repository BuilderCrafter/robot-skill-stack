"""Render standalone figures from retained research JSON and PLY artifacts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--detection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((args.output/'matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads((args.geometry/'rows.json').read_text())
    methods = ('v1_A', 'v1_B', 'surface_A', 'surface_B', 'surface_fused')
    colors = ('#64748b', '#94a3b8', '#db6d28', '#43a48b', '#286bb0')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(methods))
    accuracy = [100*np.mean([r['correct'] for r in rows if r['method'] == m]) for m in methods]
    center = [np.mean([r['center_error_mm'] for r in rows if r['method'] == m
                      and r['center_error_mm'] is not None]) for m in methods]
    axes[0].bar(x, accuracy, color=colors)
    axes[0].set_ylabel('Correct classifications (%)')
    axes[0].set_ylim(0, 105)
    axes[0].set_title('Oracle-segmented native surfaces · 48 trials/method')
    axes[1].bar(x, center, color=colors)
    axes[1].set_ylabel('Mean center error (mm)')
    axes[1].set_title('All trials, including unknown classifications')
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(methods, rotation=25, ha='right')
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.output/'geometry_comparison.png', dpi=160)
    plt.close(fig)

    failed = [r for r in rows if r['method'] == 'surface_A' and not r['correct']]
    trial = (failed[0] if failed else max(rows, key=lambda r: r.get('center_error_mm') or 0))['trial']
    fig = plt.figure(figsize=(13.5, 4.8))
    fig.subplots_adjust(left=.01, right=.94, bottom=.10, top=.80, wspace=.04)
    for index, method in enumerate(('surface_A', 'surface_B', 'surface_fused')):
        row = next(r for r in rows if r['trial'] == trial and r['method'] == method)
        points = np.loadtxt(args.geometry/'clouds'/f'{trial}_{method}.ply', skiprows=7).reshape(-1, 3)
        points = points[np.linspace(0, len(points)-1, min(len(points), 1600), dtype=int)]
        axis = fig.add_subplot(1, 3, index+1, projection='3d')
        axis.scatter(*points.T, s=2, c=points[:, 2], cmap='viridis')
        axis.scatter(*np.asarray(row['truth_center'])[:, None], marker='x', s=60, color='red')
        axis.set_xlabel('X (m)')
        axis.set_ylabel('Y (m)')
        axis.set_zlabel('Z (m)', labelpad=2)
        dims = row.get('estimated_dimensions')
        dimensions = 'unknown' if dims is None else ', '.join(f'{v*1000:.1f}' for v in dims)
        axis.set_title(f'{method}: {row["predicted"]}\n'
                       f'dimensions [{dimensions}] mm; center error {row["center_error_mm"]:.2f} mm', fontsize=9)
        axis.view_init(elev=22, azim=130)
        axis.set_box_aspect((1, 1, 1))
    truth = next(r for r in rows if r['trial'] == trial)
    dimensions = ', '.join(f'{v*1000:.0f}' for v in truth['truth_dimensions'])
    fig.suptitle(f'Complementary partial surfaces · truth {truth["truth"]}, dimensions [{dimensions}] mm')
    fig.savefig(args.output/'partial_and_fused_clouds.png', dpi=160)
    plt.close(fig)

    counts = json.loads((args.detection/'detections.json').read_text())
    methods = ('v1_A', 'v1_B', 'multi_view')
    scenarios = ('clear', 'occluded_A', 'hidden_A', 'hidden_B', 'nearby', 'elevated_clutter')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for index, method in enumerate(methods):
        rate = []
        for scenario in scenarios:
            selected = [r for r in counts if r['method'] == method and f'_{scenario}_' in r['trial']]
            rate.append(100*sum(r['matched'] for r in selected)/sum(r['expected'] for r in selected))
        axes[0].plot(scenarios, rate, marker='o', label=method)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Matched physical objects (%)')
    axes[0].set_title('Automatic detection · all seeded repetitions')
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].legend()
    timings = json.loads((args.detection/'timings.json').read_text())
    phases = ('acquisition_ms', 'projection_ms', 'fusion_ms', 'segmentation_ms', 'fitting_ms', 'render_ms')
    means = [np.mean([r[p] for r in timings]) for p in phases]
    axes[1].barh([p.replace('_ms', '') for p in phases], means, color='#286bb0')
    axes[1].set_xlabel('Mean time (ms)')
    axes[1].set_title('Multi-view pipeline · render measured separately')
    fig.savefig(args.output/'automatic_detection_and_latency.png', dpi=160)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
