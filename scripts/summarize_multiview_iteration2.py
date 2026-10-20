"""Export before/after CSV metrics and standalone research figures."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def distribution(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)),
                worst=float(values.max()), samples=len(values))


def read_ply(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() == 'end_header':
                break
        else:
            raise ValueError(f'Missing PLY header: {path}')
        return np.loadtxt(stream).reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--improved', type=Path, required=True)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--robustness', type=Path, action='append', default=[],
                        help='Matched replay directory; may be repeated for selection and holdout')
    parser.add_argument('--live-robustness', type=Path,
                        help='Separate native capture/timing directory with native_timings.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons, rows, timing_rows = {}, [], []
    for name, root in (('baseline', args.baseline), ('improved', args.improved)):
        result = json.loads((root/'result.json').read_text())
        if result['status'] != 'PASS':
            raise ValueError(f'Incomplete experiment: {root}')
        timings = json.loads((root/'timings.json').read_text())
        phases = [key for key, value in timings[0].items() if key.endswith('_ms') and isinstance(value, (int, float))]
        comparisons[name] = dict(summary=result['metrics']['summary'],
            latency_ms={phase: distribution([row[phase] for row in timings]) for phase in phases},
            effective_update_hz=1000/np.mean([row['elapsed_ms'] for row in timings]))
        rows.extend(dict(iteration=name, **row) for row in json.loads((root/'rows.json').read_text()))
        timing_rows.extend(dict(iteration=name, **row) for row in timings)
    (args.output/'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    if args.live_robustness:
        live = json.loads((args.live_robustness/'native_timings.json').read_text())
        live_summary = {}
        for method in sorted({row['method'] for row in live}):
            selected = [row for row in live if row['method'] == method]
            phases = [key for key, value in selected[0].items()
                      if key.endswith('_ms') and isinstance(value, (float, int))]
            live_summary[method] = dict(
                latency_ms={phase: distribution([row[phase] for row in selected]) for phase in phases},
                effective_update_hz=1000/np.mean([row['elapsed_ms'] for row in selected]),
                per_trial={trial: {phase: distribution([row[phase] for row in selected if row['trial'] == trial])
                                   for phase in phases} for trial in sorted({row['trial'] for row in selected})})
        (args.output/'live_holdout_comparison.json').write_text(json.dumps(live_summary, indent=2)+'\n')
        with (args.output/'live_holdout_timings.csv').open('w') as stream:
            fields = list(dict.fromkeys(key for row in live for key in row))
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in live:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
    for filename, data in (('original_rows.csv', rows), ('original_timings.csv', timing_rows)):
        fields = list(dict.fromkeys(key for row in data for key in row))
        with (args.output/filename).open('w') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in data:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('baseline', 'improved')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    colors = ['#c27925', '#2878a8']
    accuracy = [comparisons[name]['summary']['multi_view']['correct'] for name in names]
    axes[0].bar(names, accuracy, color=colors)
    axes[0].set(ylim=(0, 92), ylabel='Correct / 88 expected', title='Original 24 native scenes')
    for index, value in enumerate(accuracy):
        axes[0].text(index, value+1, str(value), ha='center')
    phases = ('acquisition_ms', 'projection_ms', 'fusion_ms', 'segmentation_ms', 'fitting_ms', 'tracking_ms')
    bottom = np.zeros(2)
    for phase in phases:
        values = [comparisons[name]['latency_ms'][phase]['mean'] for name in names]
        axes[1].bar(names, values, bottom=bottom, label=phase.removesuffix('_ms'))
        bottom += values
    axes[1].set(ylabel='Mean processing (ms)', title='Real acquisition; render excluded')
    axes[1].legend(fontsize=7)
    hz = [comparisons[name]['effective_update_hz'] for name in names]
    axes[2].bar(names, hz, color=colors)
    axes[2].axhline(5, color='gray', linestyle='--', label='5 Hz optimization goal')
    axes[2].set(ylabel='Measured loop frequency (Hz)', title='Eight real render steps + observe')
    axes[2].legend(fontsize=7)
    fig.savefig(args.output/'original_accuracy_latency.png', dpi=180)
    plt.close(fig)

    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    grouping = importlib.import_module('baseline_multiview.segmentation')
    points = read_ply(args.baseline/'full_clouds/0_hidden_B_0_fused.ply')
    fig = plt.figure(figsize=(11, 5), layout='constrained')
    for index, title in enumerate(('Original: cylinder connected to rejected occluder',
                                    'Improved: accepted object clouds')):
        axis = fig.add_subplot(1, 2, index+1, projection='3d')
        axis.scatter(*points.T*1000, color='lightgray', s=1, alpha=.35)
        if index == 0:
            selected = [cloud for cloud in grouping.cluster_points(points)
                        if not grouping.supported_object_cloud(cloud, 0.)[0]]
        else:
            selected = [read_ply(path) for path in sorted((args.improved/'clouds').glob('0_hidden_B_0_*.ply'))]
        for cloud in selected:
            axis.scatter(*cloud.T*1000, s=2, alpha=.8)
        axis.set(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', title=title,
                 xlim=(240, 590), ylim=(-240, 240), zlim=(0, 220))
        axis.view_init(elev=26, azim=-58)
    fig.savefig(args.output/'hidden_B_before_after.png', dpi=180)
    plt.close(fig)
    if args.robustness:
        fig, axes = plt.subplots(len(args.robustness), 2, figsize=(12, 4*len(args.robustness)),
                                 squeeze=False, layout='constrained')
        methods = ('v1_A', 'v1_B', 'baseline', 'improved')
        labels = ('V1 A', 'V1 B', 'Original\nmulti-view', 'Improved\nmulti-view')
        for index, directory in enumerate(args.robustness):
            result = json.loads((directory/'result.json').read_text())
            if result['status'] != 'PASS':
                raise ValueError(f'Incomplete replay: {directory}')
            partitions = result['metrics']['per_split_variant']
            key = next(key for key in partitions if key.endswith('/clean'))
            split = key.split('/')[0]
            summary = partitions[key]
            correct = [summary[method]['correct'] for method in methods]
            axes[index, 0].bar(labels, correct, color=['gray', 'silver', *colors])
            expected = summary['improved']['expected']
            axes[index, 0].set(ylabel=f'Correct / {expected} expected', ylim=(0, expected+8),
                               title=f'{split.capitalize()}: identical clean native snapshots')
            for column, value in enumerate(correct):
                axes[index, 0].text(column, value+1, str(value), ha='center')
            mixed = [summary[method]['mixed_accepted_clouds'] for method in methods]
            axes[index, 1].bar(labels, mixed, color=['gray', 'silver', *colors])
            axes[index, 1].set(ylabel='Mixed published clouds',
                               title='Includes explicit unknown observations; no failures removed')
        fig.savefig(args.output/'selection_holdout_accuracy.png', dpi=180)
        plt.close(fig)
    print(json.dumps(dict(output=str(args.output), comparison=comparisons)))


if __name__ == '__main__':
    main()
