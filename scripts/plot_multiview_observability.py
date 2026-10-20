"""Scoring-only illustration of measured versus hidden physical box extents."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from robot_skill_stack.world.perception_multiview.clouds import project_depth
from scripts.benchmark_multiview_robustness import load_frames
from tests.support.multiview_iteration2 import SnapshotRig, object_from_payload, surface_memberships


def outline(ax, center, size, *, color, label=None):
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])*size[:2]/2+center[:2]
    ax.plot(*xy.T, color=color, label=label, linewidth=1.6)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from robot_skill_stack.integrations.isaac.config import load_scene_config

    manifest = json.loads((args.capture/'capture_manifest.json').read_text())
    scene = next(row for row in manifest['scenes'] if row['key'] == 'selection_partial_box_both')
    target = object_from_payload(scene['frames'][-1]['objects'][0])
    record = next(row for row in json.loads((args.capture/'rows.json').read_text())
                  if row['scene'] == scene['key'])
    estimate = record['geometry_evidence']
    center, size = np.asarray(estimate['estimated_center']), np.asarray(estimate['box_size'])
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    provider_class = importlib.import_module('baseline_multiview.provider').MultiViewProvider
    rig = SnapshotRig()
    source = provider_class(rig, load_scene_config('config/scenes/multiview_playground.toml').scene)
    try:
        for frame in scene['frames']:
            frames = load_frames(args.capture, frame['files'])
            rig.push(frames)
            observations = source.observe().observations
        obs = min(observations, key=lambda row: np.linalg.norm(row.pose.position-center))
        cloud = source.get_point_cloud(obs.object_id)
        np.testing.assert_allclose(obs.pose.position, center, atol=1e-8)
        np.testing.assert_allclose(obs.geometry.box_size, size, atol=1e-8)
    finally:
        source.close()
    views = []
    for frame in frames:
        points = project_depth(frame, stride=1)[0]
        views.append(points[surface_memberships(points, (target,)) == 0])
    origin = target.center.copy()
    origin[2] = 0.
    local_views = [(points-origin)*1000 for points in views]
    local_cloud = (cloud-origin)*1000
    true_center, true_size = (target.center-origin)*1000, target.size*1000
    estimated_center, estimated_size = (center-origin)*1000, size*1000
    boundaries = {}
    for vi, points in enumerate(views):
        delta = points-target.center
        boundaries[str(vi)] = {f'{axis}{sign:+}': int(np.count_nonzero(
            abs(delta[:, ai]-sign*target.size[ai]/2) < .0008))
            for ai, axis in enumerate(('X', 'Y', 'Z')) for sign in (-1, 1)}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax in axes.ravel():
        ax.set(xlabel='object-local X (mm)', ylabel='object-local Y (mm)', aspect='equal',
               xlim=(-58, 58), ylim=(-58, 58))
        ax.grid(alpha=.18)
        outline(ax, true_center, true_size, color='#1f2937')
    axes[0, 0].set_title('True 90 × 90 × 50 mm object\nEvaluation geometry only')
    axes[0, 0].fill_between([-true_size[0]/2, true_size[0]/2],
                          estimated_center[1]+estimated_size[1]/2, true_size[1]/2,
                          color='#b8b8b8', alpha=.35,
                          label='Unmeasured positive-Y region')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    for index in range(2):
        axes[0, index+1].scatter(*local_views[index][:, :2].T, s=2, alpha=.6,
                                color=('#167cad', '#d18324')[index])
        axes[0, index+1].set_title(f'Camera {"AB"[index]} current target depths\nScoring surface membership only')
    axes[1, 0].scatter(*local_cloud[:, :2].T, s=3, alpha=.6, color='#167cad')
    axes[1, 0].set_title(f'Actual baseline accepted component\n{len(cloud)} current measured points')
    axes[1, 1].scatter(*local_cloud[:, :2].T, s=2, alpha=.25, color='#167cad')
    outline(axes[1, 1], estimated_center, estimated_size, color='#c0392b', label='Fitted extent')
    axes[1, 1].set_title('Incorrectly complete reconstruction\n90.0 × 43.4 mm; graspable=True')
    axes[1, 1].legend(fontsize=8)
    ax = axes[1, 2]
    ax.clear()
    ax.axis('off')
    labels = ('X−', 'X+', 'Y−', 'Y+', 'Z−', 'Z+')
    keys = ('X-1', 'X+1', 'Y-1', 'Y+1', 'Z-1', 'Z+1')
    table = ax.table(cellText=[[label, boundaries['0'][key], boundaries['1'][key]]
                              for label, key in zip(labels, keys)],
                     colLabels=['True physical plane', 'A points', 'B points'], loc='center',
                     cellLoc='center', colWidths=[.50, .25, .25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title('Measured physical plane support (±0.8 mm)\nY+ hidden; Z− is configured support prior')
    fig.suptitle('A consistent partial patch cannot establish the hidden physical extent', fontsize=15)
    fig.text(.5, .018, 'Raw camera data and actual source component; truth is used only for evaluation and drawing. '
             'No Pick/Place or gripper command.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(args.output/'hidden_box_failure.png', dpi=170)
    if args.docs_dir:
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.docs_dir/'hidden_box_failure.png', dpi=90)
    plt.close(fig)
    (args.output/'hidden_box_failure.json').write_text(json.dumps(dict(
        source=str(args.capture), raw_points=list(map(len, views)), accepted_component_points=len(cloud),
        physical_plane_points=boundaries, true_size_mm=true_size.tolist(),
        estimated_size_mm=estimated_size.tolist(), center_error_mm=record['center_error_mm'],
        baseline_graspable=estimate['graspable'], false_feasible=record['false_feasible'],
        interpretation='Physical Y+ not measured; fitted Y+ is an occlusion cut, not established extent.'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path('outputs/multiview-observability/baseline-partial'))
    parser.add_argument('--baseline-pythonpath', type=Path, default=Path('outputs/multiview-observability/baseline_snapshot'))
    parser.add_argument('--output', type=Path, default=Path('outputs/multiview-observability/figures'))
    parser.add_argument('--docs-dir', type=Path)
    main(parser.parse_args())
