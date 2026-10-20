"""Compact presentation figures from recorded final-perception experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth, voxelize
from scripts.benchmark_multiview_robustness import load_frames
from tests.support.multiview_final_evaluation import STAGES

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(figure, destination, name):
    figure.savefig(destination/name, dpi=170, bbox_inches='tight')
    plt.close(figure)


def layouts_figure(manifest, selected, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.get_cmap('tab10')
    for index, layout in enumerate(manifest['layouts']):
        eyes, targets = np.asarray(layout['eyes']), np.asarray(layout['targets'])
        for axis, coordinates in zip(axes, ((0, 1), (0, 2))):
            for camera, (eye, target) in enumerate(zip(eyes, targets)):
                xy, target_xy = eye[list(coordinates)], target[list(coordinates)]
                axis.scatter(*xy, marker='o' if camera == 0 else '^', color=colors(index), s=35)
                axis.plot(*np.array([xy, target_xy]).T, color=colors(index),
                    linewidth=1.7 if layout['name'] == selected else .6, alpha=.65)
            if axis is axes[0]:
                axis.plot([], [], color=colors(index), label=layout['name'])
    axes[0].add_patch(Rectangle((.2, -.4), .5, .8, fill=False, edgecolor='black', linestyle='--'))
    axes[1].plot([.2, .7], [0, 0], 'k--')
    for axis, label in zip(axes, ('Y (m), left + / right −', 'Z (m)')):
        axis.scatter(0, 0, marker='s', color='black', s=60)
        axis.set(xlabel='X (m), forward +', ylabel=label, aspect='equal')
        axis.grid(alpha=.2)
    axes[0].set_title('Top: actual robot base and workspace')
    axes[1].set_title('Side: fixed external cameras')
    axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    figure.suptitle(f'Bounded camera study — selected: {selected}\nA: circles; B: triangles; rays show aiming direction')
    figure.tight_layout(rect=(0, 0, 1, .86))
    save(figure, destination, 'camera_layouts.png')


def ranking_figure(coarse, shortlist, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, result, title in zip(axes, (coarse, shortlist), ('Coarse selection', 'New-scene shortlist validation')):
        values = []
        for name, methods in result['metrics']['summary'].items():
            summary = methods['current_fused']['latest_static_scene_metrics']
            values.append((name, summary))
        values.sort(key=lambda item: (-item[1]['correct_classification_rate'],
            (item[1]['dimension_correct_mm'] or {}).get('p95', 1e9)))
        labels = [name for name, _ in values]
        quality = [100*summary['correct_classification_rate'] for _, summary in values]
        axis.barh(labels, quality, color='#347d9e')
        for index, (_, summary) in enumerate(values):
            axis.text(1, index, f"{summary['correct']}/{summary['expected']}; "
                f"{summary['missing']} miss, {summary['unknown']} unknown", va='center', color='white', fontsize=8)
        axis.set(xlim=(0, 103), xlabel='Correct detection + classification (%)', title=title)
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=.2)
    figure.suptitle('One latest frame per static scene; misses retained, temporal rows reported separately')
    figure.tight_layout()
    save(figure, destination, 'camera_ranking.png')


def final_metrics_figure(result, selected, destination):
    comparisons = [('original_reference', 'v1_a', 'Original\nV1 A'),
        ('original_reference', 'baseline_fused', 'Original\npreserved fusion'),
        ('original_reference', 'current_fused', 'Original\nfinal fusion'),
        (selected, 'baseline_fused', 'Selected\npreserved fusion'),
        (selected, 'current_single_a', 'Selected\nA alone'),
        (selected, 'current_single_b', 'Selected\nB alone'),
        (selected, 'current_fused', 'Selected\nfinal fusion')]
    rows = [(layout, method, result['metrics']['summary'][layout][method]['latest_static_scene_metrics'])
        for layout, method, _ in comparisons]
    labels = [label for _, _, label in comparisons]
    colors = ['#648a9b']*(len(rows)-1)+['#23804c']
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    rates = [100*summary['correct_classification_rate'] for _, _, summary in rows]
    axes[0].bar(range(len(rows)), rates, color=colors)
    for index, ((_, _, summary), rate) in enumerate(zip(rows, rates)):
        axes[0].text(index, rate+1, f"{summary['correct']}/{summary['expected']}", ha='center', fontsize=7)
    axes[0].set(ylabel='Correct / all expected (%)', ylim=(0, 104))
    for axis, key, title in zip(axes[1:], ('center_correct_mm', 'dimension_correct_mm'),
            ('Correct-fit center p95 (mm)', 'Correct-fit max dimension error p95 (mm)')):
        axis.bar(range(len(rows)), [(summary[key] or {}).get('p95', np.nan) for _, _, summary in rows], color=colors)
        axis.axhline(5, color='red', linestyle='--', linewidth=.8)
        axis.set(ylabel=title)
    for axis in axes:
        axis.set_xticks(range(len(rows)), labels, rotation=40, ha='right', fontsize=7)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('Independent frozen holdout: one latest frame per static scene, all misses retained\n'
        'Matched physical scenes; different images across layouts; geometry conditional on correct class')
    figure.tight_layout()
    save(figure, destination, 'holdout_accuracy.png')


def latency_figure(result, destination):
    layouts = list(result['metrics']['summary'])
    if len(layouts) != 1:
        raise ValueError('Use a dedicated single-layout native profile; multi-layout averages obscure camera effects')
    values = result['metrics']['native_timing']
    if not values or any(value.get('samples', 0) < 1 or not value.get('all_samples_fresh')
                         for value in values.values()):
        raise ValueError('Use a native capture profile with fresh measured samples, not a saved replay')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = list(values)
    bottom = np.zeros(len(methods))
    for stage in STAGES[:-1]:
        heights = np.array([(values[method]['stage_ms'].get(stage) or {}).get('mean', 0) for method in methods])
        axes[0].bar(methods, heights, bottom=bottom, label=stage.removesuffix('_ms'))
        bottom += heights
    axes[0].set(ylabel='Native source mean (ms)', title='Measured stage costs')
    axes[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    rates = [values[method]['effective_hz'] for method in methods]
    axes[1].bar(methods, rates, color='#347d9e')
    axes[1].axhspan(4, 5, alpha=.12, color='green', label='Aspirational 4–5 Hz target')
    axes[1].set(ylabel='Actual render + source (Hz)', title='Eight native render steps per sample')
    axes[1].legend(fontsize=7, loc='upper right')
    for axis in axes:
        axis.tick_params(axis='x', labelrotation=15)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle(f'Selected-layout native profile: {layouts[0]}')
    figure.tight_layout()
    save(figure, destination, 'native_latency.png')


def representative_figures(capture, evaluation, selected, destination):
    manifest = read(capture/'capture_manifest.json')
    candidates = [record for record in manifest['scenes'] if record['layout']['name'] == selected and not record['temporal']]
    record = next((record for record in candidates if len(record['objects']) >= 4), candidates[0])
    snapshot = record['frames'][-1]
    frames = load_frames(capture, snapshot['files'])
    figure, axes = plt.subplots(2, 2, figsize=(10, 6))
    for index, frame in enumerate(frames):
        axes[0, index].imshow(frame.rgb)
        image = axes[1, index].imshow(np.where(np.isfinite(frame.depth), frame.depth, np.nan),
            vmin=.1, vmax=2., cmap='viridis')
        axes[0, index].set_title(f'Camera {"AB"[index]}: RGB')
        axes[1, index].set_title(f'Camera {"AB"[index]}: optical depth (m)')
    for axis in axes.ravel():
        axis.axis('off')
    figure.colorbar(image, ax=axes[1].tolist(), shrink=.8)
    figure.suptitle(f'{selected}, {record["name"]}; native 640 × 480')
    save(figure, destination, 'representative_rgb_depth.png')
    config = load_scene_config(ROOT/'config/scenes/multiview_playground.toml').scene
    clouds = [voxelize(crop_workspace(project_depth(frame, stride=2)[0], config), .002, 12000) for frame in frames]
    clouds.append(voxelize(np.concatenate(clouds), .002, 12000))
    predicted = [row for row in read(evaluation/'observations.json') if row['trial'] == record['key']
        and row['method'] == 'current_fused' and row['frame_index'] == snapshot['index']]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, (cloud, title) in enumerate(zip(clouds, ('Camera A', 'Camera B', 'Fused measurements + geometry'))):
        for row, dims in enumerate(((0, 1), (0, 2))):
            axis = axes[row, column]
            axis.scatter(cloud[:, dims[0]], cloud[:, dims[1]], c=cloud[:, 2], s=1, vmin=0, vmax=.2, cmap='viridis')
            axis.set(xlabel='X (m)', ylabel='YZ'[row]+' (m)', aspect='equal', title=title if row == 0 else None)
            axis.grid(alpha=.15)
    def geometry(axis, item, color, style, truth, vertical=False):
        center = np.asarray(item['center'])
        coordinates = center[[0, 2]] if vertical else center[:2]
        shape = item['shape']
        data = item if truth else item.get('geometry', {})
        if shape == 'sphere' and data.get('radius'):
            axis.add_patch(Circle(coordinates, data['radius'], fill=False, color=color, linestyle=style))
        if shape == 'cube':
            size = np.asarray(data['size'] if truth else data['box_size'])
            yaw = data.get('yaw')
            if yaw is None:
                yaw = data.get('metadata', {}).get('orientation_observability', {}).get('yaw_hypothesis_rad')
            if yaw is None:
                axis.plot(*coordinates, marker='x', color=color)
                return
            if vertical:
                extent = np.array([abs(np.cos(yaw))*size[0]+abs(np.sin(yaw))*size[1], size[2]])
                axis.add_patch(Rectangle(coordinates-extent/2, *extent,
                    fill=False, color=color, linestyle=style, linewidth=1.2))
                return
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            corners = np.array([[-1,-1], [1,-1], [1,1], [-1,1], [-1,-1]])*size[:2]/2
            corners = corners @ rotation.T+center[:2]
            axis.plot(*corners.T, color=color, linestyle=style, linewidth=1.2)
    for vertical in (False, True):
        for item in snapshot['objects']:
            geometry(axes[int(vertical), 2], item, 'black', '--', True, vertical)
        for item in predicted:
            geometry(axes[int(vertical), 2], item, 'red', '-', False, vertical)
    axes[0, 2].plot([], [], 'k--', label='Physical truth (scoring only)')
    axes[0, 2].plot([], [], 'r-', label='Automatic model (fitted yaw hypothesis)')
    axes[0, 2].legend(fontsize=7)
    figure.tight_layout()
    save(figure, destination, 'measured_clouds_geometry.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('coarse', 'shortlist', 'holdout', 'native', 'capture', 'evaluation', 'output'):
        parser.add_argument('--'+argument, type=Path, required=True)
    parser.add_argument('--selected', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    layouts_figure(read(args.coarse/'capture_manifest.json'), args.selected, args.output)
    ranking_figure(read(args.coarse/'result.json'), read(args.shortlist/'result.json'), args.output)
    final_metrics_figure(read(args.holdout/'result.json'), args.selected, args.output)
    latency_figure(read(args.native/'result.json'), args.output)
    representative_figures(args.capture, args.evaluation, args.selected, args.output)
