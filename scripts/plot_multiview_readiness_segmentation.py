"""Standalone scientific figures from retained selection/failure artifacts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.probe_multiview_segmentation import load_ply
from tests.support.multiview_research import ResearchObject, oracle_mask


COLORS = ('#277DA8', '#E88835', '#55A868', '#B565A7')


def save_figure(fig, args, name):
    fig.savefig(args.output/name, dpi=200)
    if args.docs_dir is not None:
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.docs_dir/name, dpi=90)


def draw_cloud(ax, clouds, *, origin, title, limits, labels=None):
    for index, cloud in enumerate(clouds):
        points = (cloud-origin)*1000
        ax.scatter(*points.T, s=5 if len(cloud) < 1000 else 2, alpha=.82,
                   color=COLORS[index % len(COLORS)], linewidths=0,
                   label=None if labels is None else labels[index])
    ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect(np.maximum(limits[:, 1]-limits[:, 0], 10))
    ax.view_init(elev=23, azim=-66)
    ax.tick_params(labelsize=7, pad=0)
    ax.title.set_fontsize(11)


def bounds(points, origin):
    limits = np.array([points.min(0)-origin, points.max(0)-origin]).T*1000
    return limits+[-3, 3]


def partition_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        points = data['fused'] if 'fused' in data else data['points']
        clouds = [data[key] for key in sorted(data.files) if key.startswith('partition_')]
        seeds = [data[key] for key in sorted(data.files) if key.startswith('broad_0_seed_')]
    return points, clouds, seeds


def sphere_figure(args, plt, rows, manifest):
    fig = plt.figure(figsize=(13, 8.7), constrained_layout=False)
    metrics = []
    for row_index, gap in enumerate((2, 10)):
        key = f'selection_gap_{gap}_sphere_sphere'
        points, clouds, seeds = partition_arrays(args.partitions/key/'retained.npz')
        scene = next(scene for scene in manifest['scenes'] if scene['key'] == key)
        truth = scene['frames'][-1]['objects']
        result = next(row for row in rows if row['scene'] == key and row['method'] == 'retained')
        origin = np.r_[points[:, :2].mean(0), 0.]
        limits = bounds(points, origin)
        display = (('Fused: one mixed unknown', [points]),
                   ('Dense seeds; sparse islands excluded', seeds),
                   ('Retained: two current sphere clouds', clouds))
        for column, (title, selected) in enumerate(display):
            ax = fig.add_subplot(2, 3, row_index*3+column+1, projection='3d')
            draw_cloud(ax, selected, origin=origin, title=title, limits=limits)
        matches = []
        for obj in truth:
            fit = min(result['fits'], key=lambda fit: np.linalg.norm(np.asarray(fit['center'])-obj['center']))
            matches.append(dict(object=obj['name'], center_error_mm=float(1000*np.linalg.norm(np.asarray(fit['center'])-obj['center'])),
                                radius_error_mm=1000*(fit['radius']-obj['radius'])))
        coverage = sum(map(len, seeds))/len(points)
        caption = (f"{gap} mm physical gap · {len(points)} measured points · dense seeds {coverage:.1%} · "
                   f"retained {sum(map(len, clouds))/len(points):.1%} · "
                   f"mean center error {np.mean([m['center_error_mm'] for m in matches]):.3f} mm · "
                   f"mean radius bias {np.mean([m['radius_error_mm'] for m in matches]):+.3f} mm")
        fig.text(.5, .515 if row_index == 0 else .055, caption, ha='center', fontsize=9)
        metrics.append(dict(scene=key, seed=scene['seed'], gap_mm=gap, measurements=len(points),
                            seed_coverage=coverage, retained_coverage=sum(map(len, clouds))/len(points), geometry_errors=matches))
    fig.suptitle('Sparse samples belong to distinct observed spheres', fontsize=15, y=.98)
    fig.text(.5, .015,
        'Selection only; fixed merged-baseline fitter isolates segmentation. Final parameter envelopes are evaluated separately.\n'
        'Colors show current measured geometric partitions; no points or hidden surfaces are reconstructed.',
        ha='center', fontsize=8)
    fig.subplots_adjust(left=.02, right=.98, bottom=.1, top=.9, hspace=.35, wspace=.08)
    save_figure(fig, args, 'sphere_pair_reconstruction.png')
    plt.close(fig)
    return metrics


def missing_band_figure(args, plt):
    fig = plt.figure(figsize=(10, 5.5))
    paths = [args.control_partitions/f'box_missing_band_{method}.npz' for method in ('baseline', 'retained')]
    for index, path in enumerate(paths):
        points, clouds, _ = partition_arrays(path)
        origin = np.r_[points[:, :2].mean(0), 0.]
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        draw_cloud(ax, clouds, origin=origin,
            title=('Previous: two known boxes (false split)' if index == 0 else 'Retained: one unknown, non-graspable'),
            limits=bounds(points, origin))
    fig.suptitle('One physical box with an 8 mm measurement band removed', fontsize=14)
    fig.text(.5, .065, 'The same remaining XYZ can fit one whole box or two partial boxes. No valid image-space gap evidence is available.',
             ha='center', fontsize=9)
    fig.text(.5, .025, 'Conservative ambiguity keeps one unknown observation; missing points are never fabricated.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.01, right=.99, top=.84, bottom=.15, wspace=.1)
    save_figure(fig, args, 'missing_measurement_ambiguity.png')
    plt.close(fig)


def narrow_gap_figure(args, plt):
    if not (args.failure_dir/'fused.ply').exists():
        return None
    points = load_ply(args.failure_dir/'fused.ply')
    shift = np.r_[np.random.default_rng(1301).uniform(-.004, .004, 2), 0.]
    cylinder = ResearchObject('upright_0', 'cylinder', np.array([.35, -.18, .045])+shift,
        np.array([.036, .036, .09]), radius=.018, length=.09)
    panel = ResearchObject('Occluder', 'cube', np.array([.32, -.14, .10]), np.array([.15, .035, .20]))
    a = points[oracle_mask(points, cylinder, tolerance=.0008)]
    b = points[oracle_mask(points, panel, tolerance=.0008)]
    closest = (np.inf, None, None)
    for start in range(0, len(a), 128):
        distance = ((a[start:start+128, None]-b[None])**2).sum(2)
        first, second = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[first, second] < closest[0]:
            closest = distance[first, second], a[start+first], b[second]
    origin = np.array([cylinder.center[0], -.1575, 0.])
    fig = plt.figure(figsize=(11, 5.8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    draw_cloud(ax, (a, b), origin=origin, title='Visible cylinder and oversized panel', limits=bounds(np.r_[a, b], origin),
               labels=('Cylinder (evaluation label)', 'Panel (evaluation label)'))
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    zoom = fig.add_subplot(1, 2, 2)
    for cloud, color in ((a, COLORS[0]), (b, COLORS[1])):
        q = (cloud-origin)*1000
        zoom.scatter(q[:, 0], q[:, 1], s=4, color=color, alpha=.6, linewidths=0)
    first, second = (closest[1]-origin)*1000, (closest[2]-origin)*1000
    zoom.plot([first[0], second[0]], [first[1], second[1]], color='#262626', marker='o', markersize=3)
    zoom.set(xlabel='x (mm)', ylabel='y (mm)', title='Top view of unresolved boundary',
             xlim=(min(first[0], second[0])-12, max(first[0], second[0])+12), ylim=(-12, 10))
    zoom.set_aspect('equal', adjustable='box'); zoom.grid(alpha=.18)
    physical = 1000*(.0045-shift[1])
    fig.suptitle('Retained limitation: one metric component, conservative omission', fontsize=14)
    fig.text(.5, .065, f'Seed 1301: physical gap {physical:.3f} mm; nearest measured 3D samples {1000*np.sqrt(closest[0]):.3f} mm; graph radius 4 mm.',
             ha='center', fontsize=9)
    fig.text(.5, .025, 'Colors use scoring-only analytic surface labels. Production receives no labels and does not publish a mixed known primitive.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.03, right=.98, bottom=.17, top=.84, wspace=.22)
    save_figure(fig, args, 'remaining_narrow_gap.png')
    plt.close(fig)
    return dict(seed=1301, physical_gap_mm=physical, closest_measured_samples_mm=1000*np.sqrt(closest[0]),
                source=str(args.failure_dir), scoring_only=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    rows = json.loads(args.evaluation.read_text())['rows']
    manifest = json.loads((args.selection_capture/'capture_manifest.json').read_text())
    spheres = sphere_figure(args, plt, rows, manifest)
    missing_band_figure(args, plt)
    narrow = narrow_gap_figure(args, plt)
    (args.output/'segmentation_figure_metrics.json').write_text(json.dumps(dict(sphere_scenes=spheres,
        narrow_gap=narrow, sources=dict(evaluation=str(args.evaluation), selection=str(args.selection_capture),
                                       partitions=str(args.partitions), controls=str(args.control_partitions)),
        fitter_stage='Fixed merged-baseline primitive fitter, before final parameter envelope; isolates segmentation',
        limitation='Offline saved selection/control/failure visualization. No live throughput or hidden surface reconstruction is implied.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-capture', type=Path, default=Path('outputs/multiview-iteration2/selection-capture-v2'))
    parser.add_argument('--evaluation', type=Path, default=Path('outputs/readiness-segmentation/retained-selection-initial.json'))
    parser.add_argument('--partitions', type=Path, default=Path('outputs/readiness-segmentation/retained-partitions'))
    parser.add_argument('--control-partitions', type=Path, default=Path('outputs/readiness-segmentation/negative-controls'))
    parser.add_argument('--failure-dir', type=Path,
        default=Path('outputs/multiview-iteration2-native/run-68mnt35r/multiview_test_close_object_recovery/1_1'))
    parser.add_argument('--output', type=Path, default=Path('outputs/multiview-readiness/figures'))
    parser.add_argument('--docs-dir', type=Path, help='Optional small90dpi review copies; high-resolution originals stay under--output')
    main(parser.parse_args())
