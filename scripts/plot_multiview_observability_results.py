"""Scientific observability figures from saved evaluation rows and native clouds.

This utility never constructs scenes, runs perception, opens Isaac, or generates
holdout fixtures. Truth outlines/labels are evaluation-only annotations.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from scripts.benchmark_multiview_grasp_readiness import plain_values
from tests.support.multiview_iteration2 import object_from_payload, surface_memberships


COLORS = dict(A='#2b6cb0', B='#d9822b', baseline='#b85b4f', improved='#277758',
              bounded='#277758', occluded='#b85b4f', missing='#936a23')


def read(path):
    return json.loads(Path(path).read_text())


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def rectangle(center, size, yaw):
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return np.asarray(center)[:2]+(signs*np.asarray(size)[:2]/2) @ rotation(yaw).T


def row_at(rows, scene, method, index=3):
    selected = [row for row in rows if row['scene'] == scene and row['method'] == method
                and row['frame_index'] == index]
    if len(selected) != 1:
        raise ValueError(f'Need exactly one saved row for{scene}/{method}/frame{index}')
    return selected[0]


def draw_cloud(ax, row, record, capture_root, *, title, color, show_legend=False):
    truth = row['truth_geometry']
    target = object_from_payload(truth)
    solids = tuple(object_from_payload(obj) for obj in record['objects']+record.get('clutter', []))
    index = row['frame_index']
    cloud_path = capture_root/'captures'/row['scene']/f'workspace_frame_{index:02d}.npz'
    origin = np.asarray(truth['center'])[:2]
    to_local = lambda xy: (np.asarray(xy)-origin) @ rotation(truth['yaw'])*1000
    with np.load(cloud_path, allow_pickle=False) as cloud:
        for ci, label in enumerate(('A', 'B')):
            points = cloud[f'camera_{ci}']
            points = points[surface_memberships(points, solids) == 0]
            points = points[np.linspace(0, len(points)-1, min(1500, len(points)), dtype=int)] if len(points) else points
            xy = to_local(points[:, :2])
            ax.scatter(xy[:, 0], xy[:, 1], s=5, alpha=.48, color=COLORS[label],
                       label=f'Camera {label}: measured 2 mm cloud', rasterized=True)
    outline = to_local(rectangle(truth['center'], truth['size'], truth['yaw']))
    ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor='#414141',
                         linewidth=1.6, linestyle='--', label='True physical footprint'))
    geometry = row['geometry_evidence']
    if geometry is None or geometry['box_size'] is None:
        raise ValueError('Representative plot requires a saved known box')
    fitted = to_local(rectangle(geometry['estimated_center'], geometry['box_size'], geometry['yaw']))
    ax.add_patch(Polygon(fitted, closed=True, fill=False, edgecolor=color,
                         linewidth=2., label='Published measured box'))
    if row['scene'].endswith('partial_box_both'):
        apparent_edge = float(fitted[:, 1].max())
        true_edge = float(outline[:, 1].max())
        ax.add_patch(Rectangle((outline[:, 0].min(), apparent_edge), np.ptp(outline[:, 0]),
            true_edge-apparent_edge, facecolor='#9a9a9a', alpha=.10, hatch='///', edgecolor='#999999'))
        ax.text(0, (apparent_edge+true_edge)/2, 'Unmeasured physical extent\n(truth annotation)',
                ha='center', va='center', fontsize=8, color='#555555')
        ax.annotate('Apparent +Y cut', xy=(fitted[:, 0].max(), apparent_edge),
                    xytext=(34, -15), fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color), ha='right')
    bound = max(np.max(abs(outline)), np.max(abs(fitted)))+8
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal',
           xlabel='True object-local X (mm)', ylabel='True object-local Y (mm)', title=title)
    ax.grid(alpha=.17)
    if show_legend:
        ax.legend(loc='lower left', fontsize=6.7, framealpha=.94)


def side_lines(row):
    evidence = row['geometry_evidence']['metadata'].get('geometric_observability', {})
    result = []
    for axis in evidence.get('axes', [])[:2]:
        for label, sign in (('negative', '−'), ('positive', '+')):
            side = axis[label]
            views = side.get('cameras', [])
            result.append(dict(side=sign+axis['axis'].upper(), status=side['status'],
                free=sum(view.get('free', 0) for view in views),
                transitions=sum(view.get('transitions', 0) for view in views),
                foreground=sum(view.get('occluded', 0) for view in views),
                supported_bins=[view.get('bins_supported', 0) for view in views]))
    return result


def eligibility_text(row):
    accepted = bool(row['estimated_plan'] and row['estimated_plan']['feasible'])
    return f"graspable={row['graspable']}  |  planner={'accept' if accepted else 'reject'}"


def save_figure(fig, name, output, docs_dir):
    fig.savefig(output/f'{name}.png', dpi=200, facecolor='white')
    if docs_dir is not None:
        fig.savefig(docs_dir/f'{name}.png', dpi=90, facecolor='white')
    plt.close(fig)


def comparison_figure(rows, records, capture_root, output, docs_dir):
    scene = next(key for key in records if key.endswith('partial_box_both'))
    before, after = (row_at(rows, scene, method) for method in ('baseline', 'improved'))
    clear_scene = next(row['scene'] for row in rows if row['method'] == 'improved' and
        row['frame_index'] == 3 and row['scene'].endswith('clear_both') and row['graspable'] is True
        and row['estimated_feasible'] and row['valid_fixture'])
    clear = row_at(rows, clear_scene, 'improved')
    fig = plt.figure(figsize=(13.2, 7.3))
    grid = fig.add_gridspec(2, 3, height_ratios=(3.8, 1.65), hspace=.4, wspace=.32)
    titles = ('Previous: confident partial box', 'Current: incomplete dimensions rejected',
              'Current: complete clear box retained')
    for column, (row, title) in enumerate(zip((before, after, clear), titles)):
        draw_cloud(fig.add_subplot(grid[0, column]), row, records[row['scene']], capture_root,
                   title=title, color=COLORS['baseline' if column == 0 else 'improved'], show_legend=column == 0)
        text_ax = fig.add_subplot(grid[1, column])
        text_ax.axis('off')
        text_ax.text(0, 1.05, eligibility_text(row), color=COLORS['baseline' if column == 0 else 'improved'],
                     transform=text_ax.transAxes, fontsize=9, fontweight='bold')
        if column == 0:
            size = np.asarray(row['geometry_evidence']['box_size'])*1000
            truth_size = np.asarray(row['truth_geometry']['size'])*1000
            jaw = row['estimated_plan']['details']['required_width_m']*1000
            actual_jaw = row['proposed_truth_checks']['true_required_width_m']*1000
            message = (f"True XY: {truth_size[0]:.1f} × {truth_size[1]:.1f} mm\nPublished XY: {size[0]:.2f} × {size[1]:.2f} mm\n"
                       f"Center error: {row['center_error_mm']:.2f} mm\n"
                       f"Class: cube; confidence={row['confidence']:.7f}\n"
                       f'Accepted jaw width {jaw:.2f} mm; actual width {actual_jaw:g} mm.')
            text_ax.text(0, .83, message, va='top', fontsize=8.5, linespacing=1.5)
        else:
            for line, side in enumerate(side_lines(row)):
                status = side['status']
                text_ax.text(0, .83-line*.19,
                    f"{side['side']}: {status}; free/nearer={side['free']}/{side['foreground']}; bins={side['supported_bins']}",
                    va='top', fontsize=8.4, color=COLORS.get(status, '#555555'))
    fig.suptitle('Current depth distinguishes an occlusion cut from a complete box boundary', fontsize=13, y=.99)
    fig.text(.5, .025,
        'Selection replay, identical saved native frames. Truth outlines/colors score visibility only; no truth enters perception.\n'
        'The partial measured box remains a known cube; unbounded sides revoke eligibility. Clear-case intervals are empirical measured bounds.',
        ha='center', va='bottom', fontsize=8)
    fig.subplots_adjust(bottom=.11, top=.91)
    save_figure(fig, 'observability_boundaries_before_after', output, docs_dir)
    return dict(scene=scene, frame_index=3, clear_scene=clear_scene, before=before, after=after, clear=clear,
                hidden_edge_evidence=side_lines(after), clear_edge_evidence=side_lines(clear))


def temporal_figure(rows, contracts, output, docs_dir):
    scene = next(row['scene'] for row in rows if row['phase'] == 'incomplete')
    selected = {method: sorted((row for row in rows if row['scene'] == scene and row['method'] == method),
                                key=lambda row: row['frame_index']) for method in ('baseline', 'improved')}
    if not all(selected.values()):
        raise ValueError('Need baseline and current temporal rows')
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.9), sharex=True)
    for ax, (field, label) in zip(axes, (('graspable', 'Published grasp eligibility'),
                                       ('planner', 'Existing planner accepts'))):
        ax.axvspan(3.5, 5.5, color='#c4b488', alpha=.23)
        for method, offset in (('baseline', .025), ('improved', -.025)):
            sequence = selected[method]
            values = []
            for row in sequence:
                value = (None if row['estimated_plan'] is None else row['estimated_plan']['feasible']) if field == 'planner' else row[field]
                values.append(np.nan if value is None else float(value)+offset)
            ax.step([row['frame_index'] for row in sequence], values, where='mid',
                    marker='o', lw=1.7, color=COLORS[method], label='Previous' if method == 'baseline' else 'Current')
        ax.set(yticks=(0, 1), yticklabels=('False', 'True'), ylim=(-.18, 1.2), ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].set_title('Same target identity: complete → incomplete → genuinely complete measurements')
    for xpos, text in ((2.6, 'Complete'), (4.5, 'Both-view extent hidden'), (7., 'Restored')):
        axes[0].text(xpos, 1.10, text, ha='center', fontsize=8.5)
    frozen = [entry for entry in contracts if entry['scene'] == scene and entry['method'] == 'improved'
              and entry['kind'] == 'frozen_frame_no_restore_or_freshness_refresh']
    frozen_ok = bool(frozen and all(entry['passed'] for entry in frozen))
    axes[1].annotate(f"Frozen-frame query: {'no eligibility/freshness refresh' if frozen_ok else 'see failed contract'}",
                     xy=(5.35, .02), xytext=(4.1, .42), fontsize=8, color=COLORS['improved'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['improved']))
    axes[1].set(xticks=range(2, 9), xlabel='Distinct saved native frame index', xlim=(1.8, 8.2))
    identities = sorted({row['object_id'] for row in selected['improved'] if row['object_id'] is not None})
    fig.text(.5, .025, f"Current object ID throughout: {', '.join(identities)}. Each point uses a distinct native capture; no robot commands.\n"
             'Planner panel uses raw plan.feasible. The separate Pick-start guard also rejects held/uncertain sentinel states at frames 4/5.\n'
             'Small vertical offsets separate coincident boolean traces.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    save_figure(fig, 'observability_temporal_eligibility', output, docs_dir)
    return dict(scene=scene, identities=identities, frozen_contract_passed=frozen_ok, rows=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=Path('outputs/multiview-observability/selection-clean-initial'))
    parser.add_argument('--output', type=Path, default=Path('outputs/multiview-observability/figures'))
    parser.add_argument('--docs-dir', type=Path)
    parser.add_argument('--freeze-manifest', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.docs_dir:
        args.docs_dir.mkdir(parents=True, exist_ok=True)
    rows = read(args.evaluation/'rows.json')
    manifest = read(args.evaluation/'capture_manifest.json')
    capture_root = Path(manifest.get('capture_origin', args.evaluation))
    records = {entry['key']: entry for entry in manifest['scenes']}
    comparison = comparison_figure(rows, records, capture_root, args.output, args.docs_dir)
    temporal = temporal_figure(rows, read(args.evaluation/'contracts.json'), args.output, args.docs_dir)
    source = read(args.evaluation/'provenance_start.json')
    metadata = dict(evaluation=str(args.evaluation.resolve()), capture_root=str(capture_root.resolve()),
        evaluation_source_provenance=source, comparison=comparison, temporal=temporal,
        note='Figures consume saved outcomes only. True geometry is scoring-only. This selected-scene illustration is not an independent holdout score.')
    if args.freeze_manifest:
        frozen = read(args.freeze_manifest)
        package = {name: value for name, value in source['sha256'].items()
                   if name.startswith('robot_skill_stack/world/perception_multiview/')}
        metadata['freeze_manifest'] = str(args.freeze_manifest.resolve())
        metadata['freeze_matches_evaluation'] = all(frozen['sha256'].get(name) == value for name, value in package.items())
    (args.output/'observability_figure_metrics.json').write_text(json.dumps(plain_values(metadata), indent=2))


if __name__ == '__main__':
    main()
