#!/usr/bin/env python3
"""Plot observed occupancy and reduced RMPflow representation on separate axes.

The input is compact physical-study evidence. No simulator, depth processing or
runtime algorithm is executed. Failed physical outcomes remain in the figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYOUTS = ('A_current', 'C_asymmetric_1.15_0.65', 'K_B_front_height_0.85_rear')
FIELDS = ('occupied_voxels', 'supported_voxels', 'candidate_boxes',
    'floor_candidate_boxes', 'hazard_candidate_boxes', 'proxy_count',
    'selected_floor_boxes', 'selected_hazard_boxes', 'represented_voxels', 'omitted_voxels')


def select_rows(document):
    rows = []
    for layout in LAYOUTS:
        for scenario in ('static', 'partial'):
            matches = [row for row in document.get('cases', []) if row.get('cohort') == 'primary'
                and row.get('measurement_valid') is True and row.get('strategy') == 'reactive'
                and row.get('sampling_policy') == 'fixed' and row.get('global_ray_budget') == 1024
                and row.get('layout_id') == layout and row.get('scenario') == scenario]
            if len(matches) != 1:
                raise ValueError(f'Expected one explicit measured row for {layout}/{scenario}, got {len(matches)}')
            source = matches[0]
            values = source.get('extraction') or {}
            if any(type(values.get(key)) not in (int, float) or values[key] < 0 for key in FIELDS):
                raise ValueError('Representation counts must be measured, nonnegative numbers')
            if values['floor_candidate_boxes']+values['hazard_candidate_boxes'] != values['candidate_boxes']:
                raise ValueError('Candidate floor/above-plane counts do not match total')
            if values['selected_floor_boxes']+values['selected_hazard_boxes'] != values['proxy_count']:
                raise ValueError('Selected floor/above-plane counts do not match total')
            if values['represented_voxels']+values['omitted_voxels'] != values['supported_voxels']:
                raise ValueError('Represented/omitted supported source voxels do not match total')
            rows.append(dict(layout_id=layout, scenario=scenario, case_id=source['case_id'],
                artifact=source['artifact'], goal_reached=source['goal_reached'],
                strict_contact_free_success=source['strict_contact_free_success'],
                physical_classification=source['classification'],
                maximum_registered_proxies=source['maximum_registered_proxies'],
                counts={key: values[key] for key in FIELDS}))
    return rows


def render(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.7), constrained_layout=True)
    x = list(range(len(rows)))
    names = [f"{row['layout_id'][0]}\n{row['scenario']}" for row in rows]
    counts = [row['counts'] for row in rows]
    occupied = [row['occupied_voxels'] for row in counts]
    supported = [row['supported_voxels'] for row in counts]
    axis = axes[0, 0]
    axis.bar([v-.19 for v in x], occupied, .36, color='#356d9a', label='Observed OCCUPIED')
    axis.bar([v+.19 for v in x], supported, .36, color='#78adcb', label='Supported for extraction')
    axis.set_title('Occupancy evidence')
    axis.set_ylabel('Source voxels')

    def stacked(axis, lower_key, upper_key, lower_label, upper_label, title, ylabel,
                colors=('#d1b574', '#487f93')):
        lower, upper = [[row[key] for row in counts] for key in (lower_key, upper_key)]
        axis.bar(x, lower, .6, label=lower_label, color=colors[0])
        axis.bar(x, upper, .6, bottom=lower, label=upper_label, color=colors[1])
        for at, first, last in zip(x, lower, upper):
            axis.annotate(str(int(first+last)), (at, first+last), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    stacked(axes[0, 1], 'floor_candidate_boxes', 'hazard_candidate_boxes',
        'Floor candidate cuboids', 'Above-plane candidate cuboids',
        'Candidate obstacle representation', 'Candidate cuboids')
    stacked(axes[1, 0], 'selected_floor_boxes', 'selected_hazard_boxes',
        'Selected floor proxies', 'Selected above-plane proxies',
        'Selected RMPflow obstacle packet', 'Selected numeric proxies')
    axes[1, 0].axhline(64, color='black', linestyle='--', linewidth=.9)
    axes[1, 0].set_ylim(0, 94)
    stacked(axes[1, 1], 'represented_voxels', 'omitted_voxels',
        'Represented source voxels', 'Omitted supported source voxels',
        'Supported evidence after the proxy budget', 'Supported source voxels',
        colors=('#5b9f8d', '#c88367'))
    for axis in axes.flat:
        axis.set_xticks(x, names)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc='upper left', fontsize=8, framealpha=.9)
    figure.suptitle('Observed geometry and bounded RMPflow representation\n'
        'A current pair; C asymmetric pair; K front pair + rear camera. One run per case.', fontsize=12)
    figure.supxlabel('Different axes count voxels or cuboids. Above-plane is not a confirmed physical-obstacle label.\n'
        'Final snapshots follow different motion durations/outcomes; more evidence does not establish better avoidance.', fontsize=9)
    figure.savefig(path, dpi=180)
    if path.suffix.lower() != '.pdf':
        figure.savefig(path.with_suffix('.pdf'))
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning', required=True)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--summary', help='Optional compact count table bound to the source evidence hash')
    args = parser.parse_args(argv)
    source = Path(args.planning)
    data = source.read_bytes()
    rows = select_rows(json.loads(data))
    render(rows, args.figure)
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(schema_version=1, status='PASS',
            planning_sha256=hashlib.sha256(data).hexdigest(), rows=rows,
            interpretation='Measured final occupancy and extractor counts only; voxel and cuboid counts are separate units. All six rows are retained regardless of physical success.',
            limitations=['These are final published snapshots following different command durations, not time-normalized visibility measurements.',
                'Above-plane proxy labels are geometric categories, not confirmed fixture association.',
                'No FREE or UNKNOWN cell is added to improve a plot or representation score.']), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows=len(rows), figure=args.figure)))


if __name__ == '__main__':
    main()
