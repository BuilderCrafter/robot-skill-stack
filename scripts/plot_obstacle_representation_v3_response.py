#!/usr/bin/env python3
"""Compact actual V3 static Lula responses and plot matched policy conditions.

Only previously measured JSON is read. No Lula, simulator or GPU is imported.
Nominal matrices/vectors retain their original precision; repeated metadata and
the six offset matrices are replaced by explicit sensitivity extrema.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


POLICY_FIELDS = ('natural_policy_force', 'natural_policy_metric', 'canonical_acceleration_rad_s2')
NORM_FIELDS = ('collision_force_norm', 'collision_metric_norm', 'canonical_acceleration_norm')


def flattened(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in flattened(part)]
    return [value]


def difference_norm(first, last):
    first, last = flattened(first), flattened(last)
    if len(first) != len(last):
        raise ValueError('Matched response arrays must have equal shapes')
    return math.sqrt(sum((a-b)**2 for a, b in zip(first, last)))


def compact(raw, *, source=None):
    grouped, cases = defaultdict(list), {}
    for row in raw.get('rows') or []:
        grouped[(row['case_id'], row['representation'], row['posture'])].append(row)
        cases.setdefault(row['case_id'], dict(case_id=row['case_id'], layout_id=row.get('layout_id'),
            scenario=row.get('scenario'), snapshot=row.get('snapshot')))
    responses, incomplete = [], []
    expected_offsets = {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'}
    for key, values in sorted(grouped.items()):
        nominal = [row for row in values if row['perturbation']['id'] == 'nominal']
        if len(nominal) != 1:
            raise ValueError('Each measured policy condition requires exactly one nominal response: '+repr(key))
        nominal = nominal[0]
        offsets = [row for row in values if row['perturbation']['id'] != 'nominal']
        labels = [row['perturbation']['id'] for row in offsets]
        if len(set(labels)) != len(labels):
            raise ValueError('Duplicate sensitivity displacement in measured condition: '+repr(key))
        missing = sorted(expected_offsets-set(labels))
        if missing or set(labels)-expected_offsets:
            incomplete.append(dict(case_id=key[0], representation=key[1], posture=key[2],
                missing_offsets=missing, unexpected_offsets=sorted(set(labels)-expected_offsets)))
        sensitivities = {}
        for field in POLICY_FIELDS:
            deltas = [(difference_norm(nominal[field], row[field]), row['perturbation']) for row in offsets]
            maximum, displacement = max(deltas, default=(None, None), key=lambda item: item[0])
            sensitivities[field] = dict(maximum_delta_norm=maximum, attaining_displacement=displacement,
                evaluated_offsets=len(offsets))
        cosine = [row.get('joint_acceleration_direction_cosine_to_baseline') for row in offsets
            if row.get('joint_acceleration_direction_cosine_to_baseline') is not None]
        responses.append(dict(case_id=key[0], representation=key[1], posture=key[2],
            registered_obstacles=nominal.get('registered_obstacles'),
            same_selected_cell_union_as_baseline=nominal.get('same_selected_cell_union_as_baseline'),
            nominal={field: nominal[field] for field in POLICY_FIELDS},
            nominal_norms={field: nominal.get(field) for field in NORM_FIELDS},
            nominal_direction_cosines={field: nominal.get(field) for field in
                ('joint_acceleration_direction_cosine_to_baseline', 'collision_force_direction_cosine_to_baseline')},
            offset_sensitivity=sensitivities,
            offset_joint_acceleration_direction_cosine_to_baseline_range=None if not cosine else [min(cosine), max(cosine)]))
    control = raw.get('identical_geometry_negative_control') or {}
    fields = {field for row in control.get('checks') or [] for field in row.get('maximum_absolute_differences') or {}}
    control_summary = {key: control.get(key) for key in ('status', 'comparisons', 'absolute_tolerance', 'relative_tolerance', 'scope')}
    control_summary['maximum_absolute_differences'] = {field: max(
        (row['maximum_absolute_differences'][field] for row in control.get('checks') or []
            if row.get('maximum_absolute_differences', {}).get(field) is not None), default=None) for field in sorted(fields)}
    control_summary['failed_comparisons'] = [
        {key: row.get(key) for key in ('case_id', 'representation', 'posture', 'perturbation', 'maximum_absolute_differences')}
        for row in control.get('checks') or [] if row.get('matched') is not True]
    return dict(schema_version=1, status='PASS' if raw.get('status') == 'PASS' and responses and not incomplete
        and control.get('status') != 'FAIL' and not raw.get('missing_packets') else 'PARTIAL_FAILURE',
        source=source, scope=raw.get('scope'), counts=dict(raw_queries=len(raw.get('rows') or []),
            cases=len(cases), nominal_conditions=len(responses),
            sensitivity_queries=sum(len(values)-1 for values in grouped.values())),
        cases=list(cases.values()), representations=raw.get('requested_representations'),
        postures=raw.get('postures'), base_frame_reference=raw.get('base_frame_reference'),
        source_hashes=raw.get('source_hashes'), packet_bundle=raw.get('packet_bundle'),
        selected_union_comparisons=raw.get('selected_union_comparisons'),
        identical_geometry_negative_control=control_summary, empty_world_baselines=raw.get('empty_world_baselines'),
        responses=responses, incomplete_conditions=incomplete,
        interpretation=[
            'Nominal natural force and metric are policy terms, not physical Newtons, torques or inertia.',
            'Collision force/metric norms subtract the matched empty-world natural terms; these arrays can be reconstructed from nominal arrays and the per-posture empty baseline.',
            'Canonical joint acceleration and its sensitivity are rad/s², not measured robot acceleration or Cartesian direction.',
            'Every plotted line holds the actual captured field and recorded q/declared zero qdot/target fixed across representations.',
            'A changed selected union changes geometry; it cannot establish a repartitioning-only effect.',
            'Identical axis_best cuboids are an actual-data negative control, not a synthetic duplication experiment.',
            'Sensitivity is the largest measured output change across the same six ±1mm world translations, not a derivative or stability guarantee.',
            'Recorded postures are not claimed exposure-matched to occupancy; finite static outputs do not establish physical avoidance.'])


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    representations = report.get('representations') or sorted({row['representation'] for row in report['responses']})
    cases = {row['case_id']: row for row in report['cases']}
    postures = [row['id'] for row in report.get('postures') or []]
    colors = dict(zip(postures, ('#246b9d', '#b56c21', '#378169', '#8f5799')))
    markers = dict(zip(cases, ('o', 's', '^', 'D', 'v', 'P')))
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), constrained_layout=True)
    quantities = [('collision_force_norm', 'Collision natural term ‖Δf‖\n(policy units)', True),
        ('collision_metric_norm', 'Collision metric ‖ΔM‖F\n(policy units)', True),
        ('canonical_acceleration_norm', 'Canonical joint acceleration norm\n(rad/s²)', False),
        ('sensitivity', 'Maximum joint acceleration change, ±1 mm\n(rad/s²)', True)]
    grouped = defaultdict(dict)
    for row in report['responses']:
        grouped[(row['case_id'], row['posture'])][row['representation']] = row
    for axis, (field, label, logarithmic) in zip(axes.flat, quantities):
        for (case, posture), rows in grouped.items():
            indices, values = [], []
            for index, representation in enumerate(representations):
                if representation not in rows:
                    continue
                row = rows[representation]
                value = (row['offset_sensitivity']['canonical_acceleration_rad_s2']['maximum_delta_norm']
                    if field == 'sensitivity' else row['nominal_norms'].get(field))
                if value is not None and (not logarithmic or value > 0):
                    indices.append(index)
                    values.append(value)
            axis.plot(indices, values, color=colors.get(posture, '#777777'), marker=markers.get(case, 'o'),
                linewidth=.9, markersize=4, alpha=.65)
        axis.set_xticks(range(len(representations)), representations)
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.2)
        if logarithmic:
            axis.set_yscale('log')
        else:
            axis.set_ylim(bottom=0)
    legend = [Line2D([], [], color=colors[name], label=name, linewidth=2) for name in postures]
    for case, marker in markers.items():
        row = cases[case]
        layout = 'A' if row['layout_id'] == 'A_current' else 'C' if str(row['layout_id']).startswith('C_') else row['layout_id']
        legend.append(Line2D([], [], color='#666666', marker=marker, linestyle='none', label=layout+' '+str(row['scenario'])))
    figure.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, -.04),
        ncol=min(7, len(legend)), frameon=False, fontsize=8)
    figure.suptitle('Measured static public Lula responses: 4 captured fields × 3 recorded postures\n'
        'Identical baseline/axis_best geometry; diversity/motion change selected cells. No physical force units.', fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=175, bbox_inches='tight')
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args(argv)
    data = args.source.read_bytes()
    report = compact(json.loads(data), source=dict(path=str(args.source), sha256=hashlib.sha256(data).hexdigest()))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    # Repeated indentation dominates this matrix artifact without adding evidence.
    args.result.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n')
    if args.figure:
        plot(report, args.figure)
    print(json.dumps(dict(status=report['status'], bytes=args.result.stat().st_size, **report['counts'])))


if __name__ == '__main__':
    main()
