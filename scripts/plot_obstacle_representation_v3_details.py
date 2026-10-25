#!/usr/bin/env python3
"""Plot fixture-scale detail from saved V3 canonical evidence and exact packets.

This reducer starts no simulator and reruns no extraction. The native raw
archives are required; committed compact summaries alone cannot reconstruct
these measured cells. Geometry outlines are evaluation-only references.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('baseline', 'diversity', 'motion')
CASES = (('A_current', 'cylinder'), ('C_asymmetric_1.15_0.65', 'narrow'))
COLORS = dict(selected='#2079ad', omitted='#e99b38', edges='#134a70', truth='#646464')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        return ROOT/path
    marker = '/robot-skill-stack/'
    if marker in str(path):
        return ROOT/str(path).split(marker, 1)[1]
    return path


def binding(path):
    path = local_path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path))


def checked_binding(value, expected):
    result = binding(value)
    if expected is not None and result['sha256'] != expected:
        raise ValueError('Source SHA256 mismatch: '+str(value))
    return result


def edges(lower, upper):
    result = []
    for axis in range(3):
        others = [item for item in range(3) if item != axis]
        for ends in product((0, 1), repeat=2):
            first, last = np.array(lower, float), np.array(lower, float)
            for other, end in zip(others, ends):
                first[other] = last[other] = upper[other] if end else lower[other]
            last[axis] = upper[axis]
            result.append(np.array([first, last]))
    return result


def faces(lower, upper):
    points = np.array([[x, y, z] for x in (lower[0], upper[0])
                       for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    return [points[list(indices)] for indices in ((0, 1, 3, 2), (4, 6, 7, 5),
            (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))]


def physical_edges(objects, sources):
    result, all_vertices = [], []
    for obj in objects:
        transform = np.asarray(obj.get('world_from_body', np.eye(4)), float).copy()
        if 'world_from_body' not in obj:
            transform[:3, 3] = obj['center']
        if obj.get('kind') == 'cooked_convex':
            path = local_path(obj['cooked_npz'])
            sources.append(checked_binding(path, obj.get('cooked_sha256')))
            with np.load(path, allow_pickle=False) as archive:
                indices = sorted(int(key.removeprefix('vertices_')) for key in archive.files
                                 if key.startswith('vertices_'))
                for index in indices:
                    vertices = archive[f'vertices_{index}']@transform[:3, :3].T+transform[:3, 3]
                    all_vertices.append(vertices)
                    counts, values = archive[f'face_counts_{index}'], archive[f'face_indices_{index}']
                    unique, cursor = set(), 0
                    for count in counts:
                        polygon = values[cursor:cursor+int(count)]
                        cursor += int(count)
                        for first, last in zip(polygon, np.roll(polygon, -1)):
                            key = tuple(sorted((int(first), int(last))))
                            if key not in unique:
                                result.append(vertices[list(key)])
                                unique.add(key)
        else:
            if obj.get('kind', 'box') not in ('box', 'cube'):
                raise ValueError('Only actual cooked convexes or fixture boxes are supported')
            half = np.asarray(obj['dimensions'], float)/2
            local = np.array([[x, y, z] for x in (-half[0], half[0])
                              for y in (-half[1], half[1]) for z in (-half[2], half[2])])
            all_vertices.append(local@transform[:3, :3].T+transform[:3, 3])
            result.extend(segment@transform[:3, :3].T+transform[:3, 3]
                          for segment in edges(-half, half))
    if not all_vertices:
        raise ValueError('Actual fixture reference required to define this detail ROI')
    vertices = np.vstack(all_vertices)
    return result, vertices.min(0), vertices.max(0)


def select_case(report, records, packet_cases, layout, scenario):
    matching = [row for row in report['cases'] if row['layout_id'] == layout and row['scenario'] == scenario]
    if not matching:
        raise ValueError(f'No actual {layout}/{scenario} case in report')
    # Same final actual snapshot used by the evaluator's per-sequence summary.
    row = matching[-1]
    record = records[row['case_id']]
    packet = packet_cases[row['case_id']]
    if row['snapshot'] != record['snapshot'] or packet['snapshot'] != row['snapshot']:
        raise ValueError('Report, canonical manifest and packet clocks/specification differ')
    return row, record, packet


def checked_arrays(row, record, packet, policy, sources):
    value = row['variants'][policy]
    if value['status'] != 'PASS_EXACT_AUDIT' or value['budget'] != 64:
        raise ValueError('Plot requires an audited64-budget experimental result')
    path = local_path(value['cell_audit_npz'])
    sources.append(binding(path))
    with np.load(path, allow_pickle=False) as archive:
        eligible, supported, selected = (np.array(archive[name], dtype=bool) for name in
                                          ('eligible', 'supported', 'selected'))
        bounds = np.array(archive['selected_bounds'], dtype=np.int64)
    spec = row['snapshot']
    lower, upper, voxel = np.asarray(spec['bounds_min']), np.asarray(spec['bounds_max']), spec['voxel_size']
    recreated = np.zeros(eligible.shape, bool)
    overlap = 0
    boxes = packet['packets'][policy]['boxes']
    if len(boxes) != len(bounds):
        raise ValueError('Saved packet and selected bounds have different counts')
    for box, bound in zip(boxes, bounds):
        if tuple(map(int, box['id'].split(':'))) != tuple(bound):
            raise ValueError('Saved packet IDs do not identify displayed exact bounds')
        lo, hi = lower+bound[[0, 2, 4]]*voxel, np.minimum(upper, lower+bound[[1, 3, 5]]*voxel)
        if not np.allclose((lo+hi)/2, box['center'], rtol=0, atol=1e-12) or not np.allclose(
                hi-lo, box['dimensions'], rtol=0, atol=1e-12):
            raise ValueError('Saved packet geometry differs from displayed cell edges')
        slices = tuple(slice(int(bound[2*a]), int(bound[2*a+1])) for a in range(3))
        overlap += int(recreated[slices].sum())
        recreated[slices] = True
    if overlap or not np.array_equal(recreated, selected) or np.any(selected & ~supported) or np.any(supported & ~eligible):
        raise ValueError('Displayed packet is not an exact disjoint subset of saved support')
    source = local_path(record['snapshot_npz'])
    sources.append(checked_binding(source, record['source_artifacts']['arrays_sha256']))
    with np.load(source, allow_pickle=False) as archive:
        states = archive['states']
        now, ttl = spec['evaluated_at'], spec['evidence_ttl']
        observed_at = archive['last_observed']
        original_eligible = ((states == 2) & archive['observed'] & ~archive['expired'] &
            np.isfinite(observed_at) & (observed_at <= now) & (now-observed_at < ttl))
        if not np.array_equal(original_eligible, eligible):
            raise ValueError('Displayed eligibility differs from canonical arrays at recorded extraction time')
        audit = dict(incorrect_cells_covered=int((selected & ~supported).sum()),
            known_free_cells_covered=int((selected & (states == 1)).sum()),
            unknown_cells_covered=int((selected & (states == 0)).sum()), overlap_cells=overlap)
    if any(audit.values()):
        raise ValueError('Displayed exact-cell invariant failed: '+repr(audit))
    return value, eligible, supported, selected, bounds, audit


def draw_cells(ax, mask, lower, upper, voxel, roi_lower, roi_upper, *, color, alpha):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cells = np.argwhere(mask)
    centers = lower+(cells+.5)*voxel
    cells = cells[np.all((centers >= roi_lower) & (centers <= roi_upper), axis=1)]
    polygons = []
    for cell in cells:
        lo, hi = lower+cell*voxel, np.minimum(upper, lower+(cell+1)*voxel)
        polygons.extend(faces(lo, hi))
    if polygons:
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=color, edgecolors=color,
                                            linewidths=.35, alpha=alpha, zsort='average'))
    return len(cells)


def make_details(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    report_path, manifest_path, packet_path = map(local_path, (args.report, args.manifest, args.packet_bundle))
    report = json.loads(report_path.read_text())
    records = {row['case_id']: row for row in json.loads(manifest_path.read_text())['records']}
    packet_cases = {row['case_id']: row for row in json.loads(packet_path.read_text())['cases']}
    sources = [binding(path) for path in (report_path, manifest_path, packet_path, Path(__file__))]
    evidence = dict(schema_version=1, kind='actual_canonical_snapshot_fixture_scale_plot', panels=[],
        source_files=sources, selection='Last actual canonical snapshot for A_current/cylinder and C_asymmetric/narrow; baseline/diversity/motion at64.',
        qualifications=['No extraction or native simulation is rerun.',
            'Filled cubes are existing supported occupied grid cells; orange is omitted support, blue is selected support.',
            'Dark blue edges are the exact saved selected packet cuboids, with no padding or gap bridging.',
            'Dashed physical cooked-prism/box outlines define only the evaluation display ROI and enter no policy.',
            'Known-FREE gap count concerns only known-FREE whole cells in the recorded gap audit; UNKNOWN is not inferred FREE.',
            'Raw cell coordinates remain in ignored native/offline archives; this companion contains only counts and SHA bindings.'])
    fig = plt.figure(figsize=(15.2, 11.2))
    # Explicit margins account for projected3D axis labels, whose extents are
    # not fully reserved by constrained layout. Leave a clear inter-row band.
    fig.subplots_adjust(left=.045, right=.94, bottom=.11, top=.91,
                        wspace=.22, hspace=.32)
    for row_index, (layout, scenario) in enumerate(CASES):
        row, record, packet = select_case(report, records, packet_cases, layout, scenario)
        actual_edges, physical_lower, physical_upper = physical_edges(record['evaluation_objects'], sources)
        margin = .04 if scenario == 'cylinder' else .025
        roi_lower, roi_upper = physical_lower-margin, physical_upper+margin
        spec = row['snapshot']
        lower, upper, voxel = np.asarray(spec['bounds_min']), np.asarray(spec['bounds_max']), spec['voxel_size']
        for column, policy in enumerate(POLICIES):
            value, eligible, supported, selected, bounds, audit = checked_arrays(row, record, packet, policy, sources)
            ax = fig.add_subplot(2, 3, row_index*3+column+1, projection='3d')
            omitted = draw_cells(ax, supported & ~selected, lower, upper, voxel, roi_lower, roi_upper,
                                 color=COLORS['omitted'], alpha=.28)
            kept = draw_cells(ax, selected, lower, upper, voxel, roi_lower, roi_upper,
                              color=COLORS['selected'], alpha=.60)
            segments = []
            for bound in bounds:
                lo, hi = lower+bound[[0, 2, 4]]*voxel, np.minimum(upper, lower+bound[[1, 3, 5]]*voxel)
                if np.all(hi >= roi_lower) and np.all(lo <= roi_upper):
                    segments.extend(edges(lo, hi))
            if segments:
                ax.add_collection3d(Line3DCollection(segments, colors=COLORS['edges'], linewidths=.9))
            if not args.hide_reference:
                ax.add_collection3d(Line3DCollection(actual_edges, colors=COLORS['truth'], linewidths=.7,
                                                    linestyles='dashed', alpha=.75))
            gap_regions = value['gap_preservation']['regions']
            gap_covered = sum(region['known_free_cells_covered'] for region in gap_regions)
            if gap_covered:
                raise ValueError('Known-FREE gap cells were covered')
            gap_text = f'Known-FREE gap covered: {gap_covered}' if gap_regions else 'Gap audit: N/A (one fixture)'
            label = 'A · actual cooked cylinder' if scenario == 'cylinder' else 'C · two-box narrow fixture'
            ax.set_title(f'{label}\n{policy.capitalize()}64 · ROI selected {kept}, omitted {omitted}', fontsize=10, pad=4)
            ax.text2D(.02, .97, f'Incorrect covered: {audit["incorrect_cells_covered"]}\nKnown-FREE covered: {audit["known_free_cells_covered"]}\n{gap_text}',
                      transform=ax.transAxes, va='top', fontsize=8,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
            ax.set(xlim=tuple((roi_lower[0], roi_upper[0])), ylim=tuple((roi_lower[1], roi_upper[1])),
                   zlim=tuple((roi_lower[2], roi_upper[2])), xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
            ax.set_box_aspect(roi_upper-roi_lower)
            ax.view_init(elev=22, azim=-60 if scenario == 'cylinder' else -65)
            ax.tick_params(labelsize=7, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.label.set_size(8)
                axis.labelpad = 1.5
                axis.set_major_locator(MaxNLocator(4))
            evidence['panels'].append(dict(case_id=row['case_id'], layout_id=layout, scenario=scenario,
                policy=policy, budget=64, global_selected_cuboids=len(bounds),
                roi_supported_cells=kept+omitted, roi_selected_cells=kept, roi_omitted_supported_cells=omitted,
                invariants=audit, gap_known_free_cells_covered=gap_covered if gap_regions else None,
                gap_known_free_cells=sum(region['known_free_cells'] for region in gap_regions) if gap_regions else None,
                gap_unknown_cells=sum(region['unknown_cells'] for region in gap_regions) if gap_regions else None,
                native_depth_attributed_objects=value['physical_objects'],
                original_snapshot_timestamp=spec['timestamp'], snapshot_evaluated_at=spec['snapshot_evaluated_at'],
                extracted_at=spec['evaluated_at'], snapshot_arrays_sha256=row['snapshot_arrays_sha256']))
    fig.suptitle('Measured occupied evidence at fixture scale · actual1cm cells and exact selected cuboids', fontsize=14)
    handles = [Patch(facecolor=COLORS['selected'], alpha=.6, label='Selected supported cell'),
               Patch(facecolor=COLORS['omitted'], alpha=.35, label='Omitted supported cell'),
               Line2D([], [], color=COLORS['edges'], linewidth=1, label='Saved selected cuboid edges')]
    if not args.hide_reference:
        handles.append(Line2D([], [], color=COLORS['truth'], linestyle='--', label='Evaluation-only physical outline'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncols=2, fontsize=9)
    output, summary = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    # Source paths repeat across panels; bind each file exactly once.
    evidence['source_files'] = list({source['path']: source for source in sources}.values())
    evidence['figure'] = binding(output)
    summary.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), summary=str(summary), panels=len(evidence['panels']))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='outputs/obstacle-representation-v3/primary-offline/full_report.json')
    parser.add_argument('--manifest', default='outputs/obstacle-representation-v3/primary-final-snapshots.json')
    parser.add_argument('--packet-bundle', default='outputs/obstacle-representation-v3/primary-offline/packet_bundle.json')
    parser.add_argument('--output', default='docs/assets/obstacle_representation_v3/fixture_scale_details.png')
    parser.add_argument('--summary', default='docs/assets/obstacle_representation_v3/fixture_scale_details.json')
    parser.add_argument('--hide-reference', action='store_true')
    make_details(parser.parse_args(argv))


if __name__ == '__main__':
    main()
