"""Compare geometric/refinement alternatives on saved clouds; truth scores only.

Run through run_isaac_python.sh. Use --capture-manifest for native selection
snapshots, or --saved-root/--singles-root for translated original cloud pairs.
All outputs remain offline replay diagnostics, never native throughput claims.
"""
from itertools import product
import argparse
import json
import importlib
from pathlib import Path
import sys
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.segmentation import (
    cluster_points, supported_object_cloud, refine_cluster,
)
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter


def load_ply(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() == 'element vertex 0':
                return np.empty((0, 3))
            if line.strip() == 'end_header':
                break
        return np.loadtxt(stream).reshape(-1, 3)


def radius_graph(points, radius=.004, normal_gate=None, adaptive=False, plane_gate=None):
    """Exact Euclidean connectivity; hash grid generates candidate edges only."""
    points = np.asarray(points, float)
    if not len(points):
        return []
    cells, inverse = np.unique(np.floor(points/radius).astype(int), axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse)
    starts = np.r_[0, counts.cumsum()]
    buckets = {tuple(cell): order[starts[i]:starts[i+1]] for i, cell in enumerate(cells)}
    parent = np.arange(len(points))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    normals = None
    spacing = None
    if normal_gate is not None or adaptive or plane_gate is not None:
        # Brute force neighborhood here is deliberate: separate accuracy probe
        # from the efficient spatial index used for production candidate edges.
        nearest = []
        for begin in range(0, len(points), 256):
            distances = ((points[begin:begin+256, None]-points[None])**2).sum(2)
            distances[np.arange(len(distances)), np.arange(begin, begin+len(distances))] = np.inf
            nearest.append(np.argsort(distances, axis=1)[:, :10])
        near = np.concatenate(nearest)
        spacing = np.linalg.norm(points-points[near[:, 0]], axis=1)
        if normal_gate is not None or plane_gate is not None:
            local = points[near]-points[:, None]
            covariance = np.einsum('nki,nkj->nij', local, local)
            _, vectors = np.linalg.eigh(covariance)
            normals = vectors[:, :, 0]
    half_offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset >= (0, 0, 0)]
    for cell, first in buckets.items():
        for offset in half_offsets:
            second = buckets.get(tuple(a+b for a, b in zip(cell, offset)))
            if second is None:
                continue
            distance = ((points[first, None]-points[second])**2).sum(2)
            allowed = distance <= radius*radius
            if adaptive:
                limits = np.minimum(radius, 1.8*np.maximum(spacing[first, None], spacing[second]))
                allowed &= distance <= limits**2
            if normal_gate is not None:
                allowed &= abs(normals[first] @ normals[second].T) >= np.cos(np.deg2rad(normal_gate))
            if plane_gate is not None:
                delta = points[first, None]-points[second]
                offsets_first = abs(np.einsum('abk,ak->ab', delta, normals[first]))
                offsets_second = abs(np.einsum('abk,bk->ab', delta, normals[second]))
                allowed &= np.maximum(offsets_first, offsets_second) <= plane_gate
            x, y = np.nonzero(allowed)
            for left, right in zip(first[x], second[y]):
                if left >= right and offset == (0, 0, 0):
                    continue
                left_root, right_root = root(left), root(right)
                if left_root != right_root:
                    parent[right_root] = left_root
    labels = np.array([root(i) for i in range(len(points))])
    return [np.flatnonzero(labels == value) for value in np.unique(labels)]


def voxel_graph(points, cell_size):
    # Matching by exact input rows preserves oracle labels without runtime truth.
    index = {tuple(point): i for i, point in enumerate(points)}
    return [np.array([index[tuple(point)] for point in cloud])
            for cloud in cluster_points(points, cell_size=cell_size, min_points=1)]


def score(groups, labels, min_points=25, expected_count=None):
    accepted = [group for group in groups if len(group) >= min_points]
    expected = np.unique(labels[labels >= 0]) if expected_count is None else np.arange(expected_count)
    split = sum(sum(np.count_nonzero(labels[group] == label) >= min_points for group in accepted) > 1
                for label in expected)
    missing = sum(not any(np.count_nonzero(labels[group] == label) >= min_points for group in accepted)
                  for label in expected)
    mixed = sum(sum(np.count_nonzero(labels[group] == label) >= max(5, .10*len(group))
                    for label in expected) > 1 for group in accepted)
    return dict(components=len(accepted), mixed=int(mixed), false_splits=int(split), missing=int(missing),
                discarded_points=sum(len(group) for group in groups if len(group) < min_points))


def model_guided(points, fitter, *, keep_known=True):
    """Research candidate: let consistent whole-object evidence veto false cuts."""
    result = []
    for broad in voxel_graph(points, .008):
        if len(broad) < 25:
            result.append(broad)
            continue
        fine = [group for group in radius_graph(points[broad], .004) if len(group) >= 25]
        if len(fine) < 2:
            result.append(broad)
            continue
        parent_supported, parent_reason = supported_object_cloud(points[broad], 0.)
        whole = None if parent_reason == 'oversized_or_robot_connected' else fitter.fit(points[broad])
        # Avoid expensive split trials for clear whole primitives. A caveat is
        # merged surfaces accidentally explainable by a single supported model.
        if keep_known and whole is not None and whole.geometry.shape.value != 'unknown':
            result.append(broad)
            continue
        fitted = [(group, fitter.fit(points[broad[group]])) for group in fine
                  if supported_object_cloud(points[broad[group]], 0.)[1] != 'oversized_or_robot_connected']
        valid = [(group, fit) for group, fit in fitted
                 if fit.geometry.shape.value != 'unknown' and abs(fit.center[2]-fit.size[2]/2) <= .004]
        supported_coverage = sum(len(group) for group, _ in valid)/len(broad)
        dense_coverage = sum(len(group) for group in fine)/len(broad)
        oversized = [group for group in fine if supported_object_cloud(points[broad[group]], 0.)[1]
                     == 'oversized_or_robot_connected']
        split_supported = len(valid) >= 2 and supported_coverage >= .85
        split_clutter = (parent_reason == 'oversized_or_robot_connected' and valid and oversized
                         and dense_coverage >= .85)
        if not (split_supported or split_clutter):
            result.append(broad)
        else:
            # This prototype leaves discarded fragments explicit. Production
            # would require evidence-based assignment and native validation.
            result.extend(broad[group] for group, _ in valid)
            result.extend(broad[group] for group in oversized)
    return result


def selected_pipeline(points, fitter):
    index = {tuple(point): i for i, point in enumerate(points)}
    result = []
    for broad in cluster_points(points):
        for cloud, _ in refine_cluster(broad, fitter):
            result.append(np.array([index[tuple(point)] for point in cloud]))
    return result


def scenarios(args):
    if args.capture_manifest:
        from tests.support.multiview_iteration2 import object_from_payload, surface_memberships
        manifest = json.loads(args.capture_manifest.read_text())
        for record in manifest['scenes']:
            if record['split'] != 'selection':
                continue
            snapshot = record['frames'][-1]
            directory = args.capture_manifest.parent/'captures'/record['key']
            with np.load(directory/f"workspace_frame_{snapshot['index']:02d}.npz") as data:
                points = data['fused']
            objects = [object_from_payload(payload) for payload in snapshot['objects']]
            labels = surface_memberships(points, objects)
            # Truth is scoring-only; exclude unassigned background in the label
            # denominator while retaining all points for every segmentation call.
            yield record['key'], points, labels, 'native_selection_workspace', len(objects)
        return
    cloud_root = args.saved_root
    kinds = ('box', 'sphere', 'upright', 'lying')
    objects = {kind: load_ply(cloud_root/f'0_{kind}_0_0_surface_fused.ply') for kind in kinds}
    singles = {path.name: load_ply(path) for path in sorted(args.singles_root.glob('*.ply'))}
    for name, points in singles.items():
        yield name, points, np.zeros(len(points), int), 'native_accepted_single', 1
    for pair in (('box', 'upright'), ('sphere', 'sphere'), ('box', 'box'), ('lying', 'upright')):
        for gap in (.002, .004, .006, .010):
            for density in (1, 2):
                left, right = [objects[kind][::density].copy() for kind in pair]
                left[:, :2] -= np.mean(left[:, :2], axis=0)
                right[:, :2] -= np.mean(right[:, :2], axis=0)
                right[:, 0] += left[:, 0].max()-right[:, 0].min()+gap
                points = np.r_[left, right]
                labels = np.r_[np.zeros(len(left), int), np.ones(len(right), int)]
                yield f'{pair[0]}_{pair[1]}_{gap*1000:g}mm_density{density}', points, labels, 'translated_native_clouds', 2


def main(args):
    if args.fitter_package_dir:
        sys.path.insert(0, str(args.fitter_package_dir.resolve()))
        fitter = importlib.import_module('baseline_multiview.fitting').SurfaceFitter()
    else:
        fitter = SurfaceFitter()
    methods = [('voxel8', lambda p: voxel_graph(p, .008)),
               ('voxel3', lambda p: voxel_graph(p, .003)),
               ('radius3', lambda p: radius_graph(p, .003)),
               ('radius4', lambda p: radius_graph(p, .004)),
               ('radius5', lambda p: radius_graph(p, .005)),
               ('adaptive5', lambda p: radius_graph(p, .005, adaptive=True)),
               ('normal5_45', lambda p: radius_graph(p, .005, normal_gate=45)),
               ('tangent8_1_2', lambda p: radius_graph(p, .008, plane_gate=.0012)),
               ('model4', lambda p: model_guided(p, fitter)),
               ('model4all', lambda p: model_guided(p, fitter, keep_known=False)),
               ('selected', lambda p: selected_pipeline(p, fitter))]
    if args.methods:
        names = set(args.methods.split(','))
        unknown = names-{name for name, _ in methods}
        if unknown:
            raise ValueError(f'Unknown methods: {sorted(unknown)}')
        methods = [(name, method) for name, method in methods if name in names]
    rows = []
    for name, points, labels, source, expected_count in scenarios(args):
        for method, function in methods:
            if args.quick and method in ('adaptive5', 'normal5_45', 'tangent8_1_2'):
                continue
            started = time.perf_counter()
            groups = function(points)
            rows.append(dict(scenario=name, method=method, source=source, points=len(points),
                             expected_objects=expected_count, visible_truth_points=int(np.count_nonzero(labels >= 0)),
                             invalid_empty_fixture=bool(expected_count and not np.any(labels >= 0)),
                             elapsed_ms=1000*(time.perf_counter()-started),
                             **score(groups, labels, expected_count=expected_count)))
        print(name, [(row['method'], row['mixed'], row['false_splits'], row['missing'])
                     for row in rows[-len(methods):]], flush=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(dict(rows=rows,
        methods=[name for name, _ in methods],
        parameters=dict(broad_cell_m=.008, metric_radius_m=.004, accepted_component_min_points=25,
                        production_partition_coverage=.85),
        capture_manifest=None if args.capture_manifest is None else str(args.capture_manifest),
        baseline_fitter_package=None if args.fitter_package_dir is None else str(args.fitter_package_dir),
        invalid_empty_fixtures=sorted({row['scenario'] for row in rows if row['invalid_empty_fixture']}),
        limitation='Offline replay, no live Isaac throughput; translated isolated clouds omit mutual occlusion'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--methods', help='Comma-separated method names (default: all)')
    parser.add_argument('--fitter-package-dir', type=Path,
                        help='Optional frozen baseline package parent, for controlled segmentation comparisons')
    parser.add_argument('--capture-manifest', type=Path,
                        help='Evaluate final frame of saved native selection scenes; holdout is never selected')
    parser.add_argument('--saved-root', type=Path, default=Path('outputs/multiview-geometry-final/clouds'))
    parser.add_argument('--singles-root', type=Path, default=Path('outputs/multiview-detection-01/clouds'))
    parser.add_argument('--result', type=Path, required=True)
    main(parser.parse_args())
