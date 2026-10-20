"""Compare frozen segmentation and independent alternatives on selection data.

No truth enters an algorithm. Ground truth scores saved current observations.
All timings are offline diagnostics; live Isaac throughput is measured elsewhere.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from robot_skill_stack.world.perception_multiview.segmentation import (
    cluster_points, _metric_components, supported_object_cloud, refine_cluster,
)
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from scripts.probe_multiview_segmentation import score
from scripts.probe_multiview_segmentation import load_ply
from tests.support.multiview_iteration2 import object_from_payload, surface_memberships, score_observations
from scripts.probe_multiview_readiness_views import gap_witness, load_frames, organized_segments


baseline_refine = refine_cluster


def residual(points, fit, *, signed=False):
    g = fit.geometry
    delta = points-fit.center
    if g.shape.value == 'sphere':
        value = np.linalg.norm(delta, axis=1)-g.radius
        return value if signed else abs(value)
    if g.shape.value == 'cube':
        c, s = np.cos(g.yaw), np.sin(g.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        q = abs(delta @ rotation)-g.box_size/2
    elif g.shape.value == 'cylinder':
        along = delta @ g.axis
        q = np.c_[np.linalg.norm(delta-along[:, None]*g.axis, axis=1)-g.radius,
                  abs(along)-g.length/2]
    else:
        return np.full(len(points), np.inf)
    value = np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0)
    return value if signed else abs(value)


def footprint(fit):
    g = fit.geometry
    if g.shape.value == 'sphere' or (g.shape.value == 'cylinder' and abs(g.axis[2]) > .99):
        angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
        # Circumscribed polygon gives conservative overlap evidence.
        return fit.center[:2]+g.radius/np.cos(np.pi/32)*np.c_[np.cos(angles), np.sin(angles)]
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if g.shape.value == 'cube':
        c, s = np.cos(g.yaw), np.sin(g.yaw)
        rotation = np.array([[c, -s], [s, c]])
        local = corners*g.box_size[:2]/2
    else:
        direction = g.axis[:2]/np.linalg.norm(g.axis[:2])
        rotation = np.c_[direction, [-direction[1], direction[0]]]
        local = corners*np.array([g.length/2, g.radius])
    return fit.center[:2]+local @ rotation.T


def overlap(first, second, tolerance=.001):
    a, b = footprint(first), footprint(second)
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0)-polygon
        normals = np.c_[-edges[:, 1], edges[:, 0]]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        pa, pb = a @ normals.T, b @ normals.T
        separation = np.maximum(pa.min(0)-pb.max(0), pb.min(0)-pa.max(0))
        if np.any(separation >= -tolerance):
            return False
    return True


def supported(cloud, fit, plane=0.):
    good, reason = supported_object_cloud(cloud, plane)
    return good or (reason == 'no_observed_support_contact' and fit.geometry.shape.value in ('sphere', 'cylinder')
                    and abs(fit.center[2]-fit.size[2]/2-plane) <= .004)


def assign_nearest(points, fine):
    distances = []
    for seed in fine:
        nearest = []
        for begin in range(0, len(points), 128):
            nearest.append(np.sqrt(((points[begin:begin+128, None]-seed[None])**2).sum(2).min(1)))
        distances.append(np.concatenate(nearest))
    distances = np.column_stack(distances)
    labels = distances.argmin(1)
    best = distances[np.arange(len(points)), labels]
    labels[best > .008] = -1
    # Keep the distinct measured dense seeds; only orphan points are assigned.
    index = {tuple(point): i for i, point in enumerate(points)}
    for label, seed in enumerate(fine):
        labels[[index[tuple(point)] for point in seed]] = label
    return labels


def augment(points, fitter, strategy, diagnostic, *, seeds=None):
    fine = _metric_components(points) if seeds is None else seeds
    diagnostic.update(parent_points=len(points), dense_seed_points=[len(cloud) for cloud in fine],
                      dense_seed_coverage=sum(map(len, fine))/max(len(points), 1))
    if not 2 <= len(fine) <= 8:
        diagnostic['rejected'] = 'seed_count'
        return baseline_refine(points, fitter)
    if not supported_object_cloud(points, 0.)[0]:
        diagnostic['rejected'] = 'retain_background_baseline'
        return baseline_refine(points, fitter)
    fine = [cloud for cloud in fine if len(cloud) >= 30 and
            supported_object_cloud(cloud, 0.)[1] not in ('too_small', 'oversized_or_robot_connected')]
    if len(fine) < 2 or (seeds is None and sum(map(len, fine)) < .60*len(points)):
        diagnostic['rejected'] = 'weak_seed_evidence'
        return baseline_refine(points, fitter)
    if strategy == 'nearest':
        labels = assign_nearest(points, fine)
    else:
        initial = [fitter.fit(cloud) for cloud in fine]
        diagnostic['seed_fits'] = [payload(fit, len(cloud)) for cloud, fit in zip(fine, initial)]
        if any(fit.geometry.shape.value == 'unknown' or not supported(cloud, fit)
               for cloud, fit in zip(fine, initial)):
            diagnostic['rejected'] = 'seed_model_unknown'
            return baseline_refine(points, fitter)
        if any(overlap(first, second) for index, first in enumerate(initial) for second in initial[index+1:]):
            diagnostic['rejected'] = 'seed_model_overlap'
            return baseline_refine(points, fitter)
        distances = np.column_stack([residual(points, fit) for fit in initial])
        labels = distances.argmin(1)
        ranked = np.sort(distances, axis=1)
        keep = (ranked[:, 0] <= .003) & (ranked[:, 1]-ranked[:, 0] >= .0005)
        labels[~keep] = -1
    clouds = [points[labels == index] for index in range(len(fine))]
    diagnostic['assigned_points'] = list(map(len, clouds))
    diagnostic['unassigned_points'] = int(np.count_nonzero(labels < 0))
    if min(map(len, clouds)) < 30 or sum(map(len, clouds)) < .90*len(points):
        diagnostic['rejected'] = 'assignment_coverage'
        return baseline_refine(points, fitter)
    fits = [fitter.fit(cloud) for cloud in clouds]
    diagnostic['final_fits'] = [payload(fit, len(cloud)) for cloud, fit in zip(clouds, fits)]
    if any(fit.geometry.shape.value == 'unknown' or not supported(cloud, fit)
           for cloud, fit in zip(clouds, fits)):
        diagnostic['rejected'] = 'final_model_unknown'
        return baseline_refine(points, fitter)
    if any(overlap(first, second) for index, first in enumerate(fits) for second in fits[index+1:]):
        diagnostic['rejected'] = 'final_model_overlap'
        return baseline_refine(points, fitter)
    if any(np.percentile(residual(cloud, fit), 95) > .003 for cloud, fit in zip(clouds, fits)):
        diagnostic['rejected'] = 'final_model_residual'
        return baseline_refine(points, fitter)
    diagnostic['accepted'] = True
    return list(zip(clouds, fits))


def guard_single_explanation(points, fitter, partitions, frames, diagnostic):
    """A comparable single solid needs measured gap evidence before a cut."""
    children = [fit for _, fit in partitions if fit is not None and fit.geometry.shape.value != 'unknown']
    if len(children) < 2:
        return partitions
    parent = fitter.fit(points)
    if parent.geometry.shape.value == 'unknown' or np.mean(residual(points, parent) <= .003) < .95:
        return partitions
    witness = gap_witness(parent, children, frames, footprint, residual)
    diagnostic['single_explanation'] = payload(parent, len(points))
    diagnostic['view_gap_evidence'] = witness
    if witness['physical_gap_supported']:
        return partitions
    if witness['occupied_bridge_supported']:
        return [(points, parent)]
    unknown = PrimitiveGeometry('unknown', parent.geometry.confidence,
        scores=dict(parent.geometry.scores), metadata={'reason': 'ambiguous_single_or_multiple_surfaces',
                                                      'single_model_residual_p80_m': parent.geometry.metadata.get('residual_p80_m')})
    return [(points, PrimitiveFit(unknown, parent.center, parent.size))]


def hybrid(points, fitter, frames, scene, diagnostic):
    lo, hi = points.min(0)-.004, points.max(0)+.004
    selected = None
    best = 0
    for frame in frames:
        proposals = [cloud[((cloud >= lo) & (cloud <= hi)).all(1)] for cloud in organized_segments(frame, scene)]
        proposals = [cloud for cloud in proposals if len(cloud) >= 30]
        evidence = {}
        partitions = augment(points, fitter, 'surface', evidence, seeds=proposals)
        diagnostic.setdefault('camera_proposals', []).append(dict(camera=frame.camera_id, evidence=evidence))
        if evidence.get('accepted'):
            count = sum(len(cloud) for cloud, _ in partitions)
            if count > best:
                selected, best = partitions, count
    return baseline_refine(points, fitter) if selected is None else selected


def payload(fit, count):
    g = fit.geometry
    return dict(points=count, shape=g.shape.value, confidence=g.confidence, center=fit.center.tolist(),
                radius=g.radius, size=fit.size.tolist(), scores=g.scores,
                residual_p80_m=g.metadata.get('residual_p80_m'))


def controls(args, frozen_segmentation, frozen_fitting):
    """Reproduce intact/missing-band controls and deterministic work bounds."""
    fitter = frozen_fitting.SurfaceFitter()
    rows, counters = [], []
    for kind in ('sphere', 'lying', 'box'):
        original = load_ply(args.geometry/f'0_{kind}_0_0_surface_fused.ply')
        for variant in ('intact', 'missing_band'):
            middle = (original[:, 0].min()+original[:, 0].max())/2
            points = original if variant == 'intact' else original[abs(original[:, 0]-middle) > .004]
            for method in args.methods:
                if method not in ('baseline', 'retained', 'nearest', 'surface'):
                    continue
                diagnostic = {}
                if method == 'baseline':
                    partitions = baseline_refine(points, fitter)
                elif method == 'retained':
                    partitions = refine_cluster(points, fitter)
                else:
                    partitions = augment(points, fitter, method, diagnostic)
                final = [payload(fitter.fit(cloud) if fit is None else fit, len(cloud)) for cloud, fit in partitions]
                row = dict(object=kind, variant=variant, method=method, physical_objects=1,
                           known_components=sum(fit['shape'] != 'unknown' for fit in final),
                           partitions=len(partitions), fits=final, diagnostics=diagnostic)
                rows.append(row)
                if args.artifacts_dir:
                    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.artifacts_dir/f'{kind}_{variant}_{method}.npz', points=points,
                        **{f'partition_{i}': cloud for i, (cloud, _) in enumerate(partitions)})
    class Counter:
        def __init__(self, limit=None): self.calls, self.limit = 0, limit
        @property
        def remaining(self): return 10000 if self.limit is None else self.limit-self.calls
        def fit(self, points, **kwargs):
            if self.limit is not None and self.calls >= self.limit:
                raise RuntimeError('Numerical budget exceeded in counter experiment')
            self.calls += 1
            lo, hi = points.min(0), points.max(0)
            return PrimitiveFit(PrimitiveGeometry('unknown', 0.), (lo+hi)/2, hi-lo)
    for count in (8, 16):
        x, z = np.meshgrid(np.linspace(-.007, .007, 7), np.linspace(.025, .043, 6))
        points = np.concatenate([np.c_[x.ravel()+.208+.032*parent,
            np.full(x.size, .007*fragment), z.ravel()] for parent in range(16) for fragment in range(count)])
        for method, segmentation in (('baseline', frozen_segmentation), ('retained', None)):
            counter = Counter(None if method == 'baseline' else 24)
            grouping = frozen_segmentation.cluster_points if segmentation else cluster_points
            refine = baseline_refine if segmentation else refine_cluster
            check = frozen_segmentation.supported_object_cloud if segmentation else supported_object_cloud
            broad = grouping(points)
            for cloud in broad:
                for child, fit in refine(cloud, counter):
                    eligible, reason = check(child, 0.)
                    if fit is None and (eligible or reason == 'no_observed_support_contact'):
                        counter.fit(child)
            counters.append(dict(method=method, fine_planes_per_parent=count, broad_groups=len(broad),
                                 points=len(points), numerical_fit_calls=counter.calls))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(kind='readiness_segmentation_controls', rows=rows,
        numerical_work_counters=counters,
        limitation='Missing-band data have no raw-depth gap evidence. Counter fitter always returns unknown; counts are not latency measurements.'), indent=2))
    print(json.dumps(dict(controls=[{key: row[key] for key in ('object', 'variant', 'method', 'known_components')}
                                  for row in rows], counters=counters), indent=2))


def main(args):
    global baseline_refine
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    frozen_segmentation = importlib.import_module('baseline_multiview.segmentation')
    frozen_fitting = importlib.import_module('baseline_multiview.fitting')
    baseline_refine = frozen_segmentation.refine_cluster
    if args.controls:
        controls(args, frozen_segmentation, frozen_fitting)
        return
    manifest = json.loads(args.manifest.read_text())
    # Hold fitter constant to isolate segmentation changes. Full combined
    # provider accuracy and source contracts are separate benchmark runs.
    fitter = frozen_fitting.SurfaceFitter()
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    scene_config = load_scene_config('config/scenes/multiview_playground.toml').scene
    rows = []
    for scene in manifest['scenes']:
        if scene['split'] != 'selection' or (args.scenario and not any(word in scene['name'] for word in args.scenario)):
            continue
        snapshot = scene['frames'][-1]
        directory = args.manifest.parent/'captures'/scene['key']
        frames = load_frames(directory, snapshot['index'])
        with np.load(directory/f"workspace_frame_{snapshot['index']:02d}.npz") as data:
            points = data['fused']
        objects = [object_from_payload(value) for value in snapshot['objects']]
        truth_labels = surface_memberships(points, objects)
        index = {tuple(point): i for i, point in enumerate(points)}
        for method in args.methods:
            groups, fits, fit_objects, diagnostics = [], [], [], []
            started = time.perf_counter()
            for broad in cluster_points(points):
                diagnostic = {}
                base_method = method.removesuffix('_guard')
                if base_method == 'baseline':
                    partitions = baseline_refine(broad, fitter)
                elif base_method == 'retained':
                    partitions = refine_cluster(broad, fitter, frames=frames)
                elif base_method == 'hybrid':
                    partitions = hybrid(broad, fitter, frames, scene_config, diagnostic)
                else:
                    partitions = augment(broad, fitter, base_method, diagnostic)
                if method.endswith('_guard'):
                    partitions = guard_single_explanation(broad, fitter, partitions, frames, diagnostic)
                support_check = frozen_segmentation.supported_object_cloud if base_method == 'baseline' else supported_object_cloud
                for cloud, fit in partitions:
                    eligible, reason = support_check(cloud, 0.)
                    if not eligible and reason != 'no_observed_support_contact':
                        continue
                    fit = fitter.fit(cloud) if fit is None else fit
                    if not eligible and not supported(cloud, fit):
                        continue
                    groups.append(np.array([index[tuple(point)] for point in cloud]))
                    fits.append(payload(fit, len(cloud)))
                    fit_objects.append(fit)
                diagnostics.append(diagnostic)
            observations = [SimpleNamespace(object_id=f'component_{i}', geometry=fit.geometry,
                pose=SimpleNamespace(position=fit.center), size=fit.size, metadata={})
                for i, fit in enumerate(fit_objects)]
            object_rows, counts, _ = score_observations(objects, observations, [points[group] for group in groups],
                method=method, trial=scene['key'], frame_index=snapshot['index'])
            known_mixed = 0
            for group, fit in zip(groups, fit_objects):
                support = np.array([np.count_nonzero(truth_labels[group] == i) for i in range(len(objects))])
                known_mixed += (fit.geometry.shape.value != 'unknown' and
                                np.count_nonzero((support >= 8) & (support >= .10*len(group))) > 1)
            row = dict(scene=scene['key'], method=method, points=len(points),
                       **score(groups, truth_labels, expected_count=len(objects)),
                       expected=len(objects), correct=sum(entry['correct'] for entry in object_rows),
                       known_mixed=int(known_mixed), observation_counts=counts, objects=object_rows,
                       offline_diagnostic_ms=1000*(time.perf_counter()-started), fits=fits, diagnostics=diagnostics)
            if args.artifacts_dir:
                export = args.artifacts_dir/scene['key']
                export.mkdir(parents=True, exist_ok=True)
                arrays = dict(fused=points, **{f'partition_{i}': points[group] for i, group in enumerate(groups)})
                for bi, broad in enumerate(cluster_points(points)):
                    arrays[f'broad_{bi}'] = broad
                    for si, seed in enumerate(_metric_components(broad)):
                        arrays[f'broad_{bi}_seed_{si}'] = seed
                np.savez_compressed(export/f'{method}.npz', **arrays)
                row['partition_export'] = str(export/f'{method}.npz')
            rows.append(row)
            print(scene['name'], method, {key: row[key] for key in ('components', 'mixed', 'false_splits', 'missing')}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(kind='research_only_island_assignment', rows=rows,
        baseline_pythonpath=str(args.baseline_pythonpath),
        limitation='Selection-only functional replay; concurrent physics may invalidate timing comparisons'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/multiview-iteration2/selection-capture-v2/capture_manifest.json'))
    parser.add_argument('--controls', action='store_true', help='Intact/missing-band objects and fragmented candidate-work counter')
    parser.add_argument('--geometry', type=Path, default=Path('outputs/multiview-geometry-final/clouds'))
    parser.add_argument('--scenario', action='append')
    parser.add_argument('--methods', nargs='+', default=['baseline', 'nearest', 'surface'])
    parser.add_argument('--baseline-pythonpath', type=Path,
                        default=Path('outputs/multiview-readiness/baseline_snapshot'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts-dir', type=Path)
    main(parser.parse_args())
