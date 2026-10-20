"""Compare frozen/new fitters on identical saved object clouds, without Isaac.

Keep all unknowns and perturbation failures. Replay timings exclude projection,
segmentation, acquisition and rendering and are never live source throughput.
The baseline-pythonpath must contain the immutable baseline_multiview package.
"""
import argparse
import cProfile
import hashlib
import importlib
import json
from pathlib import Path
import pstats
import sys
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter


def read_ply(path):
    with path.open() as stream:
        count = None
        for line in stream:
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            if line.strip() == 'end_header':
                break
        if count == 0:
            return np.empty((0, 3))
        return np.loadtxt(stream).reshape(-1, 3)


def saved_objects(geometry, detection, manifests, captures):
    if geometry:
        for row in json.loads((geometry/'rows.json').read_text()):
            if row['method'].startswith('surface') and row['repeat'] == 0:
                path = geometry/'clouds'/f"{row['trial']}_{row['method']}.ply"
                yield dict(name=str(path), points=read_ply(path), truth=row['truth'],
                           center=row['truth_center'], dimensions=row['truth_dimensions'])
    if detection:
        for path in sorted((detection/'clouds').glob('*multiview*.ply')):
            yield dict(name=str(path), points=read_ply(path), truth=None)
    for manifest in manifests:
        for record in json.loads(manifest.read_text()):
            path = manifest.parent/record['cloud']
            yield dict(record, name=str(path), points=read_ply(path))
    for capture in captures:
        for scene in json.loads((capture/'capture_manifest.json').read_text())['scenes']:
            for frame in scene['frames']:
                for record in frame.get('oracle_clouds', []):
                    obj = record['object']
                    path = capture/record['path']
                    shape = obj['shape']
                    dimensions = (sorted(obj['size']) if shape == 'cube' else
                                  [2*obj['radius']]*3 if shape == 'sphere' else
                                  [2*obj['radius'], 2*obj['radius'], obj['length']] if shape == 'cylinder' else None)
                    yield dict(name=str(path), points=read_ply(path),
                               truth=shape if shape in ('cube', 'sphere', 'cylinder') else 'unknown',
                               center=obj['center'], dimensions=dimensions, axis=obj.get('axis'), yaw=obj.get('yaw'))


def perturbations(records, seed):
    rng = np.random.default_rng(seed)
    for record in records:
        if not record.get('truth'):
            continue
        points = record['points']
        for label, fraction, noise in (('sparse25', .25, 0.), ('sparse50', .5, 0.),
                                       ('noise03mm', 1., .0003), ('noise08mm', 1., .0008)):
            count = max(1, round(len(points)*fraction)) if len(points) else 0
            index = np.sort(rng.choice(len(points), count, replace=False))
            cloud = points[index]+rng.normal(0., noise, (len(index), 3))
            yield dict(record, name=f'{label}/{record["name"]}', points=cloud)
    for count in (0, 12, 29, 30, 50, 300):
        xy = rng.uniform([.42, -.03], [.48, .03], (count, 2))
        yield dict(name=f'unknown/planar{count}', points=np.c_[xy, np.full(count, .05)], truth='unknown')
    for aspect in (1.25, 1.8, 2.5):
        direction = rng.normal(size=(1000, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        direction[:, 2] = abs(direction[:, 2])
        points = direction*np.array([.025*aspect, .025, .025])+[.45, 0., .025]
        yield dict(name=f'unknown/ellipsoid{aspect}', points=points, truth='unknown')


def fit_row(record, fit, duration):
    g = fit.geometry
    shape = g.shape.value
    dimensions = (np.sort(g.box_size) if g.box_size is not None else
                  np.full(3, 2*g.radius) if shape == 'sphere' else
                  np.array([2*g.radius, 2*g.radius, g.length]) if g.radius is not None else None)
    truth = record.get('truth')
    correct = shape == truth if truth else None
    row = dict(name=record['name'], truth=truth, predicted=shape, correct=correct,
               unknown=shape == 'unknown', confidence=g.confidence, scores=g.scores,
               input_points=len(record['points']), detected=len(record['points']) >= 30,
               residuals_m=g.metadata.get('residuals_m', {}), fit_points=g.metadata.get('fit_points'),
               estimated_center=fit.center.tolist(), estimated_dimensions=None if dimensions is None else dimensions.tolist(),
               estimated_axis=None if g.axis is None else g.axis.tolist(), estimated_yaw=g.yaw,
               center_error_mm=None, dimension_error_mm=None, axis_error_deg=None, fit_ms=duration)
    if row['detected'] and record.get('center') is not None:
        row['center_error_mm'] = float(np.linalg.norm(fit.center-record['center'])*1000)
    if correct and dimensions is not None and record.get('dimensions') is not None:
        row['dimension_error_mm'] = float(np.max(abs(dimensions-record['dimensions']))*1000)
    if correct and shape == 'cylinder' and record.get('axis') is not None and g.axis is not None:
        row['axis_error_deg'] = float(np.rad2deg(np.arccos(np.clip(abs(g.axis @ record['axis']), 0., 1.))))
    if correct and shape == 'cube' and record.get('yaw') is not None and g.yaw is not None:
        error = abs((g.yaw-record['yaw']+np.pi/4) % (np.pi/2)-np.pi/4)
        row['axis_error_deg'] = float(np.rad2deg(error))
    return row


def summarize(rows):
    summary = {}
    for variant in sorted({r['variant'] for r in rows}):
        subset = [r for r in rows if r['variant'] == variant]
        result = dict(rows=len(subset), labeled=sum(r['correct'] is not None for r in subset),
                      correct=sum(r['correct'] is True for r in subset), unknown=sum(r['unknown'] for r in subset),
                      insufficient_clouds=sum(not r['detected'] for r in subset))
        for key in ('fit_ms', 'center_error_mm', 'dimension_error_mm', 'axis_error_deg'):
            values = [r[key] for r in subset if r[key] is not None]
            result[key] = (dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                                max=float(max(values)), samples=len(values)) if values else None)
        summary[variant] = result
    return summary


def radius_diagnostics(records, baseline):
    """Separate existing depth-surface bias from the final fitter voxel stage."""
    rows = []
    for record in records:
        if record.get('truth') != 'sphere' or record.get('center') is None or record.get('dimensions') is None:
            continue
        points = record['points']
        reduced = baseline.voxelize(points, .0015, 1200)
        raw_fit, voxel_fit = baseline._radial(points), baseline._radial(reduced)
        if raw_fit is None or voxel_fit is None:
            rows.append(dict(name=record['name'], diagnostic='insufficient_radial_evidence'))
            continue
        radius = record['dimensions'][0]/2
        bias = np.linalg.norm(points-record['center'], axis=1)-radius
        rows.append(dict(name=record['name'], raw_radius_error_mm=(raw_fit[1]-radius)*1000,
                         voxel_radius_error_mm=(voxel_fit[1]-radius)*1000,
                         truth_surface_radial_error_percentiles_mm=np.percentile(bias*1000, [0, 5, 50, 95, 100]).tolist()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--detection', type=Path)
    parser.add_argument('--manifest', type=Path, action='append', default=[],
                        help='JSON list of {cloud, truth, center, dimensions}, paths relative to JSON')
    parser.add_argument('--capture', type=Path, action='append', default=[],
                        help='Iteration-2 capture directory with capture_manifest.json oracle exports')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['baseline', 'box_only', 'improved'],
                        choices=['baseline', 'box_only', 'improved', 'sample600', 'sample300'])
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--timing-context', default='Unspecified workstation contention; offline replay only')
    parser.add_argument('--profile', action='store_true', help='Separate instrumented pass; excluded from replay timings')
    parser.add_argument('--radius-diagnostics', action='store_true', help='Oracle sphere-radius bias analysis, research only')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy 1.26.4 through the Isaac wrapper, got {np.__version__}')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    old = importlib.import_module('baseline_multiview.fitting')
    box_only = type('BoxOnlyFitter', (old.SurfaceFitter,), {'_box': SurfaceFitter._box})
    fitters = dict(baseline=old.SurfaceFitter(), box_only=box_only(), improved=SurfaceFitter(),
                   sample600=SurfaceFitter(max_points=600), sample300=SurfaceFitter(max_points=300))
    original = list(saved_objects(args.geometry, args.detection, args.manifest, args.capture))
    data = list(perturbations(original, args.seed)) if args.stress else original
    if not data:
        parser.error('No saved object clouds found')
    # Store the exact numerical inputs once, including sparse/unknown/failing
    # clouds. Baseline and alternatives receive these same immutable arrays.
    np.savez_compressed(args.output/'clouds.npz', **{f'cloud_{i}': r['points'] for i, r in enumerate(data)})
    manifest = [{k: v for k, v in record.items() if k != 'points'} | dict(
                key=f'cloud_{i}', sha256=hashlib.sha256(record['points'].tobytes()).hexdigest())
                for i, record in enumerate(data)]
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in args.variants:
        fitters[name].fit(data[0]['points'])
    rows = []
    for repeat in range(args.repeats):
        for index, record in enumerate(data):
            offset = (index+repeat) % len(args.variants)
            order = args.variants[offset:]+args.variants[:offset]
            for name in order:
                started = time.perf_counter()
                fit = fitters[name].fit(record['points'])
                duration = (time.perf_counter()-started)*1000
                rows.append(dict(fit_row(record, fit, duration), variant=name, repeat=repeat))
    (args.output/'rows.json').write_text(json.dumps(rows, indent=2))
    summary = summarize(rows)
    baseline = {(r['name'], r['repeat']): r for r in rows if r['variant'] == 'baseline'}
    differences = [r for r in rows if r['variant'] != 'baseline' and baseline and
                   r['predicted'] != baseline[(r['name'], r['repeat'])]['predicted']]
    (args.output/'classification_changes.json').write_text(json.dumps(differences, indent=2))
    result = dict(kind='offline_identical_object_cloud_replay', clouds=len(data), repeats=args.repeats,
                  timings_exclude='acquisition, projection, segmentation, rendering; native throughput not measured',
                  dimensions_conditional_on_correct_classification=True, seed=args.seed,
                  numpy_version=np.__version__,
                  timing_context=args.timing_context,
                  baseline_pythonpath=str(args.baseline_pythonpath), summary=summary,
                  classification_changes=len(differences))
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    if args.radius_diagnostics:
        diagnostics = radius_diagnostics(data, old)
        (args.output/'radius-bias.json').write_text(json.dumps(diagnostics, indent=2))
    if args.profile:
        for name in args.variants:
            profiler = cProfile.Profile()
            profiler.enable()
            for record in data:
                fitters[name].fit(record['points'])
            profiler.disable()
            profiler.dump_stats(str(args.output/f'{name}.prof'))
            with (args.output/f'{name}-profile.txt').open('w') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
