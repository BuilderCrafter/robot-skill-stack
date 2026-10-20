"""Oracle-isolated cube/sphere diagnostics on saved final-camera captures.

Truth masks and signed surface errors are research diagnostics only. They never
enter an automatic provider. This compares the existing fitter on per-camera
and fused measured clouds, preserving rejected/empty rows. Orientation evidence
is a proposal serialized here; production geometry/contracts are not changed.
There are no native throughput, live freshness or manipulation-safety claims.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter, orientation_metadata
from robot_skill_stack.world.perception_multiview.observability import assess
from scripts.benchmark_multiview_robustness import ROOT, load_frames
from tests.support.multiview_final_evaluation import box_orientation, plain
from tests.support.multiview_iteration2 import distribution, object_from_payload, surface_memberships


orientation_proposal = orientation_metadata


def finalize_with_evidence(fitter, fit, points, frames, views, plane):
    result = fitter.finalize(fit, points, support_plane_z=plane, view_clouds=views)
    geometry = result.geometry
    evidence = assess(result, points, frames, view_clouds=views, support_plane_z=plane)
    geometry.metadata['geometric_observability'] = evidence
    if evidence['applicable']:
        geometry.metadata['parameter_observable'] = bool(
            geometry.metadata.get('parameter_observable', False) and evidence['observable'])
        bounds = evidence['conservative_geometry']
        if bounds is not None:
            geometry.metadata['measured_box_geometry'] = dict(center=result.center.tolist(),
                box_size=geometry.box_size.tolist(), size_world=result.size.tolist())
            result.center = np.asarray(bounds['center'])
            geometry.box_size = np.asarray(bounds['box_size'])
            c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            result.size = abs(rotation) @ geometry.box_size
            geometry.metadata.update(center=result.center.tolist(), size_world=result.size.tolist())
    return result


def fit_metrics(fit, truth, *, have_points):
    geometry = fit.geometry
    correct = geometry.shape.value == truth.shape
    result = dict(predicted=geometry.shape.value, correct=correct, confidence=geometry.confidence,
        center=fit.center.tolist(), radius=geometry.radius, box_size=geometry.box_size,
        yaw=geometry.yaw, parameter_observable=geometry.metadata.get('parameter_observable'),
        center_error_mm=None if not have_points else float(np.linalg.norm(fit.center-truth.center)*1000),
        horizontal_center_error_mm=None if not have_points else float(np.linalg.norm(fit.center[:2]-truth.center[:2])*1000),
        vertical_center_bias_mm=None if not have_points else float((fit.center[2]-truth.center[2])*1000),
        radius_bias_mm=None, dimension_error_mm=None, metadata=deepcopy(geometry.metadata))
    if correct and truth.shape == 'sphere':
        result['radius_bias_mm'] = float((geometry.radius-truth.radius)*1000)
        result['dimension_error_mm'] = 2*abs(result['radius_bias_mm'])
    elif correct and truth.shape == 'cube':
        actual = np.r_[np.sort(truth.size[:2]), truth.size[2]]
        estimate = np.r_[np.sort(geometry.box_size[:2]), geometry.box_size[2]]
        result.update(dimension_error_mm=float(max(abs(estimate-actual))*1000),
                      height_error_mm=float(abs(geometry.box_size[2]-truth.size[2])*1000),
                      orientation=box_orientation(truth, geometry))
    return result


def measured_diagnostics(points, truth, face_tolerance):
    result = dict(points=len(points))
    if not len(points):
        return result
    delta = points-truth.center
    if truth.shape == 'sphere':
        radii = np.linalg.norm(delta, axis=1)
        signed = (radii-truth.radius)*1000
        directions = delta/np.maximum(radii[:, None], 1e-12)
        result.update(signed_surface_residual_mm=dict(mean=float(signed.mean()),
            median=float(np.median(signed)), p05=float(np.percentile(signed, 5)),
            p95=float(np.percentile(signed, 95)), min=float(signed.min()), max=float(signed.max())),
            direction_span=np.ptp(directions, axis=0).tolist(),
            directional_second_moment_eigenvalues=np.linalg.eigvalsh(directions.T @ directions/len(points)).tolist())
    elif truth.shape == 'cube':
        c, s = np.cos(truth.yaw), np.sin(truth.yaw)
        local = delta @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        planes = {f'{"xyz"[axis]}{sign:+d}': int(np.count_nonzero(
            abs(local[:, axis]-sign*truth.size[axis]/2) <= face_tolerance))
            for axis in range(3) for sign in (-1, 1)}
        top = abs(local[:, 2]-truth.size[2]/2) <= face_tolerance
        result.update(physical_plane_proximity_counts=planes,
            local_measured_span_m=np.ptp(local, axis=0).tolist(),
            top_boundary_counts={f'{"xy"[axis]}{sign:+d}': int(np.count_nonzero(top &
                (abs(local[:, axis]-sign*truth.size[axis]/2) <= face_tolerance)))
                for axis in (0, 1) for sign in (-1, 1)},
            note='Truth-plane proximity includes corner points; not proof of an independently visible face')
    return result


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the established wrapper with NumPy 1.26.4')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.capture/'capture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    paths = [*list((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py')), Path(__file__)]
    hashes = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    start = hashes()
    rows, fitter = [], SurfaceFitter()
    scene = load_scene_config(ROOT/'config/scenes/multiview_playground.toml').scene
    for record in manifest['scenes']:
        layout = record.get('layout', {}).get('name', 'legacy')
        if args.layout and layout not in args.layout:
            continue
        if args.scene and not any(record.get(key) in args.scene for key in ('key', 'scene_key', 'name')):
            continue
        snapshots = record['frames'] if args.all_frames else record['frames'][-1:]
        for snapshot in snapshots:
            frames = load_frames(args.capture, snapshot['files'])
            truths = tuple(object_from_payload(obj) for obj in snapshot.get('objects', record.get('objects', [])))
            # The ordinary support crop excludes nearby table returns from the
            # analytic masks. Isolation itself remains diagnostic-only.
            raw_views = [crop_workspace(project_depth(frame, stride=args.stride)[0], scene) for frame in frames]
            labels = [surface_memberships(points, truths, tolerance=args.mask_tolerance) for points in raw_views]
            for ti, truth in enumerate(truths):
                if truth.shape not in ('cube', 'sphere'):
                    continue
                measured = [points[label == ti] for points, label in zip(raw_views, labels)]
                balanced = [voxelize(points, .002, 12000) for points in measured]
                fused = voxelize(np.concatenate(balanced), .002, 12000)
                for view, points, actual_frames, evidence_views in [
                    *[(f'camera_{ci}', cloud, (frames[ci],), (cloud,)) for ci, cloud in enumerate(balanced)],
                    ('fused', fused, frames, tuple(balanced))]:
                    raw = fitter.fit(points, support_plane_z=args.support_plane_z)
                    final = finalize_with_evidence(fitter, raw, points, actual_frames, evidence_views, args.support_plane_z)
                    unvoxelized = (np.concatenate(measured) if view == 'fused' else
                                   measured[int(view.removeprefix('camera_'))])
                    rows.append(dict(scene=record['key'], layout=layout, frame_index=snapshot['index'],
                        phase=snapshot.get('phase'), object=truth.name, truth_shape=truth.shape,
                        truth=dict(center=truth.center, size=truth.size, yaw=truth.yaw, radius=truth.radius), view=view,
                        measured=measured_diagnostics(points, truth, args.face_tolerance),
                        raw_depth_measured=measured_diagnostics(unvoxelized, truth, args.face_tolerance),
                        raw_view_point_counts=[len(cloud) for cloud in measured],
                        raw=fit_metrics(raw, truth, have_points=bool(len(points))),
                        final=fit_metrics(final, truth, have_points=bool(len(points))),
                        orientation_metadata_proposal=orientation_proposal(final, points)))
    summary = {}
    for layout, shape, view in sorted({(row['layout'], row['truth_shape'], row['view']) for row in rows}):
        selected = [row for row in rows if (row['layout'], row['truth_shape'], row['view']) == (layout, shape, view)]
        summary[f'{layout}/{shape}/{view}'] = dict(rows=len(selected),
            zero_point_rows=sum(row['measured']['points'] == 0 for row in selected),
            **{stage: dict(correct=sum(row[stage]['correct'] for row in selected),
                center_error_mm=distribution([row[stage]['center_error_mm'] for row in selected if row[stage]['correct']]),
                radius_bias_mm=distribution([row[stage]['radius_bias_mm'] for row in selected if row[stage]['correct']]),
                dimension_error_mm=distribution([row[stage]['dimension_error_mm'] for row in selected if row[stage]['correct']]))
                for stage in ('raw', 'final')})
    end = hashes()
    for filename, value in (('rows.json', rows), ('result.json', dict(status='PASS' if start == end else 'FAIL',
        kind='oracle_isolated_final_geometry_diagnostics', summary=summary, source_hashes_stable=start == end,
        input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        stride=args.stride, mask_tolerance_m=args.mask_tolerance, face_tolerance_m=args.face_tolerance,
        source_hashes=start, end_source_hashes=end, production_orientation_metadata_changed=False,
        scope='Oracle fitting diagnostics, not automatic detection or live performance; all empty/rejected rows retained'))):
        (args.output/filename).write_text(json.dumps(plain(value), indent=2)+'\n')
    print(json.dumps(plain(dict(rows=len(rows), summary=summary)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout', action='append', default=[])
    parser.add_argument('--scene', action='append', default=[])
    parser.add_argument('--all-frames', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--mask-tolerance', type=float, default=.003)
    parser.add_argument('--face-tolerance', type=float, default=.001)
    parser.add_argument('--support-plane-z', type=float, default=0.)
    args = parser.parse_args()
    if args.stride < 1 or min(args.mask_tolerance, args.face_tolerance) <= 0:
        parser.error('Use positive stride and diagnostic tolerances')
    main(args)
