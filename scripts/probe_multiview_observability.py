"""Compare boundary methods on identical frozen-source current candidate clouds.

This is functional, single-frame geometry research. It does not establish
confirmation, current WorldModel freshness, live performance or manipulation
safety. No oracle mask is used to build the candidate cache. Final native
benchmarks exercise the complete updater/planner path separately.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.perception_multiview.observability import assess
from scripts.benchmark_multiview_grasp_readiness import plain_values
from scripts.benchmark_multiview_robustness import ROOT, load_frames
from tests.support.multiview_iteration2 import SnapshotRig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain_values(value), indent=2))


def source_hashes():
    paths = sorted((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))
    paths.append(Path(__file__))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cache_candidates(args):
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    module = importlib.import_module(f'{args.baseline_package}.provider')
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    rows = []
    for capture in args.capture:
        capture = capture.resolve()
        manifest = json.loads((capture/'capture_manifest.json').read_text())
        for record in manifest['scenes']:
            if args.scene and record['key'] not in args.scene:
                continue
            snapshot = record['frames'][-1]
            frames = load_frames(capture, snapshot['files'])
            rig = SnapshotRig()
            source = module.MultiViewProvider(rig, cfg.scene)
            try:
                rig.push(frames)
                source.observe()
                clouds = [voxelize(crop_workspace(project_depth(frame, stride=source.stride,
                    bounds=(cfg.scene.workspace_min, cfg.scene.workspace_max))[0], cfg.scene),
                    source.voxel_size, source.max_points) for frame in frames]
                keys = [module._voxel_ids(cloud, source.voxel_size) for cloud in clouds]
                # Candidate tracking state is inspected before three-frame
                # confirmation; this probe makes no publication-contract claim.
                for track in source.tracker.tracks.values():
                    fit, points = track.detection.fit, track.detection.points
                    geometry = fit.geometry
                    cluster = module._voxel_ids(points, source.voxel_size)
                    views = [cloud[np.isin(key, cluster)] for cloud, key in zip(clouds, keys)]
                    path = args.output/'clouds'/f'{len(rows):04d}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, points=points,
                                        **{f'view_{i}': view for i, view in enumerate(views)})
                    rows.append(dict(capture=str(capture), scene=record['key'], cloud=str(path.resolve()),
                        frames=[str((capture/file).resolve()) for file in snapshot['files']],
                        shape=geometry.shape.value, confidence=geometry.confidence,
                        center=fit.center, size=fit.size, box_size=geometry.box_size, yaw=geometry.yaw,
                        radius=geometry.radius, length=geometry.length, axis=geometry.axis,
                        metadata=geometry.metadata))
            finally:
                source.close()
    result = dict(purpose='Frozen current candidate fits/clouds; no oracle selection; single-frame functional comparison',
                  baseline_pythonpath=str(args.baseline_pythonpath.resolve()), rows=rows)
    path = args.output/'candidate_manifest.json'
    write_json(path, result)
    return plain_values(result)


def query_states(samples, frame):
    transform, k = frame.world_from_camera, frame.intrinsics
    camera = (samples-transform[:3, 3]) @ transform[:3, :3]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    safe = np.isfinite(uv).all(1)
    pixels = np.rint(np.where(safe[:, None], uv, 0)).astype(int)
    h, w = frame.depth.shape
    inside = (safe & (camera[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    state = np.full(len(samples), 'out_of_frame', dtype='<U16')
    indices = np.flatnonzero(inside)
    measured = frame.depth[pixels[indices, 1], pixels[indices, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    values = np.full(len(indices), 'occupied', dtype='<U16')
    values[measured > camera[indices, 2]+.002] = 'free'
    values[measured < camera[indices, 2]-.002] = 'occluded'
    values[~valid] = 'missing'
    state[indices] = values
    return state


def fixed_samples(fit, axis, sign, distance, levels):
    size, yaw = fit.geometry.box_size, fit.geometry.yaw
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    tangent, height = np.meshgrid(np.linspace(-.4, .4, 13), levels)
    local = np.zeros((tangent.size, 3))
    local[:, axis] = sign*(size[axis]/2+distance)
    local[:, 1-axis] = tangent.ravel()*size[1-axis]
    local[:, 2] = (height.ravel()-.5)*size[2]
    return fit.center+local @ rotation.T


def alternatives(fit, points, frames):
    selected = assess(fit, points, frames)
    if fit.geometry.shape.value != 'cube':
        return {method: dict(observable=False, applicable=False) for method in
                ('face_only', 'anchored_strip4', 'extension4')} | {'selected': selected}
    faces = [axis[side]['physical_face']['supported'] for axis in selected['axes'][:2]
             for side in ('negative', 'positive')]
    strip, extension = [], []
    for axis in (0, 1):
        for sign in (-1, 1):
            exterior = fixed_samples(fit, axis, sign, .004, (1.,))
            interior = fixed_samples(fit, axis, sign, -.004, (1.,))
            strip.append(any(np.count_nonzero((query_states(exterior, frame) == 'free') &
                                              (query_states(interior, frame) == 'occupied')) >= 4
                             for frame in frames))
            hypothetical = fixed_samples(fit, axis, sign, .004, (.25, .5, .75, 1.))
            extension.append(any(np.count_nonzero(query_states(hypothetical, frame) == 'free') >= 4
                                 for frame in frames))
    return dict(face_only=dict(observable=all(faces), sides=faces),
                anchored_strip4=dict(observable=all(strip), sides=strip,
                                     interval_expansion_per_side_m=.004),
                extension4=dict(observable=all(extension), sides=extension,
                                interval_expansion_per_side_m=.004), selected=selected)


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the established Isaac Python wrapper with NumPy 1.26.4')
    args.output.mkdir(parents=True, exist_ok=True)
    start = source_hashes()
    write_json(args.output/'provenance_start.json', start)
    manifest = (json.loads(args.cached_manifest.read_text()) if args.cached_manifest else cache_candidates(args))
    rows = []
    for record in manifest['rows']:
        geometry = PrimitiveGeometry(record['shape'], record['confidence'], box_size=record['box_size'],
            yaw=record['yaw'], radius=record['radius'], length=record['length'], axis=record['axis'],
            metadata=deepcopy(record['metadata']))
        fit = PrimitiveFit(geometry, np.asarray(record['center']), np.asarray(record['size']))
        frames = load_frames(Path('/'), record['frames'])
        with np.load(record['cloud'], allow_pickle=False) as data:
            comparison = alternatives(fit, data['points'], frames)
        for method, result in comparison.items():
            old = bool(geometry.metadata.get('parameter_observable', False))
            eligible = old and (result['observable'] if record['shape'] == 'cube' else True)
            conservative = result.get('conservative_geometry')
            growth = (None if conservative is None else
                      (np.asarray(conservative['box_size'])-geometry.box_size)*1000)
            rows.append(dict(scene=record['scene'], shape=record['shape'], method=method,
                old_parameter_observable=old, graspable=eligible, evidence=result,
                box_dimension_growth_mm=growth, cloud=record['cloud']))
    write_json(args.output/'rows.json', rows)
    summary = {}
    for method in ('face_only', 'anchored_strip4', 'extension4', 'selected'):
        boxes = [row for row in rows if row['method'] == method and row['shape'] == 'cube']
        summary[method] = dict(box_candidates=len(boxes), box_graspable=sum(row['graspable'] for row in boxes),
            rejected_scenes=[row['scene'] for row in boxes if not row['graspable']])
    end = source_hashes()
    write_json(args.output/'provenance_end.json', end)
    result = dict(kind='current_candidate_observability_method_comparison', summary=summary,
        candidate_count=len(manifest['rows']), unchanged_sources=start == end,
        limits='No live timing, confirmation, freshness or full planner safety claims; one saved native frame per scene')
    write_json(args.output/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, nargs='+', default=[])
    parser.add_argument('--scene', action='append', default=[])
    parser.add_argument('--baseline-pythonpath', type=Path)
    parser.add_argument('--baseline-package', default='baseline_multiview')
    parser.add_argument('--cached-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.cached_manifest and (not args.capture or not args.baseline_pythonpath):
        parser.error('Use --cached-manifest or both --capture and --baseline-pythonpath')
    main(args)
