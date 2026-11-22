"""Offline, fixed-input segmentation A/B probe; run each checkout in its own process.

Analytic depths approximate the demo at 320x240 or 640x480, not Isaac rendering. Truth is used
only to render inputs and score confirmed WorldModel objects, never by perception.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import tracemalloc
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview import segmentation
from tests.support.primitive_scene import Solid, intersect, render


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_scene():
    text = Path('scenes/multiview_perception_final.usda').read_text()
    targets = {}
    for shape, name, body in re.findall(r'def (Cube|Sphere) "(\w+)"\s*\{(.*?)\n\s*\}', text, re.S):
        center = ast.literal_eval(re.search(r'xformOp:translate = (\([^\n]+\))', body)[1])
        if shape == 'Sphere':
            solid = Solid('sphere', center=center, radius=float(re.search(r'double radius = ([\d.]+)', body)[1]))
        else:
            size = ast.literal_eval(re.search(r'xformOp:scale = (\([^\n]+\))', body)[1])
            yaw = np.deg2rad(float(re.search(r'xformOp:rotateZ = ([-\d.]+)', body)[1]))
            solid = Solid('cube', center=center, size=size, yaw=float(yaw))
        targets[name] = solid
    transforms = []
    for matrix in re.findall(r'matrix4d xformOp:transform = (\([^\n]+\))', text):
        # USD row-vector camera (+Y up, -Z forward) to ROS column-vector axes.
        transforms.append(np.array(ast.literal_eval(matrix)).T @ np.diag([1., -1., -1., 1.]))
    return targets, transforms


def final_frames(solids, transforms, width):
    height = width*3//4
    focal = 24 / 20.955 * width  # Side-camera lens; same approximation for inherited main lens.
    intrinsics = np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
    y, x = np.indices((height, width))
    rays = np.c_[(x.ravel()+.5-width/2)/focal, (y.ravel()+.5-height/2)/focal, np.ones(x.size)]
    frames = []
    for i, transform in enumerate(transforms):
        directions, origin = rays @ transform[:3, :3].T, transform[:3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = -origin[2] / directions[:, 2]
        depth = np.where(depth > 0, depth, np.inf)
        for solid in solids:
            depth = np.minimum(depth, intersect(solid, origin, directions))
        frames.append(DepthFrame(str(i), (1., .1), depth.reshape(height, width).astype(np.float32),
                                 intrinsics.copy(), transform.copy(), 100.))
    return frames


def capture(root, width, selected):
    if (root/'manifest.json').exists():
        raise FileExistsError('Use a new directory; frozen observations must not be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    targets, transforms = final_scene()
    sphere = Solid('sphere')
    box = Solid('cube', center=(.45, 0., .025), size=(.068, .042, .05), yaw=np.deg2rad(-33))
    perpendicular = np.array([-np.sin(box.yaw), np.cos(box.yaw), 0.])
    near = {f'box_{i}': Solid('cube', center=tuple(np.array(box.center)+sign*.024*perpendicular),
                             size=box.size, yaw=box.yaw) for i, sign in enumerate((-1, 1))}
    axis_near = {f'box_{i}': Solid('cube', center=(.45, sign*.024, .025), size=box.size)
                 for i, sign in enumerate((-1, 1))}
    specifications = [
        ('A_final', targets, [], 'final', None),
        ('B_sphere_20pct', {'sphere': sphere}, [], 'fixture', .2),
        ('B_box_20pct', {'box': box}, [], 'fixture', .2),
        ('B_sphere_occluded', {'sphere': sphere},
         [Solid('cube', center=(.51, 0., .075), size=(.025, .04, .04))], 'fixture', None),
        ('B_box_severe', {'box': box}, [], 'fixture', .65),
        ('C_rotated', near, [], 'fixture', None),
        ('C_axis_control', axis_near, [], 'fixture', None),
        ('D_alone', {'sphere': sphere}, [], 'fixture', None),
        ('D_near', {'sphere': sphere},
         [Solid('cube', center=(.45, .185, .04), size=(.045, .24, .05))], 'fixture', None),
        ('D_coarse_connected', {'sphere': sphere},
         [Solid('cube', center=(.45, .15, .04), size=(.045, .24, .05))], 'fixture', None),
        ('D_touching', {'sphere': sphere},
         [Solid('cube', center=(.45, .145, .04), size=(.045, .24, .05))], 'fixture', None),
        ('C_rotated_close', near, [], 'close', None),
        ('D_coarse_connected_close', {'sphere': sphere},
         [Solid('cube', center=(.45, .15, .04), size=(.045, .24, .05))], 'close', None),
        ('D_touching_close', {'sphere': sphere},
         [Solid('cube', center=(.45, .145, .04), size=(.045, .24, .05))], 'close', None),
    ]
    cases = []
    for name, truth, clutter, view, missing in specifications:
        if selected and name not in selected:
            continue
        solids = list(truth.values()) + clutter
        if view == 'final':
            frames = final_frames(solids, transforms, width)
        else:
            frames = []
            eyes = ((.85, .35, .60), (.2, -.5, .45)) if view == 'close' else ((1.05, .55, .90), (.25, -.85, .65))
            for camera, eye in enumerate(eyes):
                raw = render(solids, eye=eye, resolution=(width, width*3//4), seed=71+camera)
                intrinsics = raw.intrinsics.copy()
                intrinsics[:2, 2] += .5
                frames.append(DepthFrame(str(camera), (1., .1), raw.depth, intrinsics,
                                         raw.world_from_camera, 100.))
        arrays = {}
        for camera, frame in enumerate(frames):
            depth = frame.depth.copy()
            if missing is not None:
                depth[np.random.default_rng(720+camera).random(depth.shape) < missing] = np.nan
            arrays[f'depth_{camera}'] = depth
            arrays[f'intrinsics_{camera}'] = frame.intrinsics
            arrays[f'transform_{camera}'] = frame.world_from_camera
        # Four fixed synthetic render records of each unchanged static scene.
        arrays['acquired_at'] = 100. + np.arange(4)*.2
        arrays['tokens'] = np.c_[np.arange(1, 5), .1+np.arange(4)*.2]
        path = root/f'{name}.npz'
        np.savez_compressed(path, **arrays)
        cases.append(dict(name=name, file=path.name, sha256=digest(path),
                          truth={key: asdict(solid) for key, solid in truth.items()},
                          clutter=[asdict(solid) for solid in clutter], views=view, dropout=missing))
    manifest = dict(resolution=[width, width*3//4], seed=71, stride=2, voxel_size=.002, max_points=12000,
                    scene_sha256=digest(Path('scenes/multiview_perception_final.usda')),
                    approximation='Analytic surfaces/plane; USD final camera poses, 24mm lens assumed for both; fixture views elsewhere.',
                    cases=cases)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Captured {len(cases)} fixed scenarios; manifest {digest(root/"manifest.json")}')


def owned(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def frames_at(data, index):
    return tuple(DepthFrame(str(i), tuple(data['tokens'][index]), owned(data[f'depth_{i}']),
                            owned(data[f'intrinsics_{i}']), owned(data[f'transform_{i}']),
                            float(data['acquired_at'][index])) for i in range(2))


def objects(model):
    return [dict(id=obj.object_id, shape=obj.class_name, center=obj.pose.position.tolist(),
                 dimensions=(obj.geometry.box_size.tolist() if obj.class_name == 'cube' else
                             [2*obj.geometry.radius]*3 if obj.class_name == 'sphere' else obj.size.tolist()),
                 graspable=obj.graspable, confidence=obj.confidence,
                 reason=obj.geometry.metadata.get('reason'),
                 parameter_reason=obj.geometry.metadata.get('parameter_uncertainty', {}).get('reason'))
            for obj in model.visible_objects()]


def score(predictions, truth, previous):
    assigned, used, rows = {}, set(), []
    edges = sorted((float(np.linalg.norm(np.array(obj['center'])-solid['center'])), key, i)
                   for key, solid in truth.items() for i, obj in enumerate(predictions))
    for distance, key, i in edges:
        if distance <= .06 and key not in assigned and i not in used:
            assigned[key], _ = i, used.add(i)
    for key, solid in truth.items():
        if key not in assigned:
            rows.append(dict(target=key, detected=False, usable=False))
            continue
        obj = predictions[assigned[key]]
        expected = solid['size'] if solid['shape'] == 'cube' else [2*solid['radius']]*3
        predicted = obj['dimensions']
        if solid['shape'] == 'cube' and obj['shape'] == 'cube':
            expected, predicted = sorted(expected[:2])+[expected[2]], sorted(predicted[:2])+[predicted[2]]
        center_error = float(np.linalg.norm(np.array(obj['center'])-solid['center']))*1000
        dimension_error = float(np.max(abs(np.array(predicted)-expected)))*1000
        rows.append(dict(target=key, detected=True, shape=obj['shape'], center_error_mm=center_error,
                         dimension_error_mm=dimension_error, graspable=obj['graspable'], confidence=obj['confidence'],
                         id=obj['id'], stable_id=any(p['id']==obj['id'] for p in previous), reason=obj['reason'],
                         usable=obj['shape']==solid['shape'] and center_error<=10 and dimension_error<=10))
    return dict(count=len(predictions), extras=len(predictions)-len(assigned), targets=rows)


def evaluate(root, output):
    assert Path(segmentation.__file__).resolve().is_relative_to(Path.cwd().resolve()), 'Wrong checkout on PYTHONPATH'
    manifest = json.loads((root/'manifest.json').read_text())
    scene = load_scene_config('config/scenes/multiview_perception_final.toml').scene
    results, graph = [], []
    for case in manifest['cases']:
        path = root/case['file']
        assert digest(path)==case['sha256'], 'Capture changed'
        with np.load(path, allow_pickle=False) as data:
            provider = MultiViewProvider(SimpleNamespace(close=lambda: None), scene)
            model = WorldModel()
            updater = WorldModelUpdater(model, provider)
            history, timing, fit_counts = [], [], []
            original_fit, calls = provider.processor.fitter.fit, [0]
            def measured_fit(*args, **kwargs):
                calls[0] += 1
                return original_fit(*args, **kwargs)
            provider.processor.fitter.fit = measured_fit
            for index in range(4):
                frames = frames_at(data, index)
                with patch('time.monotonic', return_value=frames[0].acquired_at+.001):
                    prepared = provider.acquire(frames=frames)
                    context = model.observation_context()
                    calls[0] = 0
                    started = time.perf_counter()
                    result = provider.processor.compute(prepared)
                    timing.append((time.perf_counter()-started)*1000)
                    fit_counts.append(calls[0])
                    assert calls[0] <= 24
                    assert updater.accepts(result.timestamp, context.revision)
                    updater.publish(provider.associate(result, context), context_revision=context.revision)
                    history.append(objects(model))
                    assert all(obj['shape'] is not None or not obj['graspable'] for obj in history[-1])
            result = score(history[-1], case['truth'], history[-2])
            result.update(scenario=case['name'], compute_ms=timing, fits=fit_counts,
                          confirmation_counts=list(map(len, history)), observations=history[-1])
            frames = frames_at(data, 0)
            clouds = [voxelize(crop_workspace(project_depth(f, bounds=(scene.workspace_min, scene.workspace_max))[0], scene), .002, 12000) for f in frames]
            fused = voxelize(np.concatenate(clouds), .002, 12000)
            clusters = segmentation.cluster_points(fused)
            result['coarse_sizes'] = list(map(len, clusters))
            results.append(result)
            if case['name'] in ('A_final', 'C_rotated', 'D_coarse_connected'):
                def run_graph(minimum=25):
                    return [segmentation._metric_components(cloud, min_points=minimum) for cloud in clusters]
                exact = run_graph(1)
                # Content hash ignores union-find root choices, retaining all point assignments.
                canonical = [sorted(tuple(sorted(map(tuple, group))) for group in groups) for groups in exact]
                assert [sum(map(len, groups)) for groups in exact] == list(map(len, clusters))
                durations = []
                for _ in range(5):
                    started = time.perf_counter(); run_graph(); durations.append((time.perf_counter()-started)*1000)
                tracemalloc.start(); run_graph(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                graph.append(dict(scenario=case['name'], points=len(fused), median_ms=float(np.median(durations)),
                                  min_ms=min(durations), max_ms=max(durations), peak_bytes=peak,
                                  components_sha256=hashlib.sha256(repr(canonical).encode()).hexdigest()))
            provider.close()
    paths = ('segmentation.py', 'provider.py', 'fitting.py', 'observability.py')
    hashes = {name: digest(Path('robot_skill_stack/world/perception_multiview')/name) for name in paths}
    output.write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                                      manifest_sha256=digest(root/'manifest.json'), sources=hashes,
                                      cases=results, graph=graph), indent=2)+'\n')
    for result in results:
        compact = [(r['target'], r.get('shape', 'missing'), round(r.get('center_error_mm', 0), 2),
                    round(r.get('dimension_error_mm', 0), 2), r.get('graspable'), round(r.get('confidence', 0), 3)) for r in result['targets']]
        print(result['scenario'], result['count'], compact, 'ms', round(float(np.median(result['compute_ms'])), 2))
    print('Graph:', graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'evaluate'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--width', type=int, choices=(320, 640), default=320)
    parser.add_argument('--cases', nargs='+', help='Capture only these named cases')
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args.data, args.width, args.cases)
    else:
        evaluate(args.data, args.output)
