"""Checkpoint B: oracle-segmented native surface fitting experiment.

Oracle masks are evaluation-only. Failed classifications remain in all metrics.
This entry point never commands Home, Move, Pick, Place, or robot joints.
"""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import csv
import json
from pathlib import Path
import time

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, save_ply
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit, SurfaceFitter
from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, research_objects, set_viewpoint
from tests.support.multiview_research import ResearchObject, metric_row, oracle_mask
from tests.support.results import fail_result, save_ppm, write_result


def reference_v1(frame, scene, obj):
    localizer = RgbdLocalizer(frame.intrinsics)
    discoverer = DepthObjectDiscoverer(localizer, support_plane_z=scene.support_plane_z,
        workspace_min=scene.workspace_min, workspace_max=scene.workspace_max)
    rgb = frame.rgb if frame.rgb is not None else np.zeros((*frame.depth.shape, 3), np.uint8)
    v1_frame = PerceptionFrame(1, rgb, frame.depth, frame.intrinsics, frame.world_from_camera, frame.acquired_at)
    candidates = discoverer.discover(v1_frame)
    # Oracle match is for this isolated experiment only, not a production filter.
    candidates = [c for c in candidates if np.linalg.norm(c.position-obj.center) < .10]
    if not candidates:
        return PrimitiveFit(PrimitiveGeometry(PrimitiveShape.UNKNOWN), np.zeros(3), np.zeros(3)), False
    candidate = min(candidates, key=lambda c: np.linalg.norm(c.position-obj.center))
    geometry = PrimitiveEstimator().estimate(candidate)
    return PrimitiveFit(geometry, np.asarray(geometry.metadata.get('center', candidate.position)),
                         np.asarray(geometry.metadata.get('size_world', candidate.size))), True


def summarize(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method'] == method]
        result[method] = {'trials': len(subset), 'correct': sum(r['correct'] for r in subset),
                          'unknown': sum(r['unknown'] for r in subset),
                          'detected': sum(r['detected'] for r in subset)}
        for metric in ('center_error_mm', 'dimension_error_mm', 'axis_error_deg', 'fit_ms'):
            values = [r[metric] for r in subset if r.get(metric) is not None]
            result[method][metric] = None if not values else {
                'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'max': float(np.max(values)),
                'samples': len(values)}
    return result


def main(args):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    world = open_experiment(simulation_app)
    scene = load_scene_config(ROOT/'config/scenes/playground.toml').scene
    rig = create_cameras(CAMERAS)
    rows, acquisition = [], []
    fitter = SurfaceFitter()
    try:
        for _ in range(60):
            world.step(render=True)
        for arrangement in range(3):
            if arrangement:
                set_viewpoint(rig.cameras[CAMERAS[1]], (.75, -.70, .55))
            for original in research_objects():
                # Translate across the table as well as changing the second view.
                from dataclasses import replace
                obj = replace(original, center=original.center+np.array([-.06 if arrangement else .02,
                                                                       .10 if arrangement else -.08, 0.]))
                objects = [obj]
                if arrangement == 2:
                    objects.append(ResearchObject('Occluder', 'cube',
                        np.array([.57, obj.center[1]+.015, .10]), np.array([.035, .06, .20])))
                populate(objects)
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    for _ in range(18):
                        world.step(render=True)
                    render_ms = (time.perf_counter()-started)*1000
                    started = time.perf_counter()
                    frames = rig.capture()
                    capture_ms = (time.perf_counter()-started)*1000
                    clouds = []
                    for frame in frames:
                        points = crop_workspace(project_depth(frame, stride=1)[0], scene)
                        clouds.append(points[oracle_mask(points, obj)])
                    started = time.perf_counter()
                    fused = fuse_clouds(clouds)
                    fusion_ms = (time.perf_counter()-started)*1000
                    label = f'{arrangement}_{obj.name}_{repeat}'
                    acquisition.append(dict(trial=label, render_ms=render_ms, capture_ms=capture_ms,
                                            fusion_ms=fusion_ms, synchronization=rig.status))
                    for method, cloud in zip(('surface_A', 'surface_B', 'surface_fused'), (*clouds, fused)):
                        started = time.perf_counter()
                        fit = fitter.fit(cloud, support_plane_z=scene.support_plane_z)
                        fit_ms = (time.perf_counter()-started)*1000
                        rows.append(metric_row(obj, fit, method=method, trial=label, arrangement=arrangement,
                            repeat=repeat, occluded=arrangement == 2,
                            detected=len(cloud) >= 30, points=len(cloud), fit_ms=fit_ms))
                        if repeat == 0:
                            save_ply(output/'clouds'/f'{label}_{method}.ply', cloud)
                    for index, frame in enumerate(frames):
                        started = time.perf_counter()
                        fit, detected = reference_v1(frame, scene, obj)
                        rows.append(metric_row(obj, fit, method=f'v1_{"AB"[index]}', trial=label,
                            arrangement=arrangement, repeat=repeat, occluded=arrangement == 2, detected=detected,
                            fit_ms=(time.perf_counter()-started)*1000))
                        if repeat == 0 and frame.rgb is not None:
                            save_ppm(output/'clouds'/f'{label}_camera_{index}.ppm', frame.rgb)
                    # Checkpoint every trial so interruption retains failures and timings.
                    (output/'rows.json').write_text(json.dumps(rows, indent=2))
                    print(label, [(r['method'], r['predicted'], r['correct']) for r in rows[-5:]], flush=True)
        summary = summarize(rows)
        (output/'acquisition.json').write_text(json.dumps(acquisition, indent=2))
        scalar_keys = ('trial', 'method', 'truth', 'predicted', 'correct', 'unknown', 'detected',
                       'center_error_mm', 'dimension_error_mm', 'axis_error_deg', 'fit_ms')
        with (output/'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summary, indent=2), flush=True)
        write_result(args.result, status='PASS', metrics={'kind': 'oracle_segmented_geometry_experiment',
                     'summary': summary, 'trials': len(acquisition), 'repeats': args.repeats,
                     'robot_commands': 0})
    finally:
        rig.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    code = 0
    try:
        main(args)
    except Exception as exc:
        fail_result(args.result, exc)
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
