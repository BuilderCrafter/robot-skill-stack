"""Matched automatic detection evaluation; oracle geometry is scoring-only."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import time

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.clouds import save_ply
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, research_objects, set_viewpoint
from tests.support.multiview_research import ResearchObject, metric_row, oracle_mask
from tests.support.results import fail_result, save_ppm, write_result


def score(objects, observations, method, trial):
    rows, assigned = [], set()
    pairs = sorted((float(np.linalg.norm(o.pose.position-obj.center)), oi, di)
                   for oi, obj in enumerate(objects) for di, o in enumerate(observations))
    matches = {}
    for distance, oi, di in pairs:
        if distance < .04 and oi not in matches and di not in assigned:
            matches[oi] = di
            assigned.add(di)
    for oi, obj in enumerate(objects):
        if oi in matches:
            obs = observations[matches[oi]]
            fit = PrimitiveFit(obs.geometry, obs.pose.position, obs.size)
            row = metric_row(obj, fit, method=method, trial=trial, detected=True)
        else:
            row = dict(object=obj.name, truth=obj.shape, predicted='missing', correct=False,
                       unknown=False, detected=False, center_error_mm=None, dimension_error_mm=None,
                       axis_error_deg=None, method=method, trial=trial)
        rows.append(row)
    duplicates = sum(any(np.linalg.norm(observations[di].pose.position-obj.center) < .04
                         for obj in objects) for di in range(len(observations)) if di not in assigned)
    return rows, dict(method=method, trial=trial, expected=len(objects), observed=len(observations),
        matched=len(matches), false_positives=len(observations)-len(assigned), duplicates=duplicates,
        missing=len(objects)-len(matches))


def main(args):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    world = open_experiment(simulation_app)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    rig = create_cameras(CAMERAS)
    base = research_objects()
    layout = ((.34, .10), (.50, -.18), (.35, -.18), (.53, .20))
    originals = [replace(obj, center=np.r_[xy, obj.center[2]]) for obj, xy in
                 zip((base[1], base[2], base[4], base[7]), layout)]
    rows, detections, timings = [], [], []
    scenarios = ('clear', 'occluded_A', 'hidden_A', 'hidden_B', 'nearby', 'elevated_clutter')
    try:
        for _ in range(60):
            world.step(render=True)
        for arrangement in range(2):
            if arrangement:
                set_viewpoint(rig.cameras[CAMERAS[1]], (.25, -.70, .55))
            for scenario in scenarios:
                for repeat in range(args.repeats):
                    # Seeded millimetre translations repeat the same conditions
                    # across all methods without replaying stale camera buffers.
                    seed = 1000*arrangement+100*scenarios.index(scenario)+repeat
                    rng = np.random.default_rng(seed)
                    shift = np.r_[rng.uniform(-.004, .004, 2), 0.]
                    objects = [replace(obj, center=obj.center+shift) for obj in originals]
                    extras = []
                    if scenario in ('occluded_A', 'hidden_A'):
                        width = .06 if scenario == 'occluded_A' else .16
                        offset = .015 if scenario == 'occluded_A' else -.02
                        extras.append(ResearchObject('Occluder', 'cube',
                            np.array([.57, objects[0].center[1]+offset, .10]), np.array([.035, width, .20])))
                    if scenario == 'hidden_B':
                        extras.append(ResearchObject('Occluder', 'cube',
                            np.array([.32, -.14, .10]), np.array([.15, .035, .20])))
                    if scenario == 'nearby':
                        objects = [replace(base[2], name='near_sphere_A', center=np.array([.44, -.045, .02])+shift),
                                   replace(base[2], name='near_sphere_B', center=np.array([.44, .045, .02])+shift)]
                    if scenario == 'elevated_clutter':
                        extras.append(ResearchObject('LinkSurrogate', 'cylinder',
                            np.array([.48, -.06+repeat*.12, .14]), np.array([.12, .03, .03]),
                            radius=.015, length=.12, axis=(1., 0., 0.)))
                    populate(objects+extras)
                    providers = {
                        'v1_A': build_perception_provider(rig.cameras[CAMERAS[0]], cfg),
                        'v1_B': build_perception_provider(rig.cameras[CAMERAS[1]], cfg),
                        'multi_view': MultiViewProvider(rig, cfg.scene)}
                    # V1 factory only needs scene settings and an already calibrated
                    # camera; selector is not changed and V1 code remains untouched.
                    observations = {}
                    for _ in range(3):
                        for _ in range(8):
                            world.step(render=True)
                        for method, provider in providers.items():
                            observations[method] = provider.observe().observations
                    trial = f'{arrangement}_{scenario}_{repeat}'
                    for method, current in observations.items():
                        selected_rows, counts = score(objects, current, method, trial)
                        mixed, checked = 0, 0
                        for obs in current:
                            cloud = providers[method].get_point_cloud(obs.object_id)
                            if cloud is not None and len(cloud):
                                checked += 1
                                fractions = [float(np.mean(oracle_mask(cloud, obj, tolerance=.0045)))
                                             for obj in objects]
                                mixed += sum(fraction >= .1 for fraction in fractions) > 1
                        counts.update(mixed_clusters=int(mixed), mixed_checked=checked)
                        rows.extend(selected_rows)
                        detections.append(counts)
                    # Time the automatic source by itself, excluding both V1
                    # pipelines, with actual render/update work between samples.
                    source = providers['multi_view']
                    for sample in range(3):
                        started = time.perf_counter()
                        for _ in range(8):
                            world.step(render=True)
                        rendered = time.perf_counter()
                        source.observe()
                        finished = time.perf_counter()
                        timings.append(dict(trial=trial, sample=sample,
                            render_ms=(rendered-started)*1000,
                            elapsed_ms=(finished-started)*1000, **source.diagnostics))
                    if repeat == 0:
                        for index, camera in enumerate(rig.cameras.values()):
                            rgb = camera.get_rgb()
                            if rgb is not None:
                                save_ppm(output/f'{trial}_camera_{index}.ppm', rgb)
                        for obs in observations['multi_view']:
                            cloud = source.get_point_cloud(obs.object_id)
                            if cloud is not None:
                                save_ply(output/'clouds'/f'{trial}_{obs.object_id}.ply', cloud)
                    (output/'rows.json').write_text(json.dumps(rows, indent=2))
                    (output/'detections.json').write_text(json.dumps(detections, indent=2))
                    (output/'timings.json').write_text(json.dumps(timings, indent=2))
                    print(trial, detections[-3:], flush=True)
        summary = {}
        for method in ('v1_A', 'v1_B', 'multi_view'):
            subset = [r for r in rows if r['method'] == method]
            counts = [r for r in detections if r['method'] == method]
            summary[method] = dict(expected=len(subset), detected=sum(r['detected'] for r in subset),
                correct=sum(r['correct'] for r in subset), unknown=sum(r['unknown'] for r in subset),
                false_positives=sum(r['false_positives'] for r in counts),
                duplicates=sum(r['duplicates'] for r in counts), missing=sum(r['missing'] for r in counts))
            summary[method].update(mixed_clusters=sum(r['mixed_clusters'] for r in counts),
                                   mixed_checked=sum(r['mixed_checked'] for r in counts))
            for key in ('center_error_mm', 'dimension_error_mm', 'axis_error_deg'):
                values = [r[key] for r in subset if r.get(key) is not None]
                summary[method][key] = None if not values else {
                    'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'samples': len(values)}
        latency = {key: {'mean': float(np.mean([r[key] for r in timings])),
                         'p95': float(np.percentile([r[key] for r in timings], 95))}
                   for key in ('render_ms', 'acquisition_ms', 'projection_ms', 'fusion_ms',
                               'segmentation_ms', 'fitting_ms', 'total_ms', 'elapsed_ms')}
        write_result(args.result, status='PASS', metrics=dict(kind='automatic_detection_experiment',
            trials=len(detections)//3, summary=summary, latency_ms=latency,
            effective_update_hz=1000/latency['elapsed_ms']['mean'],
            process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            robot_commands=0, clutter_note='Elevated moving primitive surrogate; not exhaustive robot-link validation'))
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
