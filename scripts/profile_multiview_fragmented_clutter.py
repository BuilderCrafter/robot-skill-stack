"""Analytic-depth CPU/allocation comparison for many objects and fragments.

Fixture truth constructs depth images only. Both unmodified providers receive
only calibrated DepthFrames and the ordinary workspace configuration: no masks,
labels or point clouds are injected. Three independent analytic renders precede
each measurement; their explicit synthetic fixture clock is not native freshness
evidence. Rendering, fixture checks, I/O and allocation tracing are not timed.
Run in a quiet serialized CPU window; this is not native update frequency.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.segmentation import _metric_components, cluster_points
from scripts.benchmark_multiview_robustness import ROOT
from scripts.profile_multiview_observability import allocation_peak, hashes, write_json
from tests.support.multiview_iteration2 import SnapshotRig, distribution
from tests.support.primitive_scene import Solid, render


EYES = ((1.49, -.15, 1.13), (.19, -.74, .66))


def supported_objects(count):
    rows = (-.23,) if count == 4 else (-.23, .23)
    solids = []
    for y in rows:
        solids.extend((
            Solid('cube', center=(.27, y, .020), size=(.044, .036, .040), yaw=.18),
            Solid('sphere', center=(.39, y, .023), radius=.023),
            Solid('cylinder', center=(.51, y, .028), radius=.018, length=.056),
            Solid('cylinder', center=(.64, y, .016), radius=.016, length=.068,
                  axis=(float(np.cos(.23)), float(np.sin(.23)), 0.)),
        ))
    return solids


def fixtures(seed):
    rng = np.random.default_rng(seed)
    isolated = []
    for x in np.linspace(.235, .665, 8):
        for y in np.linspace(-.335, .335, 8):
            height = float(rng.uniform(.035, .045))
            isolated.append(Solid('cube', center=(float(x), float(y), height/2),
                                  size=(.018, .011, height)))
    panels = []
    # Six millimeter pitch joins broad 8 mm grouping but offers finer 4 mm
    # partitions. Thin planar fragments intentionally need not be known boxes.
    for x in (.25, .37, .49, .61):
        for y in (-.13, .02, .17, .32):
            for offset in (-.009, -.003, .003, .009):
                height = float(rng.uniform(.031, .039))
                panels.append(Solid('cube', center=(x, y+offset, height/2), size=(.014, .0015, height)))
    return dict(supported4=supported_objects(4), supported8=supported_objects(8),
                disconnected64=isolated,
                supported4_fragmented64=supported_objects(4)+panels)


def render_frames(solids, *, seed, resolution):
    snapshots = []
    for index in range(3):
        pair = []
        for camera, eye in enumerate(EYES):
            raw = render(solids, eye=eye, resolution=resolution, seed=seed+10*index+camera)
            intrinsics = raw.intrinsics.copy()
            intrinsics[0, 2] += .5
            intrinsics[1, 2] += .5
            pair.append(DepthFrame(str(camera), (index+1., (index+1.)*.1),
                raw.depth.astype(float), intrinsics, raw.world_from_camera,
                100.+index*.1, raw.rgb))
        snapshots.append(tuple(pair))
    return snapshots


def input_diagnostics(frames, scene, output):
    clouds = [voxelize(crop_workspace(project_depth(frame, stride=2,
        bounds=(scene.workspace_min, scene.workspace_max))[0], scene), .002, 12000) for frame in frames]
    fused = fuse_clouds(clouds, voxel=.002, max_points=12000)
    broad = cluster_points(fused)
    fine = _metric_components(fused)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fused=fused, **{f'camera_{i}': cloud for i, cloud in enumerate(clouds)})
    return dict(per_camera_points=[len(cloud) for cloud in clouds], fused_points=len(fused),
                broad_components=len(broad), metric4_components=len(fine),
                broad_component_points=[len(cloud) for cloud in broad],
                metric_component_points=[len(cloud) for cloud in fine])


def measure(factory, frames, scene, *, timed=True):
    rig = SnapshotRig()
    provider = factory(rig, scene)
    try:
        # Three distinct independently rendered analytic snapshots exercise the
        # existing confirmation counter; inputs are identical across methods.
        for pair in frames[:2]:
            rig.push(pair)
            provider.observe()
        rig.push(frames[2])
        started = time.perf_counter() if timed else None
        batch = provider.observe()
        elapsed = (time.perf_counter()-started)*1000 if timed else None
        diagnostics = dict(provider.diagnostics)
        if diagnostics['numerical_fits'] > 24 or diagnostics['detections'] > 8:
            raise RuntimeError('Existing 24-fit/8-detection bound violated')
        objects = [dict(shape=obj.geometry.shape.value, graspable=bool(obj.graspable),
                        center=obj.pose.position.tolist(), size=obj.size.tolist()) for obj in batch.observations]
        return dict(elapsed_ms=elapsed, diagnostics=diagnostics, published=len(objects), outputs=objects)
    finally:
        provider.close()


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError('Use the unchanged wrapper with NumPy 1.26.4')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Choose a fresh output directory to preserve previous measurements')
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    baseline = importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider
    methods = dict(baseline=baseline, current=MultiViewProvider)
    source_paths = [*list((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py')),
        Path(__file__), ROOT/'tests/support/primitive_scene.py',
        ROOT/'scripts/profile_multiview_observability.py']
    start = hashes(source_paths)
    write_json(args.output/'provenance_start.json', start)
    scene = load_scene_config(ROOT/'config/scenes/multiview_playground.toml').scene
    rows, allocations, manifest, fixture_errors = [], [], [], []
    for index, (name, solids) in enumerate(fixtures(args.seed).items()):
        if args.case and name not in args.case:
            continue
        print(json.dumps(dict(progress='rendering analytic fixture', case=name, solids=len(solids))), flush=True)
        frames = render_frames(solids, seed=args.seed+index*100, resolution=tuple(args.resolution))
        directory = args.output/'inputs'/name
        directory.mkdir(parents=True, exist_ok=True)
        frame_files = []
        for fi, pair in enumerate(frames):
            for ci, frame in enumerate(pair):
                path = directory/f'frame_{fi:02d}_camera_{ci}.npz'
                np.savez_compressed(path, depth=frame.depth, intrinsics=frame.intrinsics,
                    world_from_camera=frame.world_from_camera, token=frame.token, acquired_at=frame.acquired_at)
                frame_files.append(path)
        counts = input_diagnostics(frames[-1], scene, directory/'workspace.npz')
        if name == 'disconnected64' and counts['broad_components'] < 24:
            fixture_errors.append(dict(case=name, reason='Fewer than 24 measured disconnected components', counts=counts))
        manifest.append(dict(case=name, fixture_solids=[asdict(solid) for solid in solids], measured=counts,
            frame_sha256=hashes(frame_files), analytic_fixture_clock=True))
        write_json(args.output/'fixture_manifest.json', manifest)
        for method, factory in methods.items():
            for _ in range(args.warmup):
                measure(factory, frames, scene)
        for repeat in range(args.repeats):
            order = ('baseline', 'current') if repeat % 2 == 0 else ('current', 'baseline')
            for method in order:
                rows.append(dict(case=name, method=method, repeat=repeat,
                                 **measure(methods[method], frames, scene)))
            if repeat % 5 == 0:
                print(json.dumps(dict(progress='paired repeat complete', case=name, repeat=repeat)), flush=True)
        # Allocation measurements never overlap a timed source call.
        if args.allocations:
            for method, factory in methods.items():
                allocations.append(dict(case=name, method=method,
                    **allocation_peak(lambda factory=factory: measure(factory, frames, scene, timed=False))))
        write_json(args.output/'rows.json', rows)
        write_json(args.output/'allocations.json', allocations)
    stages = ('acquisition_ms', 'projection_ms', 'fusion_ms', 'segmentation_ms', 'fitting_ms',
              'observability_ms', 'tracking_ms', 'total_ms', 'numerical_fits', 'detections',
              'fit_budget_exhaustions', 'workspace_points', 'clusters')
    summary = {}
    for entry in manifest:
        for method in methods:
            selected = [row for row in rows if row['case'] == entry['case'] and row['method'] == method]
            summary[f'{entry["case"]}/{method}'] = dict(elapsed_ms=distribution([row['elapsed_ms'] for row in selected]),
                **{stage: distribution([row['diagnostics'].get(stage, 0) for row in selected]) for stage in stages})
    end = hashes(source_paths)
    write_json(args.output/'provenance_end.json', end)
    result = dict(status='PASS' if start == end and not fixture_errors else 'FAIL',
        kind='analytic_rendered_depth_fragmented_clutter_cpu_profile', summary=summary,
        seed=args.seed, resolution=args.resolution, eyes=EYES, warmup=args.warmup, repeats=args.repeats,
        voxel_size_m=.002, stride=2, max_points=12000, max_numerical_fits=24, max_detections=8,
        source_hashes_stable=start == end, fixture_errors=fixture_errors, robot_or_backend_commands=[],
        memory_scope='Separate traced peak for fresh source construction and three-frame sequence; preloaded inputs, native SDK/GPU and process RSS excluded',
        scope='Analytic CPU stress only; renderer and tracing excluded from timing; no native sensor, live freshness, update-frequency or grasp-safety claim')
    write_json(args.output/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--baseline-package', default='baseline_multiview')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=8675309)
    parser.add_argument('--resolution', type=int, nargs=2, default=(320, 240), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--case', action='append', choices=('supported4', 'supported8', 'disconnected64', 'supported4_fragmented64'))
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--allocations', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or min(args.resolution) < 32:
        parser.error('Use positive repeats, nonnegative warmup and image dimensions at least 32')
    try:
        main(args)
    except Exception as error:
        if not (args.output/'result.json').exists():
            write_json(args.output/'result.json', dict(status='FAIL', error=f'{type(error).__name__}: {error}',
                kind='analytic_rendered_depth_fragmented_clutter_cpu_profile'))
        raise
