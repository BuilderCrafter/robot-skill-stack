"""Iteration-two native capture and matched replay; original benchmarks unchanged.

Capture native Isaac frames once, then feed identical arrays to iteration one,
iteration two and unchanged V1 A/B. Replay CPU times are not live throughput.
PASS means the experiment completed; every detection failure remains in rows.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, save_ply, voxelize
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from tests.support.multiview_iteration2 import (
    EYES, SEEDS, SnapshotCamera, SnapshotRig, distribution, object_from_payload, object_payload,
    perturb_frames, robustness_scenes, scene_objects, score_observations, surface_memberships,
)
from tests.support.results import fail_result, save_ppm, write_result


ROOT = Path(__file__).resolve().parents[1]
STRESS_CASES = {'gap_4_box_cylinder', 'gap_2_sphere_sphere', 'isolated', 'unknown', 'strong_A', 'temporal'}
_native_app = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def source_provenance(args):
    tracked = [Path(__file__).resolve(), ROOT/'tests/support/multiview_iteration2.py',
               ROOT/'tests/support/multiview_native.py', ROOT/'tests/support/multiview_research.py',
               ROOT/'scenes/multiview_playground.usda', ROOT/'config/scenes/multiview_playground.toml',
               ROOT/'robot_skill_stack/integrations/isaac/world/multiview.py',
               ROOT/'robot_skill_stack/integrations/isaac/sensors/rgbd_camera.py']
    tracked.extend(sorted((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py')))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    baseline_hashes = {}
    if args.baseline_pythonpath:
        baseline = Path(args.baseline_pythonpath).resolve()/args.baseline_package.replace('.', '/')
        baseline_hashes = {str(path.relative_to(baseline)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(baseline.glob('*.py'))}
    def git(*arguments):
        return subprocess.run(['git', *arguments], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return dict(git_commit=git('rev-parse', 'HEAD'), tracked_changes=git('status', '--short', '--untracked-files=no'),
                sha256=hashes, baseline_sha256=baseline_hashes, numpy_version=np.__version__,
                mode='saved_native_replay' if args.replay else 'native_capture',
                note='File hashes include uncommitted implementations; start/end files detect concurrent changes.')


def load_baseline(args):
    if args.baseline_pythonpath is None:
        raise ValueError('--baseline-pythonpath is required for matched baseline/improved evaluation')
    sys.path.insert(0, str(Path(args.baseline_pythonpath).resolve()))
    module = importlib.import_module(f'{args.baseline_package}.provider')
    return module.MultiViewProvider


def populate_scene(objects):
    # Extend only this new research fixture to unsupported anisotropic geometry.
    from tests.support.multiview_native import populate
    from pxr import Gf, UsdGeom
    import omni.usd
    populate([obj for obj in objects if obj.shape != 'ellipsoid'])
    stage = omni.usd.get_context().get_stage()
    for obj in objects:
        if obj.shape != 'ellipsoid':
            continue
        shape = UsdGeom.Sphere.Define(stage, '/World/MultiViewObjects/'+obj.name)
        shape.CreateRadiusAttr(1.)
        xform = UsdGeom.Xformable(shape.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*obj.center.tolist()))
        xform.AddScaleOp().Set(Gf.Vec3f(*(obj.size/2).tolist()))
        UsdGeom.Gprim(shape.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(.7, .4, .3)])


def save_frames(output, key, index, frames):
    directory = output/'captures'/key
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for camera_index, frame in enumerate(frames):
        path = directory/f'frame_{index:02d}_camera_{camera_index}.npz'
        rgb = frame.rgb if frame.rgb is not None else np.zeros((*frame.depth.shape, 3), np.uint8)
        np.savez_compressed(path, camera_id=frame.camera_id, token=frame.token, depth=frame.depth,
            intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera,
            acquired_at=frame.acquired_at, rgb=rgb)
        files.append(str(path.relative_to(output)))
        if index == 2 and frame.rgb is not None:
            save_ppm(directory/f'camera_{camera_index}.ppm', frame.rgb)
    return files


def load_frames(root, files):
    result = []
    for filename in files:
        with np.load(root/filename, allow_pickle=False) as data:
            result.append(DepthFrame(str(data['camera_id']), tuple(data['token'].tolist()), data['depth'],
                data['intrinsics'], data['world_from_camera'], float(data['acquired_at']), data['rgb']))
    return tuple(result)


def capture_workspace(output, key, index, frames, objects, scene):
    """Save geometry for alternatives before selecting any production method."""
    directory = output/'captures'/key
    clouds = [voxelize(crop_workspace(project_depth(frame)[0], scene), .002, 12000) for frame in frames]
    fused = fuse_clouds(clouds, voxel=.002, max_points=12000)
    np.savez_compressed(directory/f'workspace_frame_{index:02d}.npz',
                        **{f'camera_{i}': cloud for i, cloud in enumerate(clouds)}, fused=fused)
    visible = {}
    for ci, cloud in enumerate(clouds):
        save_ply(directory/f'camera_{ci}_workspace_frame_{index:02d}.ply', cloud)
        labels = surface_memberships(cloud, objects)
        visible[str(ci)] = {obj.name: int(np.count_nonzero(labels == oi)) for oi, obj in enumerate(objects)}
    save_ply(directory/f'fused_workspace_frame_{index:02d}.ply', fused)
    labels = surface_memberships(fused, objects)
    oracle_rows = []
    for oi, obj in enumerate(objects):
        path = directory/f'oracle_{obj.name}_frame_{index:02d}.ply'
        selected = fused[labels == oi]
        save_ply(path, selected)
        oracle_rows.append(dict(path=str(path.relative_to(output)), points=len(selected), object=object_payload(obj),
                                purpose='research_oracle_fitting_only'))
    return dict(visible_surface_points=visible, oracle_clouds=oracle_rows)


def validate_gap_fixture(record):
    """Rendering preconditions, separate from any algorithm acceptance score."""
    if record['gap_mm'] is None:
        return []
    errors = []
    for frame in record['frames']:
        if frame['index'] < 2:
            continue
        visible = frame.get('visible_surface_points', {})
        for obj in frame['objects']:
            count = sum(camera.get(obj['name'], 0) for camera in visible.values())
            if count < 8:
                errors.append(dict(trial=record['key'], frame_index=frame['index'], object=obj['name'],
                    kind='native_fixture_target_not_observed', surface_points_in_camera_union=count,
                    required_surface_points=8,
                    note='Invalid rendered fixture; never counted as an algorithm miss.'))
    return errors


class BorrowedRig:
    def __init__(self, rig): self.rig = rig
    def capture(self): return self.rig.capture()
    @property
    def status(self): return self.rig.status
    def close(self): pass


def capture_native(args, output, baseline_class):
    global _native_app
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': False})
    _native_app = app
    rig = None
    timings = []
    try:
        from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
        from tests.support.multiview_native import CAMERAS, open_experiment, set_viewpoint
        world = open_experiment(app)
        cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
        rig = create_cameras(CAMERAS)
        for _ in range(60):
            world.step(render=True)
        scenes = robustness_scenes(args.split)
        if args.scenario:
            scenes = [scene for scene in scenes if scene.name in args.scenario]
        if args.limit:
            scenes = scenes[:args.limit]
        manifest = dict(kind='iteration2_native_snapshots', seeds=SEEDS, viewpoints=EYES,
            numpy_version=np.__version__, comparison='identical saved native frames, original acquisition times',
            native_render_steps_between_captures=8, provenance_file='provenance_start.json', scenes=[])
        for scene in scenes:
            for camera, eye in zip(rig.cameras.values(), EYES[scene.split]):
                set_viewpoint(camera, eye)
            record = dict(key=scene.key, name=scene.name, split=scene.split, seed=scene.seed,
                gap_mm=scene.gap_mm, temporal=scene.temporal, note=scene.note,
                clutter=[object_payload(obj) for obj in scene.clutter], frames=[])
            for index in range(6 if scene.temporal else 3):
                objects = scene_objects(scene, index)
                populate_scene(objects+scene.clutter)
                for _ in range(8):
                    world.step(render=True)
                frames = rig.capture()
                if frames is None:
                    raise RuntimeError(f'Native camera buffers did not advance for {scene.key}/{index}')
                files = save_frames(output, scene.key, index, frames)
                snapshot = dict(index=index, files=files, objects=[object_payload(obj) for obj in objects])
                if index >= 2:
                    snapshot.update(capture_workspace(output, scene.key, index, frames, objects, cfg.scene))
                record['frames'].append(snapshot)
            record['fixture_errors'] = validate_gap_fixture(record)
            manifest['scenes'].append(record)
            write_json(output/'capture_manifest.json', manifest)
            print('captured', scene.key, flush=True)
            if record['fixture_errors']:
                print('INVALID FIXTURE', scene.key, record['fixture_errors'], flush=True)
            # Native timing is a separate experiment with each source alone and
            # real acquisition/rendering. No V1 replay work is timed as throughput.
            if args.native_timing_samples and scene.name in STRESS_CASES and not scene.temporal:
                populate_scene(scene.objects+scene.clutter)
                classes = [('improved', MultiViewProvider)]
                if baseline_class is not None:
                    classes.insert(0, ('baseline', baseline_class))
                for method, provider_class in classes:
                    source = provider_class(BorrowedRig(rig), cfg.scene)
                    try:
                        for sample in range(3+args.native_timing_samples):
                            started = time.perf_counter()
                            for _ in range(8):
                                world.step(render=True)
                            rendered = time.perf_counter()
                            source.observe()
                            finished = time.perf_counter()
                            if sample >= 3:
                                timings.append(dict(trial=scene.key, method=method, sample=sample-3,
                                    render_ms=(rendered-started)*1000, elapsed_ms=(finished-started)*1000,
                                    measurement='live_native_render_plus_provider', **source.diagnostics))
                    finally:
                        source.close()
                write_json(output/'native_timings.json', timings)
        return manifest, timings
    finally:
        if rig is not None:
            rig.close()


def workspace_exports(output, trial, frames, objects, clutter, scene, package):
    clouds = [voxelize(crop_workspace(project_depth(frame)[0], scene), .002, 12000) for frame in frames]
    fused = fuse_clouds(clouds, voxel=.002, max_points=12000)
    directory = output/'clouds'/trial
    directory.mkdir(parents=True, exist_ok=True)
    visible = {}
    oracle_rows = []
    for index, cloud in enumerate(clouds):
        save_ply(directory/f'camera_{index}_workspace.ply', cloud)
        labels = surface_memberships(cloud, objects)
        visible[str(index)] = {obj.name: int(np.count_nonzero(labels == oi)) for oi, obj in enumerate(objects)}
    save_ply(directory/'fused_workspace.ply', fused)
    labels = surface_memberships(fused, objects)
    for oi, obj in enumerate(objects):
        path = directory/f'oracle_{obj.name}.ply'
        selected = fused[labels == oi]
        save_ply(path, selected)
        oracle_rows.append(dict(trial=trial, path=str(path.relative_to(output)), points=len(selected),
                                object=object_payload(obj), purpose='research_oracle_fitting_only'))
    # Reconstruct the package's own grouping on exactly the saved workspace;
    # scoring truth never affects the clusters or the automatic observations.
    grouping = importlib.import_module(f'{package}.segmentation')
    coarse_clusters = grouping.cluster_points(fused)
    coarse_mixed = 0
    for ci, cloud in enumerate(coarse_clusters):
        membership = surface_memberships(cloud, objects)
        significant = sum(np.count_nonzero(membership == oi) >= max(8, .10*len(cloud))
                          for oi in range(len(objects)))
        coarse_mixed += int(significant > 1)
        save_ply(directory/f'{package.rsplit(".", 1)[-1]}_coarse_{ci}.ply', cloud)
    clusters = coarse_clusters
    if hasattr(grouping, 'refine_cluster'):
        fitter = importlib.import_module(f'{package}.fitting').SurfaceFitter()
        clusters = [child for cloud in sorted(coarse_clusters, key=len, reverse=True)[:16]
                    for child, _ in grouping.refine_cluster(cloud, fitter,
                                                           support_plane_z=scene.support_plane_z)]
    mixed_candidates, rejected_mixed, candidate_rows = 0, 0, []
    for ci, cloud in enumerate(clusters):
        membership = surface_memberships(cloud, objects)
        significant = [obj.name for oi, obj in enumerate(objects)
                       if np.count_nonzero(membership == oi) >= max(8, .10*len(cloud))]
        supported, reason = grouping.supported_object_cloud(cloud, scene.support_plane_z)
        mixed = len(significant) > 1
        mixed_candidates += int(mixed)
        rejected_mixed += int(mixed and not supported)
        path = directory/f'{package.rsplit(".", 1)[-1]}_candidate_{ci}.ply'
        save_ply(path, cloud)
        candidate_rows.append(dict(path=str(path.relative_to(output)), points=len(cloud),
            truth_members=significant, supported=supported, support_reason=reason))
    return dict(visible_surface_points=visible, coarse_mixed_candidate_clusters=coarse_mixed,
                refinement_replayed=hasattr(grouping, 'refine_cluster'),
                mixed_candidate_clusters=mixed_candidates,
                rejected_mixed_candidate_clusters=rejected_mixed, candidate_clusters=candidate_rows), oracle_rows


def summarize(rows, counts, timings):
    summaries = {}
    for method in sorted({row['method'] for row in counts}):
        selected_rows = [row for row in rows if row['method'] == method]
        selected_counts = [row for row in counts if row['method'] == method]
        summary = {key: sum(row.get(key, 0) for row in selected_counts) for key in
                   ('expected', 'observed', 'matched', 'missing', 'false_positives', 'mixed_accepted_clouds',
                    'false_splits', 'duplicate_ids', 'identity_switches')}
        summary.update(correct=sum(row['correct'] for row in selected_rows),
                       unknown=sum(row['unknown'] for row in selected_rows),
                       known_expected=sum(row['truth'] != 'unknown' for row in selected_rows),
                       known_correct=sum(row['correct'] and row['truth'] != 'unknown' for row in selected_rows),
                       expected_unknown=sum(row['truth'] == 'unknown' for row in selected_rows),
                       correct_unknown=sum(row['correct'] and row['truth'] == 'unknown' for row in selected_rows),
                       supported_shape_rejected_unknown=sum(row['unknown'] and row['truth'] != 'unknown'
                                                            for row in selected_rows),
                       unsupported_shape_overclassified=sum(row['truth'] == 'unknown' and
                           row['predicted'] not in ('unknown', 'missing') for row in selected_rows),
                       trials=len({row['trial'] for row in selected_counts}))
        for key in ('center_error_mm', 'dimension_error_mm', 'axis_error_deg', 'radius_error_mm', 'length_error_mm',
                    'box_yaw_error_deg', 'cylinder_axis_error_deg', 'confidence', 'score_margin'):
            summary[key] = distribution([row.get(key) for row in selected_rows])
        summary['offline_replay_processing_ms'] = distribution(
            [row['elapsed_ms'] for row in timings if row['method'] == method])
        for correct in (True, False):
            for key in ('confidence', 'score_margin'):
                summary[f'{key}_on_{"correct" if correct else "incorrect"}'] = distribution(
                    [row.get(key) for row in selected_rows if row['correct'] == correct])
        summaries[method] = summary
    return summaries


def replay(args, capture_root, output, manifest, baseline_class):
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    rows, counts, timings, oracle_manifest, workspace_diagnostics, contract_rows, runtime_errors = [], [], [], [], [], [], []
    variants = args.variants.split(',')
    providers_count = 2 if args.no_v1 else 4
    for record in manifest['scenes']:
        for variant in variants:
            if variant != 'clean' and record['name'] not in STRESS_CASES:
                continue
            trial = record['key']+'_'+variant
            first = perturb_frames(load_frames(capture_root, record['frames'][0]['files']), variant, seed=record['seed'])
            rigs = {key: SnapshotRig() for key in ('baseline', 'improved')}
            providers = {'baseline': baseline_class(rigs['baseline'], cfg.scene),
                         'improved': MultiViewProvider(rigs['improved'], cfg.scene)}
            cameras = {}
            if not args.no_v1:
                cameras = {f'v1_{key}': SnapshotCamera(frame) for key, frame in zip(('A', 'B'), first)}
                providers.update({method: build_perception_provider(camera, cfg) for method, camera in cameras.items()})
            previous = {method: {} for method in providers}
            try:
                for frame_record in record['frames']:
                    index = frame_record['index']
                    frames = perturb_frames(load_frames(capture_root, frame_record['files']), variant,
                                            seed=record['seed']+index)
                    objects = tuple(object_from_payload(row) for row in frame_record['objects'])
                    observations, provider_errors = {}, {}
                    for method, source in providers.items():
                        if method in rigs:
                            rigs[method].push(frames)
                        else:
                            cameras[method].push(frames[0 if method == 'v1_A' else 1])
                        started = time.perf_counter()
                        try:
                            observations[method] = source.observe().observations
                        except Exception as error:
                            observations[method] = []
                            provider_errors[method] = f'{type(error).__name__}: {error}'
                            runtime_errors.append(dict(trial=trial, method=method, frame_index=index,
                                stage='automatic_observe', error=provider_errors[method], traceback=traceback.format_exc()))
                        elapsed = (time.perf_counter()-started)*1000
                        timing = dict(trial=trial, method=method, frame_index=index, elapsed_ms=elapsed,
                                      measurement='offline_saved_native_frame_replay')
                        if method in rigs:
                            timing.update(source.diagnostics)
                        timings.append(timing)
                        if method in rigs and method not in provider_errors:
                            expected_timestamp = min(frame.acquired_at for frame in frames)
                            try:
                                repeated = source.observe()
                            except Exception as error:
                                runtime_errors.append(dict(trial=trial, method=method, frame_index=index,
                                    stage='unchanged_frame_contract', error=f'{type(error).__name__}: {error}',
                                    traceback=traceback.format_exc()))
                                repeated = None
                            contract_rows.append(dict(trial=trial, method=method, frame_index=index,
                                split=record['split'], variant=variant,
                                confirmation_early_observations=len(observations[method]) if index < 2 else 0,
                                stale_timestamp_observations=sum(
                                    obs.timestamp != expected_timestamp for obs in observations[method]),
                                repeated_frame_publications=0 if repeated is None else len(repeated.observations),
                                repeated_frame_timestamp=None if repeated is None else repeated.timestamp,
                                repeated_frame_error=repeated is None))
                    if index < 2:
                        continue
                    for method, source in providers.items():
                        current = observations[method]
                        clouds = [source.get_point_cloud(obs.object_id) for obs in current]
                        selected_rows, selected_counts, current_ids = score_observations(objects, current, clouds,
                            method=method, trial=trial, frame_index=index, previous_ids=previous[method])
                        previous[method].update(current_ids)
                        for row in selected_rows:
                            row.update(split=record['split'], scenario=record['name'], variant=variant,
                                       seed=record['seed'], gap_mm=record['gap_mm'])
                        selected_counts.update(split=record['split'], scenario=record['name'], variant=variant,
                                               gap_mm=record['gap_mm'], provider_error=provider_errors.get(method))
                        if method in provider_errors:
                            for row in selected_rows:
                                row['provider_error'] = provider_errors[method]
                        rows.extend(selected_rows)
                        counts.append(selected_counts)
                        for obs, cloud in zip(current, clouds):
                            if cloud is not None:
                                save_ply(output/'clouds'/trial/f'{method}_frame_{index}_{obs.object_id}.ply', cloud)
                        write_json(output/'diagnostics'/trial/f'{method}_frame_{index}.json', source.diagnostics)
                    if index == 2 or record['temporal']:
                        clutter = [object_from_payload(row) for row in record['clutter']]
                        for method, package in (('baseline', args.baseline_package),
                                                ('improved', 'robot_skill_stack.world.perception_multiview')):
                            try:
                                details, oracle_rows = workspace_exports(output, f'{trial}_frame_{index}', frames,
                                                                        objects, clutter, cfg.scene, package)
                            except Exception as error:
                                runtime_errors.append(dict(trial=trial, method=method, frame_index=index,
                                    stage='scoring_grouping_diagnostics', error=f'{type(error).__name__}: {error}',
                                    traceback=traceback.format_exc()))
                                details, oracle_rows = {'diagnostics_error': str(error)}, []
                            workspace_diagnostics.append(dict(trial=trial, method=method, frame_index=index, **details))
                            if method == 'improved':
                                oracle_manifest.extend(oracle_rows)
                # Save after every scene/variant; failed detections are retained,
                # and interrupted later scenes cannot erase earlier evidence.
                write_json(output/'rows.json', rows)
                write_json(output/'detections.json', counts)
                write_json(output/'replay_timings.json', timings)
                write_json(output/'workspace_diagnostics.json', workspace_diagnostics)
                write_json(output/'oracle_cloud_manifest.json', oracle_manifest)
                write_json(output/'source_contract_checks.json', contract_rows)
                write_json(output/'runtime_errors.json', runtime_errors)
                print('evaluated', trial, f'({providers_count} methods)', flush=True)
            finally:
                for source in providers.values():
                    source.close()
    for name, data in (('rows', rows), ('detections', counts), ('replay_timings', timings),
                        ('workspace_diagnostics', workspace_diagnostics), ('source_contract_checks', contract_rows),
                        ('runtime_errors', runtime_errors)):
        write_csv(output/f'{name}.csv', data)
    by_split_variant, per_scenario = {}, {}
    for split in ('selection', 'holdout'):
        for variant in variants:
            subset = [row for row in counts if row['split'] == split and row['variant'] == variant]
            if subset:
                keys = {row['trial'] for row in subset}
                by_split_variant[f'{split}/{variant}'] = summarize(
                    [row for row in rows if row['trial'] in keys], subset,
                    [row for row in timings if row['trial'] in keys])
                for scenario in sorted({row['scenario'] for row in subset}):
                    scenario_counts = [row for row in subset if row['scenario'] == scenario]
                    scenario_keys = {row['trial'] for row in scenario_counts}
                    per_scenario[f'{split}/{variant}/{scenario}'] = summarize(
                        [row for row in rows if row['trial'] in scenario_keys], scenario_counts,
                        [row for row in timings if row['trial'] in scenario_keys])
    contract_summary = {key: sum(row[key] for row in contract_rows) for key in
                        ('confirmation_early_observations', 'stale_timestamp_observations',
                         'repeated_frame_publications')}
    contract_summary['checks'] = len(contract_rows)
    contract_summary['repeated_frame_nonempty_timestamp'] = sum(
        row['repeated_frame_timestamp'] is not None for row in contract_rows)
    visibility_coverage = []
    for record in manifest['scenes']:
        if not record['name'].startswith('one_view_'):
            continue
        hidden_camera = '0' if record['name'].endswith('_A') else '1'
        other_camera = '1' if hidden_camera == '0' else '0'
        for snapshot in record['frames']:
            if snapshot['index'] < 2 or 'visible_surface_points' not in snapshot:
                continue
            visible = snapshot['visible_surface_points']
            hidden_points = sum(visible[hidden_camera].values())
            other_points = sum(visible[other_camera].values())
            visibility_coverage.append(dict(trial=record['key'], frame_index=snapshot['index'],
                hidden_camera=hidden_camera, hidden_points=hidden_points, other_points=other_points,
                one_view_condition_met=hidden_points == 0 and other_points >= 8))
    return dict(summary=summarize(rows, counts, timings), per_split_variant=by_split_variant,
                per_scenario=per_scenario, one_view_coverage=visibility_coverage,
                source_contract_checks=contract_summary,
                runtime_error_count=len(runtime_errors),
                runtime_errors_by_stage={stage: sum(row['stage'] == stage for row in runtime_errors)
                    for stage in {row['stage'] for row in runtime_errors}},
                native_scenes=len(manifest['scenes']), replay_variants=len({row['trial'] for row in counts}),
                expected_rows=len(rows), failure_rows=sum(not row['correct'] for row in rows),
                scoring='One-to-one center matching <40 mm; exclusive nearest surface <=1.5 mm; '
                        'mixed/false-split support >=10% and >=8 points; every failure retained',
                oracle_clouds='Scoring/fitting diagnostics only, never automatic provider inputs',
                native_visibility='Physical occluders; inspect per-camera visible_surface_points before claiming one-view coverage')


def main(args):
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'provenance_start.json', source_provenance(args))
    baseline = None if args.capture_only and not args.baseline_pythonpath else load_baseline(args)
    if args.replay:
        capture_root = Path(args.replay).resolve()
        manifest = json.loads((capture_root/'capture_manifest.json').read_text())
        if args.split != 'both':
            manifest['scenes'] = [record for record in manifest['scenes'] if record['split'] == args.split]
        if args.scenario:
            manifest['scenes'] = [record for record in manifest['scenes'] if record['name'] in args.scenario]
        if args.limit:
            manifest['scenes'] = manifest['scenes'][:args.limit]
        native_timings = []
    else:
        capture_root = output
        manifest, native_timings = capture_native(args, output, baseline)
    if not manifest['scenes']:
        raise ValueError('No benchmark scenes selected')
    fixture_errors = [error for record in manifest['scenes'] for error in validate_gap_fixture(record)]
    write_json(output/'fixture_validation.json', dict(errors=fixture_errors,
        checked_gap_scenes=sum(record['gap_mm'] is not None for record in manifest['scenes'])))
    if fixture_errors:
        write_result(args.result, status='FAIL', metrics=dict(kind='invalid_native_fixture_capture',
            invalid_targets=len(fixture_errors), invalid_scenes=len({error['trial'] for error in fixture_errors}),
            scenes=len(manifest['scenes']), automatic_algorithm_evaluation_run=False,
            note='Saved frames retained. Failed rendering preconditions are fixture failures, not detector misses.'),
            error='Native gap targets have fewer than8 scoring surface points in the union of healthy cameras')
        return 1
    if args.capture_only:
        write_result(args.result, status='PASS', metrics=dict(kind='native_capture_only',
            scenes=len(manifest['scenes']), manifest=str(output/'capture_manifest.json'),
            note='No accuracy/throughput claim; original native snapshots retained'))
        return
    metrics = replay(args, capture_root, output, manifest, baseline)
    metrics.update(kind='iteration2_matched_robustness_experiment', baseline_package=args.baseline_package,
                   baseline_pythonpath=str(Path(args.baseline_pythonpath).resolve()),
                   robot_commands=0, experiment_completed=True,
                   pass_note='PASS records experiment completion, not an accuracy acceptance gate.')
    if native_timings:
        metrics['live_native_latency_ms'] = {
            method: {key: distribution([row.get(key) for row in native_timings if row['method'] == method])
                     for key in ('render_ms', 'acquisition_ms', 'projection_ms', 'fusion_ms', 'segmentation_ms',
                                 'fitting_ms', 'tracking_ms', 'total_ms', 'elapsed_ms')}
            for method in {row['method'] for row in native_timings}}
        metrics['live_effective_hz'] = {
            method: 1000/values['elapsed_ms']['mean']
            for method, values in metrics['live_native_latency_ms'].items()}
    status = 'FAIL' if metrics['runtime_error_count'] else 'PASS'
    write_result(args.result, status=status, metrics=metrics,
                 error=None if status == 'PASS' else 'Runtime errors retained in runtime_errors.json and missed-object rows')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--baseline-pythonpath')
    parser.add_argument('--baseline-package', default='baseline_multiview')
    parser.add_argument('--replay', help='Directory containing capture_manifest.json; does not launch Isaac')
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--split', choices=('selection', 'holdout', 'both'), default='both')
    parser.add_argument('--scenario', action='append', help='Optional exact scenario name; may be repeated')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--native-timing-samples', type=int, default=0)
    parser.add_argument('--variants', default='clean,depth_noise,calibration,sparse')
    parser.add_argument('--no-v1', action='store_true', help='Quick research replay; final comparison should retain V1')
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error('--limit must be positive')
    if arguments.native_timing_samples < 0:
        parser.error('--native-timing-samples must be nonnegative')
    if not set(arguments.variants.split(',')) <= {'clean', 'depth_noise', 'calibration', 'sparse'}:
        parser.error('Unknown --variants entry')
    try:
        exit_code = main(arguments)
    except Exception as error:
        fail_result(arguments.result, error)
        raise SystemExit(1)
    finally:
        try:
            write_json(Path(arguments.artifacts_dir)/'provenance_end.json', source_provenance(arguments))
        finally:
            # Kit can terminate the interpreter inside close(). Write every
            # result/provenance artifact first, as the existing native scripts do.
            if _native_app is not None:
                _native_app.close()
    raise SystemExit(exit_code or 0)
