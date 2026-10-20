"""Bounded native camera study and matched, simulator-free perception replay.

Each layout sees the same physical scenes. Baseline/current/single-view methods
share exact saved images WITHIN a layout; different layouts necessarily have
different images. PASS means the experiment and source contracts completed,
not 95% accuracy or manipulation authorization. No robot backend is created.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.clouds import save_ply
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from scripts.benchmark_multiview_robustness import (
    BorrowedRig, capture_workspace, load_frames, populate_scene, save_frames, write_csv,
)
from tests.support.multiview_final_evaluation import (
    PerceptionSession, STAGES, latest_scene_summary, plain, settle_physics, summarize, temporal_recovery,
)
from tests.support.multiview_iteration2 import distribution, object_from_payload, object_payload


FIXTURE = 'tests/support/multiview_final_research.py'
METHODS = ('current_fused', 'baseline_fused', 'current_single_a', 'current_single_b', 'v1_a', 'v1_b')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2), encoding='utf-8')


def file_hashes():
    paths = sorted((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))
    paths += [ROOT/FIXTURE, Path(__file__).resolve(), ROOT/'tests/support/multiview_final_evaluation.py',
              ROOT/'scenes/multiview_playground.usda', ROOT/'config/scenes/multiview_playground.toml']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def layout_payload(layout):
    return plain(asdict(layout))


def layout_hash(layouts):
    return hashlib.sha256(json.dumps([layout_payload(value) for value in layouts],
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_freeze(path, layouts):
    """Run before generating holdout scenes, including source AND camera layout."""
    if path is None:
        raise ValueError('Holdout generation requires --freeze-manifest with package and selected-layout hashes')
    frozen = json.loads(Path(path).read_text())
    current = file_hashes()
    required = {key: value for key, value in current.items()
                if key.startswith('robot_skill_stack/world/perception_multiview/')}
    differences = [key for key, value in required.items() if frozen.get('sha256', {}).get(key) != value]
    if differences:
        raise ValueError(f'Frozen source differs or omits package files: {differences}')
    actual_layout = layout_hash(layouts)
    if frozen.get('layout_sha256') != actual_layout:
        raise ValueError('Frozen selected layout configuration differs or is omitted')
    return dict(path=str(Path(path).resolve()), manifest_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                package_sha256=required, layout_sha256=actual_layout)


def provenance(args):
    def git(*parts):
        return subprocess.run(['git', *parts], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    baseline = {}
    if args.baseline_pythonpath:
        directory = Path(args.baseline_pythonpath).resolve()/args.baseline_package.replace('.', '/')
        baseline = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob('*.py'))}
    return dict(git_commit=git('rev-parse', 'HEAD'), tracked_changes=git('status', '--short', '--untracked-files=no'),
        sha256=file_hashes(), baseline_sha256=baseline, numpy_version=np.__version__,
        argv=sys.argv, invocation=plain({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}),
        mode='saved_native_replay' if args.replay else 'native_capture', no_robot_backend=True,
        no_planning_or_manipulation=True,
        timing_scope='Offline CPU only; saved original acquisition timestamps, no live Hz.' if args.replay else
            'Source-only live timings, when requested, exclude all comparisons, truth scoring and diagnostic exports.')


def balanced_pairs(scenes, layouts):
    """Cyclic Latin order balances every layout's position within each scene."""
    return [(scene, layouts[(position+index) % len(layouts)])
            for index, scene in enumerate(scenes) for position in range(len(layouts))]


def scene_record(scene, layout):
    return dict(key=f'{scene.key}__{layout.name}', scene_key=scene.key, name=scene.name, split=scene.split,
        seed=scene.seed, temporal=scene.temporal, note=scene.note, requirements=plain(scene.requirements),
        objects=[object_payload(obj) for obj in scene.objects], layout=layout_payload(layout), frames=[])


def scene_from_record(record):
    from tests.support.multiview_final_research import FinalScene
    return FinalScene(name=record['name'], split=record['split'], seed=record['seed'],
        objects=tuple(object_from_payload(obj) for obj in record['objects']),
        temporal=record['temporal'], note=record.get('note', ''), requirements=record.get('requirements', {}))


def apply_layout(rig, layout):
    from pxr import Gf
    from tests.support.multiview_native import set_viewpoint
    matrices = layout.world_from_cameras
    for index, (camera, eye, target) in enumerate(zip(rig.cameras.values(), layout.eyes, layout.targets)):
        matrix = None if matrices is None else matrices[index]
        if matrix is None:
            set_viewpoint(camera, eye, target)
        else:
            matrix = np.asarray(matrix, float)
            quaternion = Gf.Matrix3d(*matrix[:3, :3].T.ravel().tolist()).ExtractRotation().GetQuat()
            camera.camera.set_world_pose(position=matrix[:3, 3], orientation=np.r_[quaternion.GetReal(),
                np.asarray(quaternion.GetImaginary())], camera_axes='ros')


def native_robot_diagnostics(layout):
    """Read actual resolved USD transforms/bounds; never issue robot commands."""
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                     UsdGeom.Tokens.proxy])
    base = stage.GetPrimAtPath('/Franka')
    result = dict(base_valid=bool(base.IsValid()), transforms={}, camera_robot_bounds=[])
    if not base.IsValid():
        return result
    for prim in Usd.PrimRange(base):
        name = str(prim.GetPath())
        if prim == base or 'link' in prim.GetName() or 'hand' in prim.GetName():
            if prim.IsA(UsdGeom.Xformable):
                # USD matrices use row vectors; store a conventional column transform.
                result['transforms'][name] = np.asarray(cache.GetLocalToWorldTransform(prim)).T.tolist()
    extent = bounds.ComputeWorldBound(base).ComputeAlignedRange()
    low, high = np.asarray(extent.GetMin()), np.asarray(extent.GetMax())
    result.update(world_aabb_min=low.tolist(), world_aabb_max=high.tolist(),
        descendant_meshes=sum(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(base)),
        bounds_scope='Conservative resolved robot world AABB; inside does not prove a mesh intersection.')
    for eye in layout.eyes:
        point = np.asarray(eye)
        delta = np.maximum(np.maximum(low-point, point-high), 0.)
        result['camera_robot_bounds'].append(dict(eye=point.tolist(),
            outside_robot_aabb=bool(np.any(delta > 0.)), distance_to_aabb_m=float(np.linalg.norm(delta))))
    return result


def native_settling(world):
    """Read cached USD rigid-link transforms without constructing a backend."""
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    base = stage.GetPrimAtPath('/Franka')
    if not base.IsValid():
        raise RuntimeError('Cannot measure initial physics settling without resolved /Franka')
    tracked = [prim for prim in Usd.PrimRange(base) if prim.IsA(UsdGeom.Xformable) and
               (prim == base or 'link' in prim.GetName() or 'hand' in prim.GetName())]
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                  UsdGeom.Tokens.proxy])
    def transforms():
        cache.Clear()
        return {str(prim.GetPath()): np.asarray(cache.GetLocalToWorldTransform(prim)).T.tolist() for prim in tracked}
    def bounds():
        box.Clear()
        extent = box.ComputeWorldBound(base).ComputeAlignedRange()
        return [np.asarray(extent.GetMin()).tolist(), np.asarray(extent.GetMax()).tolist()]
    return settle_physics(lambda: world.step(render=True), transforms, bounds)


def native_timing(world, rig, scene, classes, sample_count, record):
    """Serial sources alone, fresh render+source loops; scoring is outside."""
    values = []
    ordered = list(classes)
    if record.get('order_index', 0) % 2:
        ordered.reverse()
    for method_order, (method, cls) in enumerate(ordered):
        source = cls(BorrowedRig(rig), scene)
        try:
            for index in range(3+sample_count):
                started = time.perf_counter()
                for _ in range(8):
                    world.step(render=True)
                rendered = time.perf_counter()
                batch = source.observe()
                finished = time.perf_counter()
                if index >= 3:
                    values.append(dict(trial=record['key'], layout=record['layout']['name'], method=method,
                        sample=index-3, render_ms=(rendered-started)*1000,
                        method_order=method_order, trial_order=record.get('order_index'),
                        render_plus_source_ms=(finished-started)*1000,
                        fresh=batch.timestamp is not None, acquisition_timestamp=batch.timestamp,
                        **plain(source.diagnostics)))
        finally:
            source.close()
    return values


def method_specs(args):
    names = args.methods.split(',')
    unknown = set(names)-set(METHODS)
    if unknown:
        raise ValueError(f'Unknown methods: {sorted(unknown)}')
    baseline = None
    if 'baseline_fused' in names:
        if not args.baseline_pythonpath:
            raise ValueError('baseline_fused requires --baseline-pythonpath')
        sys.path.insert(0, str(Path(args.baseline_pythonpath).resolve()))
        baseline = importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider
    specs = {'current_fused': (MultiViewProvider, 'fused'), 'baseline_fused': (baseline, 'fused'),
             'current_single_a': (MultiViewProvider, 'single_a'), 'current_single_b': (MultiViewProvider, 'single_b'),
             'v1_a': (None, 'v1_a'), 'v1_b': (None, 'v1_b')}
    return {name: specs[name] for name in names}


def annotate_visibility(rows, audit):
    measured = {entry.get('object', entry.get('name')): entry for entry in audit.get('objects', [])}
    for row in rows:
        item = measured.get(row['object'], {})
        row['visibility'] = item
        row['sufficiently_visible'] = item.get('sufficiently_visible')
        counts = [camera['raw_points'] for camera in item.get('cameras', [])]
        row['strict_true_one_view'] = bool(len(counts) == 2 and min(counts) == 0 and max(counts) >= 8)
        row['effective_one_view'] = bool(len(counts) == 2 and sum(value >= 8 for value in counts) == 1)


def completion_gate(*, errors, hashes_stable, audits, summaries, capture_only):
    if errors or not hashes_stable or not audits or any(not row['valid'] for row in audits):
        return False
    return capture_only or bool(summaries) and all(value['source_contracts_passed'] for value in summaries.values())


def run(args, app=None):
    from tests.support.multiview_final_research import (
        audit_final_scene, camera_layouts, final_scenes, scene_frame_content, scene_frame_count,
    )
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = provenance(args)
    write_json(output/'provenance_start.json', start)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    capture_root = Path(args.replay).resolve() if args.replay else output
    if args.replay:
        manifest = json.loads((capture_root/'capture_manifest.json').read_text())
        records = manifest['scenes']
        selected = [(scene_from_record(record), record, None) for record in records]
    else:
        layouts = list(camera_layouts())
        if args.layout:
            layouts = [layout for layout in layouts if layout.name in args.layout]
        if not layouts:
            raise ValueError('No requested camera layouts exist')
        # Critically before final_scenes('holdout') can be called.
        freeze = verify_freeze(args.freeze_manifest, layouts) if args.split == 'holdout' else None
        scenes = list(final_scenes(args.split, holdout_authorized=freeze is not None))
        if args.scenario:
            scenes = [scene for scene in scenes if scene.name in args.scenario or scene.key in args.scenario]
        if args.limit:
            scenes = scenes[:args.limit]
        selected = [(scene, scene_record(scene, layout), layout) for scene, layout in balanced_pairs(scenes, layouts)]
        manifest = dict(kind='final_perception_native_camera_study', split=args.split,
            layouts=[layout_payload(layout) for layout in layouts], layout_sha256=layout_hash(layouts),
            source_freeze=freeze, resolution=[640, 480], render_steps_between_captures=8,
            order='Scene-major cyclic Latin layout order; matched physical scenes across all layouts.',
            comparison='Identical frames within each layout; matched physical scenes across layouts.', scenes=[])
    if args.replay:
        if args.layout:
            selected = [entry for entry in selected if entry[1]['layout']['name'] in args.layout]
        if args.scenario:
            selected = [entry for entry in selected if entry[0].name in args.scenario or entry[0].key in args.scenario]
        if args.limit:
            keys = list(dict.fromkeys(entry[0].key for entry in selected))[:args.limit]
            selected = [entry for entry in selected if entry[0].key in keys]
        manifest = {**manifest, 'scenes': [entry[1] for entry in selected], 'capture_origin': str(capture_root)}
    if not selected:
        raise ValueError('An empty evaluation cannot establish a perception checkpoint')
    specs = {} if args.capture_only else method_specs(args)
    rows, samples, detections, observations, contracts, audits, errors, timings = [], [], [], [], [], [], [], []
    world, rig = None, None
    try:
        if not args.replay:
            from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
            from tests.support.multiview_native import CAMERAS, open_experiment
            world = open_experiment(app)
            rig = create_cameras(CAMERAS, resolution=(640, 480))
            manifest['physics_settling'] = native_settling(world)
            write_json(output/'physics_settling.json', manifest['physics_settling'])
            write_json(output/'capture_manifest.json', manifest)
            if not manifest['physics_settling']['stable']:
                raise RuntimeError('Initial physics did not settle within the declared600-step bound')
        for order_index, (scene, record, layout) in enumerate(selected):
            sessions = {name: PerceptionSession(cls, cfg, method=name, mode=mode) for name, (cls, mode) in specs.items()}
            record['order_index'] = order_index
            try:
                if not args.replay:
                    apply_layout(rig, layout)
                    for _ in range(16):
                        world.step(render=True)
                    record['resolved_robot'] = native_robot_diagnostics(layout)
                count = len(record['frames']) if args.replay else scene_frame_count(scene)
                for index in range(count):
                    if args.replay:
                        snapshot = record['frames'][index]
                        frames = load_frames(capture_root, snapshot['files'])
                        objects = tuple(object_from_payload(obj) for obj in snapshot['objects'])
                        phase = snapshot.get('phase', 'static')
                    else:
                        objects, phase = scene_frame_content(scene, index)
                        populate_scene(objects)
                        for _ in range(8):
                            world.step(render=True)
                        frames = rig.capture()
                        if frames is None:
                            raise RuntimeError('Native camera pair did not advance')
                        if len(frames) != 2 or any(frame.depth.shape != (480, 640) for frame in frames):
                            raise RuntimeError('Expected two calibrated 640x480 frames')
                        snapshot = dict(index=index, objects=[object_payload(obj) for obj in objects], phase=phase,
                            files=save_frames(output, record['key'], index, frames))
                        if index >= 2:
                            snapshot.update(capture_workspace(output, record['key'], index, frames, objects, cfg.scene))
                        record['frames'].append(snapshot)
                    context = dict(trial=record['key'], scene_key=scene.key, layout=record['layout']['name'],
                        split=scene.split, seed=scene.seed, phase=phase, temporal=scene.temporal, condition=scene.name)
                    audit = audit_final_scene(scene, frames, index=index, config=cfg.scene)
                    audit.update(**context, frame_index=index, scored=index >= 2)
                    audits.append(audit)
                    for name, session in sessions.items():
                        before = len(session.rows)
                        accepted, clouds = [], []
                        try:
                            accepted, clouds = session.consume(frames, objects, index=index, **context)
                        except Exception as error:
                            message = f'{type(error).__name__}: {error}'
                            errors.append(dict(method=name, frame_index=index, error=message, **context))
                            session.retain_failure(objects, index=index, error=message, **context)
                        if index >= 2 and args.export_clouds:
                            for observation, cloud in zip(accepted, clouds):
                                if cloud is not None:
                                    path = output/'processed'/name/record['key']/f'frame_{index:02d}_{observation.object_id}.ply'
                                    save_ply(path, cloud)
                        annotate_visibility(session.rows[before:], audit)
                    if not args.replay:
                        write_json(output/'capture_manifest.json', {**manifest, 'scenes': manifest['scenes']+[record]})
                if not args.replay:
                    manifest['scenes'].append(record)
                if not args.replay and args.native_timing_samples:
                    # Static content only. All diagnostic/truth work is finished.
                    if not scene.temporal:
                        native_classes = [('current_fused', MultiViewProvider)]
                        if args.time_baseline:
                            native_classes.insert(0, ('baseline_fused', method_specs(argparse.Namespace(**{
                                **vars(args), 'methods': 'baseline_fused'}))['baseline_fused'][0]))
                        timings.extend(native_timing(world, rig, cfg.scene, native_classes,
                                                     args.native_timing_samples, record))
                        write_json(output/'native_timings.json', timings)
                for session in sessions.values():
                    rows.extend(session.rows)
                    samples.extend(session.samples)
                    detections.extend(session.detections)
                    observations.extend(session.observations)
                    contracts.extend(session.contracts)
                for filename, value in [('capture_manifest', manifest), ('rows', rows), ('samples', samples),
                    ('detections', detections), ('observations', observations), ('contracts', contracts),
                    ('fixture_audit', audits), ('runtime_errors', errors)]:
                    write_json(output/f'{filename}.json', value)
                print('captured' if args.capture_only else 'evaluated', record['key'], flush=True)
            finally:
                for session in sessions.values():
                    session.close()
        summaries = {}
        for layout_name in dict.fromkeys(record['layout']['name'] for _, record, _ in selected):
            summaries[layout_name] = {}
            for method in specs:
                def subset(values):
                    return [row for row in values if row['layout'] == layout_name and row['method'] == method]
                summaries[layout_name][method] = summarize(subset(rows), subset(samples), subset(detections), subset(contracts))
                summaries[layout_name][method]['latest_static_scene_metrics'] = latest_scene_summary(
                    subset(rows), subset(samples), subset(detections), subset(contracts))
                summaries[layout_name][method]['temporal_recovery'] = temporal_recovery(subset(rows), subset(samples))
        end = provenance(args)
        stable = start['sha256'] == end['sha256']
        flat = {f'{layout}/{method}': value for layout, methods in summaries.items() for method, value in methods.items()}
        completed = completion_gate(errors=errors, hashes_stable=stable, audits=audits,
                                    summaries=flat, capture_only=args.capture_only)
        result = dict(status='PASS' if completed else 'FAIL', error=None, metrics=dict(
            kind='final_perception_capture_only' if args.capture_only else 'final_perception_matched_replay',
            summary=summaries, physical_scenes=len({scene.key for scene, _, _ in selected}),
            layout_scene_trials=len(selected), native_samples=sum(len(record['frames']) for _, record, _ in selected),
            runtime_errors=len(errors), invalid_audit_frames=sum(not value['valid'] for value in audits),
            source_and_harness_hashes_stable=stable, no_robot_backend_or_planning=True,
            input_images_identical_within_layout=True, matched_physical_scenes_across_layouts=True,
            engineering_targets_are_aspirational=True,
            completion_scope='All requested trials retained, fixture audits valid, source freshness/confirmation contracts pass; '
                'classification/geometry failures remain data and do not disappear behind PASS.',
            native_timing={method: dict(samples=len(values),
                render_plus_source_ms=distribution([value['render_plus_source_ms'] for value in values]),
                effective_hz=1000/np.mean([value['render_plus_source_ms'] for value in values]),
                stage_ms={stage: distribution([value.get(stage) for value in values]) for stage in STAGES},
                all_samples_fresh=all(value['fresh'] for value in values))
                for method in dict.fromkeys(value['method'] for value in timings)
                for values in [[value for value in timings if value['method'] == method]]}))
        # Replay inherits the capture's measured native timings; it does not
        # infer render rate from current CPU replay or old acquisition gaps.
        native_path = capture_root/'native_timings.json'
        native_samples = json.loads(native_path.read_text()) if args.replay and native_path.exists() else timings
        requested_keys = {record['key'] for _, record, _ in selected}
        native_samples = [value for value in native_samples if value['trial'] in requested_keys]
        result['metrics']['native_timing_by_layout'] = {
            name: {method: dict(samples=len(values),
                effective_hz=1000/np.mean([value['render_plus_source_ms'] for value in values]),
                render_plus_source_ms=distribution([value['render_plus_source_ms'] for value in values]),
                render_ms=distribution([value['render_ms'] for value in values]),
                stages_ms={stage: distribution([value.get(stage) for value in values]) for stage in STAGES},
                all_samples_fresh=all(value['fresh'] for value in values),
                origin=str(native_path), scope='Actual native source-only measurements saved during capture.')
                for method in dict.fromkeys(value['method'] for value in native_samples if value['layout'] == name)
                for values in [[value for value in native_samples if value['layout'] == name and value['method'] == method]]}
            for name in dict.fromkeys(value['layout'] for value in native_samples)}
        recovery = [event for methods in summaries.values() for value in methods.values() for event in value['temporal_recovery']]
        write_json(output/'temporal_recovery.json', recovery)
        for filename, values in [('rows', rows), ('samples', samples), ('detections', detections),
                                 ('observations', observations), ('contracts', contracts)]:
            write_csv(output/f'{filename}.csv', plain(values))
        write_json(output/'provenance_end.json', end)
        write_json(args.result, result)
        return 0 if completed else 1
    finally:
        if rig is not None:
            rig.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--result', required=True)
    result.add_argument('--artifacts-dir')
    result.add_argument('--capture-only', action='store_true')
    result.add_argument('--replay', type=Path)
    result.add_argument('--split', choices=('coarse', 'shortlist', 'holdout'), default='coarse')
    result.add_argument('--layout', action='append')
    result.add_argument('--scenario', action='append')
    result.add_argument('--limit', type=int, help='Maximum physical scenes, all requested layouts retained')
    result.add_argument('--methods', default='current_fused', help=','.join(METHODS))
    result.add_argument('--baseline-pythonpath', type=Path)
    result.add_argument('--baseline-package', default='baseline_multiview')
    result.add_argument('--freeze-manifest', type=Path)
    result.add_argument('--native-timing-samples', type=int, default=0)
    result.add_argument('--time-baseline', action='store_true')
    result.add_argument('--export-clouds', action='store_true')
    return result


def cli():
    command = parser()
    args = command.parse_args()
    args.artifacts_dir = args.artifacts_dir or str(Path(args.result).parent)
    if args.capture_only and args.replay:
        command.error('--capture-only and --replay are mutually exclusive')
    if args.limit is not None and args.limit < 1 or args.native_timing_samples < 0:
        command.error('--limit must be positive and --native-timing-samples nonnegative')
    if args.replay and args.native_timing_samples:
        command.error('Native timing cannot be inferred from saved-frame replay')
    if args.time_baseline and not args.baseline_pythonpath:
        command.error('--time-baseline requires --baseline-pythonpath')
    app, code = None, 1
    try:
        if not args.replay:
            if args.split == 'holdout':
                from tests.support.multiview_final_research import camera_layouts
                layouts = [layout for layout in camera_layouts() if not args.layout or layout.name in args.layout]
                verify_freeze(args.freeze_manifest, layouts)
            if np.__version__ != '1.26.4':
                raise RuntimeError(f'Expected NumPy1.26.4 before SimulationApp, got {np.__version__}')
            from isaacsim import SimulationApp
            app = SimulationApp({'headless': False})
        code = run(args, app)
    except Exception as error:
        write_json(args.result, dict(status='FAIL', error=f'{type(error).__name__}: {error}', metrics=dict(runtime_errors=1)))
    finally:
        write_json(Path(args.artifacts_dir)/'provenance_exit.json', provenance(args))
        if app is not None:
            app.close()
    raise SystemExit(code)


if __name__ == '__main__':
    cli()
