"""Native current-depth observability and read-only grasp evaluation.

Capture once for baseline/current on identical frames. Saved-frame replay needs
no simulator: original tokens/timestamps survive and an explicit research clock
follows the native capture timeline. No robot, motion backend or Pick is created.
Strict acceptance applies to the current source; baseline failures remain data.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.clouds import save_ply
from scripts.benchmark_multiview_grasp_readiness import plain_values, write_csv, write_json
from scripts.benchmark_multiview_robustness import (
    capture_workspace, load_frames, populate_scene, save_frames, source_provenance,
)
from tests.support.multiview_iteration2 import object_from_payload, object_payload, perturb_frames
from tests.support.multiview_observability_evaluation import EvaluationSession, method_summary


def package_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))}


def verify_freeze(path):
    if path is None:
        raise ValueError('Holdout capture requires --freeze-manifest before generating any holdout scenes')
    frozen = json.loads(Path(path).read_text())['sha256']
    current = package_hashes()
    differences = [name for name, value in current.items() if frozen.get(name) != value]
    if differences:
        raise ValueError(f'Frozen source differs or omits package files: {differences}')
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                verified_package_sha256=current)


def provenance(args):
    result = source_provenance(args)
    for relative in ('scripts/benchmark_multiview_observability.py',
                     'tests/support/multiview_observability_research.py',
                     'tests/support/multiview_observability_evaluation.py',
                     'tests/support/multiview_readiness_research.py'):
        path = ROOT/relative
        if path.exists():
            result['sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result.update(clock='Original native monotonic acquisition times; replay clock follows saved timeline.'
        if args.replay else 'Actual native monotonic acquisition clock.',
        no_robot_created=True, no_backend_commands=True)
    return result


def scene_payload(scene):
    return dict(key=scene.key, name=scene.name, split=scene.split, seed=scene.seed,
        objects=[object_payload(obj) for obj in scene.objects], clutter=[object_payload(obj) for obj in scene.clutter],
        occluded_clutter=[object_payload(obj) for obj in scene.occluded_clutter],
        eyes=scene.eyes, targets=scene.targets, requirements=scene.requirements,
        temporal=scene.temporal, note=scene.note, frames=[])


def scene_from_record(record):
    # No holdout generation when replaying: reconstruct the captured fixture.
    from tests.support.multiview_observability_research import ObservabilityScene
    fields = {key: record[key] for key in ('name', 'split', 'seed', 'eyes', 'targets', 'requirements', 'temporal', 'note')}
    return ObservabilityScene(objects=tuple(object_from_payload(obj) for obj in record['objects']),
        clutter=tuple(object_from_payload(obj) for obj in record.get('clutter', [])),
        occluded_clutter=tuple(object_from_payload(obj) for obj in record.get('occluded_clutter', [])), **fields)


def classes(args):
    result = {'improved': MultiViewProvider}
    if args.baseline_pythonpath:
        sys.path.insert(0, str(Path(args.baseline_pythonpath).resolve()))
        result = {'baseline': importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider, **result}
    if args.baseline_only:
        return {'baseline': result['baseline']}
    return result


def apply_variant(frames, variant, seed):
    # Only measurements are changed; native timestamps and identities persist.
    return perturb_frames(frames, 'sparse' if variant == 'dropout' else variant, seed=seed)


def result_gate(summary, audits, errors, *, strict, baseline_only=False, hashes_stable=True, capture_only=False):
    """Invalid fixtures never disappear into a passing zero-unsafe denominator."""
    if errors or any(not audit['valid'] for audit in audits) or (not capture_only and not hashes_stable):
        return False
    if capture_only:
        return True
    selected = summary.get('baseline' if baseline_only else 'improved')
    return bool(selected and selected['contracts_passed'] and (not strict or
        (selected['safe_valid_geometry'] and selected['full_scene_safety_passed'])))


def run(args, app=None):
    from tests.support.multiview_observability_research import (
        audit_observability_scene, observability_scenes, scene_frame_content, scene_frame_count,
    )
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = provenance(args)
    write_json(output/'provenance_start.json', start)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    capture_root = Path(args.replay).resolve() if args.replay else output
    freeze = verify_freeze(args.freeze_manifest) if not args.replay and args.split == 'holdout' else None
    if args.replay:
        manifest = json.loads((capture_root/'capture_manifest.json').read_text())
        records = manifest['scenes']
        selected = [(scene_from_record(record), record) for record in records]
    else:
        # Holdout function is never called before the explicit hash guard above.
        selected = [(scene, scene_payload(scene)) for scene in observability_scenes(args.split)]
        manifest = dict(kind='native_observability_current_depth_capture', split=args.split,
                        source_freeze=freeze, scenes=[])
    if args.scenario:
        selected = [(scene, record) for scene, record in selected
                    if scene.name in args.scenario or scene.key in args.scenario]
        if not selected:
            raise ValueError('No requested scenarios exist in this capture/split')
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No capture scenes are available; an empty benchmark cannot establish readiness')
    if args.replay:
        manifest = {**manifest, 'scenes': [record for _, record in selected],
                    'capture_origin': str(capture_root)}
    world, rig = None, None
    rows, samples, detections, contracts, audits, errors = [], [], [], [], [], []
    providers = {} if args.capture_only else classes(args)
    try:
        if not args.replay:
            from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
            from tests.support.multiview_native import CAMERAS, open_experiment, set_viewpoint
            world = open_experiment(app)
            rig = create_cameras(CAMERAS)
        for scene, record in selected:
            sessions = {method: EvaluationSession(cls, cfg.scene, method=method, scene_key=scene.key,
                        replay=bool(args.replay), temporal=scene.temporal) for method, cls in providers.items()}
            loaded = []
            if not args.replay:
                for camera, eye, target in zip(rig.cameras.values(), scene.eyes, scene.targets):
                    set_viewpoint(camera, eye, target)
                for _ in range(30):
                    world.step(render=True)
            try:
                count = len(record['frames']) if args.replay else scene_frame_count(scene)
                for index in range(count):
                    if args.replay:
                        snapshot = record['frames'][index]
                        raw_frames = load_frames(capture_root, snapshot['files'])
                        objects = tuple(object_from_payload(obj) for obj in snapshot['objects'])
                        clutter = tuple(object_from_payload(obj) for obj in snapshot.get('clutter', record.get('clutter', [])))
                        phase = snapshot.get('phase', 'static')
                    else:
                        objects, clutter, phase = scene_frame_content(scene, index)
                        populate_scene(objects+clutter)
                        for _ in range(8):
                            world.step(render=True)
                        raw_frames = rig.capture()
                        if raw_frames is None:
                            raise RuntimeError('No new synchronized native pair')
                        snapshot = dict(index=index, objects=[object_payload(obj) for obj in objects],
                            clutter=[object_payload(obj) for obj in clutter], phase=phase,
                            files=save_frames(output, scene.key, index, raw_frames))
                        snapshot.update(capture_workspace(output, scene.key, index, raw_frames, objects, cfg.scene))
                        record['frames'].append(snapshot)
                    loaded.append((snapshot, raw_frames))
                    frames = apply_variant(raw_frames, args.variant, scene.seed+index)
                    for method, session in sessions.items():
                        try:
                            session.consume(frames, objects, clutter, index=index, phase=phase)
                            if index >= 2 and session.source.batch is not None:
                                for observation in session.source.batch.observations:
                                    cloud = session.provider.get_point_cloud(observation.object_id)
                                    if cloud is not None:
                                        cloud_path = output/'processed'/method/scene.key/f'frame_{index:02d}_{observation.object_id}.ply'
                                        save_ply(cloud_path, cloud)
                                        for detection in session.detections:
                                            if detection['frame_index'] == index:
                                                for details in detection['clouds']:
                                                    if details['object_id'] == observation.object_id:
                                                        details['cloud'] = str(cloud_path.relative_to(output))
                            if scene.temporal and phase == 'incomplete' and (
                                index+1 == count or scene_frame_content(scene, index+1)[2] != 'incomplete'):
                                session.frozen()
                        except Exception as error:
                            errors.append(dict(scene=scene.key, method=method, frame_index=index,
                                               error=f'{type(error).__name__}: {error}'))
                            session.retain_failure(objects, clutter, index=index, phase=phase,
                                                   error=errors[-1]['error'])
                    # Preserve progress before any optional diagnostic can fail.
                    if not args.replay:
                        write_json(output/'capture_manifest.json', {**manifest, 'scenes': manifest['scenes']+[record]})
                try:
                    frame_audits = [audit_observability_scene(scene, frames,
                        frame_index=snapshot['index'], phase=snapshot['phase'], config=cfg.scene)
                        for snapshot, frames in loaded if snapshot['index'] >= 2]
                    audit = dict(valid=bool(frame_audits) and all(value['valid'] for value in frame_audits),
                        errors=[dict(frame_index=entry['frame_index'], error=error)
                                for value in frame_audits for entry in value['measurements'] for error in entry['errors']],
                        measurements=[entry for value in frame_audits for entry in value['measurements']])
                except Exception as error:
                    audit = dict(valid=False, errors=[f'Fixture audit {type(error).__name__}: {error}'], measurements=[])
                audit.update(scene=scene.key)
                audits.append(audit)
                measurements = {entry['frame_index']: entry for entry in audit.get('measurements', [])}
                for session in sessions.values():
                    try:
                        contracts.extend(session.finish())
                    except Exception as error:
                        errors.append(dict(scene=scene.key, method=session.method,
                                           error=f'{type(error).__name__}: {error}'))
                    for value in session.rows+session.detections:
                        measurement = measurements.get(value['frame_index'], {})
                        value.update(valid_fixture=bool(measurement.get('valid_fixture', audit['valid'])),
                            fixture_errors=measurement.get('errors', audit.get('errors', [])),
                            split=scene.split, seed=scene.seed, variant=args.variant)
                    if (scene.requirements or {}).get('case') == 'mandatory_legacy_90mm':
                        contracts.append(dict(scene=scene.key, method=session.method,
                            kind='mandatory_90mm_both_camera_hidden_box_rejected',
                            passed=bool(session.rows and all(row['valid_fixture'] and row['detected'] and
                                row['classification_correct'] and row['graspable'] is False and
                                not row['estimated_feasible'] for row in session.rows))))
                    rows.extend(session.rows)
                    samples.extend(session.samples)
                    detections.extend(session.detections)
                if not args.replay:
                    manifest['scenes'].append(record)
                for filename, values in [('capture_manifest', manifest), ('rows', rows), ('samples', samples),
                    ('detections', detections), ('contracts', contracts), ('fixture_audit', audits), ('runtime_errors', errors)]:
                    write_json(output/f'{filename}.json', values)
                print('observability captured' if args.capture_only else 'observability evaluated', scene.key, flush=True)
            finally:
                for session in sessions.values():
                    session.updater.stop()
        summary = {method: method_summary([row for row in rows if row['method'] == method],
            [row for row in samples if row['method'] == method], [row for row in contracts if row['method'] == method],
            [row for row in detections if row['method'] == method]) for method in providers}
        invalid = [audit for audit in audits if not audit['valid']]
        end = provenance(args)
        stable = start['sha256'] == end['sha256']
        passed = result_gate(summary, audits, errors, strict=args.strict, baseline_only=args.baseline_only,
                             hashes_stable=stable, capture_only=args.capture_only)
        result = dict(status='PASS' if passed else 'FAIL', metrics=dict(
            kind='observability_capture_only' if args.capture_only else 'full_path_readonly_observability',
            summary=summary, scenes=len(selected), invalid_scenes=len(invalid), runtime_errors=len(errors),
            strict_current_zero_unsafe=args.strict, source_hashes_stable=stable, no_robot_or_backend_commands=True,
            source_executed=not args.capture_only,
            geometry_gate_scope='Target geometry feasibility and actual proposed jaw width, target-palm and support clearance. '
                'Physical cover/clutter obstacle verdicts are separately retained in full_scene fields.',
            full_scene_safety_passed=None if args.capture_only else summary.get('improved', summary.get('baseline', {})).get('full_scene_safety_passed'),
            overall_readiness_evidence_passed=bool(not args.capture_only and passed and
                summary.get('improved', summary.get('baseline', {})).get('full_scene_safety_passed')),
            input_frames_identical_between_methods=True, variant=args.variant,
            clock_scope='Saved original acquisition timeline; logical replay freshness, not live native rate.'
                if args.replay else 'Actual native capture/update clock.',
            no_pick_place_or_gripper_execution=True), error=None)
        for filename, values in [('rows', rows), ('samples', samples), ('detections', detections), ('contracts', contracts)]:
            write_csv(output/f'{filename}.csv', values)
        write_json(args.result, result)
        write_json(output/'provenance_end.json', end)
        return 0 if passed else 1
    finally:
        if rig is not None:
            rig.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--result', required=True)
    result.add_argument('--artifacts-dir')
    result.add_argument('--capture-only', action='store_true')
    result.add_argument('--replay', type=Path)
    result.add_argument('--split', choices=('selection', 'holdout'), default='selection')
    result.add_argument('--scenario', action='append')
    result.add_argument('--limit', type=int)
    result.add_argument('--variant', choices=('clean', 'depth_noise', 'calibration', 'dropout', 'sparse'), default='clean')
    result.add_argument('--baseline-pythonpath')
    result.add_argument('--baseline-package', default='baseline_multiview')
    result.add_argument('--baseline-only', action='store_true')
    result.add_argument('--freeze-manifest', type=Path)
    result.add_argument('--strict', action='store_true')
    return result


def cli(strict=False):
    command = parser()
    args = command.parse_args()
    args.strict = args.strict or strict
    args.artifacts_dir = args.artifacts_dir or str(Path(args.result).parent/'observability')
    if args.limit is not None and args.limit < 1:
        command.error('--limit must be positive')
    if args.baseline_only and not args.baseline_pythonpath:
        command.error('--baseline-only requires --baseline-pythonpath')
    if args.capture_only and args.replay:
        command.error('--capture-only and --replay are mutually exclusive')
    app, code = None, 1
    try:
        if not args.replay:
            if args.split == 'holdout':
                verify_freeze(args.freeze_manifest)
            if np.__version__ != '1.26.4':
                raise RuntimeError(f'Expected NumPy1.26.4 before SimulationApp, got {np.__version__}')
            from isaacsim import SimulationApp
            app = SimulationApp({'headless': False})
        code = run(args, app)
    except Exception as error:
        write_json(args.result, dict(status='FAIL', metrics=dict(runtime_errors=1),
                                    error=f'{type(error).__name__}: {error}'))
    finally:
        write_json(Path(args.artifacts_dir)/'provenance_exit.json', provenance(args))
        if app is not None:
            app.close()
    raise SystemExit(code)


if __name__ == '__main__':
    cli()
