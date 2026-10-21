"""Read-only native perception-to-planner evaluation. No Pick is submitted.

Only the existing synchronous Pick precondition query is called. Planner and IK
queries never execute trajectories or gripper commands. Ground truth scores the
unchanged planner after automatic perception, never segmentation or fitting.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from scripts.benchmark_multiview_robustness import (
    ROOT, capture_workspace, populate_scene, save_frames, source_provenance,
    write_csv as _write_csv, write_json as _write_json,
)
from tests.support.multiview_iteration2 import SnapshotRig, distribution, object_payload
from tests.support.multiview_readiness_research import (
    EYES, QueryOnlyBackend, evaluate_grasp, grasp_width_scenes, invalid_geometry_queries,
    make_planner, pick_start_guard,
)
from tests.support.results import fail_result, write_result as _write_result
from tests.support.multiview_research import metric_row


def plain_values(value):
    """Keep numerical evidence intact while converting NumPy scalar containers."""
    if isinstance(value, np.ndarray):
        return plain_values(value.tolist())
    if isinstance(value, np.generic):
        return plain_values(value.item())
    if isinstance(value, dict):
        return {key: plain_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_values(item) for item in value]
    return value


def write_json(path, value):
    return _write_json(path, plain_values(value))


def write_csv(path, rows):
    return _write_csv(path, plain_values(rows))


def write_result(path, **values):
    return _write_result(path, **plain_values(values))


def match_objects(objects, visible):
    assigned, matches = set(), {}
    for distance, oi, di in sorted((float(np.linalg.norm(obj.center-obs.pose.position)), oi, di)
                                  for oi, obj in enumerate(objects) for di, obs in enumerate(visible)):
        if distance < .04 and oi not in matches and di not in assigned:
            matches[oi] = di
            assigned.add(di)
    return {objects[oi].name: visible[di] for oi, di in matches.items()}, len(visible)-len(assigned)


def source_classes(args):
    classes = {'improved': MultiViewProvider}
    if args.baseline_pythonpath:
        sys.path.insert(0, str(Path(args.baseline_pythonpath).resolve()))
        baseline = importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider
        classes = {'baseline': baseline, **classes}
    return {'baseline': classes['baseline']} if args.baseline_only else classes


def frame_step(world, rig):
    for _ in range(8):
        world.step(render=True)
    frames = rig.capture()
    if frames is None:
        raise RuntimeError('Native cameras did not supply a new synchronized pair')
    return frames


def contract_queries(world, rig, source_rig, updater, truth, backend):
    """Native remeasurement plus isolated read-only fault/health queries."""
    model = updater.world_model
    matched, _ = match_objects((truth,), model.visible_objects())
    if truth.name not in matched:
        return dict(passed=False, error='Ordinary object missing; contract precondition unavailable')
    obj = matched[truth.name]
    key = obj.object_id
    planner = make_planner(model, backend)
    result = dict(initial_id=key, initially_allowed=pick_start_guard(model, key, planner, backend)['allowed'])
    stamp = updater.health.last_frame_at
    updater.update()
    result['unchanged_frame_does_not_refresh'] = updater.health.last_frame_at == stamp
    model.require(key).graspable = False
    updater.update()
    result['prior_rejection_retained_without_new_measurement'] = not model.require(key).graspable
    source_rig.push(frame_step(world, rig))
    updater.update()
    result['new_known_measurement_restores_graspable'] = bool(model.require(key).class_name is not None and
        model.require(key).graspable and planner.plan(model.require(key)).ok)
    result['restoring_measurement_is_new'] = updater.health.last_frame_at > stamp
    result['invalid_geometry_rejected'] = invalid_geometry_queries(obj, model, backend)
    now = time.monotonic()
    with patch('time.monotonic', return_value=now+1.2):
        updater.cleanup()
        result['stale_pick_guard'] = pick_start_guard(model, key, planner, backend)
    # Restore freshness from a new real capture, not a retimestamped old frame.
    source_rig.push(frame_step(world, rig))
    updater.update()
    result['fresh_measurement_restored'] = updater.health.status == 'fresh' and model.require(key).visible
    held = []
    for uncertain in (False, True):
        model.set_held(key, uncertain=uncertain)
        held.append(dict(uncertain=uncertain, **pick_start_guard(model, key, planner, backend)))
    result['held_pick_guards'] = held
    populate_scene(())
    source_rig.push(frame_step(world, rig))
    updater.update()
    updater.cleanup(now=time.monotonic()+updater.stale_object_ttl+1)
    result['uncertain_id_survives_missing_and_ttl'] = model.exists(key) and model.held_object_id == key
    old_pose = model.require(key).pose.position.copy()
    moved = replace(truth, center=truth.center+np.array([.055, .075, 0.]))
    model.expect_object_at(key, moved.center, ttl=10., reason='readiness_placement_hint')
    result['hint_does_not_fabricate_pose'] = bool(np.array_equal(model.require(key).pose.position, old_pose))
    model.set_held(None)
    populate_scene((moved,))
    source_rig.push(frame_step(world, rig))
    updater.update()
    result['hint_reacquired_same_id'] = model.exists(key) and model.require(key).visible
    result['hint_consumed'] = not model.association_hints()
    result['measured_reacquired_center_error_mm'] = float(np.linalg.norm(model.require(key).pose.position-moved.center)*1000)
    result['failed_source_guard'] = None
    updater.health.error = 'Read-only injected source-health failure'
    result['failed_source_guard'] = pick_start_guard(model, key, planner, backend)
    updater.health.error = None
    result['passed'] = bool(result['initially_allowed'] and result['unchanged_frame_does_not_refresh'] and
        result['prior_rejection_retained_without_new_measurement'] and
        result['new_known_measurement_restores_graspable'] and result['restoring_measurement_is_new'] and
        not result['stale_pick_guard']['allowed'] and result['fresh_measurement_restored'] and
        not any(item['allowed'] for item in held) and result['uncertain_id_survives_missing_and_ttl'] and
        result['hint_does_not_fabricate_pose'] and result['hint_reacquired_same_id'] and result['hint_consumed'] and
        not result['failed_source_guard']['allowed'] and
        all(not value['feasible'] for value in result['invalid_geometry_rejected'].values()) and
        result['measured_reacquired_center_error_mm'] < 10)
    return result


def main(args, app):
    from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
    from tests.support.multiview_native import CAMERAS, open_experiment, set_viewpoint
    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(args)
    for filename in (Path(__file__).resolve(), ROOT/'tests/support/multiview_readiness_research.py'):
        provenance['sha256'][str(filename.relative_to(ROOT))] = hashlib.sha256(filename.read_bytes()).hexdigest()
    write_json(output/'provenance_start.json', provenance)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    world = open_experiment(app)
    native_backend = None
    if args.ik:
        from robot_skill_stack.integrations.isaac.manipulation.factory import create_backend, create_robot
        robot = create_robot(world, cfg.robot)
        world.reset()
        world.play()
        for _ in range(30):
            world.step(render=True)
        native_backend, _ = create_backend(world, robot, cfg.robot)
    backend = QueryOnlyBackend(native_backend)
    rig = create_cameras(CAMERAS)
    rows, samples, contracts, errors, manifest, oracle = [], [], [], [], [], []
    try:
        for camera, eye in zip(rig.cameras.values(), EYES[args.split]):
            set_viewpoint(camera, eye)
        for _ in range(60):
            world.step(render=True)
        scenes = grasp_width_scenes(args.split)
        if args.limit:
            scenes = scenes[:args.limit]
        classes = source_classes(args)
        for scene in scenes:
            populate_scene(scene.objects+scene.clutter)
            source_rigs = {method: SnapshotRig() for method in classes}
            sources = {method: cls(source_rigs[method], cfg.scene) for method, cls in classes.items()}
            updaters = {method: WorldModelUpdater(WorldModel(), source, stale_object_ttl=2.5)
                        for method, source in sources.items()}
            record = dict(key=scene.key, name=scene.name, split=scene.split, seed=scene.seed,
                          gap_mm=scene.gap_mm, clutter=[], frames=[], note=scene.note)
            prior_ids = {method: {} for method in classes}
            prior_classes = {method: {} for method in classes}
            try:
                for index in range(4):
                    frames = frame_step(world, rig)
                    snapshot = dict(index=index, files=save_frames(output, scene.key, index, frames),
                                    objects=[object_payload(obj) for obj in scene.objects])
                    if index >= 2:
                        snapshot.update(capture_workspace(output, scene.key, index, frames, scene.objects, cfg.scene))
                    record['frames'].append(snapshot)
                    for method, updater in updaters.items():
                        source_rigs[method].push(frames)
                        updater.update()
                        model = updater.world_model
                        visible = model.visible_objects()
                        matches, false_positives = match_objects(scene.objects, visible)
                        ids = {name: obj.object_id for name, obj in matches.items()}
                        labels = {name: obj.class_name for name, obj in matches.items()}
                        switches = sum(name in prior_ids[method] and prior_ids[method][name] != key
                                       for name, key in ids.items())
                        class_switches = sum(name in prior_classes[method] and prior_classes[method][name] != label
                                             for name, label in labels.items())
                        sample = dict(scene=scene.key, method=method, frame_index=index,
                            health=updater.health.status, source_error=updater.health.error,
                            observed=len(visible), identities=ids, classifications=labels,
                            classification_switches=class_switches,
                            identity_switches=switches, false_positives=false_positives,
                            early_confirmations=len(visible) if index < 2 else 0,
                            diagnostics=sources[method].diagnostics)
                        samples.append(sample)
                        prior_ids[method].update(ids)
                        prior_classes[method].update(labels)
                        if index != 3:
                            continue
                        for truth in scene.objects:
                            row = evaluate_grasp(truth, matches.get(truth.name), model,
                                                 scene.objects+scene.clutter, backend=backend)
                            row.update(scene=scene.key, split=scene.split, method=method, seed=scene.seed,
                                       truth_geometry=object_payload(truth))
                            observed = matches.get(truth.name)
                            row['classification_correct'] = observed is not None and observed.class_name == (
                                None if truth.shape == 'ellipsoid' else truth.shape)
                            row['dimension_error_mm'] = None
                            row['axis_or_yaw_error_deg'] = None
                            if observed is not None and truth.shape != 'ellipsoid':
                                geometric = metric_row(truth, SimpleNamespace(geometry=observed.geometry,
                                    center=observed.pose.position, size=observed.size))
                                row['dimension_error_mm'] = geometric['dimension_error_mm']
                                row['axis_or_yaw_error_deg'] = geometric['axis_error_deg']
                            row['geometry_evidence'] = None if observed is None else dict(
                                scores=observed.geometry.scores, metadata=observed.geometry.metadata,
                                graspable=observed.graspable, estimated_size=observed.size.tolist(),
                                estimated_center=observed.pose.position.tolist(),
                                box_size=None if observed.geometry.box_size is None else observed.geometry.box_size.tolist(),
                                yaw=observed.geometry.yaw, radius=observed.geometry.radius, length=observed.geometry.length,
                                axis=None if observed.geometry.axis is None else observed.geometry.axis.tolist())
                            rows.append(row)
                if scene.name.endswith('_ordinary'):
                    for method, updater in updaters.items():
                        populate_scene(scene.objects)
                        source_rigs[method].push(frame_step(world, rig))
                        updater.update()
                        contracts.append(dict(method=method,
                            **contract_queries(world, rig, source_rigs[method], updater, scene.objects[0], backend)))
                oracle.extend(record['frames'][-1].get('oracle_clouds', []))
                manifest.append(record)
                write_json(output/'capture_manifest.json', dict(kind='native_readonly_grasp_readiness', scenes=manifest))
                write_json(output/'rows.json', rows)
                write_json(output/'samples.json', samples)
                write_json(output/'contract_queries.json', contracts)
                write_json(output/'oracle_cloud_manifest.json', oracle)
                print('read-only planned', scene.key, flush=True)
            except Exception as error:
                errors.append(dict(scene=scene.key, error=f'{type(error).__name__}: {error}'))
                write_json(output/'runtime_errors.json', errors)
                raise
            finally:
                for updater in updaters.values():
                    updater.stop()
        write_csv(output/'rows.csv', rows)
        write_csv(output/'samples.csv', samples)
        summary = {}
        for method in classes:
            selected = [row for row in rows if row['method'] == method]
            summary[method] = dict(expected=len(selected), detected=sum(row['detected'] for row in selected),
                truth_feasible=sum(row['truth_feasible'] for row in selected),
                estimated_feasible=sum(row['estimated_feasible'] for row in selected),
                false_feasible=sum(row['false_feasible'] for row in selected),
                false_infeasible=sum(row['false_infeasible'] for row in selected),
                correct_classifications=sum(row['classification_correct'] for row in selected),
                physical_clearance_violations=sum(bool(row['proposed_truth_checks'] and
                    row['proposed_truth_checks']['clearance_violation']) for row in selected),
                physical_overlaps=sum(bool(row['proposed_truth_checks'] and
                    row['proposed_truth_checks']['physical_overlap']) for row in selected),
                proposed_width_violations=sum(bool(row['proposed_truth_checks'] and
                    row['proposed_truth_checks']['width_violation']) for row in selected),
                endpoint_ik_unreachable=sum(row['endpoint_ik_all_reachable'] is False for row in selected),
                center_error_mm=distribution([row['center_error_mm'] for row in selected]),
                dimension_error_mm=distribution([row['dimension_error_mm'] for row in selected]),
                axis_or_yaw_error_deg=distribution([row['axis_or_yaw_error_deg'] for row in selected]))
        unsafe = any(row['false_feasible'] or row['physical_clearance_violations'] or row['proposed_width_violations']
                     for row in summary.values())
        contract_failed = any(not row['passed'] for row in contracts)
        measured_contract_failure = any(row['identity_switches'] or row['classification_switches'] or row['early_confirmations'] or row['health'] != 'fresh'
                                        for row in samples)
        passed = not (backend.commands or errors or contract_failed or measured_contract_failure or (args.strict and unsafe))
        write_result(args.result, status='PASS' if passed else 'FAIL', metrics=dict(
            kind='read_only_perception_to_unchanged_grasp_planner', summary=summary, scenes=len(scenes),
            backend_commands=backend.commands, ik_query_count=backend.ik_queries,
            ik_available=native_backend is not None, contract_queries_passed=not contract_failed,
            sample_contracts_passed=not measured_contract_failure, runtime_errors=len(errors),
            source_health_failure_samples=sum(row['health'] != 'fresh' for row in samples),
            safe_geometry_decisions=not unsafe, strict_readiness_gate=args.strict,
            no_pick_or_place_submitted=True,
            scope='Geometry/known-object-envelope planning plus optional endpoint IK. No path, physical grasp or manipulation validation.',
            freshness='Actual native captures applied by unchanged WorldModelUpdater; stale/failed checks use existing synchronous Pick preconditions.'))
        return 0 if passed else 1
    finally:
        rig.close()


def parser(strict=False):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--result', required=True)
    result.add_argument('--artifacts-dir')
    result.add_argument('--split', choices=('selection', 'holdout'), default='selection')
    result.add_argument('--baseline-pythonpath')
    result.add_argument('--baseline-package', default='baseline_multiview')
    result.add_argument('--baseline-only', action='store_true')
    result.add_argument('--no-ik', dest='ik', action='store_false', default=True)
    result.add_argument('--limit', type=int)
    result.add_argument('--strict', action='store_true', default=strict)
    return result


def cli(strict=False):
    args = parser(strict).parse_args()
    if args.artifacts_dir is None:
        args.artifacts_dir = str(Path(args.result).parent/'grasp_readiness')
    if args.baseline_only and not args.baseline_pythonpath:
        raise SystemExit('--baseline-only needs --baseline-pythonpath')
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be positive')
    args.replay = None
    app = None
    code = 1
    try:
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy1.26.4 before SimulationApp, got{np.__version__}')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': False})
        code = main(args, app)
    except Exception as error:
        fail_result(args.result, error)
    finally:
        ending = source_provenance(args)
        for filename in (Path(__file__).resolve(), ROOT/'tests/support/multiview_readiness_research.py'):
            ending['sha256'][str(filename.relative_to(ROOT))] = hashlib.sha256(filename.read_bytes()).hexdigest()
        write_json(Path(args.artifacts_dir)/'provenance_end.json', ending)
        if app is not None:
            app.close()
    raise SystemExit(code)


if __name__ == '__main__':
    cli()
