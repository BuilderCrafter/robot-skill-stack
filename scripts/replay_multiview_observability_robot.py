"""Persistent saved-Franka replay without simulator, robot or motion commands.

Both providers consume the original ordered camera arrays/tokens/timestamps.
An explicit logical clock follows native capture times through the unchanged
updater. Reacquisition latency uses the SAVED capture timeline; CPU source stage
times are measured separately. Neither is new live latency or update frequency.
Robot bodies are not supplied as oracle solids or planning obstacles.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from scripts.benchmark_multiview_grasp_readiness import plain_values, write_csv, write_json
from scripts.benchmark_multiview_observability import package_hashes
from scripts.benchmark_multiview_robustness import load_frames
from tests.support.multiview_iteration2 import SnapshotRig, distribution, object_from_payload, score_observations
from tests.support.multiview_observability_evaluation import RecordedSource


def replay_robot(input_dir, output, baseline_class, *, current_class=MultiViewProvider, minimum_visible_points=8):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir/'capture_manifest.json').read_text())
    original_samples = json.loads((input_dir/'samples.json').read_text())
    native_counts = json.loads((input_dir/'detections.json').read_text())
    original_confirmation = {(row['trial'], row['frame_index']): row.get('confirmation_frame', False) for row in native_counts}
    frame_lookup = {(scene['key'], frame['index']): frame for scene in manifest['scenes'] for frame in scene['frames']}
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    scene = replace(cfg.scene, workspace_min=np.asarray(manifest['workspace_min']),
                    workspace_max=np.asarray(manifest['workspace_max']))
    rows, samples, detections, observations, events, errors, checks = [], [], [], [], [], [], []
    instances = {}
    for method, cls in [('baseline', baseline_class), ('improved', current_class)]:
        rig = SnapshotRig()
        source = cls(rig, scene)
        model = WorldModel()
        recorder = RecordedSource(source)
        instances[method] = dict(rig=rig, source=source, recorder=recorder, model=model,
            updater=WorldModelUpdater(model, recorder, stale_object_ttl=15.), previous_ids={}, states={})
        instances[method]['previous_labels'] = {}
    start_hashes = package_hashes()
    write_json(output/'provenance_start.json', dict(package_sha256=start_hashes,
        baseline_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(sys.modules[baseline_class.__module__].__file__).parent.glob('*.py')},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={name: hashlib.sha256((input_dir/name).read_bytes()).hexdigest() for name in
                      ('capture_manifest.json', 'samples.json', 'detections.json')}))
    try:
        for sample_index, original in enumerate(original_samples):
            key = original['phase'], original.get('frame_index')
            snapshot = frame_lookup.get(key)
            frames = None if snapshot is None else load_frames(input_dir, snapshot['files'])
            objects = () if snapshot is None else tuple(object_from_payload(obj) for obj in snapshot['objects'])
            now = (min(frame.acquired_at for frame in frames)+1e-6 if frames is not None
                   else original['delivered_at'])
            for method, instance in instances.items():
                source, updater, model = (instance[name] for name in ('source', 'updater', 'model'))
                accepted, failed, before_updates = [], None, updater.update_count
                try:
                    if frames is not None:
                        instance['rig'].push(frames)
                    started = time.perf_counter()
                    with patch('time.monotonic', return_value=now):
                        accepted = updater.update()
                        health = updater.health.status
                        batch = instance['recorder'].batch
                        stamp = None if batch is None else batch.timestamp
                        if frames is not None:
                            expected_stamp = min(frame.acquired_at for frame in frames)
                            stamp_ok = (stamp == expected_stamp and updater.update_count == before_updates+1 and
                                        all(obs.timestamp == expected_stamp for obs in accepted))
                        else:
                            stamp_ok = stamp is None and not accepted
                    elapsed = (time.perf_counter()-started)*1000
                    diagnostic = copy.deepcopy(source.diagnostics)
                except Exception as error:
                    failed = f'{type(error).__name__}: {error}'
                    errors.append(dict(method=method, sample_index=sample_index, error=failed))
                    health, stamp, stamp_ok, elapsed, diagnostic = 'failed', None, False, None, {}
                confirmed = original_confirmation.get(key, False)
                sample = dict(method=method, sample_index=sample_index, phase=original['phase'],
                    frame_index=original.get('frame_index'), confirmation_frame=confirmed,
                    original_acquired_at=None if frames is None else min(frame.acquired_at for frame in frames),
                    original_delivered_at=original.get('delivered_at'), fresh_capture=frames is not None,
                    fresh_consumed=updater.update_count > before_updates, source_health=health,
                    batch_timestamp=stamp, timestamps_preserved=stamp_ok, offline_provider_ms=elapsed,
                    observed=len(accepted), diagnostics=diagnostic, error=failed)
                samples.append(sample)
                scored, count, ids = score_observations(objects, accepted,
                    [source.get_point_cloud(obs.object_id) for obs in accepted], method=method,
                    trial=original['phase'], frame_index=original.get('frame_index'),
                    previous_ids=instance['previous_ids'])
                instance['previous_ids'].update(ids)
                count.update(sample_index=sample_index, confirmation_frame=confirmed,
                    unmatched_workspace_observations=count['false_positives'])
                detections.append(count)
                for row in scored:
                    name = row['object']
                    visible = {camera: values.get(name, 0) for camera, values in snapshot['visible_surface_points'].items()}
                    union = sum(visible.values())
                    row.update(sample_index=sample_index, confirmation_frame=confirmed,
                               camera_surface_points=visible, union_surface_points=union)
                    measured = next((obs for obs in accepted if obs.object_id == row.get('object_id')), None)
                    row['graspable'] = None if measured is None else measured.graspable
                    row['geometry_metadata'] = None if measured is None else measured.geometry.metadata
                    row['classification_switch'] = bool(row['detected'] and name in instance['previous_labels'] and
                        instance['previous_labels'][name] != row['predicted'])
                    if row['detected']:
                        instance['previous_labels'][name] = row['predicted']
                    rows.append(row)
                    state = instance['states'].setdefault(name, dict(last_id=None, event=None))
                    if union == 0 and state['event'] is None:
                        state['event'] = dict(method=method, object=name, hidden_sample=sample_index,
                            prior_id=state['last_id'], first_revisible_sample=None, first_revisible_capture_time=None,
                            first_correct_sample=None, first_correct_capture_time=None, reacquired_id=None)
                        events.append(state['event'])
                    event = state['event']
                    if event is not None and union >= minimum_visible_points:
                        if event['first_revisible_sample'] is None:
                            event.update(first_revisible_sample=sample_index, first_revisible_capture_time=now-1e-6)
                        if row['correct']:
                            event.update(first_correct_sample=sample_index, first_correct_capture_time=now-1e-6,
                                         reacquired_id=row['object_id'])
                            state['event'] = None
                    if measured is not None:
                        state['last_id'] = measured.object_id
                for obs in accepted:
                    observations.append(dict(method=method, sample_index=sample_index, phase=original['phase'],
                        object_id=obs.object_id, class_name=obs.class_name, timestamp=obs.timestamp,
                        graspable=obs.graspable, center=obs.pose.position, size=obs.size,
                        confidence=obs.geometry.confidence, geometry_metadata=obs.geometry.metadata,
                        geometry=dict(shape=obs.geometry.shape.value, box_size=obs.geometry.box_size,
                            yaw=obs.geometry.yaw, radius=obs.geometry.radius, length=obs.geometry.length,
                            axis=obs.geometry.axis),
                        observation_metadata=obs.metadata))
                if frames is not None:
                    with patch('time.monotonic', return_value=now+1e-5):
                        before = updater.health.last_frame_at
                        empty = updater.update()
                        unchanged = instance['recorder'].batch
                        checks.append(dict(method=method, sample_index=sample_index,
                            frozen_silent=not empty and unchanged is not None and unchanged.timestamp is None,
                            timestamp_unchanged=updater.health.last_frame_at == before))
            write_json(output/'samples.json', samples)
        for event in events:
            event['identity_preserved'] = None if event['prior_id'] is None or event['reacquired_id'] is None else event['prior_id'] == event['reacquired_id']
            event['saved_capture_timeline_reacquisition_ms'] = None if event['first_correct_capture_time'] is None else 1000*(
                event['first_correct_capture_time']-event['first_revisible_capture_time'])
            event['censored'] = event['first_correct_sample'] is None
        summaries = {}
        for method in instances:
            selected = [row for row in rows if row['method'] == method and not row['confirmation_frame']]
            selected_samples = [row for row in samples if row['method'] == method]
            selected_counts = [row for row in detections if row['method'] == method and not row['confirmation_frame']]
            summaries[method] = dict(samples=len(selected_samples), fresh_native_samples=sum(row['fresh_capture'] for row in selected_samples),
                consumed_native_samples=sum(row['fresh_consumed'] for row in selected_samples), expected=len(selected),
                detected=sum(row['detected'] for row in selected), correct=sum(row['correct'] for row in selected),
                unknown=sum(row['unknown'] for row in selected), missing=sum(not row['detected'] for row in selected),
                classification_switches=sum(row['classification_switch'] for row in selected),
                graspable=sum(row['graspable'] is True for row in selected),
                missing_visible=sum(not row['detected'] and row['union_surface_points'] >= minimum_visible_points for row in selected),
                missing_fully_hidden=sum(not row['detected'] and row['union_surface_points'] == 0 for row in selected),
                missing_low_density=sum(not row['detected'] and 0 < row['union_surface_points'] < minimum_visible_points for row in selected),
                **{field: sum(row.get(field, 0) for row in selected_counts) for field in
                   ('identity_switches', 'duplicate_ids', 'false_splits', 'mixed_accepted_clouds', 'unmatched_workspace_observations')},
                timestamp_contract_failures=sum(not row['timestamps_preserved'] for row in selected_samples),
                source_health_failures=sum(row['source_health'] != 'fresh' for row in selected_samples),
                offline_provider_ms=distribution([row['offline_provider_ms'] for row in selected_samples]),
                offline_observability_ms=distribution([row['diagnostics'].get('observability_ms') for row in selected_samples]),
                numerical_fit_count=distribution([row['diagnostics'].get('numerical_fits') for row in selected_samples]),
                fit_budget_exhaustions=sum(row['diagnostics'].get('fit_budget_exhaustions', 0) for row in selected_samples),
                fit_cap_violations=sum(row['diagnostics'].get('numerical_fits', 0) > row['diagnostics'].get('fit_budget', 24)
                                       for row in selected_samples),
                offline_stage_ms={field: distribution([row['diagnostics'].get(field) for row in selected_samples])
                    for field in sorted({field for row in selected_samples for field in row['diagnostics'] if field.endswith('_ms')})},
                reacquisition=[event for event in events if event['method'] == method])
        end_hashes = package_hashes()
        current = summaries['improved']
        completed = not errors and start_hashes == end_hashes and not any(
            current[field] for field in ('identity_switches', 'duplicate_ids', 'timestamp_contract_failures', 'source_health_failures'))
        completed = completed and all(row['frozen_silent'] and row['timestamp_unchanged'] for row in checks)
        result = dict(status='PASS' if completed else 'FAIL', metrics=dict(summary=summaries, runtime_errors=len(errors),
            source_hashes_stable=start_hashes == end_hashes, phases=len(manifest['scenes']),
            input=str(input_dir), persistent_sources_across_native_phases=True, robot_commands=0,
            scope='Functional saved-motion replay only. No live update rate, current wall-clock freshness, IK, '
                  'robot geometry oracle or manipulation/scene-planning safety claim. PASS verifies replay/source contracts; misses remain in metrics.',
            reacquisition_clock='First sufficiently visible native capture to accepted native capture on SAVED timeline; excludes new CPU delivery time.'), error=None)
        for name, values in [('rows', rows), ('samples', samples), ('observations', observations), ('detections', detections),
                             ('reacquisition', events), ('source_contract_checks', checks), ('runtime_errors', errors)]:
            write_json(output/f'{name}.json', values)
            write_csv(output/f'{name}.csv', values)
        write_json(output/'provenance_end.json', dict(package_sha256=end_hashes))
        write_json(output/'result.json', result)
        return result
    finally:
        for instance in instances.values():
            instance['updater'].stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-pythonpath', type=Path, required=True)
    parser.add_argument('--baseline-package', default='baseline_multiview')
    parser.add_argument('--minimum-visible-points', type=int, default=8)
    args = parser.parse_args()
    if args.minimum_visible_points < 1:
        parser.error('--minimum-visible-points must be positive')
    sys.path.insert(0, str(args.baseline_pythonpath.resolve()))
    baseline = importlib.import_module(f'{args.baseline_package}.provider').MultiViewProvider
    result = replay_robot(args.input, args.output, baseline, minimum_visible_points=args.minimum_visible_points)
    print(json.dumps(plain_values(result), indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
