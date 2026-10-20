"""Analyze saved real-Franka perception evidence without Isaac or model fitting.

Visibility ratios are point counts divided by each object's observed run maximum,
not calibrated physical visibility fractions. Optional depth proximity measures
TCP-proximal non-target geometry, not a semantic gripper mask or collision proof.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def distribution(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not values else dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
        min=float(np.min(values)), worst=float(np.max(values)), samples=len(values))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def depth_proximity(root, frame, sample, *, stride, roi_radius, support_z):
    from robot_skill_stack.world.perception_multiview.clouds import project_depth
    from tests.support.multiview_iteration2 import object_from_payload, surface_memberships
    tcp = sample.get('tcp_position')
    if tcp is None:
        return dict(available=False, reason='No measured TCP position stored')
    objects = [object_from_payload(obj) for obj in frame['objects']]
    clouds, labels = [], []
    for filename in frame['files']:
        with np.load(root/filename, allow_pickle=False) as saved:
            snapshot = SimpleNamespace(depth=saved['depth'], intrinsics=saved['intrinsics'],
                                       world_from_camera=saved['world_from_camera'])
            cloud = project_depth(snapshot, stride=stride)[0]
        keep = np.isfinite(cloud).all(1) & (cloud[:, 2] > support_z+.005)
        cloud = cloud[keep]
        clouds.append(cloud)
        labels.append(surface_memberships(cloud, objects))
    cloud = np.concatenate(clouds) if clouds else np.empty((0, 3))
    membership = np.concatenate(labels) if labels else np.empty(0, int)
    nearby = cloud[(membership < 0) & (np.linalg.norm(cloud-np.asarray(tcp), axis=1) <= roi_radius)]
    per_object = {}
    for oi, obj in enumerate(objects):
        surface = cloud[membership == oi]
        minimum = None
        if len(surface) and len(nearby):
            minimum = min(float(np.min(np.sum((surface[begin:begin+128, None]-nearby[None])**2, axis=2)))
                          for begin in range(0, len(surface), 128))**.5
        per_object[obj.name] = dict(target_depth_points=len(surface), tcp_roi_other_depth_points=len(nearby),
            measured_minimum_depth_surface_distance_mm=None if minimum is None else minimum*1000)
    return dict(available=True, depth_stride=stride, tcp_roi_radius_mm=roi_radius*1000,
                support_plane_exclusion_mm=5., objects=per_object,
                attribution='Non-target depth within measured TCP ROI; may include gripper, wrist, arm or background. '
                            'Sampled current-depth distance, not a continuous-surface lower bound or collision certificate.')


def analyze(root, *, minimum_visible_points=8, include_depth_proximity=False, depth_stride=8, tcp_roi_radius=.12):
    root = Path(root).resolve()
    manifest = json.loads((root/'capture_manifest.json').read_text())
    samples = json.loads((root/'samples.json').read_text())
    rows = json.loads((root/'rows.json').read_text())
    detections = json.loads((root/'detections.json').read_text())
    frames = {(scene['key'], frame['index']): frame for scene in manifest['scenes'] for frame in scene['frames']}
    scene_by_key = {scene['key']: scene for scene in manifest['scenes']}
    row_lookup = {(row['trial'], row['frame_index'], row['object']): row for row in rows}
    counts = {(row['trial'], row['frame_index']): row for row in detections}
    maxima = {}
    for frame in frames.values():
        for camera, objects in frame.get('visible_surface_points', {}).items():
            for name, value in objects.items():
                maxima[(camera, name)] = max(maxima.get((camera, name), 0), value)
    frame_rows, object_rows, visibility_rows, events = [], [], [], []
    identity_states = {}
    identity_changes, first_ids = [], {}
    last_timestamp, previous_sample = None, None
    previous_tokens = {}
    timestamp_issues = []
    integrity_issues = []
    phases = {}
    for sample_index, sample in enumerate(samples):
        phase = sample['phase']
        key = (phase, sample.get('frame_index'))
        fresh = sample.get('fresh_capture', sample.get('batch_timestamp') is not None)
        stamp, delivered = sample.get('batch_timestamp'), sample.get('delivered_at')
        count = counts.get(key, {})
        confirmation = count.get('confirmation_frame', False)
        if not count and key in frames:
            confirmation = scene_by_key[phase].get('needs_confirmation', False) and sample['frame_index'] < 2
        record = dict(phase=phase, sample_index=sample_index, frame_index=sample.get('frame_index'),
            expected_objects=len(frames.get(key, {}).get('objects', [])),
            fresh_capture=fresh, confirmation_frame=confirmation, source_health=sample.get('source_health'),
            source_error=sample.get('source_error'), batch_timestamp=stamp, captured_at=sample.get('captured_at'),
            delivered_at=delivered, **{key: value for key, value in sample.items() if key.endswith('_ms')})
        diagnostics = sample.get('diagnostics', {})
        record.update({key: value for key, value in diagnostics.items() if key.endswith('_ms')})
        for field in ('numerical_fits', 'fit_budget', 'fit_budget_exhaustions'):
            record[field] = diagnostics.get(field)
        synchronization = diagnostics.get('synchronization', {})
        record['synchronization_state'] = synchronization.get('state')
        record['frame_skew_ms'] = None if synchronization.get('skew_seconds') is None else synchronization['skew_seconds']*1000
        if fresh:
            if stamp is None or not np.isfinite(stamp):
                timestamp_issues.append(dict(sample_index=sample_index, kind='fresh_capture_without_batch_timestamp'))
            else:
                if last_timestamp is not None and stamp <= last_timestamp:
                    timestamp_issues.append(dict(sample_index=sample_index, kind='nonincreasing_fresh_batch_timestamp'))
                if delivered is not None and stamp > delivered:
                    timestamp_issues.append(dict(sample_index=sample_index, kind='future_batch_timestamp'))
                if sample.get('captured_at') is not None and abs(stamp-sample['captured_at']) > 1e-8:
                    timestamp_issues.append(dict(sample_index=sample_index, kind='batch_timestamp_differs_from_native_capture'))
                last_timestamp = stamp
            for camera, token in synchronization.get('tokens', {}).items():
                if previous_tokens.get(camera) == token:
                    timestamp_issues.append(dict(sample_index=sample_index, camera=camera,
                                                 kind='reused_camera_token_in_fresh_batch'))
                previous_tokens[camera] = token
        elif stamp is not None:
            timestamp_issues.append(dict(sample_index=sample_index, kind='batch_timestamp_without_fresh_capture'))
        if previous_sample is not None:
            record['interval_crosses_phase'] = previous_sample['phase'] != phase
            if delivered is not None and previous_sample.get('delivered_at') is not None:
                record['measured_delivery_interval_ms'] = (delivered-previous_sample['delivered_at'])*1000
            if sample.get('simulation_time_s') is not None and previous_sample.get('simulation_time_s') is not None:
                record['physics_interval_ms'] = (sample['simulation_time_s']-previous_sample['simulation_time_s'])*1000
            if sample.get('physics_step') is not None and previous_sample.get('physics_step') is not None:
                record['physics_step_interval'] = sample['physics_step']-previous_sample['physics_step']
        previous_sample = sample
        frame_rows.append(record)
        phases.setdefault(phase, []).append(record)
        frame = frames.get(key)
        if frame is None:
            if fresh:
                integrity_issues.append(dict(sample_index=sample_index, phase=phase,
                                             kind='fresh_sample_missing_capture_manifest_frame'))
            continue
        proximity = (depth_proximity(root, frame, sample, stride=depth_stride, roi_radius=tcp_roi_radius,
                                     support_z=manifest.get('workspace_min', [0., 0., 0.])[2])
                     if include_depth_proximity and frame['objects'] else None)
        for truth in frame['objects']:
            name = truth['name']
            visible = {camera: values.get(name, 0) for camera, values in frame.get('visible_surface_points', {}).items()}
            union = sum(visible.values())
            row = row_lookup.get((phase, frame['index'], name), {})
            if not row:
                integrity_issues.append(dict(sample_index=sample_index, object=name,
                                             kind='capture_truth_missing_scoring_row'))
            measured = bool(row.get('detected'))
            current_id = row.get('object_id') if measured else None
            result = dict(phase=phase, sample_index=sample_index, frame_index=frame['index'], object=name,
                confirmation_frame=confirmation, camera_surface_points=visible, camera_union_surface_points=union,
                fully_hidden_all_cameras=bool(visible) and union == 0,
                camera_visibility_available=bool(visible), camera_count=len(visible),
                fully_hidden_both_cameras=len(visible) >= 2 and union == 0,
                sufficient_visible_points=union >= minimum_visible_points,
                one_view_only=sum(value > 0 for value in visible.values()) == 1,
                detected=measured, correct=bool(row.get('correct')), truth=row.get('truth'), predicted=row.get('predicted', 'unscored'),
                object_id=current_id, confidence=row.get('confidence'), score_margin=row.get('score_margin'),
                missing=not measured, missing_with_sufficient_visibility=not measured and union >= minimum_visible_points,
                missing_with_fully_hidden_depth=not measured and bool(visible) and union == 0,
                detected_while_fully_hidden=measured and bool(visible) and union == 0,
                center_error_mm=row.get('center_error_mm'), dimension_error_mm=row.get('dimension_error_mm'),
                radius_error_mm=row.get('radius_error_mm'), length_error_mm=row.get('length_error_mm'),
                axis_error_deg=row.get('axis_error_deg'))
            if proximity is not None:
                result['tcp_depth_proximity'] = proximity.get('objects', {}).get(name)
            object_rows.append(result)
            for camera, value in visible.items():
                maximum = maxima[(camera, name)]
                visibility_rows.append(dict(phase=phase, object=name, camera=camera, sample_index=sample_index,
                    surface_points=value, run_max_surface_points=maximum,
                    ratio_to_observed_run_max=None if maximum == 0 else value/maximum))
            state = identity_states.setdefault(name, dict(last_id=None, hidden_event=None, previous_union=None))
            if measured and current_id is not None:
                first_ids.setdefault(name, current_id)
                if state['last_id'] is not None and state['last_id'] != current_id:
                    identity_changes.append(dict(object=name, sample_index=sample_index, phase=phase,
                                                 previous_id=state['last_id'], current_id=current_id))
            if visible and union == 0 and state['hidden_event'] is None:
                state['hidden_event'] = dict(object=name, start_sample=sample_index, start_phase=phase,
                    start_capture_time=sample.get('captured_at', stamp), prior_id=state['last_id'],
                    previously_detected=state['last_id'] is not None, fully_hidden_samples=0,
                    first_revisible_capture_time=None, first_revisible_delivery_time=None,
                    first_detected_capture_time=None, first_detected_delivery_time=None,
                    first_correct_delivery_time=None, reacquired_id=None, correct_reacquired_id=None)
                events.append(state['hidden_event'])
            event = state['hidden_event']
            if event is not None:
                if visible and union == 0:
                    event['fully_hidden_samples'] += 1
                if union >= minimum_visible_points and event['first_revisible_capture_time'] is None:
                    event['first_revisible_sample'] = sample_index
                    event['first_revisible_capture_time'] = sample.get('captured_at', stamp)
                    event['first_revisible_delivery_time'] = delivered
                if union >= minimum_visible_points and event['first_revisible_capture_time'] is not None:
                    if measured and event['first_detected_capture_time'] is None:
                        event['first_detected_capture_time'] = sample.get('captured_at', stamp)
                        event['first_detected_delivery_time'] = delivered
                        event['reacquired_id'] = current_id
                    if row.get('correct'):
                        event['first_correct_delivery_time'] = delivered
                        event['correct_reacquired_id'] = current_id
                        state['hidden_event'] = None
            if measured and current_id is not None:
                state['last_id'] = current_id
            state['previous_union'] = union
    for event in events:
        event['identity_preserved'] = (None if event['prior_id'] is None or event['correct_reacquired_id'] is None
                                       else event['prior_id'] == event['correct_reacquired_id'])
        event['detected_reacquisition_from_first_revisible_capture_ms'] = (None if event['first_detected_delivery_time'] is None
            else (event['first_detected_delivery_time']-event['first_revisible_capture_time'])*1000)
        event['correct_reacquisition_from_first_revisible_capture_ms'] = (None if event['first_correct_delivery_time'] is None
            else (event['first_correct_delivery_time']-event['first_revisible_capture_time'])*1000)
        event['event_kind'] = 'reacquisition' if event['previously_detected'] else 'initial_acquisition_after_hidden_start'
        event['censored_without_correct_reacquisition'] = event['first_correct_delivery_time'] is None
    phase_summaries = {}
    for phase, selected_frames in phases.items():
        selected_objects = [row for row in object_rows if row['phase'] == phase and not row['confirmation_frame']]
        selected_counts = [row for row in detections if row['trial'] == phase and not row.get('confirmation_frame', False)]
        measures = sorted({key for row in selected_frames for key, value in row.items()
                           if key.endswith('_ms') and isinstance(value, (float, int))})
        fit_rows = [row for row in selected_frames if row['numerical_fits'] is not None and row['fit_budget'] is not None]
        phase_summaries[phase] = dict(samples=len(selected_frames), expected_after_confirmation=len(selected_objects),
            correct_after_confirmation=sum(row['correct'] for row in selected_objects),
            unknown_after_confirmation=sum(row['predicted'] == 'unknown' for row in selected_objects),
            predicted_classes={name: sum(row['predicted'] == name for row in selected_objects)
                               for name in sorted({row['predicted'] for row in selected_objects})},
            correct_geometry_errors={field: distribution([row[field] for row in selected_objects if row['correct']])
                                     for field in ('center_error_mm', 'dimension_error_mm', 'radius_error_mm',
                                                   'length_error_mm', 'axis_error_deg')},
            accepted_confidence=distribution([row['confidence'] for row in selected_objects if row['detected']]),
            accepted_score_margin=distribution([row['score_margin'] for row in selected_objects if row['detected']]),
            missing_after_confirmation=sum(row['missing'] for row in selected_objects),
            missing_visible=sum(row['missing_with_sufficient_visibility'] for row in selected_objects),
            missing_fully_hidden=sum(row['missing_with_fully_hidden_depth'] for row in selected_objects),
            missing_low_density=sum(row['missing'] and 0 < row['camera_union_surface_points'] < minimum_visible_points for row in selected_objects),
            fully_hidden_object_samples=sum(row['fully_hidden_all_cameras'] for row in selected_objects),
            one_view_object_samples=sum(row['one_view_only'] for row in selected_objects),
            **{key: sum(row.get(key, 0) for row in selected_counts) for key in
               ('false_positives', 'false_splits', 'mixed_accepted_clouds', 'duplicate_ids', 'identity_switches')},
            source_failures=sum(row['source_health'] == 'failed' for row in selected_frames),
            source_nonfresh_states=sum(row['source_health'] != 'fresh' for row in selected_frames),
            no_fresh_capture_samples=sum(not row['fresh_capture'] for row in selected_frames),
            latency_ms={key: distribution([row.get(key) for row in selected_frames]) for key in measures},
            physics_step_intervals=distribution([row.get('physics_step_interval') for row in selected_frames]),
            numerical_fit_count=distribution([row['numerical_fits'] for row in selected_frames]),
            fit_budget_exhaustions=sum(row.get('fit_budget_exhaustions') or 0 for row in selected_frames),
            fit_cap_violations=sum(row['numerical_fits'] > row['fit_budget'] for row in fit_rows))
    visibility_summary = []
    for phase, name, camera in sorted({(row['phase'], row['object'], row['camera']) for row in visibility_rows}):
        selected = [row for row in visibility_rows if (row['phase'], row['object'], row['camera']) == (phase, name, camera)]
        visibility_summary.append(dict(phase=phase, object=name, camera=camera,
            points=distribution([row['surface_points'] for row in selected]),
            ratio_to_observed_run_max=distribution([row['ratio_to_observed_run_max'] for row in selected])))
    evaluated = [row for row in object_rows if not row['confirmation_frame']]
    proximity_summaries = []
    if include_depth_proximity:
        for phase, name in sorted({(row['phase'], row['object']) for row in object_rows}):
            selected = [row['tcp_depth_proximity'] for row in object_rows
                        if (row['phase'], row['object']) == (phase, name) and row.get('tcp_depth_proximity')]
            proximity_summaries.append(dict(phase=phase, object=name,
                sampled_minimum_surface_distance_mm=distribution([
                    row['measured_minimum_depth_surface_distance_mm'] for row in selected]),
                samples_with_target_and_tcp_roi_geometry=sum(
                    row['measured_minimum_depth_surface_distance_mm'] is not None for row in selected)))
    summary = dict(kind='offline_real_franka_artifact_analysis', input=str(root), samples=len(samples),
        source_result_status=None if not (root/'result.json').exists() else json.loads((root/'result.json').read_text()).get('status'),
        expected_after_confirmation=len(evaluated), correct_after_confirmation=sum(row['correct'] for row in evaluated),
        missing_after_confirmation=sum(row['missing'] for row in evaluated),
        fully_hidden_object_samples=sum(row['fully_hidden_all_cameras'] for row in evaluated),
        fully_hidden_both_cameras_established=any(row['fully_hidden_both_cameras'] for row in evaluated),
        one_view_object_samples=sum(row['one_view_only'] for row in evaluated),
        missing_visible=sum(row['missing_with_sufficient_visibility'] for row in evaluated),
        missing_fully_hidden=sum(row['missing_with_fully_hidden_depth'] for row in evaluated),
        missing_low_density=sum(row['missing'] and 0 < row['camera_union_surface_points'] < minimum_visible_points for row in evaluated),
        missing_visibility_unavailable=sum(row['missing'] and not row['camera_visibility_available'] for row in evaluated),
        detected_while_fully_hidden=sum(row['detected_while_fully_hidden'] for row in evaluated),
        unknown_after_confirmation=sum(row['predicted'] == 'unknown' for row in evaluated),
        correct_geometry_errors={field: distribution([row[field] for row in evaluated if row['correct']])
                                 for field in ('center_error_mm', 'dimension_error_mm', 'radius_error_mm',
                                               'length_error_mm', 'axis_error_deg')},
        accepted_confidence=distribution([row['confidence'] for row in evaluated if row['detected']]),
        accepted_score_margin=distribution([row['score_margin'] for row in evaluated if row['detected']]),
        **{key: sum(value[key] for value in phase_summaries.values()) for key in
           ('false_positives', 'false_splits', 'mixed_accepted_clouds', 'duplicate_ids', 'identity_switches',
            'fit_cap_violations', 'fit_budget_exhaustions', 'source_nonfresh_states')},
        numerical_fit_count=distribution([row['numerical_fits'] for row in frame_rows]),
        latency_ms={key: distribution([row.get(key) for row in frame_rows]) for key in
                    sorted({key for row in frame_rows for key in row if key.endswith('_ms')})},
        occupied_fresh_sample_latency_ms={key: distribution([row.get(key) for row in frame_rows
            if row['fresh_capture'] and row['expected_objects']]) for key in
            sorted({key for row in frame_rows for key in row if key.endswith('_ms')})},
        physics_step_intervals=distribution([row.get('physics_step_interval') for row in frame_rows]),
        timestamp_issues=timestamp_issues, source_failures=sum(row.get('source_health') == 'failed' for row in samples),
        artifact_integrity_issues=integrity_issues, minimum_visible_points=minimum_visible_points,
        fresh_capture_samples=sum(row['fresh_capture'] for row in frame_rows),
        identity_changes=identity_changes, initial_ids=first_ids, occlusion_events=events,
        successful_correct_reacquisitions=sum(event['event_kind'] == 'reacquisition' and not event['censored_without_correct_reacquisition'] for event in events),
        reacquisition_latency_ms=distribution([event['correct_reacquisition_from_first_revisible_capture_ms']
            for event in events if event['event_kind'] == 'reacquisition']),
        per_phase=phase_summaries, visibility=visibility_summary,
        depth_proximity_computed=include_depth_proximity,
        tcp_depth_proximity=proximity_summaries,
        methodological_limits=[
            'Confirmation frames remain in raw exports and are excluded only from explicitly named confirmed summaries.',
            'Fully hidden means zero scoring-surface points in all saved camera views; low-density visibility is separately reported.',
            'Ratios use observed run maxima; viewpoint, size, sampling and motion affect counts, so they are not physical occlusion percentages.',
            'Reacquisition time starts at the first sampled capture with at least the configured visible points, not the unobserved physical reveal instant.',
            'Delivery intervals include recording/analysis overhead between calls and are not deployment throughput.',
            'Overall timing distributions include empty scenes; occupied fresh-sample timing distributions are exported separately.',
            'TCP depth proximity is sampled distance to non-target measured geometry near the TCP, not semantic gripper clearance.',
            'IK, swept-path safety and manipulation success are not established by this perception analyzer.'],
        input_sha256={filename: hashlib.sha256((root/filename).read_bytes()).hexdigest() for filename in
                      ('capture_manifest.json', 'samples.json', 'rows.json', 'detections.json')})
    return summary, frame_rows, object_rows, visibility_rows


def main(args):
    summary, frames, objects, visibility = analyze(args.input, minimum_visible_points=args.minimum_visible_points,
        include_depth_proximity=args.depth_proximity, depth_stride=args.depth_stride, tcp_roi_radius=args.tcp_roi_radius)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, rows in (('frames', frames), ('objects', objects), ('visibility', visibility),
                            ('reacquisition', summary['occlusion_events'])):
        (output/f'{filename}.json').write_text(json.dumps(rows, indent=2)+'\n')
        write_csv(output/f'{filename}.csv', rows)
    write_csv(output/'phases.csv', [dict(phase=phase, **values) for phase, values in summary['per_phase'].items()])
    write_csv(output/'depth_proximity.csv', summary['tcp_depth_proximity'])
    print(json.dumps({key: summary[key] for key in ('samples', 'expected_after_confirmation',
        'correct_after_confirmation', 'missing_after_confirmation', 'fully_hidden_both_cameras_established',
        'source_failures', 'successful_correct_reacquisitions', 'depth_proximity_computed')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--minimum-visible-points', type=int, default=8)
    parser.add_argument('--depth-proximity', action='store_true')
    parser.add_argument('--depth-stride', type=int, default=8)
    parser.add_argument('--tcp-roi-radius', type=float, default=.12)
    arguments = parser.parse_args()
    if min(arguments.minimum_visible_points, arguments.depth_stride, arguments.tcp_roi_radius) <= 0:
        parser.error('Visibility threshold, depth stride and TCP ROI radius must be positive')
    main(arguments)
