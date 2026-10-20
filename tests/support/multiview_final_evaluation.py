"""Perception-only final checkpoint scoring and saved-frame source adapters.

Truth enters only after automatic source.observe(). No planner, controller,
robot backend or semantic/oracle mask is supplied to a source. Frame replay
preserves acquired_at and tokens; its logical clock is not a live rate claim.
Historical research scorers remain unchanged.
"""
from copy import deepcopy
from unittest.mock import patch
import time

import numpy as np

from robot_skill_stack.world.perception.factory import build_perception_provider
from tests.support.multiview_iteration2 import (
    SnapshotCamera, SnapshotRig, distribution, surface_memberships,
)


STAGES = ('acquisition_ms', 'projection_ms', 'fusion_ms', 'segmentation_ms',
          'fitting_ms', 'observability_ms', 'tracking_ms', 'total_ms')


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def angular_distance_degrees(first, second, period):
    return float(np.rad2deg(abs((first-second+period/2) % period-period/2)))


def transform_drift(reference, current):
    """Rigid translation/rotation drift, removing SDK floating-point scale error."""
    if set(reference) != set(current) or not reference:
        raise ValueError('Settling requires the same nonempty tracked link/hand transforms')
    shifts, angles = [], []
    for name, previous in reference.items():
        previous, value = np.asarray(previous), np.asarray(current[name])
        shifts.append(float(np.linalg.norm(value[:3, 3]-previous[:3, 3])))
        a, _, b = np.linalg.svd(previous[:3, :3]); before = a @ b
        a, _, b = np.linalg.svd(value[:3, :3]); after = a @ b
        angles.append(float(np.rad2deg(np.arccos(np.clip((np.trace(before.T @ after)-1)/2, -1., 1.)))))
    return max(shifts), max(angles)


def settle_physics(step, transforms, bounds, *, max_steps=600, stable_steps=60,
                   translation_m=.0001, rotation_deg=.01, aabb_m=.0001):
    """Measure passive initial physics settling; no pose/controller changes.

    Cheap cached link transforms are read every rendered physics step. World
    AABB is evaluated only at window starts and candidate stable endpoints.
    Every transform must stay close to the SAME window start, rather than
    permitting slow cumulative drift through consecutive adjacent comparisons.
    """
    if min(max_steps, stable_steps) < 1 or min(translation_m, rotation_deg, aabb_m) <= 0:
        raise ValueError('Settling limits and step counts must be positive')
    initial = transforms()
    initial_bounds = np.asarray(bounds(), float)
    reference, reference_bounds = initial, initial_bounds
    final, final_bounds = initial, initial_bounds
    consecutive, resets, steps, shift, angle, bound_delta = 0, 0, 0, 0., 0., 0.
    window_shift, window_angle = 0., 0.
    stable = False
    for steps in range(1, max_steps+1):
        step()
        final = transforms()
        shift, angle = transform_drift(reference, final)
        if shift > translation_m or angle > rotation_deg:
            reference, reference_bounds = final, np.asarray(bounds(), float)
            consecutive, resets = 0, resets+1
            window_shift, window_angle = 0., 0.
            continue
        window_shift, window_angle = max(window_shift, shift), max(window_angle, angle)
        consecutive += 1
        if consecutive < stable_steps:
            continue
        final_bounds = np.asarray(bounds(), float)
        bound_delta = float(np.max(abs(final_bounds-reference_bounds)))
        if bound_delta <= aabb_m:
            stable = True
            break
        reference, reference_bounds = final, final_bounds
        consecutive, resets = 0, resets+1
        window_shift, window_angle = 0., 0.
    if not stable:
        final_bounds = np.asarray(bounds(), float)
    return plain(dict(stable=stable, steps=steps, max_steps=max_steps,
        required_consecutive_steps=stable_steps, consecutive_stable_steps=consecutive, window_resets=resets,
        limits=dict(translation_m=translation_m, rotation_deg=rotation_deg, aabb_m=aabb_m),
        final_window_max_translation_m=window_shift, final_window_max_rotation_deg=window_angle,
        final_window_aabb_delta_m=bound_delta, initial_transforms=initial, final_transforms=final,
        initial_aabb=initial_bounds, final_aabb=final_bounds, no_robot_backend_or_commands=True,
        scope='Initial passive rendered physics settling before camera-layout trials; no robot pose or controller changes.'))


def box_orientation(truth, geometry):
    """Compare physical horizontal axes, preserving the rectangle's 180° rule.

    The near-square stratum max(1 mm, 2% span) describes evaluation ambiguity
    at roughly the existing voxel/dimension scale. It is not a source policy,
    confidence interval or a new rotational symmetry. Raw errors remain saved.
    """
    result = dict(yaw_error_deg=None, yaw_raw_major_axis_error_deg=None,
        yaw_symmetry_degrees=None, yaw_precision_eligible=False,
        yaw_ambiguity='not_a_correct_box', estimated_yaw=None, truth_yaw=float(truth.yaw))
    if geometry is None or geometry.shape.value != 'cube' or geometry.box_size is None:
        return result
    orientation = geometry.metadata.get('orientation_observability')
    hypothesis = geometry.yaw if geometry.yaw is not None else (
        None if orientation is None else orientation.get('yaw_hypothesis_rad'))
    if hypothesis is None:
        return result
    actual, predicted = np.asarray(truth.size), np.asarray(geometry.box_size)
    first = float(truth.yaw)+(np.pi/2 if actual[1] > actual[0] else 0.)
    second = float(hypothesis)+(np.pi/2 if predicted[1] > predicted[0] else 0.)
    exact_square = bool(abs(actual[0]-actual[1]) <= 1e-8)
    period = np.pi/2 if exact_square else np.pi
    raw = angular_distance_degrees(first, second, period)
    # Squares have observable face-grid yaw modulo90, but no unique major axis.
    truth_near = not exact_square and abs(actual[0]-actual[1]) <= max(.001, .02*max(actual[:2]))
    prediction_near = not exact_square and abs(predicted[0]-predicted[1]) <= max(.001, .02*max(predicted[:2]))
    evidence = geometry.metadata.get('parameter_observable')
    orientation_reason = None if orientation is None else (
        'face_grid_evidence_ambiguous' if exact_square and not orientation.get('face_grid_observable', False) else
        'major_axis_evidence_ambiguous' if not exact_square and not orientation.get('major_axis_observable', False) else None)
    reason = orientation_reason
    if reason is None:
        reason = ('near_square_truth' if truth_near else 'near_square_estimate' if prediction_near else
                  'incomplete_geometry_evidence' if evidence is False else None)
    result.update(yaw_error_deg=raw if reason is None else None,
        yaw_raw_major_axis_error_deg=raw, yaw_symmetry_degrees=90 if exact_square else 180,
        yaw_precision_eligible=reason is None, yaw_ambiguity=reason,
        estimated_yaw=None if geometry.yaw is None else float(geometry.yaw),
        fitted_yaw_hypothesis_rad=float(hypothesis), truth_major_axis_yaw=first,
        estimated_major_axis_yaw=second, truth_footprint_square=exact_square,
        near_square_truth=truth_near, near_square_estimate=prediction_near,
        source_face_grid_observable=None if orientation is None else orientation.get('face_grid_observable'),
        source_major_axis_observable=None if orientation is None else orientation.get('major_axis_observable'),
        near_square_tolerance_mm=float(max(.001, .02*max(actual[:2]))*1000))
    return result


def geometry_row(obj, observation):
    geometry = None if observation is None else observation.geometry
    predicted = 'missing' if observation is None else ('unknown' if geometry is None else geometry.shape.value)
    correct = predicted == obj.shape
    row = dict(object=obj.name, truth=obj.shape, predicted=predicted, correct=correct,
        detected=observation is not None, unknown=predicted == 'unknown',
        object_id=None if observation is None else observation.object_id,
        center_error_mm=None, dimension_error_mm=None, height_error_mm=None,
        footprint_long_error_mm=None, footprint_short_error_mm=None,
        radius_error_mm=None, radius_bias_mm=None, diameter_error_mm=None,
        cylinder_axis_error_deg=None, truth_center=obj.center.tolist(),
        truth_dimensions=obj.size.tolist(), estimated_dimensions=None,
        parameter_observable=None, graspable=None, confidence=None,
        estimated_center=None, observation_metadata=None, geometry_metadata=None,
        yaw_error_deg=None, yaw_raw_major_axis_error_deg=None,
        yaw_precision_eligible=False, yaw_ambiguity='not_a_box' if obj.shape != 'cube' else 'missing_or_wrong_class')
    if observation is None:
        return row
    row.update(center_error_mm=float(np.linalg.norm(observation.pose.position-obj.center)*1000),
        estimated_center=observation.pose.position.tolist(), estimated_size=observation.size.tolist(),
        graspable=observation.graspable, observation_timestamp=observation.timestamp,
        observation_metadata=deepcopy(observation.metadata))
    if geometry is None:
        return row
    row.update(confidence=float(geometry.confidence), scores=deepcopy(geometry.scores),
        geometry_metadata=deepcopy(geometry.metadata),
        parameter_observable=geometry.metadata.get('parameter_observable'),
        contributing_views=observation.metadata.get('contributing_views'))
    if obj.shape == 'cube':
        row.update(box_orientation(obj, geometry))
        if correct and geometry.box_size is not None:
            truth = np.r_[np.sort(obj.size[:2])[::-1], obj.size[2]]
            estimate = np.r_[np.sort(geometry.box_size[:2])[::-1], geometry.box_size[2]]
            errors = abs(estimate-truth)*1000
            row.update(truth_dimensions=truth.tolist(), estimated_dimensions=estimate.tolist(),
                estimated_box_size=np.asarray(geometry.box_size).tolist(),
                dimension_error_mm=float(max(errors)), footprint_long_error_mm=float(errors[0]),
                footprint_short_error_mm=float(errors[1]), height_error_mm=float(errors[2]),
                historical_sorted_xyz_error_mm=float(max(abs(np.sort(geometry.box_size)-np.sort(obj.size)))*1000))
    elif correct and obj.shape in ('sphere', 'cylinder') and geometry.radius is not None:
        bias = float((geometry.radius-obj.radius)*1000)
        row.update(radius_error_mm=abs(bias), radius_bias_mm=bias, diameter_error_mm=abs(2*bias))
        if obj.shape == 'sphere':
            row.update(dimension_error_mm=abs(2*bias), estimated_dimensions=[2*float(geometry.radius)]*3)
        elif geometry.length is not None:
            row.update(dimension_error_mm=max(abs(2*bias), abs(geometry.length-obj.length)*1000),
                estimated_dimensions=[2*float(geometry.radius), 2*float(geometry.radius), float(geometry.length)])
            if geometry.axis is not None:
                row['cylinder_axis_error_deg'] = float(np.rad2deg(np.arccos(np.clip(
                    abs(np.asarray(obj.axis) @ geometry.axis), 0., 1.))))
    return row


def raw_observation(observation):
    g = observation.geometry
    return plain(dict(object_id=observation.object_id, class_name=observation.class_name,
        shape='unknown' if g is None else g.shape.value, center=observation.pose.position,
        orientation=observation.pose.orientation, size=observation.size,
        timestamp=observation.timestamp, graspable=observation.graspable,
        metadata=deepcopy(observation.metadata), geometry=None if g is None else dict(
            confidence=g.confidence, box_size=g.box_size, yaw=g.yaw, radius=g.radius,
            length=g.length, axis=g.axis, metadata=deepcopy(g.metadata), scores=deepcopy(g.scores))))


def score_frame(objects, observations, clouds, *, previous_ids=None, **context):
    """All expected objects and all raw observations survive the 40 mm match."""
    matched, assigned = {}, set()
    for distance, oi, di in sorted((float(np.linalg.norm(obs.pose.position-obj.center)), oi, di)
            for oi, obj in enumerate(objects) for di, obs in enumerate(observations)):
        if distance < .04 and oi not in matched and di not in assigned:
            matched[oi] = di
            assigned.add(di)
    supports = np.zeros(len(objects), int)
    details, mixed = [], []
    for index, observation in enumerate(observations):
        cloud = clouds[index] if index < len(clouds) else None
        counts = np.zeros(len(objects), int)
        if cloud is not None and len(cloud):
            labels = surface_memberships(cloud, objects)
            counts = np.array([np.count_nonzero(labels == oi) for oi in range(len(objects))])
        fractions = counts/max(1, 0 if cloud is None else len(cloud))
        significant = (counts >= 8) & (fractions >= .10)
        supports += significant
        is_mixed = bool(np.count_nonzero(significant) > 1)
        detail = dict(**raw_observation(observation), matched_truth_index=next(
            (oi for oi, di in matched.items() if di == index), None),
            points=0 if cloud is None else len(cloud), truth_surface_counts=counts.tolist(),
            truth_surface_fractions=fractions.tolist(), mixed=is_mixed, **context)
        details.append(detail)
        if is_mixed:
            mixed.append(detail)
    rows, ids, switches = [], {}, 0
    for oi, obj in enumerate(objects):
        obs = observations[matched[oi]] if oi in matched else None
        row = dict(**geometry_row(obj, obs), **context)
        if obs is not None:
            ids[obj.name] = obs.object_id
            changed = bool(previous_ids and obj.name in previous_ids and previous_ids[obj.name] != obs.object_id)
            row['identity_switch'] = changed
            switches += changed
        rows.append(row)
    counts = dict(expected=len(objects), observed=len(observations), matched=len(matched),
        missing=len(objects)-len(matched), false_positives=len(observations)-len(assigned),
        false_splits=int(np.maximum(supports-1, 0).sum()),
        duplicate_ids=len(observations)-len({obs.object_id for obs in observations}), identity_switches=switches,
        mixed_accepted_clouds=len(mixed), known_mixed=sum(item['shape'] != 'unknown' for item in mixed),
        graspable_mixed=sum(item['graspable'] is True for item in mixed),
        checked_clouds=sum(cloud is not None and len(cloud) > 0 for cloud in clouds), **context)
    return rows, counts, ids, details


class PerceptionSession:
    """One persistent source per scene/method, with one consumption per push."""
    def __init__(self, provider_class, config, *, method, mode='fused'):
        self.provider_class, self.config, self.method, self.mode = provider_class, config, method, mode
        self.rig, self.camera, self.source = SnapshotRig(), None, None
        self.previous_ids, self.rows, self.samples, self.detections, self.observations, self.contracts = {}, [], [], [], [], []

    def consume(self, frames, objects, *, index, **context):
        selected = frames if self.mode == 'fused' else (frames[0 if self.mode.endswith('_a') else 1],)
        now = min(frame.acquired_at for frame in selected)
        if self.source is None:
            if self.mode.startswith('v1'):
                self.camera = SnapshotCamera(selected[0])
                self.source = build_perception_provider(self.camera, self.config)
            else:
                self.source = self.provider_class(self.rig, self.config.scene)
        if self.camera is not None:
            self.camera.push(selected[0])
        else:
            self.rig.push(selected)
        started = time.perf_counter()
        with patch('time.monotonic', return_value=now):
            batch = self.source.observe()
        elapsed = (time.perf_counter()-started)*1000
        accepted = list(batch.observations)
        stamp_ok = batch.timestamp == now and all(obs.timestamp == now for obs in accepted)
        diagnostics = deepcopy(self.source.diagnostics)
        with patch('time.monotonic', return_value=now+1e-6):
            repeated = self.source.observe()
        frozen_ok = repeated.timestamp is None and not repeated.observations
        metadata = dict(method=self.method, frame_index=index, scored=index >= 2, **context)
        clouds = [self.source.get_point_cloud(obs.object_id) for obs in accepted]
        rows, counts, ids, raw = score_frame(objects, accepted, clouds, previous_ids=self.previous_ids, **metadata)
        self.previous_ids.update(ids)
        self.rows.extend(rows)
        self.detections.append(counts)
        self.observations.extend(raw)
        self.samples.append(dict(**metadata, original_acquired_at=now, batch_timestamp=batch.timestamp,
            camera_ids=[frame.camera_id for frame in selected], tokens=[frame.token for frame in selected],
            timestamps_preserved=stamp_ok, frozen_silent=frozen_ok, observed=len(accepted),
            offline_processing_ms=elapsed, diagnostics=diagnostics))
        self.contracts.append(dict(**metadata, kind='original_stamp_and_unchanged_frame_silent',
            passed=bool(stamp_ok and frozen_ok), timestamps_preserved=stamp_ok, frozen_silent=frozen_ok))
        for obs in accepted:
            if obs.geometry is not None and obs.geometry.shape.value == 'sphere':
                self.contracts.append(dict(**metadata, kind='featureless_sphere_orientation_absent',
                    passed=obs.pose.orientation is None and obs.geometry.yaw is None, object_id=obs.object_id))
        if index < 2:
            self.contracts.append(dict(**metadata, kind='three_distinct_frames_before_confirmation', passed=not accepted))
        return accepted, clouds

    def retain_failure(self, objects, *, index, error, **context):
        metadata = dict(method=self.method, frame_index=index, scored=index >= 2, **context)
        rows, counts, _, _ = score_frame(objects, [], [], **metadata)
        for row in rows:
            row['runtime_error'] = error
        self.rows.extend(rows)
        self.detections.append(dict(**counts, runtime_error=error))
        self.samples.append(dict(**metadata, runtime_error=error, timestamps_preserved=False,
                                 frozen_silent=False, offline_processing_ms=None, diagnostics={}))
        self.contracts.append(dict(**metadata, kind='source_runtime_error', passed=False, error=error))

    def close(self):
        if self.source is not None:
            self.source.close()


def summarize(rows, samples, detections, contracts):
    selected = [row for row in rows if row.get('scored', True)]
    counts = [row for row in detections if row.get('scored', True)]
    correct = [row for row in selected if row['correct']]
    result = dict(expected=len(selected), detected=sum(row['detected'] for row in selected),
        correct=sum(row['correct'] for row in selected), unknown=sum(row['unknown'] for row in selected),
        missing=sum(not row['detected'] for row in selected),
        recall=None if not selected else sum(row['detected'] for row in selected)/len(selected),
        correct_classification_rate=None if not selected else len(correct)/len(selected),
        runtime_error_rows=sum(bool(row.get('runtime_error')) for row in selected),
        ambiguous_box_yaw=sum(row['truth'] == 'cube' and not row['yaw_precision_eligible'] for row in selected),
        observable_yaw_samples=sum(row['yaw_precision_eligible'] for row in correct),
        parameter_observable_known=sum(row.get('parameter_observable') is True for row in correct),
        **{field: sum(row.get(field, 0) for row in counts) for field in
           ('false_positives', 'false_splits', 'duplicate_ids', 'identity_switches',
            'mixed_accepted_clouds', 'known_mixed', 'graspable_mixed')},
        center_all_detected_mm=distribution([row['center_error_mm'] for row in selected if row['detected']]),
        center_correct_mm=distribution([row['center_error_mm'] for row in correct]),
        dimension_correct_mm=distribution([row['dimension_error_mm'] for row in correct]),
        cube_dimension_correct_mm=distribution([row['dimension_error_mm'] for row in correct if row['truth'] == 'cube']),
        cube_height_correct_mm=distribution([row['height_error_mm'] for row in correct if row['truth'] == 'cube']),
        sphere_radius_correct_mm=distribution([row['radius_error_mm'] for row in correct if row['truth'] == 'sphere']),
        yaw_observable_deg=distribution([row['yaw_error_deg'] for row in correct]),
        yaw_raw_major_axis_deg=distribution([row['yaw_raw_major_axis_error_deg'] for row in correct]),
        source_contracts_passed=bool(contracts) and all(row['passed'] for row in contracts),
        source_contract_failures=sum(not row['passed'] for row in contracts),
        offline_processing_ms=distribution([row.get('offline_processing_ms') for row in samples]),
        offline_stages_ms={field: distribution([row['diagnostics'].get(field) for row in samples
            if isinstance(row['diagnostics'], dict)]) for field in STAGES})
    visible = [row for row in selected if row.get('sufficiently_visible') is True]
    severe = [row for row in selected if row.get('sufficiently_visible') is False]
    result['visibility_strata'] = {name: dict(expected=len(values), correct=sum(row['correct'] for row in values),
        detected=sum(row['detected'] for row in values), unknown=sum(row['unknown'] for row in values),
        missing=sum(not row['detected'] for row in values)) for name, values in [('sufficiently_visible', visible), ('severe_or_sparse', severe)]}
    # Aspirations are separate from completion/source-contract status.
    targets = dict(correct_at_least_95pct=result['correct_classification_rate'] is not None and
        result['correct_classification_rate'] >= .95,
        center_p95_at_most_5mm=bool(result['center_correct_mm'] and result['center_correct_mm']['p95'] <= 5.),
        dimensions_p95_at_most_5mm=bool(result['dimension_correct_mm'] and result['dimension_correct_mm']['p95'] <= 5.),
        observable_yaw_p95_at_most_5deg=bool(result['yaw_observable_deg'] and result['yaw_observable_deg']['p95'] <= 5.))
    result['engineering_targets'] = targets
    result['by_class'] = {shape: dict(expected=len(values), detected=sum(row['detected'] for row in values),
        correct=sum(row['correct'] for row in values), unknown=sum(row['unknown'] for row in values),
        missing=sum(not row['detected'] for row in values),
        center_correct_mm=distribution([row['center_error_mm'] for row in values if row['correct']]),
        dimension_correct_mm=distribution([row['dimension_error_mm'] for row in values if row['correct']]),
        radius_correct_mm=distribution([row['radius_error_mm'] for row in values if row['correct']]))
        for shape in sorted({row['truth'] for row in selected})
        for values in [[row for row in selected if row['truth'] == shape]]}
    result['by_condition'] = {condition: basic_counts(values)
        for condition in sorted({row.get('condition', row.get('scene_key', 'unspecified')) for row in selected})
        for values in [[row for row in selected if row.get('condition', row.get('scene_key', 'unspecified')) == condition]]}
    result['one_view_strata'] = {name: basic_counts(values) for name, flag in
        [('strict_true_one_view', 'strict_true_one_view'), ('effective_one_view', 'effective_one_view')]
        for values in [[row for row in selected if row.get(flag) is True]]}
    return plain(result)


def basic_counts(rows):
    return dict(expected=len(rows), detected=sum(row['detected'] for row in rows),
        correct=sum(row['correct'] for row in rows), unknown=sum(row['unknown'] for row in rows),
        missing=sum(not row['detected'] for row in rows),
        center_correct_mm=distribution([row['center_error_mm'] for row in rows if row['correct']]),
        dimension_correct_mm=distribution([row['dimension_error_mm'] for row in rows if row['correct']]))


def temporal_recovery(rows, samples):
    """Saved capture-to-capture recovery; never new live delivery latency.

    A reappearance requires >=8 raw truth-associated samples in the camera
    union. Fully hidden means EXACTLY zero, independently of the predeclared
    stronger sufficient-visibility flag. Every unresolved event stays censored.
    """
    rows = [row for row in rows if row.get('temporal') and row.get('scored', True)]
    stamps = {(row['trial'], row['frame_index']): row.get('original_acquired_at') for row in samples}
    events = []
    for trial, name in sorted({(row['trial'], row['object']) for row in rows}):
        last_id, active = None, None
        sequence = sorted([row for row in rows if row['trial'] == trial and row['object'] == name],
                          key=lambda row: row['frame_index'])
        for row in sequence:
            count = sum(camera['raw_points'] for camera in row.get('visibility', {}).get('cameras', []))
            stamp = stamps.get((trial, row['frame_index']))
            # A new both-view hiding interval censors the earlier visible
            # failure episode. Its clock must not include a later period in
            # which the object cannot be observed at all.
            if (active is not None and count == 0 and
                    (active['onset_kind'] != 'fully_hidden' or active['first_revisible_frame'] is not None)):
                active.update(end_reason='interrupted_by_full_occlusion', interrupted_before_frame=row['frame_index'])
                active = None
            if active is None and (not row['correct'] or count == 0):
                active = dict(trial=trial, object=name, method=row.get('method'), layout=row.get('layout'),
                    onset_frame=row['frame_index'], onset_phase=row.get('phase'), prior_id=last_id,
                    onset_kind='fully_hidden' if count == 0 else 'low_density' if count < 8 else 'visible_classification_failure',
                    fully_hidden_frames=[], low_density_frames=[], visible_failure_frames=[],
                    first_revisible_frame=None, first_revisible_acquired_at=None,
                    first_sufficiently_visible_frame=None, recovered_frame=None,
                    recovered_acquired_at=None, recovered_id=None, end_reason='censored_end_of_capture')
                events.append(active)
            if active is not None:
                if count == 0:
                    active['fully_hidden_frames'].append(row['frame_index'])
                elif count < 8:
                    active['low_density_frames'].append(row['frame_index'])
                elif not row['correct']:
                    active['visible_failure_frames'].append(row['frame_index'])
                if count >= 8 and active['first_revisible_frame'] is None:
                    active.update(first_revisible_frame=row['frame_index'], first_revisible_acquired_at=stamp)
                if row.get('sufficiently_visible') and active['first_sufficiently_visible_frame'] is None:
                    active['first_sufficiently_visible_frame'] = row['frame_index']
                if count >= 8 and row['correct']:
                    active.update(recovered_frame=row['frame_index'], recovered_acquired_at=stamp,
                                  recovered_id=row['object_id'], end_reason='recovered')
                    active = None
            if row['correct']:
                last_id = row['object_id']
    for event in events:
        event['censored'] = event['recovered_frame'] is None
        event['identity_preserved'] = None if event['prior_id'] is None or event['recovered_id'] is None else (
            event['prior_id'] == event['recovered_id'])
        event['saved_capture_reacquisition_ms'] = None if (event['recovered_acquired_at'] is None or
            event['first_revisible_acquired_at'] is None) else 1000*(
                event['recovered_acquired_at']-event['first_revisible_acquired_at'])
    return events


def latest_scene_summary(rows, samples, detections, contracts):
    """One latest static frame per scene, temporal continuation separate.

    Repeated scored static frames quantify jitter; they do not create additional
    independent physical scenes or dominate camera-placement selection.
    """
    static = [row for row in detections if row.get('scored', True) and not row.get('temporal', False)]
    latest = {trial: max(row['frame_index'] for row in static if row['trial'] == trial)
              for trial in {row['trial'] for row in static}}
    def take(values):
        return [row for row in values if row.get('trial') in latest and row['frame_index'] == latest[row['trial']]]
    summary = summarize(take(rows), take(samples), take(detections), take(contracts))
    summary['physical_static_scenes'] = len(latest)
    temporal = [row for row in rows if row.get('scored', True) and row.get('temporal', False)]
    summary['temporal_rows_separate'] = dict(expected=len(temporal), correct=sum(row['correct'] for row in temporal),
        missing=sum(not row['detected'] for row in temporal), unknown=sum(row['unknown'] for row in temporal))
    return summary
