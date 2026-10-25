#!/usr/bin/env python3
"""Offline matched global-route analysis and measured V4.3 figures.

Native measurements, source-bound numeric fixture proofs and strong detour
acceptance remain separate. No simulator is imported or planner invoked. All
failed or missing declared attempts remain in the evidence inventory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from scripts import analyze_roundtrip_v4_2 as recorded


ARM = recorded.ARM
FINGERS = ('panda_finger_joint1', 'panda_finger_joint2')
MEANINGFUL_DEVIATION_RAD = .05
ENHANCED_MODEL_SHA256 = '6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d'
CONDITIONS = ('baseline_free', 'free', 'obstacle', 'near', 'goal_blocked')
EXECUTION_KEYS = ('tracking_limit_rad', 'tracking_grace_seconds', 'tracking_abort_rad',
    'terminal_velocity_limit_rad_s', 'terminal_settle_seconds', 'maximum_control_updates')


def route_array(value):
    q = np.asarray(value, float)
    if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
        raise ValueError('A finite nonempty seven-joint route is required')
    return q


def named_route(value, names):
    q = route_array(value)
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(ARM):
        raise ValueError('Route coordinates require exactly seven unique arm joint names')
    return q[:, [names.index(name) for name in ARM]]


def distances_to_polyline(points, line):
    """Exact point-to-polyline distances in seven-revolute-joint radians."""
    points, line = route_array(points), route_array(line)
    if len(line) == 1:
        return np.linalg.norm(points-line[0], axis=1)
    if len(points)*(len(line)-1) > 10000000:
        raise ValueError('Route comparison exceeds the bounded point/segment query budget')
    origins, vectors = line[:-1], np.diff(line, axis=0)
    length2 = np.einsum('ij,ij->i', vectors, vectors)
    output = np.full(len(points), np.inf)
    for first in range(0, len(points), 512):
        block = points[first:first+512]
        for begin in range(0, len(vectors), 128):
            end = begin+128
            delta = block[:, None, :]-origins[begin:end]
            fractions = np.divide(np.einsum('nij,ij->ni', delta, vectors[begin:end]), length2[begin:end],
                out=np.zeros((len(block), len(vectors[begin:end]))), where=length2[begin:end] > 0)
            nearest = origins[begin:end]+np.clip(fractions, 0., 1.)[:, :, None]*vectors[begin:end]
            output[first:first+len(block)] = np.minimum(output[first:first+len(block)],
                np.linalg.norm(block[:, None, :]-nearest, axis=2).min(axis=1))
    return output


def sample_polyline(value, *, step_rad=.01):
    q = route_array(value)
    if not np.isfinite(step_rad) or step_rad <= 0:
        raise ValueError('A positive finite sampling spacing is required')
    counts = np.maximum(1, np.ceil(np.linalg.norm(np.diff(q, axis=0), axis=1)/step_rad).astype(int))
    if counts.sum() > 200000:
        raise ValueError('Route comparison exceeds the declared sample budget')
    return np.concatenate([*[np.linspace(first, last, count+1)[:-1]
        for first, last, count in zip(q[:-1], q[1:], counts)], q[-1:]])


def route_geometry(value):
    q = route_array(value)
    direct = q[[0, -1]]
    return dict(waypoints=len(q), joint_space_length_rad=float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
        direct_connection_length_rad=float(np.linalg.norm(q[-1]-q[0])),
        maximum_deviation_from_direct_rad=float(distances_to_polyline(q, direct).max()),
        route_sha256=recorded.numeric_hash(q.tolist()), start_q=q[0].tolist(), goal_q=q[-1].tolist(),
        metric='Euclidean norm of all seven revolute joint coordinates in radians; no TCP-only inference')


def compare_routes(reference, candidate):
    a, b = route_array(reference), route_array(candidate)
    direct = a[[0, -1]]
    directed = float(distances_to_polyline(sample_polyline(b), a).max())
    reverse = float(distances_to_polyline(sample_polyline(a), b).max())
    deviation = float(distances_to_polyline(b, direct).max())
    return dict(reference=route_geometry(a), candidate=route_geometry(b),
        exact_waypoint_array_equal=bool(np.array_equal(a, b)),
        maximum_deviation_from_reference_direct_rad=deviation,
        sampled_symmetric_route_distance_rad=max(directed, reverse),
        route_comparison_sampling_step_rad=.01,
        same_start=bool(np.array_equal(a[0], b[0])), same_goal=bool(np.array_equal(a[-1], b[-1])),
        meaningful_threshold_rad=MEANINGFUL_DEVIATION_RAD,
        meaningful_multisegment_detour=bool(len(b) >= 3 and deviation > MEANINGFUL_DEVIATION_RAD),
        qualification='Direct-segment deviation is evaluated at every route vertex; symmetric polyline distance is sampled and is not a swept-volume certificate')


def _number(value):
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _same_numeric_vector(actual, expected, length):
    try:
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        return bool(a.shape == b.shape == (length,) and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=1e-7, rtol=0))
    except (TypeError, ValueError):
        return False


def _named_positions(state, names):
    positions = (state.get('feedback') or state.get('joint_feedback') or {}).get('positions')
    if positions is None or len(names) != len(set(names)) or not set((*ARM, *FINGERS)) <= set(names):
        return None
    q = np.asarray(positions, float)
    if q.shape != (len(names),) or not np.isfinite(q).all():
        return None
    return {name: float(q[names.index(name)]) for name in (*ARM, *FINGERS)}


def _native_finger_targets(state, names):
    response = (state.get('live_drive_feedback') or {}).get('robot_applied_action') or {}
    action = response.get('value') if response.get('available') else None
    if not action or action.get('joint_positions') is None:
        return None
    values = np.asarray(action['joint_positions'], float)
    indices = action.get('joint_indices')
    if values.ndim != 1 or not np.isfinite(values).all():
        return None
    if indices is None:
        if len(values) != len(names):
            return None
        mapping = dict(zip(names, values.tolist()))
    else:
        if len(indices) != len(values) or len(set(indices)) != len(indices):
            return None
        if any(not isinstance(index, int) or not 0 <= index < len(names) for index in indices):
            return None
        mapping = {names[index]: value for index, value in zip(indices, values.tolist())}
    return {name: mapping[name] for name in FINGERS} if set(FINGERS) <= set(mapping) else None


def _operation(case, label):
    return next((row for row in case.get('operations', []) if row['summary']['label'] == label), None)


def _operation_pass(operation):
    if operation is None:
        return False
    row = operation['summary']
    errors = [_number(row.get(key)) for key in ('final_position_error_m', 'final_orientation_error_rad')]
    return bool(row.get('returned_success') is True
        and (_number(row.get('maximum_actual_arm_displacement_rad')) or 0.) > .02
        and (_number(row.get('sampled_all_scene_minimum_m')) or 0.) > 0.
        and row.get('observed_physical_contact_steps') == 0
        and all(value is not None and value >= 0. for value in errors)
        and errors[0] <= .025 and errors[1] <= .10
        and _number(row.get('final_measured_arm_velocity_max_rad_s')) is not None
        and 0. <= row['final_measured_arm_velocity_max_rad_s'] <= .01
        and row.get('solver_position_iterations_restored') is True
        and (_number(row.get('observed_terminal_settling_sim_seconds')) or 0.) >= .25-1e-9)


def read_case(path, declaration=None):
    path = Path(path)
    declared = declaration or {}
    summary = dict(result=str(path), condition=declared.get('condition'), seed=declared.get('seed'),
        repeat=declared.get('repeat', path.parent.name), measurement_valid=False, operations=[])
    case = dict(summary=summary, operations=[], raw={}, trace={}, controls={})
    if not path.exists():
        summary.update(status='NO_RESULT_AFTER_LAUNCH' if path.with_name('launch.json').exists() else 'NOT_RUN',
            actual_trace_status='NOT_CAPTURED')
        return case
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError('Native result envelope must be an object')
        metrics = result.get('metrics') or {}
        summary.update(status=result.get('status', 'INVALID_RESULT'), condition=metrics.get('condition', summary['condition']),
            seed=metrics.get('seed', summary['seed']), repeat=metrics.get('repeat', summary['repeat']),
            error=result.get('error'), result_sha256=recorded.digest(path), contracts=result.get('contracts') or {},
            source_manifest_sha256=recorded.numeric_hash(result['hashes']) if result.get('hashes') else None,
            hashes=result.get('hashes') or {}, fixed_target={key: value for key, value in (metrics.get('fixed_target') or {}).items()
                if key in ('schema_version', 'joint_names', 'q', 'target', 'end_effector_frame', 'baseline_result_sha256')},
            fixture_sha256=metrics.get('fixture_document_sha256', metrics.get('frozen_fixture_sha256')),
            final_strategy=metrics.get('strategy_after_close'))
        trace_path = path.parent/metrics.get('trace_artifact', 'roundtrip_trace.json')
        if not trace_path.exists():
            summary['actual_trace_status'] = 'NOT_CAPTURED'
            return case
        trace = json.loads(trace_path.read_text())
        loaded = recorded.load_case(path)
        summary.update({key: value for key, value in loaded['summary'].items()
                        if key not in ('mode', 'model', 'seed', 'repeat', 'operations', 'contracts', 'error')})
        summary['operations'] = loaded['summary']['operations']
        case.update(operations=loaded['operations'], raw=result, trace=trace)
        for measured, original in zip(case['operations'], trace.get('operations', [])):
            measured['raw_operation'] = original
            if measured['route'] is not None:
                native_names = (original.get('plan_trace') or {}).get('joint_names') or []
                measured['route'] = named_route(measured['route'], native_names)
                measured['summary'].update(native_route_joint_names=native_names, route_joint_names=list(ARM),
                    route_joint_polyline_sha256=recorded.numeric_hash(measured['route'].tolist()))
                measured['summary']['route_geometry'] = route_geometry(measured['route'])
        outward = _operation(case, 'outward')
        operation = outward.get('raw_operation', {}) if outward else {}
        diag = operation.get('strategy_diagnostics') or {}
        names = trace.get('metadata', {}).get('robot_dof_names') or metrics.get('robot_dof_names') or []
        actual = _named_positions(operation.get('before') or metrics.get('initial_state') or {}, names)
        selected = diag.get('selected_ik_index')
        selected_candidate = next((row for row in diag.get('ik_attempts', []) if row.get('index') == selected), {})
        fixed = metrics.get('fixed_target') or {}
        fixed_q = fixed.get('q')
        fixed_names = fixed.get('joint_names', fixed.get('names'))
        if fixed_q is not None and fixed_names is not None:
            if len(fixed_names) != len(set(fixed_names)) or set(fixed_names) != set(ARM):
                raise ValueError('Fixed target must identify exactly the seven arm joints')
            fixed_q = np.asarray(fixed_q, float)[[fixed_names.index(name) for name in ARM]].tolist()
        planning_record = next((row for row in metrics.get('planning_records', []) if row.get('target') == operation.get('target')), {})
        planning_controls = (dict(limits=planning_record['limits'],
            joint_names=planning_record.get('joint_names'), position_tolerance_m=planning_record.get('position_tolerance_m'),
            orientation_tolerance_rad=planning_record.get('orientation_tolerance_rad')) if planning_record.get('limits') else None)
        planning_controls = planning_controls or metrics.get('planning_controls') or metrics.get('planning_limits')
        if planning_controls is None:
            planning_controls = (diag.get('research_fixed_joint_target') or {}).get('planning_controls')
        targets = _native_finger_targets(operation.get('before') or metrics.get('initial_state') or {}, names)
        selected_q = selected_candidate.get('ik_q')
        if selected_q is not None:
            selected_q = named_route([selected_q], diag.get('active_joint_names') or (operation.get('plan_trace') or {}).get('joint_names') or [])[0].tolist()
        case['controls'] = dict(start_q=[actual[name] for name in ARM] if actual else None,
            measured_fingers=[actual[name] for name in FINGERS] if actual else None,
            native_finger_targets=targets, full_pose=operation.get('target'), fixed_q=fixed_q,
            selected_ik_q=selected_q, model=diag.get('global_collision_model', metrics.get('collision_model')),
            robot_base_pose=(operation.get('before') or metrics.get('initial_state') or {}).get('robot_base_pose'),
            world_snapshot=(operation.get('plan_trace') or {}).get('snapshot'),
            model_sha256=(diag.get('sdk_configuration_sha256') or {}).get('robot_description_path'),
            max_iterations=diag.get('max_iterations'), rrt_step_size_rad=diag.get('rrt_step_size_rad'),
            planning_controls=planning_controls, execution_controls={key: diag.get(key) for key in EXECUTION_KEYS},
            selected_fk_errors={key: selected_candidate.get(key) for key in ('fk_position_error', 'fk_orientation_error')},
            planning_validation=dict(route_samples=selected_candidate.get('route_validation_samples'),
                trajectory=diag.get('trajectory_validation')),
            fixed_adapter=(diag.get('research_fixed_joint_target') or {}))
        summary['comparison_controls'] = case['controls']
        driver_path = path.with_name('driver_result.json')
        driver = json.loads(driver_path.read_text()) if driver_path.exists() else None
        if driver:
            summary['driver'] = {key: driver.get(key) for key in ('status', 'return_code', 'driver_error', 'fatal_gpu_messages')}
            summary['driver_sha256'] = recorded.digest(driver_path)
        valid_driver = driver is None or (driver.get('status') == 'PASS' and driver.get('return_code') == 0
            and not driver.get('driver_error') and not driver.get('fatal_gpu_messages'))
        contracts = summary['contracts']
        paths = trace.get('metadata', {}).get('body_paths', [])
        expected_frames = {*(f'panda_link{index}' for index in range(8)), 'panda_hand', 'panda_leftfinger', 'panda_rightfinger'}
        summary['all11_reference'] = bool(len(paths) == 11 and {path.rsplit('/', 1)[-1] for path in paths} == expected_frames)
        summary['measurement_valid'] = bool(summary['status'] == 'PASS' and contracts and all(value is True for value in contracts.values())
            and not summary.get('error') and valid_driver
            and summary['all11_reference']
            and metrics.get('meters_per_unit') == 1.
            and summary.get('actual_body_poses_match_cooked_archive') is True)
        summary['outward_physical_acceptance'] = bool(summary['measurement_valid'] and _operation_pass(outward))
        summary['home_physical_acceptance'] = bool(summary['measurement_valid'] and _operation_pass(_operation(case, 'home_return')))
        summary['roundtrip_physical_acceptance'] = bool(summary['outward_physical_acceptance'] and summary['home_physical_acceptance'])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        summary.update(status='INVALID_CAPTURE', error=f'{type(error).__name__}: {error}', measurement_valid=False)
    return case


def _present_equal(a, b):
    return a is not None and b is not None and a == b


def match_controls(free, obstacle, *, fixture_path=None):
    a, b = free.get('controls') or {}, obstacle.get('controls') or {}
    sa, sb = free['summary'], obstacle['summary']
    checks = {key: _present_equal(a.get(key), b.get(key)) for key in (
        'start_q', 'measured_fingers', 'native_finger_targets', 'fixed_q', 'selected_ik_q', 'full_pose',
        'model_sha256', 'max_iterations', 'rrt_step_size_rad', 'planning_controls', 'execution_controls', 'robot_base_pose')}
    checks.update(both_measurements_valid=sa.get('measurement_valid') is True and sb.get('measurement_valid') is True,
        same_seed=_present_equal(sa.get('seed'), sb.get('seed')),
        same_source_cohort=_present_equal(sa.get('source_manifest_sha256'), sb.get('source_manifest_sha256')),
        enhanced_model=a.get('model') == b.get('model') == 'enhanced',
        frozen_102_sphere_description=a.get('model_sha256') == b.get('model_sha256') == ENHANCED_MODEL_SHA256,
        frozen_rrt_budget=a.get('max_iterations') == b.get('max_iterations') == 10000,
        frozen_rrt_step=a.get('rrt_step_size_rad') == b.get('rrt_step_size_rad') == .05,
        execution_feedback_available=all(controls.get('execution_controls') and all(value is not None
            for value in controls['execution_controls'].values()) for controls in (a, b)),
        unchanged_fixed_goal=a.get('fixed_q') is not None and a.get('fixed_q') == b.get('fixed_q')
            and a.get('selected_ik_q') == a.get('fixed_q')
            and b.get('selected_ik_q') == b.get('fixed_q'),
        gripper_range_supported=all(values is not None and len(values) == 2 and all(.001 <= value <= .005 for value in values)
            for values in (a.get('measured_fingers'), b.get('measured_fingers'))),
        native_finger_targets_supported=all(targets is not None and len(targets) == 2
            and all(.001 <= value <= .005 for value in targets.values())
            for targets in (a.get('native_finger_targets'), b.get('native_finger_targets'))),
        full_pose_fk_validated=all(all(_number(errors.get(key)) is not None and errors[key] <= limit
            for key, limit in (('fk_position_error', .025), ('fk_orientation_error', .10)))
            for errors in (a.get('selected_fk_errors') or {}, b.get('selected_fk_errors') or {})))
    for side, controls in (('free', a), ('obstacle', b)):
        route = _operation(free if side == 'free' else obstacle, 'outward')
        checks[side+'_route_uses_measured_start_and_fixed_goal'] = bool(route is not None and route['route'] is not None
            and np.array_equal(route['route'][0], controls.get('start_q'))
            and np.array_equal(route['route'][-1], controls.get('fixed_q')))
        checks[side+'_production_validation_preserved'] = controls.get('fixed_adapter', {}).get('production_validation_preserved') is True
        validation = controls.get('planning_validation') or {}
        checks[side+'_route_and_trajectory_model_validated'] = bool((validation.get('route_samples') or 0) > 0
            and (validation.get('trajectory') or {}).get('collision_validation') == 'PASS_SAMPLED_SDK_MODEL')
    def normalized(snapshot):
        return {key: value for key, value in snapshot.items() if key not in ('timestamp', 'obstacles')}
    wa, wb = a.get('world_snapshot') or {}, b.get('world_snapshot') or {}
    added = [row for row in wb.get('obstacles', []) if row.get('path') == fixture_path]
    original = [row for row in wb.get('obstacles', []) if row.get('path') != fixture_path]
    checks['only_frozen_obstacle_added_to_world'] = bool(fixture_path and wa and wb and len(added) == 1
        and all(row.get('path') != fixture_path for row in wa.get('obstacles', []))
        and normalized(wa) == normalized(wb) and wa.get('obstacles') == original)
    return checks


def design_checks(frozen, cases):
    proof = frozen.get('obstruction_proof') or {}
    witness = frozen.get('witness') or {}
    baseline = frozen.get('baseline') or {}
    references = baseline.get('source_sha256') or {}
    actual_baselines = [case for case in cases if case['summary'].get('condition') == 'baseline_free']
    baseline_hashes = {case['summary'].get('result_sha256') for case in actual_baselines}
    bound = references.get(baseline.get('result'))
    if isinstance(bound, dict):
        bound = bound.get('sha256')
    return dict(original_global_route_positive_volume_obstruction=proof.get('status') == 'VERIFIED_POSITIVE_INTERSECTION_BALL'
            and proof.get('reference') == 'independentFKplannedtrajectory'
            and (_number(proof.get('positive_ball_radius_lower_bound_m')) or 0.) > 0.,
        baseline_source_bound=bool(bound) and bound in baseline_hashes,
        independent_witness_declared=witness.get('independent_of_candidate_rrt') is True,
        independent_witness_external_clearance_positive=(_number(witness.get('numericcontinuousexternal_lower_bound_m')) or 0.) > 0.,
        independent_witness_self_clearance_positive=(_number(witness.get('self_lower_bound_m')) or 0.) > 0.)


def paired_results(cases, proof_checks, frozen=None, fixture_sha256=None):
    frozen = frozen or {}
    design_baseline = frozen.get('baseline') or {}
    reference = next((case for case in cases if case['summary'].get('condition') == 'baseline_free'
        and design_baseline.get('result') and Path(case['summary']['result']).resolve() == Path(design_baseline['result']).resolve()), None)
    grouped = defaultdict(lambda: defaultdict(list))
    for case in cases:
        summary = case['summary']
        if summary.get('condition') in ('free', 'obstacle'):
            grouped[(summary.get('seed'), summary.get('repeat'))][summary['condition']].append(case)
    pairs = []
    for (seed, repeat), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(seed=seed, repeat=repeat, strong_detour_accepted=False)
        if len(group['free']) != 1 or len(group['obstacle']) != 1:
            row.update(status='MISSING_OR_AMBIGUOUS_PAIR', free_attempts=len(group['free']), obstacle_attempts=len(group['obstacle']))
            pairs.append(row)
            continue
        free, obstacle = group['free'][0], group['obstacle'][0]
        checks = match_controls(free, obstacle, fixture_path=(frozen.get('fixture') or {}).get('prim_path'))
        a, b = _operation(free, 'outward'), _operation(obstacle, 'outward')
        geometry, geometry_error = None, None
        if a is not None and b is not None and a['route'] is not None and b['route'] is not None:
            try:
                geometry = compare_routes(a['route'], b['route'])
            except ValueError as error:
                geometry_error = str(error)
        checks.update(independent_original_route_obstructed=proof_checks['original_global_route_positive_volume_obstruction'],
            source_bound_independent_feasibility=all(proof_checks.values()),
            route_changed=bool(geometry and not geometry['exact_waypoint_array_equal']),
            meaningful_multisegment_detour=bool(geometry and geometry['meaningful_multisegment_detour']),
            free_outward_physical_acceptance=free['summary'].get('outward_physical_acceptance') is True,
            obstacle_outward_physical_acceptance=obstacle['summary'].get('outward_physical_acceptance') is True,
            obstacle_home_physical_acceptance=obstacle['summary'].get('home_physical_acceptance') is True)
        checks['same_design_start_and_goal'] = all(np.array_equal(case.get('controls', {}).get('start_q'), design_baseline.get('start_q'))
            and np.array_equal(case.get('controls', {}).get('fixed_q'), design_baseline.get('goal_q'))
            and design_baseline.get('start_q') is not None and design_baseline.get('goal_q') is not None
            for case in (free, obstacle))
        design_pose = design_baseline.get('target_pose') or {}
        checks['same_design_full_pose'] = all(key in design_pose for key in ('position', 'orientation')) and all(all((case.get('controls', {}).get('full_pose') or {}).get(key) == design_pose.get(key)
            for key in ('position', 'orientation')) for case in (free, obstacle))
        checks['same_design_finger_state_and_targets'] = bool(reference) and all(
            _present_equal(case.get('controls', {}).get(key), reference.get('controls', {}).get(key))
            for case in (free, obstacle) for key in ('measured_fingers', 'native_finger_targets'))
        reference_outward = _operation(reference, 'outward') if reference else None
        checks['free_restores_original_global_route'] = bool(reference_outward is not None and a is not None
            and reference_outward['route'] is not None and a['route'] is not None
            and np.array_equal(reference_outward['route'], a['route']))
        checks['same_frozen_fixture_document'] = bool(fixture_sha256) and all(case['summary'].get('fixture_sha256') == fixture_sha256
            for case in (free, obstacle))
        checks['obstacle_both_legs_positive_physical_box_clearance'] = all((_number((op or {}).get('summary', {}).get('sampled_fixture_minimum_m')) or 0.) > 0.
            for op in (b, _operation(obstacle, 'home_return')))
        geometry_rows = [row for row in (obstacle.get('controls', {}).get('world_snapshot') or {}).get('obstacles', [])
                         if row.get('path') == (frozen.get('fixture') or {}).get('prim_path')]
        fixture = frozen.get('fixture') or {}
        checks['frozen_obstacle_geometry_matches'] = bool(len(geometry_rows) == 1 and fixture
            and geometry_rows[0].get('kind') == 'cuboid'
            and all(_same_numeric_vector(geometry_rows[0].get(actual), fixture.get(expected), length)
                for actual, expected, length in (('position', 'position_m', 3), ('dimensions', 'dimensions_m', 3), ('orientation', 'orientation_wxyz', 4))))
        row.update(status='PASS_STRONG_DETOUR' if all(checks.values()) else 'INCOMPLETE_OR_FAILED_ACCEPTANCE',
            free_result=free['summary']['result'], obstacle_result=obstacle['summary']['result'], controls_and_acceptance=checks,
            route_comparison=geometry, strong_detour_accepted=all(checks.values()),
            route_comparison_error=geometry_error,
            free_outward=a['summary'] if a else None, obstacle_outward=b['summary'] if b else None,
            obstacle_home=(_operation(obstacle, 'home_return') or {}).get('summary'))
        pairs.append(row)
    return pairs


def negative_controls(cases, frozen=None):
    frozen = frozen or {}
    proofs = frozen.get('control_proofs') or {}
    near = [case for case in cases if case['summary'].get('condition') == 'near']
    blocked = [case for case in cases if case['summary'].get('condition') == 'goal_blocked']
    output = dict(near=[], goal_blocked=[])
    for case in near:
        physical = proofs.get('near') or {}
        checks = dict(native_request_completed=case['summary'].get('outward_physical_acceptance') is True,
            independently_irrelevant_obstacle=physical.get('passed') is True
                and (_number(physical.get('original_planned_and_actual_global_route_aabb_clearance_lower_bound_m')) or 0.) > 0.)
        output['near'].append(dict(result=case['summary']['result'], checks=checks, passed=all(checks.values())))
    for case in blocked:
        outward = _operation(case, 'outward')
        op = outward.get('raw_operation', {}) if outward else {}
        compact = next((row for row in case.get('raw', {}).get('metrics', {}).get('operations', []) if row.get('label') == 'outward'), {})
        diag = op.get('strategy_diagnostics') or {}
        action_count = compact.get('action_count')
        checks = dict(valid_measurement=case['summary'].get('measurement_valid') is True,
            request_rejected=op.get('returned_success') is False,
            global_strategy=diag.get('motion_strategy') == 'global',
            no_arm_commands=action_count == 0,
            no_home_after_rejection=_operation(case, 'home_return') is None)
        physical = proofs.get('goal_blocked') or {}
        intersection = physical.get('goal_positive_volume_intersection') or {}
        checks['independent_goal_volume_obstruction'] = bool(physical.get('passed') is True
            and intersection.get('status') == 'VERIFIED_POSITIVE_INTERSECTION_BALL'
            and (_number(intersection.get('positive_ball_radius_lower_bound_m')) or 0.) > 0.)
        checks['independent_start_home_clear'] = (_number((physical.get('endpoints') or {}).get('actual_start_home', {}).get('minimum_m')) or 0.) > 0.
        output['goal_blocked'].append(dict(result=case['summary']['result'], checks=checks, passed=all(checks.values()),
            failure_code=diag.get('failure_code'), action_count=action_count,
            qualification='No obstacle-unaware command fallback observed in this rejected request; not a proof over arbitrary future failures'))
    output['near_passed'] = bool(output['near']) and all(row['passed'] for row in output['near'])
    output['goal_blocked_passed'] = bool(output['goal_blocked']) and all(row['passed'] for row in output['goal_blocked'])
    return output


def summarize(cases, frozen, fixture_sha256=None):
    proof = design_checks(frozen, cases)
    pairs = paired_results(cases, proof, frozen, fixture_sha256)
    controls = negative_controls(cases, frozen)
    accepted = [row for row in pairs if row['strong_detour_accepted']]
    seeds = {row['seed'] for row in accepted}
    primary = sum(row['seed'] == 123456 for row in accepted)
    robustness = {str(seed): any(row['seed'] == seed and row['strong_detour_accepted'] for row in pairs)
                  for seed in (123456, 234567, 345678)}
    all_required = bool(pairs) and len(accepted) == len(pairs) and seeds >= {123456, 234567, 345678} and primary >= 3
    return dict(schema_version=1, status='PASS_DECLARED_STATIC_DETOUR' if all_required and controls['near_passed'] and controls['goal_blocked_passed'] else 'PARTIAL_OR_FAILED_STATIC_DETOUR',
        producer_sha256=recorded.digest(__file__), helper_source_sha256=recorded.digest(recorded.__file__),
        meaningful_deviation_threshold_rad=MEANINGFUL_DEVIATION_RAD, design_checks=proof,
        cases=[case['summary'] for case in cases], matched_pairs=pairs, negative_controls=controls,
        detour_demonstrated=bool(accepted), accepted_pairs=len(accepted), retained_pair_attempts=len(pairs),
        seeds=robustness, accepted_primary_fresh_pairs=primary, all_declared_pair_attempts_passed=all_required,
        qualifications=['Configuration-space geometry is measured in all seven named revolute coordinates.',
            'Actual cooked-solid clearance and self geometry remain sampled; no continuous executed swept-volume certificate is inferred.',
            'Independent witness feasibility does not guarantee bounded RRT discovery or execution success.',
            'Strict zero reports and observed physical-contact-free completion remain separate.',
            'Failures, missing captures and unpaired attempts remain visible rather than becoming PASS.'])


def box_geometry(frozen):
    box = frozen['fixture']
    center = np.asarray(box['position_m'], float)
    half = np.asarray(box['dimensions_m'], float)/2
    from robot_skill_stack.common.rotations import quat_matrix
    rotation = quat_matrix(box.get('orientation_wxyz', (1., 0., 0., 0.)))
    corners = np.asarray(list(itertools.product((-1., 1.), repeat=3)))*half
    return corners@rotation.T+center


def draw_box(axis, frozen, *, color='#555555', alpha=.35):
    vertices = box_geometry(frozen)
    local = list(itertools.product((-1, 1), repeat=3))
    for i, a in enumerate(local):
        for j, b in enumerate(local[i+1:], i+1):
            if sum(x != y for x, y in zip(a, b)) == 1:
                axis.plot(*vertices[[i, j]].T, color=color, alpha=alpha)


def _save(fig, path, plt):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def render_physical_obstruction(cases, frozen, directory, plt):
    """Plot the actual cooked solids placed at source-bound planned q states."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scripts.audit_global_collision_v4_1 import pose_matrix_xyzw
    from scripts.audit_rejected_ik_v4_2 import IndependentUrdfFK, URDF

    reference = next((case for case in cases if case['summary'].get('condition') == 'baseline_free'
        and _operation(case, 'outward') is not None), None)
    if reference is None:
        return None
    operation = _operation(reference, 'outward')
    plan = operation['raw_operation'].get('plan_trace') or {}
    commands = plan.get('trajectory') or []
    proof = frozen.get('obstruction_proof') or {}
    planned_time = proof.get('planned_time')
    if not commands or planned_time is None:
        return None
    mid = min(commands, key=lambda row: abs(row['time']-planned_time))
    if abs(mid['time']-planned_time) > 1e-8:
        raise ValueError('Frozen obstruction time must identify an actual retained planned joint sample')
    result = reference['raw']
    expected_urdf = (operation['raw_operation'].get('strategy_diagnostics', {}).get('sdk_configuration_sha256') or {}).get('urdf_path')
    if expected_urdf is None or recorded.digest(URDF) != expected_urdf:
        raise ValueError('Physical scene FK source does not match the native baseline URDF')
    trace = reference['trace']
    body_paths = trace['metadata']['body_paths']
    first = trace['samples'][operation['raw_operation']['first_sample']]
    base = pose_matrix_xyzw(first['poses'][body_paths.index('/Franka/panda_link0')])
    finger_values = reference['controls']['measured_fingers']
    fk = IndependentUrdfFK(URDF)
    selected = [commands[0], mid, commands[-1]]
    records = result['metrics']['geometry_reference']['cooking']
    parts = []
    with np.load(Path(reference['summary']['result']).parent/'geometry_reference.npz', allow_pickle=False) as archive:
        index = 0
        for record in records:
            for _ in range(record['cooked_hulls']):
                body_from_mesh = archive[f'body_from_mesh_{index}']
                vertices = archive[f'vertices_{index}']@body_from_mesh[:3, :3].T+body_from_mesh[:3, 3]
                parts.append((record['body'].rsplit('/', 1)[-1], vertices, archive[f'triangles_{index}'].copy()))
                index += 1
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    vertices_world = []
    body = str(proof.get('robot_body', '')).rsplit('/', 1)[-1]
    for state_index, command in enumerate(selected):
        command_q = named_route([command['q']], plan.get('joint_names') or [])[0]
        poses = fk.poses((*ARM, *FINGERS), [*command_q, *finger_values])
        for frame, vertices, triangles in parts:
            pose = base@poses[frame]
            world = vertices@pose[:3, :3].T+pose[:3, 3]
            vertices_world.append(world)
            color = '#b34230' if state_index == 1 and frame == body else ('#8d9aa2', '#3175a0', '#c79748')[state_index]
            for column, axis in enumerate(axes):
                if column == 1 and state_index != 1:
                    continue
                axis.add_collection3d(Poly3DCollection(world[triangles], facecolor=color,
                    edgecolor='none', alpha=.55 if state_index == 1 else .10))
    for axis in axes:
        draw_box(axis, frozen, color='#222222', alpha=.95)
        axis.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
        axis.view_init(elev=24, azim=-58)
    cloud = np.concatenate([*vertices_world, box_geometry(frozen)])
    low, high = cloud.min(0), cloud.max(0)
    for dimension, setter in enumerate((axes[0].set_xlim, axes[0].set_ylim, axes[0].set_zlim)):
        setter(low[dimension]-.025, high[dimension]+.025)
    center = np.asarray(frozen['fixture']['position_m'])
    half = np.asarray(frozen['fixture']['dimensions_m'])/2
    radius = max(.10, float(half.max())+.07)
    axes[1].set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
    axes[0].set_title('Original planned start, obstruction and goal configurations')
    axes[1].set_title(f'Original planned q at t={planned_time:.4f} s; {body}')
    fig.suptitle('Captured PhysX-cooked solids placed by independent, source-checked URDF FK')
    fig.text(.02, .01, 'Frozen box is superposed on the no-obstacle reference. Red body has an independent positive-volume intersection witness.', fontsize=8)
    return _save(fig, Path(directory)/'physical_obstruction_scene.png', plt)


def render_figures(cases, frozen, directory):
    """Measured actual curves and native waypoints; no illustrative replacement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    candidates = [case for case in cases if case['summary'].get('condition') == 'obstacle'
        and case['summary'].get('seed') == 123456 and _operation(case, 'outward') is not None]
    primary_b = next((case for case in candidates if case['summary'].get('roundtrip_physical_acceptance') is True),
        candidates[0] if candidates else None)
    primary_a = next((case for case in cases if case['summary'].get('condition') == 'free'
        and primary_b is not None and case['summary'].get('seed') == primary_b['summary'].get('seed')
        and case['summary'].get('repeat') == primary_b['summary'].get('repeat') and _operation(case, 'outward') is not None), None)
    if primary_a is None or primary_b is None:
        return []
    a, b, home = _operation(primary_a, 'outward'), _operation(primary_b, 'outward'), _operation(primary_b, 'home_return')
    operations = [(a, 'No obstacle outward'), (b, 'Obstacle outward')]+([(home, 'Obstacle Home return')] if home else [])
    paths = []
    scene = render_physical_obstruction(cases, frozen, directory, plt)
    if scene:
        paths.append(scene)

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    axis = fig.add_subplot(111, projection='3d')
    for op, label in operations:
        axis.plot(*op['tcp'].T, label=label, lw=1.8)
    draw_box(axis, frozen)
    axis.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)', title='Actual TCP trajectories in matched conditions')
    axis.legend(fontsize=8)
    fig.text(.02, .01, 'Box is counterfactual for the no-obstacle capture; physical-body obstruction is proved separately.', fontsize=8)
    paths.append(_save(fig, directory/'actual_tcp_conditions.png', plt))

    fig, axes = plt.subplots(7, len(operations), figsize=(5.2*len(operations), 11), sharex='col', squeeze=False, constrained_layout=True)
    for column, (op, label) in enumerate(operations):
        for joint in range(7):
            axis = axes[joint, column]
            axis.plot(op['times'], op['q'][:, joint], label='Actual joint', color='#174966')
            if len(op['command_q']):
                axis.step(op['command_times'], op['command_q'][:, joint], where='post', label='Planner command', color='#d65a31', ls='--', lw=1)
            if len(op['applied_q']):
                axis.plot(op['action_times'], op['applied_q'][:, joint], '.', ms=1.2, label='Native target', color='#6e9f62')
            axis.set_ylabel(f'q{joint+1} (rad)')
            axis.grid(alpha=.2)
            if joint == 0:
                axis.set_title(label)
                axis.legend(fontsize=7)
        axes[-1, column].set_xlabel('Operation simulation time (s)')
    paths.append(_save(fig, directory/'actual_joint_targets.png', plt))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    all_routes = [op['route'] for op, _ in operations if op['route'] is not None]
    if all_routes:
        cloud = np.concatenate(all_routes)
        center = cloud.mean(0)
        _, _, directions = np.linalg.svd(cloud-center, full_matrices=True)
        for op, label in operations:
            if op['route'] is None:
                continue
            route = op['route']
            projected = (route-center)@directions[:2].T
            axes[0].plot(projected[:, 0], projected[:, 1], 'o-', label=label, ms=3)
            distance = distances_to_polyline(route, route[[0, -1]])
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
            axes[1].plot(arc, distance, 'o-', label=label, ms=3)
        axes[0].set(xlabel='Joint-space principal coordinate 1 (rad)', ylabel='Joint-space principal coordinate 2 (rad)', title='Recorded native RRT waypoints (7D projection)')
        axes[1].set(xlabel='Joint-space polyline arc length (rad)', ylabel='Distance from direct joint segment (rad)', title='Deviation measured in all seven joints')
        axes[1].axhline(MEANINGFUL_DEVIATION_RAD, color='#777777', ls=':', label='Frozen 0.05 rad criterion')
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    paths.append(_save(fig, directory/'global_joint_route_comparison.png', plt))

    fig, axes = plt.subplots(7, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for op, label in operations:
        if op['route'] is None:
            continue
        route = op['route']
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        for joint, axis in enumerate(axes):
            axis.plot(arc, route[:, joint], 'o-', label=label, ms=2.5)
            axis.set_ylabel(f'q{joint+1} (rad)')
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Recorded joint-space route arc length (rad)')
    fig.suptitle('Joint-space waypoint coordinates, without artificial intermediate waypoints')
    paths.append(_save(fig, directory/'joint_waypoint_coordinates.png', plt))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for op, label in operations:
        if op['curve']:
            axes[0].plot([row['simulation_time']-op['origin'] for row in op['curve']], [row['minimum_m']*1000 for row in op['curve']], label=label)
        axes[1].plot(op['times'], op['position_error']*1000, label=label)
    axes[0].axhline(0., color='#9a3020', ls=':')
    axes[0].set(ylabel='Sampled all-body box clearance (mm)', title='Recorded cooked-solid clearance; counterfactual box in no-obstacle case')
    axes[1].axhline(25., color='#777777', ls=':')
    axes[1].set(xlabel='Operation simulation time (s)', ylabel='Actual TCP position error (mm)')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    paths.append(_save(fig, directory/'physical_clearance_outward_home.png', plt))

    rows = [(case, op) for case in cases for op in case.get('operations', [])
            if case['summary'].get('condition') in ('free', 'obstacle')]
    if rows:
        fig, axis = plt.subplots(figsize=(max(10., len(rows)*.7), 5), constrained_layout=True)
        x = np.arange(len(rows))
        for index, (key, label) in enumerate((('rrt_seconds', 'RRT search'), ('planning_and_validation_wall_seconds', 'Total planning and validation'))):
            values = [float('nan') if _number(op['summary']['planning_seconds'].get(key)) is None
                else op['summary']['planning_seconds'][key]*1000 for _, op in rows]
            axis.bar(x+(index-.5)*.32, values, .32, label=label)
        axis.set_xticks(x, [f"{case['summary']['condition']} {case['summary']['seed']} {case['summary']['repeat']} {op['summary']['label']}" for case, op in rows], rotation=40, ha='right')
        axis.set(ylabel='Measured wall time (ms)', title='Recorded planning time; failed attempts remain in the inventory')
        axis.legend(fontsize=8)
        paths.append(_save(fig, directory/'planning_time_comparison.png', plt))
    return paths


def declarations(paths):
    result = {}
    for path in paths:
        document = json.loads(Path(path).read_text())
        for row in document if isinstance(document, list) else document.get('jobs', []):
            arguments = row['arguments']
            def argument(flag):
                return arguments[arguments.index(flag)+1] if flag in arguments else None
            result[row['result']] = dict(condition=argument('--condition'), seed=int(argument('--seed')) if argument('--seed') else None,
                repeat=Path(row['artifacts_dir']).name, declared_label=row['label'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', action='append', default=[], type=Path)
    parser.add_argument('--jobs', action='append', default=[], type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--figures-dir', type=Path)
    args = parser.parse_args(argv)
    declared = declarations(args.jobs)
    paths = list(dict.fromkeys([*declared, *map(str, args.result)]))
    if not paths:
        parser.error('At least one result or finite job declaration is required')
    if args.output.exists() or args.figures_dir is not None and args.figures_dir.exists():
        parser.error('Refusing to overwrite an existing V4.3 reduction')
    frozen = json.loads(args.fixture.read_text())
    cases = [read_case(path, declared.get(path)) for path in paths]
    summary = summarize(cases, frozen, recorded.digest(args.fixture))
    summary.update(fixture=dict(path=str(args.fixture), sha256=recorded.digest(args.fixture)), figures=[])
    if args.figures_dir is not None:
        summary['figures'] = render_figures(cases, frozen, args.figures_dir)
    summary['figure_sha256'] = {path: recorded.digest(path) for path in summary['figures']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'], cases=len(cases), pairs=len(summary['matched_pairs']), accepted_pairs=summary['accepted_pairs'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
