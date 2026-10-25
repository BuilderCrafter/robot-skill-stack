#!/usr/bin/env python3
"""Reduce explicitly supplied V4.1 evidence; no simulator or directory scans.

Measurement validity, motion acceptance and observed post-hold stability stay
separate. Raw poses, trajectories and contact arrays are never copied out.
Read raw count aggregates only after native timing jobs have finished.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.summarize_global_motion_v4 import digest, fingerprint, load_attempt, number


OP_FIELDS = ('id label kind runtime_generation cancelled returned_success strict_success '
    'measured_full_pose_within_tolerances position_error_m orientation_error_rad '
    'maximum_actual_joint_displacement_rad wall_seconds sample_count action_count '
    'final_applied_arm_velocity_zero observed_physical_contact_steps ownership_released '
    'overlap_rejected active_before_cancel first_command_observed').split()
DIAG_FIELDS = ('failure_code state trajectory_duration_seconds execution_wall_seconds execution_sim_seconds '
    'planning_and_validation_wall_seconds rrt_seconds commanded_updates final_hold tracking_error_rad '
    'position_error_m orientation_error_rad measured_arm_velocity_rad_s terminal_velocity_limit_rad_s '
    'terminal_settle_seconds tracking_limit_rad tracking_grace_seconds tracking_abort_rad '
    'position_tolerance_m orientation_tolerance_rad execution_solver joint_tracking_error_rad control_interval_ms').split()
HOLD_FIELDS = ('available readings_available requested_seconds observed_simulation_seconds duration_sim_seconds '
    'additional_arm_commands maximum_applied_target_gap_rad maximum_actual_target_gap_rad '
    'maximum_continuous_gap_over_015_rad_seconds longest_sampled_above_sustained_limit_seconds '
    'maximum_position_error_m final_position_error_m maximum_orientation_error_rad final_orientation_error_rad '
    'all_requested_arm_velocity_targets_zero all_applied_arm_velocity_targets_zero '
    'all_native_actor_counts_expected all_native_actor_counts_nominal expected_native_actor_position_count '
    'kinematic_stability_passed validation_passed tracking_within_original_bounds no_additional_arm_commands '
    'ownership_released_throughout final_actual_target_gap_max_rad final_measured_arm_velocity_max_rad_s '
    'maximum_measured_arm_velocity_rad_s').split()
STATIC_FIELDS = ('scenario strategy seed mode cohort measurement_status measurement_valid strict_success '
    'movable_arm_strict_success failure_code classification final_pose_errors motion_wall_seconds '
    'reported_goal_reached max_iterations geometric_joint_path_length rrt_waypoints '
    'global_trajectory_validation_passed trajectory_validation joint_path_deviation final_hold '
    'no_requested_movement_joint_delta_rad minimum_reported_separation_m physical_contact_classification '
    'clearance_by_link_m clearance_by_group_m clearance_by_obstacle_m '
    'planning_wall_seconds rrt_seconds trajectory_generation_seconds trajectory_duration_seconds '
    'execution_wall_seconds execution_sim_seconds joint_tracking_error_rad control_interval_ms kit_interval_ms '
    'controller_wait_ms controller_waits_over_100ms commanded_updates minimum_all_link_clearance_m '
    'minimum_movable_arm_clearance_m reported_contact_points observed_physical_contact_steps '
    'maximum_observed_penetration_m maximum_observed_normal_pair_impulse_ns source_unchanged').split()


def scalar_tree(value):
    if isinstance(value, dict):
        return {key: scalar_tree(item) for key, item in value.items() if not isinstance(item, (list, tuple))}
    return value if value is None or isinstance(value, (bool, int, float, str)) else None


def pick(value, fields):
    return {key: scalar_tree(value[key]) for key in fields if key in value and not isinstance(value[key], (list, tuple))}


def bind(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path) if path.is_file() else None)


def maximum_absolute(values):
    valid = [number(value) for value in values or []]
    return max(map(abs, valid)) if valid and all(value is not None for value in valid) else None


def compact_operation(op):
    row = pick(op, OP_FIELDS)
    row['diagnostics'] = pick(op.get('strategy_diagnostics') or {}, DIAG_FIELDS)
    code = row['diagnostics'].get('failure_code')
    row['outcome_class'] = ('goal_rejected_before_execution' if code == 'invalid_goal' and op.get('action_count') == 0
        else 'completed' if op.get('returned_success') else 'cancelled' if op.get('cancelled')
        else 'hold_observation' if op.get('kind') in ('post_hold', 'hold_observation')
        else 'lifecycle_interruption' if code == 'lifecycle_invalidated'
        else 'execution_tracking_failure' if code == 'tracking_failure' else 'other_noncompletion')
    row['clearance'] = pick(op.get('clearance') or {}, ('minimum_m', 'movable_arm_minimum_m'))
    row['contacts'] = pick(op.get('contacts') or {}, ('reported_points', 'reported_pairs', 'penetrating_pairs', 'contact_buffer_saturated'))
    for key in ('before_target_gap_arm_rad', 'after_target_gap_arm_rad'):
        row[key.replace('_arm_rad', '_maximum_rad')] = maximum_absolute(op.get(key))
    for key in ('post_home_hold', 'retained_home_hold', 'hold_observation'):
        if key in op:
            row[key] = pick(op[key], HOLD_FIELDS)
            velocity = maximum_absolute(op[key].get('final_actual_arm_velocity_rad_s'))
            row[key]['final_measured_arm_velocity_max_rad_s'] = (velocity if velocity is not None
                else op[key].get('final_measured_arm_velocity_max_rad_s'))
    probe = op.get('solver_velocity_probe')
    if probe:
        row['solver_probe'] = pick(probe, ('requested_velocity_iterations', 'requested_position_iterations',
            'no_physics_step_or_time_advance_during_set', 'joint_state_change_during_set_rad',
            'original_session_opinion_restored', 'baseline_comparison', 'native_actor_properties_before',
            'native_actor_properties_after_set', 'native_actor_properties_after_restore'))
    return row


def native_count_aggregates(reference):
    path = Path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Raw trace SHA mismatch before native count aggregation')
    raw = json.loads(path.read_text())
    rows = []
    for generation, document in enumerate(raw if isinstance(raw, list) else [raw]):
        for op in document['operations']:
            samples = document['samples'][op['first_sample']:op['last_sample']]
            transitions, available = [], 0
            for sample in samples:
                reading = sample.get('native_actor_solver_properties') or {}
                if not reading.get('available'):
                    continue
                available += 1
                counts = dict(position=reading['position'], velocity=reading['velocity'])
                if not transitions or counts != transitions[-1]:
                    transitions.append(counts)
            rows.append(dict(runtime_generation=generation, id=op['id'], label=op['label'],
                available_samples=available, unavailable_samples=len(samples)-available, transitions=transitions))
    return dict(operations=rows, scope='Native actor requested counts; effective island counts are not exposed')


def controlled_record(path, *, raw_counts=False, analysis=None):
    row = dict(artifact=bind(path))
    if row['artifact']['sha256'] is None:
        return dict(row, measurement_status='UNAVAILABLE', measurement_valid=False)
    raw = json.loads(Path(path).read_text())
    section = raw.get('acceptance') or raw.get('controlled')
    if section is None:
        return dict(row, measurement_status=raw.get('status', 'UNAVAILABLE'), measurement_valid=False,
            measurement_assessment='missing_observation_section', error_recorded=bool(raw.get('error')),
            behavior_acceptance_passed=None, post_hold_passed=None, operations=[])
    observations = section.get('observation_checks') or section.get('checks') or {}
    valid = raw.get('status') == 'PASS' and bool(observations) and all(value is True for value in observations.values())
    holds = [op[key] for op in section['operations'] for key in ('post_home_hold', 'retained_home_hold') if key in op]
    row.update(mode=section['mode'], strategy=section.get('strategy', 'global'), measurement_status=raw.get('status'),
        measurement_valid=valid, observation_checks=observations, behavior_acceptance_passed=section['acceptance_passed'],
        measurement_assessment='valid_observation' if valid else 'invalid_observation_not_an_executor_verdict',
        failed_observation_checks=[key for key, value in observations.items() if value is not True],
        error_recorded=bool(raw.get('error')),
        post_hold_passed=all(hold.get('validation_passed', hold.get('kinematic_stability_passed')) is True for hold in holds) if holds else None,
        behavior_checks=section.get('checks') if 'acceptance' in raw else None,
        operations=[compact_operation(op) for op in section['operations']], raw_trace=section.get('raw_trace'),
        raw_trace_verified=raw_counts,
        source_manifest_sha256=fingerprint(section.get('source_hashes_before') or {}),
        source_unchanged=section.get('source_hashes_before') == section.get('source_hashes_after'))
    if not valid:
        for operation in row['operations']:
            operation['outcome_class'] = 'not_assessed_invalid_observation'
    if raw_counts:
        row['native_counts'] = native_count_aggregates(section['raw_trace'])
    if analysis:
        matched = next((case for case in analysis['cases'] if case['artifact_sha256'] == row['artifact']['sha256']), None)
        if matched:
            row['independent_analysis'] = [dict(id=op['id'], label=op['label'],
                physical_joint1=op.get('physical_joint1'), physical_hand_vs_model_fk=op.get('physical_hand_vs_model_fk'),
                target_agreement={key: pick(value, ('available', 'samples', 'maximum_absolute'))
                    for key, value in (op.get('targets') or {}).items() if isinstance(value, dict)},
                dynamics=scalar_tree(op.get('dynamics'))) for op in matched['operations']]
    return row


def static_record(path):
    row, signature, contracts = load_attempt(path)  # Preserve the authoritative V4 metric definitions.
    public = dict(artifact=bind(path), contracts=scalar_tree(contracts), **pick(row, STATIC_FIELDS))
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raw = {}
    metrics = raw.get('metrics') or {}
    diagnostics = dict(((metrics.get('motion') or {}).get('result') or {}).get('details') or {})
    diagnostics.update(metrics.get('strategy_diagnostics') or {})
    public['execution_solver'] = scalar_tree(diagnostics.get('execution_solver'))
    public.update(source_manifest_sha256=signature['source_manifest_sha256'],
        physical_world_sha256=signature['physical_world_sha256'],
        physical_configuration_sha256=fingerprint({k: v for k, v in signature.items() if k != 'source_manifest_sha256'}))
    return public, row


def static_comparisons(candidates, references):
    key = lambda item: tuple(item.get(field) for field in ('scenario', 'strategy', 'seed', 'mode'))
    indexed, comparisons = defaultdict(list), []
    for public, internal in references:
        indexed[key(public)].append((public, internal))
    for candidate, candidate_raw in candidates:
        matches = indexed[key(candidate)]
        record = dict(candidate=candidate['artifact'], key=list(key(candidate)), matching_references=len(matches))
        if len(matches) == 1:
            reference, reference_raw = matches[0]
            before, after = reference_raw.get('initial_joints') or [], candidate_raw.get('initial_joints') or []
            q_error = maximum_absolute([a-b for a, b in zip(before, after)]) if (
                len(before) == len(after) >= 7 and all(number(value) is not None for value in before+after)) else None
            same_fixture = bool(candidate['physical_world_sha256'] and reference['physical_world_sha256']
                and candidate['physical_configuration_sha256'] == reference['physical_configuration_sha256'])
            record.update(reference=reference['artifact'], physical_configuration_equal=same_fixture,
                source_cohorts=[reference['source_manifest_sha256'], candidate['source_manifest_sha256']],
                source_cohorts_equal=reference['source_manifest_sha256'] == candidate['source_manifest_sha256'],
                actual_initial_joint_max_difference_rad=q_error,
                both_measurements_valid=bool(reference['measurement_valid'] and candidate['measurement_valid']))
            if same_fixture and record['both_measurements_valid']:
                record['descriptive_deltas'] = {field: dict(before=reference.get(field), after=candidate.get(field),
                    delta=candidate[field]-reference[field], percent_change=100*(candidate[field]/reference[field]-1) if reference[field] else None)
                    for field in ('motion_wall_seconds', 'planning_wall_seconds', 'rrt_seconds', 'execution_wall_seconds', 'execution_sim_seconds')
                    if number(candidate.get(field)) is not None and number(reference.get(field)) is not None}
        record['scope'] = 'Same scenario/strategy/seed fixture comparison only; changed source cohorts and measured initial-state differences are explicit. Single runs are descriptive, not controlled performance replicates.'
        comparisons.append(record)
    return comparisons


def regression_record(path):
    record = dict(artifact=bind(path))
    if record['artifact']['sha256'] is None:
        return dict(record, status='UNAVAILABLE')
    raw = json.loads(Path(path).read_text())
    groups = {key: pick(value, ('status', 'return_code')) for key, value in raw.items()
        if isinstance(value, dict) and 'status' in value}
    groups_passed = all(value.get('status') == 'PASS' for value in groups.values()) if groups else None
    checks = {key: scalar_tree(raw[key]) for key in ('checks', 'native_acceptance') if isinstance(raw.get(key), dict)}
    native_passed = (raw.get('status') == 'PASS' and bool(checks) and not raw.get('error')
        and all(bool(group) and all(value is True for value in group.values()) for group in checks.values()))
    return dict(record, groups=groups, all_groups_passed=groups_passed,
        native_status=raw.get('status'), native_checks=checks, native_envelope_passed=native_passed if 'status' in raw else None,
        regression_passed=native_passed if 'status' in raw else groups_passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'probes', 'stabilized', 'acceptance', 'static', 'v4-static', 'regressions'):
        parser.add_argument('--'+name, nargs='*', default=[])
    parser.add_argument('--analysis')
    parser.add_argument('--read-raw-counts', action='store_true')
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    analysis = json.loads(Path(args.analysis).read_text()) if args.analysis else None
    sections = {name: [controlled_record(path, raw_counts=args.read_raw_counts, analysis=analysis)
        for path in getattr(args, name)] for name in ('baseline', 'probes', 'stabilized', 'acceptance')}
    statics, old = [static_record(path) for path in args.static], [static_record(path) for path in args.v4_static]
    report = dict(schema_version=1, producer=bind(__file__), sections=sections,
        section_counts={name: dict(records=len(rows), valid=sum(row['measurement_valid'] for row in rows),
            behavior_pass=sum(row.get('behavior_acceptance_passed') is True for row in rows),
            post_hold_pass=sum(row.get('post_hold_passed') is True for row in rows)) for name, rows in sections.items()},
        static=[row for row, _ in statics], v4_static=[row for row, _ in old],
        static_comparisons=static_comparisons(statics, old), regressions=[regression_record(path) for path in args.regressions],
        independent_analysis_artifact=bind(args.analysis) if args.analysis else None,
        meaning='Evidence inventory, not a blanket PASS. Observation validity, behavior acceptance, post-hold stability and static rejection outcomes remain separate. No raw poses, trajectory arrays or contact samples are published.')
    target = Path(args.result);target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(bytes=target.stat().st_size, section_counts=report['section_counts'])))


if __name__ == '__main__':
    main()
