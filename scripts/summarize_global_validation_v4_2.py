#!/usr/bin/env python3
"""Reduce V4.2 result envelopes after native timing, retaining every attempt.

No simulator, raw trace, mesh, contact-array or plotting imports are needed.
Measurement validity, reported completion, sampled physical clearance, strict
zero-report success and rejection feasibility remain separate assessments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from scripts.summarize_global_motion_v4 import digest, fingerprint, label, number
from scripts.summarize_global_stabilization_v4_1 import compact_operation, pick


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT/'outputs/motion-planning-v4-2'
ASSETS = ROOT/'docs/assets/motion_planning_v4_2'
TIMING = ('snapshot_seconds world_update_seconds ik_seconds rrt_seconds route_validation_seconds '
    'trajectory_generation_seconds trajectory_validation_seconds planning_and_validation_wall_seconds '
    'configuration_validation_calls configuration_validation_wall_seconds native_rrt_collision_query_count '
    'native_rrt_collision_query_seconds trajectory_duration_seconds trajectory_duration '
    'execution_wall_seconds execution_sim_seconds').split()
METRICS = ('configuration final_pose_errors strict_success movable_arm_strict_success goal_reached '
    'classification no_requested_movement_joint_delta_rad clearance initial_clearance '
    'closest_robot_link_by_obstacle control_interval_ms kit_interval_ms controller_wait_ms '
    'controller_waits_over_100ms memory_before_native memory_after_runtime_close '
    'memory_after_offline_reduction actual_model_selection_verified').split()
DIAGNOSTICS = ('failure_code failure_message global_collision_model state rrt_waypoints '
    'geometric_joint_path_length trajectory_validation execution_solver joint_tracking_error_rad '
    'position_error_m orientation_error_rad measured_arm_velocity_rad_s commanded_updates '
    'final_hold configuration_validation_scope rrt_self_collision_policy self_collision_mask '
    'enhanced_model_metadata_sha256 sdk_configuration_sha256').split()


def read_object(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('Result envelope must be a JSON object')
        return value, None
    except (OSError, ValueError, TypeError) as error:
        return {}, f'{type(error).__name__}: {error}'


def reference(path):
    path = Path(path)
    return dict(path=label(path), sha256=digest(path) if path.is_file() else None)


def checks_pass(value):
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def envelope(path, declared=None):
    path = Path(path)
    raw, error = read_object(path)
    driver_path = path.with_name('driver_result.json')
    driver, driver_error = read_object(driver_path)
    driver_valid = (not driver_path.is_file() or driver_error is None and driver.get('status') == 'PASS'
        and driver.get('return_code') == 0 and not driver.get('driver_error')
        and not driver.get('fatal_gpu_messages'))
    state = (raw.get('status', 'INVALID_RESULT') if path.is_file() else
        'NO_RESULT_AFTER_LAUNCH' if path.with_name('launch.json').is_file() else 'NOT_RUN')
    return raw, dict(artifact=reference(path), launch=reference(path.with_name('launch.json')),
        driver_artifact=reference(path.with_name('driver_result.json')),
        declared_label=declared, measurement_status=state,
        driver=pick(driver, ('status', 'return_code', 'wall_seconds', 'driver_error')),
        driver_read_error=driver_error if driver_path.is_file() else None,
        driver_valid=driver_valid, error=raw.get('error') or error,
        source_manifest_sha256=fingerprint(raw['hashes']) if raw.get('hashes') else None)


def static_record(path, declared=None):
    raw, record = envelope(path, declared)
    metrics, contracts = raw.get('metrics') or {}, raw.get('contracts') or {}
    valid = (record['measurement_status'] == 'PASS' and checks_pass(contracts)
        and not record['error'] and record['driver_valid'])
    outcome = (metrics.get('motion') or {}).get('result') or {}
    details = dict(outcome.get('details') or {})
    details.update(metrics.get('strategy_diagnostics') or {})
    details['failure_code'] = details.get('failure_code') or outcome.get('failure_code')
    details['failure_message'] = details.get('failure_message') or outcome.get('message')
    record.update(measurement_valid=valid, contracts=contracts,
        metrics=pick(metrics, METRICS), diagnostics=pick(details, DIAGNOSTICS),
        timing={key: number(details.get(key)) for key in TIMING},
        physical_world_sha256=fingerprint(metrics['physical_world']) if metrics.get('physical_world') else None)
    # These are compact configurations, not the raw seven-joint time series.
    record['metrics']['configuration'] = metrics.get('configuration') or {}
    record['initial_joint_positions_rad'] = metrics.get('initial_joints')
    record['motion_result_status'] = outcome.get('status')
    record['motion_wall_seconds'] = number((metrics.get('motion') or {}).get('wall_seconds'))
    record['rejection_phase'] = ('upstream_workspace_gate_before_rrt'
        if details['failure_code'] == 'unreachable' and details.get('state') == 'ready'
        and metrics.get('no_requested_movement_joint_delta_rad') == 0 else
        'global_planner_or_validator' if details['failure_code'] else None)
    severity = metrics.get('contact_severity') or {}
    contacts = metrics.get('contacts') or {}
    separation = number(contacts.get('minimum_reported_separation_m'))
    record['contacts'] = dict(reported_points=contacts.get('reported_points'),
        minimum_reported_separation_m=separation,
        positive_separation_reported_points=(contacts.get('reported_points') if separation is not None and separation > 0 else
            0 if contacts.get('reported_points') == 0 else None),
        observed_physical_contact_steps=severity.get('observed_physical_contact_steps'),
        maximum_penetration_m=(severity.get('depth_m') or {}).get('maximum'),
        maximum_normal_pair_impulse_ns=(severity.get('normal_pair_impulse_ns') or {}).get('maximum'),
        physical_classification=severity.get('physical_classification'),
        contact_buffer_saturated=contacts.get('contact_buffer_saturated'),
        acceptability_threshold='NONE_DEFINED')
    clearance = number((metrics.get('clearance') or {}).get('minimum_m'))
    complete = metrics.get('goal_reached') is True
    values = [clearance, number(severity.get('observed_physical_contact_steps')),
        number((severity.get('depth_m') or {}).get('maximum')),
        number((severity.get('normal_pair_impulse_ns') or {}).get('maximum'))]
    if contacts.get('reported_points') == 0:
        values[2:] = [0., 0.]  # No observed penetration/impulse reports; independent clearance is still required.
    record['sampled_physical_contact_free_completion'] = (None if not valid or any(value is None for value in values)
        else bool(complete and values[0] > 0 and all(value == 0 for value in values[1:])))
    record['rejection_physical_feasibility'] = 'NOT_ESTABLISHED_BY_PLANNER_REJECTION'
    record['ik_attempts'] = [{key: attempt[key] for key in ('index', 'ik_success', 'status',
        'ik_seconds', 'rrt_seconds', 'route_validation_seconds', 'trajectory_generation_seconds',
        'trajectory_validation_seconds', 'route_validation_samples') if key in attempt}
        for attempt in details.get('ik_attempts') or []]
    return record


def lifecycle_record(path, declared=None):
    raw, record = envelope(path, declared)
    section = raw.get('acceptance') or {}
    valid = (record['measurement_status'] == 'PASS' and checks_pass(raw.get('contracts'))
        and checks_pass(section.get('observation_checks')) and not record['error'] and record['driver_valid'])
    record.update(measurement_valid=valid, mode=section.get('mode'), seed=section.get('seed'),
        observation_checks=section.get('observation_checks') or {}, behavior_checks=section.get('checks') or {},
        behavior_acceptance_passed=(section.get('acceptance_passed') if valid else None),
        source_unchanged=(section.get('source_hashes_before') == section.get('source_hashes_after')
            if section.get('source_hashes_before') else None),
        operations=[compact_operation(operation) for operation in section.get('operations') or []])
    episode = section.get('episode') or {}
    returned = episode.get('result') or {}
    returned_details = returned.get('details') or {}
    record['episode_flags'] = {key: value for key, value in episode.items() if isinstance(value, bool)}
    record['post_fault_result'] = (dict(status=returned.get('status'),
        failure_code=returned_details.get('failure_code') or returned.get('failure_code'),
        execution_solver=returned_details.get('execution_solver')) if returned else None)
    record['final_execution_solver'] = ((raw.get('metrics') or {}).get('strategy_after_close') or {}).get('execution_solver')
    record['interruption_snapshot_scope'] = ('Stop/reset/close operation observations end before the fault; '
        'their transient solver restored=False is not the final cleanup result. '
        'post_fault_result records the awaited interruption result; final_execution_solver records runtime-close diagnostics.')
    for compact, original in zip(record['operations'], section.get('operations') or []):
        compact['diagnostics'].update(pick(original.get('strategy_diagnostics') or {}, DIAGNOSTICS))
        compact['timing'] = {key: number((original.get('strategy_diagnostics') or {}).get(key)) for key in TIMING}
        compact['contact_severity'] = pick(original.get('contact_severity') or {},
            ('observed_physical_contact_steps', 'minimum_reported_separation_m', 'depth_m', 'normal_pair_impulse_ns'))
        if not valid:
            compact['outcome_class'] = 'not_assessed_invalid_observation'
    return record


def regression_record(path, declared=None):
    raw, record = envelope(path, declared)
    groups = {key: pick(value, ('status', 'return_code', 'error')) for key, value in raw.items()
        if isinstance(value, dict) and 'status' in value} if 'status' not in raw else {}
    checks = {key: raw[key] for key in ('checks', 'native_acceptance', 'contracts') if isinstance(raw.get(key), dict)}
    passed = (all(item.get('status') == 'PASS' and item.get('return_code') == 0 for item in groups.values())
        if groups else record['measurement_status'] == 'PASS' and bool(checks)
            and all(checks_pass(value) for value in checks.values()))
    record.update(regression_passed=bool(passed and not record['error'] and record['driver_valid']),
        checks=checks, groups=groups, metrics=pick(raw.get('metrics') or {},
            ('home', 'move', 'pick', 'place', 'motion_strategy', 'lifecycle', 'control_interval_ms',
                'kit_interval_ms', 'contacts', 'avoidance_success')))
    if groups:
        record['measurement_status'] = 'PASS' if passed else 'FAIL'
    return record


def inventory(root, folders, manifest_paths=()):
    """Include frozen declarations, every recorded result and failed launches."""
    found = {}
    for manifest in manifest_paths:
        document, error = read_object(manifest)
        if error:
            raise ValueError(f'Cannot read frozen job manifest: {error}')
        for job in document.get('jobs') or []:
            requested = Path(job['result'])
            parts = requested.parts
            marker = 'motion-planning-v4-2'
            if marker not in parts:
                raise ValueError('Declared result is outside the V4.2 output tree')
            result = root.joinpath(*parts[parts.index(marker)+1:])
            found[result] = job['label']
    for folder in folders:
        base = root/folder
        if base.is_dir():
            for pattern in ('result.json', 'launch.json', 'driver_result.json'):
                for path in base.rglob(pattern):
                    found.setdefault(path.with_name('result.json'), None)
    return [(path, found[path]) for path in sorted(found)]


def distribution(values):
    values = [value for item in values if (value := number(item)) is not None]
    return dict(count=len(values), median=statistics.median(values) if values else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None)


def performance(records):
    groups = defaultdict(list)
    rows = []
    for record in records:
        config = record['metrics'].get('configuration') or {}
        model = record['diagnostics'].get('global_collision_model') or config.get('global_collision_model')
        row = dict(artifact=record['artifact'], measurement_valid=record['measurement_valid'],
            model=model, scenario=config.get('scenario'), seed=config.get('seed'), timing=record['timing'],
            kit_interval_ms=record['metrics'].get('kit_interval_ms'),
            controller_wait_ms=record['metrics'].get('controller_wait_ms'),
            joint_tracking_error_rad=record['diagnostics'].get('joint_tracking_error_rad'),
            memory={key: record['metrics'].get(key) for key in ('memory_before_native',
                'memory_after_runtime_close', 'memory_after_offline_reduction')})
        calls, seconds = number(row['timing'].get('configuration_validation_calls')), number(row['timing'].get('configuration_validation_wall_seconds'))
        row['mean_explicit_configuration_check_seconds'] = seconds/calls if calls and seconds is not None else None
        rows.append(row)
        if record['measurement_valid']:
            groups[(model, config.get('strategy'))].append(row)
    cohorts = [dict(model=key[0], strategy=key[1], valid_attempts=len(values),
        timing={field: distribution(row['timing'].get(field) for row in values) for field in TIMING})
        for key, values in sorted(groups.items(), key=lambda item: str(item[0]))]
    return dict(attempts=rows, descriptive_cohorts=cohorts, successful_matched_comparisons=paired_performance(records),
        interpretation='Single frozen-scene attempts; descriptive cohorts include valid rejections and completions separately identifiable by source artifact. No confidence interval or controlled runtime speedup claim.',
        configuration_query_scope='Explicit external/self/domain validator calls, including failed calls; native RRT internal query counts and time remain unavailable.',
        memory_scope='RSS includes Kit and retained observations; process differences alone do not establish a resource leak.')


def paired_performance(records):
    indexed = defaultdict(dict)
    for record in records:
        config = record['metrics'].get('configuration') or {}
        model = record['diagnostics'].get('global_collision_model') or config.get('global_collision_model')
        indexed[(config.get('scenario'), config.get('seed'))].setdefault(model, []).append(record)
    matched, excluded = [], []
    for key, models in sorted(indexed.items(), key=lambda item: str(item[0])):
        if len(models.get('stock', [])) != 1 or len(models.get('enhanced', [])) != 1:
            excluded.append(dict(scenario=key[0], seed=key[1], reason='missing or ambiguous stock/enhanced pair'))
            continue
        before, after = models['stock'][0], models['enhanced'][0]
        configs = [{name: value for name, value in record['metrics']['configuration'].items()
                    if name != 'global_collision_model'} for record in (before, after)]
        checks = dict(both_valid=before['measurement_valid'] and after['measurement_valid'],
            both_completed=before['metrics'].get('goal_reached') is True and after['metrics'].get('goal_reached') is True,
            same_task_configuration=configs[0] == configs[1],
            same_physical_world=bool(before['physical_world_sha256']) and before['physical_world_sha256'] == after['physical_world_sha256'],
            same_recorded_initial_joints=bool(before['initial_joint_positions_rad'])
                and before['initial_joint_positions_rad'] == after['initial_joint_positions_rad'],
            same_source_cohort=bool(before['source_manifest_sha256'])
                and before['source_manifest_sha256'] == after['source_manifest_sha256'])
        row = dict(scenario=key[0], seed=key[1], stock=before['artifact'], enhanced=after['artifact'], checks=checks)
        if not all(checks.values()):
            excluded.append(row); continue
        row['timing'] = {field: dict(stock=before['timing'].get(field), enhanced=after['timing'].get(field)) for field in TIMING}
        matched.append(row)
    comparison = {}
    for field in TIMING:
        stocks = [row['timing'][field]['stock'] for row in matched]
        enhanced = [row['timing'][field]['enhanced'] for row in matched]
        before, after = distribution(stocks), distribution(enhanced)
        comparison[field] = dict(stock=before, enhanced=after,
            median_change_percent=(100*(after['median']/before['median']-1)
                if before['median'] and after['median'] is not None else None))
    return dict(matched_pairs=matched, excluded_pairs=excluded, descriptive_timing=comparison,
        interpretation='Only pairs with valid completed movements, identical frozen task/world/initial joints/source cohort are pooled. Different sphere geometry can change RRT routes and query counts: elapsed RRT time is not per-query collision cost or an algorithmic speedup proof. One attempt per model/task/seed gives descriptive measurements, not statistical confidence.')


def static_table(records):
    groups = defaultdict(list)
    for record in records:
        config = record['metrics'].get('configuration') or {}
        model = record['diagnostics'].get('global_collision_model') or config.get('global_collision_model')
        groups[(model, config.get('scenario'))].append(record)
    rows = []
    for (model, scenario), cases in sorted(groups.items(), key=lambda item: str(item[0])):
        valid = [record for record in cases if record['measurement_valid']]
        clearances = [number((record['metrics'].get('clearance') or {}).get('movable_arm_minimum_m')) for record in valid]
        rows.append(dict(model=model, scenario=scenario, attempts=len(cases), valid=len(valid),
            reported_completions=sum(record['metrics'].get('goal_reached') is True for record in valid),
            strict_zero_report_successes=sum(record['metrics'].get('strict_success') is True for record in valid),
            sampled_physical_contact_free_completions=sum(record['sampled_physical_contact_free_completion'] is True for record in valid),
            minimum_movable_arm_clearance_m=min((value for value in clearances if value is not None), default=None),
            reported_contact_points=sum(record['contacts'].get('reported_points') or 0 for record in valid),
            positive_separation_reported_points=sum(record['contacts'].get('positive_separation_reported_points') or 0 for record in valid),
            observed_physical_contact_steps=sum(record['contacts'].get('observed_physical_contact_steps') or 0 for record in valid),
            rejection_codes=dict(Counter(record['diagnostics'].get('failure_code') for record in valid
                if record['diagnostics'].get('failure_code')))))
    return rows


def build_reports(root=INPUT_ROOT, assets=ASSETS):
    root, assets = Path(root), Path(assets)
    static = [static_record(path, declared) for path, declared in inventory(root, ('static',), (assets/'static_jobs.json',))]
    lifecycle = [lifecycle_record(path, declared) for path, declared in inventory(root, ('lifecycle',), (assets/'lifecycle_jobs.json',))]
    regressions = [regression_record(path, declared) for path, declared in inventory(root, ('regression',),
        (assets/'direct_jobs.json', assets/'reactive_jobs.json'))]
    for folder in sorted(root.glob('units*')):
        regressions.extend(regression_record(path) for path in sorted(folder.rglob('summary.json')))
    common = dict(schema_version=1, producer=reference(__file__), input_root=label(root),
        meaning='Evidence inventory only. Missing/failed attempts are retained. A planner rejection does not establish physical infeasibility; zero contact reports alone do not establish geometric clearance.')
    return {'static_results.json': dict(common, attempts=static,
                counts=dict(Counter(record['measurement_status'] for record in static)), scenario_table=static_table(static)),
        'performance.json': dict(common, **performance(static)),
        'lifecycle.json': dict(common, attempts=lifecycle),
        'regressions.json': dict(common, attempts=regressions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=ASSETS)
    parser.add_argument('--protocol-dir', type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    reports = build_reports(args.root, args.protocol_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (args.output_dir/name).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: len(report.get('attempts') or []) for name, report in reports.items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
