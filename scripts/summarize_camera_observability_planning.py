#!/usr/bin/env python3
"""Reduce explicit camera-study planner cases without hiding failed attempts.

This is an SDK-free result reader. Historical ground-truth results are labelled
reference context and never enter new RGB-D observations or controller inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_FIELDS = ('occupied_voxels', 'supported_voxels', 'spatial_supported_voxels',
    'strong_sparse_retained', 'rejected_sparse_voxels', 'candidate_boxes', 'proxy_count',
    'represented_voxels', 'omitted_voxels', 'floor_candidate_boxes', 'hazard_candidate_boxes',
    'selected_floor_boxes', 'selected_hazard_boxes', 'represented_floor_voxels',
    'omitted_floor_voxels', 'run_budget_omitted_voxels', 'max_obstacles', 'truncated')


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def path_label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def known_delta(before, after, key):
    start, stop = number(before.get(key)), number(after.get(key))
    return None if start is None or stop is None or stop < start else stop-start


def launch_context(folder):
    """Read only known experiment switches, never unrelated command arguments."""
    try:
        payload = json.loads((folder/'launch.json').read_text())
        command = payload.get('command', [])
    except (OSError, ValueError):
        return {}
    result = {}
    for option, key in [('--layout-id', 'layout_id'), ('--scenario', 'scenario'),
        ('--strategy', 'strategy'), ('--source', 'source'), ('--max-rays', 'max_rays'),
        ('--sampling-policy', 'sampling_policy'), ('--mapping-hz', 'mapping_hz')]:
        if option in command:
            index = command.index(option)+1
            if index < len(command):
                result[key] = command[index]
    return result


def load_case(path, *, cohort='primary'):
    path = Path(path)
    path = path/'result.json' if path.is_dir() else path
    folder = path.parent
    context = launch_context(folder)
    digest = None
    try:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise ValueError('Expected a result object')
    except (OSError, ValueError) as exc:
        raw = dict(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
    metrics, research = raw.get('metrics') or {}, raw.get('research') or {}
    config = metrics.get('configuration') or {}
    contracts = raw.get('contracts') or {}
    camera_contracts = research.get('camera_contracts') or {}
    valid = (raw.get('status') == 'PASS' and bool(contracts)
             and all(value is True for value in contracts.values())
             and all(value is True for value in camera_contracts.values())
             and not raw.get('error') and not research.get('error'))
    motion = metrics.get('motion') or {}
    outcome = motion.get('result') or {}
    contacts, severity = metrics.get('contacts') or {}, metrics.get('contact_severity') or {}
    clearance = metrics.get('clearance') or {}
    controller, updates = metrics.get('controller') or {}, metrics.get('obstacle_updates') or {}
    extraction = updates.get('extraction') or {}
    initial, final = metrics.get('initial') or {}, metrics.get('final') or {}
    strategy = config.get('strategy', context.get('strategy'))
    source = config.get('mapping_source', context.get('source'))
    sample_policy = research.get('sampling_policy', context.get('sampling_policy', 'fixed'))
    errors = [str(value)[:2000] for value in (raw.get('error'), research.get('error')) if value]
    max_depth = number((severity.get('depth_m') or {}).get('maximum'))
    max_impulse = number((severity.get('normal_pair_impulse_ns') or {}).get('maximum'))
    if max_depth is None and contacts.get('reported_points') == 0:
        max_depth = 0.
    if max_depth is None and number(contacts.get('minimum_reported_separation_m')) is not None:
        max_depth = max(0., -contacts['minimum_reported_separation_m'])
    if max_impulse is None:
        max_impulse = number(contacts.get('maximum_pair_impulse'))
    row = dict(case_id=path_label(folder), cohort=cohort,
        artifact=dict(path=path_label(path), sha256=digest),
        layout_id=research.get('layout_id', context.get('layout_id')),
        scenario=context.get('scenario'), source=source, strategy=strategy,
        sampling_policy=sample_policy,
        configured_mapping_hz=number(config.get('mapping_hz', context.get('mapping_hz'))),
        global_ray_budget=research.get('global_ray_budget', number(context.get('max_rays'))),
        research_source_hashes={key: research.get(key) for key in
            ('wrapper_sha256', 'layout_file_sha256', 'sampling_script_sha256')},
        runtime_source_manifest_sha256=(hashlib.sha256(json.dumps(raw['hashes'], sort_keys=True).encode()).hexdigest()
            if isinstance(raw.get('hashes'), dict) else None),
        measurement_status=raw.get('status', 'UNAVAILABLE'), measurement_valid=valid,
        measurement_contracts=contracts, camera_contracts=camera_contracts, errors=errors,
        classification=metrics.get('classification'), goal_status=outcome.get('status'),
        goal_reached=None if not valid else outcome.get('status') == 'success',
        goal_failure_code=outcome.get('failure_code'), goal_message=outcome.get('message'),
        strict_contact_free_success=metrics.get('avoidance_success') if valid else None,
        motion_seconds=number(motion.get('seconds')),
        minimum_cooked_link_clearance_m=number(clearance.get('minimum_m')),
        clearance_by_link_group_m=clearance.get('by_group_m'),
        closest_pair=clearance.get('closest_pair'), clearance_reference=clearance.get('reference'),
        reported_contact_points=contacts.get('reported_points'),
        penetrating_pairs=contacts.get('penetrating_pairs'),
        maximum_observed_penetration_m=max_depth, maximum_observed_normal_pair_impulse_ns=max_impulse,
        physical_contact_classification=severity.get('physical_classification'),
        severity_pass=severity.get('severity_pass'),
        severity_acceptability=severity.get('limited_low_severity_assessment'),
        observed_physical_contact_steps=severity.get('observed_physical_contact_steps'),
        sampled_physical_contact_duration_s=number(severity.get('sampled_physical_contact_duration_s')),
        contact_point_displacement_speed_m_s=severity.get('robot_contact_point_displacement_speed_m_s'),
        contact_buffer_saturated=contacts.get('buffer_saturated'),
        controller_interval_ms=controller.get('apply_interval_ms'),
        controller_interval_p99_ms=number((controller.get('apply_interval_ms') or {}).get('p99')),
        controller_interval_worst_ms=number((controller.get('apply_interval_ms') or {}).get('worst')),
        kit_interval_ms=controller.get('kit_interval_ms'),
        controller_waits_over_100ms=controller.get('waits_over_100_ms'),
        representation_applicable=strategy == 'reactive',
        maximum_registered_proxies=updates.get('maximum_proxy_count') if strategy == 'reactive' else None,
        consumed_map_age_ms=updates.get('map_age_ms') if strategy == 'reactive' else None,
        consumed_map_age_worst_ms=number((updates.get('map_age_ms') or {}).get('worst')) if strategy == 'reactive' else None,
        final_mapping_source_age_ms=(None if number((final.get('mapping') or {}).get('age_seconds')) is None
            else final['mapping']['age_seconds']*1000),
        final_mapping_fresh=(final.get('mapping') or {}).get('fresh'),
        reactive_source_failures_total=(metrics.get('closed_strategy') or {}).get('source_failures'),
        observation_rates=metrics.get('observation_rates'),
        map_states=metrics.get('map_states'), extraction={key: extraction.get(key) for key in EXTRACTION_FIELDS},
        original_camera_calibration=research.get('actual_camera_calibration'))
    if row['scenario'] is None:
        # The physical fixture records boxes rather than a scenario label.
        # Optional wrapper/driver metadata is explicit; never guess from geometry.
        row['scenario'] = research.get('scenario', raw.get('scenario'))
    candidate, selected = number(extraction.get('candidate_boxes')), number(extraction.get('proxy_count'))
    row['extraction']['unselected_candidate_boxes'] = (None if candidate is None or selected is None
        else max(0., candidate-selected))
    row['worker_counts_in_command'] = {}
    for label in ('mapping', 'semantic'):
        before, after = initial.get(label) or {}, final.get(label) or {}
        values = {key: known_delta(before, after, key) for key in
            ('submitted', 'attempts', 'published', 'rejected', 'source_failures', 'worker_failures', 'failures')}
        values['worker'] = {key: known_delta(before.get('worker') or {}, after.get('worker') or {}, key)
            for key in ('submitted', 'started', 'completed', 'replaced', 'discarded', 'errors', 'delivered')}
        row['worker_counts_in_command'][label] = values
    return row


def comparisons(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['cohort'] != 'primary':
            continue
        key = (row['layout_id'], row['scenario'], row['source'], row['sampling_policy'],
               row['global_ray_budget'], row['configured_mapping_hz'])
        groups[key][row['strategy']].append(row)
    result = []
    for key, strategies in sorted(groups.items(), key=lambda item: repr(item[0])):
        direct, reactive = strategies.get('direct', []), strategies.get('reactive', [])
        entry = dict(layout_id=key[0], scenario=key[1], source=key[2], sampling_policy=key[3],
            global_ray_budget=key[4], configured_mapping_hz=key[5], direct_cases=[r['case_id'] for r in direct],
            reactive_cases=[r['case_id'] for r in reactive])
        if len(direct) != 1 or len(reactive) != 1:
            entry['status'] = 'UNPAIRED' if not direct or not reactive else 'MULTIPLE_ATTEMPTS_RETAINED'
        elif not direct[0]['measurement_valid'] or not reactive[0]['measurement_valid']:
            entry['status'] = 'INVALID_MEASUREMENT_RETAINED'
        else:
            entry['status'] = 'MATCHED_VALID_MEASUREMENTS'
            for field in ('motion_seconds', 'minimum_cooked_link_clearance_m',
                'maximum_observed_penetration_m', 'maximum_observed_normal_pair_impulse_ns',
                'reported_contact_points', 'controller_interval_p99_ms'):
                first, last = number(direct[0].get(field)), number(reactive[0].get(field))
                entry[field] = dict(direct=first, reactive=last,
                    reactive_minus_direct=None if first is None or last is None else last-first)
            entry['goal_reached'] = dict(direct=direct[0]['goal_reached'], reactive=reactive[0]['goal_reached'])
            entry['strict_contact_free_success'] = dict(direct=direct[0]['strict_contact_free_success'],
                reactive=reactive[0]['strict_contact_free_success'])
        result.append(entry)
    return result


def ground_truth_context(path):
    if path is None:
        return None
    path = Path(path)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    rows = []
    for case in raw.get('final_cases', []):
        if case.get('source') != 'ground_truth' or case.get('strategy') != 'reactive':
            continue
        if case.get('configured_mapping_hz', 2.) != 2.:
            continue
        severity = case.get('severity') or {}
        rows.append(dict(scenario=case.get('scenario'), measurement_status=case.get('measurement_status'),
            goal_status=case.get('goal_status'), strict_contact_free_success=case.get('strict_avoidance_success'),
            classification=case.get('classification'), motion_seconds=case.get('motion_seconds'),
            minimum_cooked_link_clearance_m=(case.get('clearance') or {}).get('minimum_m'),
            reported_contact_points=(case.get('contacts') or {}).get('reported_points'),
            maximum_observed_penetration_m=(severity.get('depth_m') or {}).get('maximum'),
            maximum_observed_normal_pair_impulse_ns=(severity.get('normal_pair_impulse_ns') or {}).get('maximum'),
            artifact=case.get('artifact')))
    return dict(path=path_label(path), sha256=hashlib.sha256(raw_bytes).hexdigest(), cases=rows,
        scope='Historical V2b ground-truth occupancy reference only; not newly measured camera results and never RGB-D runtime augmentation.')


def summarize(paths, *, retained_trials=(), ground_truth_reference=None):
    rows, seen, duplicates = [], set(), []
    for cohort, values in [('primary', paths), ('retained_trial', retained_trials)]:
        for value in values:
            path = Path(value)
            path = path/'result.json' if path.is_dir() else path
            identity = str(path.resolve())
            if identity in seen:
                duplicates.append(path_label(path))
                continue
            seen.add(identity)
            rows.append(load_case(path, cohort=cohort))
    primary = [row for row in rows if row['cohort'] == 'primary']
    valid = [row for row in primary if row['measurement_valid']]
    reactive = [row for row in valid if row['strategy'] == 'reactive']
    physical = ('NOT_MEASURED' if not reactive else 'PARTIAL_FAILURE' if any(
        row['strict_contact_free_success'] is not True for row in reactive) else
        'INCOMPLETE' if len(valid) != len(primary) else 'PASS')
    return dict(schema_version=1, status='RESEARCH_SUMMARY',
        measurement_status='NOT_MEASURED' if not primary else 'PASS' if len(valid) == len(primary) else 'PARTIAL_FAILURE',
        physical_assessment_status=physical,
        meaning='Valid measurement and strict contact-free target completion are separate. Direct controls may intentionally expose obstacle-unaware contacts; no severity threshold is invented.',
        counts=dict(primary_attempts=len(primary), valid_measurements=len(valid),
            invalid_measurements=len(primary)-len(valid), retained_trials=len(rows)-len(primary),
            reactive_valid=len(reactive), reactive_goal_reached=sum(row['goal_reached'] is True for row in reactive),
            reactive_strict_successes=sum(row['strict_contact_free_success'] is True for row in reactive)),
        cases=rows, paired_comparisons=comparisons(rows), duplicate_requested_artifacts=duplicates,
        historical_ground_truth_reference=ground_truth_context(ground_truth_reference),
        limitations=[
            'No failed attempt is silently removed. Explicit calibration/preparation trials remain labelled separately from the primary comparisons.',
            'Multiple attempts in one matching group remain uncollapsed; the reducer never chooses the fastest or successful attempt.',
            'Missing metrics remain null. Direct motion has no registered obstacle product, so its proxy fields are N/A rather than zero sensing evidence.',
            'Cooked-link clearance is sampled unsigned surface distance; it does not certify continuous clearance or include PhysX contact offsets.',
            'Reported contact points are sampled reports, not unique impacts. Raw penetration and normal impulse do not establish harmlessness.',
            'Consumed map-age distributions apply only to reactive consumers; direct final map age is an endpoint observation, not a worst-age claim.',
            'Ground-truth results are historical evaluation context and are never merged with new camera observations.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True, help='Explicit case directories or result.json files')
    parser.add_argument('--retained-trials', nargs='*', default=[])
    parser.add_argument('--ground-truth-reference')
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    result = summarize(args.cases, retained_trials=args.retained_trials,
        ground_truth_reference=args.ground_truth_reference)
    output = Path(args.result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(measurement_status=result['measurement_status'],
        physical_assessment_status=result['physical_assessment_status'], counts=result['counts'])))


if __name__ == '__main__':
    main()
