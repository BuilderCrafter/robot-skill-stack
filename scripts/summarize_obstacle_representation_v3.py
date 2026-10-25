#!/usr/bin/env python3
"""Compact V3 native and static-policy evidence without selecting good attempts.

This SDK-free reader keeps measurement validity, physical outcomes, captures,
timing eligibility and historical controls separate. Plot imports are deferred.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.summarize_camera_observability_planning import load_case as physical_case
from scripts.summarize_camera_observability_planning import number, path_label


FIELDS = ('motion_seconds', 'minimum_cooked_link_clearance_m',
    'maximum_observed_penetration_m', 'maximum_observed_normal_pair_impulse_ns',
    'controller_interval_p99_ms', 'controller_interval_worst_ms', 'consumed_map_age_worst_ms',
    'path_length_m', 'path_length_over_straight', 'maximum_x_detour_m', 'maximum_z_detour_m',
    'longest_low_progress_seconds', 'registry_sync_p99_ms', 'registry_update_p99_ms',
    'worker_compute_p95_ms', 'worker_queue_p95_ms', 'worker_transport_p95_ms')


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path)
    return dict(path=path_label(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def distribution(values):
    values = sorted(value for item in values if (value := number(item)) is not None)
    if not values:
        return dict(count=0)
    def percentile(fraction):
        index = (len(values)-1)*fraction
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(count=len(values), minimum=values[0], median=statistics.median(values),
        mean=statistics.fmean(values), p50=percentile(.5), p95=percentile(.95),
        p99=percentile(.99), worst=values[-1])


def equal_numeric(first, second, tolerance=1e-8):
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(equal_numeric(a, b, tolerance) for a, b in zip(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(equal_numeric(first[key], second[key], tolerance) for key in first)
    a, b = number(first), number(second)
    return abs(a-b) <= tolerance if a is not None and b is not None else first == second


def observed_worker_latency(trace):
    """Deduplicate sparse owner observations; never claim all worker publications."""
    observed = {}
    for row in trace.get('runtime') or []:
        mapping = row.get('mapping') or {}
        key = mapping.get('input_timestamp')
        if key is not None:
            observed[key] = mapping
    fields = dict(compute='compute_seconds', queue='queue_seconds', transport='transport_seconds',
        worker='worker_seconds', publication='worker_to_publication_seconds',
        original_publish_age='publish_age_seconds', robot_filter='filter_seconds')
    return dict(scope='Distinct completed source jobs observed by the sparse owner runtime sampler; not all worker publications.',
        source_jobs=len(observed), **{name+'_ms': distribution(
            number(row.get(key))*1000 if number(row.get(key)) is not None else None
            for row in observed.values()) for name, key in fields.items()})


def calibration_signature(calibration):
    """Keep physical calibration, excluding frame tokens and run clocks."""
    if not isinstance(calibration, dict):
        return None
    keys = ('camera_id', 'intrinsics', 'world_from_camera_ros', 'resolution')
    result = {}
    for camera, row in calibration.items():
        if not isinstance(row, dict) or any(key not in row for key in keys[1:]):
            return None
        result[camera] = {key: row.get(key, camera) for key in keys}
    return result


def load_case(path, *, cohort='primary'):
    path = Path(path)
    path = path/'result.json' if path.is_dir() else path
    row = physical_case(path, cohort=cohort)
    try:
        raw = read(path)
    except (OSError, ValueError):
        raw = {}
    metrics, representation = raw.get('metrics') or {}, raw.get('representation_v3') or {}
    config, research = metrics.get('configuration') or {}, raw.get('research') or {}
    registry, progress = representation.get('registry') or {}, representation.get('progress') or {}
    activation = representation.get('activation') or {}
    guard = activation.get('guard') or {}
    inherited = cohort == 'reused_direct'
    specific_valid = bool(representation) and bool(guard) and all(value is True for value in guard.values())
    specific_valid = specific_valid and representation.get('production_sources_unchanged') is True
    specific_valid = specific_valid and representation.get('no_wait_for_mapping_during_motion') is True
    if row['strategy'] == 'reactive':
        specific_valid = specific_valid and registry.get('owner_thread_all') is True
    row['measurement_valid'] = row['measurement_valid'] and (specific_valid or inherited)
    row['scenario'] = representation.get('scenario', row['scenario'])
    row.update(candidate_id=representation.get('candidate_id', 'historical_direct' if inherited else None),
        packing=representation.get('packing'), selection=representation.get('selection'), budget=representation.get('budget'),
        fixture_kind=representation.get('fixture_kind'), representation_metadata=representation,
        activation_guard=guard, initial_condition_match=activation.get('initial_condition_match'),
        initial_joints=metrics.get('initial_joints'), fixture_setup=metrics.get('fixture_setup'),
        timing_eligible=row['measurement_valid'] and cohort == 'primary' and bool(representation)
            and representation.get('full_capture_excluded_from_timing_claims') is False,
        source_protocol=dict(mapping_source=config.get('mapping_source'), mapping_hz=config.get('mapping_hz'),
            global_rays=research.get('global_ray_budget'), sampling_policy=research.get('sampling_policy'),
            start=config.get('start'), goal=config.get('goal'), orientation=config.get('orientation'),
            voxel_size=config.get('voxel_size'), physical_objects=config.get('objects'),
            backend_settings=metrics.get('backend_settings'),
            actual_camera_calibration=calibration_signature(research.get('actual_camera_calibration')),
            production_source_hashes=raw.get('hashes'), original_harness_sha256=representation.get('original_harness_sha256')),
        registry=registry, path=metrics.get('path'), progress=progress,
        contact_severity=metrics.get('contact_severity'), cylinder_reference=metrics.get('cylinder_reference'))
    if not row['measurement_valid']:
        row['goal_reached'] = row['strict_contact_free_success'] = None
    path_metrics = metrics.get('path') or {}
    length, straight = number(path_metrics.get('length_m')), number(path_metrics.get('straight_distance_m'))
    row.update(path_length_m=length, path_length_over_straight=None if length is None or not straight else length/straight,
        maximum_x_detour_m=number(path_metrics.get('maximum_x_detour_m')),
        maximum_z_detour_m=number(path_metrics.get('maximum_z_detour_m')),
        progress_reversals=path_metrics.get('progress_reversals'),
        longest_low_progress_seconds=number(progress.get('longest_low_progress_seconds')),
        registry_sync_p99_ms=number((registry.get('sync_ms') or {}).get('p99')),
        registry_update_p99_ms=number((registry.get('actual_update_ms') or {}).get('p99')))
    try:
        trace = read(path.parent/'trace.json')
        row['trace_artifact'] = artifact(path.parent/'trace.json')
    except (OSError, ValueError):
        trace = {}
        row['trace_artifact'] = None
    row['observed_worker_latency'] = observed_worker_latency(trace)
    for name in ('compute', 'queue', 'transport'):
        row['worker_'+name+'_p95_ms'] = number((row['observed_worker_latency'].get(name+'_ms') or {}).get('p95'))
    updates = trace.get('sdk_updates') or []
    revisions = [(item.get('generation'), item.get('revision')) for item in updates]
    row['per_step_packet_checks'] = dict(control_observations=len(updates),
        distinct_revisions=len(set(revisions)), consecutive_same_revision_steps=sum(a == b for a, b in zip(revisions, revisions[1:])),
        original_source_age_ms=distribution(number(item.get('map_age_seconds'))*1000
            if number(item.get('map_age_seconds')) is not None else None for item in updates),
        scope='Repeated consumption of a still-fresh revision between 2Hz publications is expected; it is not a rejected/repeated source job.')
    final_diagnostics = (metrics.get('final') or {}).get('mapping_diagnostics') or {}
    row['final_extraction'] = final_diagnostics.get('obstacle_extraction')
    return row


def match_conditions(baseline, candidate):
    q, r = baseline.get('initial_joints'), candidate.get('initial_joints')
    maximum = max((abs(a-b) for a, b in zip(q, r)), default=None) if q and r and len(q) == len(r) else None
    first, last = baseline.get('source_protocol') or {}, candidate.get('source_protocol') or {}
    required = ('mapping_source', 'mapping_hz', 'global_rays', 'sampling_policy', 'start', 'goal',
        'orientation', 'voxel_size', 'physical_objects', 'backend_settings', 'actual_camera_calibration',
        'production_source_hashes', 'original_harness_sha256')
    checks = {key: first.get(key) is not None and last.get(key) is not None
        and equal_numeric(first[key], last[key]) for key in required}
    checks['initial_joints'] = maximum is not None and maximum <= .01
    before = (baseline.get('fixture_setup') or {}).get('actual_actor_poses')
    after = (candidate.get('fixture_setup') or {}).get('actual_actor_poses')
    if baseline.get('scenario') == candidate.get('scenario') == 'empty':
        before, after = before or [], after or []
    checks['actual_fixture_poses'] = before is not None and after is not None and equal_numeric(before, after, 1e-4)
    return dict(matched=all(checks.values()), checks=checks, maximum_initial_joint_error_rad=maximum,
        joint_tolerance_rad=.01, fixture_pose_component_tolerance=1e-4)


def native_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['cohort'] == 'primary':
            grouped[(row['layout_id'], row['scenario'], row['candidate_id'], row['budget'], row['strategy'])].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        valid = [row for row in values if row['measurement_valid']]
        timed = [row for row in valid if row['timing_eligible']]
        result.append(dict(layout_id=key[0], scenario=key[1], candidate_id=key[2], budget=key[3], strategy=key[4],
            attempts=len(values), valid_measurements=len(valid), timing_eligible_runs=len(timed),
            goal_reached=sum(row['goal_reached'] is True for row in valid),
            strict_contact_free_success=sum(row['strict_contact_free_success'] is True for row in valid),
            case_ids=[row['case_id'] for row in values],
            physical_metrics={name: distribution(row.get(name) for row in valid) for name in FIELDS[:4]+FIELDS[7:12]},
            runtime_metrics={name: distribution(row.get(name) for row in timed) for name in FIELDS[4:7]+FIELDS[12:]},
            aggregation='Median/min/max and distribution of per-run metrics; no pooling of control samples across repeats.'))
    return result


def paired_comparisons(rows):
    result = []
    for candidate in rows:
        if candidate['cohort'] != 'primary' or candidate['candidate_id'] in ('baseline', None):
            continue
        baselines = [row for row in rows if row['cohort'] == 'primary' and row['candidate_id'] == 'baseline'
            and (row['layout_id'], row['scenario'], row['strategy']) ==
            (candidate['layout_id'], candidate['scenario'], candidate['strategy'])]
        # Retain all pairwise repeat relationships; never select a favorable baseline.
        if not baselines:
            result.append(dict(candidate=candidate['case_id'], status='UNPAIRED_NO_BASELINE'))
        for baseline in baselines:
            match = match_conditions(baseline, candidate)
            entry = dict(baseline=baseline['case_id'], candidate=candidate['case_id'], initial_conditions=match,
                status='MATCHED' if match['matched'] and baseline['measurement_valid'] and candidate['measurement_valid'] else 'UNPROVEN_OR_INVALID')
            entry['deltas'] = {name: None if number(baseline.get(name)) is None or number(candidate.get(name)) is None
                else candidate[name]-baseline[name] for name in FIELDS}
            entry['goal_reached'] = dict(baseline=baseline['goal_reached'], candidate=candidate['goal_reached'])
            entry['strict_contact_free_success'] = dict(baseline=baseline['strict_contact_free_success'], candidate=candidate['strict_contact_free_success'])
            entry['timing_comparison_eligible'] = match['matched'] and baseline['timing_eligible'] and candidate['timing_eligible']
            result.append(entry)
    return result


def summarize_probe(path):
    if path is None:
        return None
    raw = read(path)
    groups = defaultdict(list)
    for row in raw.get('rows') or []:
        groups[(row.get('representation'), row.get('posture'), row.get('perturbation', {}).get('id') == 'nominal')].append(row)
    timing_units = defaultdict(lambda: dict(registration={}, pose_update={}, world_update={}, evaluation=[]))
    for row in raw.get('rows') or []:
        unit = timing_units[row.get('representation')]
        case, displacement = row.get('case_id'), row.get('perturbation', {}).get('id')
        unit['registration'][case] = row.get('registration_ms')
        unit['pose_update'][(case, displacement)] = row.get('numeric_pose_update_ms')
        unit['world_update'][(case, displacement)] = row.get('world_update_ms')
        unit['evaluation'].append(row.get('policy_evaluation_ms'))
    fields = ('registered_obstacles', 'collision_force_norm', 'collision_metric_norm',
        'canonical_acceleration_norm', 'perturbation_acceleration_delta_norm',
        'joint_acceleration_direction_cosine_to_baseline', 'collision_force_direction_cosine_to_baseline',
        'registration_ms', 'numeric_pose_update_ms', 'world_update_ms', 'policy_evaluation_ms')
    return dict(artifact=artifact(path), status=raw.get('status'), rows=len(raw.get('rows') or []),
        selected_case_ids=raw.get('selected_case_ids'), requested_representations=raw.get('requested_representations'),
        missing_packets=raw.get('missing_packets'), base_frame_reference=raw.get('base_frame_reference'),
        selected_union_comparisons=raw.get('selected_union_comparisons'),
        identical_geometry_negative_control=raw.get('identical_geometry_negative_control'),
        postures=raw.get('postures'), source_hashes=raw.get('source_hashes'),
        groups=[dict(representation=key[0], posture=key[1], nominal=key[2], samples=len(values),
            metrics={field: distribution(row.get(field) for row in values) for field in fields})
            for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))],
        overhead=[dict(representation=name, registration_ms=distribution(unit['registration'].values()),
            numeric_pose_update_ms=distribution(unit['pose_update'].values()),
            world_update_ms=distribution(unit['world_update'].values()),
            policy_evaluation_ms=distribution(unit['evaluation']),
            scope='One registration per case/representation; one world/pose update per displacement; one query per posture/displacement. Shared costs repeated in query rows are deduplicated.')
            for name, unit in sorted(timing_units.items())],
        scope='Static installed public Lula policy outputs and query overhead; no physical force units or avoidance success.',
        limitations=raw.get('limitations'))


def summarize(cases, *, retained_trials=(), reused_direct=(), static_probe=None, offline=None):
    rows, seen, duplicates = [], set(), []
    for cohort, paths in [('primary', cases), ('retained_trial', retained_trials), ('reused_direct', reused_direct)]:
        for value in paths:
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
    return dict(schema_version=1, status='RESEARCH_SUMMARY',
        measurement_status='NOT_MEASURED' if not primary else 'PASS' if len(valid) == len(primary) else 'PARTIAL_FAILURE',
        counts=dict(primary_attempts=len(primary), valid_measurements=len(valid),
            invalid_measurements=len(primary)-len(valid), timing_eligible=sum(row['timing_eligible'] for row in primary),
            retained_trials=sum(row['cohort'] == 'retained_trial' for row in rows),
            historical_direct_controls=sum(row['cohort'] == 'reused_direct' for row in rows)),
        cases=rows, groups=native_groups(rows), paired_comparisons=paired_comparisons(rows),
        static_probe=summarize_probe(static_probe), offline_evidence=None if offline is None else artifact(offline),
        duplicate_requested_artifacts=duplicates,
        limitations=['Failed and invalid attempts are retained; missing values stay null.',
            'Full forensic captures may describe outcomes but are excluded from live runtime-cost aggregation.',
            'Pairwise repeat comparisons retain every baseline; no fastest/successful attempt is selected.',
            'Exact proxy cell coverage does not establish missing physical volume coverage or continuous collision avoidance.',
            'Sampled cooked-link clearance is independent of proxies but does not certify continuous separation.',
            'Depth/normal impulse/contact reports have no validated harmless-contact threshold.',
            'Low goal-progress windows can describe detours or orientation correction; they do not prove local minima.',
            'Historical direct controls are context until explicit source, fixture, calibration and actual initial-state matching is proven.',
            'Static Lula natural terms are not physical forces and do not replace measured native outcomes.'])


def plot_summary(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [row for row in report['groups'] if row['strategy'] == 'reactive']
    if not groups:
        return False
    labels = [row['layout_id']+' / '+row['scenario']+'\n'+str(row['candidate_id']) for row in groups]
    figure, axes = plt.subplots(3, 1, figsize=(max(9., len(groups)*.42), 9.), constrained_layout=True)
    quantities = [('motion_seconds', 'Motion duration (s)', False),
        ('minimum_cooked_link_clearance_m', 'Sampled cooked clearance (m)', False),
        ('controller_interval_p99_ms', 'Control interval p99 (ms)', True)]
    for axis, (field, label, timed) in zip(axes, quantities):
        for index, group in enumerate(groups):
            values = (group['runtime_metrics'] if timed else group['physical_metrics']).get(field) or {}
            if values.get('count', 0):
                center = values['median']
                axis.errorbar(index, center, yerr=[[center-values['minimum']], [values['worst']-center]],
                    fmt='o', color='#176a85' if group['candidate_id'] == 'baseline' else '#b65d27', capsize=3)
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.25)
        axis.set_xticks(range(len(labels)), labels if axis is axes[-1] else ['']*len(labels), rotation=90)
    figure.suptitle('Native representations: per-run medians and ranges; full captures excluded from timing')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(figure)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--retained-trials', nargs='*', default=[])
    parser.add_argument('--reused-direct', nargs='*', default=[])
    parser.add_argument('--static-probe')
    parser.add_argument('--offline')
    parser.add_argument('--result', required=True)
    parser.add_argument('--figure')
    args = parser.parse_args(argv)
    report = summarize(args.cases, retained_trials=args.retained_trials,
        reused_direct=args.reused_direct, static_probe=args.static_probe, offline=args.offline)
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.figure:
        plot_summary(report, args.figure)
    print(json.dumps(dict(measurement_status=report['measurement_status'], **report['counts'])))


if __name__ == '__main__':
    main()
