#!/usr/bin/env python3
"""Compact every V4 attempt and plot recorded routes, execution and contacts.

This reducer starts no simulator and evaluates no controller.  A comparison
exists only when the candidate's launch explicitly names its reference result.
Raw joint trajectories, meshes and contact samples remain in ignored outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('direct', 'reactive', 'global')
QUANTILES = ('count', 'mean', 'p50', 'p95', 'p99', 'worst', 'maximum')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def quantiles(value):
    return {key: value[key] for key in QUANTILES if key in value} if isinstance(value, dict) else None


def declaration(folder):
    path = Path(folder)/'launch.json'
    if not path.exists():
        return dict(reference=None, witness=None, error=None,
            scope='No driver launch record; no paired reference is inferred')
    try:
        raw = read(path)
        command = raw.get('command')
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise ValueError('Launch command must be an exact string argument vector')
        indices = [i for i, item in enumerate(command) if item == '--reference-result']
        if len(indices) > 1 or indices and indices[0]+1 >= len(command):
            raise ValueError('Ambiguous reference-result argument')
        reference = None
        if indices:
            reference = Path(command[indices[0]+1])
            if not reference.is_absolute():
                reference = Path(raw.get('cwd') or ROOT)/reference
            reference = label(reference)
        return dict(reference=reference, witness='--witness' in command, error=None,
            launch_sha256=digest(path))
    except (OSError, ValueError, TypeError) as error:
        return dict(reference=None, witness=None, error=str(error))


def _artifact(path):
    path = Path(path)
    return (path/'result.json' if path.is_dir() else path).resolve()


def compact_lifecycle(value, *, artifact, artifact_sha256):
    """Keep episode outcomes, never copy recorded joint-action trajectories."""
    if not value:
        return None
    episodes = []
    for index, episode in enumerate(value.get('episodes') or []):
        result = episode.get('result') or {}
        details = dict(result.get('details') or {})
        details.update(episode.get('strategy') or {})
        actions = episode.get('actions')
        episodes.append(dict(index=index, name=episode.get('name'),
            result_status=result.get('status'), backend_ok=result.get('ok'),
            public_failure_code=result.get('failure_code'),
            internal_failure_code=details.get('failure_code'),
            message=result.get('message') or details.get('failure_message'),
            strategy_state=details.get('state'), strategy_closed=details.get('closed'),
            recorded_action_count=len(actions) if isinstance(actions, list) else None,
            timing={key: details[key] for key in ('planning_and_validation_wall_seconds',
                'rrt_seconds', 'trajectory_generation_seconds', 'trajectory_duration_seconds',
                'execution_wall_seconds', 'execution_sim_seconds') if key in details},
            flags={key: item for key, item in episode.items() if isinstance(item, bool)}))
    return dict(status=value.get('status'), checks=value.get('checks'), episodes=episodes,
        excluded_from_primary_timing=value.get('excluded_from_primary_timing'),
        limitation=value.get('limitation'),
        source=dict(artifact=artifact, sha256=artifact_sha256),
        scope='Recorded episode outcomes/checks and action counts only; raw action targets and verbose plan traces remain in the hashed ignored native result.')


def load_attempt(path):
    path = _artifact(path)
    errors, raw = [], {}
    try:
        raw = read(path)
        if not isinstance(raw, dict):
            raise ValueError('Native result must be a JSON object')
        result_sha = digest(path)
    except (OSError, ValueError, TypeError) as error:
        errors.append(type(error).__name__+': '+str(error))
        result_sha = None
        raw = {}
    launch = declaration(path.parent)
    if launch.get('error'):
        errors.append(launch['error'])
    metrics = raw.get('metrics') or {}
    config = metrics.get('configuration') or {}
    contracts = raw.get('contracts') or {}
    valid = (raw.get('status') == 'PASS' and bool(contracts)
        and all(value is True for value in contracts.values()) and not raw.get('error') and not errors)
    match = metrics.get('reference_match') or {}
    if launch.get('reference') is not None:
        valid = (valid and match.get('matched') is True and bool(match.get('checks'))
            and all(value is True for value in match['checks'].values()))
    outcome = (metrics.get('motion') or {}).get('result') or {}
    details = outcome.get('details') or {}
    diagnostics = metrics.get('strategy_diagnostics') or {}
    planning = dict(details)
    planning.update(diagnostics)
    attempts = planning.get('ik_attempts') or []
    rrt_seconds = number(planning.get('rrt_seconds'))
    if rrt_seconds is None:
        measured = [number(item.get('rrt_seconds')) for item in attempts]
        rrt_seconds = sum(value for value in measured if value is not None) if any(value is not None for value in measured) else None
    contacts, severity = metrics.get('contacts') or {}, metrics.get('contact_severity') or {}
    clearance = metrics.get('clearance') or {}
    witness = bool(config.get('witness') or metrics.get('witness') or launch.get('witness'))
    lifecycle = metrics.get('lifecycle') or {}
    mode = 'witness' if witness else 'primary'
    cohort = ('witness' if witness else 'lifecycle' if lifecycle else
        'pilot' if 'pilot' in path.parts else 'seeds' if path.parent.parent.name == 'seeds' else 'primary')
    signature = dict(source_manifest_sha256=fingerprint(raw['hashes']) if raw.get('hashes') else None,
        physical_world_sha256=fingerprint(metrics['physical_world']) if metrics.get('physical_world') is not None else None,
        scenario=config.get('scenario'), start=config.get('start'), goal=config.get('goal'),
        orientation=config.get('orientation'), objects=config.get('objects'),
        position_tolerance_m=config.get('position_tolerance_m'),
        orientation_tolerance_rad=config.get('orientation_tolerance_rad'),
        max_motion_steps=config.get('max_motion_steps'))
    row = dict(id=path.parent.name, artifact=label(path), artifact_sha256=result_sha,
        measurement_status=raw.get('status', 'UNAVAILABLE'), measurement_valid=bool(valid),
        scenario=config.get('scenario'), strategy=config.get('strategy'), seed=config.get('seed'),
        max_iterations=config.get('max_iterations'), mode=mode, cohort=cohort,
        initial_joints=metrics.get('initial_joints'),
        declared_reference=launch.get('reference'), launch_sha256=launch.get('launch_sha256'),
        reference_runtime_match=match.get('matched'),
        initial_q_maximum_error_rad=match.get('maximum_joint_error_rad'),
        initial_q_tolerance_rad=match.get('joint_tolerance_rad'),
        goal_status=outcome.get('status'), reported_goal_reached=metrics.get('goal_reached'),
        strict_success=metrics.get('strict_success') if valid else None,
        movable_arm_strict_success=metrics.get('movable_arm_strict_success') if valid else None,
        mounted_base_qualification=metrics.get('mounted_base_qualification'),
        failure_code=planning.get('failure_code') or outcome.get('failure_code'),
        failure_message=planning.get('failure_message') or outcome.get('message'),
        classification=metrics.get('classification'),
        final_pose_errors=metrics.get('final_pose_errors'),
        motion_wall_seconds=number((metrics.get('motion') or {}).get('wall_seconds')),
        planning_wall_seconds=number(planning.get('planning_and_validation_wall_seconds')),
        rrt_seconds=rrt_seconds,
        ik_candidates_attempted=planning.get('ik_candidates_attempted'),
        ik_attempt_statuses=[{key: item[key] for key in ('index', 'ik_success', 'status', 'ik_seconds',
            'rrt_seconds', 'route_validation_samples', 'trajectory_generation_seconds',
            'trajectory_validation_seconds') if key in item} for item in attempts],
        rrt_waypoints=planning.get('rrt_waypoints'), geometric_joint_path_length=planning.get('geometric_joint_path_length'),
        static_obstacles=planning.get('static_obstacles'), world_update_seconds=planning.get('world_update_seconds'),
        trajectory_generation_seconds=planning.get('trajectory_generation_seconds'),
        trajectory_duration_seconds=planning.get('trajectory_duration_seconds', planning.get('trajectory_duration')),
        trajectory_validation=planning.get('trajectory_validation'),
        execution_wall_seconds=planning.get('execution_wall_seconds'),
        execution_sim_seconds=planning.get('execution_sim_seconds'),
        execution_uses_physics_time=planning.get('execution_uses_physics_time'),
        planning_blocks_owner_loop=planning.get('planning_blocks_owner_loop'),
        static_snapshot_refreshes=planning.get('static_snapshot_refreshes'),
        joint_tracking_error_rad=quantiles(planning.get('joint_tracking_error_rad')),
        joint_path_deviation=metrics.get('joint_path_deviation'),
        final_hold=planning.get('final_hold'), commanded_updates=planning.get('commanded_updates'),
        no_requested_movement_joint_delta_rad=metrics.get('no_requested_movement_joint_delta_rad'),
        tcp_path=metrics.get('tcp_path'),
        control_interval_ms=quantiles(metrics.get('control_interval_ms')),
        kit_interval_ms=quantiles(metrics.get('kit_interval_ms')),
        controller_wait_ms=quantiles(metrics.get('controller_wait_ms')),
        controller_waits_over_100ms=metrics.get('controller_waits_over_100ms'),
        minimum_all_link_clearance_m=clearance.get('minimum_m'),
        minimum_movable_arm_clearance_m=clearance.get('movable_arm_minimum_m'),
        clearance_by_link_m=clearance.get('by_link_m'), clearance_by_group_m=clearance.get('by_group_m'),
        clearance_by_obstacle_m=clearance.get('by_obstacle_m'), closest_pair=clearance.get('closest_pair'),
        reported_contact_points=contacts.get('reported_points'),
        penetrating_contact_pairs=contacts.get('penetrating_pairs'),
        minimum_reported_separation_m=contacts.get('minimum_reported_separation_m'),
        reported_contact_links=contacts.get('links'), mounted_base_excluded=contacts.get('mounted_base_excluded'),
        observed_physical_contact_steps=severity.get('observed_physical_contact_steps'),
        maximum_observed_penetration_m=(severity.get('depth_m') or {}).get('maximum'),
        maximum_observed_normal_pair_impulse_ns=(severity.get('normal_pair_impulse_ns') or {}).get('maximum'),
        sampled_physical_contact_seconds=severity.get('sampled_physical_contact_duration_s'),
        physical_contact_classification=severity.get('physical_classification'),
        contact_acceptability='UNVALIDATED_NO_THRESHOLD',
        lifecycle=compact_lifecycle(lifecycle, artifact=label(path), artifact_sha256=result_sha),
        witness_segments=metrics.get('witness_segments') or None,
        witness_protocol_sha256=config.get('witness_protocol_sha256'),
        source_unchanged=contracts.get('source_unchanged'),
        snapshot=metrics.get('physical_world'), errors=errors or ([raw['error']] if raw.get('error') else None))
    artifacts = {}
    for filename in ('trace.json', 'plan_trace.json'):
        location = path.parent/filename
        if location.exists():
            artifacts[filename] = dict(path=label(location), sha256=digest(location))
    row['raw_artifacts'] = artifacts
    row['global_trajectory_validation_passed'] = (row['strategy'] == 'global' and
        (row.get('trajectory_validation') or {}).get('collision_validation') == 'PASS_SAMPLED_SDK_MODEL')
    return row, signature, contracts


def comparison(candidate, reference, signatures):
    checks = dict(reference_included=reference is not None,
        candidate_valid=candidate['measurement_valid'],
        reference_valid=bool(reference and reference['measurement_valid']),
        runtime_guard=candidate['reference_runtime_match'] is True)
    if reference is not None:
        first, last = reference.get('initial_joints') or [], candidate.get('initial_joints') or []
        tolerance = number(candidate.get('initial_q_tolerance_rad'))
        compatible = (len(first) == len(last) and len(first) >= 7 and
            all(number(value) is not None for value in first+last))
        q_error = max(abs(a-b) for a, b in zip(first, last)) if compatible else None
        checks.update(physical_and_source_conditions=signatures[candidate['condition_fingerprint']] ==
            signatures[reference['condition_fingerprint']],
            same_mode=candidate['mode'] == reference['mode'],
            actual_start_joints=q_error is not None and tolerance is not None and tolerance > 0 and q_error <= tolerance)
    eligible = all(checks.values())
    delta = None
    if eligible and candidate['motion_wall_seconds'] is not None and reference['motion_wall_seconds'] is not None:
        delta = candidate['motion_wall_seconds']-reference['motion_wall_seconds']
    return dict(candidate=candidate['artifact'], reference=candidate['declared_reference'],
        reference_sha256=reference['artifact_sha256'] if reference else None,
        comparison_eligible=eligible, checks=checks,
        strict_success=[reference['strict_success'], candidate['strict_success']] if reference else None,
        motion_wall_seconds_delta=delta,
        meaning='One declared edge, not an independent baseline replicate; strategies and RRT seeds may differ.')


def summarize(paths):
    rows, conditions, contract_sets, seen, duplicates = [], {}, {}, set(), []
    for value in paths:
        identity = str(_artifact(value))
        if identity in seen:
            duplicates.append(label(identity))
            continue
        seen.add(identity)
        row, signature, contracts = load_attempt(value)
        source_id, contract_id = fingerprint(signature)[:20], fingerprint(contracts)[:20]
        if source_id in conditions and conditions[source_id] != signature:
            raise ValueError('Unexpected condition fingerprint collision')
        conditions[source_id], contract_sets[contract_id] = signature, contracts
        row.update(condition_fingerprint=source_id, contract_set=contract_id)
        rows.append(row)
    by_path = {row['artifact']: row for row in rows}
    edges = [comparison(row, by_path.get(row['declared_reference']), conditions)
        for row in rows if row['declared_reference'] is not None]
    counts = dict(unique_attempts=len(rows), valid_measurements=sum(row['measurement_valid'] for row in rows),
        strict_successes=sum(row['strict_success'] is True for row in rows),
        declared_reference_edges=len(edges), eligible_reference_edges=sum(edge['comparison_eligible'] for edge in edges),
        unique_declared_baselines=len({edge['reference'] for edge in edges}))
    cohort_counts = dict(Counter(row['cohort'] for row in rows))
    cohort_outcomes = {}
    for cohort in cohort_counts:
        selected = [row for row in rows if row['cohort'] == cohort]
        cohort_outcomes[cohort] = dict(attempts=len(selected),
            valid_measurements=sum(row['measurement_valid'] for row in selected),
            reported_goals_reached=sum(row['reported_goal_reached'] is True for row in selected),
            valid_goal_completions=sum(row['reported_goal_reached'] is True and row['measurement_valid'] for row in selected),
            strict_successes=sum(row['strict_success'] is True for row in selected))
    return dict(schema_version=1, status='COMPACT_NATIVE_V4_EVIDENCE', counts=counts,
        cohort_counts=cohort_counts, cohort_outcomes=cohort_outcomes,
        modes=dict(Counter(row['mode'] for row in rows)),
        strategy_attempts=dict(Counter(row['strategy'] or 'unknown' for row in rows)),
        cases=rows, declared_comparisons=edges, condition_fingerprints=conditions, contract_sets=contract_sets,
        duplicate_requested_artifacts=duplicates,
        scope=['Every distinct requested attempt is retained, including failed seeds and invalid measurements.',
            'Process exit/status alone does not establish motion or collision-free success.',
            'Only explicit launch reference-result edges are paired; repeats do not form Cartesian joins.',
            'Explicit cohorts, not mode labels, define denominators; the original request inside a lifecycle artifact is preserved but excluded from primary comparisons.',
            'Witness/lifecycle/pilot artifacts remain separate from valid primary and additional-seed comparison figures.',
            'Reported proximity, penetration, impulse and sampled geometry remain separate, without a safe-contact threshold.',
            'Minimum clearance is sampled full-robot cooked geometry, not TCP-only or continuous swept clearance.',
            'Planning and execution timings have different boundaries; no missing duration is replaced by an estimate.'],
        reducer_sha256=digest(__file__))


def _plot_setup():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/motion-planning-v4/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _load_bound_artifact(row, name):
    artifact = row['raw_artifacts'].get(name)
    if artifact is None:
        return None
    path = Path(artifact['path'])
    path = path if path.is_absolute() else ROOT/path
    if digest(path) != artifact['sha256']:
        raise ValueError('Raw plot source changed after summary: '+str(path))
    return read(path)


def _box_edges(item):
    import numpy as np
    from robot_skill_stack.common.rotations import quat_matrix
    if item.get('kind') != 'cuboid' or item.get('source_kind') == 'Plane':
        return []
    half = np.asarray(item['dimensions'])/2
    from itertools import product
    corners = np.asarray(list(product((-1, 1), repeat=3)))*half
    corners = corners@quat_matrix(item['orientation']).T+np.asarray(item['position'])
    return [corners[[i, j]] for i in range(8) for j in range(i+1, 8)
        if sum(a != b for a, b in zip(list(product((-1, 1), repeat=3))[i],
            list(product((-1, 1), repeat=3))[j])) == 1]


def figures(report, destination):
    """Plot actual stored values only; return compact source/figure bindings."""
    import numpy as np
    plt = _plot_setup()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = report['cases']
    colors = {'direct': '#777777', 'reactive': '#e58926', 'global': '#2075b8'}
    bound = []

    def save(figure, name, selected, caption):
        path = destination/name
        figure.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(figure)
        bound.append(dict(path=label(path), sha256=digest(path), caption=caption,
            sources=[dict(artifact=row['artifact'], sha256=row['artifact_sha256'],
                raw_artifacts=row['raw_artifacts']) for row in selected]))

    primary = [row for row in rows if row['cohort'] in ('primary', 'seeds') and row['measurement_valid']]
    scenes = list(dict.fromkeys(row['scenario'] for row in primary))
    if scenes:
        figure = plt.figure(figsize=(15, 5*math.ceil(len(scenes)/3)))
        for panel, scene in enumerate(scenes, 1):
            axis = figure.add_subplot(math.ceil(len(scenes)/3), 3, panel, projection='3d')
            selected = [row for row in primary if row['scenario'] == scene]
            for row in selected:
                trace = _load_bound_artifact(row, 'trace.json') or {}
                tcp = np.asarray([point['position'] for point in trace.get('tcp', [])], float)
                legend = row['strategy']+(f" seed{row['seed']}" if row['strategy'] == 'global' else '')
                if tcp.ndim == 2 and len(tcp):
                    axis.plot(*tcp.T, color=colors.get(row['strategy']), alpha=.8,
                        label=legend+(' ✓' if row['strict_success'] else ' ×'))
                else:
                    axis.plot([], [], [], color=colors.get(row['strategy']), label=legend+' (no recorded TCP)')
            for item in selected[0].get('snapshot') or []:
                for edge in _box_edges(item):
                    axis.plot(*edge.T, color='#555555', alpha=.45, lw=.8)
            axis.set(title=scene, xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
            axis.legend(fontsize=7)
        figure.suptitle('Recorded primary and additional-seed TCP paths with actual collision boxes\n✓ strict success; × failure or non-strict outcome; all valid requested seeds retained')
        save(figure, 'tcp_trajectories.png', primary,
            'Every valid primary/additional-seed TCP trace; fixed physical boxes are drawn from the matched snapshot. '
            'No planned Cartesian path is invented. Curves are sampled TCP motion, not whole-arm swept clearance.')

    globals_ = [row for row in primary if row['strategy'] == 'global']
    representative = None
    for row in sorted(globals_, key=lambda item: (item['scenario'] not in ('large', 'static', 'narrow'),
            str(item['scenario']), item['seed'] if isinstance(item['seed'], int) else math.inf, item['id'])):
        trace = _load_bound_artifact(row, 'plan_trace.json')
        if trace and trace.get('route') and trace.get('trajectory') and trace.get('actual'):
            representative = row, trace
            break
    if representative is not None:
        row, trace = representative
        route = np.asarray(trace['route'], float)
        joint_names = trace.get('joint_names') or [f'Arm index {index}' for index in range(7)]
        figure, axis = plt.subplots(figsize=(10, 4.5))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        for joint in range(7):
            axis.plot(arc, route[:, joint], '.-', label=joint_names[joint])
        axis.set(xlabel='RRT joint-polyline arc length (Euclidean radians)', ylabel='Joint position (rad)',
            title=f"Recorded seven-joint RRT route: {row['id']}")
        axis.legend(ncol=4)
        save(figure, 'joint_route.png', [row],
            'First available recorded route in deterministic scene/seed order (prefer obstructed scenes). '
            'All seven RRT joint configurations are preserved; the axis is geometric joint-space arc length, not time.')
        figure, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        sampled = trace['trajectory']
        times = np.asarray([item['time'] for item in sampled])
        planned = np.asarray([item['q'] for item in sampled])
        measured = trace['actual']
        origin = measured[0]['sim_time']
        actual_times = np.asarray([item['sim_time']-origin for item in measured])
        actual = np.asarray([item['q'] for item in measured])
        for joint, axis in enumerate(axes.flat[:7]):
            axis.plot(times, planned[:, joint], color='#2075b8', label='Generated/validated plan')
            axis.plot(actual_times, actual[:, joint], '--', color='#bf5928', label='Actual articulation')
            axis.set(ylabel=joint_names[joint]+' (rad)', xlabel='Physics time from execution start (s)')
        axes.flat[0].legend(fontsize=8)
        axes.flat[-1].axis('off')
        figure.suptitle('Generated joint trajectory versus actual measured articulation\n'+row['id'])
        figure.tight_layout()
        save(figure, 'planned_vs_actual_joints.png', [row],
            'Generated plan samples and actual q from the same attempt. Time uses measured simulation timestamps, '
            'not rendered update count; traces are neither retimed nor projected to hide tracking error.')

    if primary:
        names = [f"{row['scenario']}\n{row['strategy']}"+(f"/{row['seed']}" if row['strategy'] == 'global' else '') for row in primary]
        x = np.arange(len(primary))
        figure, axes = plt.subplots(2, 1, figsize=(max(11, len(primary)*.45), 8), sharex=True)
        clearances = [np.nan if number(row['minimum_movable_arm_clearance_m']) is None else row['minimum_movable_arm_clearance_m']*1000 for row in primary]
        axes[0].bar(x, clearances, color=[colors.get(row['strategy'], '#555555') for row in primary])
        axes[0].set(ylabel='Minimum movable-arm clearance (mm)', title='Sampled cooked whole-arm clearance and separate contact evidence')
        axes[1].bar(x-.18, [np.nan if number(row.get('reported_contact_points')) is None else row['reported_contact_points'] for row in primary], width=.35, label='Reported points (includes proximity)')
        axes[1].bar(x+.18, [np.nan if number(row.get('observed_physical_contact_steps')) is None else row['observed_physical_contact_steps'] for row in primary], width=.35, label='Observed physical-contact steps')
        axes[1].set(yscale='symlog', ylabel='Recorded count (symlog)', xticks=x, xticklabels=names)
        axes[1].tick_params(axis='x', labelrotation=60, labelsize=7)
        axes[1].legend()
        figure.tight_layout()
        save(figure, 'clearance_and_contacts.png', primary,
            'Movable-arm clearance excludes the mounted base in this panel; all-link/base minima remain in JSON. '
            'Reported points and physical-contact steps have distinct definitions. Missing counts are annotated as unavailable in the compact table.')
        figure, axes = plt.subplots(3, 1, figsize=(max(11, len(primary)*.45), 10), sharex=True)
        missing = lambda key: [np.nan if number(row.get(key)) is None else row[key] for row in primary]
        axes[0].bar(x, missing('planning_wall_seconds'), width=.6, label='Planning + validation wall time')
        axes[0].plot(x, missing('rrt_seconds'), 'k.', label='RRT search only')
        axes[0].set(ylabel='Measured wall time (s)', title='Planning and execution have separate measured timing boundaries')
        axes[0].legend(fontsize=8)
        axes[1].bar(x-.18, missing('motion_wall_seconds'), width=.35, label='Whole movement request (includes global planning)')
        axes[1].bar(x+.18, missing('execution_wall_seconds'), width=.35, label='Global trajectory execution only')
        axes[1].set(ylabel='Measured wall time (s)')
        axes[1].legend(fontsize=8)
        axes[2].bar(x, [(row.get('kit_interval_ms') or {}).get('worst', np.nan) for row in primary])
        axes[2].set(ylabel='Worst recorded Kit interval (ms)', xticks=x, xticklabels=names)
        axes[2].tick_params(axis='x', labelrotation=60, labelsize=7)
        figure.tight_layout()
        save(figure, 'planning_and_execution_timing.png', primary,
            'Recorded whole-request, planning-and-validation, execution and RRT-call wall durations; '
            'direct/reactive request durations include their controller work and global request durations also include preplanning. Missing values are absent, '
            'not zero estimates. Worst Kit interval captures owner-loop responsiveness and is not a planning deadline guarantee.')
    return dict(schema_version=1, reducer_sha256=digest(__file__), figures=bound,
        selection='All valid primary and additional-seed attempts in TCP/contact/timing panels; lifecycle/witness/pilot and invalid attempts remain in compact JSON. One deterministic available route in joint panels.',
        limitations='Figures use recorded raw artifacts bound by SHA256. No new native execution, collision query, IK, FK, or timing benchmark is performed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--figures-dir', type=Path)
    parser.add_argument('--max-bytes', type=int, default=350_000)
    args = parser.parse_args(argv)
    report = summarize(args.cases)
    encoded = (json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if len(encoded) > args.max_bytes:
        raise ValueError(f'Compact evidence is {len(encoded)} bytes, above {args.max_bytes}; no attempts were removed')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_bytes(encoded)
    if args.figures_dir:
        metadata = figures(report, args.figures_dir)
        (args.figures_dir/'figures.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(bytes=len(encoded), **report['counts'])))


if __name__ == '__main__':
    main()
