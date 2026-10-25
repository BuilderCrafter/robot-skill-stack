#!/usr/bin/env python3
"""Compact unique native V3 attempts and plot representative actual TCP traces.

The only baseline edge for a candidate is its declared --reference-result in
the sibling launch record. Repeats remain individual attempts. No simulator,
policy, registry or native measurement helper is imported or modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re

from scripts.summarize_obstacle_representation_v3 import (
    calibration_signature, observed_worker_latency)


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('primary', 'repeats', 'direct', 'optional_k', 'retained', 'historical_direct')
QUANTILES = ('count', 'p50', 'p95', 'p99', 'worst')
POLICIES = ('baseline', 'diversity', 'motion')


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def quantiles(value, *, worker=False):
    keys = ('count', 'p50', 'p95', 'worst') if worker else QUANTILES
    return {key: value[key] for key in keys if key in value} if isinstance(value, dict) else None


def known_delta(first, last, key):
    before, after = finite(first.get(key)), finite(last.get(key))
    return None if before is None or after is None or after < before else after-before


def launch(folder):
    try:
        raw = read(folder/'launch.json')
        command = raw.get('command') or []
        if not isinstance(command, list):
            raise ValueError('An exact argument-vector launch record is required')
        values = {}
        for flag in ('--reference-result', '--layout-id', '--scenario', '--strategy', '--candidate-id'):
            indices = [index for index, value in enumerate(command) if value == flag]
            if len(indices) > 1 or indices and indices[0]+1 >= len(command):
                raise ValueError('Ambiguous experiment switch: '+flag)
            values[flag[2:].replace('-', '_')] = command[indices[0]+1] if indices else None
        reference = values.pop('reference_result')
        if reference:
            reference = Path(reference)
            if not reference.is_absolute():
                reference = Path(raw.get('cwd') or ROOT)/reference
            values['reference_result'] = reference.resolve()
        else:
            values['reference_result'] = None
        values['error'] = None
        return values
    except (OSError, ValueError, TypeError) as exc:
        return dict(reference_result=None, error=type(exc).__name__+': '+str(exc))


def repeat_index(folder):
    found = re.search(r'(?:^|[-_])r([1-9][0-9]*)(?:$|[-_])', folder.name)
    return None if found is None else int(found.group(1))


def pose_errors(metrics):
    details = ((metrics.get('motion') or {}).get('result') or {}).get('details') or {}
    position, orientation = finite(details.get('position_error_m')), finite(details.get('orientation_error_rad'))
    final, config = metrics.get('final_tcp') or {}, metrics.get('configuration') or {}
    if position is None and len(final.get('position') or []) == len(config.get('goal') or []) == 3:
        position = math.sqrt(sum((a-b)**2 for a, b in zip(final['position'], config['goal'])))
    first, last = final.get('orientation') or [], config.get('orientation') or []
    if orientation is None and len(first) == len(last) == 4:
        divisor = math.sqrt(sum(x*x for x in first)*sum(x*x for x in last))
        if divisor > 1e-12:
            orientation = 2*math.acos(min(1., abs(sum(a*b for a, b in zip(first, last)))/divisor))
    return dict(position_m=position, orientation_rad=orientation,
        position_tolerance_m=finite(details.get('position_tolerance_m')),
        orientation_tolerance_rad=finite(details.get('orientation_tolerance_rad')))


def load_attempt(path, cohort):
    path = Path(path)
    path = path/'result.json' if path.is_dir() else path
    path = path.resolve()
    folder, errors = path.parent, []
    declaration = launch(folder)
    try:
        data = path.read_bytes()
        raw, digest = json.loads(data), sha(data)
        if not isinstance(raw, dict):
            raise ValueError('A result object is required')
    except (OSError, ValueError) as exc:
        raw, digest = {}, None
        errors.append(type(exc).__name__+': '+str(exc))
    metrics, rep, research = raw.get('metrics') or {}, raw.get('representation_v3') or {}, raw.get('research') or {}
    contracts, camera = raw.get('contracts') or {}, research.get('camera_contracts') or {}
    config, registry = metrics.get('configuration') or {}, rep.get('registry') or {}
    activation = rep.get('activation') or {}
    guard, match = activation.get('guard') or {}, activation.get('initial_condition_match') or {}
    strategy = config.get('strategy', declaration.get('strategy'))
    is_historical = cohort == 'historical_direct'
    valid = (raw.get('status') == 'PASS' and bool(contracts) and all(v is True for v in contracts.values())
        and bool(camera) and all(v is True for v in camera.values()) and not raw.get('error') and not research.get('error'))
    if not is_historical:
        valid = (valid and bool(rep) and bool(guard) and all(v is True for v in guard.values())
            and rep.get('production_sources_unchanged') is True and rep.get('no_wait_for_mapping_during_motion') is True
            and (strategy != 'reactive' or registry.get('owner_thread_all') is True))
    if declaration.get('reference_result') is not None:
        valid = (valid and match.get('matched') is True and bool(match.get('checks'))
            and all(value is True for value in match['checks'].values()))
    for value in (raw.get('error'), research.get('error'), declaration.get('error')):
        if value:
            errors.append(str(value)[:500])
    initial, final = metrics.get('initial') or {}, metrics.get('final') or {}
    contact, severity = metrics.get('contacts') or {}, metrics.get('contact_severity') or {}
    controller, updates, path_metrics = metrics.get('controller') or {}, metrics.get('obstacle_updates') or {}, metrics.get('path') or {}
    outcome = (metrics.get('motion') or {}).get('result') or {}
    duration = finite((metrics.get('motion') or {}).get('seconds'))
    reported = contact.get('reported_points')
    depth = finite((severity.get('depth_m') or {}).get('maximum'))
    impulse = finite((severity.get('normal_pair_impulse_ns') or {}).get('maximum'))
    if reported == 0:
        depth = 0. if depth is None else depth
        impulse = 0. if impulse is None else impulse
    source_signature = dict(production_manifest_sha256=fingerprint(raw.get('hashes')) if raw.get('hashes') else None,
        diagnostic_sha256=rep.get('original_harness_sha256'), wrapper_sha256=rep.get('wrapper_sha256'),
        extractor_sha256=rep.get('extractor_sha256'), layout_file_sha256=research.get('layout_file_sha256'),
        calibration_sha256=fingerprint(calibration_signature(research.get('actual_camera_calibration'))),
        global_rays=research.get('global_ray_budget'), sampler=research.get('sampling_policy'),
        mapping_source=config.get('mapping_source'), mapping_hz=config.get('mapping_hz'),
        voxel_size_m=config.get('voxel_size'), backend_settings=metrics.get('backend_settings'))
    row = dict(id=folder.name, cohort=cohort, repeat=repeat_index(folder), artifact=label(path), artifact_sha256=digest,
        layout_id=research.get('layout_id', declaration.get('layout_id')),
        scenario=rep.get('scenario', declaration.get('scenario')), strategy=strategy,
        candidate=rep.get('candidate_id', declaration.get('candidate_id')), budget=rep.get('budget'),
        fixture_kind=rep.get('fixture_kind'), measurement_status=raw.get('status', 'UNAVAILABLE'), measurement_valid=bool(valid),
        goal_status=outcome.get('status'), goal_reached=(outcome.get('status') == 'success') if valid else None,
        strict_contact_free_success=metrics.get('avoidance_success') if valid else None,
        failure_code=outcome.get('failure_code'), classification=metrics.get('classification'),
        motion_seconds=duration, ee_error=pose_errors(metrics),
        minimum_cooked_link_clearance_m=finite((metrics.get('clearance') or {}).get('minimum_m')),
        minimum_reported_separation_m=finite(contact.get('minimum_reported_separation_m')),
        reported_contact_points=reported, observed_physical_contact_steps=severity.get('observed_physical_contact_steps'),
        maximum_observed_penetration_m=depth, maximum_observed_normal_pair_impulse_ns=impulse,
        sampled_physical_contact_seconds=finite(severity.get('sampled_physical_contact_duration_s')),
        physical_contact_classification=severity.get('physical_classification'), contact_acceptability='UNVALIDATED_NO_THRESHOLD',
        contact_buffer_saturated=contact.get('buffer_saturated'),
        path={key: path_metrics.get(key) for key in ('length_m', 'straight_distance_m', 'maximum_x_detour_m', 'maximum_z_detour_m', 'progress_reversals')},
        progress={key: (rep.get('progress') or {}).get(key) for key in
            ('longest_low_progress_seconds', 'stationary_windows', 'low_goal_progress_windows')},
        control_interval_ms=quantiles(controller.get('apply_interval_ms')),
        kit_interval_ms=quantiles(controller.get('kit_interval_ms')),
        controller_waits_over_100ms=controller.get('waits_over_100_ms'),
        consumed_original_source_age_ms=quantiles(updates.get('map_age_ms')) if strategy == 'reactive' else None,
        final_source_age_ms=None if finite((final.get('mapping') or {}).get('age_seconds')) is None else final['mapping']['age_seconds']*1000,
        registry_sync_ms=quantiles(registry.get('sync_ms')),
        registry_actual_update_ms=quantiles(registry.get('actual_update_ms')),
        registry_operations=registry.get('operations'), registry_changed_calls=registry.get('changed_calls'),
        registry_change_hz=None if not duration else (registry.get('changed_calls') or 0)/duration,
        sdk_world_update_ms=quantiles(controller.get('sdk_world_update_ms')),
        maximum_registered_proxies=updates.get('maximum_proxy_count') if strategy == 'reactive' else None,
        accepted_motion_revisions=updates.get('accepted_revisions') if strategy == 'reactive' else None,
        publication_hz={key: (metrics.get('observation_rates') or {}).get(key) for key in ('semantic_hz', 'mapping_hz')},
        source_failures_lifetime=(metrics.get('closed_strategy') or {}).get('source_failures'),
        timing_eligible=bool(valid) and not is_historical and cohort != 'retained'
            and rep.get('full_capture_excluded_from_timing_claims') is False,
        activation_guard=guard, matched_checks=match.get('checks'), q_maximum_error_rad=match.get('maximum_joint_error_rad'),
        q_tolerance_rad=match.get('joint_tolerance_rad'),
        declared_reference=label(declaration['reference_result']) if declaration.get('reference_result') else None,
        reference_runtime_match=match.get('matched') if declaration.get('reference_result') else None,
        errors=errors or None)
    row['mapping_counts'] = {key: known_delta(initial.get('mapping') or {}, final.get('mapping') or {}, key)
        for key in ('submitted', 'published', 'rejected', 'source_failures', 'worker_failures', 'processing_failures')}
    row['mapping_worker_counts'] = {key: known_delta((initial.get('mapping') or {}).get('worker') or {},
        (final.get('mapping') or {}).get('worker') or {}, key) for key in ('completed', 'replaced', 'discarded', 'errors')}
    extraction = updates.get('extraction') or (final.get('mapping_diagnostics') or {}).get('obstacle_extraction') or {}
    row['final_extraction'] = {key: extraction.get(key) for key in ('occupied_voxels', 'supported_voxels',
        'candidate_boxes', 'proxy_count', 'represented_voxels', 'omitted_voxels', 'selected_floor_boxes', 'selected_hazard_boxes')}
    try:
        trace = read(folder/'trace.json')
        latency = observed_worker_latency(trace)
        row['observed_worker_ms'] = {key: quantiles(latency.get(key), worker=True) for key in
            ('compute_ms', 'queue_ms', 'transport_ms', 'publication_ms', 'original_publish_age_ms')}
        row['observed_completed_source_jobs'] = latency['source_jobs']
    except (OSError, ValueError, TypeError):
        row['observed_worker_ms'], row['observed_completed_source_jobs'] = None, None
    return row, source_signature, raw


def summarize(cohorts):
    rows, sources, contracts, seen, duplicates = [], {}, {}, set(), []
    for cohort in COHORTS:
        for value in cohorts.get(cohort) or []:
            path = Path(value)
            path = path/'result.json' if path.is_dir() else path
            identity = str(path.resolve())
            if identity in seen:
                duplicates.append(dict(artifact=label(path), requested_cohort=cohort))
                continue
            seen.add(identity)
            row, source, raw = load_attempt(path, cohort)
            source_id = fingerprint(source)[:16]
            if source_id in sources and sources[source_id] != source:
                raise ValueError('Unexpected source fingerprint collision')
            sources[source_id] = source
            row['source_fingerprint'] = source_id
            contract = dict(native=raw.get('contracts'), camera=(raw.get('research') or {}).get('camera_contracts'),
                activation=row.pop('activation_guard'), matching=row.pop('matched_checks'))
            contract_id = fingerprint(contract)[:16]
            if contract_id in contracts and contracts[contract_id] != contract:
                raise ValueError('Unexpected contract-set fingerprint collision')
            contracts[contract_id] = contract
            row['contract_set'] = contract_id
            rows.append(row)
    by_artifact = {row['artifact']: row for row in rows}
    edges = []
    for row in rows:
        reference = row.get('declared_reference')
        if reference is None:
            continue
        baseline = by_artifact.get(reference)
        if baseline is None:
            try:
                reference_sha = sha((ROOT/reference).read_bytes())
            except OSError:
                reference_sha = None
        else:
            reference_sha = baseline['artifact_sha256']
        compatible = bool(baseline and baseline['candidate'] == 'baseline' and all(
            baseline[key] == row[key] for key in ('layout_id', 'scenario', 'strategy')))
        eligible = bool(compatible and baseline['measurement_valid'] and row['measurement_valid']
            and row['reference_runtime_match'] is True)
        edge = dict(candidate_id=row['id'], candidate_artifact=row['artifact'],
            declared_baseline=reference, declared_baseline_sha256=reference_sha,
            baseline_included=baseline is not None, comparison_eligible=eligible,
            status='MATCHED_DECLARED_REFERENCE' if eligible else 'UNPROVEN_OR_INVALID_REFERENCE',
            timing_comparison_eligible=bool(eligible and baseline['timing_eligible'] and row['timing_eligible']))
        if eligible:
            edge['goal_reached'] = [baseline['goal_reached'], row['goal_reached']]
            edge['strict_contact_free_success'] = [baseline['strict_contact_free_success'], row['strict_contact_free_success']]
            edge['motion_seconds_delta'] = (None if baseline['motion_seconds'] is None or row['motion_seconds'] is None
                else row['motion_seconds']-baseline['motion_seconds'])
        edges.append(edge)
    return dict(schema_version=1, status='COMPACT_NATIVE_RESEARCH_EVIDENCE',
        counts=dict(unique_attempts=len(rows), valid_measurements=sum(row['measurement_valid'] for row in rows),
            timing_eligible=sum(row['timing_eligible'] for row in rows), declared_reference_edges=len(edges),
            eligible_reference_edges=sum(row['comparison_eligible'] for row in edges),
            unique_declared_baselines=len({row['declared_baseline'] for row in edges})),
        cohort_counts=dict(Counter(row['cohort'] for row in rows)), cases=rows,
        declared_comparisons=edges, source_fingerprints=sources, contract_sets=contracts,
        declared_baseline_use_counts=dict(Counter(row['declared_baseline'] for row in edges)),
        duplicate_requested_artifacts=duplicates,
        scope=[
            'Each unique attempt appears once, with its declared cohort and explicit repeat suffix. No attempt is selected for favorable speed or outcome.',
            'Each candidate has at most one baseline edge: its exact launch --reference-result. Repeat baselines are not multiplied into Cartesian comparison pairs.',
            'Multiple candidates can share a declared baseline; reference edges are not an independent-baseline sample count. Use counts and unique-baseline counts remain explicit.',
            'Baseline SHA is read from the retained file at reduction time; the runtime guard recorded matching checks, not a motion-time reference-file hash.',
            'Measurement validity, target completion, strict zero-reported-contact success and ungraded contact severity remain separate.',
            'Cooked-link separation is a sampled independent reference, not a continuous collision certificate. Raw impulse/depth have no acceptability threshold.',
            'Worker latency distributions cover distinct completed source jobs seen by the sparse owner sampler, not every publication.',
            'Registry change rate counts measured motion calls with actual geometry/world changes, not every control callback or source publication.',
            'Historical direct controls are context; missing explicit matching proofs never become matched V3 comparisons.',
            'Source fingerprints store exact canonical calibration and source hashes; runtime calibration equivalence uses the recorded tolerance checks.',
            'Repeated matching/native/camera/activation Boolean sets are interned once under contract_sets; each attempt records its exact set ID.',
            'Full forensic captures and retained preparation attempts are excluded from live timing claims.'])


def rounded_scalars(value):
    if isinstance(value, float):
        return float(f'{value:.10g}') if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: rounded_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded_scalars(item) for item in value]
    return value


def columnar_cases(report):
    """Store scalar leaf headers once without dropping nulls or any attempt."""
    def leaves(value, prefix=''):
        if isinstance(value, dict) and value:
            result = {}
            for key, item in value.items():
                if '.' in key:
                    raise ValueError('Columnar case field names cannot contain dots')
                result.update(leaves(item, prefix+'.'+key if prefix else key))
            return result
        return {prefix: value}
    flattened = [leaves(row) for row in report['cases']]
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    values, presence = [], []
    for row in flattened:
        values.append([row.get(key) for key in columns])
        presence.append(format(sum(1 << index for index, key in enumerate(columns) if key in row), 'x'))
    return {**{key: value for key, value in report.items() if key != 'cases'},
        'schema_version': 2, 'case_columns': columns, 'case_rows': values, 'case_presence': presence,
        'case_encoding': 'Scalar leaf columns use dotted dictionary paths. case_presence stores a hexadecimal bit mask per row so absent leaves and explicit null containers round-trip exactly. expand_case_table() restores ordinary case dictionaries.'}


def expand_case_table(report):
    if 'cases' in report:
        return report
    columns, rows = report['case_columns'], []
    if len(report['case_rows']) != len(report['case_presence']):
        raise ValueError('Columnar row/presence counts differ')
    for values, encoded_presence in zip(report['case_rows'], report['case_presence']):
        if len(values) != len(columns):
            raise ValueError('Columnar case/header widths differ')
        presence, row = int(encoded_presence, 16), {}
        for index, (column, value) in enumerate(zip(columns, values)):
            if not presence & (1 << index):
                continue
            parts, parent = column.split('.'), row
            for part in parts[:-1]:
                if part not in parent:
                    parent[part] = {}
                if not isinstance(parent[part], dict):
                    raise ValueError('Inconsistent nested columnar case shape')
                parent = parent[part]
            parent[parts[-1]] = value
        rows.append(row)
    return {**{key: value for key, value in report.items()
        if key not in ('case_columns', 'case_rows', 'case_presence', 'case_encoding')}, 'cases': rows}


def world_box_vertices(dimensions, pose):
    if len(pose) != 7 or len(dimensions) != 3:
        raise ValueError('Actual XYZ/xyzw box pose and dimensions are required')
    x, y, z, w = pose[3:]
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not norm:
        raise ValueError('Actual physical fixture orientation is invalid')
    x, y, z, w = [value/norm for value in (x, y, z, w)]
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    signs = list(product((-1, 1), repeat=3))
    vertices = []
    for sign in signs:
        local = [sign[axis]*dimensions[axis]*.5 for axis in range(3)]
        vertices.append([pose[axis]+sum(rotation[axis][j]*local[j] for j in range(3)) for axis in range(3)])
    edges = [(i, j) for i in range(8) for j in range(i+1, 8) if sum(a != b for a, b in zip(signs[i], signs[j])) == 1]
    return vertices, edges


def representative_attempts(report):
    panels = [('A_current', 'static'), ('A_current', 'partial'),
        ('C_asymmetric_1.15_0.65', 'static'), ('C_asymmetric_1.15_0.65', 'narrow')]
    result = []
    for layout, scenario in panels:
        selected = {}
        for policy in POLICIES:
            choices = [row for row in report['cases'] if row['cohort'] == 'primary' and row['repeat'] == 1
                and row['layout_id'] == layout and row['scenario'] == scenario and row['candidate'] == policy
                and row['strategy'] == 'reactive']
            if len(choices) > 1:
                raise ValueError('Ambiguous primary r1 trajectory; refusing favorable selection: '+repr((layout, scenario, policy)))
            selected[policy] = choices[0] if choices else None
        result.append((layout, scenario, selected))
    return result


def configure_plot_cache():
    directory = Path(os.environ.setdefault('MPLCONFIGDIR', str(
        ROOT/'outputs/obstacle-representation-v3/matplotlib-cache')))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot(report, path, *, contact_points=False):
    configure_plot_cache()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(baseline='#236da2', diversity='#c06b24', motion='#368668')
    styles = dict(baseline='--', diversity='-', motion=':')
    figure = plt.figure(figsize=(12., 9.4), constrained_layout=True)
    metadata, global_points = [], []
    for panel, (layout, scenario, selected) in enumerate(representative_attempts(report), 1):
        axis = figure.add_subplot(2, 2, panel, projection='3d')
        reference = next((row for row in selected.values() if row is not None), None)
        goal = None
        outlines = []
        for policy, row in selected.items():
            if row is None:
                metadata.append(dict(layout_id=layout, scenario=scenario, candidate=policy, status='MISSING_PRIMARY_R1'))
                continue
            result_path = ROOT/row['artifact']
            raw, trace = read(result_path), read(result_path.parent/'trace.json')
            metrics = raw.get('metrics') or {}
            config = metrics.get('configuration') or {}
            points = [value['position'] for value in trace.get('tcp') or []]
            if points:
                axis.plot(*zip(*points), color=colors[policy], linestyle=styles[policy], linewidth=1.6, alpha=.9)
                axis.scatter(*points[0], color=colors[policy], s=15)
                global_points.extend(points)
            contacts = [point for sample in trace.get('samples') or [] for pair in sample.get('contacts') or []
                for point in pair.get('points_world_m') or []]
            plotted = contacts[::max(1, math.ceil(len(contacts)/2000))] if contact_points else []
            if plotted:
                axis.scatter(*zip(*plotted), marker='x', s=9, color='#a33547', alpha=.45)
                global_points.extend(plotted)
            if row is reference:
                objects = config.get('objects') or []
                poses = (metrics.get('fixture_setup') or {}).get('actual_actor_poses') or []
                if len(objects) != len(poses):
                    raise ValueError('Actual inserted fixture poses are required; nominal outlines are not substituted')
                for obj, pose in zip(objects, poses):
                    vertices, edges = world_box_vertices(obj['dimensions'], pose)
                    outlines.append(dict(name=obj['name'], dimensions_m=obj['dimensions'], actual_pose=pose))
                    global_points.extend(vertices)
                    for first, last in edges:
                        axis.plot(*zip(vertices[first], vertices[last]), color='#777777', linewidth=.8, alpha=.55)
                goal = config.get('goal')
                if goal:
                    global_points.append(goal)
            metadata.append(dict(layout_id=layout, scenario=scenario, candidate=policy, attempt=row['id'],
                artifact=row['artifact'], artifact_sha256=row['artifact_sha256'],
                tcp_samples=len(points), raw_reported_contact_points=len(contacts), plotted_contact_points=len(plotted),
                goal_status=row['goal_status'], strict_contact_free_success=row['strict_contact_free_success']))
        if goal:
            axis.scatter(*goal, marker='*', color='#222222', s=70)
        axis.set_title(('A' if layout == 'A_current' else 'C')+' '+scenario, fontsize=11)
        axis.set_xlabel('World X (m)')
        axis.set_ylabel('World Y (m)')
        axis.set_zlabel('World Z (m)')
        axis.view_init(elev=25, azim=-58)
        axis.set_box_aspect((1., 1., .8))
        metadata.append(dict(layout_id=layout, scenario=scenario, physical_fixture_outlines=outlines))
    if global_points:
        bounds = []
        for dimension in range(3):
            lower, upper = min(p[dimension] for p in global_points), max(p[dimension] for p in global_points)
            margin = max(.025, (upper-lower)*.07)
            bounds.append((lower-margin, upper+margin))
        for axis in figure.axes:
            for dimension, setter in enumerate((axis.set_xlim, axis.set_ylim, axis.set_zlim)):
                setter(*bounds[dimension])
            axis.set_box_aspect([upper-lower for lower, upper in bounds])
    handles = [Line2D([], [], color=colors[name], linestyle=styles[name], linewidth=2, label=name) for name in POLICIES]
    handles += [Line2D([], [], color='#777777', linewidth=1, label='Actual physical fixture outline'),
        Line2D([], [], color='#222222', marker='*', linestyle='none', label='Goal TCP')]
    if contact_points:
        handles.append(Line2D([], [], color='#a33547', marker='x', linestyle='none', label='PhysX surface contact points (not TCP)'))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.015), ncol=3, frameon=False, fontsize=9)
    figure.suptitle('Actual primary r1 TCP trajectories under three observed-geometry policies\n'
        'TCP paths do not show whole-arm motion and do not certify continuous clearance.', fontsize=12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=165, bbox_inches='tight')
    plt.close(figure)
    return dict(figure=str(path), selected=metadata, contact_points_enabled=contact_points,
        caption='Curves connect actual measured TCP samples from the predeclared primary r1 attempts. Gray wireframes use actual inserted actor poses and physical box dimensions. Red markers, when enabled, are sampled PhysX robot-surface contact points, not TCP positions; at most2000 actual points per attempt are displayed. This is not a whole-arm swept volume or a continuous-clearance certificate. All panels share world-coordinate limits.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in COHORTS:
        parser.add_argument('--'+cohort.replace('_', '-'), nargs='*', default=[])
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--figure-metadata', type=Path)
    parser.add_argument('--contact-points', action='store_true')
    parser.add_argument('--max-bytes', type=int, default=250_000)
    parser.add_argument('--columnar', action='store_true', help='Store scalar leaf names once; preserve all attempts/fields with exact decoder')
    args = parser.parse_args(argv)
    report = summarize({cohort: getattr(args, cohort) for cohort in COHORTS})
    report['scalar_precision'] = '10 significant digits; complete original precision remains in each hashed raw result. Exact Boolean checks and source hashes are preserved.'
    output = columnar_cases(report) if args.columnar else report
    encoded = (json.dumps(rounded_scalars(output), separators=(',', ':'), allow_nan=False)+'\n').encode()
    if len(encoded) > args.max_bytes:
        raise ValueError(f'Compact evidence is {len(encoded)}bytes; exceeds explicit {args.max_bytes}byte budget. No attempts were removed.')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_bytes(encoded)
    if args.figure:
        metadata = plot(report, args.figure, contact_points=args.contact_points)
        destination = args.figure_metadata or args.figure.with_suffix('.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(bytes=len(encoded), **report['counts'])))


if __name__ == '__main__':
    main()
