#!/usr/bin/env python3
"""Compare repeated native camera-study windows without pooling frame samples.

Reading results is SDK-free. Plotting imports matplotlib only when --figure is
requested after all hardware windows finish. A25% Kitp99 cost is a reported
trade-off, never an automatic rejection or a collision-avoidance success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ('kit_update_ms', 'kit_interval_ms', 'physics_interval_ms', 'physics_step_ms',
    'physics_dt_ms', 'control_interval_ms', 'control_wait_ms', 'control_apply_ms',
    'mapping_source_age_ms', 'mapping_compute_ms', 'mapping_queue_ms',
    'mapping_worker_roundtrip_ms', 'mapping_transport_envelope_ms',
    'mapping_worker_to_publication_ms', 'mapping_publish_age_ms',
    'semantic_compute_ms', 'semantic_publish_age_ms', 'cpu_utilization_percent',
    'available_ram_bytes')
STATISTICS = ('mean', 'p50', 'p95', 'p99', 'worst')


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def aggregate(values):
    values = [number(value) for value in values]
    values = [value for value in values if value is not None]
    return (dict(count=0, minimum=None, median=None, maximum=None, mean=None)
            if not values else dict(count=len(values), minimum=min(values),
                median=median(values), maximum=max(values), mean=sum(values)/len(values)))


def label(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def json_file(path):
    try:
        data = Path(path).read_bytes()
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError('Expected an object')
        return result, hashlib.sha256(data).hexdigest(), None
    except (OSError, ValueError) as exc:
        return {}, None, f'{type(exc).__name__}: {exc}'


def launch_context(folder):
    payload, _, _ = json_file(folder/'launch.json')
    command = payload.get('command')
    if not isinstance(command, list):
        return {}
    result = {}
    for option, key in [('--layout-id', 'layout_id'), ('--mode', 'mode'),
        ('--sampling-policy', 'sampling_policy'), ('--max-rays', 'global_ray_budget')]:
        if option in command and command.index(option)+1 < len(command):
            result[key] = command[command.index(option)+1]
    return result


def log_errors(folder, payload):
    supplied = payload.get('runtime_log_errors')
    if isinstance(supplied, dict):
        return supplied
    try:
        lines = [line[:1000] for line in (folder/'native.log').read_text(errors='replace').splitlines()
                 if '[error]' in line.lower() or '[fatal]' in line.lower()]
        return dict(count=len(lines), first_lines=lines[:5])
    except OSError:
        return dict(count=None, first_lines=[], limitation='Native log not available')


def distribution_value(summary, key, statistic):
    value = summary.get(key) or {}
    count = number(value.get('count'))
    return None if count is None or count <= 0 else number(value.get(statistic))


def load_case(value):
    path = Path(value)
    path = path/'result.json' if path.is_dir() else path
    raw, digest, read_error = json_file(path)
    folder = path.parent
    context = launch_context(folder)
    summary, research = raw.get('summary') or {}, raw.get('research') or {}
    protocol = raw.get('protocol') or {}
    trace, _, trace_error = json_file(folder/'trace.json')
    events = trace.get('events') or {}
    motions = events.get('motions') or []
    motion_seconds = [number(row.get('seconds')) for row in motions]
    motion_total = (sum(motion_seconds) if motions and all(v is not None for v in motion_seconds) else None)
    camera_contracts = research.get('contracts') or {}
    cameras = (research.get('layout') or {}).get('cameras') or []
    third_resolution = list(cameras[2].get('resolution', [])) if len(cameras) == 3 else None
    runtime_errors = log_errors(folder, raw)
    execution_errors = list(raw.get('errors') or [])
    errors = list(execution_errors)
    if read_error:
        errors.append(read_error)
    jobs = summary.get('job_counts') or {}
    job_totals = {}
    for key in ('source_failures', 'worker_failures', 'rejected', 'replaced', 'discarded', 'errors'):
        values = []
        for channel in ('semantic', 'mapping'):
            counters = jobs.get(channel) or {}
            owner = counters.get('worker') or {} if key in ('replaced', 'discarded', 'errors') else counters
            values.append(number(owner.get(key)))
        job_totals[key] = sum(values) if all(v is not None for v in values) else None
    worst_control = distribution_value(summary, 'control_interval_ms', 'worst')
    severe_control = None if worst_control is None else worst_control >= 300.
    repeated_source = None if job_totals['source_failures'] is None else job_totals['source_failures'] >= 2
    observed_errors = bool(execution_errors) or (number(runtime_errors.get('count')) or 0) > 0
    observed_job_error = any((job_totals[key] or 0) > 0 for key in ('source_failures', 'worker_failures', 'errors'))
    gpu_backed = bool(raw.get('gpu_devices'))
    instability = ('NATIVE_UNAVAILABLE' if raw.get('status') == 'BLOCKED' else
        'NOT_MEASURED' if not gpu_backed else
        'INSTABILITY_OBSERVED' if observed_errors or observed_job_error or severe_control else
        'NOT_MEASURED' if worst_control is None or job_totals['source_failures'] is None else
        'NO_INSTABILITY_OBSERVED')
    required = ('kit_update_ms', 'physics_interval_ms', 'physics_step_ms', 'control_interval_ms',
                'mapping_source_age_ms', 'mapping_compute_ms', 'mapping_queue_ms', 'mapping_publish_age_ms')
    valid = (raw.get('status') == 'PASS' and not errors and not (number(runtime_errors.get('count')) or 0)
        and gpu_backed and bool(camera_contracts) and all(v is True for v in camera_contracts.values())
        and summary.get('motion_complete') is True and summary.get('frame_checks') is True
        and len(motions) == 4 and all(row.get('ok') is True for row in motions)
        and all(distribution_value(summary, key, 'p99') is not None for key in required)
        and number(summary.get('semantic_hz')) is not None and number(summary.get('mapping_hz')) is not None)
    ram_samples = [number(row.get('available_ram_bytes')) for row in events.get('cpu', [])]
    ram_samples = [v for v in ram_samples if v is not None]
    row = dict(case_id=label(folder), artifact=dict(path=label(path), sha256=digest),
        layout_id=research.get('layout_id', context.get('layout_id')),
        mode=research.get('mode', context.get('mode')),
        sampling_policy=research.get('sampling_policy', context.get('sampling_policy', 'fixed')),
        global_ray_budget=number(research.get('global_ray_budget', context.get('global_ray_budget',
            protocol.get('global_mapping_ray_budget')))), third_resolution=third_resolution,
        native_status=raw.get('status', 'UNAVAILABLE'), measurement_valid=valid,
        gpu_backed_native_rendering=gpu_backed, gpu_devices=raw.get('gpu_devices'),
        errors=errors, runtime_log_errors=runtime_errors,
        wall_seconds=number(summary.get('wall_seconds')),
        configured_measurement_seconds=number(protocol.get('measurement_seconds')),
        warmup_seconds=number(protocol.get('warmup_seconds')),
        metric_distributions={key: summary.get(key) for key in TIMINGS},
        semantic_hz=number(summary.get('semantic_hz')), mapping_hz=number(summary.get('mapping_hz')),
        mapping_fresh_sample_fraction=number(summary.get('mapping_fresh_sample_fraction')),
        motion_complete=summary.get('motion_complete'), frame_checks=summary.get('frame_checks'),
        calibration_contracts=camera_contracts,
        motion_attempts=len(motions) if motions else None,
        successful_moves=sum(row.get('ok') is True for row in motions) if motions else None,
        movement_total_seconds=motion_total,
        movements=[{key: motion.get(key) for key in ('command', 'seconds', 'ok')} for motion in motions],
        trace_available=trace_error is None, trace_limitation=trace_error,
        minimum_available_ram_bytes=min(ram_samples) if ram_samples else None,
        worker_counts=jobs, worker_count_totals=job_totals,
        allocated_mapping_rays=summary.get('allocated_mapping_rays'),
        mapping_rays_per_camera=summary.get('mapping_rays_per_camera'),
        instability=dict(status=instability, control_interval_worst_ms=worst_control,
            control_stall_at_least_300ms=severe_control, repeated_source_failures=repeated_source,
            worker_job_pressure_observed=any((job_totals[key] or 0) > 0 for key in ('rejected', 'replaced', 'discarded'))),
        optional_gpu_snapshots=raw.get('optional_gpu_snapshots'),
        research_source_hashes={key: research.get(key) for key in
            ('wrapper_sha256', 'layout_file_sha256', 'sampling_script_sha256')})
    return row


def group_cases(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['layout_id'], row['mode'], row['sampling_policy'], row['global_ray_budget'],
               tuple(row['third_resolution']) if row['third_resolution'] else None)
        grouped[key].append(row)
    groups = []
    for key, attempts in sorted(grouped.items(), key=lambda item: repr(item[0])):
        measured = [row for row in attempts if row['measurement_valid']]
        metrics = {name: {statistic: aggregate(distribution_value(row['metric_distributions'], name, statistic)
                for row in measured) for statistic in STATISTICS} for name in TIMINGS}
        scalars = {name: aggregate(row.get(name) for row in measured) for name in
            ('semantic_hz', 'mapping_hz', 'movement_total_seconds', 'wall_seconds',
             'mapping_fresh_sample_fraction', 'minimum_available_ram_bytes')}
        groups.append(dict(layout_id=key[0], mode=key[1], sampling_policy=key[2],
            global_ray_budget=key[3], third_resolution=key[4], attempts=len(attempts),
            valid_measurements=len(measured), invalid_measurements=len(attempts)-len(measured),
            case_ids=[row['case_id'] for row in attempts],
            invalid_case_ids=[row['case_id'] for row in attempts if not row['measurement_valid']],
            measurement_status='NOT_MEASURED' if not measured else 'PASS' if len(measured) == len(attempts) else 'PARTIAL_FAILURE',
            instability_status=('INSTABILITY_OBSERVED' if any(row['instability']['status'] == 'INSTABILITY_OBSERVED' for row in attempts)
                else 'NOT_MEASURED' if any(row['instability']['status'] in ('NOT_MEASURED', 'NATIVE_UNAVAILABLE') for row in attempts)
                else 'NO_INSTABILITY_OBSERVED'),
            metrics=metrics, scalars=scalars,
            aggregation_scope='Median/min/max of per-run statistics from valid native windows, not pooled frame samples; all invalid attempts remain visible and prevent an unqualified measurement PASS.'))
    return groups


def add_comparisons(groups):
    matches = [g for g in groups if g['layout_id'] == 'A_current' and g['mode'] == 'baseline'
        and g['sampling_policy'] == 'fixed' and g['global_ray_budget'] == 1024 and g['third_resolution'] is None]
    baseline = matches[0] if len(matches) == 1 else None
    first = None if baseline is None else baseline['metrics']['kit_update_ms']['p99']
    for group in groups:
        measured = group['metrics']['kit_update_ms']['p99']
        value = measured['median']
        if first is None or first['median'] is None or first['median'] <= 0 or value is None:
            group['baseline_comparison'] = dict(cost_flag='NOT_MEASURED')
            continue
        base = first['median']
        group['baseline_comparison'] = dict(
            baseline_kit_p99_median_ms=base,
            baseline_kit_p99_run_range_ms=[first['minimum'], first['maximum']],
            candidate_kit_p99_run_range_ms=[measured['minimum'], measured['maximum']],
            median_kit_p99_absolute_cost_ms=value-base,
            median_kit_p99_relative_cost_percent=100*(value/base-1),
            median_cost_against_baseline_range_percent=[100*(value/first['maximum']-1),
                100*(value/first['minimum']-1)] if first['minimum'] > 0 else None,
            cost_flag='MATERIAL_KIT_P99_COST' if value > base*1.25 else 'WITHIN_25_PERCENT_KIT_P99_REFERENCE',
            cost_flag_is_automatic_veto=False,
            qualification='A reported runtime trade-off. Visibility/planner utility and observed instability must be assessed separately; this is not a physical motion-success assessment.')
    return None if baseline is None else dict(case_ids=baseline['case_ids'],
        valid_measurements=baseline['valid_measurements'], invalid_measurements=baseline['invalid_measurements'],
        kit_p99_ms=first, scope='All explicitly supplied A_current/two-camera/fixed1024 baseline runs, including interleaved repeats.')


def summarize(paths):
    rows, seen, duplicates = [], set(), []
    for value in paths:
        path = Path(value)
        path = path/'result.json' if path.is_dir() else path
        identity = str(path.resolve())
        if identity in seen:
            duplicates.append(label(path))
            continue
        seen.add(identity)
        rows.append(load_case(path))
    groups = group_cases(rows)
    baseline = add_comparisons(groups)
    valid = sum(row['measurement_valid'] for row in rows)
    return dict(schema_version=1, status='RESEARCH_PERFORMANCE_SUMMARY',
        measurement_status='NOT_MEASURED' if not rows or not valid else 'PASS' if valid == len(rows) else 'PARTIAL_FAILURE',
        counts=dict(attempts=len(rows), valid_measurements=valid, invalid_measurements=len(rows)-valid,
            groups=len(groups), observed_instability=sum(row['instability']['status'] == 'INSTABILITY_OBSERVED' for row in rows)),
        baseline=baseline, groups=groups, cases=rows, duplicate_requested_artifacts=duplicates,
        interpretation='Performance measurement validity, runtime cost, resource instability and physical avoidance are separate assessments. This file does not score collision avoidance.',
        limitations=[
            'Short windows and few repeats give descriptive ranges, not confidence intervals. Frame samples are not treated as independent repeated experiments.',
            'The25% Kitp99 cost flag is a visibility/performance trade-off indicator, not an automatic reason to discard a useful camera layout.',
            'Invalid and failed attempts remain visible. Group cost statistics use valid GPU-backed windows only and explicitly report excluded invalid attempt counts.',
            'Control timing covers the identical movement sequence; Kit, physics and worker rates cover the full measurement window.',
            'Physics pre/post wall time is not GPU-kernel time. Worker transport is an envelope, not isolated IPC latency.',
            'Mapping source age uses the original first owner acquisition timestamp, not a measured GPU exposure wall clock.',
            'Optional GPU readings are outside-window snapshots, not continuous utilization or peak VRAM measurements.',
            'No missing metric is replaced with zero, and no CPU-only startup is treated as native camera performance.'])


def plot_summary(summary, path):
    """Call after native windows; matplotlib is optional and lazily imported."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = summary['groups']
    figure, axes = plt.subplots(2, 2, figsize=(max(11., len(groups)*1.35), 8.), constrained_layout=True)
    specs = [('kit_update_ms', 'p99', 'Kit update p99 (ms)'),
        ('control_interval_ms', 'p99', 'Control interval p99 (ms)'),
        ('mapping_compute_ms', 'p50', 'Mapping compute p50 (ms)'),
        ('cpu_utilization_percent', 'mean', 'CPU mean (%)')]
    aliases = {'A_current': 'A', 'C_asymmetric_1.15_0.65': 'C',
        'H_third_side_-1.00': 'H', 'K_B_front_height_0.85_rear': 'K'}
    names = []
    for group in groups:
        short = aliases.get(group['layout_id'], str(group['layout_id']))
        workload = ('edge' if group['sampling_policy'] == 'depth_edge' else group['sampling_policy'])
        if group['mode'] in ('mapping', 'full'):
            workload = 'map' if group['mode'] == 'mapping' else 'full'
        budget = '—' if group['global_ray_budget'] is None else f"{group['global_ray_budget']:g} rays"
        names.append(f"{short} {workload}\n{budget}\nn={group['valid_measurements']}/{group['attempts']}")
    by_id = {row['case_id']: row for row in summary['cases']}
    for axis, (metric, statistic, title) in zip(axes.flat, specs):
        for i, group in enumerate(groups):
            samples = [distribution_value({'value': by_id[case]['metric_distributions'].get(metric)}, 'value', statistic)
                for case in group['case_ids'] if by_id[case]['measurement_valid']]
            samples = [v for v in samples if v is not None]
            if samples:
                offsets = [0.] if len(samples) == 1 else [-.12+.24*j/(len(samples)-1) for j in range(len(samples))]
                axis.scatter([i+x for x in offsets], samples, s=32, alpha=.8)
                axis.plot([i-.2, i+.2], [median(samples)]*2, color='black', linewidth=2)
            else:
                axis.text(i, .03, 'unmeasured', transform=axis.get_xaxis_transform(),
                          rotation=90, ha='center', va='bottom', color='gray', fontsize=8)
        axis.set_title(title)
        axis.set_xticks(range(len(groups)), names, ha='center', fontsize=8)
        axis.grid(axis='y', alpha=.25)
        axis.set_ylim(bottom=0)
    baseline = summary.get('baseline') or {}
    reference = baseline.get('kit_p99_ms') or {}
    if reference.get('median') is not None:
        axes[0, 0].axhspan(reference['minimum'], reference['maximum'], color='gray', alpha=.14,
            label='Baseline run range')
        axes[0, 0].axhline(reference['median']*1.25, color='darkorange', linestyle='--',
            label='+25% cost reference')
        axes[0, 0].legend(fontsize=8)
    figure.suptitle('Camera workload: independent valid runs and group medians; n=valid/attempted.\nA current, C asymmetric, H current+side, K front+rear. Cost does not assess collision avoidance.', fontsize=10)
    figure.savefig(path, dpi=170)
    if path.suffix.lower() != '.pdf':
        figure.savefig(path.with_suffix('.pdf'))
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--figure', help='Optional runtime-cost figure; import plotting tools only after measurements finish')
    args = parser.parse_args(argv)
    summary = summarize(args.cases)
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if args.figure:
        plot_summary(summary, args.figure)
    print(json.dumps(dict(measurement_status=summary['measurement_status'], counts=summary['counts'])))


if __name__ == '__main__':
    main()
