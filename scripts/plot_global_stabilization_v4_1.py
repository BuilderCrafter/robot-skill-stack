#!/usr/bin/env python3
"""Plot recorded V4.1 Home execution, native targets, tracking and TCP return.

Accepts any measured result paths and explicit labels. It creates no synthetic
after/fixed result and does not import Isaac or change simulator state.
Run only after native timing jobs have finished.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.analyze_global_stabilization_v4_1 import ARM, _arm_values, digest


def quaternion_distance(first, last):
    """Quaternion angular separation; shared component ordering is sufficient."""
    first, last = np.asarray(first, float), np.asarray(last, float)
    if (first.shape != (4,) or last.shape != (4,) or not np.isfinite(first).all()
            or not np.isfinite(last).all() or min(np.linalg.norm(first), np.linalg.norm(last)) < 1e-12):
        raise ValueError('Recorded TCP orientations require finite nonzero quaternions')
    dot = abs(float(first@last)/(np.linalg.norm(first)*np.linalg.norm(last)))
    return 2*math.acos(np.clip(dot, 0., 1.))


def _load_trace(path):
    path = Path(path)
    result = json.loads(path.read_text())
    reference = result['controlled']['raw_trace']
    trace_path = Path(reference['path'])
    if not trace_path.exists():
        trace_path = path.parent/'controlled_trace.json'
    if digest(trace_path) != reference['sha256']:
        raise ValueError('Controlled trace does not match its recorded SHA')
    trace = json.loads(trace_path.read_text())
    return path, result, reference, trace_path, trace


def extract_case(path, *, operation='home', meters_per_unit=1.):
    """Read only recorded arrays, preserving native simulation timestamps."""
    path, result, reference, trace_path, trace = _load_trace(path)
    selected = [row for row in trace['operations'] if row['label'] == operation]
    if len(selected) != 1:
        raise ValueError(f'Exactly one recorded operation {operation!r} is required')
    op = selected[0]
    samples = trace['samples'][op['first_sample']:op['last_sample']]
    actions = trace['actions'][op['first_action']:op['last_action']]
    if not samples:
        raise ValueError('Selected operation has no measured samples')
    meta = trace['metadata']
    native_index = meta['robot_dof_names'].index(ARM[0])
    plan = op.get('plan_trace') or {}
    actual = plan.get('actual') or []
    commanded = plan.get('commanded') or []
    origin = (actual[0]['sim_time'] if actual else commanded[0]['sim_time']
              if commanded else samples[0]['simulation_time'])
    origin_source = ('First recorded planner actual simulation time' if actual else
                     'First recorded planner command simulation time' if commanded else
                     'First observed operation simulation time; no planner execution trace')
    names = plan.get('joint_names') or []
    plan_joint = names.index(ARM[0]) if ARM[0] in names else None
    trajectory = plan.get('trajectory') or []
    plan_time = np.asarray([row['time'] for row in trajectory], float)
    plan_q1 = (np.asarray([row['q'][plan_joint] for row in trajectory], float)
               if plan_joint is not None else np.array([]))
    if len(plan_time) != len(plan_q1):
        raise ValueError('Recorded trajectory lacks an explicit joint1 name mapping')
    sample_time = np.asarray([row['simulation_time']-origin for row in samples], float)
    measured_q1 = np.asarray([row['feedback']['positions'][native_index] for row in samples], float)
    target_time, requested_q1, applied_q1, target_gap = [], [], [], []
    for row in actions:
        applied = row.get('actual_physx_targets') or {}
        if not applied.get('available'):
            continue
        requested = dict(joint_positions=row.get('requested_positions'), joint_indices=row.get('requested_indices'))
        requested_arm = _arm_values(requested, 'joint_positions', meta)
        applied_arm = _arm_values(applied['value'], 'joint_positions', meta)
        if requested_arm is None or applied_arm is None:
            continue
        target_time.append(row['simulation_time']-origin)
        requested_q1.append(requested_arm[0]);applied_q1.append(applied_arm[0])
        target_gap.append(np.max(np.abs(applied_arm-requested_arm)))
    # Match q actual and q desired at the exact same recorded planner-loop time.
    # No interpolation is used to reconstruct acceptance errors. The aborting
    # step is absent from plan.actual, so its max error is a separate marker.
    commands_by_time = {row['sim_time']: row for row in commanded}
    exact_time, exact_q1_error, guard_time, guard_error = [], [], [], []
    for row in actual:
        guard_time.append(row['sim_time']-origin);guard_error.append(row['tracking_error_rad'])
        same_time = commands_by_time.get(row['sim_time'])
        if plan_joint is not None and same_time is not None:
            exact_time.append(row['sim_time']-origin)
            exact_q1_error.append(abs(row['q'][plan_joint]-same_time['q'][plan_joint]))
    target = op.get('target')
    tcp_position, tcp_orientation = [], []
    for row in samples:
        if target is None or row.get('tcp_pose') is None:
            tcp_position.append(math.nan);tcp_orientation.append(math.nan)
            continue
        tcp_position.append(float(np.linalg.norm(np.asarray(row['tcp_pose']['position'])-target['position']))*meters_per_unit)
        tcp_orientation.append(quaternion_distance(row['tcp_pose']['orientation'], target['orientation']))
    diagnostics = op.get('strategy_diagnostics') or {}
    abort_error = diagnostics.get('tracking_error_rad') if diagnostics.get('failure_code') == 'tracking_failure' else None
    return dict(result_path=str(path), result_sha256=digest(path), trace_path=str(trace_path),
        trace_sha256=reference['sha256'], mode=result['controlled']['mode'],
        strategy=result['controlled']['strategy'], operation=operation,
        returned_success=op['returned_success'], failure_code=diagnostics.get('failure_code'),
        origin_simulation_time=origin, origin_source=origin_source,
        sample_time=sample_time, measured_q1=measured_q1, plan_time=plan_time, plan_q1=plan_q1,
        target_time=np.asarray(target_time), requested_q1=np.asarray(requested_q1), applied_q1=np.asarray(applied_q1),
        target_gap=np.asarray(target_gap), exact_time=np.asarray(exact_time), exact_q1_error=np.asarray(exact_q1_error),
        guard_time=np.asarray(guard_time), guard_error=np.asarray(guard_error),
        abort_error=abort_error, tcp_position=np.asarray(tcp_position), tcp_orientation=np.asarray(tcp_orientation),
        thresholds={key: diagnostics.get(key) for key in ('tracking_limit_rad', 'tracking_abort_rad',
            'position_tolerance_m', 'orientation_tolerance_rad')})


def extract_hold_case(path, *, meters_per_unit=1.):
    """Retain actual no-action hold windows, measured from Home completion."""
    path, result, reference, trace_path, trace = _load_trace(path)
    homes = [row for row in trace['operations'] if row['label'] == 'home']
    holds = [row for row in trace['operations'] if row.get('kind') == 'post_hold']
    if len(homes) != 1 or not holds:
        raise ValueError('Exactly one Home plus recorded post_hold windows are required')
    home = homes[0]
    origin = home['after']['simulation_time']
    target = home['target'];meta = trace['metadata']
    indices = [meta['robot_dof_names'].index(name) for name in ARM]
    time, position, orientation, speed, gap, counts = [], [], [], [], [], []
    windows = []
    for op in holds:
        samples = trace['samples'][op['first_sample']:op['last_sample']]
        for row in samples:
            time.append(row['simulation_time']-origin)
            position.append(np.linalg.norm(np.asarray(row['tcp_pose']['position'])-target['position'])*meters_per_unit)
            orientation.append(quaternion_distance(row['tcp_pose']['orientation'], target['orientation']))
            speed.append(np.abs(np.asarray(row['feedback']['velocities'])[indices]).max())
            values = row.get('actual_minus_applied_target_arm_rad')
            gap.append(math.nan if values is None else float(np.abs(values).max()))
            native = row.get('native_actor_solver_properties') or {}
            counts.append(native['position'] if native.get('available') else math.nan)
        windows.append(dict(label=op['label'], first_time=op['before']['simulation_time']-origin,
            last_time=op['after']['simulation_time']-origin,
            additional_actions=op['last_action']-op['first_action']))
    diag = home.get('strategy_diagnostics') or {}
    return dict(result_path=str(path), result_sha256=digest(path), trace_path=str(trace_path),
        trace_sha256=reference['sha256'], mode=result['controlled']['mode'],
        strategy=result['controlled']['strategy'], origin_simulation_time=origin,
        windows=windows, time=np.asarray(time), position=np.asarray(position), orientation=np.asarray(orientation),
        speed=np.asarray(speed), target_gap=np.asarray(gap), native_position_count=np.asarray(counts),
        thresholds={key: diag.get(key) for key in ('position_tolerance_m', 'orientation_tolerance_rad',
            'tracking_limit_rad', 'tracking_abort_rad')})


def render_holds(cases, labels, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    colors = ('#D86435', '#2878B5', '#23956F', '#805AA8')
    provenance = []
    for case, label, color in zip(cases, labels, (colors[i % len(colors)] for i in range(len(cases)))):
        for axis, field, scale in zip(axes, ('position', 'orientation', 'speed', 'target_gap', 'native_position_count'),
                (1000., 1., 1., 1., 1.)):
            axis.plot(case['time'], case[field]*scale, color=color, label=label)
        provenance.append({key: case[key] for key in ('result_path', 'result_sha256', 'trace_path',
            'trace_sha256', 'mode', 'strategy', 'origin_simulation_time', 'windows', 'thresholds')})
        provenance[-1]['label'] = label
    for row, key, scale in ((0, 'position_tolerance_m', 1000.), (1, 'orientation_tolerance_rad', 1.),
            (3, 'tracking_limit_rad', 1.), (3, 'tracking_abort_rad', 1.)):
        limits = {case['thresholds'][key] for case in cases if case['thresholds'][key] is not None}
        for value in limits:
            axes[row].axhline(value*scale, color='#777777', ls=':', lw=1, label=key)
    axes[2].axhline(.01, color='#777777', ls=':', lw=1, label='Proposed settling threshold 0.01 rad/s')
    for axis, label in zip(axes, ('TCP position error (mm)', 'TCP orientation error (rad)',
            'Max arm speed (rad/s)', 'Max joint target gap (rad)',
            'Native position iterations')):
        axis.set_ylabel(label, fontsize=9);axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[-1].set_xlabel('Native physics time after original Home returned success (s)')
    axes[-1].set_yticks([32, 255])
    figure.suptitle('Recorded post-Home hold: solver restoration and residual movement\n'
        'Supplied no-action observation windows only; count readback is native actor request, not island count.', fontsize=11)
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160);plt.close(figure)
    manifest = dict(schema_version=1, producer=dict(path=str(Path(__file__)), sha256=digest(__file__)),
        figure=dict(path=str(output), sha256=digest(output)), cases=provenance,
        method='Existing recorded zero-velocity target hold windows; native measured feedback and actor iteration requests. No extra commands or simulated after-data are introduced by this analysis. The 0.01 rad/s line is a proposed stricter completion criterion, not an original V4 acceptance threshold.')
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def render(cases, labels, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, len(cases), figsize=(6.2*len(cases), 13),
        squeeze=False, sharex=True, sharey='row', constrained_layout=True)
    provenance = []
    gap_axes, orientation_axes = [], []
    for column, (case, label) in enumerate(zip(cases, labels)):
        first, targets, error, tcp = axes[:, column]
        title = f"{label}\n{case['strategy']} / {case['mode']}: "
        title += 'returned success' if case['returned_success'] else 'returned failure'
        if case['failure_code']:
            title += ' ('+case['failure_code']+')'
        first.set_title(title, fontsize=10)
        if len(case['plan_time']):
            first.plot(case['plan_time'], case['plan_q1'], '--', color='#2878B5', label='Generated plan samples')
        else:
            first.text(.02, .03, 'No generated-plan trace recorded', transform=first.transAxes, fontsize=8)
        first.plot(case['sample_time'], case['measured_q1'], color='#222222', label='Native measured q1')
        first.set_ylabel('Joint1 position (rad)');first.legend(fontsize=8)
        targets.step(case['target_time'], case['requested_q1'], where='post', color='#2878B5', label='Requested q1')
        targets.plot(case['target_time'], case['applied_q1'], '.', ms=3, color='#D86435', label='Native PhysX target q1')
        targets.set_ylabel('Joint1 target (rad)');targets.legend(fontsize=8, loc='upper left')
        gap = targets.twinx()
        gap_axes.append(gap)
        gap.plot(case['target_time'], case['target_gap'], color='#23956F', alpha=.65, label='Max arm readback gap')
        gap.set_ylabel('Max |applied − requested| (rad)', color='#23956F', fontsize=8)
        gap.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        error.plot(case['guard_time'], case['guard_error'], color='#D86435', label='Recorded max-joint guard error')
        error.plot(case['exact_time'], case['exact_q1_error'], '--', color='#2878B5', label='|q1 − exact same-time desired|')
        if case['abort_error'] is not None:
            error.plot(case['sample_time'][-1], case['abort_error'], 'rx', ms=8, label='Terminal guard failure')
        for key, color in (('tracking_limit_rad', '#777777'), ('tracking_abort_rad', '#B72A31')):
            if case['thresholds'][key] is not None:
                error.axhline(case['thresholds'][key], color=color, ls=':', lw=1, label=key)
        error.set_ylabel('Tracking error (rad)');error.legend(fontsize=7)
        tcp.plot(case['sample_time'], case['tcp_position'], color='#2878B5', label='TCP position error')
        tcp.set_ylabel('TCP target position error (m)', color='#2878B5')
        ori = tcp.twinx()
        orientation_axes.append(ori)
        ori.plot(case['sample_time'], case['tcp_orientation'], color='#D86435', label='TCP orientation error')
        ori.set_ylabel('TCP target orientation error (rad)', color='#D86435')
        if case['thresholds']['position_tolerance_m'] is not None:
            tcp.axhline(case['thresholds']['position_tolerance_m'], color='#2878B5', ls=':', lw=1)
        if case['thresholds']['orientation_tolerance_rad'] is not None:
            ori.axhline(case['thresholds']['orientation_tolerance_rad'], color='#D86435', ls=':', lw=1)
        tcp.set_xlabel('Native physics time from recorded execution start (s)')
        tcp.legend(fontsize=7, loc='upper left');ori.legend(fontsize=7, loc='upper right')
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
            axis.axvline(case['sample_time'][-1], color='#777777', ls=':', lw=.7)
        provenance.append({key: case[key] for key in ('result_path', 'result_sha256', 'trace_path',
            'trace_sha256', 'mode', 'strategy', 'operation', 'returned_success', 'failure_code',
            'origin_simulation_time', 'origin_source', 'thresholds')})
        provenance[-1]['label'] = label
    # Both left and right scales match across conditions. A smaller trace must
    # not look equally severe merely because each column rescales itself.
    for row in (gap_axes, orientation_axes):
        limits = [axis.get_ylim() for axis in row]
        for axis in row:
            axis.set_ylim(min(value[0] for value in limits), max(value[1] for value in limits))
    figure.suptitle('Recorded Home execution: plan, actual PhysX target buffers, guard errors and TCP return\n'
        'Only supplied measured records; no time warping, synthetic after-data or reconstructed acceptance errors.', fontsize=11)
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160);plt.close(figure)
    manifest = dict(schema_version=1, producer=dict(path=str(Path(__file__)), sha256=digest(__file__)),
        figure=dict(path=str(output), sha256=digest(output)), cases=provenance,
        method='Same-time command/actual pairs form exact q1 errors. The terminal failing guard error is recorded separately because the aborted step is absent from the planner actual trace. TCP is the existing recorded backend FK, independently cross-checked against native body transforms by the separate analysis helper.')
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--operation', default='home')
    parser.add_argument('--post-hold', action='store_true', help='Plot recorded no-action post-Home hold windows')
    parser.add_argument('--meters-per-unit', type=float, default=1.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0:
        parser.error('meters_per_unit must be finite and positive')
    if args.labels is not None and len(args.labels) != len(args.cases):
        parser.error('One explicit label is required per supplied case')
    cases = [extract_hold_case(path, meters_per_unit=args.meters_per_unit) if args.post_hold else
        extract_case(path, operation=args.operation, meters_per_unit=args.meters_per_unit) for path in args.cases]
    labels = args.labels or [f"{case['mode']} ({Path(case['result_path']).parent.name})" for case in cases]
    (render_holds if args.post_hold else render)(cases, labels, args.output)
    print(json.dumps(dict(cases=len(cases), output=args.output)))


if __name__ == '__main__':
    main()
