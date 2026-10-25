#!/usr/bin/env python3
"""Installed public Lula response to actual captured V3 obstacle packets.

This is a bounded CPU-only static policy experiment, not physical robot motion.
The same recorded q, declared zero qdot, target and geometry perturbation are
used for every representation. Natural force/metric are not physical forces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TARGET = [.45, .25, .35]
ORIENTATION = [0., 0., 1., 0.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_postures(capture_path, names=('home', 'raised', 'low')):
    source = json.loads(Path(capture_path).read_text())
    records = source.get('snapshots') or []
    rows = []
    for name in names:
        selected = [row for row in records if row.get('posture', row.get('id')) == name]
        if len(selected) != 1 or len(selected[0].get('joints', [])) < 7:
            raise ValueError('A unique recorded arm posture is required: '+name)
        values = selected[0]
        rows.append(dict(id=name, q=values['joints'][:7], qd=[0.]*7,
            target=TARGET, orientation_wxyz=ORIENTATION,
            source_simulation_time=values.get('simulation_time'), actual_tcp_position=values.get('tcp_position')))
    return dict(schema_version=1, postures=rows,
        source=dict(path=str(capture_path), sha256=sha(capture_path)),
        scope='q values were captured after legitimate native robot movements. qdot=0 is an explicit static-rest condition, not recorded dynamic velocity. These q values are not asserted exposure-matched to each occupancy snapshot.')


def choose_cases(bundle, *, case_ids=None, maximum=4):
    if maximum < 1:
        raise ValueError('A positive bounded case count is required')
    values = bundle.get('cases') or []
    if case_ids:
        values = [row for row in values if row.get('case_id') in case_ids]
        if set(case_ids) != {row.get('case_id') for row in values}:
            raise ValueError('An explicitly requested captured packet case is missing')
    return values[:maximum]


def validate_packet(packet, *, evaluated_at=None):
    boxes = packet.get('boxes') or []
    if len(boxes) > 128:
        raise ValueError('Static probe is bounded to128numeric cuboids per packet')
    if len({box['id'] for box in boxes}) != len(boxes):
        raise ValueError('Actual packet box identities must be unique')
    for box in boxes:
        for key in ('center', 'dimensions'):
            value = box.get(key)
            if (not isinstance(value, (tuple, list)) or len(value) != 3
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
                raise ValueError('Numeric finite XYZ packet geometry is required')
        if any(v <= 0 for v in box['dimensions']):
            raise ValueError('Numeric cuboids require positive side lengths')
        if evaluated_at is not None and box.get('expires_at', evaluated_at+1) <= evaluated_at:
            raise ValueError('Already-expired geometry must not enter the static packet probe')
    return boxes


def perturbations(millimeters):
    if not math.isfinite(millimeters) or not 0 < millimeters <= 2:
        raise ValueError('Declared sensitivity displacement must be within(0,2]mm')
    values = [dict(id='nominal', translation_m=[0., 0., 0.])]
    for axis, name in enumerate('xyz'):
        for sign in (-1, 1):
            displacement = [0., 0., 0.]
            displacement[axis] = sign*millimeters/1000
            values.append(dict(id=name+('-' if sign < 0 else '+'), translation_m=displacement))
    return values


def recorded_base(result_path, trace_path=None):
    """Use the actual rigid-body URDF base, with a fixed-base drift check."""
    result_path = Path(result_path)
    trace_path = Path(trace_path) if trace_path else result_path.parent/'trace.json'
    result = json.loads(result_path.read_text())
    trace = json.loads(trace_path.read_text())
    paths = ((result.get('metrics') or {}).get('capture') or {}).get('body_paths') or []
    if paths.count('/Franka/panda_link0') != 1 or not trace.get('samples'):
        raise ValueError('Actual recorded panda_link0 body poses are required for world/base conversion')
    index = paths.index('/Franka/panda_link0')
    poses = [row['poses'][index] for row in trace['samples']]
    if any(len(pose) != 7 or any(not math.isfinite(value) for value in pose) for pose in poses):
        raise ValueError('Finite actual XYZ/xyzw base poses are required')
    first = poses[0]
    drift = max(abs(a-b) for pose in poses for a, b in zip(first, pose))
    if drift > 1e-6:
        raise ValueError('This static probe requires a fixed recorded robot base')
    x, y, z, w = first[3:]
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-10:
        raise ValueError('Actual recorded base orientation is invalid')
    x, y, z, w = [value/norm for value in (x, y, z, w)]
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return dict(translation_world_m=first[:3], world_from_base_rotation=rotation,
        urdf_base_frame='panda_link0', maximum_pose_component_drift=drift, samples=len(poses),
        result=dict(path=str(result_path), sha256=sha(result_path)),
        trace=dict(path=str(trace_path), sha256=sha(trace_path)),
        scope='Actual fixed URDF base rigid-body pose; world cuboids, perturbations and targets are transformed into this base frame before standalone Lula queries.')


def selected_cell_union(packet, snapshot):
    """Identify equal selected unions across different exact cuboid partitions."""
    size = snapshot['voxel_size']
    minimum = snapshot['bounds_min']
    cells = set()
    for box in packet.get('boxes') or []:
        bounds = []
        for axis in range(3):
            low = (box['center'][axis]-.5*box['dimensions'][axis]-minimum[axis])/size
            high = (box['center'][axis]+.5*box['dimensions'][axis]-minimum[axis])/size
            if abs(low-round(low)) > 1e-6 or abs(high-round(high)) > 1e-6 or high <= low:
                raise ValueError('Actual selected packet cuboids must align to the recorded voxel grid')
            bounds.append((round(low), round(high)))
        count = math.prod(high-low for low, high in bounds)
        if count > 2_000_000 or len(cells)+count > 2_000_000:
            raise ValueError('Static selected-cell union audit exceeds its explicit2M-cell bound')
        cells.update((x, y, z) for x in range(*bounds[0]) for y in range(*bounds[1]) for z in range(*bounds[2]))
    return frozenset(cells)


def identical_cuboids(first, second):
    def keys(packet):
        return sorted(tuple(box['center'])+tuple(box['dimensions']) for box in packet.get('boxes') or [])
    return keys(first) == keys(second)


def negative_controls(rows, unions, *, absolute_tolerance=1e-12, relative_tolerance=1e-12):
    """Identical numeric cubes should reproduce the same public policy output."""
    baseline = {(row['case_id'], row['posture'], row['perturbation']['id']): row
        for row in rows if row['representation'] == 'baseline'}
    identical = {(row['case_id'], row['representation']) for row in unions
        if row.get('identical_numeric_cuboids_as_baseline') is True and row['representation'] != 'baseline'}
    fields = ('natural_policy_force', 'natural_policy_metric', 'canonical_acceleration_rad_s2')
    def flatten(value):
        if isinstance(value, (list, tuple)):
            return [item for part in value for item in flatten(part)]
        return [value]
    checks = []
    for row in rows:
        if (row['case_id'], row['representation']) not in identical:
            continue
        original = baseline.get((row['case_id'], row['posture'], row['perturbation']['id']))
        differences, passed = {}, original is not None
        if original is not None:
            for field in fields:
                first, last = flatten(original[field]), flatten(row[field])
                equal_length = len(first) == len(last)
                delta = max((abs(a-b) for a, b in zip(first, last)), default=0.) if equal_length else None
                differences[field] = delta
                passed = passed and equal_length and all(abs(a-b) <= absolute_tolerance+
                    relative_tolerance*max(abs(a), abs(b)) for a, b in zip(first, last))
        checks.append(dict(case_id=row['case_id'], representation=row['representation'], posture=row['posture'],
            perturbation=row['perturbation']['id'], matched=bool(passed), maximum_absolute_differences=differences))
    return dict(status='NOT_TESTED' if not checks else 'PASS' if all(row['matched'] for row in checks) else 'FAIL',
        comparisons=len(checks), absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
        checks=checks, scope='Identical numerical cuboids, q, qdot, target and displacement only. Equal cell unions with different partitions are not assumed policy-equivalent.')


def run(args, bundle, postures):
    import numpy as np
    if np.__version__ != '1.26.4':
        raise RuntimeError('Installed Lula requires NumPy1.26.4; use ./run_isaac_python.sh')
    sdk = Path(os.environ.get('ISAAC_SIM', str(Path.home()/'isaacsim'))).expanduser().resolve()
    bindings = sdk/'exts/isaacsim.robot_motion.lula/pip_prebundle'
    config_dir = sdk/'exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka'
    files = dict(robot_descriptor=config_dir/'rmpflow/robot_descriptor.yaml',
        robot_urdf=config_dir/'lula_franka_gen.urdf',
        policy_config=config_dir/'rmpflow/franka_rmpflow_common.yaml',
        integration_config=config_dir/'rmpflow/config.json')
    if not bindings.is_dir() or any(not path.is_file() for path in files.values()):
        raise FileNotFoundError('Installed public Lula/Franka configuration is unavailable')
    sys.path.insert(0, str(bindings))
    import lula
    base = recorded_base(args.base_reference_result, args.base_trace)
    base_from_world_rotation = np.asarray(base['world_from_base_rotation'], float).T
    base_world_position = np.asarray(base['translation_world_m'], float)

    def local_point(position):
        return base_from_world_rotation@(np.asarray(position, float)-base_world_position)

    robot = lula.load_robot(str(files['robot_descriptor']), str(files['robot_urdf']))
    end_frame = json.loads(files['integration_config'].read_text())['end_effector_frame_name']

    def make_policy(boxes):
        started = time.perf_counter()
        world = lula.create_world()
        view = world.add_world_view()
        config = lula.create_rmpflow_config(str(files['policy_config']), robot, end_frame, view)
        policy = lula.create_rmpflow(config)
        construction = time.perf_counter()-started
        started = time.perf_counter()
        handles = []
        for box in boxes:
            obstacle = lula.create_obstacle(lula.Obstacle.Type.CUBE)
            obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(box['dimensions'], float))
            handles.append(world.add_obstacle(obstacle,
                lula.Pose3(lula.Rotation3(base_from_world_rotation), local_point(box['center']))))
        registration = time.perf_counter()-started
        return world, view, policy, handles, construction, registration

    def evaluate(policy, posture):
        q, qd = np.asarray(posture['q'], float), np.asarray(posture.get('qd', [0.]*7), float)
        if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError('Static policy conditions require finite7-dimensional q andqd')
        policy.set_end_effector_position_attractor(local_point(posture['target']))
        target_world_rotation = lula.Rotation3(*posture['orientation_wxyz']).matrix()
        policy.set_end_effector_orientation_attractor(lula.Rotation3(base_from_world_rotation@target_world_rotation))
        started = time.perf_counter()
        force, metric = policy.eval_force_and_metric(q, qd)
        acceleration = np.empty_like(q)
        policy.eval_accel(q, qd, acceleration)
        seconds = time.perf_counter()-started
        if not all(np.isfinite(value).all() for value in (force, metric, acceleration)):
            raise RuntimeError('Installed Lula returned a nonfinite static policy output')
        return np.asarray(force), np.asarray(metric), acceleration, seconds

    empty_world, empty_view, empty_policy, _, _, _ = make_policy([])
    empty_policy.update_world_view()
    empty = {posture['id']: evaluate(empty_policy, posture) for posture in postures}
    rows, missing, union_comparisons = [], [], []
    selected_cases = choose_cases(bundle, case_ids=args.case_ids, maximum=args.max_cases)
    for case in selected_cases:
        packets = case.get('packets') or {}
        baseline_cells = None if 'baseline' not in packets else selected_cell_union(packets['baseline'], case['snapshot'])
        names = args.representations or (['baseline']+sorted(name for name in packets if name != 'baseline')[:2])
        for name in names:
            if name not in packets:
                missing.append(dict(case_id=case['case_id'], representation=name))
                continue
            boxes = validate_packet(packets[name], evaluated_at=(case.get('snapshot') or {}).get('evaluated_at'))
            cells = selected_cell_union(packets[name], case['snapshot'])
            union = dict(case_id=case['case_id'], representation=name, selected_cells=len(cells),
                identical_numeric_cuboids_as_baseline=None if 'baseline' not in packets else identical_cuboids(packets[name], packets['baseline']),
                same_selected_cell_union_as_baseline=None if baseline_cells is None else cells == baseline_cells,
                baseline_only_cells=None if baseline_cells is None else len(baseline_cells-cells),
                candidate_only_cells=None if baseline_cells is None else len(cells-baseline_cells))
            union_comparisons.append(union)
            world, view, policy, handles, construction, registration = make_policy(boxes)
            for displacement in perturbations(args.perturbation_mm):
                offset = np.asarray(displacement['translation_m'], float)
                before = time.perf_counter()
                for handle, box in zip(handles, boxes):
                    world.set_pose(handle, lula.Pose3(lula.Rotation3(base_from_world_rotation),
                        local_point(np.asarray(box['center'])+offset)))
                pose_update = time.perf_counter()-before
                before = time.perf_counter()
                policy.update_world_view()
                world_update = time.perf_counter()-before
                for posture in postures:
                    force, metric, acceleration, evaluation = evaluate(policy, posture)
                    base_force, base_metric, base_acceleration, _ = empty[posture['id']]
                    delta_force, delta_metric = force-base_force, metric-base_metric
                    rows.append(dict(case_id=case['case_id'], layout_id=case.get('layout_id'), scenario=case.get('scenario'),
                        snapshot=case.get('snapshot'), representation=name, posture=posture['id'],
                        same_selected_cell_union_as_baseline=union['same_selected_cell_union_as_baseline'],
                        perturbation=displacement, registered_obstacles=int(view.num_enabled_obstacles()),
                        natural_policy_force=force.tolist(), natural_policy_metric=metric.tolist(),
                        canonical_acceleration_rad_s2=acceleration.tolist(),
                        collision_force_delta=delta_force.tolist(), collision_metric_delta=delta_metric.tolist(),
                        collision_force_norm=float(np.linalg.norm(delta_force)),
                        collision_metric_norm=float(np.linalg.norm(delta_metric)),
                        canonical_acceleration_norm=float(np.linalg.norm(acceleration)),
                        policy_construction_ms=construction*1000, registration_ms=registration*1000,
                        numeric_pose_update_ms=pose_update*1000, world_update_ms=world_update*1000,
                        policy_evaluation_ms=evaluation*1000))
    nominal = {(row['case_id'], row['representation'], row['posture']): row for row in rows
        if row['perturbation']['id'] == 'nominal'}
    def direction(first, second):
        first, second = np.asarray(first), np.asarray(second)
        norm = np.linalg.norm(first)*np.linalg.norm(second)
        return None if norm <= 1e-20 else float(np.clip(first@second/norm, -1., 1.))
    for row in rows:
        own = nominal[(row['case_id'], row['representation'], row['posture'])]
        baseline = nominal.get((row['case_id'], 'baseline', row['posture']))
        row['perturbation_acceleration_delta_norm'] = float(np.linalg.norm(np.asarray(
            row['canonical_acceleration_rad_s2'])-np.asarray(own['canonical_acceleration_rad_s2'])))
        row['joint_acceleration_direction_cosine_to_baseline'] = None if baseline is None else direction(
            row['canonical_acceleration_rad_s2'], baseline['canonical_acceleration_rad_s2'])
        row['collision_force_direction_cosine_to_baseline'] = None if baseline is None else direction(
            row['collision_force_delta'], baseline['collision_force_delta'])
    files['lula_binding'] = Path(lula.__file__)
    files['probe'] = Path(__file__)
    controls = negative_controls(rows, union_comparisons)
    return dict(schema_version=1, status='PASS' if rows and not missing and controls['status'] != 'FAIL' else 'PARTIAL_FAILURE',
        scope='Static public installed Lula CPU queries only; no SimulationApp, GPU rendering, articulation or physical movement.',
        packet_bundle=dict(path=str(args.packet_bundle), sha256=sha(args.packet_bundle)),
        postures=postures, base_frame_reference=base,
        selected_case_ids=[case['case_id'] for case in selected_cases],
        requested_representations=args.representations, missing_packets=missing, rows=rows,
        selected_union_comparisons=union_comparisons,
        identical_geometry_negative_control=controls,
        empty_world_baselines={name:dict(natural_force=values[0].tolist(), metric=values[1].tolist(),
            canonical_acceleration_rad_s2=values[2].tolist()) for name, values in empty.items()},
        source_hashes={name:dict(path=str(path), sha256=sha(path)) for name, path in files.items()},
        limitations=['Natural force/metric are policy terms, not physical Newtons, torques or inertia.',
            'Joint-acceleration direction is not Cartesian TCP direction or a measured robot trajectory.',
            'Actual captured packets share one snapshot per case; different budgeted selections need not have equal geometric unions.',
            'The same declared small translation is applied to every representation as a counterfactual sensitivity probe; perturbed boxes are never published to RGB-D runtime.',
            'Recorded postures and declared static qdot are matched across policies but not claimed exposure-matched to occupancy inputs.',
            'Timing is CPU-query/registration overhead, not native Kit control or source freshness. Finite policy outputs do not establish obstacle avoidance.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-bundle', type=Path)
    parser.add_argument('--postures-json', type=Path)
    parser.add_argument('--base-reference-result', type=Path)
    parser.add_argument('--base-trace', type=Path)
    parser.add_argument('--posture-capture', type=Path)
    parser.add_argument('--posture-ids', nargs='+', default=['home', 'raised', 'low'])
    parser.add_argument('--export-postures', type=Path)
    parser.add_argument('--representations', nargs='+')
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--max-cases', type=int, default=4)
    parser.add_argument('--perturbation-mm', type=float, default=1.)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args(argv)
    if args.export_postures:
        if args.posture_capture is None:
            parser.error('--posture-capture is required for a pure recorded-posture export')
        document = export_postures(args.posture_capture, args.posture_ids)
        args.export_postures.parent.mkdir(parents=True, exist_ok=True)
        args.export_postures.write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
        if not args.packet_bundle:
            print(json.dumps(dict(postures=len(document['postures']), exported=str(args.export_postures))))
            return
    else:
        if args.postures_json is None:
            parser.error('--postures-json or --export-postures is required')
        document = json.loads(args.postures_json.read_text())
    if args.packet_bundle is None or args.result is None or args.base_reference_result is None:
        parser.error('--packet-bundle, --base-reference-result and --result are required for actual static policy queries')
    report = run(args, json.loads(args.packet_bundle.read_text()), document['postures'])
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], policy_evaluations=len(report['rows']))))


if __name__ == '__main__':
    main()
