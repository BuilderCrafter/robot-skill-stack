#!/usr/bin/env python3
"""Post-run physical-body, native joint and target analysis; no Isaac imports.

This script does not move a robot, change a controller, or infer a motor torque.
Raw traces remain local; the output contains scalar diagnostic summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ARM = tuple(f'panda_joint{i}' for i in range(1, 8))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyzw_rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Physical pose requires a finite nonzero xyzw quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)


def rotation_error(first, last):
    return float(np.arccos(np.clip((np.trace(np.asarray(first).T@last)-1)/2, -1., 1.)))


def joint_axis_from_urdf(path, joint_name='panda_joint1'):
    """Establish the joint axis from authored mechanics, never measured q."""
    document = ET.parse(path).getroot()
    joint = next((x for x in document.findall('joint') if x.get('name') == joint_name), None)
    if joint is None or joint.get('type') not in ('continuous', 'revolute'):
        raise ValueError('Named revolute joint is absent from the supplied URDF')
    origin, axis = joint.find('origin'), joint.find('axis')
    rpy = np.fromstring('0 0 0' if origin is None else origin.get('rpy', '0 0 0'), sep=' ')
    vector = np.fromstring('1 0 0' if axis is None else axis.get('xyz', '1 0 0'), sep=' ')
    if (rpy.shape != (3,) or vector.shape != (3,) or not np.isfinite(rpy).all()
            or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-12):
        raise ValueError('URDF joint axis/origin must be finite')
    rotation = (axis_rotation((0, 0, 1), rpy[2]) @ axis_rotation((0, 1, 0), rpy[1])
                @ axis_rotation((1, 0, 0), rpy[0]))
    vector = rotation @ (vector/np.linalg.norm(vector))
    return dict(joint_name=joint_name, parent=joint.find('parent').get('link'),
        child=joint.find('child').get('link'), axis_in_parent=vector.tolist(),
        origin_rpy=rpy.tolist(), urdf_path=str(path), urdf_sha256=digest(path),
        method='URDF joint origin Rz(yaw)Ry(pitch)Rx(roll) times joint-local axis; no native q used to infer axis.')


def _body_index(paths, name):
    matches = [i for i, path in enumerate(paths) if path.rstrip('/').rsplit('/', 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f'Exactly one physical body named {name} is required')
    return matches[0]


def joint1_body_series(samples, metadata, axis_proof):
    """Independent parent/child-body angle delta and native encoder delta.

    Relative-body rotations remove root motion; subtracting the initial frame
    removes the child's fixed orientation. Authored joint axis supplies sign.
    """
    if not samples:
        raise ValueError('Physical yaw comparison requires samples')
    names, paths = metadata['robot_dof_names'], metadata['body_paths']
    index = names.index(axis_proof['joint_name'])
    parent = _body_index(paths, axis_proof['parent'])
    child = _body_index(paths, axis_proof['child'])
    axis = np.asarray(axis_proof['axis_in_parent'], float)
    axis = axis/np.linalg.norm(axis)
    relatives, q = [], []
    for row in samples:
        poses = np.asarray(row['poses'], float)
        relatives.append(xyzw_rotation(poses[parent, 3:]).T @ xyzw_rotation(poses[child, 3:]))
        q.append(float(row['feedback']['positions'][index]))
    if not np.isfinite(q).all():
        raise ValueError('Native joint encoder samples must be finite')
    relative = np.asarray(relatives)
    delta = relative @ relative[0].T
    sine = np.array([axis @ np.array([x[2, 1]-x[1, 2], x[0, 2]-x[2, 0], x[1, 0]-x[0, 1]])/2 for x in delta])
    cosine = (np.trace(delta, axis1=1, axis2=2)-1)/2
    angle = np.unwrap(np.arctan2(sine, cosine))
    native = np.unwrap(np.asarray(q))-q[0]
    residual = np.asarray([rotation_error(axis_rotation(axis, a), r) for a, r in zip(angle, delta)])
    return dict(physical_delta_rad=angle, native_delta_rad=native,
        difference_rad=angle-native, off_axis_residual_rad=residual)


def joint1_body_summary(samples, metadata, axis_proof):
    data = joint1_body_series(samples, metadata, axis_proof)
    return dict(samples=len(samples), physical_delta_rad=float(data['physical_delta_rad'][-1]),
        native_delta_rad=float(data['native_delta_rad'][-1]),
        maximum_physical_minus_native_delta_rad=float(np.max(np.abs(data['difference_rad']))),
        rms_physical_minus_native_delta_rad=float(np.sqrt(np.mean(data['difference_rad']**2))),
        maximum_off_axis_rotation_residual_rad=float(np.max(data['off_axis_residual_rad'])),
        method='Signed unwrapped angle of (Rparent.T Rchild)_t (Rparent.T Rchild)_0.T about authored parent-frame joint axis; physical xyzw quaternions normalized.')


def hand_fk_summary(samples, metadata, *, meters_per_unit):
    """Test consistency even when hand and FK frames have a fixed offset."""
    index = _body_index(metadata['body_paths'], 'panda_hand')
    offsets, rotations, absolute_position, absolute_rotation = [], [], [], []
    missing = 0
    for row in samples:
        fk = row.get('lula_model_fk', {}).get('frames', {}).get('panda_hand', {})
        if not fk.get('available'):
            missing += 1
            continue
        position, rotation = fk['value']
        physical = np.asarray(row['poses'][index], float)
        body_rotation = xyzw_rotation(physical[3:])
        difference = (np.asarray(position)-physical[:3])*meters_per_unit
        offsets.append(body_rotation.T@difference)
        rotations.append(body_rotation.T@np.asarray(rotation))
        absolute_position.append(float(np.linalg.norm(difference)))
        absolute_rotation.append(rotation_error(body_rotation, np.asarray(rotation)))
    if not offsets:
        return dict(samples=0, missing_samples=missing, available=False)
    offsets, rotations = np.asarray(offsets), np.asarray(rotations)
    return dict(available=True, samples=len(offsets), missing_samples=missing,
        maximum_absolute_position_difference_m=max(absolute_position),
        maximum_absolute_rotation_difference_rad=max(absolute_rotation),
        maximum_hand_to_fk_translation_offset_variation_m=float(np.linalg.norm(offsets-offsets[0], axis=1).max()),
        maximum_hand_to_fk_rotation_offset_variation_rad=max(rotation_error(rotations[0], x) for x in rotations),
        method='Physical hand body versus Lula FK of native joints; variation of Tphysical_hand^-1 Tmodel_hand distinguishes dynamic disagreement from a fixed frame offset.')


def _arm_values(action, field, metadata):
    value = action.get(field)
    if value is None:
        return None
    values = np.asarray(value, float).reshape(-1)
    indices = action.get('joint_indices')
    names = metadata['robot_dof_names']
    arm_indices = [names.index(name) for name in ARM]
    if indices is None:
        if len(values) != len(names):
            return None
        return values[arm_indices]
    if len(indices) != len(values) or len(set(map(int, indices))) != len(indices):
        return None
    lookup = dict(zip(map(int, indices), values))
    return np.asarray([lookup[i] for i in arm_indices]) if all(i in lookup for i in arm_indices) else None


def target_summaries(samples, actions, metadata):
    desired_gaps, velocity_gaps, actual_gaps = [], [], []
    names = metadata['robot_dof_names']
    indices = [names.index(name) for name in ARM]
    for row in actions:
        applied = row.get('actual_physx_targets', {})
        if not applied.get('available'):
            continue
        requested = dict(joint_positions=row.get('requested_positions'),
            joint_velocities=row.get('requested_velocities'), joint_indices=row.get('requested_indices'))
        for field, values in (('joint_positions', desired_gaps), ('joint_velocities', velocity_gaps)):
            first, last = _arm_values(requested, field, metadata), _arm_values(applied['value'], field, metadata)
            if first is not None and last is not None:
                values.append(last-first)
    for row in samples:
        applied = row.get('live_drive_feedback', {}).get('robot_applied_action', {})
        if not applied.get('available'):
            continue
        target = _arm_values(applied['value'], 'joint_positions', metadata)
        if target is not None:
            actual_gaps.append(np.asarray(row['feedback']['positions'], float)[indices]-target)
    def summarize(values):
        if not values:
            return dict(samples=0, available=False)
        values = np.asarray(values)
        return dict(available=True, samples=len(values), maximum_absolute=float(np.abs(values).max()),
            per_joint_maximum_absolute=np.abs(values).max(axis=0).tolist(), final=values[-1].tolist())
    return dict(requested_vs_actual_physx_positions_rad=summarize(desired_gaps),
        requested_vs_actual_physx_velocities_rad_s=summarize(velocity_gaps),
        measured_vs_latched_physx_position_targets_rad=summarize(actual_gaps),
        meaning='Applied-buffer agreement uses exact action readback. Measured-minus-latched-target samples differ from time-matched planned-trajectory tracking errors.')


def _matrix(value, dimension):
    array = np.asarray(value, float)
    while array.ndim > dimension and array.shape[0] == 1:
        array = array[0]
    return array


def control_state_summary(samples, metadata):
    """Distinguish authored solver flags from native drive readbacks."""
    fields = {
        'dynamics': ('get_solver_position_iteration_counts', 'get_solver_velocity_iteration_counts',
            'get_enabled_self_collisions', 'get_drive_types', 'get_dof_drive_model_properties'),
        'live_drive_feedback': ('controller_get_gains', 'controller_get_effort_modes'),
    }
    result = {}
    authored_fields = {'get_solver_position_iteration_counts', 'get_solver_velocity_iteration_counts',
        'get_enabled_self_collisions', 'controller_get_effort_modes'}
    for source, names in fields.items():
        for name in names:
            readings, unavailable = [], set()
            for row in samples:
                item = row.get(source, {}).get(name, {})
                if item.get('available'):
                    readings.append(item['value'])
                else:
                    unavailable.add(item.get('error', 'Not recorded'))
            if not readings:
                result[name] = dict(available=False, samples=0, unavailable_reasons=sorted(unavailable))
                continue
            encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in readings]
            # Public count/self-collision getters read USD attributes, not
            # native solver state. Native gains remain in native DOF order.
            unique = list(dict.fromkeys(encoded))
            result[name] = dict(available=True, samples=len(readings),
                distinct_states=len(unique), unchanged=len(unique) == 1,
                states=[dict(value=json.loads(value), samples=encoded.count(value)) for value in unique],
                provenance=('USD DriveAPI type attribute; public effort-mode getter can apply DriveAPI if absent'
                    if name == 'controller_get_effort_modes' else
                    'USD attribute through public getter; actual native/island solver state unverified'
                    if name in authored_fields else 'Native drive buffer through recorded SDK getter'),
                unavailable_reasons=sorted(unavailable))
    arm_indices = [metadata['robot_dof_names'].index(name) for name in ARM]
    effort_readings = {}
    for name in ('measured_efforts', 'explicit_applied_efforts'):
        values = [np.asarray(row['feedback'][name], float)[arm_indices] for row in samples
            if row.get('feedback', {}).get(name) is not None]
        if values and not np.isfinite(values).all():
            raise ValueError('Invalid recorded effort signal')
        effort_readings[name] = dict(available=bool(values), samples=len(values),
            maximum_absolute=None if not values else float(np.abs(values).max()),
            per_arm_joint_maximum_absolute=None if not values else np.abs(values).max(axis=0).tolist())
    return dict(native_dof_names=metadata['robot_dof_names'], arm_names=list(ARM),
        observed_states=result, effort_signals=effort_readings,
        meaning='Solver-count, self-collision and effort-mode public getters read USD attributes; actual native solver counts can differ due to scene/island minima. Native drive readbacks are labeled separately. Measured joint efforts are reaction signals; explicit effort buffers omit implicit position/velocity drive forces. Neither alone is calibrated actuator torque.')


def dynamics_summary(samples, metadata):
    """Read-only native model quantities; not calibrated motor-drive torques."""
    indices = [metadata['robot_dof_names'].index(name) for name in ARM]
    availability, errors = Counter(), {}
    condition, pair_condition, correlation, mineig, symmetry = [], [], [], [], []
    signals = {'get_coriolis_and_centrifugal_forces': [], 'get_generalized_gravity_forces': []}
    for row in samples:
        dynamics = row.get('dynamics', {})
        for key, item in dynamics.items():
            if item.get('available'):
                availability[key] += 1
            else:
                errors.setdefault(key, set()).add(item.get('error', 'Unavailable'))
        mass = dynamics.get('get_mass_matrices', {})
        if mass.get('available'):
            matrix = _matrix(mass['value'], 2)[np.ix_(indices, indices)]
            if matrix.shape != (7, 7) or not np.isfinite(matrix).all():
                raise ValueError('Invalid measured arm mass matrix')
            symmetry.append(float(np.max(np.abs(matrix-matrix.T))))
            eigen = np.linalg.eigvalsh((matrix+matrix.T)/2)
            mineig.append(float(eigen.min()))
            condition.append(float(np.linalg.cond(matrix)))
            pair = matrix[np.ix_((0, 2), (0, 2))]
            pair_condition.append(float(np.linalg.cond(pair)))
            correlation.append(float(matrix[0, 2]/math.sqrt(matrix[0, 0]*matrix[2, 2])))
        for key in signals:
            item = dynamics.get(key, {})
            if item.get('available'):
                signal = _matrix(item['value'], 1)[indices]
                if not np.isfinite(signal).all():
                    raise ValueError('Invalid measured dynamics signal')
                signals[key].append(signal)
    def extent(values):
        return None if not values else dict(minimum=float(min(values)), maximum=float(max(values)),
            initial=float(values[0]), final=float(values[-1]))
    return dict(samples=len(samples), availability_counts=dict(availability),
        unavailable_reasons={k: sorted(v) for k, v in errors.items()},
        arm_mass_matrix_condition_number=extent(condition),
        joint1_joint3_mass_submatrix_condition_number=extent(pair_condition),
        joint1_joint3_mass_correlation=extent(correlation),
        minimum_arm_mass_eigenvalue=extent(mineig), mass_matrix_symmetry_error=extent(symmetry),
        signal_summaries={key: None if not values else dict(samples=len(values),
            maximum_absolute=float(np.abs(values).max()), joint1_maximum_absolute=float(np.abs(np.asarray(values)[:, 0]).max()))
            for key, values in signals.items()},
        interpretation='Configuration-dependent native model coupling, not a proven failure cause. Tiny wrist inertia can dominate full matrix condition; joint1/joint3 submatrix is shown separately. Signals are not calibrated motor torque.')


def analyze_case(path, axis_proof, *, meters_per_unit=1.):
    path = Path(path)
    payload = json.loads(path.read_text())
    controlled = payload['controlled']
    trace_path = Path(controlled['raw_trace']['path'])
    if not trace_path.exists():
        trace_path = path.parent/'controlled_trace.json'
    if digest(trace_path) != controlled['raw_trace']['sha256']:
        raise ValueError('Controlled trace does not match its recorded SHA')
    document = json.loads(trace_path.read_text())
    metadata = document['metadata']
    recorded = {op['strategy_diagnostics'].get('sdk_configuration_sha256', {}).get('urdf_path')
        for op in document['operations'] if op.get('strategy_diagnostics')}
    recorded.discard(None)
    if recorded and recorded != {axis_proof['urdf_sha256']}:
        raise ValueError('Joint-axis proof does not match the recorded planner URDF')
    rows = []
    for op in document['operations']:
        samples = document['samples'][op['first_sample']:op['last_sample']]
        actions = document['actions'][op['first_action']:op['last_action']]
        if not samples:
            rows.append(dict(id=op['id'], label=op['label'], samples=0, analysis_error='No samples'))
            continue
        rows.append(dict(id=op['id'], label=op['label'], kind=op['kind'], cancelled=op['cancelled'],
            returned_success=op['returned_success'], simulation_seconds=float(samples[-1]['simulation_time']-samples[0]['simulation_time']),
            first_physics_step=samples[0]['physics_step'], last_physics_step=samples[-1]['physics_step'],
            physical_joint1=joint1_body_summary(samples, metadata, axis_proof),
            physical_hand_vs_model_fk=hand_fk_summary(samples, metadata, meters_per_unit=meters_per_unit),
            targets=target_summaries(samples, actions, metadata), dynamics=dynamics_summary(samples, metadata),
            control_state=control_state_summary(samples, metadata),
            original_strategy_diagnostics={key: op['strategy_diagnostics'].get(key) for key in
                ('failure_code', 'tracking_error_rad', 'joint_tracking_error_rad', 'trajectory_duration_seconds',
                 'position_error_m', 'orientation_error_rad') if key in (op.get('strategy_diagnostics') or {})}))
    return dict(artifact=str(path), artifact_sha256=digest(path), trace=str(trace_path), trace_sha256=digest(trace_path),
        mode=controlled['mode'], strategy=controlled['strategy'], measurement_status=payload['status'],
        acceptance_passed=controlled['acceptance_passed'], units=dict(meters_per_unit=meters_per_unit,
            source='Explicit matched-stage analysis input; all original V4.1 scenes use meters'),
        axis_matches_recorded_global_urdf=True if recorded else None, operations=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--meters-per-unit', type=float, default=1.)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0:
        parser.error('meters_per_unit must be finite and positive')
    proof = joint_axis_from_urdf(args.urdf)
    cases = [analyze_case(path, proof, meters_per_unit=args.meters_per_unit) for path in args.cases]
    output = dict(schema_version=1, status='POST_RUN_OBSERVATIONAL_ANALYSIS', axis_proof=proof,
        producer=dict(path=str(Path(__file__)), sha256=digest(__file__)), cases=cases,
        meaning='Independent physical-body deltas, native readback, target-buffer and model-coupling observations. Analysis PASS is not a motion or root-cause acceptance claim.')
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(cases), bytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
