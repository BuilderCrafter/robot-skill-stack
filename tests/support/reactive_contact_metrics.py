"""Observational contact and joint-feedback metrics for native experiments.

No function applies actions, changes physics, or infers contact from tracking
error alone. Reductions run after the measured motion. Contact buffers queried
with SDK ``dt=1`` contain normal impulses in N s, not forces in N; tangential
friction is outside this metric. The existing V2 strict result is an input and
is retained separately from descriptive severity measurements.
"""
from collections import defaultdict

import numpy as np


def _numbers(value, shape=None):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(array).all() or shape is not None and array.shape != shape:
        return None
    return array


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), minimum=float(values.min()), mean=float(values.mean()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=float(values.max()))


def _pose_matrices(poses, scale):
    poses = _numbers(poses)
    if poses is None or poses.ndim != 2 or poses.shape[1] != 7:
        return None
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(norms < 1e-12):
        return None
    x, y, z, w = (poses[:, 3:] / norms[:, None]).T
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]).transpose(2, 0, 1)
    result[:, :3, 3] = poses[:, :3] * scale
    return result


def contact_point_velocity(points_world_m, world_from_body, local_com_m, velocity_com_world):
    """World contact-point velocity from COM linear/angular SDK velocities.

    Raw RigidBodyView velocities use the center of mass for their linear part.
    Using the body origin instead gives incorrect speeds for rotating links.
    """
    points = np.asarray(points_world_m, dtype=float).reshape(-1, 3)
    transform = np.asarray(world_from_body, dtype=float)
    com = transform[:3, :3] @ np.asarray(local_com_m, dtype=float) + transform[:3, 3]
    velocity = np.asarray(velocity_com_world, dtype=float)
    return velocity[:3] + np.cross(velocity[3:], points-com)


def _step_duration(sample, previous):
    """Verify one observed SDK physics step before converting impulse to force."""
    dt = _numbers(sample.get('physics_dt_s'))
    if dt is None or dt.shape != () or float(dt) <= 0:
        return None, 'missing_or_invalid_sdk_physics_dt'
    if previous is None:
        return None, 'no_previous_physics_sample'
    before, after = previous.get('physics_step'), sample.get('physics_step')
    if not isinstance(before, (int, np.integer)) or not isinstance(after, (int, np.integer)):
        return None, 'missing_physics_step_index'
    if after-before != 1:
        return None, 'physics_steps_skipped_or_repeated'
    times = _numbers([previous.get('simulation_time'), sample.get('simulation_time')])
    if times is None or not np.isclose(times[1]-times[0], float(dt), rtol=1e-5, atol=1e-8):
        return None, 'simulation_time_does_not_match_sdk_physics_dt'
    return float(dt), None


def capture_joint_feedback(robot, action=None, *, controller_velocity_feedback=None):
    """Copy bounded public SDK feedback without adding a sensor or controller.

    Missing/invalid SDK handles remain unavailable. Measured efforts and child-
    joint-frame reaction wrenches include normal dynamics and controller loads;
    applied efforts are explicit effort commands, not inferred drive torque.
    """
    result, errors = {}, {}
    methods = dict(positions='get_joint_positions', velocities='get_joint_velocities',
                   measured_efforts='get_measured_joint_efforts',
                   explicit_applied_efforts='get_applied_joint_efforts',
                   incoming_joint_wrenches='get_measured_joint_forces')
    for name, method in methods.items():
        try:
            value = _numbers(getattr(robot, method)())
            result[name] = None if value is None else value.copy().tolist()
            if value is None:
                errors[name] = 'unavailable_or_nonfinite'
        except Exception as exc:
            result[name] = None
            errors[name] = f'{type(exc).__name__}: {exc}'
    if action is None:
        try:
            action = robot.get_applied_action()
        except Exception as exc:
            errors['action'] = f'{type(exc).__name__}: {exc}'
    result['action'] = {}
    for name in ('joint_positions', 'joint_velocities', 'joint_efforts', 'joint_indices'):
        value = getattr(action, name, None)
        result['action'][name] = None if value is None else np.asarray(value).copy().tolist()
    feedback = _numbers(controller_velocity_feedback)
    result['controller_velocity_feedback'] = None if feedback is None else feedback.copy().tolist()
    result['errors'] = errors
    return result


def analyze_joint_feedback(samples, *, arm_joint_indices=tuple(range(7))):
    """Compare arm targets/feedback observationally; never classify contact."""
    tracking, velocity, efforts, wrenches = [], [], [], []
    available, unavailable = defaultdict(int), defaultdict(int)
    for sample in samples:
        feedback = sample.get('joint_feedback') or {}
        for field in ('positions', 'velocities', 'measured_efforts',
                      'explicit_applied_efforts', 'incoming_joint_wrenches'):
            (available if _numbers(feedback.get(field)) is not None else unavailable)[field] += 1
        positions = _numbers(feedback.get('positions'))
        action = feedback.get('action') or {}
        targets = action.get('joint_positions')
        if positions is not None and positions.ndim == 1 and targets is not None:
            indices = action.get('joint_indices')
            if indices is None and len(targets) == len(positions):
                indices = range(len(positions))
            if indices is not None and len(indices) == len(targets):
                for index, target in zip(indices, targets):
                    target = _numbers(target)
                    if (isinstance(index, (int, np.integer)) and index in arm_joint_indices
                            and 0 <= index < len(positions) and target is not None and target.shape == ()):
                        tracking.append(abs(float(target)-positions[index]))
        values = _numbers(feedback.get('velocities'))
        control = _numbers(feedback.get('controller_velocity_feedback'))
        if values is not None and control is not None and values.ndim == control.ndim == 1:
            for local, index in enumerate(arm_joint_indices):
                if index < len(values) and local < len(control):
                    velocity.append(abs(values[index]-control[local]))
        values = _numbers(feedback.get('measured_efforts'))
        if values is not None and values.ndim == 1:
            efforts.extend(abs(values[index]) for index in arm_joint_indices if index < len(values))
        values = _numbers(feedback.get('incoming_joint_wrenches'))
        if values is not None and values.ndim == 2 and values.shape[1] == 6:
            wrenches.extend(np.linalg.norm(values[:, :3], axis=1))
    return dict(available_samples=dict(available), unavailable_samples=dict(unavailable),
                arm_position_tracking_error_rad=_distribution(tracking),
                raw_vs_controller_arm_velocity_error_rad_s=_distribution(velocity),
                arm_solver_projected_effort_abs_nm=_distribution(efforts),
                incoming_joint_reaction_force_norm_n=_distribution(wrenches),
                contact_detector_validated=False,
                interpretation='Tracking error, effort and joint reaction loads also arise from gravity, inertia, drives and friction. They do not independently identify unexpected contact or a wrist external wrench.')


def analyze_contact_samples(samples, body_paths, local_com_poses=None, *, meters_per_unit=1.,
                            goal_status=None, failure_code=None, minimum_clearance_m=None,
                            strict_v2_success=None):
    """Reduce copied physics observations without choosing permissive limits.

    A sample has simulation_time, physics_step, physics_dt_s, poses (XYZ/XYZW),
    body_velocities (world COM linear/angular), contacts and optional joint_feedback.
    Each contact has sensor/obstacle paths and optional normal_impulses_ns,
    points_world_m, normals_world and separations_m arrays. Legacy V2 count,
    minimum_separation_m and impulse_magnitude still support raw depth/impulse
    reporting, but cannot recover per-point speeds or forces.
    """
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('meters_per_unit must be finite and positive')
    indices = {path: index for index, path in enumerate(body_paths)}
    coms = _numbers(local_com_poses)
    if coms is not None:
        if coms.ndim != 2 or coms.shape[0] != len(body_paths) or coms.shape[1] not in (3, 7):
            raise ValueError('Local COM poses must correspond to body_paths')
        coms = coms[:, :3] * meters_per_unit
    unique = {}
    for number, sample in enumerate(samples):
        key = (sample.get('physics_step'), sample.get('simulation_time'))
        if key == (None, None):
            key = ('unindexed', number)
        unique[key] = sample
    ordered = sorted(unique.values(), key=lambda sample: sample.get('simulation_time', 0.))
    pair_impulses, point_impulses, depths, forces = [], [], [], []
    robot_speeds, relative_speeds, normal_speeds, displacement_speeds = [], [], [], []
    force_unavailable = defaultdict(int)
    by_link, by_pair, last_pair_step = defaultdict(int), defaultdict(int), {}
    episodes, points_count, reported_steps, physical_steps = 0, 0, 0, 0
    reported_unverified_steps = physical_unverified_steps = missing_step_pairs = 0
    kinematic_points = pair_samples = 0
    signed_separations, invalid_buffers = [], defaultdict(int)
    reported_duration, physical_duration = 0., 0.
    complete_impulses = complete_separations = True
    previous, previous_matrices = None, None
    saturated = False
    for sample in ordered:
        dt, reason = _step_duration(sample, previous)
        matrices = _pose_matrices(sample.get('poses'), meters_per_unit)
        velocities = _numbers(sample.get('body_velocities'), (len(body_paths), 6))
        if velocities is not None:
            velocities = velocities.copy()
            velocities[:, :3] *= meters_per_unit
        contacts = sample.get('contacts') or []
        saturated |= bool(sample.get('contact_buffer_full', False))
        step_physical = False
        if contacts:
            reported_steps += 1
            reported_duration += dt or 0.
            reported_unverified_steps += dt is None
        for contact in contacts:
            pair_samples += 1
            sensor, obstacle = contact.get('sensor'), contact.get('obstacle')
            count = int(contact.get('count', 0))
            points_count += count
            by_link[str(sensor)] += count
            pair = (str(sensor), str(obstacle))
            by_pair[pair] += count
            step = sample.get('physics_step')
            missing_step_pairs += step is None
            if step is None or last_pair_step.get(pair) != step-1:
                episodes += 1
            last_pair_step[pair] = step
            separations = _numbers(contact.get('separations_m'))
            if separations is not None and separations.size != count:
                invalid_buffers['separation_count_mismatch'] += 1
                complete_separations = False
            if separations is None:
                minimum = _numbers(contact.get('minimum_separation_m'))
                separations = None if minimum is None else minimum.reshape(-1)
            if separations is None or not separations.size:
                complete_separations = False
            else:
                signed_separations.extend(separations.ravel())
                depths.extend(np.maximum(0., -separations).ravel())
                step_physical |= bool(np.any(separations < 0.))
            impulses = _numbers(contact.get('normal_impulses_ns'))
            if impulses is not None and impulses.size:
                impulses = abs(impulses).reshape(-1)
                if impulses.size != count:
                    invalid_buffers['impulse_count_mismatch'] += 1
                    complete_impulses = False
                point_impulses.extend(impulses)
                impulse = float(impulses.sum())
            else:
                legacy = _numbers(contact.get('impulse_magnitude'))
                impulse = None if legacy is None or legacy.shape != () else abs(float(legacy))
            if impulse is None:
                complete_impulses = False
                force_unavailable['normal_impulse_unavailable'] += 1
            else:
                pair_impulses.append(impulse)
                step_physical |= impulse > 0.
                if dt is not None:
                    forces.append(impulse/dt)
                else:
                    force_unavailable[reason] += 1
            points = _numbers(contact.get('points_world_m'))
            if (points is not None and points.shape == (count, 3) and matrices is not None
                    and len(matrices) == len(body_paths)
                    and velocities is not None and coms is not None
                    and sensor in indices and obstacle in indices):
                kinematic_points += count
                first, second = indices[sensor], indices[obstacle]
                robot_velocity = contact_point_velocity(points, matrices[first], coms[first], velocities[first])
                obstacle_velocity = contact_point_velocity(points, matrices[second], coms[second], velocities[second])
                relative = robot_velocity-obstacle_velocity
                robot_speeds.extend(np.linalg.norm(robot_velocity, axis=1))
                relative_speeds.extend(np.linalg.norm(relative, axis=1))
                normals = _numbers(contact.get('normals_world'), (count, 3))
                if normals is not None:
                    lengths = np.linalg.norm(normals, axis=1)
                    valid = lengths > 1e-12
                    normal_speeds.extend(abs(np.sum(relative[valid]*normals[valid], axis=1)/lengths[valid]))
                if previous_matrices is not None and previous is not None:
                    elapsed = sample['simulation_time']-previous['simulation_time']
                    if elapsed > 0:
                        local = (points-matrices[first, :3, 3]) @ matrices[first, :3, :3]
                        old_points = local @ previous_matrices[first, :3, :3].T + previous_matrices[first, :3, 3]
                        displacement_speeds.extend(np.linalg.norm(points-old_points, axis=1)/elapsed)
        if step_physical:
            physical_steps += 1
            physical_duration += dt or 0.
            physical_unverified_steps += dt is None
        previous, previous_matrices = sample, matrices
    reference_overlap = minimum_clearance_m is not None and minimum_clearance_m <= 0.
    if saturated:
        classification = 'contact_buffer_saturated_severity_incomplete'
    elif not points_count:
        classification = 'reference_contact_disagreement' if reference_overlap else 'no_physical_contact_observed'
    elif not complete_impulses or not complete_separations:
        classification = 'contact_severity_unavailable'
    elif max(depths, default=0.) == 0. and max(pair_impulses, default=0.) == 0. and not reference_overlap:
        classification = ('contact_offset_proximity' if minimum_clearance_m is not None
                          else 'nonpenetrating_zero_impulse_reports_reference_unavailable')
    else:
        classification = 'measured_physical_contact_ungraded'
    return dict(strict_v2_success=strict_v2_success, strict_zero_reported_contacts=points_count == 0,
                goal_status=goal_status, failure_code=failure_code, physical_classification=classification,
                limited_low_severity_assessment='UNVALIDATED_NO_ACCEPTABILITY_THRESHOLD',
                severity_pass=None, reported_points=points_count, reported_physics_steps=reported_steps,
                observed_physical_contact_steps=physical_steps,
                pair_contact_episodes=None if missing_step_pairs else episodes,
                pair_contact_samples=pair_samples, pairs_missing_physics_step_index=missing_step_pairs,
                unique_physics_samples=len(ordered), duplicate_samples_removed=len(samples)-len(ordered),
                sampled_reported_contact_duration_s=reported_duration,
                sampled_physical_contact_duration_s=physical_duration,
                reported_steps_with_unverified_duration=reported_unverified_steps,
                physical_steps_with_unverified_duration=physical_unverified_steps,
                minimum_reported_separation_m=min(signed_separations, default=None),
                sum_observed_pair_normal_impulses_ns=(None if points_count and not pair_impulses
                                                       else float(sum(pair_impulses))),
                depth_m=_distribution(depths), normal_pair_impulse_ns=_distribution(pair_impulses),
                normal_point_impulse_ns=_distribution(point_impulses),
                normal_pair_step_average_force_n=_distribution(forces),
                force_unavailable_reasons=dict(force_unavailable),
                robot_contact_point_speed_m_s=_distribution(robot_speeds),
                relative_contact_point_speed_m_s=_distribution(relative_speeds),
                absolute_normal_relative_speed_m_s=_distribution(normal_speeds),
                robot_contact_point_displacement_speed_m_s=_distribution(displacement_speeds),
                contact_points_with_kinematics=kinematic_points,
                contact_points_without_kinematics=points_count-kinematic_points,
                invalid_buffer_fields=dict(invalid_buffers),
                points_by_robot_link=dict(by_link),
                points_by_pair=[dict(sensor=a, obstacle=b, points=count) for (a, b), count in sorted(by_pair.items())],
                joint_feedback=analyze_joint_feedback(ordered), contact_buffer_saturated=saturated,
                methods=dict(normal_impulse='Sum of absolute normal point impulses per robot-link/actor pair and observed physics step; tangential friction excluded',
                    force='Normal pair impulse divided by verified SDK physics dt; step-average, not peak force',
                    speed='SDK world COM linear velocity plus angular cross contact-point offset; relative velocity subtracts actual actor point velocity',
                    absolute_normal_speed='Absolute projection of relative point velocity onto normalized contact normal; no unvalidated normal-orientation closing sign',
                    depth='Per-point negative separation clipped at zero; legacy V2 input supplies pair-minimum depth only',
                    episode='Consecutive observed physics steps for one explicit robot-link/actor pair; unavailable without step indices'),
                limitations=[
                    'No calibrated material/force tolerance establishes harmless contact or a severity PASS. Depth, impulse, speed and duration remain raw experimental measurements.',
                    'Normal impulses exclude friction. Force is a last-physics-step average only for verified one-step intervals; missing/skipped intervals are not zero force.',
                    'SDK velocities are post-solver instantaneous velocities, not proven pre-impact approach speeds. Displacement velocities average captured frames.',
                    'Kit samples may miss substeps and brief contacts. Episode/point counts are sampled reports, not unique impacts. Sampled durations omit unverified intervals.',
                    'Joint tracking/effort/reaction signals are observations, not a validated unexpected-contact detector or calibrated external force/torque sensor.',
                ])
