#!/usr/bin/env python3
"""Compare frozen stock/enhanced Lula spheres with actual captured cooked solids.

This offline measurement imports native CPU Lula without SimulationApp. Every
audited posture comes from recorded physical feedback/body poses. Surface
quadrature, finite-cell solid containment, model queries and observed physical
clearance are reported separately. Optional query timings require a quiescent
machine and do not stand in for native planning/execution timings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from scripts.audit_global_collision_v4_1 import (
    body_mesh_vertices, clear_numeric_scene, coverage_statistics, pose_matrix_xyzw, register_numeric_scene,
    sphere_union_gaps, surface_quadrature,
)
from scripts.generate_franka_collision_model_v4_2 import FINGER_BODIES, finger_transforms
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import adjacent_self_collision_pairs
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import (
    PlaneDomain, StaticObstacle, StaticWorldSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = Path('/home/etfrobotics/isaacsim')
DESCRIPTOR_SHA256 = '6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d'
DOF_NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))+('panda_finger_joint1', 'panda_finger_joint2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_label(path):
    folder = Path(path).resolve().parent
    return str(folder.relative_to(ROOT/'outputs')) if folder.is_relative_to(ROOT/'outputs') else str(folder)


def representative_indices(samples, per_phase=3):
    """Select first/middle/last unique measured states within each phase."""
    if type(per_phase) is not int or not 1 <= per_phase <= 21:
        raise ValueError('One to twenty-one representative states per phase are required')
    grouped, seen = {}, set()
    for index, sample in enumerate(samples):
        phase = sample.get('phase', 'recorded_route')
        key = phase, sample.get('physics_step'), sample.get('simulation_time')
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(phase, []).append(index)
    return [indices[i] for indices in grouped.values()
            for i in np.rint(np.linspace(0, len(indices)-1, min(per_phase, len(indices)))).astype(int)]


def measured_finger_pose(native_pose, hand_rotation, axis, actual, fixed):
    """Correct fixed-descriptor FK using the measured independent prismatic DOF."""
    pose = np.asarray(native_pose, float).copy()
    pose[:3, 3] += np.asarray(hand_rotation)@np.asarray(axis)*(float(actual)-float(fixed))
    return pose


@dataclass
class CapturedCase:
    name: str
    samples: list
    body_paths: tuple
    parts: list
    dof_names: tuple
    snapshot: object
    sources: dict
    posture_goal_checks: dict
    source_status: str


def _parts(cooking, archive, body_paths):
    result = []
    for record in cooking:
        if record['body'] not in body_paths:
            raise ValueError('Cooked body is missing from actual pose observations')
        for _ in range(record['cooked_hulls']):
            index = len(result)
            vertices = body_mesh_vertices(archive[f'vertices_{index}'], archive[f'body_from_mesh_{index}'])
            triangles = archive[f'triangles_{index}'].copy()
            points, weights = surface_quadrature(vertices, triangles, subdivisions=8)
            result.append(dict(body=record['body'], vertices=vertices, triangles=triangles,
                points=points, weights=weights, hull_index=index))
    names = {part['body'].rsplit('/', 1)[-1] for part in result}
    if names != {'panda_hand', *FINGER_BODIES, *(f'panda_link{i}' for i in range(8))}:
        raise ValueError('Independent geometry must include all eleven cooked robot bodies')
    return result


def _snapshot(data):
    if data is None:
        return None
    return StaticWorldSnapshot(**dict(data,
        obstacles=tuple(StaticObstacle(**item) for item in data['obstacles']),
        plane_domains=tuple(PlaneDomain(**item) for item in data.get('plane_domains', []))))


def load_case(path):
    """Read standard native fixture captures or one acceptance generation."""
    path = Path(path)
    if path.name == 'acceptance_trace.json':
        generations = json.loads(path.read_text())
        generation = generations[0]
        metadata = generation['metadata']
        samples = generation['samples']
        body_paths = tuple(metadata['body_paths'])
        names = tuple(metadata['robot_dof_names'])
        geometry_path = path.parent/'acceptance_geometry_0.npz'
        snapshot = _snapshot(generation.get('startup_world'))
        cooking = generation['cooking']
        sources = {str(path): digest(path), str(geometry_path): digest(geometry_path)}
        goals, source_status = {}, 'RECORDED_ACCEPTANCE_GENERATION'
    else:
        payload = json.loads(path.read_text())
        if payload.get('measurement_status', payload['status']) != 'PASS':
            raise ValueError('Invalid native observations cannot become coverage evidence')
        metrics = payload['metrics']
        if not metrics['geometry_reference']['cooked_only']:
            raise ValueError('Only independent PhysX-cooked references are accepted')
        trace_path = path.parent/metrics.get('trace_artifact', 'trace.json')
        trace = json.loads(trace_path.read_text())
        samples = trace['samples']
        body_paths = tuple(metrics['contact_reference']['sensor_paths'])
        names = tuple(metrics.get('robot_dof_names', trace.get('metadata', {}).get('robot_dof_names', DOF_NAMES)))
        geometry_path = path.parent/'geometry_reference.npz'
        plan_path = path.parent/'plan_trace.json'
        snapshot = _snapshot(json.loads(plan_path.read_text())['snapshot'] if plan_path.exists()
                             else metrics.get('snapshot'))
        cooking = metrics['geometry_reference']['cooking']
        sources = {str(item): digest(item) for item in (path, trace_path, geometry_path)}
        goals, source_status = metrics.get('posture_goal_checks', {}), payload['status']
        if metrics.get('meters_per_unit', 1.) != 1.:
            raise ValueError('Only actual meter-stage captures are supported')
        if plan_path.exists():
            sources[str(plan_path)] = digest(plan_path)
    if (len(body_paths) != len(set(body_paths)) or len(names) != 9 or set(names) != set(DOF_NAMES)
            or not samples or (snapshot is not None and snapshot.meters_per_unit != 1.)):
        raise ValueError('Unambiguous nine-DOF meter-stage captures are required')
    with np.load(geometry_path, allow_pickle=False) as archive:
        if 'body_poses' in archive:
            shape = archive['body_poses'].shape
            # Historical captures append physical obstacle poses after the
            # named robot sensor bodies. Retain/compare that duplicate data.
            if len(shape) != 3 or shape[0] != len(samples) or shape[1] < len(body_paths) or shape[2] != 7:
                raise ValueError('NPZ poses and physical trace differ in shape')
            for index, sample in enumerate(samples):
                if not np.array_equal(sample['poses'], archive['body_poses'][index]):
                    raise ValueError('NPZ poses and physical trace differ')
        elif path.name != 'acceptance_trace.json':
            raise ValueError('Standard native capture is missing duplicate NPZ poses')
        parts = _parts(cooking, archive, body_paths)
    return CapturedCase(case_label(path), samples, body_paths, parts, names, snapshot, sources, goals, source_status)


def native_model(isaac_root, descriptor, urdf, *, extra_exclusions=frozenset()):
    import yaml
    sys.path.insert(0, str(Path(isaac_root)/'exts/isaacsim.robot_motion.lula/pip_prebundle'))
    import lula
    document = yaml.safe_load(Path(descriptor).read_text())
    robot = lula.load_robot(str(descriptor), str(urdf))
    world = lula.create_world()
    inspector = lula.create_robot_world_inspector(robot, world.add_world_view())
    inspector.update_world_view()
    entries = [(frame, row) for group in document['collision_spheres'] for frame, rows in group.items() for row in rows]
    if len(entries) != inspector.num_collision_spheres():
        raise ValueError('Descriptor/native primitive counts differ')
    fixed = {rule['name']: rule['value'] for rule in document['cspace_to_urdf_rules']}
    return dict(lula=lula, robot=robot, world=world, inspector=inspector, kinematics=robot.kinematics(),
        descriptor=Path(descriptor), urdf=Path(urdf), document=document, entries=entries,
        fixed=fixed, exclusions=adjacent_self_collision_pairs(urdf)|extra_exclusions)


def _pose(model, q, frame):
    pose = model['kinematics'].pose(np.asarray(q)[:, None], frame)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = pose.rotation.matrix(), pose.translation
    return result


def query_timing(inspector, q, batches):
    """Same-configuration cached/native query microbenchmark, in bounded batches."""
    if type(batches) is not int or not 0 <= batches <= 101:
        raise ValueError('Timing batches must be bounded from zero through101')
    if not batches:
        return dict(status='NOT_MEASURED')
    result = {}
    for label, query in (('external_collision', inspector.in_collision_with_obstacle),
                         ('self_collision_pairs', inspector.frames_in_self_collision),
                         ('sphere_positions', inspector.collision_sphere_positions),
                         ('combined_external_self', lambda value: (inspector.in_collision_with_obstacle(value),
                             inspector.frames_in_self_collision(value))),
                         ('combined_external_self_positions', lambda value: (inspector.in_collision_with_obstacle(value),
                             inspector.frames_in_self_collision(value), inspector.collision_sphere_positions(value)))):
        query(q)
        values = []
        for _ in range(batches):
            start = time.perf_counter_ns()
            for _ in range(128):
                query(q)
            values.append((time.perf_counter_ns()-start)/128/1000)
        result[label] = dict(median_us=float(np.median(values)), p95_us=float(np.percentile(values, 95)),
            batches=batches, queries_per_batch=128)
    return dict(status='MEASURED_OFFLINE_SAME_Q', queries=result,
        limitation='Repeated same-q native CPU calls include Python overhead and caching; excludes RRT, route generation and Kit update cost.')


def physical_clearance(parts, physical, snapshot):
    """Independent vertex-hull distance against captured physical box/plane world."""
    if snapshot is None:
        return dict(status='NO_CAPTURED_WORLD', minimum_m=None)
    from robot_skill_stack.common.rotations import quat_matrix
    from scripts.obstacle_representation_v3_cylinder import convex_distance_details
    minimum, closest, width = np.inf, None, 0.
    for part in parts:
        pose = physical[part['body']]
        vertices = part['vertices']@pose[:3, :3].T+pose[:3, 3]
        for obstacle in snapshot.obstacles:
            rotation = quat_matrix(obstacle.orientation)
            if obstacle.source_kind == 'Plane':
                point = np.asarray(obstacle.position)+rotation[:, 2]*obstacle.dimensions[2]/2
                distance = float(np.min((vertices-point)@rotation[:, 2]))
            elif obstacle.kind == 'cuboid' and obstacle.approximation == 'exact':
                half = np.asarray(obstacle.dimensions)/2
                corners = np.array([[x, y, z] for x in (-half[0], half[0])
                    for y in (-half[1], half[1]) for z in (-half[2], half[2])])
                detail = convex_distance_details(vertices, corners@rotation.T+obstacle.position)
                distance = detail['distance_m']
                width = max(width, detail['upper_bound_m']-detail['lower_bound_m'])
            else:
                raise ValueError('Independent captured-world clearance requires exact physical boxes/planes')
            if distance < minimum:
                minimum = distance
                closest = dict(body=part['body'], obstacle=obstacle.path)
    return dict(status='MEASURED_COOKED_SOLIDS', minimum_m=float(minimum) if np.isfinite(minimum) else None,
        closest_pair=closest, maximum_numerical_distance_certificate_width_m=float(width),
        limitation='Actual sampled pose only, excludes contact/rest offsets and continuous swept clearance.')


def physical_self_pair_distances(parts, physical, pairs):
    """Check reported nonadjacent model pairs against independent cooked solids."""
    if not pairs:
        return []
    from scripts.obstacle_representation_v3_cylinder import convex_distance_details
    by_frame = defaultdict(list)
    for part in parts:
        pose = physical[part['body']]
        by_frame[part['body'].rsplit('/', 1)[-1]].append(
            part['vertices']@pose[:3, :3].T+pose[:3, 3])
    result = []
    for first, last in pairs:
        if first not in by_frame or last not in by_frame:
            result.append(dict(frames=[first, last], status='REFERENCE_UNAVAILABLE'))
            continue
        measurements = [convex_distance_details(left, right)
            for left in by_frame[first] for right in by_frame[last]]
        result.append(dict(frames=[first, last], status='MEASURED_COOKED_SOLIDS',
            distance_m=min(row['distance_m'] for row in measurements),
            lower_bound_m=min(row['lower_bound_m'] for row in measurements),
            upper_bound_m=min(row['upper_bound_m'] for row in measurements)))
    return result


def audit_case(case, models, enhanced, *, per_phase=3, timing_batches=0):
    indices = representative_indices(case.samples, per_phase)
    arm_indices = [case.dof_names.index(name) for name in DOF_NAMES[:7]]
    finger_indices = [case.dof_names.index(name) for name in DOF_NAMES[7:]]
    results, render = [], None
    for index in indices:
        sample = case.samples[index]
        feedback = sample.get('joint_feedback', sample.get('feedback'))
        joints = np.asarray(feedback['positions'], float)
        if joints.shape != (9,) or not np.isfinite(joints).all():
            raise ValueError('Finite measured nine-DOF posture is required')
        physical = {path: pose_matrix_xyzw(pose) for path, pose in zip(case.body_paths, sample['poses'])}
        base = physical['/Franka/panda_link0']
        q, fingers = joints[arm_indices], joints[finger_indices]
        cooked_clearance = physical_clearance(case.parts, physical, case.snapshot)
        model_rows = {}
        for name, model in models.items():
            if case.snapshot is not None:
                register_numeric_scene(model, case.snapshot, base)
            else:
                clear_numeric_scene(model)
            inspector = model['inspector']
            centers = np.asarray(inspector.collision_sphere_positions(q))@base[:3, :3].T+base[:3, 3]
            radii = np.asarray(inspector.collision_sphere_radii())
            external = bool(inspector.in_collision_with_obstacle(q))
            pairs = [tuple(pair) for pair in inspector.frames_in_self_collision(q)
                     if frozenset(pair) not in model['exclusions']]
            body_rows, alignment = [], []
            hand_pose = base@_pose(model, q, 'panda_hand')
            attachments = finger_transforms(model['urdf'], 0.)
            for path, actual in physical.items():
                frame = path.rsplit('/', 1)[-1]
                if frame not in {part['body'].rsplit('/', 1)[-1] for part in case.parts}:
                    continue
                expected = base@_pose(model, q, frame)
                nominal_error = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
                if frame in FINGER_BODIES:
                    finger = FINGER_BODIES.index(frame)
                    expected = measured_finger_pose(expected, hand_pose[:3, :3], attachments[frame][1],
                        fingers[finger], model['fixed'][DOF_NAMES[7+finger]])
                alignment.append(dict(frame=frame, measured_prismatic_corrected_translation_error_m=float(
                    np.linalg.norm(expected[:3, 3]-actual[:3, 3])),
                    nominal_descriptor_translation_error_m=nominal_error,
                    rotation_matrix_max_error=float(np.max(np.abs(expected[:3, :3]-actual[:3, :3])))))
            for part in case.parts:
                pose = physical[part['body']]
                points = part['points']@pose[:3, :3].T+pose[:3, 3]
                body_rows.append(dict(body=part['body'], hull_index=part['hull_index'],
                    **coverage_statistics(sphere_union_gaps(points, centers, radii), part['weights'])))
            base_preflight = None
            if name == 'enhanced' and case.snapshot is not None:
                try:
                    base_preflight = dict(status='PASS', **enhanced.preflight_base(case.snapshot,
                        base[:3, 3], base[:3, :3]))
                except Exception as error:
                    base_preflight = dict(status='REJECTED', message=str(error), details=getattr(error, 'details', {}))
            supported_fingers = bool(np.all(fingers >= enhanced.finger_range_m[0]) and
                                     np.all(fingers <= enhanced.finger_range_m[1]))
            false_obstruction = (external and cooked_clearance['minimum_m'] is not None
                                 and cooked_clearance['minimum_m'] > 0.)
            physical_self = physical_self_pair_distances(case.parts, physical, pairs)
            model_rows[name] = dict(primitive_count=len(radii), bodies=body_rows, frame_alignment=alignment,
                external_model_collision=external, unmasked_model_self_pairs=pairs,
                independent_cooked_self_pair_distances=physical_self,
                independently_self_clear_but_model_obstructed=bool(physical_self) and all(
                    row.get('lower_bound_m', 0.) > 0 for row in physical_self),
                independently_clear_but_model_obstructed=bool(false_obstruction),
                enhanced_finger_contract_supported=supported_fingers, fixed_base_preflight=base_preflight,
                cpu_query_timing=query_timing(inspector, q, timing_batches))
            if render is None:
                frame_spheres = []
                for frame, row in model['entries']:
                    frame_pose = base@_pose(model, q, frame)
                    frame_spheres.append(dict(frame=frame,
                        center=frame_pose[:3, :3]@np.asarray(row['center'])+frame_pose[:3, 3],
                        radius=row['radius']))
                # Native inspector sphere order is not the YAML frame order.
                # Keep frame-labelled entries separately for the hand figure.
                model_rows[name]['render'] = dict(centers=centers, radii=radii, frame_spheres=frame_spheres)
        if render is None:
            render = dict(parts=[dict(body=part['body'], points=part['points']@
                physical[part['body']][:3, :3].T+physical[part['body']][:3, 3]) for part in case.parts],
                hand_pose=physical['/Franka/panda_hand'], models={name: row.pop('render')
                    for name, row in model_rows.items()}, case=case.name, sample=index)
        results.append(dict(case=case.name, sample=index, phase=sample.get('phase', 'recorded_route'),
            requested_phase_goal_achieved=case.posture_goal_checks.get(sample.get('phase')),
            actual_tcp_pose=sample.get('tcp_pose'),
            physics_step=sample.get('physics_step'), simulation_time=sample.get('simulation_time'),
            measured_joint_positions=joints.tolist(), finger_positions_m=fingers.tolist(),
            independent_physical_world_clearance=cooked_clearance, models=model_rows))
    return results, render


def compact_results(states, models):
    summary = {}
    for name, model in models.items():
        grouped = defaultdict(list)
        for state in states:
            for body in state['models'][name]['bodies']:
                grouped[body['body']].append(body)
        bodies = []
        for body, rows in grouped.items():
            bodies.append(dict(body=body, states=len(rows), sphere_union_surface_coverage_minimum=min(
                row['estimated_covered_surface_fraction'] for row in rows),
                sphere_union_surface_coverage_mean=float(np.mean([row['estimated_covered_surface_fraction'] for row in rows])),
                maximum_sampled_outside_gap_m=max(row['maximum_sampled_outside_gap_m'] for row in rows)))
        alignments = [row for state in states for row in state['models'][name]['frame_alignment']]
        summary[name] = dict(descriptor_sha256=digest(model['descriptor']), primitive_count=len(model['entries']),
            primitive_counts_by_frame=dict(Counter(frame for frame, _ in model['entries'])), bodies=bodies,
            maximum_measured_prismatic_corrected_frame_translation_error_m=max(
                row['measured_prismatic_corrected_translation_error_m'] for row in alignments),
            maximum_rotation_matrix_error=max(row['rotation_matrix_max_error'] for row in alignments),
            externally_obstructed_states=sum(state['models'][name]['external_model_collision'] for state in states),
            independently_clear_but_model_obstructed_states=sum(
                state['models'][name]['independently_clear_but_model_obstructed'] for state in states),
            unmasked_self_obstructed_states=sum(bool(state['models'][name]['unmasked_model_self_pairs']) for state in states))
        summary[name]['independently_self_clear_but_model_obstructed_states'] = sum(
            state['models'][name]['independently_self_clear_but_model_obstructed'] for state in states)
        summary[name]['enhanced_contract_unsupported_states'] = sum(
            not state['models'][name]['enhanced_finger_contract_supported'] for state in states)
        for scope, include_base in (('moving_bodies', False), ('including_base', True)):
            rows = [body for state in states for body in state['models'][name]['bodies']
                    if include_base or not body['body'].endswith('/panda_link0')]
            summary[name][scope+'_area_weighted_sample_coverage'] = float(sum(
                row['surface_area_m2']*row['estimated_covered_surface_fraction'] for row in rows)/
                sum(row['surface_area_m2'] for row in rows))
        timings = defaultdict(list)
        for state in states:
            for query, measured in state['models'][name]['cpu_query_timing'].get('queries', {}).items():
                timings[query].append(measured['median_us'])
        summary[name]['cpu_query_timing'] = dict(status='MEASURED_OFFLINE_SAME_Q' if timings else 'NOT_MEASURED',
            queries={query: dict(states=len(values), median_state_batch_median_us=float(np.median(values)),
                p95_state_batch_median_us=float(np.percentile(values, 95))) for query, values in timings.items()},
            limitation='Explicit repeated Python/native inspector calls, not native RRT internal query counts, which are unexposed.')
    return summary


def audit_gap_witnesses(source, manifest_path, models, enhanced):
    """Recheck historical source-bound finger counterexamples without simulation."""
    from scripts.obstacle_representation_v3_cylinder import convex_distance_details

    source, manifest_path = Path(source), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['raw_artifact']['sha256'] != digest(source):
        raise ValueError('Historical witness list differs from its retained compact manifest')
    descriptions = {case['case']: case for case in manifest['cases']}
    rows, provenance = [], {str(source): digest(source), str(manifest_path): digest(manifest_path)}
    for case_record in json.loads(source.read_text()):
        description = descriptions[case_record['case']]
        for artifact in description['source_artifacts'].values():
            if digest(artifact['path']) != artifact['sha256']:
                raise ValueError('Historical independent witness source changed')
            provenance[artifact['path']] = artifact['sha256']
        case = load_case(description['source_artifacts']['result.json']['path'])
        if case.snapshot is None:
            raise ValueError('Historical witness source is missing its original frozen world')
        for witness in case_record['witnesses']:
            samples = [sample for sample in case.samples if sample['physics_step'] == witness['physics_step']
                and np.allclose(np.asarray(sample['joint_feedback']['positions'])[:7], witness['native_q'], atol=1e-12, rtol=0)]
            if not samples:
                raise ValueError('Witness q does not match an actual retained source state')
            sample = samples[0]
            physical = {path: pose_matrix_xyzw(sample['poses'][index])
                for index, path in enumerate(case.body_paths)}
            base = physical['/Franka/panda_link0']
            fingers = np.asarray(sample['joint_feedback']['positions'])[7:9]
            supported = bool(np.all(fingers >= enhanced.finger_range_m[0]) and np.all(fingers <= enhanced.finger_range_m[1]))
            part = next(part for part in case.parts if part['body'] == witness['body'])
            pose = physical[part['body']]
            vertices = part['vertices']@pose[:3, :3].T+pose[:3, 3]
            point = np.asarray(witness['physical_cooked_surface_center'])
            distance = convex_distance_details(vertices, point[None])
            physical_intersection = distance['upper_bound_m'] < witness['radius_m']
            q, query_rows = np.asarray(witness['native_q']), {}
            for name, model in models.items():
                register_numeric_scene(model, case.snapshot, base)
                inspector = model['inspector']
                baseline = bool(inspector.in_collision_with_obstacle(q))
                self_pairs = [tuple(pair) for pair in inspector.frames_in_self_collision(q)
                    if frozenset(pair) not in model['exclusions']]
                clear_numeric_scene(model)
                lula = model['lula']
                ball = lula.create_obstacle(lula.Obstacle.Type.SPHERE)
                ball.set_attribute(lula.Obstacle.Attribute.RADIUS, witness['radius_m'])
                handle = model['world'].add_obstacle(ball, lula.Pose3(lula.Rotation3(np.eye(3)),
                    base[:3, :3].T@(point-base[:3, 3])))
                try:
                    inspector.update_world_view()
                    detected = bool(inspector.in_collision_with_obstacle(q))
                finally:
                    model['world'].remove_obstacle(handle)
                    inspector.update_world_view()
                query_rows[name] = dict(original_world_external_collision=baseline,
                    original_world_unmasked_self_pairs=self_pairs, isolated_witness_collision=detected)
            passed = (physical_intersection and supported and not query_rows['stock']['original_world_external_collision']
                and not query_rows['stock']['original_world_unmasked_self_pairs']
                and not query_rows['stock']['isolated_witness_collision']
                and query_rows['enhanced']['isolated_witness_collision'])
            rows.append(dict(case=case.name, witness=witness, measured_finger_positions_m=fingers.tolist(),
                enhanced_finger_contract_supported=supported, physical_point_to_cooked_solid_distance=distance,
                witness_ball_intersects_cooked_solid=physical_intersection, native_models=query_rows,
                prevented_stock_model_gap=bool(passed)))
    return dict(status='PASS' if rows and all(row['prevented_stock_model_gap'] for row in rows) else 'FAIL',
        witnesses=len(rows), prevented_stock_model_gaps=sum(row['prevented_stock_model_gap'] for row in rows),
        source_sha256=provenance, models={name: digest(model['descriptor']) for name, model in models.items()},
        rows=rows, scope='Actual recorded cooked-solid configuration counterexamples and numeric native Lula obstacle queries only; never authored or physically executed witness balls. Not a route or continuous-motion certificate.')


def plot_measurements(summary, render, models, directory):
    """Render only the actual source posture, native spheres and measured gaps."""
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/motion-planning-v4-2/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 15), np.linspace(0, np.pi, 9))
    unit = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=-1)
    paths = []
    figure = plt.figure(figsize=(12, 6.2), constrained_layout=True)
    all_points = np.concatenate([part['points'] for part in render['parts']])
    low, high = all_points.min(0)-.06, all_points.max(0)+.06
    for column, name in enumerate(('stock', 'enhanced'), 1):
        axis = figure.add_subplot(1, 2, column, projection='3d')
        for part in render['parts']:
            points = part['points'][::max(1, len(part['points'])//400)]
            axis.scatter(*points.T, s=1.2, color='#345973', alpha=.6)
        positions = render['models'][name]
        for center, radius in zip(positions['centers'], positions['radii']):
            sphere = center+radius*unit
            axis.plot_wireframe(sphere[:, :, 0], sphere[:, :, 1], sphere[:, :, 2],
                linewidth=.3, color='#d98926', alpha=.35)
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
            xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)',
            title=f'{name.capitalize()}: {len(positions["radii"])} native planning spheres')
        axis.set_box_aspect(high-low)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.zaxis.label.set_size(9)
        axis.view_init(elev=21, azim=-55)
    figure.suptitle(f'Actual captured cooked surfaces and native planning spheres\n{render["case"]}, sample {render["sample"]}')
    path = directory/'collision_geometry_comparison.png'
    figure.savefig(path, dpi=170, bbox_inches='tight', pad_inches=.2)
    plt.close(figure)
    paths.append(path)

    figure = plt.figure(figsize=(12, 6), constrained_layout=True)
    hand = render['hand_pose']
    rotation, origin = hand[:3, :3], hand[:3, 3]
    reference = np.concatenate([part['points'] for part in render['parts']
        if part['body'].rsplit('/', 1)[-1] in ('panda_hand', *FINGER_BODIES)])
    local = (reference-origin)@rotation
    for column, name in enumerate(('stock', 'enhanced'), 1):
        axis = figure.add_subplot(1, 2, column, projection='3d')
        positions = render['models'][name]
        gaps = sphere_union_gaps(reference, positions['centers'], positions['radii'])
        for covered, color in ((True, '#3477a5'), (False, '#ca4a38')):
            selected = local[(gaps <= 1e-9) == covered][::3]
            if len(selected):
                axis.scatter(*selected.T, s=1.2, color=color, alpha=.6)
        for item in positions['frame_spheres']:
            if item['frame'] not in ('panda_hand', *FINGER_BODIES):
                continue
            center = (item['center']-origin)@rotation
            sphere = center+item['radius']*unit
            axis.plot_wireframe(sphere[:, :, 0], sphere[:, :, 1], sphere[:, :, 2],
                linewidth=.4, color='#d98926', alpha=.35)
        axis.set(xlim=(-.085, .085), ylim=(-.13, .13), zlim=(-.05, .15),
            xlabel='Hand X (m)', ylabel='Hand Y (m)', zlabel='Hand Z (m)', title=name.capitalize())
        axis.set_box_aspect((.17, .26, .20))
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.zaxis.label.set_size(9)
        axis.view_init(elev=19, azim=-53)
    figure.legend(handles=[Line2D([], [], marker='.', linestyle='', color='#3477a5', label='Covered physical samples'),
        Line2D([], [], marker='.', linestyle='', color='#ca4a38', label='Uncovered physical samples'),
        Line2D([], [], color='#d98926', label='Native spheres')], loc='lower center', ncol=3, fontsize=9)
    figure.suptitle('Measured hand and finger coverage at the same actual posture\nPhysical finger feedback; enhanced range 1–5 mm per joint')
    path = directory/'hand_finger_coverage.png'
    figure.savefig(path, dpi=170, bbox_inches='tight', pad_inches=.2)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6.2), constrained_layout=True)
    labels = [row['body'].rsplit('/', 1)[-1].replace('panda_', '') for row in summary['stock']['bodies']]
    y = np.arange(len(labels))
    for offset, name, color in ((-.18, 'stock', '#7d9eb8'), (.18, 'enhanced', '#28628d')):
        rows = summary[name]['bodies']
        axes[0].barh(y+offset, [100*row['sphere_union_surface_coverage_mean'] for row in rows], .35,
                     color=color, label=name.capitalize())
        axes[1].barh(y+offset, [1000*row['maximum_sampled_outside_gap_m'] for row in rows], .35, color=color)
    for axis in axes:
        axis.set(yticks=y, yticklabels=labels)
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=.2)
    axes[0].set(xlim=(0, 100), xlabel='Area-weighted surface sample coverage (%)')
    axes[1].set(xlabel='Maximum sampled outside gap (mm)')
    axes[0].legend(loc='lower right')
    figure.suptitle('Actual cooked geometry across recorded postures\nBase sphere-union gap retained; fixed-base containment uses its separate preflight')
    path = directory/'collision_coverage.png'
    figure.savefig(path, dpi=170, bbox_inches='tight', pad_inches=.2)
    plt.close(figure)
    paths.append(path)
    return [dict(path=str(path), sha256=digest(path)) for path in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, type=Path,
        help='Standard result.json or acceptance_trace.json; repeat for all recorded datasets')
    parser.add_argument('--isaac-root', type=Path, default=ISAAC_ROOT)
    parser.add_argument('--per-phase', type=int, default=3)
    parser.add_argument('--timing-batches', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/motion-planning-v4-2/collision-audit')
    parser.add_argument('--compact', type=Path)
    parser.add_argument('--figure-dir', type=Path)
    parser.add_argument('--gap-witnesses', type=Path)
    parser.add_argument('--gap-manifest', type=Path)
    args = parser.parse_args(argv)
    enhanced = load_enhanced_collision_model()
    if enhanced.descriptor_sha256 != DESCRIPTOR_SHA256:
        raise ValueError('Frozen enhanced model changed')
    motion = args.isaac_root/'exts/isaacsim.robot_motion.motion_generation'
    urdf = motion/'motion_policy_configs/franka/lula_franka_gen.urdf'
    if digest(urdf) != enhanced.urdf_sha256:
        raise ValueError('Native URDF differs from the certified source')
    models = dict(stock=native_model(args.isaac_root, motion/'motion_policy_configs/franka/rmpflow/robot_descriptor.yaml', urdf),
        enhanced=native_model(args.isaac_root, enhanced.descriptor_path, urdf,
            extra_exclusions=enhanced.certified_self_collision_exclusions))
    states, sources, captures, render = [], {}, [], None
    for path in args.case:
        case = load_case(path)
        rows, illustration = audit_case(case, models, enhanced, per_phase=args.per_phase, timing_batches=args.timing_batches)
        if render is None:
            render = illustration
        states.extend(rows)
        sources.update(case.sources)
        captures.append(dict(case=case.name, source_status=case.source_status,
            posture_goal_checks=case.posture_goal_checks, measured_states=len(rows)))
    result = dict(schema_version=1, status='MEASURED_RECORDED_STATES', states=len(states),
        cases=[str(path) for path in args.case], source_sha256=sources,
        captured_goal_outcomes=captures,
        producer_sha256=digest(__file__), models=compact_results(states, models),
        enhanced_solid_certificate_sha256=digest(enhanced.descriptor_path.with_suffix('.json')),
        surface_method='Equal-area cooked-triangle centroid quadrature,64centroids per input triangle. Area fractions/max outside gaps are sample estimates; finite-cell solid containment is a separate source-bound certificate.',
        base_interpretation='The native base sphere union remains incomplete. Fixed-base enclosing-sphere/AABB preflight is reported separately and conditionally contains the whole captured base solid.',
        limitations=['No invented FK-only postures: only actual captured joint/body states are measured.',
            'A physically clear but sphere-obstructed sampled configuration measures conservative obstruction, not continuous passage infeasibility.',
            'Measured posture clearance excludes contact/rest offsets and does not certify future tracking or continuous execution.',
            'Missing posture captures, timing and native planning measurements remain unmeasured.'])
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output/'states.json'
    raw.write_text(json.dumps(states, indent=2, allow_nan=False)+'\n')
    result['raw_artifact'] = dict(path=str(raw), sha256=digest(raw))
    if args.gap_witnesses:
        gap = audit_gap_witnesses(args.gap_witnesses,
            args.gap_manifest or args.gap_witnesses.with_name('compact.json'), models, enhanced)
        artifact = args.output/'model_gap_witnesses.json'
        artifact.write_text(json.dumps(gap, indent=2, allow_nan=False)+'\n')
        result['model_gap_witnesses'] = {key: value for key, value in gap.items() if key != 'rows'}
        result['model_gap_witnesses']['raw_artifact'] = dict(path=str(artifact), sha256=digest(artifact))
    if args.figure_dir:
        result['figures'] = plot_measurements(result['models'], render, models, args.figure_dir)
        result['figure_source'] = dict(case=render['case'], sample=render['sample'])
    (args.output/'compact.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.compact:
        args.compact.parent.mkdir(parents=True, exist_ok=True)
        args.compact.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(states=len(states), models={name: dict(primitive_count=row['primitive_count'],
        external_obstructions=row['externally_obstructed_states'],
        conservative_obstructions=row['independently_clear_but_model_obstructed_states'])
        for name, row in result['models'].items()})))


if __name__ == '__main__':
    main()
