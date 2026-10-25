#!/usr/bin/env python3
"""Bounded independent cooked-geometry fixture design, with no RRT searches.

Read one newly captured obstacle-free global route, then apply the predeclared
grid and independent witness library. A numeric witness is used only to prove
geometric feasibility; it never supplies waypoints to the candidate planner.
Large hulls, joint polylines and detailed rejection records stay under outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import adjacent_self_collision_pairs
from scripts.audit_global_collision_v4_1 import body_mesh_vertices, pose_matrix_xyzw
from scripts.audit_rejected_ik_v4_2 import IndependentUrdfFK, NAMES, URDF, candidate_world_poses, intersection_ball
from scripts.obstacle_representation_v3_cylinder import convex_distance_details


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/assets/motion_planning_v4_3/design_protocol.json'
MODEL_SHA = '6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d'
ALLOWANCE_M = 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False)+'\n')


def reference(path):
    return dict(path=str(path), sha256=digest(path))


def box_vertices(center, dimensions):
    center, dimensions = np.asarray(center, float), np.asarray(dimensions, float)
    if (center.shape != (3,) or dimensions.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
        raise ValueError('Finite center and positive box dimensions required')
    return np.array(list(itertools.product(*zip(-dimensions/2, dimensions/2))))+center


def aabb_distance(lower, upper, center, half):
    """Lower bound on convex-solid distance, since AABBs enclose the solids."""
    delta = np.maximum(np.maximum(lower-(center+half), center-half-upper), 0.)
    return np.linalg.norm(delta, axis=-1)


def densify(knots, maximum_step):
    knots = np.asarray(knots, float)
    if (knots.ndim != 2 or knots.shape[1] != 9 or len(knots) < 2
            or not np.isfinite(knots).all() or maximum_step <= 0):
        raise ValueError('Finite nine-DOF witness knots and positive step required')
    rows = [knots[0]]
    for first, last in zip(knots[:-1], knots[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(last[:7]-first[:7]))/maximum_step)))
        rows.extend(first+(last-first)*(index/count) for index in range(1, count+1))
    return np.asarray(rows)


def displacement_weights(fk, frames, point_radii, names, finger_maximum):
    """Conservative point displacement per ancestor angle/prismatic increment."""
    children = {joint.child: joint for joint in fk.joints}
    weights = np.zeros((len(frames), len(names)))
    for row, frame in enumerate(frames):
        chain, child = [], frame
        while child in children:
            joint = children[child]
            chain.append(joint)
            child = joint.parent
        chain.reverse()
        for index, joint in enumerate(chain):
            if joint.name not in names:
                continue
            column = names.index(joint.name)
            if joint.kind == 'prismatic':
                weights[row, column] = np.linalg.norm(joint.axis)
            elif joint.kind in ('revolute', 'continuous'):
                radius = float(point_radii[row])
                for descendant in chain[index+1:]:
                    radius += np.linalg.norm(descendant.origin[:3, 3])
                    if descendant.kind == 'prismatic':
                        radius += np.linalg.norm(descendant.axis)*finger_maximum
                weights[row, column] = radius*np.linalg.norm(joint.axis)
    return weights


def relative_weights(first, second):
    """Shared ancestor joints apply the same isometry and cancel in distance."""
    first, second = np.asarray(first).copy(), np.asarray(second).copy()
    shared = (first > 0) & (second > 0)
    first[shared] = second[shared] = 0.
    return first+second


def continuous_aabb_scores(lower, upper, maximum_interval_motion, centers, dimensions):
    scores = []
    for start in range(0, len(centers), 24):
        batch = centers[start:start+24]
        distances = aabb_distance(lower[None], upper[None], batch[:, None, None], dimensions/2)
        # Any interval point is within its full displacement envelope from an
        # endpoint. Subtracting each body's maximum interval envelope from its
        # global sampled AABB distance is conservative over the whole polyline.
        bounds = distances.min(axis=1)-maximum_interval_motion[None]-ALLOWANCE_M
        scores.extend(bounds.min(axis=1))
    return np.asarray(scores)


def ordered_candidates(centers, dimensions, scores, minimum):
    rows = []
    for size, values in zip(dimensions, scores):
        rows.extend(dict(center=np.asarray(center), dimensions=np.asarray(size), score=float(score))
            for center, score in zip(centers, values) if score >= minimum)
    return sorted(rows, key=lambda row: (-row['score'], float(np.prod(row['dimensions'])),
        *row['center'], *row['dimensions']))


@dataclass
class Baseline:
    result_path: Path
    result: dict
    trace: dict
    operation: dict
    samples: list
    parts: list
    names: tuple
    body_paths: tuple
    fk: object
    base: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    actual_clouds: list
    planned_clouds: list
    planned_times: list
    endpoint_clouds: dict
    source_sha256: dict
    snapshot: dict
    alignment: dict


def clouds(parts, poses):
    return [part['vertices']@poses[part['frame']][:3, :3].T+poses[part['frame']][:3, 3]
            for part in parts]


def actual_clouds(parts, sample, body_paths):
    poses = {path.rsplit('/', 1)[-1]: pose_matrix_xyzw(sample['poses'][index])
             for index, path in enumerate(body_paths)}
    return clouds(parts, poses)


def load_baseline(path, urdf=URDF):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get('measurement_status', payload['status']) != 'PASS':
        raise ValueError('A valid newly captured no-obstacle native baseline is required')
    metrics = payload['metrics']
    trace_path = path.parent/metrics.get('trace_artifact', 'roundtrip_trace.json')
    geometry = path.parent/'geometry_reference.npz'
    trace = json.loads(trace_path.read_text())
    operation = next(row for row in trace['operations'] if row['label'] == 'outward')
    if not operation.get('returned_success') or not operation.get('measured_full_pose_within_tolerances'):
        raise ValueError('Initial obstacle-free global route must execute successfully')
    plan = operation['plan_trace']
    if len(plan['route']) != 2 or tuple(plan['joint_names']) != NAMES[:7]:
        raise ValueError('This predeclared design requires the preferred direct global C-space baseline')
    names = tuple(trace['metadata']['robot_dof_names'])
    if names != NAMES:
        raise ValueError('Unambiguous actual seven-arm/two-finger coordinate order is required')
    paths = tuple(trace['metadata']['body_paths'])
    samples = trace['samples'][operation['first_sample']:operation['last_sample']]
    if not samples:
        raise ValueError('Baseline has no actual outward observations')
    parts = []
    cooking = metrics['geometry_reference']['cooking']
    if (not metrics['geometry_reference']['cooked_only'] or len(cooking) != 11
            or any(row['cooked_hulls'] != 1 for row in cooking)):
        raise ValueError('All eleven frozen actual cooked convex bodies are required')
    with np.load(geometry, allow_pickle=False) as archive:
        if not np.array_equal(archive['body_poses'], np.asarray([row['poses'] for row in trace['samples']])):
            raise ValueError('Actual pose archive differs from its trace')
        for index, row in enumerate(cooking):
            vertices = body_mesh_vertices(archive[f'vertices_{index}'], archive[f'body_from_mesh_{index}'])
            parts.append(dict(body=row['body'], frame=row['body'].rsplit('/', 1)[-1], vertices=vertices))
    if {part['frame'] for part in parts} != {'panda_hand', 'panda_leftfinger', 'panda_rightfinger', *(f'panda_link{i}' for i in range(8))}:
        raise ValueError('Base, hand, fingers and every arm body must be retained')
    base = pose_matrix_xyzw(samples[0]['poses'][paths.index('/Franka/panda_link0')])
    fk = IndependentUrdfFK(urdf)
    start = np.asarray(samples[0]['joint_feedback']['positions'], float)
    goal = start.copy()
    goal[:7] = np.asarray(plan['route'][-1], float)
    if np.max(np.abs(start[:7]-plan['route'][0])) > 1e-12:
        raise ValueError('Actual initial arm state differs from the original global route')
    if np.any(start[7:] < .001) or np.any(start[7:] > .005):
        raise ValueError('New native baseline finger state lies outside the frozen contract')
    maximum_translation = maximum_rotation = maximum_vertex = 0.
    for sample in samples:
        q = np.asarray(sample['joint_feedback']['positions'], float)
        predicted = candidate_world_poses(fk, names, q, base)
        for part in parts:
            actual = pose_matrix_xyzw(sample['poses'][paths.index(part['body'])])
            expected = predicted[part['frame']]
            maximum_translation = max(maximum_translation, float(np.linalg.norm(expected[:3, 3]-actual[:3, 3])))
            maximum_rotation = max(maximum_rotation, float(np.max(np.abs(expected[:3, :3]-actual[:3, :3]))))
            radius = float(np.max(np.linalg.norm(part['vertices'], axis=1)))
            error = np.linalg.norm(expected[:3, 3]-actual[:3, 3])+np.linalg.norm(expected[:3, :3]-actual[:3, :3], ord=2)*radius
            maximum_vertex = max(maximum_vertex, float(error))
    if maximum_vertex > ALLOWANCE_M:
        raise ValueError('Independent URDF/cooked-body alignment exceeds the declared allowance')
    planned, stamps = [], []
    for row in plan['trajectory']:
        q = start.copy()
        q[:7] = row['q']
        planned.append(clouds(parts, candidate_world_poses(fk, names, q, base)))
        stamps.append(row['time'])
    if not planned:
        raise ValueError('Actual global generated trajectory must be retained')
    observed = [actual_clouds(parts, sample, paths) for sample in samples]
    endpoint = dict(actual_start_home=observed[0], actual_goal=observed[-1],
        fixed_validated_joint_goal=clouds(parts, candidate_world_poses(fk, names, goal, base)))
    sources = {str(item): digest(item) for item in (path, trace_path, geometry, Path(urdf), PROTOCOL, Path(__file__))}
    target_path = path.parent/'frozen_target.json'
    target_record = json.loads(target_path.read_text())
    selected_goal = np.asarray(target_record['q'], float)
    if (selected_goal.shape != (7,) or np.max(np.abs(selected_goal-goal[:7])) > 1e-12
            or target_record['model_sha256'] != MODEL_SHA
            or np.max(np.abs(np.asarray(target_record['target']['position'])-operation['target']['position'])) > 1e-12
            or np.max(np.abs(np.asarray(target_record['target']['orientation'])-operation['target']['orientation'])) > 1e-12):
        raise ValueError('Validated selected IK goal/full pose differs from the captured global route')
    sources[str(target_path)] = digest(target_path)
    snapshot = plan['snapshot']
    if snapshot['meters_per_unit'] != 1. or any(row['source_kind'] != 'Plane' for row in snapshot['obstacles']):
        raise ValueError('The declared no-obstacle baseline world must contain only its unchanged support plane')
    return Baseline(path, payload, trace, operation, samples, parts, names, paths, fk, base, start, goal,
        observed, planned, stamps, endpoint, sources, snapshot,
        dict(maximum_translation_m=maximum_translation, maximum_rotation_matrix_component=maximum_rotation,
            maximum_cooked_vertex_error_bound_m=maximum_vertex, allowance_m=ALLOWANCE_M))


def witness_knots(baseline, protocol):
    historical = ROOT/protocol['retained_witness_source']['path']
    if historical.is_file():
        captured = json.loads(historical.read_text())[protocol['retained_witness_source']['generation']]
        old_names = tuple(captured['metadata']['robot_dof_names'])
        order = [old_names.index(name) for name in NAMES]
        operations = {row['label']: row for row in captured['operations']}
        rows = [baseline.start.copy()]
        for label in protocol['retained_witness_source']['operations_reversed_in_order']:
            operation = operations[label]
            rows.append(np.asarray(operation['after']['feedback']['positions'])[order])
            rows.extend(np.asarray(row.get('joint_feedback', row.get('feedback'))['positions'])[order]
                for row in reversed(captured['samples'][operation['first_sample']:operation['last_sample']]))
            rows.append(np.asarray(operation['before']['feedback']['positions'])[order])
        rows.append(baseline.goal.copy())
        rows = np.asarray(rows)
        rows[:, 7:] = baseline.start[7:]
        yield protocol['witness_priority'][0], rows, {str(historical): digest(historical)}
    for label, index, offset in ((protocol['witness_priority'][1], 0, .6),
            (protocol['witness_priority'][2], 0, -.6), (protocol['witness_priority'][3], 2, .6),
            (protocol['witness_priority'][4], 2, -.6), (protocol['witness_priority'][5], 0, 1.),
            (protocol['witness_priority'][6], 0, -1.)):
        middle = (baseline.start+baseline.goal)/2
        middle[index] += offset
        yield label, np.array([baseline.start, middle, baseline.goal]), {}


def urdf_limits(path=URDF):
    joints = {element.get('name'): element for element in ET.parse(path).getroot().findall('joint')}
    limits = [joints[name].find('limit') for name in NAMES[:7]]
    return np.array([float(item.get('lower')) for item in limits]), np.array([float(item.get('upper')) for item in limits])


def witness_geometry(baseline, knots, protocol):
    q = densify(knots, protocol['maximum_joint_interval_rad'])
    lower_limit, upper_limit = urdf_limits()
    if np.any(q[:, :7] < lower_limit) or np.any(q[:, :7] > upper_limit):
        raise ValueError('Independent witness leaves the unchanged arm joint limits')
    frames = [part['frame'] for part in baseline.parts]
    weights = displacement_weights(baseline.fk, frames,
        [np.linalg.norm(part['vertices'], axis=1).max() for part in baseline.parts], baseline.names, .005)
    actual = [clouds(baseline.parts, candidate_world_poses(baseline.fk, baseline.names, row, baseline.base)) for row in q]
    lower = np.array([[points.min(0) for points in world] for world in actual])
    upper = np.array([[points.max(0) for points in world] for world in actual])
    motion = np.abs(np.diff(q, axis=0))@weights.T
    return dict(q=q, world=actual, lower=lower, upper=upper, weights=weights,
        maximum_interval_motion=motion.max(axis=0), intervals=len(q)-1)


def self_certificate(baseline, geometry):
    exclusions = adjacent_self_collision_pairs(URDF)
    pairs = [(first, last) for first, last in itertools.combinations(range(len(baseline.parts)), 2)
        if frozenset((baseline.parts[first]['frame'], baseline.parts[last]['frame'])) not in exclusions]
    minima = np.full(len(pairs), np.inf)
    queries = 0
    for world in geometry['world']:
        for index, (first, last) in enumerate(pairs):
            a, b = world[first], world[last]
            bound = float(aabb_distance(a.min(0), a.max(0), (b.min(0)+b.max(0))/2, (b.max(0)-b.min(0))/2))
            if bound >= minima[index]:
                continue
            result = convex_distance_details(a, b)
            minima[index] = min(minima[index], result['lower_bound_m'])
            queries += 1
    differences = np.abs(np.diff(geometry['q'], axis=0))
    rows = []
    for index, (first, last) in enumerate(pairs):
        relative = relative_weights(geometry['weights'][first], geometry['weights'][last])
        maximum_motion = float((differences@relative).max())
        rows.append(dict(frames=[baseline.parts[first]['frame'], baseline.parts[last]['frame']],
            sampled_distance_lower_bound_m=float(minima[index]), maximum_interval_relative_motion_m=maximum_motion,
            continuous_lower_bound_m=float(minima[index]-maximum_motion-2*ALLOWANCE_M)))
    return dict(nonadjacent_pairs=len(rows), minimum_continuous_lower_bound_m=min(row['continuous_lower_bound_m'] for row in rows),
        pair_bounds=rows, independent_convex_queries=queries,
        structural_exclusions=[sorted(pair) for pair in sorted(exclusions, key=lambda pair: sorted(pair))],
        method='Independent cooked convex solids; cancel shared ancestor rigid motions, then subtract conservative relative point displacement from every pair global sampled GJK separation lower bound. Physical finger pair retained.')


def endpoint_clearance(baseline, center, dimensions):
    corners = box_vertices(center, dimensions)
    rows = {}
    for label, world in baseline.endpoint_clouds.items():
        by_body = {part['body']: convex_distance_details(points, corners)['lower_bound_m']
            for part, points in zip(baseline.parts, world)}
        rows[label] = dict(minimum_m=min(by_body.values()), by_body_m=by_body)
    return rows


def intersection_proof(worlds, parts, center, dimensions, minimum_radius, *, times=None, samples=None):
    corners = box_vertices(center, dimensions)
    low, high = np.asarray(center)-np.asarray(dimensions)/2, np.asarray(center)+np.asarray(dimensions)/2
    for sample, world in enumerate(worlds):
        for part, vertices in zip(parts, world):
            if np.any(vertices.min(0) > high-minimum_radius) or np.any(vertices.max(0) < low+minimum_radius):
                continue
            witness = intersection_ball([vertices, corners])
            if witness['status'] == 'VERIFIED_POSITIVE_INTERSECTION_BALL' and witness['radius_m'] >= minimum_radius:
                return dict(status=witness['status'], positive_ball_radius_lower_bound_m=witness['radius_m'],
                    robot_body=part['body'], baseline_sample=sample,
                    planned_time=times[sample] if times is not None else None,
                    physics_step=samples[sample]['physics_step'] if samples is not None else None,
                    ball_center_m=witness['center_m'], maximum_constraint_residual_m=witness['maximum_constraint_residual_m'])
    return None


def model_witness(baseline, geometry):
    """Pure native model queries; deliberately no RRT planner construction."""
    from scripts.audit_franka_collision_model_v4_2 import native_model, _snapshot
    from scripts.audit_global_collision_v4_1 import register_numeric_scene
    enhanced = load_enhanced_collision_model()
    if enhanced.descriptor_sha256 != MODEL_SHA:
        raise ValueError('Frozen enhanced 102-sphere model changed')
    model = native_model(Path('/home/etfrobotics/isaacsim'), enhanced.descriptor_path, URDF,
        extra_exclusions=enhanced.certified_self_collision_exclusions)
    snapshot = _snapshot(baseline.snapshot)
    register_numeric_scene(model, snapshot, baseline.base)
    inspector = model['inspector']
    centers, self_collisions, world_collisions = [], [], []
    for index, q in enumerate(geometry['q']):
        if inspector.in_collision_with_obstacle(q[:7]):
            world_collisions.append(index)
        pairs = [list(pair) for pair in inspector.frames_in_self_collision(q[:7])
            if frozenset(pair) not in model['exclusions']]
        if pairs:
            self_collisions.append(dict(sample=index, pairs=pairs))
        points = np.asarray(inspector.collision_sphere_positions(q[:7]))@baseline.base[:3, :3].T+baseline.base[:3, 3]
        snapshot.assert_within_domains(points, inspector.collision_sphere_radii())
        centers.append(points)
    # Match native ordering to explicit frozen frame entries once, then verify
    # it at every witness state before using frame-specific displacement bounds.
    from scipy.optimize import linear_sum_assignment
    frames = [frame for frame, row in model['entries']]
    local_points = [np.asarray(row['center']) for frame, row in model['entries']]
    weights = displacement_weights(baseline.fk, frames, [np.linalg.norm(point) for point in local_points],
        baseline.names, .005)
    maximum_alignment = 0.
    fixed_q = geometry['q'][0].copy()
    fixed_q[7:] = .003
    pose = candidate_world_poses(baseline.fk, baseline.names, fixed_q, baseline.base)
    expected = np.array([pose[frame][:3, :3]@point+pose[frame][:3, 3] for frame, point in zip(frames, local_points)])
    native = np.asarray(centers)
    _, ordering = linear_sum_assignment(np.linalg.norm(expected[:, None]-native[0][None], axis=2))
    for index, q in enumerate(geometry['q']):
        fixed_q = q.copy()
        fixed_q[7:] = .003
        pose = candidate_world_poses(baseline.fk, baseline.names, fixed_q, baseline.base)
        expected = np.array([pose[frame][:3, :3]@point+pose[frame][:3, 3] for frame, point in zip(frames, local_points)])
        maximum_alignment = max(maximum_alignment, float(np.linalg.norm(expected-native[index, ordering], axis=1).max()))
    if maximum_alignment > 1e-8:
        raise ValueError('Independent frame-labelled sphere placement differs from the native frozen model')
    motion = np.abs(np.diff(geometry['q'], axis=0))@weights.T
    return dict(centers=native[:, ordering], radii=np.asarray(inspector.collision_sphere_radii())[ordering],
        maximum_interval_motion=motion.max(axis=0), maximum_sphere_alignment_error_m=maximum_alignment,
        world_collisions=world_collisions, self_collisions=self_collisions, model_sha256=enhanced.descriptor_sha256)


def model_obstacle_bound(model, center, dimensions):
    gaps = np.linalg.norm(np.maximum(np.abs(model['centers']-center)-dimensions/2, 0.), axis=-1)-model['radii']
    return float((gaps.min(axis=0)-model['maximum_interval_motion']-1e-9).min())


def plane_clearance(baseline, geometry):
    rows = []
    for domain in baseline.snapshot['plane_domains']:
        normal = quat_matrix(domain['orientation'])[:, 2]
        point = np.asarray(domain['position'])
        for index, part in enumerate(baseline.parts):
            minimum = min(float(np.min((world[index]-point)@normal)) for world in geometry['world'])
            bound = minimum-geometry['maximum_interval_motion'][index]-ALLOWANCE_M
            # The fixed mount has no motion and its captured source geometry
            # intentionally meets the support plane. Do not invent a safety
            # margin there or inflate the physical base below its plane.
            if part['frame'] == 'panda_link0':
                bound = minimum
            rows.append(dict(body=part['body'], plane=domain['path'], sampled_signed_minimum_m=minimum,
                continuous_signed_lower_bound_m=float(bound), qualified_mount=part['frame'] == 'panda_link0'))
    return dict(checks=rows, passed=all(row['continuous_signed_lower_bound_m'] >= 0. for row in rows))


def verify_controls(baseline, protocol):
    controls, proofs = {}, {}
    for name, item in protocol['negative_controls'].items():
        center = np.asarray(baseline.operation['target']['position'] if isinstance(item['position_m'], str) else item['position_m'])
        dimensions = np.asarray(item['dimensions_m'])
        geometry = dict(name=item['name'], prim_path='/World/ReactiveEvaluation/V43'+item['name'], kind='cuboid',
            position_m=center.tolist(), dimensions_m=dimensions.tolist(), orientation_wxyz=item['orientation_wxyz'], stationary=True)
        controls[name] = geometry
        endpoints = endpoint_clearance(baseline, center, dimensions)
        if name == 'near':
            all_world = baseline.planned_clouds+baseline.actual_clouds
            lower = min(float(aabb_distance(points.min(0), points.max(0), center, dimensions/2))
                for world in all_world for points in world)
            proofs[name] = dict(original_planned_and_actual_global_route_aabb_clearance_lower_bound_m=lower,
                endpoints=endpoints, passed=lower >= protocol['minimum_endpoint_cooked_clearance_m'])
        else:
            positive = intersection_proof([baseline.endpoint_clouds['fixed_validated_joint_goal']], baseline.parts,
                center, dimensions, protocol['minimum_original_route_intersection_ball_radius_m'])
            proofs[name] = dict(goal_positive_volume_intersection=positive, endpoints=endpoints,
                passed=positive is not None and endpoints['actual_start_home']['minimum_m'] >= protocol['minimum_endpoint_cooked_clearance_m'])
    return controls, proofs


def select_fixture(baseline, protocol, output):
    grid = protocol['candidate_grid_integer_steps']
    axes = [np.arange(first, last+1)*grid['unit_m'] for first, last in
        (grid['x_inclusive'], grid['y_inclusive'], grid['z_inclusive'])]
    centers = np.array(list(itertools.product(*axes)))
    dimensions = np.asarray(protocol['candidate_dimensions_m'])
    if len(centers)*len(dimensions) != protocol['candidates_per_witness']:
        raise ValueError('Declared grid count differs from its finite enumeration')
    records = []
    for witness_index, (name, knots, sources) in enumerate(witness_knots(baseline, protocol)):
        record = dict(witness=name, source_sha256=sources, status='STARTED', tested_candidates=[])
        records.append(record)
        try:
            geometry = witness_geometry(baseline, knots, protocol)
            self_check = self_certificate(baseline, geometry)
            plane = plane_clearance(baseline, geometry)
            record.update(self_certificate=self_check, support_plane_certificate=plane,
                knots=len(knots), states=len(geometry['q']))
            if self_check['minimum_continuous_lower_bound_m'] <= 0. or not plane['passed']:
                record['status'] = 'INDEPENDENT_WITNESS_SELF_OR_PLANE_NOT_CERTIFIED'
                continue
            scores = [continuous_aabb_scores(geometry['lower'], geometry['upper'], geometry['maximum_interval_motion'],
                centers, size) for size in dimensions]
            ordered = ordered_candidates(centers, dimensions, scores,
                protocol['minimum_continuous_witness_external_lower_bound_m'])
            record['continuous_aabb_candidates'] = len(ordered)
            model = model_witness(baseline, geometry)
            record['model_existing_world_collision_samples'] = model['world_collisions']
            record['model_self_collision_samples'] = model['self_collisions']
            if model['world_collisions'] or model['self_collisions']:
                record['status'] = 'WITNESS_NOT_CLEAR_UNDER_EXISTING_ENHANCED_CHECKS'
                continue
            for rank, candidate in enumerate(ordered):
                center, size = candidate['center'], candidate['dimensions']
                row = dict(rank=rank, center_m=center.tolist(), dimensions_m=size.tolist(),
                    continuous_cooked_aabb_lower_bound_m=candidate['score'])
                record['tested_candidates'].append(row)
                # Endpoint AABBs can reject conservatively before exact queries.
                endpoint_aabb = min(float(aabb_distance(points.min(0), points.max(0), center, size/2))
                    for world in baseline.endpoint_clouds.values() for points in world)
                if endpoint_aabb < protocol['minimum_endpoint_cooked_clearance_m']:
                    row['status'] = 'ENDPOINT_AABB_CLEARANCE_NOT_ESTABLISHED'
                    continue
                planned = intersection_proof(baseline.planned_clouds, baseline.parts, center, size,
                    protocol['minimum_original_route_intersection_ball_radius_m'], times=baseline.planned_times)
                if planned is None:
                    row['status'] = 'ORIGINAL_PLANNED_GLOBAL_ROUTE_VOLUME_OBSTRUCTION_NOT_ESTABLISHED'
                    continue
                actual = intersection_proof(baseline.actual_clouds, baseline.parts, center, size,
                    protocol['minimum_original_route_intersection_ball_radius_m'], samples=baseline.samples)
                if actual is None:
                    row['status'] = 'ACTUAL_GLOBAL_BASELINE_VOLUME_CORROBORATION_NOT_ESTABLISHED'
                    continue
                model_bound = model_obstacle_bound(model, center, size)
                row['planning_sphere_witness_continuous_margin_m'] = model_bound
                if model_bound <= protocol['minimum_planning_sphere_witness_margin_m']:
                    row['status'] = 'ENHANCED_WITNESS_CANDIDATE_CLEARANCE_NOT_ESTABLISHED'
                    continue
                endpoints = endpoint_clearance(baseline, center, size)
                if min(item['minimum_m'] for item in endpoints.values()) < protocol['minimum_endpoint_cooked_clearance_m']:
                    row['status'] = 'EXACT_ENDPOINT_CLEARANCE_NOT_ESTABLISHED'
                    continue
                row['status'] = record['status'] = 'SELECTED_BEFORE_CANDIDATE_RRT'
                raw_path = output/'independent_witness.npz'
                np.savez_compressed(raw_path, knots=knots, q=geometry['q'], hull_lower=geometry['lower'],
                    hull_upper=geometry['upper'], point_displacement_weights=geometry['weights'])
                raw_proof = output/'design_proof.json'
                write_json(raw_proof, dict(witness_records=records, selected_candidate=row,
                    planned_obstruction=planned, actual_obstruction=actual, endpoints=endpoints,
                    baseline_alignment=baseline.alignment, source_sha256={**baseline.source_sha256, **sources}))
                planned['reference'] = 'independentFKplannedtrajectory'
                actual['reference'] = 'actualNativeBaselineCookedBodyPoses'
                controls, control_proofs = verify_controls(baseline, protocol)
                if not all(item['passed'] for item in control_proofs.values()):
                    raise ValueError('Predeclared negative controls did not pass independent geometric checks')
                fixture = dict(schema_version=1, status='FROZEN_BEFORE_ANY_V4_3_CANDIDATE_RRT',
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(), candidate_RRT_calls_before_freeze=0,
                    fixture_revision_after_candidate_RRT_allowed=False,
                    source_baseline_result_sha256=digest(baseline.result_path),
                    design_protocol_sha256=digest(PROTOCOL), producer_sha256=digest(__file__),
                    fixture=dict(name='GlobalRouteDetour', prim_path='/World/ReactiveEvaluation/GlobalRouteDetourV43',
                        kind='cuboid', position_m=center.tolist(), dimensions_m=size.tolist(),
                        orientation_wxyz=[1., 0., 0., 0.], stationary=True),
                    baseline=dict(result=str(baseline.result_path), source_sha256=baseline.source_sha256,
                        start_q=baseline.start[:7].tolist(), goal_q=baseline.goal[:7].tolist(),
                        finger_positions_m=baseline.start[7:].tolist(), target_pose=baseline.operation['target'],
                        baseline_waypoints=len(baseline.operation['plan_trace']['route']),
                        independent_FK_alignment=baseline.alignment),
                    target_source=reference(baseline.result_path.parent/'frozen_target.json')
                        if (baseline.result_path.parent/'frozen_target.json').exists() else None,
                    obstruction_proof=planned, actual_obstruction_proof=actual,
                    endpoint_clearance=endpoints,
                    witness=dict(status='INDEPENDENT_NUMERIC_CONTINUOUS_WITNESS_CERTIFIED',
                        independent_of_candidate_rrt=True, name=name, knots=len(knots), states=len(geometry['q']),
                        numericcontinuousexternal_lower_bound_m=candidate['score'],
                        self_lower_bound_m=self_check['minimum_continuous_lower_bound_m'],
                        planning_model_continuous_external_lower_bound_m=model_bound,
                        planning_model_sampled_self_and_original_world_clear=True,
                        maximum_native_sphere_alignment_error_m=model['maximum_sphere_alignment_error_m'],
                        source_sha256=sources, raw_artifact=reference(raw_path), proof_artifact=reference(raw_proof),
                        qualification='Source-bound floating-point joint-linear witness, not a certificate of candidate RRT, generated spline or actual physical execution. No witness waypoints are supplied to RRT.'),
                    controls=controls, control_proofs=control_proofs,
                    acceptance=protocol['detour_acceptance'],
                    limitations=['PhysX contact/rest offsets and arbitrary physical-asset changes are outside numeric geometry.',
                        'Continuous certificate belongs only to the independent witness; actual execution remains separately sampled.',
                        'All candidate RRT successes and failures must be retained without fixture revision.'])
                return fixture, records
        except Exception as error:
            record.update(status='INDEPENDENT_DESIGN_WITNESS_REJECTED', error=f'{type(error).__name__}: {error}')
        write_json(output/'design_progress.json', records)
    return None, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/motion-planning-v4-3/fixture-design')
    parser.add_argument('--fixture', type=Path, default=ROOT/'docs/assets/motion_planning_v4_3/frozen_fixture.json')
    args = parser.parse_args(argv)
    if args.output.exists() or args.fixture.exists():
        parser.error('Refusing to overwrite a prior design or frozen fixture')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['candidate_RRT_calls_allowed_during_design'] != 0:
        raise ValueError('Design protocol permits impermissible candidate RRT calls')
    args.output.mkdir(parents=True)
    write_json(args.output/'launch.json', dict(arguments=dict(baseline_result=str(args.baseline_result),
        output=str(args.output), fixture=str(args.fixture)), protocol=reference(PROTOCOL), producer=reference(__file__)))
    began = time.monotonic()
    baseline = load_baseline(args.baseline_result)
    fixture, records = select_fixture(baseline, protocol, args.output)
    summary = dict(status='FROZEN' if fixture else 'BOUNDED_DESIGN_INFEASIBLE',
        source_sha256=baseline.source_sha256, witness_records=records, candidate_RRT_calls=0,
        elapsed_seconds=time.monotonic()-began)
    write_json(args.output/'summary.json', summary)
    if fixture:
        write_json(args.fixture, fixture)
        write_json(args.output/'frozen_fixture.json', fixture)
    print(json.dumps(dict(status=summary['status'], fixture=str(args.fixture) if fixture else None,
        fixture_sha256=digest(args.fixture) if fixture else None, witnesses_tested=len(records))))
    return 0 if fixture else 1


if __name__ == '__main__':
    raise SystemExit(main())
