#!/usr/bin/env python3
"""Independent numeric cooked-solid audit of retained rejected IK endpoints.

No SimulationApp or native Lula import is used. Candidate poses are numeric
counterfactuals placed by URDF-only FK, validated against actual recorded body
poses. Positive convex distance and verified positive-radius intersection balls
are different evidence. Checking four IK branches never proves global physical
infeasibility; failed route/trajectory interiors cannot be diagnosed at endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix
from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import adjacent_self_collision_pairs
from scripts.audit_global_collision_v4_1 import body_mesh_vertices, pose_matrix_xyzw
from scripts.obstacle_representation_v3_cylinder import convex_distance_details


ROOT = Path(__file__).resolve().parents[1]
URDF = Path('/home/etfrobotics/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf')
NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))+('panda_finger_joint1', 'panda_finger_joint2')
ALLOWANCE = 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_label(path):
    folder = Path(path).resolve().parent
    return str(folder.relative_to(ROOT/'outputs')) if folder.is_relative_to(ROOT/'outputs') else str(folder)


def vector(text):
    value = np.fromstring(text, sep=' ')
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Finite URDF xyz/rpy/axis triples are required')
    return value


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if axis.shape != (3,) or not np.isfinite(axis).all() or norm < 1e-15 or not np.isfinite(angle):
        raise ValueError('Finite nonzero URDF axis and angle required')
    x, y, z = axis/norm
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class IndependentUrdfFK:
    """URDF origin Rz(yaw)Ry(pitch)Rx(roll), followed by joint motion."""
    def __init__(self, path):
        document = ET.parse(path).getroot()
        self.joints = []
        children = set()
        for element in document.findall('joint'):
            origin = element.find('origin')
            xyz = vector('0 0 0' if origin is None else origin.get('xyz', '0 0 0'))
            rpy = vector('0 0 0' if origin is None else origin.get('rpy', '0 0 0'))
            transform = np.eye(4)
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2])@
                axis_rotation([0, 1, 0], rpy[1])@axis_rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            axis_element = element.find('axis')
            axis = vector('0 0 1' if axis_element is None else axis_element.get('xyz', '0 0 1'))
            kind = element.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported URDF joint type')
            if kind != 'fixed' and not np.isclose(np.linalg.norm(axis), 1., atol=1e-12, rtol=0):
                raise ValueError('The frozen physical model requires unit joint axes')
            child = element.find('child').get('link')
            children.add(child)
            self.joints.append(Joint(element.get('name'), kind, element.find('parent').get('link'),
                                     child, transform, axis))
        roots = {entry.get('name') for entry in document.findall('link')}-children
        if len(roots) != 1:
            raise ValueError('A connected URDF with one root is required')
        self.root, = roots

    def poses(self, names, positions):
        values = np.asarray(positions, float)
        if len(names) != len(set(names)) or values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError('Unique names and finite measured joint positions are required')
        coordinates = dict(zip(names, values))
        result, remaining = {self.root: np.eye(4)}, list(self.joints)
        while remaining:
            progress = False
            for joint in remaining[:]:
                if joint.parent not in result:
                    continue
                motion = np.eye(4)
                if joint.kind != 'fixed' and joint.name not in coordinates:
                    raise ValueError('Missing independent physical joint coordinate: '+joint.name)
                value = coordinates.get(joint.name, 0.)
                if joint.kind in ('revolute', 'continuous'):
                    motion[:3, :3] = axis_rotation(joint.axis, value)
                elif joint.kind == 'prismatic':
                    motion[:3, 3] = joint.axis*value
                # Both captured finger coordinates are independent physical
                # observations; never replace finger2 by its URDF mimic rule.
                result[joint.child] = result[joint.parent]@joint.origin@motion
                remaining.remove(joint)
                progress = True
            if not progress:
                raise ValueError('Disconnected/cyclic URDF joint graph')
        return result


def intersection_ball(vertex_sets, extra_halfspaces=()):
    """Verify a common positive-radius ball; zero distance alone is insufficient."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    equations = [ConvexHull(np.asarray(vertices, float)).equations for vertices in vertex_sets]
    if len(extra_halfspaces):
        equations.append(np.asarray(extra_halfspaces, float))
    planes = np.concatenate(equations)
    normals, offsets = planes[:, :3], planes[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    if not np.isfinite(planes).all() or np.any(norms <= 0):
        raise ValueError('Finite nondegenerate halfspaces are required')
    normals, offsets = normals/norms[:, None], offsets/norms
    matrix = np.c_[normals, np.ones(len(normals))]
    answer = linprog([0., 0., 0., -1.], A_ub=matrix, b_ub=-offsets,
        bounds=[(None, None)]*3+[(0., None)], method='highs')
    if not answer.success:
        return dict(status='NO_VERIFIED_INTERSECTION_BALL', solver_status=int(answer.status))
    center, raw_radius = answer.x[:3], float(answer.x[3])
    residual = float(np.max(normals@center+offsets+raw_radius))
    radius = max(0., raw_radius-max(0., residual)-1e-9)
    verified_residual = float(np.max(normals@center+offsets+radius))
    positive = radius > 0. and verified_residual <= 0.
    return dict(status='VERIFIED_POSITIVE_INTERSECTION_BALL' if positive else 'NO_POSITIVE_RADIUS_PROVEN',
        center_m=center.tolist(), radius_m=radius, raw_radius_m=raw_radius,
        maximum_constraint_residual_m=verified_residual,
        scope='Floating-point verified halfspace witness of numeric solid intersection, not observed native contact.')


def classify_relation(detail, ball, *, mounted_support=False, allowance=ALLOWANCE):
    if mounted_support and detail['lower_bound_m'] >= 0.:
        return 'QUALIFIED_NONPENETRATING_MOUNT'
    if detail['lower_bound_m'] > allowance:
        return 'SEPARATED_BEYOND_FRAME_ALLOWANCE'
    if ball and ball.get('status') == 'VERIFIED_POSITIVE_INTERSECTION_BALL' and ball['radius_m'] > allowance:
        return 'SOLID_INTERSECTION_WITNESS'
    if ball and ball.get('status') == 'VERIFIED_POSITIVE_INTERSECTION_BALL':
        return 'SMALL_NUMERIC_INTERSECTION_WITHIN_FRAME_ALLOWANCE'
    return 'TOUCH_PROXIMITY_OR_NUMERICALLY_UNRESOLVED'


def classify_candidate(status, error, relations):
    if any(row['classification'] == 'SOLID_INTERSECTION_WITNESS' for row in relations):
        return 'NUMERIC_PHYSICAL_INTERSECTION_AT_IK_ENDPOINT'
    if any(row['classification'] == 'SMALL_NUMERIC_INTERSECTION_WITHIN_FRAME_ALLOWANCE' for row in relations):
        return 'NUMERIC_INTERSECTION_WITHIN_ALLOWANCE_UNCERTAIN_PHYSICAL_ATTRIBUTION'
    clear = bool(relations) and all(row['classification'] in (
        'SEPARATED_BEYOND_FRAME_ALLOWANCE', 'QUALIFIED_NONPENETRATING_MOUNT') for row in relations)
    collision_rejection = status == 'invalid_goal' and (
        'Configuration intersects' in error or 'non-adjacent self collision' in error)
    if clear and collision_rejection:
        return 'PHYSICALLY_CLEAR_ENDPOINT_REJECTED_BY_CONSERVATIVE_MODEL'
    if clear and status == 'search_exhausted':
        return 'PHYSICALLY_CLEAR_IK_ENDPOINT_BOUNDED_SEARCH_FAILED'
    if status in ('route_invalid', 'trajectory_failed'):
        return 'ENDPOINT_ONLY_CANNOT_DIAGNOSE_FAILED_ROUTE_OR_TRAJECTORY'
    if clear:
        return 'PHYSICALLY_CLEAR_ENDPOINT_OTHER_REJECTION_REASON'
    return 'ENDPOINT_PHYSICAL_CLASSIFICATION_UNRESOLVED'


def _world_vertices(vertices, pose):
    return vertices@pose[:3, :3].T+pose[:3, 3]


def _solid_relation(first, last):
    try:
        detail = convex_distance_details(first, last)
        ball = intersection_ball([first, last]) if detail['lower_bound_m'] <= ALLOWANCE else None
        return dict(**detail, intersection_ball=ball, classification=classify_relation(detail, ball))
    except Exception as error:
        return dict(distance_m=None, lower_bound_m=None, upper_bound_m=None,
            classification='NUMERIC_QUERY_FAILED', error=str(error))


def physical_relations(parts, poses, obstacles, exclusions):
    """All eleven cooked bodies externally; nonadjacent pairs independently."""
    world = {part['body']: _world_vertices(part['vertices'], poses[part['body']]) for part in parts}
    result = []
    for body, vertices in world.items():
        for obstacle in obstacles:
            rotation = quat_matrix(obstacle['orientation'])
            if obstacle['source_kind'] == 'Plane':
                point = np.asarray(obstacle['position'])+rotation[:, 2]*obstacle['dimensions'][2]/2
                normal = rotation[:, 2]
                distance = float(np.min((vertices-point)@normal))
                detail = dict(distance_m=distance, lower_bound_m=distance, upper_bound_m=distance)
                ball = intersection_ball([vertices], [np.r_[normal, -normal@point]]) if distance <= ALLOWANCE else None
                mount = body == 'panda_link0'
                relation = dict(**detail, intersection_ball=ball,
                    classification=classify_relation(detail, ball, mounted_support=mount))
            elif obstacle['kind'] == 'cuboid' and obstacle['approximation'] == 'exact':
                half = np.asarray(obstacle['dimensions'])/2
                corners = np.array(list(itertools.product(*[(-value, value) for value in half])))
                corners = corners@rotation.T+obstacle['position']
                relation = _solid_relation(vertices, corners)
            else:
                raise ValueError('Independent endpoint audit supports actual exact boxes and physical planes only')
            result.append(dict(kind='external', body=body, obstacle=obstacle['path'], **relation))
    for first, last in itertools.combinations(world, 2):
        if frozenset((first, last)) in exclusions:
            continue
        result.append(dict(kind='self', bodies=[first, last], **_solid_relation(world[first], world[last])))
    return result


def candidate_world_poses(fk, names, positions, base):
    relative = fk.poses(names, positions)
    root = base@np.linalg.inv(relative['panda_link0'])
    return {name: root@pose for name, pose in relative.items()}


def audit_case(path, urdf, *, budget):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get('measurement_status', payload['status']) != 'PASS':
        raise ValueError('Only valid retained native measurements can be audited')
    metrics = payload['metrics']
    if not metrics['geometry_reference']['cooked_only']:
        raise ValueError('Independent cooked geometry required')
    folder = path.parent
    plan_path, geometry_path, trace_path = (folder/name for name in ('plan_trace.json', 'geometry_reference.npz', 'trace.json'))
    plan, trace = json.loads(plan_path.read_text()), json.loads(trace_path.read_text())
    snapshot = plan.get('snapshot')
    diagnostics = metrics.get('strategy_diagnostics', {})
    attempts = diagnostics.get('ik_attempts', [])
    if not attempts:
        return dict(case=case_label(path), candidates=[], classification='NO_RETAINED_IK_ATTEMPTS',
            source_failure=metrics.get('motion'), source_sha256={str(path): digest(path), str(urdf): digest(urdf)},
            limitation='No physical candidate or goal infeasibility is inferred.'), 0
    if snapshot is None or snapshot['meters_per_unit'] != 1.:
        raise ValueError('A retained meter-stage frozen planning snapshot is required')
    body_paths = tuple(metrics['contact_reference']['sensor_paths'])
    names = tuple(metrics.get('robot_dof_names', trace.get('metadata', {}).get('robot_dof_names', NAMES)))
    active_names = tuple(diagnostics['active_joint_names'])
    if len(names) != 9 or set(names) != set(NAMES) or len(set(names)) != 9 or set(active_names) != set(NAMES[:7]):
        raise ValueError('Unambiguous retained physical/RRT joint ordering is required')
    recorded_urdf_hash = diagnostics.get('sdk_configuration_sha256', {}).get('urdf_path')
    if recorded_urdf_hash is not None and recorded_urdf_hash != digest(urdf):
        raise ValueError('Independent FK URDF differs from the retained planning source')
    parts = []
    with np.load(geometry_path, allow_pickle=False) as archive:
        for record in metrics['geometry_reference']['cooking']:
            if record['cooked_hulls'] != 1:
                raise ValueError('This bounded audit requires the captured one-hull-per-body model')
            index = len(parts)
            parts.append(dict(body=record['body'].rsplit('/', 1)[-1],
                vertices=body_mesh_vertices(archive[f'vertices_{index}'], archive[f'body_from_mesh_{index}'])))
    if len(parts) != 11 or len({part['body'] for part in parts}) != 11:
        raise ValueError('All eleven independent cooked bodies are required')
    fk = IndependentUrdfFK(urdf)
    checks = []
    for sample_index in sorted({0, len(trace['samples'])//2, len(trace['samples'])-1}):
        sample = trace['samples'][sample_index]
        measured = np.asarray(sample['joint_feedback']['positions'], float)
        actual = {name.rsplit('/', 1)[-1]: pose_matrix_xyzw(sample['poses'][index])
            for index, name in enumerate(body_paths)}
        poses = candidate_world_poses(fk, names, measured, actual['panda_link0'])
        for body, physical in actual.items():
            vertices = next(part['vertices'] for part in parts if part['body'] == body)
            checks.append(dict(sample=sample_index, body=body,
                translation_error_m=float(np.linalg.norm(poses[body][:3, 3]-physical[:3, 3])),
                maximum_cooked_vertex_alignment_error_m=float(np.linalg.norm(
                    _world_vertices(vertices, poses[body])-_world_vertices(vertices, physical), axis=1).max()),
                rotation_matrix_max_error=float(np.max(abs(poses[body][:3, :3]-physical[:3, :3])))))
    if max(row['maximum_cooked_vertex_alignment_error_m'] for row in checks) > ALLOWANCE:
        raise ValueError('Independent nine-DOF FK does not align with captured physical frames')
    initial = trace['samples'][0]
    measured = np.asarray(initial['joint_feedback']['positions'], float)
    finger_feedback = diagnostics.get('collision_model_fingers', {}).get('measured_positions_m')
    finger_source = 'First retained actual trace observation'
    if finger_feedback is not None:
        values = np.asarray(finger_feedback, float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('Invalid retained planning-time measured fingers')
        for name, value in zip(NAMES[7:], values):
            measured[names.index(name)] = value
        finger_source = 'Retained native planning-time measured enhanced-model guard'
    base = pose_matrix_xyzw(initial['poses'][body_paths.index('/Franka/panda_link0')])
    exclusions = adjacent_self_collision_pairs(urdf)
    candidates, used = [], 0
    for entry in attempts:
        row = dict(attempt=entry, numeric_scope='Counterfactual IK endpoint at actual planning-start finger positions; not observed physical execution.')
        if 'ik_q' not in entry:
            row['classification'] = 'NO_RETAINED_NUMERIC_IK_CANDIDATE'
        elif entry.get('status') in ('planned', 'duplicate_ik'):
            row['classification'] = 'ACCEPTED_OR_DUPLICATE_ENDPOINT_RETAINED_WITHOUT_REAUDIT'
        elif used >= budget:
            row['classification'] = 'NOT_AUDITED_BOUNDED_BUDGET'
        else:
            candidate = measured.copy()
            q = np.asarray(entry['ik_q'], float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('Invalid retained numeric IK candidate')
            for name, value in zip(active_names, q):
                candidate[names.index(name)] = value
            poses = candidate_world_poses(fk, names, candidate, base)
            relations = physical_relations(parts, poses, snapshot['obstacles'], exclusions)
            row.update(physical_joint_positions=candidate.tolist(), physical_relations=relations,
                classification=classify_candidate(entry.get('status', ''), entry.get('error', ''), relations))
            used += 1
        candidates.append(row)
    return dict(case=case_label(path), candidates=candidates, independent_fk_alignment=checks,
        global_collision_model=diagnostics.get('global_collision_model', 'stock'),
        physical_finger_feedback_source=finger_source,
        measured_finger_positions_m=measured[[names.index(name) for name in NAMES[7:]]].tolist(),
        numeric_frame_allowance_m=ALLOWANCE, source_sha256={str(source): digest(source)
            for source in (path, plan_path, geometry_path, trace_path, urdf)},
        physical_self_scope='All cooked nonadjacent pairs, including actual finger/finger and finger/link6; only existing URDF fixed/adjacent pairs excluded.'), used


def audit_actual_case(path, urdf, *, per_phase=3):
    """Measure every actual nonadjacent cooked pair, independently of Lula flags."""
    from scripts.audit_franka_collision_model_v4_2 import representative_indices

    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get('measurement_status', payload['status']) != 'PASS':
        raise ValueError('Actual self audit requires a valid retained physical capture')
    metrics = payload['metrics']
    if not metrics['geometry_reference']['cooked_only'] or metrics.get('meters_per_unit', 1.) != 1.:
        raise ValueError('Actual cooked meter-stage reference required')
    folder = path.parent
    trace_path = folder/metrics.get('trace_artifact', 'trace.json')
    geometry_path = folder/'geometry_reference.npz'
    trace = json.loads(trace_path.read_text())
    body_paths = tuple(metrics['contact_reference']['sensor_paths'])
    parts = []
    with np.load(geometry_path, allow_pickle=False) as archive:
        for record in metrics['geometry_reference']['cooking']:
            if record['cooked_hulls'] != 1:
                raise ValueError('The frozen actual audit expects one hull per body')
            index = len(parts)
            parts.append(dict(body=record['body'].rsplit('/', 1)[-1],
                vertices=body_mesh_vertices(archive[f'vertices_{index}'], archive[f'body_from_mesh_{index}'])))
    if len(parts) != 11 or len({part['body'] for part in parts}) != 11:
        raise ValueError('All eleven actual cooked bodies required')
    physical_frames = {part['body'] for part in parts}
    exclusions = adjacent_self_collision_pairs(urdf)
    excluded_pairs = sorted(sorted(pair) for pair in exclusions if pair <= physical_frames)
    states = []
    for index in representative_indices(trace['samples'], per_phase):
        sample = trace['samples'][index]
        actual = {name.rsplit('/', 1)[-1]: pose_matrix_xyzw(sample['poses'][body])
            for body, name in enumerate(body_paths)}
        relations = physical_relations(parts, actual, [], exclusions)
        finger = next(row for row in relations if set(row['bodies']) ==
                      {'panda_leftfinger', 'panda_rightfinger'})
        states.append(dict(case=case_label(path), sample=index, phase=sample.get('phase', 'recorded_route'),
            simulation_time=sample.get('simulation_time'), physics_step=sample.get('physics_step'),
            actual_tcp_pose=sample.get('tcp_pose'),
            measured_joint_positions=sample['joint_feedback']['positions'],
            nonadjacent_pair_queries=len(relations), pairs=relations,
            physical_finger_pair=finger))
    return dict(case=case_label(path), states=states, structural_exclusions=excluded_pairs,
        source_sha256={str(source): digest(source) for source in (path, trace_path, geometry_path, urdf)},
        scope='Direct actual body poses and all cooked nonadjacent solid pairs, without consulting Lula flags or PhysX self-contact reports. Physical finger pair is retained separately and is not excluded.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', default=[], type=Path)
    parser.add_argument('--actual-case', action='append', default=[], type=Path,
        help='Audit all actual cooked nonadjacent pairs from each retained standard native result')
    parser.add_argument('--per-phase', type=int, default=3)
    parser.add_argument('--urdf', type=Path, default=URDF)
    parser.add_argument('--max-candidates', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/motion-planning-v4-2/rejected-ik-audit')
    parser.add_argument('--compact', type=Path)
    args = parser.parse_args(argv)
    if not args.case and not args.actual_case:
        raise ValueError('At least one rejected-endpoint or actual-state source is required')
    if not 1 <= args.max_candidates <= 64:
        raise ValueError('Candidate budget must be from1through64')
    budget, cases = args.max_candidates, []
    for path in args.case:
        result, used = audit_case(path, args.urdf, budget=budget)
        cases.append(result)
        budget -= used
    actual_cases = [audit_actual_case(path, args.urdf, per_phase=args.per_phase) for path in args.actual_case]
    counts = Counter(row['classification'] for case in cases for row in case['candidates'])
    conspicuous = [dict(case=case['case'], candidate=row['attempt'], classification=row['classification'])
        for case in cases for row in case['candidates']
        if row['classification'] == 'PHYSICALLY_CLEAR_ENDPOINT_REJECTED_BY_CONSERVATIVE_MODEL']
    summary = dict(schema_version=1, status='BOUNDED_NUMERIC_ENDPOINT_AND_ACTUAL_SELF_AUDIT', producer_sha256=digest(__file__),
        candidate_classifications=dict(counts), physically_clear_model_rejected_endpoints=conspicuous,
        limitations=['Physical candidate poses are independently checked URDF counterfactuals, not actual observed configurations.',
            'No finite IK candidate audit establishes whole-goal or whole-path infeasibility.',
            'Route/trajectory rejection requires its failed interior configuration; endpoint clearance alone cannot diagnose it.',
            'Positive intersection balls establish numeric solid overlap, separate from native contacts, offsets and physical execution.',
            'The nominal nonpenetrating mount remains explicitly qualified and source/frame conditional.'])
    actual_pairs = [pair for case in actual_cases for state in case['states'] for pair in state['pairs']]
    actual_bounds = [row['lower_bound_m'] for row in actual_pairs if row.get('lower_bound_m') is not None]
    summary['actual_self_geometry'] = dict(states=sum(len(case['states']) for case in actual_cases),
        source_sha256={path: value for case in actual_cases for path, value in case['source_sha256'].items()},
        pair_queries=len(actual_pairs), pair_classifications=dict(Counter(row['classification'] for row in actual_pairs)),
        minimum_separation_lower_bound_m=min(actual_bounds) if actual_bounds else None,
        nonseparated_or_unresolved=[dict(case=case['case'], sample=state['sample'], phase=state['phase'], pair=pair)
            for case in actual_cases for state in case['states'] for pair in state['pairs']
            if pair['classification'] != 'SEPARATED_BEYOND_FRAME_ALLOWANCE'],
        scope='All measured nonadjacent pairs, including physical finger/finger; structural exclusions are explicitly listed in raw evidence. No absence-of-contact-report inference.')
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output/'candidates.json'
    raw.write_text(json.dumps(dict(rejected_ik_cases=cases, actual_self_cases=actual_cases), indent=2, allow_nan=False)+'\n')
    summary['raw_artifact'] = dict(path=str(raw), sha256=digest(raw))
    (args.output/'compact.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.compact:
        args.compact.parent.mkdir(parents=True, exist_ok=True)
        args.compact.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(candidate_classifications=summary['candidate_classifications'],
        actual_self_states=summary['actual_self_geometry']['states'],
        actual_self_pair_queries=len(actual_pairs),
        minimum_actual_self_lower_bound_m=summary['actual_self_geometry']['minimum_separation_lower_bound_m'])))


if __name__ == '__main__':
    main()
