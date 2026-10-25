#!/usr/bin/env python3
"""Check the frozen original global trajectory in the native numeric model.

This is an independent CPU inspector receipt, not an RRT search or simulation.
It verifies that the enhanced model rejects the physically obstructed original
trajectory while its endpoints remain valid, and that removing the candidate
restores the same original trajectory. The independently frozen cooked-solid
positive-volume obstruction remains a separate physical-geometry argument.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle
from scripts.audit_franka_collision_model_v4_2 import native_model, _snapshot
from scripts.audit_global_collision_v4_1 import pose_matrix_xyzw, register_numeric_scene, clear_numeric_scene


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = '6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d'
PROTOCOL = ROOT/'docs/assets/motion_planning_v4_3/design_protocol.json'
DESIGNER = ROOT/'scripts/design_global_detour_v4_3.py'
CONDITIONS = ('free', 'obstacle', 'removed', 'near', 'goal_blocked')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def local_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def read_bound(reference):
    path = local_path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Frozen source changed: '+str(path))
    return json.loads(path.read_text())


def _states(values, label):
    array = np.asarray(values, float)
    if array.ndim != 2 or array.shape[1:] != (7,) or not len(array) or not np.isfinite(array).all():
        raise ValueError(label+' requires finite seven-joint states')
    return array


def _clear(row):
    return (row.get('external_collision') is False and row.get('unmasked_self_pairs') == []
        and row.get('domain_error') is None)


def evaluate_reference(route, endpoints, obstruction_index, query_condition):
    """Evaluate one immutable state array in five independently set worlds.

    ``query_condition(name, states)`` returns one inspector result per state.
    Both explicit endpoint names are mandatory; the helper never silently
    drops an endpoint or turns absence of a query into evidence of clearance.
    """
    route = _states(route, 'Original generated trajectory')
    if len(route) < 2 or set(endpoints) != {'start', 'goal'}:
        raise ValueError('Original trajectory and both named endpoints are mandatory')
    endpoint_values = _states([endpoints['start'], endpoints['goal']], 'Fixed endpoints')
    if endpoint_values.shape != (2, 7):
        raise ValueError('Both named endpoints must have seven joints')
    if (type(obstruction_index) is not int or not 0 <= obstruction_index < len(route)):
        raise ValueError('Declared physical obstruction index is outside the original trajectory')
    if (np.max(np.abs(route[0]-endpoint_values[0])) > 1e-6
            or np.max(np.abs(route[-1]-endpoint_values[1])) > 1e-6):
        raise ValueError('Original trajectory endpoints differ from the fixed validated states')
    states = np.vstack((route, endpoint_values))
    records = {}
    for condition in CONDITIONS:
        rows = query_condition(condition, states.copy())
        if len(rows) != len(states) or any(not isinstance(row, dict) or any(key not in row
                for key in ('external_collision', 'unmasked_self_pairs', 'domain_error')) for row in rows):
            raise ValueError('Incomplete native model results for '+condition)
        records[condition] = rows
    checks = dict(
        original_world_entire_reference_clear=all(_clear(row) for row in records['free']),
        candidate_obstructs_declared_physical_intersection_state=
            records['obstacle'][obstruction_index]['external_collision'] is True,
        candidate_start_clear=_clear(records['obstacle'][-2]),
        candidate_goal_clear=_clear(records['obstacle'][-1]),
        removal_restores_entire_original_reference=all(_clear(row) for row in records['removed']),
        removal_restores_identical_query_results=records['free'] == records['removed'],
        irrelevant_control_entire_reference_clear=all(_clear(row) for row in records['near']),
        blocked_goal_control_start_clear=_clear(records['goal_blocked'][-2]),
        blocked_goal_control_goal_collision=records['goal_blocked'][-1]['external_collision'] is True)
    summary = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        reference_trajectory_samples=len(route), declared_physical_obstruction_index=obstruction_index,
        fixed_endpoint_names=['start', 'goal'], conditions={})
    for condition, rows in records.items():
        summary['conditions'][condition] = dict(
            trajectory_external_collision_samples=[index for index, row in enumerate(rows[:len(route)])
                if row['external_collision']],
            trajectory_self_or_domain_failures=[index for index, row in enumerate(rows[:len(route)])
                if row['unmasked_self_pairs'] or row['domain_error'] is not None],
            start=rows[-2], goal=rows[-1])
    return summary, records


def cuboid(item):
    if item.get('kind') != 'cuboid':
        raise ValueError('Frozen V4.3 model receipt requires exact supported cuboids')
    return StaticObstacle(path=item['prim_path'], kind='cuboid', position=item['position_m'],
        orientation=item['orientation_wxyz'], dimensions=item['dimensions_m'], source_kind='Cube')


def selected_proof(fixture, raw_proof):
    """Normalize either frozen proof format without weakening any source link."""
    chosen = raw_proof['selected_candidate']
    if fixture.get('algorithm_revision', 1) == 1:
        proof, center, dimensions = raw_proof['planned_obstruction'], chosen['center_m'], chosen['dimensions_m']
    elif fixture['algorithm_revision'] == 2:
        if raw_proof['producer_sha256'] != fixture['producer_sha256'] or raw_proof['candidate_RRT_calls'] != 0:
            raise ValueError('Exact convex proof does not bind its frozen no-RRT producer')
        proof, center, dimensions = chosen['planned'], chosen['center'], chosen['dimensions']
    else:
        raise ValueError('Unsupported independently frozen design revision')
    if any(value != fixture['obstruction_proof'].get(key) for key, value in proof.items()):
        raise ValueError('Frozen original-route obstruction differs from the bound raw proof')
    if center != fixture['fixture']['position_m'] or dimensions != fixture['fixture']['dimensions_m']:
        raise ValueError('Frozen physical cuboid differs from the independently selected box')


def verify_control_override(fixture, design, declaration, lock, proof):
    """Only the separately predeclared goal-control size may differ."""
    permitted = 'negative_controls.goal_blocked.dimensions_m'
    if (set(fixture['protocol_overrides']) != {permitted}
            or declaration['permitted_protocol_overrides'] != [permitted]
            or declaration['candidate_RRT_calls'] != 0 or declaration['main_selection_rerun_allowed']
            or declaration['main_geometry_must_remain_identical'] is not True):
        raise ValueError('Control finalization permits a change outside its declared scope')
    for key in ('position_m', 'dimensions_m', 'orientation_wxyz'):
        if fixture['fixture'][key] != lock['locked_main_fixture'][key]:
            raise ValueError('Control finalization changed the locked main geometry')
    if (fixture['obstruction_proof'] != lock['planned_obstruction']
            or fixture['actual_obstruction_proof'] != lock['actual_obstruction']
            or fixture['witness']['raw_artifact'] != lock['raw_witness']):
        raise ValueError('Control finalization replaced main obstruction/witness evidence')
    override = fixture['protocol_overrides'][permitted]
    side = proof['selected_cube_edge_length_m']
    if (override['original'] != design['negative_controls']['goal_blocked']['dimensions_m']
            or override['active'] != [side]*3
            or fixture['controls']['goal_blocked']['dimensions_m'] != override['active']
            or side not in declaration['goal_blocked_control']['candidate_cube_edge_lengths_m']):
        raise ValueError('Active goal-control dimensions differ from the declared override')
    for name, control in fixture['controls'].items():
        expected = design['negative_controls'][name]
        position = fixture['baseline']['target_pose']['position'] if isinstance(expected['position_m'], str) else expected['position_m']
        if (control['position_m'] != position or control['orientation_wxyz'] != expected['orientation_wxyz']
                or (name == 'near' and control['dimensions_m'] != expected['dimensions_m'])):
            raise ValueError('Control finalization changed a control field beyond goal dimensions')
    if proof['producer_sha256'] != fixture['finalization_producer_sha256'] or proof['candidate_RRT_calls'] != 0:
        raise ValueError('Control selection receipt does not bind its no-RRT producer')
    records = proof['records']
    edges = declaration['goal_blocked_control']['candidate_cube_edge_lengths_m']
    if not records or [row['cube_edge_length_m'] for row in records] != edges[:len(records)] or records[-1]['cube_edge_length_m'] != side:
        raise ValueError('Goal-control selection did not follow declared largest-first order')
    criteria = declaration['unchanged_criteria']
    def acceptable(row):
        return (row['actual_home_clearance_m'] >= criteria['minimum_actual_home_clearance_m']
            and row['enhanced_home_clear'] and row['enhanced_goal_blocked']
            and row['goal_intersection_radius_m'] >= criteria['minimum_goal_intersection_ball_radius_m'])
    if not acceptable(records[-1]) or any(acceptable(row) for row in records[:-1]):
        raise ValueError('Selected control is not the first geometrically certified declared cube')


def load_inputs(fixture_path, protocol_path=PROTOCOL, designer_path=DESIGNER):
    fixture_path = local_path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    if (fixture['status'] != 'FROZEN_BEFORE_ANY_V4_3_CANDIDATE_RRT'
            or fixture['candidate_RRT_calls_before_freeze'] != 0
            or fixture['fixture_revision_after_candidate_RRT_allowed'] is not False):
        raise ValueError('The independent fixture was not frozen before candidate RRT evaluation')
    protocol_path, designer_path = local_path(protocol_path), local_path(designer_path)
    if digest(protocol_path) != fixture['design_protocol_sha256'] or digest(designer_path) != fixture['producer_sha256']:
        raise ValueError('Frozen design protocol or producer changed')
    protocol = json.loads(protocol_path.read_text())
    if protocol['candidate_RRT_calls_allowed_during_design'] != 0:
        raise ValueError('Frozen design protocol allows candidate RRT calls')
    sources = {str(fixture_path): digest(fixture_path), str(protocol_path):digest(protocol_path),
               str(designer_path):digest(designer_path)}
    if fixture.get('algorithm_revision', 1) == 2:
        if digest(PROTOCOL) != protocol['original_protocol_sha256']:
            raise ValueError('Exact revision changed the original design protocol')
        previous = protocol['previous_bounded_design']
        if digest(local_path(previous['path'])) != previous['sha256'] or previous['candidate_RRT_calls'] != 0:
            raise ValueError('The initial bounded negative design was not preserved')
        if fixture['previous_bounded_design'] != previous:
            raise ValueError('Exact fixture and protocol disagree about the prior bounded design')
        # The exact producer imports these first-design geometry helpers.
        # Their hash must be explicitly recorded in the frozen baseline chain.
        original_names = {str(DESIGNER), str(DESIGNER.relative_to(ROOT))}
        if not original_names.intersection(fixture['baseline']['source_sha256']):
            raise ValueError('Exact revision does not bind the original geometry helper source')
        sources[str(PROTOCOL)] = digest(PROTOCOL)
        sources[str(local_path(previous['path']))] = previous['sha256']
    for name, expected in fixture['baseline']['source_sha256'].items():
        path = local_path(name)
        if digest(path) != expected:
            raise ValueError('Frozen baseline/design source changed: '+name)
        sources[str(path)] = expected
    for name, expected in fixture['witness']['source_sha256'].items():
        path = local_path(name)
        if digest(path) != expected:
            raise ValueError('Independent witness source changed: '+name)
        sources[str(path)] = expected
    references = [fixture['witness']['raw_artifact'], fixture['witness']['proof_artifact'], fixture['target_source']]
    if 'candidate_pool' in fixture['witness']:
        references.append(fixture['witness']['candidate_pool'])
    for reference in references:
        path = local_path(reference['path'])
        if digest(path) != reference['sha256']:
            raise ValueError('Frozen proof/archive/target changed: '+str(path))
        sources[str(path)] = reference['sha256']
    if 'finalization_protocol' in fixture:
        finalizer = ROOT/'scripts/finalize_global_detour_v4_3.py'
        if (digest(finalizer) != fixture['finalization_producer_sha256']
                or fixture['original_design_producer_sha256'] != fixture['producer_sha256']
                or fixture['original_design_protocol_sha256'] != fixture['design_protocol_sha256']):
            raise ValueError('Independent control finalizer or original exact emitter changed')
        declaration = read_bound(fixture['finalization_protocol'])
        lock = read_bound(fixture['main_design_lock'])
        read_bound(fixture['main_design_proof'])
        control_proof = read_bound(fixture['control_finalization_proof'])
        for name, expected in declaration['original_design_source_sha256'].items():
            if digest(local_path(name)) != expected:
                raise ValueError('Control finalization original source changed: '+name)
            sources[str(local_path(name))] = expected
        verify_control_override(fixture, protocol, declaration, lock, control_proof)
        for reference in (fixture['finalization_protocol'], fixture['main_design_lock'],
                fixture['main_design_proof'], fixture['control_finalization_proof']):
            sources[str(local_path(reference['path']))] = reference['sha256']
        sources[str(finalizer)] = digest(finalizer)
    result_path = local_path(fixture['baseline']['result'])
    result = json.loads(result_path.read_text())
    if (result['status'] != 'PASS' or not result['metrics']['roundtrip_acceptance']
            or digest(result_path) != fixture['source_baseline_result_sha256']):
        raise ValueError('Original global baseline is not the accepted frozen result')
    trace_path = result_path.parent/result['metrics']['trace_artifact']
    trace = json.loads(trace_path.read_text())
    target = read_bound(fixture['target_source'])
    raw_proof = read_bound(fixture['witness']['proof_artifact'])
    selected_proof(fixture, raw_proof)
    operation = next(row for row in trace['operations'] if row['label'] == 'outward')
    plan = operation['plan_trace']
    # Independent capture and planner capture have distinct read timestamps;
    # every geometry/domain/robot-exclusion field must still match exactly.
    geometric = lambda snapshot:{key:value for key, value in snapshot.items() if key != 'timestamp'}
    if canonical_hash(geometric(plan['snapshot'])) != canonical_hash(geometric(result['metrics']['snapshot'])):
        raise ValueError('Original planning and independent captured worlds differ')
    if tuple(plan['joint_names']) != tuple(target['joint_names']):
        raise ValueError('Frozen goal and original trajectory have different named coordinate order')
    rows = plan['trajectory']
    trajectory = _states([row['q'] for row in rows], 'Original generated trajectory')
    endpoints = dict(start=fixture['baseline']['start_q'], goal=target['q'])
    proof = fixture['obstruction_proof']
    index = proof['baseline_sample']
    if (type(index) is not int or not 0 <= index < len(rows)
            or rows[index]['time'] != proof['planned_time']
            or proof['reference'] != 'independentFKplannedtrajectory'
            or proof['positive_ball_radius_lower_bound_m'] <= 0.):
        raise ValueError('Physical obstruction witness does not identify the original planned state')
    if 'planned_q' in proof and not np.array_equal(proof['planned_q'], trajectory[index]):
        raise ValueError('Frozen positive-volume proof q differs from the indexed original command')
    names = trace['metadata']['robot_dof_names']
    initial = result['metrics']['initial_state']
    if tuple(names[:7]) != tuple(plan['joint_names']) or len(names) != 9:
        raise ValueError('Original actual state has an unsupported joint order')
    actual = np.asarray(initial['feedback']['positions'], float)
    if actual.shape != (9,) or not np.isfinite(actual).all() or np.any(actual[7:] < .001) or np.any(actual[7:] > .005):
        raise ValueError('Original measured fingers leave the enhanced model contract')
    paths = trace['metadata']['body_paths']
    sample = trace['samples'][operation['first_sample']]
    base = pose_matrix_xyzw(sample['poses'][paths.index('/Franka/panda_link0')])
    snapshot = _snapshot(plan['snapshot'])
    if snapshot.meters_per_unit != 1. or any(item.source_kind != 'Plane' for item in snapshot.obstacles):
        raise ValueError('Original global world must be the declared meter-stage support plane only')
    original_paths = {item.path for item in snapshot.obstacles}
    additions = {name: cuboid(item) for name, item in dict(obstacle=fixture['fixture'], **fixture['controls']).items()}
    if any(item.path in original_paths for item in additions.values()):
        raise ValueError('Candidate/control colliders must be additions to the original world')
    snapshots = dict(free=snapshot, removed=snapshot,
        **{name:replace(snapshot, obstacles=(*snapshot.obstacles, item)) for name, item in additions.items()})
    return fixture, target, trajectory, endpoints, index, base, snapshots, sources


def run(args):
    output, compact = local_path(args.output), local_path(args.compact)
    if output.exists() or compact.exists():
        raise ValueError('Refusing to overwrite prior model verification evidence')
    fixture, target, trajectory, endpoints, index, base, snapshots, sources = load_inputs(
        args.fixture, args.protocol, args.designer)
    enhanced = load_enhanced_collision_model()
    if enhanced.descriptor_sha256 != MODEL_SHA or target['model_sha256'] != MODEL_SHA:
        raise ValueError('Frozen enhanced 102-sphere description changed')
    urdf = Path(args.isaac_root)/'exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf'
    if digest(urdf) != enhanced.urdf_sha256:
        raise ValueError('Independent numeric model URDF differs from its frozen certificate')
    model = native_model(args.isaac_root, enhanced.descriptor_path, urdf,
        extra_exclusions=enhanced.certified_self_collision_exclusions)
    def query_condition(name, states):
        snapshot = snapshots[name]
        register_numeric_scene(model, snapshot, base)
        rows = []
        for q in states:
            external = bool(model['inspector'].in_collision_with_obstacle(q))
            pairs = [list(pair) for pair in model['inspector'].frames_in_self_collision(q)
                if frozenset(pair) not in model['exclusions']]
            points = np.asarray(model['inspector'].collision_sphere_positions(q))@base[:3, :3].T+base[:3, 3]
            domain_error = None
            try:
                snapshot.assert_within_domains(points, model['inspector'].collision_sphere_radii())
            except ValueError as error:
                domain_error = str(error)
            rows.append(dict(external_collision=external, unmasked_self_pairs=pairs, domain_error=domain_error))
        return rows
    try:
        summary, records = evaluate_reference(trajectory, endpoints, index, query_condition)
    finally:
        clear_numeric_scene(model)
    alignment = float(fixture['witness']['maximum_native_sphere_alignment_error_m'])
    summary['checks']['frozen_model_alignment_within_design_subtracted_allowance'] = bool(
        np.isfinite(alignment) and 0. <= alignment <= 1e-9)
    summary['status'] = 'PASS' if all(summary['checks'].values()) else 'FAIL'
    output.mkdir(parents=True)
    raw = output/'queries.json'
    raw.write_text(json.dumps(dict(records=records, trajectory=trajectory.tolist(), endpoints=endpoints),
        indent=2, allow_nan=False)+'\n')
    for path in (Path(__file__), enhanced.descriptor_path, enhanced.descriptor_path.with_suffix('.json'), urdf,
                 ROOT/'scripts/audit_franka_collision_model_v4_2.py', ROOT/'scripts/audit_global_collision_v4_1.py'):
        sources[str(path)] = digest(path)
    summary.update(schema_version=1, producer_sha256=digest(__file__), model_sha256=enhanced.descriptor_sha256,
        primitive_count=len(model['entries']), fixture_sha256=digest(local_path(args.fixture)),
        frozen_witness_maximum_native_sphere_alignment_error_m=alignment,
        source_sha256=sources, reference_q_sha256=canonical_hash(trajectory[index].tolist()),
        reference_q_time_seconds=fixture['obstruction_proof']['planned_time'],
        raw_artifact=dict(path=str(raw), sha256=digest(raw)), RRT_calls=0, SimulationApp_started=False,
        meaning='Native CPU inspector only: the physically proven blocked original generated trajectory is rejected by the enhanced planning model, while endpoints remain clear and removal restores the same reference. No candidate RRT search or execution is performed here.',
        limitations=['All trajectory queries are sampled; this receipt is not a continuous motion or physical execution certificate.',
            'Physical positive-volume obstruction comes from the independently frozen cooked-solid proof, not these planning spheres.',
            'Internal native RRT search policy is not measured by inspector queries.'])
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'compact.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'], checks=summary['checks'], samples=len(trajectory))))
    return summary['status'] == 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT/'docs/assets/motion_planning_v4_3/frozen_fixture.json')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL,
        help='The explicitly frozen protocol, including a source-bound exact-convex revision if selected')
    parser.add_argument('--designer', type=Path, default=DESIGNER,
        help='Source file whose SHA256 is the selected fixture producer')
    parser.add_argument('--isaac-root', type=Path, default=Path('/home/etfrobotics/isaacsim'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/motion-planning-v4-3/reference-model-check')
    parser.add_argument('--compact', type=Path, default=ROOT/'docs/assets/motion_planning_v4_3/reference_model_check.json')
    args = parser.parse_args(argv)
    return 0 if run(args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
