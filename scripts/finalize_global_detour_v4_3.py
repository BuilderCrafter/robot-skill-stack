#!/usr/bin/env python3
"""Finalize control geometry around an already locked V4.3 main fixture.

No main-box search or RRT search is repeated. The original exact design and
its failed historical-control finalization remain retained and source-bound.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.collision_model import load_enhanced_collision_model
from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle
from scripts import design_global_detour_exact_v4_3 as exact
from scripts import design_global_detour_v4_3 as original


ROOT=original.ROOT
PROTOCOL=ROOT/'docs/assets/motion_planning_v4_3/control_finalization_protocol.json'
LOCK=ROOT/'docs/assets/motion_planning_v4_3/main_design_lock.json'


def assert_locked_main(lock,selected):
    geometry=lock['locked_main_fixture']
    if (selected['center'].tolist()!=geometry['position_m']
            or selected['dimensions'].tolist()!=geometry['dimensions_m']
            or geometry['orientation_wxyz']!=[1.,0.,0.,0.]):
        raise ValueError('Control finalization must not change locked main geometry')
    if (selected['planned']!=lock['planned_obstruction']
            or selected['actual']!=lock['actual_obstruction']):
        raise ValueError('Control finalization must not replace obstruction evidence')


def override_goal_control(protocol,side):
    updated=copy.deepcopy(protocol)
    updated['negative_controls']['goal_blocked']['dimensions_m']=[side]*3
    for key in protocol:
        if key!='negative_controls' and updated[key]!=protocol[key]:
            raise ValueError('Main design protocol changed during control finalization')
    before=copy.deepcopy(protocol['negative_controls'])
    after=copy.deepcopy(updated['negative_controls'])
    before['goal_blocked'].pop('dimensions_m')
    after['goal_blocked'].pop('dimensions_m')
    if before!=after:
        raise ValueError('Only goal-control dimensions may change')
    return updated


def select_control(protocol,verify):
    records=[]
    for side in protocol['goal_blocked_control']['candidate_cube_edge_lengths_m']:
        result=verify(side)
        record=dict(cube_edge_length_m=side,**result)
        records.append(record)
        if (result['actual_home_clearance_m']>=protocol['unchanged_criteria']['minimum_actual_home_clearance_m']
                and result['enhanced_home_clear'] and result['enhanced_goal_blocked']
                and result['goal_intersection_radius_m']>=protocol['unchanged_criteria']['minimum_goal_intersection_ball_radius_m']):
            return side,records
    return None,records


def native_control_verifier(baseline):
    from scripts.audit_franka_collision_model_v4_2 import native_model,_snapshot
    from scripts.audit_global_collision_v4_1 import register_numeric_scene
    enhanced=load_enhanced_collision_model()
    model=native_model(Path('/home/etfrobotics/isaacsim'),enhanced.descriptor_path,original.URDF,
        extra_exclusions=enhanced.certified_self_collision_exclusions)
    snapshot=_snapshot(baseline.snapshot)
    center=np.asarray(baseline.operation['target']['position'])

    def verify(side):
        dimensions=np.full(3,side)
        endpoints=original.endpoint_clearance(baseline,center,dimensions)
        proof=original.intersection_proof([baseline.endpoint_clouds['fixed_validated_joint_goal']],
            baseline.parts,center,dimensions,.003)
        obstacle=StaticObstacle('/World/ReactiveEvaluation/V43BlockedGoal',kind='cuboid',
            position=tuple(center),dimensions=tuple(dimensions),source_kind='Cube',approximation='exact')
        world=replace(snapshot,obstacles=(*snapshot.obstacles,obstacle))
        register_numeric_scene(model,world,baseline.base)
        inspector=model['inspector']
        home_collision=bool(inspector.in_collision_with_obstacle(baseline.start[:7]))
        goal_collision=bool(inspector.in_collision_with_obstacle(baseline.goal[:7]))
        return dict(actual_home_clearance_m=endpoints['actual_start_home']['minimum_m'],
            enhanced_home_clear=not home_collision,enhanced_goal_blocked=goal_collision,
            goal_intersection_radius_m=proof['positive_ball_radius_lower_bound_m'] if proof else 0.,
            physical_endpoints=endpoints,physical_goal_intersection=proof,
            method='Actual cooked solids and isolated native enhanced world inspector; no RRT instance or search.')
    return verify


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/motion-planning-v4-3/control-finalization')
    parser.add_argument('--fixture',type=Path,default=ROOT/'docs/assets/motion_planning_v4_3/frozen_fixture.json')
    args=parser.parse_args(argv)
    if args.output.exists() or args.fixture.exists():
        parser.error('Refusing to overwrite a finalization attempt or frozen fixture')
    protocol=json.loads(PROTOCOL.read_text())
    if protocol['candidate_RRT_calls']!=0 or protocol['main_selection_rerun_allowed']:
        raise ValueError('Invalid control-only finalization declaration')
    for path,digest in protocol['original_design_source_sha256'].items():
        if original.digest(path)!=digest:
            raise ValueError('Locked original design source changed: '+path)
    lock=json.loads(LOCK.read_text())
    for key in ('raw_proof','raw_pool','raw_witness'):
        if original.digest(lock[key]['path'])!=lock[key]['sha256']:
            raise ValueError('Locked main artifact changed: '+key)
    proof=json.loads(Path(lock['raw_proof']['path']).read_text())
    selected=proof['selected_candidate']
    selected['center']=np.asarray(selected['center'])
    selected['dimensions']=np.asarray(selected['dimensions'])
    assert_locked_main(lock,selected)
    records=proof['witness_records']
    source_witness=records[-1]
    baseline=original.load_baseline(args.baseline_result)
    if not any(Path(path).resolve()==args.baseline_result.resolve() and digest==original.digest(args.baseline_result)
            for path,digest in proof['source_sha256'].items()):
        raise ValueError('Finalization baseline differs from the locked main-design baseline')
    for path,digest in proof['source_sha256'].items():
        if original.digest(path)!=digest:
            raise ValueError('Original independent proof source changed: '+path)
    baseline.source_sha256.update({str(path):original.digest(path) for path in
        (Path(__file__),PROTOCOL,LOCK,Path(lock['raw_proof']['path']),Path(lock['raw_pool']['path']),Path(lock['raw_witness']['path']))})
    args.output.mkdir(parents=True)
    original.write_json(args.output/'launch.json',dict(arguments=dict(baseline_result=str(args.baseline_result),
        output=str(args.output),fixture=str(args.fixture)),producer=original.reference(__file__),protocol=original.reference(PROTOCOL),
        original_exact_emitter=original.reference(exact.__file__),candidate_RRT_calls=0))
    side,control_records=select_control(protocol,native_control_verifier(baseline))
    original.write_json(args.output/'control_checks.json',dict(records=control_records,
        selected_cube_edge_length_m=side,candidate_RRT_calls=0,producer_sha256=original.digest(__file__)))
    if side is None:
        print(json.dumps(dict(status='NO_DECLARED_GOAL_CONTROL_CERTIFIED',candidate_RRT_calls=0)))
        return 1
    design_protocol=json.loads(exact.PROTOCOL.read_text())
    active_protocol=override_goal_control(design_protocol,side)
    with np.load(lock['raw_witness']['path'],allow_pickle=False) as archive:
        knots=archive['knots'].copy()
        geometry=dict(q=archive['q'].copy(),lower=archive['hull_lower'].copy(),
            upper=archive['hull_upper'].copy(),weights=archive['point_displacement_weights'].copy())
    geometry['maximum_interval_motion']=(np.abs(np.diff(geometry['q'],axis=0))@geometry['weights'].T).max(axis=0)
    model=original.model_witness(baseline,geometry)
    if model['world_collisions'] or model['self_collisions']:
        raise ValueError('Unchanged locked witness is no longer clear under the frozen enhanced description')
    rebound=original.model_obstacle_bound(model,selected['center'],selected['dimensions'])
    if abs(rebound-selected['planning_model_bound_m'])>1e-10:
        raise ValueError('Model witness margin changed while finalizing controls')
    fixture=exact.freeze_document(baseline,active_protocol,selected,source_witness['witness'],knots,geometry,
        model,source_witness['self_certificate'],proof['independent_physical_witness'],
        source_witness['source_sha256'],records,lock['raw_pool'],args.output)
    # The exact emitter remains the actual emitter. Preserve its identity and
    # both old/new raw proofs, then bind this separate control component.
    fixture.update(original_design_producer_sha256=original.digest(exact.__file__),
        original_design_protocol_sha256=original.digest(exact.PROTOCOL),
        main_design_lock=original.reference(LOCK),main_design_proof=lock['raw_proof'],
        finalization_producer_sha256=original.digest(__file__),
        finalization_protocol=original.reference(PROTOCOL),
        protocol_overrides={'negative_controls.goal_blocked.dimensions_m':dict(
            original=design_protocol['negative_controls']['goal_blocked']['dimensions_m'],active=[side]*3)},
        control_finalization_proof=original.reference(args.output/'control_checks.json'))
    regenerated=fixture['witness']['raw_artifact']
    fixture['witness']['raw_artifact']=lock['raw_witness']
    fixture['witness']['regenerated_equal_numeric_arrays_artifact']=regenerated
    with np.load(regenerated['path'],allow_pickle=False) as archive:
        if not np.array_equal(archive['q'],geometry['q']) or not np.array_equal(archive['knots'],knots):
            raise ValueError('Finalization changed the independent witness')
    if (fixture['fixture']['position_m']!=lock['locked_main_fixture']['position_m']
            or fixture['fixture']['dimensions_m']!=lock['locked_main_fixture']['dimensions_m']):
        raise ValueError('Finalization changed the locked main fixture')
    original.write_json(args.fixture,fixture)
    original.write_json(args.output/'frozen_fixture.json',fixture)
    original.write_json(args.output/'summary.json',dict(status='FROZEN_WITH_INDEPENDENT_CONTROL_ONLY_FINALIZATION',
        fixture=original.reference(args.fixture),main_geometry_unchanged=True,witness_unchanged=True,
        candidate_RRT_calls=0,goal_control_side_m=side))
    print(json.dumps(dict(status='FROZEN',fixture=str(args.fixture),fixture_sha256=original.digest(args.fixture),
        main_geometry_unchanged=True,goal_control_side_m=side,candidate_RRT_calls=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
