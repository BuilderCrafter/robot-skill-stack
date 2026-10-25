#!/usr/bin/env python3
"""Second predeclared geometric filter: exact convex queries for ambiguous AABBs.

The first completed AABB-only design remains immutable. This revision keeps
its grid, dimensions, witness library and all acceptance margins, and still
constructs no RRT search. It tightens geometric knowledge, not constraints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np

from scripts import design_global_detour_v4_3 as original


ROOT = original.ROOT
PROTOCOL = ROOT/'docs/assets/motion_planning_v4_3/design_exact_protocol.json'


def fast_intersection(worlds, parts, lower, upper, center, dimensions, minimum_radius,
                      *, times=None, samples=None):
    """Same verified convex LP as v1, with vectorized broad-phase rejection."""
    corners = original.box_vertices(center, dimensions)
    lo, hi = center-dimensions/2, center+dimensions/2
    possible = np.all((lower <= hi-minimum_radius) & (upper >= lo+minimum_radius), axis=-1)
    for sample, body in np.argwhere(possible):
        proof = original.intersection_ball([worlds[sample][body], corners])
        if proof['status'] == 'VERIFIED_POSITIVE_INTERSECTION_BALL' and proof['radius_m'] >= minimum_radius:
            return dict(status=proof['status'], positive_ball_radius_lower_bound_m=proof['radius_m'],
                robot_body=parts[body]['body'], baseline_sample=int(sample),
                planned_time=times[sample] if times is not None else None,
                physics_step=samples[sample]['physics_step'] if samples is not None else None,
                ball_center_m=proof['center_m'], maximum_constraint_residual_m=proof['maximum_constraint_residual_m'])
    return None


def certify_distances(worlds, parts, center, dimensions, required_by_body):
    """Accept AABB certificates; query GJK only when the AABB is insufficient.

    A failed lower bound never counts as a physical collision. The returned
    status only says whether the original required margin was established.
    """
    corners = original.box_vertices(center, dimensions)
    minima = np.full(len(parts), np.inf)
    queries = 0
    for sample, world in enumerate(worlds):
        for index, points in enumerate(world):
            bound = float(original.aabb_distance(points.min(0), points.max(0), center, dimensions/2))
            method = 'enclosing_AABB_separation_lower_bound'
            if bound < required_by_body[index]:
                detail = original.convex_distance_details(points, corners)
                bound = detail['lower_bound_m']
                queries += 1
                method = 'independent_cooked_convex_GJK_lower_bound'
            minima[index] = min(minima[index], bound)
            if bound < required_by_body[index]:
                return dict(passed=False, failed_sample=sample, failed_body=parts[index]['body'],
                    lower_bound_m=bound, required_lower_bound_m=float(required_by_body[index]),
                    method=method, convex_queries=queries,
                    interpretation='Required positive clearance was not established; this is not itself a collision proof.')
    return dict(passed=True, by_body_lower_bound_m={part['body']:float(value) for part, value in zip(parts, minima)},
        minimum_lower_bound_m=float(minima.min()), convex_queries=queries,
        method='Conservative AABB certificate when sufficient, exact cooked-convex GJK lower bound otherwise.')


def candidate_pool(baseline, protocol, output):
    grid = protocol['candidate_grid_integer_steps']
    axes = [np.arange(first, last+1)*grid['unit_m'] for first, last in
        (grid['x_inclusive'], grid['y_inclusive'], grid['z_inclusive'])]
    centers = np.array(list(itertools.product(*axes)))
    sizes = np.asarray(protocol['candidate_dimensions_m'])
    if len(centers)*len(sizes) != protocol['candidates_per_witness']:
        raise ValueError('Exact revision changed the original finite grid')
    planned_low = np.array([[points.min(0) for points in world] for world in baseline.planned_clouds])
    planned_high = np.array([[points.max(0) for points in world] for world in baseline.planned_clouds])
    actual_low = np.array([[points.min(0) for points in world] for world in baseline.actual_clouds])
    actual_high = np.array([[points.max(0) for points in world] for world in baseline.actual_clouds])
    required = protocol['minimum_endpoint_cooked_clearance_m']
    radius = protocol['minimum_original_route_intersection_ball_radius_m']
    endpoints = list(baseline.endpoint_clouds.values())
    records, candidates = [], []
    for size in sizes:
        # One vectorized cull for the complete unchanged grid. A candidate
        # whose bounding intervals cannot overlap by 2r cannot contain the
        # required radius-r intersection ball.
        possible = []
        for start in range(0, len(centers), 32):
            batch = centers[start:start+32]
            mask = np.any(np.all((planned_low[None] <= batch[:,None,None]+size/2-radius)
                & (planned_high[None] >= batch[:,None,None]-size/2+radius),axis=-1),axis=(1,2))
            possible.extend(mask)
        for index in np.flatnonzero(possible):
            center = centers[index]
            record = dict(center_m=center.tolist(), dimensions_m=size.tolist())
            records.append(record)
            endpoint = certify_distances(endpoints, baseline.parts, center, size,
                np.full(len(baseline.parts), required))
            record['endpoint_margin_proof'] = endpoint
            if not endpoint['passed']:
                record['status'] = 'EXACT_ENDPOINT_MARGIN_NOT_ESTABLISHED'
                continue
            planned = fast_intersection(baseline.planned_clouds, baseline.parts, planned_low, planned_high,
                center, size, radius, times=baseline.planned_times)
            if planned is None:
                record['status'] = 'PLANNED_GLOBAL_VOLUME_OBSTRUCTION_NOT_ESTABLISHED'
                continue
            actual = fast_intersection(baseline.actual_clouds, baseline.parts, actual_low, actual_high,
                center, size, radius, samples=baseline.samples)
            if actual is None:
                record['status'] = 'ACTUAL_GLOBAL_VOLUME_CORROBORATION_NOT_ESTABLISHED'
                continue
            planned['reference'] = 'independentFKplannedtrajectory'
            planned['planned_q'] = baseline.operation['plan_trace']['trajectory'][planned['baseline_sample']]['q']
            actual['reference'] = 'actualNativeBaselineCookedBodyPoses'
            record.update(status='INDEPENDENT_BASELINE_OBSTRUCTION_AND_ENDPOINTS_CERTIFIED',
                planned_obstruction=planned, actual_obstruction=actual)
            candidates.append(dict(center=center, dimensions=size, endpoint=endpoint, planned=planned, actual=actual))
        print(json.dumps(dict(stage='baseline_geometric_pool', dimensions_m=size.tolist(),
            examined=len(records), independently_obstructing_clear_endpoint_boxes=len(candidates))),flush=True)
    candidates.sort(key=lambda row: (-row['endpoint']['minimum_lower_bound_m'],
        -min(row['planned']['positive_ball_radius_lower_bound_m'],row['actual']['positive_ball_radius_lower_bound_m']),
        float(np.prod(row['dimensions'])),*row['center'],*row['dimensions']))
    artifact = output/'candidate_pool.json'
    original.write_json(artifact,dict(candidate_RRT_calls=0,total_grid_boxes=len(centers)*len(sizes),
        retained=len(candidates),status_counts=dict(Counter(row['status'] for row in records)),records=records))
    return candidates, original.reference(artifact)


def freeze_document(baseline, protocol, selected, witness_name, knots, geometry, model, self_check,
                    physical, sources, records, pool_ref, output):
    center, size = selected['center'], selected['dimensions']
    witness_path = output/'independent_witness.npz'
    np.savez_compressed(witness_path,knots=knots,q=geometry['q'],hull_lower=geometry['lower'],
        hull_upper=geometry['upper'],point_displacement_weights=geometry['weights'])
    proof_path = output/'design_proof.json'
    original.write_json(proof_path,dict(witness_records=records,selected_candidate=selected,
        independent_physical_witness=physical,source_sha256={**baseline.source_sha256,**sources},
        candidate_pool=pool_ref,producer_sha256=original.digest(__file__),candidate_RRT_calls=0))
    endpoints = original.endpoint_clearance(baseline,center,size)
    controls,control_proofs = original.verify_controls(baseline,protocol)
    if not all(row['passed'] for row in control_proofs.values()):
        raise ValueError('Unchanged predeclared negative controls did not pass geometry verification')
    return dict(schema_version=1,status='FROZEN_BEFORE_ANY_V4_3_CANDIDATE_RRT',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),candidate_RRT_calls_before_freeze=0,
        fixture_revision_after_candidate_RRT_allowed=False,
        source_baseline_result_sha256=original.digest(baseline.result_path),
        design_protocol_sha256=original.digest(PROTOCOL),producer_sha256=original.digest(__file__),
        fixture=dict(name='GlobalRouteDetour',prim_path='/World/ReactiveEvaluation/GlobalRouteDetourV43',
            kind='cuboid',position_m=center.tolist(),dimensions_m=size.tolist(),
            orientation_wxyz=[1.,0.,0.,0.],stationary=True),
        baseline=dict(result=str(baseline.result_path),source_sha256=baseline.source_sha256,
            start_q=baseline.start[:7].tolist(),goal_q=baseline.goal[:7].tolist(),
            finger_positions_m=baseline.start[7:].tolist(),target_pose=baseline.operation['target'],
            baseline_waypoints=len(baseline.operation['plan_trace']['route']),
            independent_FK_alignment=baseline.alignment),
        target_source=original.reference(baseline.result_path.parent/'frozen_target.json'),
        obstruction_proof=selected['planned'],actual_obstruction_proof=selected['actual'],
        endpoint_clearance=endpoints,
        witness=dict(status='INDEPENDENT_NUMERIC_CONTINUOUS_WITNESS_CERTIFIED',
            independent_of_candidate_rrt=True,name=witness_name,knots=len(knots),states=len(geometry['q']),
            numericcontinuousexternal_lower_bound_m=physical['minimum_continuous_lower_bound_m'],
            self_lower_bound_m=self_check['minimum_continuous_lower_bound_m'],
            planning_model_continuous_external_lower_bound_m=selected['planning_model_bound_m'],
            planning_model_sampled_self_and_original_world_clear=True,
            maximum_native_sphere_alignment_error_m=model['maximum_sphere_alignment_error_m'],
            source_sha256=sources,raw_artifact=original.reference(witness_path),
            proof_artifact=original.reference(proof_path),candidate_pool=pool_ref,
            qualification='Independent source-bound floating-point joint-linear witness, not candidate RRT, spline or executed-motion swept-volume proof. No witness waypoints are supplied to RRT.'),
        controls=controls,control_proofs=control_proofs,acceptance=protocol['detour_acceptance'],
        algorithm_revision=2,previous_bounded_design=protocol['previous_bounded_design'],
        limitations=['AABB insufficiency was tightened with independent convex lower bounds; no geometric threshold was relaxed.',
            'Physical contact/rest offsets and arbitrary asset changes are not certified.',
            'All candidate RRT outcomes must be retained with the frozen geometry unchanged.'])


def select_fixture(baseline,protocol,output):
    candidates,pool_ref = candidate_pool(baseline,protocol,output)
    records=[]
    for name,knots,sources in original.witness_knots(baseline,protocol):
        record=dict(witness=name,source_sha256=sources,status='STARTED',candidate_checks=[])
        records.append(record)
        geometry=original.witness_geometry(baseline,knots,protocol)
        self_check=original.self_certificate(baseline,geometry)
        plane=original.plane_clearance(baseline,geometry)
        record.update(states=len(geometry['q']),self_certificate=self_check,support_plane_certificate=plane)
        if self_check['minimum_continuous_lower_bound_m']<=0. or not plane['passed']:
            record['status']='WITNESS_SELF_OR_SUPPORT_PLANE_NOT_CERTIFIED'
            continue
        model=original.model_witness(baseline,geometry)
        if model['world_collisions'] or model['self_collisions']:
            record.update(status='WITNESS_NOT_CLEAR_UNDER_UNCHANGED_MODEL',
                existing_world_collision_samples=model['world_collisions'],self_collisions=model['self_collisions'])
            continue
        required=(geometry['maximum_interval_motion']+original.ALLOWANCE_M+
            protocol['minimum_continuous_witness_external_lower_bound_m'])
        for rank,candidate in enumerate(candidates):
            center,size=candidate['center'],candidate['dimensions']
            check=dict(rank=rank,center_m=center.tolist(),dimensions_m=size.tolist())
            record['candidate_checks'].append(check)
            model_bound=original.model_obstacle_bound(model,center,size)
            check['planning_model_continuous_external_lower_bound_m']=model_bound
            if model_bound<=protocol['minimum_planning_sphere_witness_margin_m']:
                check['status']='UNCHANGED_ENHANCED_WITNESS_OBSTACLE_MARGIN_NOT_ESTABLISHED'
                continue
            physical=certify_distances(geometry['world'],baseline.parts,center,size,required)
            check['physical_witness']=physical
            if not physical['passed']:
                check['status']='EXACT_CONTINUOUS_PHYSICAL_WITNESS_MARGIN_NOT_ESTABLISHED'
                continue
            continuous={part['body']:physical['by_body_lower_bound_m'][part['body']]
                -float(geometry['maximum_interval_motion'][index])-original.ALLOWANCE_M
                for index,part in enumerate(baseline.parts)}
            physical.update(continuous_by_body_lower_bound_m=continuous,
                minimum_continuous_lower_bound_m=min(continuous.values()))
            candidate['planning_model_bound_m']=model_bound
            check['status']=record['status']='SELECTED_BEFORE_CANDIDATE_RRT'
            original.write_json(output/'design_progress.json',records)
            return freeze_document(baseline,protocol,candidate,name,knots,geometry,model,self_check,
                physical,sources,records,pool_ref,output),records
        record['status']='DECLARED_CANDIDATES_EXHAUSTED_WITHOUT_CERTIFIED_FIXTURE'
        original.write_json(output/'design_progress.json',records)
        print(json.dumps(dict(stage='witness_exhausted',witness=name,checks=len(record['candidate_checks']))),flush=True)
    return None,records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/motion-planning-v4-3/fixture-design-exact')
    parser.add_argument('--fixture',type=Path,default=ROOT/'docs/assets/motion_planning_v4_3/frozen_fixture.json')
    args=parser.parse_args(argv)
    if args.output.exists() or args.fixture.exists():
        parser.error('Refusing to overwrite a previous attempt or frozen fixture')
    protocol=json.loads(PROTOCOL.read_text())
    if (protocol['candidate_RRT_calls_allowed_during_design']!=0
            or protocol['original_protocol_sha256']!=original.digest(original.PROTOCOL)):
        raise ValueError('Predeclared independent protocol chain changed')
    old=protocol['previous_bounded_design']
    if original.digest(old['path'])!=old['sha256']:
        raise ValueError('The initial bounded negative design was not preserved')
    args.output.mkdir(parents=True)
    original.write_json(args.output/'launch.json',dict(arguments=dict(baseline_result=str(args.baseline_result),
        output=str(args.output),fixture=str(args.fixture)),producer=original.reference(__file__),
        protocol=original.reference(PROTOCOL),original_geometry_helpers=original.reference(original.__file__)))
    began=time.monotonic()
    baseline=original.load_baseline(args.baseline_result)
    baseline.source_sha256.update({str(path):original.digest(path) for path in
        (Path(__file__),PROTOCOL,Path(protocol['previous_bounded_design']['path']))})
    fixture,records=select_fixture(baseline,protocol,args.output)
    original.write_json(args.output/'summary.json',dict(status='FROZEN' if fixture else 'BOUNDED_EXACT_DESIGN_INFEASIBLE',
        source_sha256=baseline.source_sha256,witness_records=records,candidate_RRT_calls=0,
        elapsed_seconds=time.monotonic()-began))
    if fixture:
        original.write_json(args.fixture,fixture)
        original.write_json(args.output/'frozen_fixture.json',fixture)
    print(json.dumps(dict(status='FROZEN' if fixture else 'BOUNDED_EXACT_DESIGN_INFEASIBLE',
        fixture=str(args.fixture) if fixture else None,fixture_sha256=original.digest(args.fixture) if fixture else None,
        witnesses_tested=len(records))))
    return 0 if fixture else 1


if __name__=='__main__':
    raise SystemExit(main())
