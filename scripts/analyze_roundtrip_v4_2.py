#!/usr/bin/env python3
"""Offline actual roundtrip reduction and measured figures; no simulator imports.

Only recorded actual physics samples and explicitly recorded commands are
plotted. Clearance curves retain the native cooked-solid sampling limitation.
The independent design witness is never substituted for executed motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ARM = tuple(f'panda_joint{i}' for i in range(1, 8))
QUALIFICATIONS = (
    'The fixed direct-reference criterion passes: actual ordinary direct RMPflow finger solids intersect the unchanged counterfactual box, while the observed global roundtrips remain clear.',
    'All accepted global routes have two waypoints and identical per-direction joint/trajectory hashes across recorded seeds and both models. These results do not prove an obstacle-induced change to the global path, a multisegment RRT detour, stochastic search robustness, or a geometry-specific benefit of the enhanced model.',
    'Executed physical clearance is sampled. The separately frozen continuous geometric witness is not a continuous certificate of these executed trajectories.',
    'Outward cooked-solid clearance and contacts are confirmed retrospectively after close; the native Home request is gated on outward runtime/full-pose success rather than an online independent physical-clearance reduction.',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def unique_samples(rows):
    return list({(row['physics_step'], row['simulation_time']):row for row in rows}.values())


def arm_values(values, indices, names):
    if values is None:
        return None
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        return None
    native = [names.index(name) for name in ARM]
    if indices is None:
        return values[native] if len(values) == len(names) else None
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        return None
    mapping = dict(zip(indices, values))
    return np.array([mapping[index] for index in native]) if set(native) <= set(mapping) else None


def quaternion_error(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    first, second = first/np.linalg.norm(first), second/np.linalg.norm(second)
    return float(2*np.arccos(np.clip(abs(first@second), 0., 1.)))


def extract_operation(trace, operation, compact=None):
    names = trace['metadata']['robot_dof_names']
    if len(names) != len(set(names)) or not set(ARM) <= set(names):
        raise ValueError('Unambiguous explicit seven-arm joint names are required')
    samples = unique_samples(trace['samples'][operation['first_sample']:operation['last_sample']])
    if not samples:
        raise ValueError('Operation has no actual samples')
    native = [names.index(name) for name in ARM]
    origin = float(samples[0]['simulation_time'])
    times = np.array([row['simulation_time']-origin for row in samples])
    q = np.array([row['joint_feedback']['positions'] for row in samples])[:,native]
    qd = np.array([row['joint_feedback']['velocities'] for row in samples])[:,native]
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError('Nonfinite actual encoder data')
    tcp = np.array([row['tcp_pose']['position'] for row in samples])
    target = operation.get('target')
    position_error = np.array([np.linalg.norm(row['tcp_pose']['position']-np.asarray(target['position']))
        for row in samples]) if target else np.full(len(samples), np.nan)
    orientation_error = np.array([quaternion_error(row['tcp_pose']['orientation'],target['orientation'])
        for row in samples]) if target else np.full(len(samples),np.nan)
    plan = operation.get('plan_trace') or {}
    plan_names = plan.get('joint_names') or []
    commands = plan.get('commanded') or []
    if commands and not set(ARM) <= set(plan_names):
        raise ValueError('Command trace lacks explicit seven-joint names')
    order = [plan_names.index(name) for name in ARM] if commands else []
    command_times = np.array([row['sim_time']-origin for row in commands])
    command_q = np.array([np.asarray(row['q'])[order] for row in commands]).reshape(-1,7)
    action_times, applied_q = [], []
    for row in trace['actions'][operation['first_action']:operation['last_action']]:
        feedback = row.get('actual_physx_targets') or {}
        action = feedback.get('value') if feedback.get('available') else None
        values = arm_values(action.get('joint_positions'),action.get('joint_indices'),names) if action else None
        if values is not None:
            action_times.append(row['simulation_time']-origin)
            applied_q.append(values)
    guard = plan.get('actual') or []
    guard_times = np.array([row['sim_time']-origin for row in guard])
    guard_error = np.array([row['tracking_error_rad'] for row in guard])
    by_time = {row['sim_time']:row for row in commands}
    exact_errors = [float(np.max(np.abs(np.asarray(row['q'])-np.asarray(by_time[row['sim_time']]['q']))))
        for row in guard if row['sim_time'] in by_time]
    diag = operation.get('strategy_diagnostics') or {}
    route = plan.get('route')
    terminal_start, terminal_span = None, None
    duration = diag.get('trajectory_duration_seconds')
    if commands and duration is not None:
        start = float(commands[0]['sim_time'])
        terminal_commands = [row for row in commands if row['sim_time']-start >= duration-1e-8
            and np.max(np.abs(row['qd'])) <= 1e-12]
        if terminal_commands:
            terminal_start = terminal_commands[0]['sim_time']
            final_q = np.asarray(terminal_commands[-1]['q'])[order]
            good = ((np.max(np.abs(q-final_q),axis=1) <= .05)
                & (np.max(np.abs(qd),axis=1) <= diag.get('terminal_velocity_limit_rad_s',.01))
                & (position_error <= diag.get('position_tolerance_m',.025))
                & (orientation_error <= diag.get('orientation_tolerance_rad',.10)))
            began = None
            for row,valid in zip(samples,good):
                if row['simulation_time'] < terminal_start or not valid:
                    began = None
                elif began is None:
                    began = row['simulation_time']
            terminal_span = None if began is None else samples[-1]['simulation_time']-began
    solver = [row.get('dynamics',{}).get('get_solver_position_iteration_counts',{}) for row in samples]
    known = all(row.get('available') and row.get('value') is not None for row in solver)
    solver_restored = solver[0]['value'] == solver[-1]['value'] if known else None
    curve = trace.get('clearance_curves',{}).get(operation['label'],[])
    curve = unique_samples(curve)
    compact = compact or {}
    severity = compact.get('contact_severity') or {}
    summary = dict(label=operation['label'],returned_success=operation['returned_success'],
        failure_code=diag.get('failure_code'),actual_unique_physics_samples=len(samples),
        maximum_actual_arm_displacement_rad=float(np.max(np.abs(q-q[0]))),
        final_position_error_m=float(position_error[-1]),final_orientation_error_rad=float(orientation_error[-1]),
        final_measured_arm_velocity_max_rad_s=float(np.max(np.abs(qd[-1]))),
        sampled_fixture_minimum_m=min((row['minimum_m'] for row in curve),default=None),
        sampled_all_scene_minimum_m=compact.get('clearance',{}).get('minimum_m'),
        minimum_clearance_robot_link=compact.get('fixture_clearance',{}).get('closest_pair',{}),
        runtime_guard_error_max_rad=float(guard_error.max()) if len(guard_error) else None,
        exact_same_time_actual_command_error_max_rad=max(exact_errors,default=None),
        exact_same_time_pairs=len(exact_errors),
        observed_terminal_settling_sim_seconds=terminal_span,
        recorded_required_terminal_settle_seconds=diag.get('terminal_settle_seconds'),
        solver_position_iterations_restored=solver_restored,
        solver_position_iterations_before=solver[0].get('value'),
        solver_position_iterations_after=solver[-1].get('value'),
        route_waypoints=len(route) if route is not None else None,
        route_joint_names=plan.get('joint_names'),
        route_joint_polyline_sha256=numeric_hash(route) if route is not None else None,
        generated_trajectory_sha256=numeric_hash(plan['trajectory']) if plan.get('trajectory') is not None else None,
        selected_ik_index=diag.get('selected_ik_index'),
        ik_candidates_attempted=diag.get('ik_candidates_attempted'),
        observed_physical_contact_steps=severity.get('observed_physical_contact_steps'),
        maximum_penetration_depth_m=(severity.get('depth_m') or {}).get('maximum'),
        maximum_normal_pair_impulse_ns=(severity.get('normal_pair_impulse_ns') or {}).get('maximum'),
        strict_zero_reports=(compact.get('contacts') or {}).get('reported_points') == 0,
        planning_seconds={key:diag.get(key) for key in ('rrt_seconds','route_validation_seconds',
            'trajectory_generation_seconds','trajectory_validation_seconds','planning_and_validation_wall_seconds')},
        execution_sim_seconds=diag.get('execution_sim_seconds'),execution_wall_seconds=diag.get('execution_wall_seconds'))
    return dict(summary=summary,times=times,q=q,qd=qd,tcp=tcp,position_error=position_error,
        orientation_error=orientation_error,command_times=command_times,command_q=command_q,
        action_times=np.array(action_times),applied_q=np.array(applied_q).reshape(-1,7),
        guard_times=guard_times,guard_error=guard_error,curve=curve,origin=origin,
        route=None if route is None else np.asarray(route,float),
        terminal_start=None if terminal_start is None else terminal_start-origin)


def load_case(path):
    path = Path(path)
    result = json.loads(path.read_text())
    metrics = result.get('metrics') or {}
    trace_path = path.parent/metrics.get('trace_artifact','roundtrip_trace.json')
    summary = dict(result=str(path),result_sha256=digest(path),status=result['status'],
        mode=metrics.get('mode'),model=metrics.get('collision_model'),seed=metrics.get('seed'),
        repeat=path.parent.name,source_manifest_sha256=numeric_hash(result.get('hashes',{})),
        global_runtime_sha256=result.get('hashes',{}).get('robot_skill_stack/integrations/isaac/manipulation/motion/global_rrt.py'),
        native_roundtrip_acceptance=metrics.get('roundtrip_acceptance'),
        native_strict_zero_report_roundtrip=metrics.get('strict_zero_report_roundtrip'),
        contracts=result.get('contracts'),error=result.get('error'),operations=[])
    if not trace_path.exists():
        summary['actual_trace_status']='NOT_CAPTURED'
        return dict(summary=summary,operations=[])
    trace = json.loads(trace_path.read_text())
    summary.update(trace=str(trace_path),trace_sha256=digest(trace_path))
    archive_path = path.parent/'geometry_reference.npz'
    if archive_path.exists():
        with np.load(archive_path,allow_pickle=False) as archive:
            aligned = np.array_equal(archive['body_poses'],np.asarray([row['poses'] for row in trace['samples']]))
        summary['actual_body_poses_match_cooked_archive']=bool(aligned)
        if not aligned:
            raise ValueError('Actual trace/body pose archive mismatch')
        summary['geometry_reference_sha256']=digest(archive_path)
    operations=[]
    by_label={row['label']:row for row in metrics.get('operations',[])}
    for operation in trace.get('operations',[]):
        extracted=extract_operation(trace,operation,by_label.get(operation['label']))
        operations.append(extracted)
        summary['operations'].append(extracted['summary'])
    summary['actual_trace_status']='CAPTURED' if operations else 'NO_OPERATION_SAMPLES'
    summary['missing_roundtrip_directions']=[label for label in ('outward','home_return')
        if label not in {row['summary']['label'] for row in operations}]
    return dict(summary=summary,operations=operations,fixture=metrics.get('fixture_before'))


def intersection_ball(vertices,center,half):
    """Positive-volume witness, with an explicit inscribed-ball feasibility check."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    vertices,center,half=np.asarray(vertices,float),np.asarray(center,float),np.asarray(half,float)
    hull=ConvexHull(vertices)
    normals=hull.equations[:,:3];offsets=-hull.equations[:,3]
    lengths=np.linalg.norm(normals,axis=1)
    normals,offsets=normals/lengths[:,None],offsets/lengths
    A=np.r_[normals,np.eye(3),-np.eye(3)]
    b=np.r_[offsets,center+half,-center+half]
    result=linprog([0.,0.,0.,-1.],A_ub=np.c_[A,np.ones(len(A))],b_ub=b,
        bounds=[(None,None)]*3+[(0.,None)],method='highs')
    if not result.success:return dict(status='NO_POSITIVE_BALL_FOUND',solver_message=result.message)
    point=result.x[:3];slack=b-A@point
    allowance=1e-9
    radius=max(0.,float(slack.min())-allowance)
    residual=max(0.,float(np.max(A@point+result.x[3]-b)))
    return dict(status='POSITIVE_VOLUME_INTERSECTION' if radius>0. else 'NO_POSITIVE_BALL_FOUND',
        positive_ball_radius_lower_bound_m=radius,ball_center_world_m=point.tolist(),
        solver_optimal_inradius_m=float(result.x[3]),maximum_halfspace_feasibility_residual_m=residual,
        hull_vertex_halfspace_residual_m=max(0.,float(np.max(vertices@normals.T-offsets))),
        numerical_allowance_m=allowance,
        interpretation='An inscribed ball proves positive-volume intersection under the cooked-vertex convex-hull model; the1nm allowance is numerical, not a harmless-contact threshold')


def direct_intersection(case,frozen,detail_folder):
    summary=case['summary'];result_path=Path(summary['result'])
    result=json.loads(result_path.read_text());metrics=result['metrics']
    trace_path=result_path.parent/metrics.get('trace_artifact','roundtrip_trace.json')
    trace=json.loads(trace_path.read_text());paths=trace['metadata']['body_paths']
    if metrics.get('meters_per_unit')!=1.:
        raise ValueError('Fresh direct positive-volume audit requires recorded meter units')
    if metrics.get('fixture_before',{}).get('physically_present'):
        raise ValueError('Direct reference must have no physical external fixture')
    records=metrics['geometry_reference']['cooking']
    center=np.asarray(frozen['fixture']['position_m']);half=np.asarray(frozen['fixture']['dimensions_m'])/2
    witnesses=[];counts={};queries=0
    with np.load(result_path.parent/'geometry_reference.npz',allow_pickle=False) as archive:
        for name in ('panda_leftfinger','panda_rightfinger'):
            path='/Franka/'+name;indices=[index for index,row in enumerate(records) if row['body']==path]
            if len(indices)!=1 or records[indices[0]]['cooked_hulls']!=1:
                raise ValueError('Exactly one actual cooked finger hull is required')
            index=indices[0];body=paths.index(path);mesh=archive[f'body_from_mesh_{index}'];v=archive[f'vertices_{index}']
            v=v@mesh[:3,:3].T+mesh[:3,3]
            best=None;positive=0
            for sample in unique_samples(trace['samples']):
                pose=np.asarray(sample['poses'][body]);x,y,z,w=pose[3:]/np.linalg.norm(pose[3:])
                rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                world=v@rotation.T+pose[:3]
                if np.any(world.min(0)>=center+half) or np.any(world.max(0)<=center-half):continue
                measured=intersection_ball(world,center,half);queries+=1
                if measured['status']=='POSITIVE_VOLUME_INTERSECTION':
                    positive+=1
                    measured.update(robot_body=path,physics_step=sample['physics_step'],
                        simulation_time=sample['simulation_time'],actual_joint_positions=sample['joint_feedback']['positions'])
                    if best is None or measured['positive_ball_radius_lower_bound_m']>best['positive_ball_radius_lower_bound_m']:
                        best=measured
            counts[path]=positive
            if best is not None:witnesses.append(best)
    detail_folder.mkdir(parents=True,exist_ok=True)
    detail_path=detail_folder/'positive_volume_finger_witnesses.json'
    if detail_path.exists():raise ValueError('Refusing to overwrite a fresh direct intersection audit')
    detail=dict(result_sha256=digest(result_path),trace_sha256=digest(trace_path),
        geometry_sha256=digest(result_path.parent/'geometry_reference.npz'),witnesses=witnesses,
        positive_sample_counts=counts,lp_queries=queries)
    detail_path.write_text(json.dumps(detail,indent=2)+'\n')
    return dict(status='BOTH_FINGERS_POSITIVE_VOLUME_INTERSECTION' if len(witnesses)==2 else 'INCOMPLETE_POSITIVE_VOLUME_CORROBORATION',
        positive_sample_counts=counts,lp_queries=queries,
        witness_ball_radius_lower_bound_by_body_m={row['robot_body']:row['positive_ball_radius_lower_bound_m'] for row in witnesses},
        minimum_two_finger_witness_radius_m=min((row['positive_ball_radius_lower_bound_m'] for row in witnesses),default=None),
        maximum_halfspace_feasibility_residual_m=max((row['maximum_halfspace_feasibility_residual_m'] for row in witnesses),default=None),
        raw_details=str(detail_path),raw_details_sha256=digest(detail_path),
        qualification='Counterfactual unchanged box against actual captured direct motion; no physical impact was performed')


def path_comparison(cases):
    rows=[];primary={}
    for case in cases:
        if case['summary'].get('mode')!='roundtrip':continue
        for op in case['operations']:
            label=op['summary']['label'];route=op['route'];difference=None
            if route is not None:
                if label not in primary:primary[label]=route
                if route.shape==primary[label].shape:difference=float(np.max(np.abs(route-primary[label])))
            rows.append(dict(model=case['summary']['model'],seed=case['summary']['seed'],
                result=case['summary']['result'],label=label,route_waypoints=op['summary']['route_waypoints'],
                global_runtime_sha256=case['summary'].get('global_runtime_sha256'),
                route_joint_polyline_sha256=op['summary']['route_joint_polyline_sha256'],
                selected_ik_index=op['summary']['selected_ik_index'],maximum_joint_difference_to_first_route_rad=difference))
    return dict(rows=rows,interpretation=QUALIFICATIONS[1])


def render_case(case, folder, frozen,direct_reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    operations=case['operations']
    if not operations:
        return []
    folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    fig,axes=plt.subplots(7,len(operations),figsize=(6.4*len(operations),11),squeeze=False,sharex='col',constrained_layout=True)
    fig.suptitle(f"Measured {case['summary']['model']}, seed {case['summary']['seed']}, {case['summary']['repeat']}")
    for column,op in enumerate(operations):
        for joint in range(7):
            axis=axes[joint,column]
            axis.plot(op['times'],op['q'][:,joint],color='#16324f',label='Measured joint')
            if len(op['command_times']):
                axis.step(op['command_times'],op['command_q'][:,joint],where='post',color='#d65a31',ls='--',lw=1,label='Planner command')
            if len(op['action_times']):
                axis.plot(op['action_times'],op['applied_q'][:,joint],'.',ms=1.5,color='#639e6d',label='Native target readback')
            if op['terminal_start'] is not None:
                axis.axvline(op['terminal_start'],color='#888888',ls=':',lw=.8)
            axis.set_ylabel(f'q{joint+1} (rad)');axis.grid(alpha=.2)
            if joint==0:axis.set_title(op['summary']['label']);axis.legend(fontsize=7)
        axes[-1,column].set_xlabel('Operation simulation time (s)')
    path=folder/'actual_joint_trajectories.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    fig=plt.figure(figsize=(8,6),constrained_layout=True);axis=fig.add_subplot(111,projection='3d')
    for index,op in enumerate(operations):
        axis.plot(*op['tcp'].T,label=op['summary']['label'],color=('#2878b5','#d86435')[index%2])
        axis.scatter(*op['tcp'][0],s=22)
    if direct_reference is not None and case['summary'].get('mode')=='roundtrip':
        axis.plot(*direct_reference['tcp'].T,color='#555555',ls='--',lw=1.5,
            label='Ordinary direct reference (box absent)')
    box=frozen['fixture'];center=np.asarray(box['position_m']);half=np.asarray(box['dimensions_m'])/2
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half+center
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(corners[i]!=corners[j])==1:
                axis.plot(*corners[[i,j]].T,color='#777777',alpha=.6,lw=1)
    title='Measured actual TCP trajectories and frozen obstacle'
    if case['summary'].get('mode')=='direct_reference':title+=' (counterfactual box)'
    axis.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title=title)
    axis.legend()
    fig.text(.02,.01,'TCP curves are measured frame paths; finger-solid intersection is assessed independently.',fontsize=8)
    path=folder/'actual_tcp_trajectory.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    fig,axes=plt.subplots(2,len(operations),figsize=(6.4*len(operations),6),squeeze=False,constrained_layout=True)
    for column,op in enumerate(operations):
        curve=op['curve']
        if curve:
            axes[0,column].plot([row['simulation_time']-op['origin'] for row in curve],
                [row['minimum_m']*1000 for row in curve],color='#2878b5')
        else:
            axes[0,column].text(.5,.5,'No executed clearance curve captured',ha='center',transform=axes[0,column].transAxes)
        axes[0,column].axhline(0.,color='#a83030',ls=':');axes[0,column].set_ylabel('All 11 cooked bodies → obstacle (mm)')
        axes[0,column].set_title(op['summary']['label'])
        axes[1,column].plot(op['times'],op['position_error']*1000,label='TCP position error (mm)')
        axes[1,column].axhline(25.,ls=':',color='#777777',label='Original 25 mm tolerance')
        twin=axes[1,column].twinx();twin.plot(op['times'],op['orientation_error'],color='#d86435',label='Orientation error')
        twin.set_ylabel('Orientation error (rad)');twin.axhline(.10,color='#d86435',ls=':')
        axes[1,column].set_xlabel('Operation simulation time (s)');axes[1,column].legend(fontsize=7)
    fig.suptitle('Measured sampled clearance and TCP errors')
    path=folder/'actual_clearance_and_tcp_errors.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    return paths


def render_planning(cases,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for case in cases:
        for op in case['operations']:
            values=op['summary']['planning_seconds']
            if any(value is not None for value in values.values()):
                rows.append((f"{case['summary']['model']} {case['summary']['seed']} {case['summary']['repeat']} {op['summary']['label']}",values))
    if not rows:return None
    fig,axis=plt.subplots(figsize=(max(8.,len(rows)*.9),5),constrained_layout=True)
    x=np.arange(len(rows));width=.18
    legend=('RRT search','Route validation','Trajectory validation','Total planning and validation')
    for index,key in enumerate(('rrt_seconds','route_validation_seconds','trajectory_validation_seconds','planning_and_validation_wall_seconds')):
        values=[np.nan if row[1].get(key) is None else row[1][key]*1000 for row in rows]
        axis.bar(x+(index-1.5)*width,values,width,label=legend[index])
    axis.set_xticks(x,[row[0] for row in rows],rotation=35,ha='right');axis.set_ylabel('Recorded planning wall time (ms)')
    axis.legend(fontsize=8);path=folder/'planning_time_comparison.png';fig.savefig(path,dpi=150);plt.close(fig);return str(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures-dir',type=Path)
    parser.add_argument('--primary-only-figures',action='store_true')
    parser.add_argument('--direct-intersection-details',type=Path)
    parser.add_argument('--frozen-fixture',type=Path,default=Path('docs/assets/motion_planning_v4_2/roundtrip_fixture.json'))
    args=parser.parse_args(argv)
    if args.output.exists() or (args.figures_dir is not None and args.figures_dir.exists()) or (args.direct_intersection_details is not None and args.direct_intersection_details.exists()):
        parser.error('Refusing to overwrite a previous actual-data reduction')
    cases=[load_case(path) for path in args.result]
    frozen=json.loads(args.frozen_fixture.read_text())
    summary=dict(schema_version=1,source_script_sha256=digest(__file__),cases=[case['summary'] for case in cases],
        interpretation='Actual recorded motion, commands and cooked-solid clearance samples. Missing experiments remain missing. No future-execution continuous guarantee is inferred from the independent frozen numeric witness.',
        qualifications=list(QUALIFICATIONS),path_comparison=path_comparison(cases),figures=[])
    if args.direct_intersection_details is not None:
        for index,case in enumerate(cases):
            if case['summary'].get('mode')=='direct_reference':
                case['summary']['fresh_positive_volume_intersection']=direct_intersection(case,frozen,
                    args.direct_intersection_details/str(index))
    if args.figures_dir is not None:
        args.figures_dir.mkdir(parents=True)
        direct=next((case['operations'][0] for case in cases if case['summary'].get('mode')=='direct_reference' and case['operations']),None)
        global_rendered=False
        for index,case in enumerate(cases):
            if args.primary_only_figures and case['summary'].get('mode')=='roundtrip':
                if global_rendered:continue
                global_rendered=True
            label=re.sub('[^a-zA-Z0-9_-]+','_',f"{index}_{case['summary']['model']}_{case['summary']['seed']}_{case['summary']['repeat']}")
            summary['figures'].extend(render_case(case,args.figures_dir/label,frozen,direct_reference=direct))
        planning=render_planning(cases,args.figures_dir)
        if planning:summary['figures'].append(planning)
    summary['figure_sha256']={name:digest(name) for name in summary['figures']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(cases),figures=len(summary['figures']),output=str(args.output))))


if __name__=='__main__':main()
