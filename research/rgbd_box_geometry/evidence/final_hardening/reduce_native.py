"""Compact numeric receipts; archived posthoc inspection never admits a live source."""
import json, math
from pathlib import Path
ROOT=Path('outputs/motion-planning-v5/final-hardening-20261114')
def read(path):return json.loads(path.read_text())
def distribution(values):
    values=sorted(v for v in values if v is not None)
    if not values:return {'count':0}
    def quantile(q):
        at=(len(values)-1)*q;lo=math.floor(at);hi=math.ceil(at)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(count=len(values),p50=quantile(.5),p95=quantile(.95),worst=values[-1])
def health(value):
    return {k:value.get(k) for k in ('status','reason','generation','resets','published','failures',
        'source_failures','worker_failures','processing_failures','worker','last_error')}
rows=[]
for folder in sorted(ROOT.glob('*/attempt-1')):
    if not (folder/'result.json').exists():continue
    result=read(folder/'result.json');trace=read(folder/'motion_trace.json')
    session=read(folder/'session.json') if (folder/'session.json').exists() else None
    requests=session['requests'] if session else [dict(index=1,operation_index=0,
        selected_result=result['metrics']['selected_result'],record=read(folder/'autonomous.json'),
        category=read(folder/'autonomous.json')['category'],physical_evaluation=result['physical_evaluation'])]
    allpubs=read(folder/'mapping_publications.json')
    previous_end=None
    for req in requests:
        record=req['record'];selected=req['selected_result'] or {};details=selected.get('details',{})
        index=req['operation_index'];op=trace['operations'][index];reduced=result['operations'][index]
        pubs=[p for p in allpubs if p['at']<=op['wall_started'] and (previous_end is None or p['at']>previous_end)]
        rate=None if len(pubs)<2 else (pubs[-1]['health']['published']-pubs[0]['health']['published'])/(pubs[-1]['at']-pubs[0]['at'])
        matched=[p for p in pubs if p['health'].get('completion_status')=='published' and p['timestamp']==p['health'].get('input_timestamp')]
        source=details.get('planning_source',{});age=source.get('map_age_seconds')
        model=record.get('inferred');actions=trace['actions'][op['first_action']:op['last_action']]
        row=dict(case=str(folder.relative_to(ROOT)),request=req['index'],label=op['label'],
            category=req['category'],measurement_status=result['status'],contracts=result['contracts'],
            error=result['error'],selected_status=selected.get('status'),failure_code=details.get('failure_code') or selected.get('failure_code'),message=selected.get('message'),
            actual_commands=len(actions),operation={k:reduced.get(k) for k in ('id','label','returned_success','cancelled',
                'position_error_m','orientation_error_rad','measured_full_pose_within_tolerances',
                'maximum_actual_joint_displacement_rad','wall_seconds','contacts','physics_step_coverage')},
            moving_arm_sampled_clearance_m=reduced.get('clearance',{}).get('movable_arm_minimum_m'),
            source_age_s=age,freshness_headroom_s=None if age is None else 1-age,
            source_identity={k:source.get(k) for k in ('generation','revision','timestamp','original_acquisition_timestamp','evaluated_at')},
            admission=details.get('rgbd_source_admission'),source_captures=details.get('planning_source_captures'),
            selected_union=source.get('occupied_coverage'),base_preflight=details.get('base_geometry_preflight'),
            primary=record.get('primary'),selected_uses_fixed_joint_target=record.get('selected_uses_fixed_joint_target'),
            physical_evaluation=req['physical_evaluation'],detour=record.get('detour'),
            execution={k:details.get(k) for k in ('closed','final_hold','execution_solver','seed','max_iterations',
                'ik_seconds','rrt_seconds','route_validation_seconds','trajectory_validation_seconds',
                'planning_and_validation_wall_seconds','execution_wall_seconds','joint_tracking_error_rad',
                'control_interval_ms','commanded_updates')},
            timing=dict(accepted_publication_hz=rate,accepted_publication_count=len(pubs),
                timing_records=len(matched),
                request_to_first_command_s=actions[0]['at']-op['wall_started'] if actions else None,
                **{k:distribution([p['health'].get(k) for p in matched]) for k in
                    ('worker_seconds','compute_seconds','transport_seconds','queue_seconds','publish_age_seconds','worker_to_publication_seconds')},
                reconstruction_s=distribution([p['diagnostics'].get('reconstruction',{}).get('elapsed_seconds') for p in matched]),
                mapping_only_compute_s=distribution([p['diagnostics'].get('mapping_compute_seconds') for p in matched])),
            mapping_before=health(record.get('ordinary_request',{}).get('health',{})),
            mapping_after=health(record.get('mapping_after',{}).get('health',{})),
            runtime_after=record.get('runtime_after') if session else result['metrics']['runtime_after'],
            inference=None if model is None else dict(status=model['status'],reason=model['reason'],
                generation=model['generation'],revision=model['revision'],timestamp=model['timestamp'],
                elapsed_s=model['diagnostics']['elapsed_seconds'],
                models=[dict(component=m['component'],status=m['status'],geometry=m['geometry']) for m in model['models']],
                robot_components=[{k:c.get(k) for k in ('component','points','robot_envelope_points','reason')} for c in
                    model['diagnostics'].get('reconstruction',{}).get('components',[]) if c.get('reason')=='possible_robot_component']))
        rows.append(row);previous_end=op['wall_started']+op['wall_seconds']
summary=dict(raw_root=str(ROOT),qualification='Fixed-time sampled physical validation; no continuous safety certificate or general scene support.',requests=rows)
(ROOT/'numeric-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
for row in rows:
    print(row['case'],row['label'],row['selected_status'],row['failure_code'],row['source_age_s'],row['actual_commands'],
        row['operation']['position_error_m'],row['operation']['contacts'],row['timing']['accepted_publication_hz'])
