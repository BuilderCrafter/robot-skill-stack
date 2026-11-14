"""Offline numeric reduction; run with the unchanged Isaac Python, without Kit."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json
import numpy as np
from research.rgbd_box_geometry.score import compare,evidence_audit
from robot_skill_stack.world.spatial_mapping.reconstruction import InferredGeometry,combine
ROOT=Path('outputs/motion-planning-v5/autonomous-integration-20261114')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distribution(values):
 a=np.asarray([v for v in values if v is not None],float)
 return {} if not len(a) else dict(count=len(a),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),worst=float(a.max()))
seal=read(ROOT/'predictions-seal.json')
assert all(digest(ROOT/p)==h for p,h in seal['files'].items())
rows=[]
for folder in sorted(ROOT.glob('*/attempt-*')):
 if not (folder/'result.json').exists():continue
 result=read(folder/'result.json');metrics=result['metrics'];selected=metrics['selected_result'] or {};details=selected.get('details',{})
 autonomous=read(folder/'autonomous.json') if (folder/'autonomous.json').exists() else {}
 pubs=read(folder/'mapping_publications.json');camera=read(folder/'camera_acquisition.json');trace=read(folder/'motion_trace.json')
 operation=trace['operations'][0];period=operation['wall_started']-camera[0]['at'];semantic=metrics['runtime_before']['semantic']
 pubs=[p for p in pubs if p['at']<=operation['wall_started']]
 rate=None if len(pubs)<2 else (pubs[-1]['health']['published']-pubs[0]['health']['published'])/(pubs[-1]['at']-pubs[0]['at'])
 timing_pubs=[p for p in pubs if p['health'].get('completion_status')=='published' and p['timestamp']==p['health'].get('input_timestamp')]
 source=details.get('planning_source',{});age=source.get('map_age_seconds')
 if age is None and 'timestamp' in details and 'evaluated_at' in details:age=details['evaluated_at']-details['timestamp']
 row=dict(case=str(folder.relative_to(ROOT)),category=autonomous.get('category','GT_COMPATIBILITY_CONTROL'),
  selected_status=selected.get('status'),failure_code=details.get('failure_code') or selected.get('failure_code'),message=selected.get('message'),
  contracts=result['contracts'],error=result['error'],source_age_s=age,freshness_headroom_s=None if age is None else 1-age,
  readiness=autonomous.get('readiness'),production_capture_unmodified=autonomous.get('production_capture_unmodified'),
  physical_evaluation=result['physical_evaluation'],detour=autonomous.get('detour'),operations=result['operations'],
  timings=dict(accepted_mapping_publication_hz=rate,matched_accepted_completion_records=len(timing_pubs),semantic_published=semantic['published'],semantic_attempts=semantic['attempts'],
   semantic_audit_window_s=period,semantic_published_per_audit_second=semantic['published']/period,
   source_assembly_s=details.get('snapshot_seconds'),request_to_first_command_s=metrics.get('request_to_first_command_seconds'),
   kit_interval_ms=result['kit_interval_ms'],controller_wait_ms=result['controller_wait_ms'],
   **{key:distribution([p['health'].get(key) for p in timing_pubs]) for key in ('worker_seconds','compute_seconds','transport_seconds','queue_seconds','publish_age_seconds','worker_to_publication_seconds')},
   reconstruction_s=distribution([p['diagnostics'].get('reconstruction',{}).get('elapsed_seconds') for p in timing_pubs]),
   mapping_only_compute_s=distribution([p['diagnostics'].get('mapping_compute_seconds') for p in timing_pubs])),
  inference=None,posthoc_observed_union=None,posthoc_reference_scores=[])
 # Compact operation evidence; retain complete curves and diagnostics in hashed raw files.
 row['operations']=[{k:v for k,v in o.items() if k not in ('strategy_diagnostics',)} for o in result['operations']]
 product=autonomous.get('inferred')
 if product is not None:
  components=product['diagnostics'].get('reconstruction',{}).get('components',[])
  row['inference']=dict(identity={k:product[k] for k in ('generation','revision','timestamp')},status=product['status'],reason=product['reason'],
   elapsed_s=product['diagnostics']['elapsed_seconds'],models=product['models'],source_frames=product['source_frames'],
   components=[{k:c[k] for k in ('component','points','robot_envelope_points','status','reason') if k in c} for c in components],
   per_face_evidence={str(c['component']):c['faces'] for c in components if c['component'] in {m['component'] for m in product['models']}},
   qualifications=[{k:q[k] for k in ('component','status','reason','reference_status','alternative_score')} for q in product['diagnostics'].get('qualifications',[])])
  if product['status'] in ('qualified','empty'):
   occupancy=read(folder/'occupancy.json');cameras=read(folder/'camera_metadata.json');packet=occupancy['packet'];snapshot=occupancy['snapshot']
   frozen=NS(generation=snapshot['generation'],revision=snapshot['revision'],timestamp=snapshot['timestamp'],
    view=NS(inferred=InferredGeometry(**product)),spec=NS(voxel_size=snapshot['voxel_size']),
    metadata={'source_frames':cameras,'extraction':packet['diagnostics']},
    boxes=tuple(NS(**{k:b[k] for k in ('id','center','dimensions')},orientation_wxyz=(1.,0.,0.,0.)) for b in packet['boxes']))
   try:
    _,audit,_=combine(frozen);row['posthoc_observed_union']=dict(audit,live_source_revalidated=False)
   except ValueError as error:row['posthoc_observed_union']=dict(assembly_error=str(error),live_source_revalidated=False)
   # Predictions were sealed before reading these independent references. No fitting/tuning occurs here.
   physical=[o for o in read(folder/'geometry_reference.json')['physical_obstacles'] if o['vertices'] is not None]
   matched=set()
   for model in product['models']:
    geometry=model['geometry'];obstacle=min(physical,key=lambda o:np.linalg.norm(np.mean(o['vertices'],axis=0)-geometry['center']))
    assert obstacle['path'] not in matched;matched.add(obstacle['path'])
    row['posthoc_reference_scores'].append(dict(component=model['component'],reference_path=obstacle['path'],
      **compare(geometry,np.asarray(obstacle['vertices'])),free_evidence=evidence_audit(geometry,folder)))
 rows.append(row)
summary=dict(schema_version=1,status='BLOCKED_BY_RELIABILITY',starting_v5='71c07f815410a308dc7ea176148c017055c7fd7a',
 raw_root=str(ROOT),geometry_predictions_seal=seal,qualification='Posthoc archived assembly/scoring is not fresh live-source acceptance. Sampled physical evaluation is not continuous collision certification.',
 scenes=rows)
(ROOT/'numeric-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('reduced',len(rows),'native cases')
