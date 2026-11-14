"""Independent scoring only; verify blind predictions before reading GT."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from . import reference as ref
from .reference import CORNERS,DEPTH
from .inputs import CASES,expected_inputs
from .run import verify_seal


def compare(model, vertices):
    center = np.asarray(model['center'])
    rotation = np.asarray(model['rotation'])
    half = np.asarray(model['dimensions'])/2
    q = (vertices-center)@rotation
    predicted = np.c_[-half, half]
    truth = np.c_[q.min(0), q.max(0)]
    errors = predicted-truth
    slack = float((half-abs(q)).min())
    return dict(maximum_extent_error_m=float(abs(errors).max()),
                signed_face_errors_m=errors.tolist(),
                minimum_containment_slack_m=slack,
                contains_reference_without_allowance=slack >= 0,
                contains_reference_with_existing_10um_allowance=slack >= -1e-5)


def evidence_audit(model, folder):
    meta = json.loads((folder/'occupancy.json').read_text())
    packet = meta['packet']['boxes']
    spec = meta['snapshot']
    center, r, half = (np.asarray(model[k]) for k in ('center','rotation','dimensions'))
    half = half/2
    covered, protected = [], []
    for box in packet:
        points = np.asarray(box['center'])+CORNERS*np.asarray(box['dimensions'])
        floor = points[:, 2].min() <= .004
        if floor:
            protected.append(box['id'])
        if not floor and ((abs((points-center)@r)+1e-12) <= half).all():
            covered.append(box['id'])
    with np.load(folder/'occupancy.npz', allow_pickle=False) as archive:
        indices = np.argwhere(archive['states'] == 1)
    voxel = spec['voxel_size']
    points = np.asarray(spec['bounds_min'])+(indices+.5)*voxel
    q = abs((points-center)@r)
    raw_free_centers = int((q < half).all(1).sum())
    # Strong contradictions only: entire FREE grid cell lies in strict interior.
    projected_half_cell = (voxel/2)*abs(r).sum(0)
    full_free_cells = int((q+projected_half_cell < half-DEPTH).all(1).sum())
    total = len(packet)-len(covered)+1
    return dict(original_selected_boxes=len(packet), selected_floor_boxes_preserved=len(protected),
                fully_contained_nonfloor_boxes=covered,
                total_proxies_preserving_all_selected_boxes=total,
                fits_64_with_whole_cuboid_preservation=total <= 64,
                raw_FREE_cell_centers_in_model=raw_free_centers,
                complete_FREE_cells_strictly_in_model=full_free_cells,
                grid_alias_qualification='center hits near boundaries are not exact free-solid contradictions',
                archive_source_usable=meta['source_usable'],
                current_publication_freshness_not_revalidated=True)


def score(captures,output):
    captures,output=Path(captures),Path(output)
    if (output/'scoring.json').exists():raise ValueError('Refusing to overwrite scores')
    verify_seal(output)
    inputs=expected_inputs()
    for relative,expected in json.loads((output/'input-manifest.json').read_text()).items():
        if ref.digest(captures/relative)!=expected:raise ValueError('Archived input changed: '+relative)
    results=dict(scoring_started_utc=datetime.now(timezone.utc).isoformat(),
                 blind_sha256=ref.digest(output/'blind-results.json'),cases={},failures=[])
    predictions=json.loads((output/'blind-results.json').read_text())
    for case,record in predictions.items():
        folder=captures/CASES[case]
        relative=f'{CASES[case]}/geometry_reference.json'
        if ref.digest(folder/'geometry_reference.json')!=inputs[relative]:
            raise ValueError('Reference hash mismatch: '+relative)
        truth=json.loads((folder/'geometry_reference.json').read_text())
        obstacles=[o for o in truth['physical_obstacles'] if o['kind']=='convex']
        candidates=[c for c in record['reference']['components'] if c.get('hypothesis') is not None]
        if len(obstacles)!=1 or len(candidates)!=1:raise ValueError('Ambiguous scoring association')
        original=candidates[0]
        model=next(c for c in record['models'] if c['component']==original['component'])
        vertices=np.asarray(obstacles[0]['vertices'])
        true_volume=float(np.prod(np.ptp(vertices,axis=0)))
        row=dict(status=model['status'],reference_path_scoring_only=obstacles[0]['path'],
                 reference_geometry_sha256=inputs[relative],
                 raw_surface_hypothesis=compare(original['hypothesis'],vertices),
                 original_free_rays=original['free_rays'],
                 robot_envelope_points=original['robot_envelope_points'])
        if model['geometry'] is None:
            results['failures'].append(case+': no qualified box')
        else:
            geometry=model['geometry']
            row['geometry']=geometry
            row['containment']=compare(geometry,vertices)
            row['model_volume_m3']=float(np.prod(geometry['dimensions']))
            row['reference_volume_m3']=true_volume
            row['excess_volume_m3']=row['model_volume_m3']-true_volume
            row['relative_excess_volume']=row['model_volume_m3']/true_volume-1
            row['evidence']=evidence_audit(geometry,folder)
            if not row['containment']['contains_reference_with_existing_10um_allowance']:
                results['failures'].append(case+': physical containment failed')
            if not row['evidence']['fits_64_with_whole_cuboid_preservation']:
                results['failures'].append(case+': 64-proxy limit failed')
            if row['evidence']['complete_FREE_cells_strictly_in_model']:
                results['failures'].append(case+': complete FREE voxel inside strict model interior')
        results['cases'][case]=row
    status=json.loads((output/'status.json').read_text())
    results['offline_gate_pass']=status['all_declared_tests_pass'] and not results['failures'] and len(results['cases'])==2
    verify_seal(output)
    ref.save(output/'scoring.json',results)
    print(json.dumps(results,indent=2))
    return results['offline_gate_pass']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    return 0 if score(args.captures,args.output) else 2


if __name__=='__main__':raise SystemExit(main())
