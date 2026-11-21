"""Measured-box numerical reference, extracted unchanged from V5 research.

Only the module documentation differs; the original research copy stays frozen.
"""
from itertools import product

import numpy as np

from . import reference as ref

POLICY = dict(
    name='visibility_aware_observed_plane_patch',
    assumption='One static convex yaw box; empirical depth/plane bands, no shape guarantee',
    grid_fractions=[-.25,0.,.25], required_unique_supported_pixels=6,
    minimum_supported_rows=2, minimum_supported_columns=2,
    direct_plane_support='Unchanged reference minimum/span/rank/residual checks',
    axis_families=['x','y','z'], required_bounded_faces=6,
    residual_tolerance_m=ref.TOL, residual_score_min=.65, depth_band_m=ref.DEPTH,
    unobserved=['outside_fov','behind_camera','occluded','invalid_depth','back_facing'],
    contradiction='Any current visible plane-probe free contradiction rejects model',
    all_observable_bins_must_have_support=True,
    original_bounds_robot_free_checks='unchanged',added_fits=0,
    fit_point_max=1200,cloud_point_max=12000,hypothesis_max=4,
    mapping_rays_added=0, new_workers=0,
)


def classify(inside: bool, expected: float, measured: float) -> str:
    """Classify a projected sample as supported, free, occluded, or unobserved."""
    if not inside:
        return 'outside_fov'
    if not np.isfinite(expected) or expected<=0:
        return 'behind_camera'
    if not np.isfinite(measured) or measured<=0:
        return 'invalid_depth'
    if measured < expected-ref.DEPTH:
        return 'occluded'
    if measured > expected+ref.DEPTH:
        return 'contradictory_free'
    return 'supported'


def patch_decision(samples: list[dict]) -> dict:
    """Decide whether a face has enough distributed, noncontradictory pixels."""
    supported=[s for s in samples if s['state']=='supported']
    observable=[s for s in samples if s['state'] in ('supported','contradictory_free')]
    unique=len({tuple(s['pixel_xy']) for s in supported})
    rows={s['grid_bin'][0] for s in supported}
    columns={s['grid_bin'][1] for s in supported}
    observable_rows={s['grid_bin'][0] for s in observable}
    observable_columns={s['grid_bin'][1] for s in observable}
    contradictions=sum(s['state']=='contradictory_free' for s in samples)
    passed=(not contradictions and unique>=6 and len(rows)>=2 and len(columns)>=2
            and rows==observable_rows and columns==observable_columns)
    return dict(qualified=passed,unique_supported=unique,rows=len(rows),columns=len(columns),
                contradictions=contradictions,
                state_counts={name:sum(s['state']==name for s in samples)
                              for name in ('supported','contradictory_free',*POLICY['unobserved'])})


def face_pixels(model: dict, face: dict, axis: int, sign: int, frame) -> dict:
    """Project one candidate face into a camera and test its visible samples."""
    r,center,size=(np.asarray(model[k]) for k in ('rotation','center','dimensions'))
    plane=face['direct']['plane_local_m']
    origin=(frame.world_from_camera[:3,3]-center)@r
    if sign*(origin[axis]-plane)<=0:
        return dict(camera_id=frame.camera_id,qualified=False,unique_supported=0,
                    rows=0,columns=0,contradictions=0,samples=[],
                    state_counts={'back_facing':9},reason='back_facing')
    other=[a for a in range(3) if a!=axis]
    bins=np.array(list(product(range(3),repeat=2)))
    local=np.zeros((9,3));local[:,axis]=plane
    local[:,other]=np.array(POLICY['grid_fractions'])[bins]*size[other]
    world=center+local@r.T
    camera=(world-frame.world_from_camera[:3,3])@frame.world_from_camera[:3,:3]
    k=frame.intrinsics
    with np.errstate(divide='ignore',invalid='ignore'):
        projected=camera[:,:2]/camera[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]-.5
    finite=np.isfinite(projected).all(1)
    pixels=np.zeros((9,2),dtype=int)
    pixels[finite]=np.rint(projected[finite]).astype(int)
    h,w=frame.depth.shape
    inside=finite&((pixels>=0)&(pixels<[w,h])).all(1)&(camera[:,2]>0)
    rays=np.c_[(pixels[:,0]+.5-k[0,2])/k[0,0],
               (pixels[:,1]+.5-k[1,2])/k[1,1],np.ones(9)]
    directions=rays@frame.world_from_camera[:3,:3].T@r
    with np.errstate(divide='ignore',invalid='ignore'):
        expected=(plane-origin[axis])/directions[:,axis]
    samples=[]
    for i in range(9):
        depth=float(frame.depth[pixels[i,1],pixels[i,0]]) if inside[i] else np.nan
        state=classify(bool(inside[i]),float(expected[i]),depth)
        if camera[i,2]<=0:state='behind_camera'
        samples.append(dict(grid_bin=bins[i].tolist(),pixel_xy=pixels[i].tolist(),state=state,
                            depth_residual_m=float(depth-expected[i]) if np.isfinite(depth) and np.isfinite(expected[i]) else None))
    decision=patch_decision(samples)
    return dict(camera_id=frame.camera_id,samples=samples,**decision,
                reason='observed_spanning_patch' if decision['qualified'] else 'insufficient_or_contradictory_observed_patch')


def qualify(reference: dict, frames) -> dict:
    """Require measured face patches before accepting inferred box geometry."""
    result=dict(status='rejected',reason=None,geometry=None,coverage={},
                reference_status=reference['status'],reference_score=None,residual_score=None,
                alternative_score=0.,assumption=POLICY['assumption'])
    model=reference.get('hypothesis')
    if reference.get('robot_envelope_points',1) or model is None:
        return dict(result,reason=reference['reason'])
    faces=reference.get('faces',{})
    result['reference_score']=model['score']
    result['residual_score']=float(np.exp(-.5*(model['residual_p80_m']/ref.TOL)**2))
    if reference['free_rays']['contradictions']:
        return dict(result,reason='contradictory_current_free_rays')
    if len(faces)!=6 or not all(f['qualified'] for f in faces.values()):
        return dict(result,status='unqualified',reason='unqualified_hidden_bound')
    axes=[];contradictions=0
    for axis in range(3):
        trials=[]
        for sign in (-1,1):
            name=f'{"xyz"[axis]}{sign:+d}'
            if faces[name]['direct']['supported']:
                trials.extend(dict(face=name,**face_pixels(model,faces[name],axis,sign,f)) for f in frames)
        passed=any(t['qualified'] for t in trials)
        axes.append(passed);contradictions+=sum(t['contradictions'] for t in trials)
        result['coverage']['xyz'[axis]]=dict(qualified=passed,views=trials)
    if contradictions:
        return dict(result,reason='contradictory_visible_plane_pixels')
    result['alternative_score']=result['residual_score']*int(all(axes))
    if result['alternative_score']<.65:
        return dict(result,reason='insufficient_observed_three_axis_patch_or_residual')
    bounds=np.array([[faces[f'{a}-1']['outer_local_m'],faces[f'{a}+1']['outer_local_m']] for a in 'xyz'])
    rotation=np.asarray(model['rotation'])
    geometry=dict(center=np.asarray(model['center'])+bounds.mean(1)@rotation.T,
                  dimensions=np.diff(bounds,axis=1).ravel(),rotation=rotation,
                  yaw_rad=model['yaw_rad'],approximation='rgbd_inferred_box',occupancy_completion=False)
    return dict(result,status='qualified',reason='observed_three_axis_patches_and_six_bounds',geometry=geometry)
