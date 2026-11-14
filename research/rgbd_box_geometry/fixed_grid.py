"""ONE experimental obstacle-specific coverage gate; no fitting or GT reader.

Reference geometry, robot checks, per-face bounds and free-ray checks are reused
unchanged. Coverage requires current pixels on three orthogonal measured planes.
No top-area percentage is lowered or resampled. Single yaw-box assumption only.
"""
from copy import deepcopy
from itertools import product

import numpy as np

from . import reference as ref

POLICY = dict(
    name='calibrated_three_axis_face_coverage',
    model='single convex yaw box; no support-height prior; empirical uncertainty',
    fit_point_max=1200, cloud_point_max=12000, hypothesis_max=4,
    numerical_fits_added=0, required_direct_plane_normal_axes=['x','y','z'],
    required_unique_occupied_pixels_per_face_view=6,
    sample_grid_fractions=[-.25,0.,.25], required_rows=3,required_columns=3,
    depth_tolerance_m=ref.DEPTH,residual_tolerance_m=ref.TOL,
    residual_score_min=.65,
    unchanged_checks=['robot component exclusion','all six measured extent bounds',
                      'current calibrated free-ray contradiction rejection',
                      'finite box hypothesis and 1200-point fit'],
    no_mapping_ray_integration=True, no_gt_geometry=True,
    stop='First fixed alternative only; any baseline/control failure stops development',
)


def face_pixels(model, face, axis, sign, frame):
    """Nine existing image pixels, with six unique witnesses spanning all bins."""
    r, center, size = (np.array(model[k]) for k in ('rotation','center','dimensions'))
    other = [a for a in range(3) if a != axis]
    plane = face['direct']['plane_local_m']
    out = dict(camera_id=frame.camera_id, qualified=False, unique_supported=0,
               rows=0,columns=0,witnesses=[],missing=0,foreground=0,free=0)
    origin = (frame.world_from_camera[:3,3]-center)@r
    if sign*(origin[axis]-plane) <= 0:
        return dict(out,reason='back_facing')
    bins = np.array(list(product(range(3),repeat=2)))
    q = np.zeros((len(bins),3))
    grid=np.array(POLICY['sample_grid_fractions'])
    q[:,axis]=plane
    q[:,other]=grid[bins]*size[other]
    world=center+q@r.T
    camera=(world-frame.world_from_camera[:3,3])@frame.world_from_camera[:3,:3]
    k=frame.intrinsics
    if np.any(camera[:,2] <= 0):
        return dict(out,reason='behind_camera')
    projected=camera[:,:2]/camera[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]-.5
    pixels=np.rint(projected).astype(int)
    h,w=frame.depth.shape
    inside=((pixels>=0)&(pixels<[w,h])).all(1)
    selected=np.flatnonzero(inside)
    measured=np.full(9,np.nan)
    measured[selected]=frame.depth[pixels[selected,1],pixels[selected,0]]
    rays=np.c_[(pixels[:,0]+.5-k[0,2])/k[0,0],
               (pixels[:,1]+.5-k[1,2])/k[1,1],np.ones(9)]
    directions=rays@frame.world_from_camera[:3,:3].T@r
    with np.errstate(divide='ignore',invalid='ignore'):
        expected=(plane-origin[axis])/directions[:,axis]
    valid=inside&np.isfinite(measured)&(measured>0)&np.isfinite(expected)&(expected>0)
    occupied=valid&(abs(measured-expected)<=ref.DEPTH)
    supported=np.flatnonzero(occupied)
    unique=len(np.unique(pixels[supported],axis=0))
    rows=len(np.unique(bins[supported,0]))
    columns=len(np.unique(bins[supported,1]))
    out.update(unique_supported=unique,rows=rows,columns=columns,
               missing=int((~valid).sum()),
               foreground=int((valid&(measured<expected-ref.DEPTH)).sum()),
               free=int((valid&(measured>expected+ref.DEPTH)).sum()),
               witnesses=[dict(pixel_xy=pixels[i].tolist(),grid_bin=bins[i].tolist(),
                               depth_residual_m=float(measured[i]-expected[i])) for i in supported])
    good=unique>=6 and rows==3 and columns==3
    return dict(out,qualified=good,reason='distributed_current_plane_pixels' if good else 'insufficient_plane_pixels')


def qualify(reference,frames):
    """Use one reference fit/check result; never refit or silently override failure."""
    result=dict(status='rejected',reason=None,geometry=None,coverage={},
                reference_status=reference['status'],reference_reason=reference['reason'],
                reference_score=None,residual_score=None,alternative_score=0.)
    model=reference.get('hypothesis')
    if reference.get('robot_envelope_points',1) or model is None:
        return dict(result,reason=reference['reason'])
    result['reference_score']=model['score']
    result['residual_score']=float(np.exp(-.5*(model['residual_p80_m']/ref.TOL)**2))
    faces=reference['faces']
    if reference['free_rays']['contradictions']:
        return dict(result,reason='contradictory_current_free_rays')
    if not all(face['qualified'] for face in faces.values()):
        return dict(result,status='unqualified',reason='unqualified_hidden_bound')
    if len(faces)!=6:
        return dict(result,reason='missing_face_record')
    axes=[]
    for axis in range(3):
        trials=[]
        for sign in (-1,1):
            name=f'{"xyz"[axis]}{sign:+d}'
            face=faces[name]
            if face['direct']['supported']:
                trials.extend(dict(face=name,**face_pixels(model,face,axis,sign,f)) for f in frames)
        supported=any(t['qualified'] for t in trials)
        axes.append(supported)
        result['coverage']['xyz'[axis]]=dict(qualified=supported,views=trials)
    result['alternative_score']=result['residual_score']*int(all(axes))
    if result['alternative_score'] < .65:
        return dict(result,reason='insufficient_three_axis_pixel_coverage_or_residual')
    bounds=np.array([[faces[f'{a}-1']['outer_local_m'],faces[f'{a}+1']['outer_local_m']] for a in 'xyz'])
    rotation=np.array(model['rotation'])
    geometry=dict(center=np.array(model['center'])+bounds.mean(1)@rotation.T,
                  dimensions=np.diff(bounds,axis=1).ravel(),rotation=rotation,
                  yaw_rad=model['yaw_rad'],approximation='rgbd_inferred_box',occupancy_completion=False)
    return dict(result,status='qualified',reason='three_orthogonal_current_face_grids_and_six_bounds',
                geometry=geometry)
