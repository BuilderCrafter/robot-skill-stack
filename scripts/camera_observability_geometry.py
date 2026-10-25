#!/usr/bin/env python3
"""Offline camera positioning research using calibrated rays and copied meshes.

Geometry is evaluation-only. Nothing here supplies a depth frame, occupancy
measurement or obstacle packet to the production runtime. Mesh occlusion uses
the exact existing immutable triangle BVH; analytic fixtures have exact ray
intersections and area-weighted surface quadrature. This is a sampled geometry
screen, not a substitute for native rendered-depth validation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import _mesh_depth, MeshInstance, TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'docs/assets/multiview_final/selection.json'
CURRENT = ROOT/'docs/assets/multiview_final/selected-layout.json'
CAMERAS = ('/World/PerceptionCamera', '/World/PerceptionCameraSide', '/World/CameraObservabilityThird')
BOUNDS = np.array([[-.3, -.6, -.02], [.85, .6, 1.12]])
CENTER = np.array([.275, 0., .55])
DEFAULT_AIM = np.array([.35, 0., .35])
BODY_GROUPS = ('gripper', 'fingertips', 'fingertips', 'base', 'shoulder', 'shoulder',
               'elbow', 'elbow', 'forearm', 'forearm', 'wrist')


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')


def look_at_ros(eye, target):
    """Optical ROS columns: right, down, forward; preserve explicit authored roll."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target-eye
    length = np.linalg.norm(forward)
    if length < 1e-9:
        raise ValueError('Camera eye must differ from aim')
    forward /= length
    up = np.array([0., 0., 1.])
    if abs(forward@up) > .999:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = eye
    return transform


def quaternion_wxyz(rotation):
    """Stable rotation-to-quaternion, returning the native wxyz convention."""
    r = np.asarray(rotation, float)
    candidates = np.array([1+r[0,0]+r[1,1]+r[2,2], 1+r[0,0]-r[1,1]-r[2,2],
                           1-r[0,0]+r[1,1]-r[2,2], 1-r[0,0]-r[1,1]+r[2,2]])
    index = int(np.argmax(candidates))
    q = np.empty(4)
    q[index] = math.sqrt(max(0., candidates[index]))/2
    denominator = 4*q[index]
    if index == 0:
        q[1:] = [(r[2,1]-r[1,2])/denominator, (r[0,2]-r[2,0])/denominator,
                 (r[1,0]-r[0,1])/denominator]
    elif index == 1:
        q[[0,2,3]] = [(r[2,1]-r[1,2])/denominator, (r[0,1]+r[1,0])/denominator,
                      (r[0,2]+r[2,0])/denominator]
    elif index == 2:
        q[[0,1,3]] = [(r[0,2]-r[2,0])/denominator, (r[0,1]+r[1,0])/denominator,
                      (r[1,2]+r[2,1])/denominator]
    else:
        q[[0,1,2]] = [(r[1,0]-r[0,1])/denominator, (r[0,2]+r[2,0])/denominator,
                      (r[1,2]+r[2,1])/denominator]
    return q if q[0] >= 0 else -q


def camera_record(index, eye=None, aim=None, transform=None, resolution=(640, 480), role='fixed'):
    history = json.loads(HISTORY.read_text())
    intrinsic = np.asarray(history['calibration']['native_intrinsics_flat_per_camera'][min(index, 1)]).reshape(3, 3)
    intrinsic = intrinsic.copy()
    intrinsic[0] *= resolution[0]/640
    intrinsic[1] *= resolution[1]/480
    matrix = np.asarray(transform, float) if transform is not None else look_at_ros(eye, aim)
    eye = matrix[:3, 3]
    forward = matrix[:3, 2]
    if aim is None:
        aim = eye+forward
    return dict(camera_id=CAMERAS[min(index, 2)], position_world_m=eye,
        aim_world_m=np.asarray(aim, float), world_from_camera_ros=matrix,
        orientation_ros_wxyz=quaternion_wxyz(matrix[:3, :3]),
        resolution=resolution, intrinsics=intrinsic, role=role,
        height_m=float(eye[2]), horizontal_radius_to_workspace_m=float(np.linalg.norm((eye-CENTER)[:2])),
        distance_to_mapping_center_m=float(np.linalg.norm(eye-CENTER)),
        downward_pitch_deg=float(np.degrees(np.arctan2(-forward[2], np.linalg.norm(forward[:2])))),
        azimuth_deg=float(np.degrees(np.arctan2(forward[1], forward[0]))),
        horizontal_fov_deg=float(np.degrees(2*np.arctan(resolution[0]/(2*intrinsic[0,0])))),
        vertical_fov_deg=float(np.degrees(2*np.arctan(resolution[1]/(2*intrinsic[1,1])))),
        calibration_scope='Recorded native optics inherited unchanged; reverify actual K and pose in shortlisted native captures.')


def catalog():
    """One-factor variations, historical seeds, and explicit fixed/wrist families."""
    current = json.loads(CURRENT.read_text())
    rows = []

    def add(identifier, family, eyes=None, aims=None, transforms=None, variable='', value=None, role='fixed'):
        n = len(transforms) if transforms is not None else len(eyes)
        cameras = []
        for i in range(n):
            transform = None if transforms is None else transforms[i]
            cameras.append(camera_record(i, None if eyes is None else eyes[i],
                None if aims is None else aims[i], transform,
                (320, 240) if i == 2 else (640, 480), role if i == 2 else 'fixed'))
        rows.append(dict(layout_id=identifier, family=family, variant=dict(variable=variable, value=value),
            cameras=cameras, baseline_separation_m=float(np.linalg.norm(
                np.asarray(cameras[0]['position_world_m'])-cameras[1]['position_world_m'])),
            physical_mount_scope='Optical points only; no permanent sensor housing/mount/cable model is authored.'))

    transforms = current['world_from_cameras']
    add('A_current', 'A', aims=current['targets'], transforms=transforms,
        variable='authored', value='exact validated current transforms including roll')
    # Same common elevated aim, preserving all but the explicitly named factor.
    for height in (.55, .85, 1.15):
        add(f'B_front_height_{height:.2f}', 'B', [(1.05,.70,height),(1.05,-.70,height)],
            [DEFAULT_AIM]*2, variable='height_m', value=height)
    for high, low in ((.95,.55),(1.15,.65),(1.35,.75)):
        add(f'C_asymmetric_{high:.2f}_{low:.2f}', 'C', [(1.05,.70,high),(.18,-.80,low)],
            [DEFAULT_AIM]*2, variable='high_low_height_m', value=[high,low])
    for lateral in (.50,.75,1.):
        add(f'D_baseline_{2*lateral:.2f}', 'D', [(.95,lateral,.85),(.95,-lateral,.85)],
            [DEFAULT_AIM]*2, variable='lateral_baseline_m', value=2*lateral)
    for distance in (.65,.85,1.05):
        add(f'E_distance_{distance:.2f}', 'E', [(distance,.55,.75),(distance,-.55,.75)],
            [DEFAULT_AIM]*2, variable='front_x_m', value=distance)
    for height in (1.05,1.30,1.55):
        add(f'F_high_{height:.2f}', 'F', [(.95,.65,height),(.95,-.65,height)],
            [DEFAULT_AIM]*2, variable='height_m', value=height)
    for target_z in (.10,.35,.60):
        add(f'B_front_aim_{target_z:.2f}', 'B', [(1.05,.70,.85),(1.05,-.70,.85)],
            [np.array([.35,0.,target_z])]*2, variable='aim_z_m', value=target_z)
    for side in (-.25,0.,.25):
        add(f'G_overhead_offset_{side:+.2f}', 'G', [(.35,side,1.35),(.90,-.75,.70)],
            [np.array([.35,0.,.25]),DEFAULT_AIM], variable='overhead_y_m', value=side)
    for y in (-.70,-1.,1.):
        add(f'H_third_side_{y:+.2f}', 'H',
            transforms=[*transforms, look_at_ros((-.25,y,.85),DEFAULT_AIM)],
            aims=[*current['targets'],DEFAULT_AIM], variable='third_side_y_m', value=y)
    for height in (1.,1.25,1.50):
        add(f'I_third_overhead_{height:.2f}', 'I',
            transforms=[*transforms, look_at_ros((.35,0.,height),(.35,0.,.20))],
            aims=[*current['targets'],(.35,0.,.20)], variable='third_height_m', value=height)
    for offset in (-.05,.05):
        add(f'J_wrist_offset_{offset:+.2f}', 'J',
            transforms=[*transforms, look_at_ros((.45,offset,.60),(.45,offset,.20))],
            aims=[*current['targets'],(.45,offset,.20)], variable='hand_lateral_offset_m', value=offset,
            role='hypothetical_wrist')
        rows[-1]['cameras'][-1]['hand_rigid_from_camera_ros']=np.array([
            [1.,0.,0.,offset],[0.,1.,0.,0.],[0.,0.,1.,-.03],[0.,0.,0.,1.]])
        rows[-1]['cameras'][-1]['position_scope']='Catalog pose is a nominal illustration; actual world optical pose is kinematically recomputed from each captured hand pose.'
    for base_name in ('B_front_height_0.85','C_asymmetric_1.15_0.65','G_overhead_offset_+0.00'):
        base=next(row for row in rows if row['layout_id']==base_name)
        for label,eye in [('rear',(-.25,-.90,.85)),('front_side',(.90,.90,.75)),('overhead',(.35,0.,1.25))]:
            add('K_'+base_name+'_'+label,'K',
                transforms=[*[camera['world_from_camera_ros'] for camera in base['cameras']],look_at_ros(eye,DEFAULT_AIM)],
                aims=[*[camera['aim_world_m'] for camera in base['cameras']],DEFAULT_AIM],
                variable='optimized_pair_plus_third',value=dict(base_layout=base_name,third=label))
    # Historical tabletop-aim variants explicitly extend the old semantic study
    # with robot-mesh occlusion and a full mapping-volume target set.
    history = json.loads(HISTORY.read_text())
    for old in history['layouts']:
        if old['name'] in ('front_symmetric','asymmetric_high_left','front_wider'):
            add('historical_'+old['name'], 'historical', old['eyes'], old['targets'],
                variable='historical_semantic_seed', value=old['name'])
    return rows


def _rotation(shape):
    yaw = float(shape.get('yaw', 0.))
    return np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])


def shape_surface(shape, density=5):
    """Deterministic quadrature with true area weights, including hidden backs."""
    kind, center = shape['kind'], np.asarray(shape['center'], float)
    points, normals, weights = [], [], []
    if kind == 'box':
        dimensions = np.asarray(shape['dimensions'], float)
        grid = (np.arange(density)+.5)/density-.5
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for sign in (-1,1):
                for u in grid:
                    for v in grid:
                        p = np.zeros(3);p[axis]=sign*dimensions[axis]/2
                        p[others] = np.array([u,v])*dimensions[others]
                        n=np.zeros(3);n[axis]=sign
                        points.append(p);normals.append(n)
                        weights.append(np.prod(dimensions[others])/density**2)
    elif kind == 'sphere':
        n=max(48, density*density*6)
        index=np.arange(n)
        z=1-2*(index+.5)/n
        theta=index*np.pi*(3-np.sqrt(5))
        normal=np.c_[np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z]
        points=list(normal*shape['radius']);normals=list(normal)
        weights=[4*np.pi*shape['radius']**2/n]*n
    elif kind == 'cylinder':
        radius,height=shape['radius'],shape['height']
        sectors=max(16,density*4)
        for z in (np.arange(density)+.5)/density*height-height/2:
            for theta in (np.arange(sectors)+.5)/sectors*2*np.pi:
                normal=np.array([np.cos(theta),np.sin(theta),0.])
                points.append(normal*radius+[0.,0.,z]);normals.append(normal)
                weights.append(2*np.pi*radius*height/(density*sectors))
        for sign in (-1,1):
            for j in range(density):
                r=radius*np.sqrt((j+.5)/density)
                for theta in (np.arange(sectors)+.5)/sectors*2*np.pi:
                    points.append([r*np.cos(theta),r*np.sin(theta),sign*height/2])
                    normals.append([0.,0.,sign]);weights.append(np.pi*radius**2/(density*sectors))
    else:
        raise ValueError('Unsupported independent fixture '+str(kind))
    rotation=_rotation(shape)
    return np.asarray(points)@rotation.T+center, np.asarray(normals)@rotation.T, np.asarray(weights)


def shape_depth(shape, origin, directions):
    """First positive ray parameter against analytic evaluation shapes."""
    r=_rotation(shape); origin=(np.asarray(origin)-shape['center'])@r
    rays=np.asarray(directions)@r
    if shape['kind']=='box':
        half=np.asarray(shape['dimensions'])/2
        first,second=np.full_like(rays,-np.inf),np.full_like(rays,np.inf)
        moving=abs(rays)>1e-14
        np.divide(-half-origin,rays,out=first,where=moving)
        np.divide(half-origin,rays,out=second,where=moving)
        near=np.minimum(first,second).max(1);far=np.maximum(first,second).min(1)
        outside=(~moving&((origin < -half)|(origin > half))).any(1)
        return np.where(~outside&(far>=np.maximum(near,0)),np.where(near>0,near,far),np.inf)
    if shape['kind']=='sphere':
        a=np.sum(rays*rays,1);b=2*rays@origin;c=origin@origin-shape['radius']**2
        disc=b*b-4*a*c
        root=np.sqrt(np.maximum(0,disc))
        first=(-b-root)/(2*a);second=(-b+root)/(2*a)
        return np.where(disc>=0,np.where(first>0,first,np.where(second>0,second,np.inf)),np.inf)
    if shape['kind']=='cylinder':
        radius,half=shape['radius'],shape['height']/2
        a=np.sum(rays[:,:2]**2,1);b=2*rays[:,:2]@origin[:2];c=origin[:2]@origin[:2]-radius**2
        disc=b*b-4*a*c;root=np.sqrt(np.maximum(0,disc))
        best=np.full(len(rays),np.inf)
        for sign in (-1,1):
            t=np.divide(-b+sign*root,2*a,out=np.full(len(rays),np.inf),where=a>1e-20)
            z=origin[2]+t*rays[:,2]
            best=np.minimum(best,np.where((disc>=0)&(t>0)&(abs(z)<=half+1e-10),t,np.inf))
            t=np.divide(sign*half-origin[2],rays[:,2],out=np.full(len(rays),np.inf),where=abs(rays[:,2])>1e-20)
            xy=origin[:2]+t[:,None]*rays[:,:2]
            best=np.minimum(best,np.where((t>0)&(np.sum(xy*xy,1)<=radius*radius+1e-12),t,np.inf))
        return best
    raise ValueError('Unsupported shape')


def projected(camera, points):
    transform=np.asarray(camera['world_from_camera_ros'])
    xyz=(np.asarray(points)-transform[:3,3])@transform[:3,:3]
    k=np.asarray(camera['intrinsics']);z=xyz[:,2]
    uv=np.full((len(xyz),2),np.nan)
    good=z>1e-8
    uv[good]=xyz[good,:2]/z[good,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    width,height=camera['resolution']
    valid=good&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    return xyz,uv,valid


def visibility(camera, points, parts=(), occluders=(), normals=None):
    """Visibility at finite surface/probe samples, with exact copied mesh LOS."""
    points=np.asarray(points,float)
    _,uv,valid=projected(camera,points)
    eye=np.asarray(camera['world_from_camera_ros'])[:3,3]
    if normals is not None:
        valid &= np.sum(np.asarray(normals)*(eye-points),axis=1)>1e-10
    ids=np.flatnonzero(valid)
    if not len(ids):return valid,uv
    directions=points[ids]-eye
    endpoint_margin=np.maximum(1e-6,.0005/np.linalg.norm(directions,axis=1))
    limit=1-endpoint_margin
    blocked=np.zeros(len(ids),bool)
    for part in parts:
        inverse=part._local_from_world
        origin=eye@inverse[:3,:3].T+inverse[:3,3]
        local=directions@inverse[:3,:3].T
        depth=_mesh_depth(part.mesh,origin,local,limit)
        blocked |= depth < limit-1e-10
    for shape in occluders:
        distance=shape_depth(shape,eye,directions)
        blocked |= (distance>0)&(distance<limit)
    valid[ids[blocked]]=False
    return valid,uv


def fixtures():
    """Evaluation fixtures span table, arm-height hazards, gaps and occlusion."""
    return [
        dict(scenario_id='empty',objects=[]),
        dict(scenario_id='small_cube_sphere',objects=[
            dict(id='cube',kind='box',center=[.43,-.18,.025],dimensions=[.05,.05,.05]),
            dict(id='sphere',kind='sphere',center=[.64,.24,.03],radius=.03)]),
        dict(scenario_id='thin_post',objects=[dict(id='post',kind='cylinder',center=[.45,.03,.17],radius=.012,height=.34)]),
        dict(scenario_id='large_obstruction',objects=[dict(id='large',kind='box',center=[.50,.05,.25],dimensions=[.18,.24,.50])]),
        dict(scenario_id='narrow_gap',objects=[
            dict(id='gap_left',kind='box',center=[.48,.105,.18],dimensions=[.08,.10,.36]),
            dict(id='gap_right',kind='box',center=[.48,-.105,.18],dimensions=[.08,.10,.36])]),
        dict(scenario_id='object_occlusion',objects=[
            dict(id='hidden_target',kind='box',center=[.40,.08,.10],dimensions=[.06,.06,.20]),
            dict(id='occluder',kind='box',center=[.60,.08,.15],dimensions=[.12,.16,.30])]),
        dict(scenario_id='arm_occlusion',objects=[
            dict(id='rear_of_arm',kind='box',center=[-.12,.12,.55],dimensions=[.06,.06,.12])]),
        dict(scenario_id='entering_path',objects=[
            dict(id='moving_post',kind='cylinder',center=[.46,.12,.20],radius=.018,height=.40)]),
        dict(scenario_id='upper_arm_hazards',objects=[
            dict(id='shoulder_neighbor',kind='sphere',center=[-.10,-.15,.38],radius=.035),
            dict(id='elbow_neighbor',kind='box',center=[.02,.16,.72],dimensions=[.06,.06,.08]),
            dict(id='forearm_neighbor',kind='cylinder',center=[.28,-.16,.53],radius=.018,height=.16)])]


def background_shapes():
    result=[]
    for item in json.loads(CURRENT.read_text())['objects']:
        result.append(dict(id='background_'+item['name'],kind='box' if item['shape']=='cube' else item['shape'],
            center=item['center'],dimensions=item['size'],radius=item['radius'],yaw=item.get('yaw',0.)))
    return result


@dataclass
class GeometryPose:
    pose_id: str
    parts: tuple
    source: str
    part_names: tuple = ()


def load_poses(manifest=None):
    """Load trusted local capture inputs; never deserialize external pickles."""
    if manifest is not None:
        manifest=Path(manifest)
        document=json.loads(manifest.read_text())
        rows=[]
        for item in document['snapshots']:
            if item.get('mapping_input_pickle'):
                path=Path(item['mapping_input_pickle'])
                if not path.is_absolute():path=manifest.parent/path
                payload=pickle.loads(path.read_bytes())
                snapshot=next(iter(payload.robot_snapshots.values()))
                parts=snapshot.parts
            else:
                path=Path(item['geometry_npz'])
                if not path.is_absolute():path=manifest.parent/path
                with np.load(path) as values:
                    count=int(values['part_count']) if 'part_count' in values else len(item['part_names'])
                    parts=tuple(MeshInstance(TriangleMesh(values[f'vertices_{i}'],values[f'triangles_{i}']),
                        values[f'world_from_local_{i}']) for i in range(count))
            rows.append(GeometryPose(item.get('pose_id',item.get('id',item.get('name'))),parts,str(path),tuple(item.get('part_names',()))))
        return rows
    folder=ROOT/'outputs/motion-planning-v2b-baseline-rgbd-static-direct-capture/mapping-audit'
    paths=sorted(folder.glob('*-input.pkl'))
    if len(paths)<3:raise FileNotFoundError('Need existing V2b trusted robot snapshots or --geometry-manifest')
    indices=[0,len(paths)//2,len(paths)-1]
    rows=[]
    for index in indices:
        path=paths[index];payload=pickle.loads(path.read_bytes())
        snapshot=next(iter(payload.robot_snapshots.values()))
        rows.append(GeometryPose(f'recorded_motion_{index:02d}',snapshot.parts,str(path.relative_to(ROOT))))
    return rows


def wrist_camera(layout, pose):
    """Hypothetical hand-fixed camera, evaluated kinematically without attachment."""
    offset=float(layout['variant']['value'])
    hand=pose.parts[0].world_from_local
    # Mesh transforms can include authored uniform scale100. Physical mounting
    # offsets are meters in a rigid hand frame, never scaled mesh coordinates.
    u,_,vt=np.linalg.svd(hand[:3,:3]);rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vt
    rigid=np.eye(4);rigid[:3,:3]=rotation;rigid[:3,3]=hand[:3,3]
    local=np.eye(4);local[:3,3]=[offset,0.,-.03]
    optical=rigid@local
    camera=camera_record(2,aim=optical[:3,3]+.35*optical[:3,2],transform=optical,
        resolution=(320,240),role='hypothetical_wrist')
    camera['hand_rigid_from_camera_ros']=local
    return camera


def region_probes(pose):
    """External face probes around actual link AABBs; not a swept-volume proof."""
    points,groups=[],[]
    for index,part in enumerate(pose.parts):
        vertices=part.mesh.vertices@part.world_from_local[:3,:3].T+part.world_from_local[:3,3]
        low,high=vertices.min(0),vertices.max(0)
        center=(low+high)/2
        name=BODY_GROUPS[index] if index<len(BODY_GROUPS) else 'other'
        if pose.part_names:
            path=pose.part_names[index]
            if 'finger' in path:name='fingertips'
            elif 'hand' in path:name='gripper'
            else:
                for number,group in ((0,'base'),(1,'shoulder'),(2,'shoulder'),(3,'elbow'),
                                     (4,'elbow'),(5,'forearm'),(6,'forearm'),(7,'wrist')):
                    if 'panda_link'+str(number) in path:name=group;break
        for axis in range(3):
            for sign in (-1,1):
                point=center.copy();point[axis]=(high[axis] if sign>0 else low[axis])+sign*.04
                if np.all(point>=BOUNDS[0]) and np.all(point<=BOUNDS[1]):
                    points.append(point);groups.append(name)
    # Identical small swept-path neighborhood for every layout, following the
    # legitimate hover/side/return sequence, not all workspace voxels.
    for y in np.linspace(-.04,.12,9):
        for x in (.40,.45,.50):
            for z in (.18,.25,.32):
                points.append([x,y,z]);groups.append('tcp_path')
    return np.asarray(points),np.asarray(groups)


def workspace_grid(spacing=.15):
    axes=[np.arange(a+spacing/2,b,spacing) for a,b in zip(BOUNDS[0],BOUNDS[1])]
    return np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)


def blind_spots(points, counts, spacing=.15):
    """Connected unobserved probe-bin extents, separate from map occupancy."""
    origin=np.min(points,axis=0)
    cells=np.rint((points-origin)/spacing).astype(int)
    lookup={tuple(cell):i for i,cell in enumerate(cells) if counts[i]==0}
    remaining=set(lookup);clusters=[]
    while remaining:
        seed=remaining.pop();stack=[seed];indices=[]
        while stack:
            cell=stack.pop();indices.append(lookup[cell])
            for axis in range(3):
                for sign in (-1,1):
                    neighbor=list(cell);neighbor[axis]+=sign;neighbor=tuple(neighbor)
                    if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
        selected=points[indices]
        clusters.append(dict(probes=len(indices),lower_m=np.maximum(BOUNDS[0],selected.min(0)-spacing/2),
            upper_m=np.minimum(BOUNDS[1],selected.max(0)+spacing/2),nominal_probe_bin_volume_m3=len(indices)*spacing**3))
    return sorted(clusters,key=lambda row:row['probes'],reverse=True)[:6]


def fixture_robot_clearance(pose, shape):
    """Conservative actual-mesh/enclosing-box overlap flag, never a motion proof."""
    from tests.support.reactive_geometry import mesh_box_distance
    if shape['kind']=='box':half=np.asarray(shape['dimensions'])/2
    elif shape['kind']=='sphere':half=np.full(3,shape['radius'])
    else:half=np.array([shape['radius'],shape['radius'],shape['height']/2])
    matrix=np.eye(4);matrix[:3,:3]=_rotation(shape).T;matrix[:3,3]=-matrix[:3,:3]@np.asarray(shape['center'])
    best=np.inf
    for part in pose.parts:
        best=min(best,mesh_box_distance(part.mesh,matrix@part.world_from_local,half,upper_bound=best))
        if best==0:break
    return dict(clearance_to_enclosing_box_m=float(best),potential_overlap=best<=1e-9,
        reference='Exact actual robot visible triangle meshes; exact box fixture or conservative enclosing box for sphere/cylinder.',
        motion_validity='Potential overlap must be excluded/checked before any physical movement claim; visibility trial retained.')


def pareto_rows(summaries, camera_count):
    selected=[row for row in summaries if row['camera_count']==camera_count and row['family']!='J']
    fields=('nonoverlap_object_macro_visible_mean','arm_region_min_mean','relevant_surface_visible_one_min')
    return [row['layout_id'] for row in selected if not any(
        all(other[key]>=row[key]-1e-12 for key in fields) and any(other[key]>row[key]+1e-12 for key in fields)
        for other in selected if other is not row)]


def camera_housing_clearances(geometry_manifest, native_paths, layouts):
    """Evaluate an explicit5cmworld-aligned housing assumption at optical eyes."""
    poses=load_poses(geometry_manifest)
    native={}
    for location in native_paths:
        path=Path(location)
        if path.is_dir():path=path/'capture.json'
        document=json.loads(path.read_text())
        native[document['layout_id']]={item.get('id',item.get('pose_id')):item['cameras'] for item in document['snapshots']}
    selected=[layout for layout in layouts if layout['layout_id'] in native or layout['family']=='J']
    rows=[]
    for layout in selected:
        for pose in poses:
            if layout['family']=='J':
                cameras=[*layout['cameras'][:-1],wrist_camera(layout,pose)]
            else:
                cameras=native[layout['layout_id']][pose.pose_id]
            for camera in cameras:
                eye=np.asarray(camera['world_from_camera_ros'])[:3,3]
                shape=dict(kind='box',center=eye,dimensions=[.05,.05,.05])
                clearance=fixture_robot_clearance(pose,shape)
                rows.append(dict(layout_id=layout['layout_id'],pose_id=pose.pose_id,camera_id=camera['camera_id'],
                    optical_origin_world_m=eye,housing_clearance_m=clearance['clearance_to_enclosing_box_m'],
                    housing_intersects_robot=clearance['potential_overlap'],
                    camera_pose_source='kinematic hand-fixed hypothesis' if layout['family']=='J' and camera['camera_id']==CAMERAS[2] else 'actual native fixed-camera calibration'))
    summaries=[]
    for layout in selected:
        chosen=[row for row in rows if row['layout_id']==layout['layout_id']]
        summaries.append(dict(layout_id=layout['layout_id'],minimum_housing_clearance_m=min(row['housing_clearance_m'] for row in chosen),
            intersecting_camera_pose_samples=sum(row['housing_intersects_robot'] for row in chosen),sampled_camera_poses=len(chosen),
            per_camera={identifier:dict(minimum_housing_clearance_m=min(row['housing_clearance_m'] for row in chosen if row['camera_id']==identifier),
                intersecting_postures=[row['pose_id'] for row in chosen if row['camera_id']==identifier and row['housing_intersects_robot']])
                for identifier in sorted({row['camera_id'] for row in chosen})}))
    return dict(status='MEASURED_GEOMETRIC_ENVELOPE',summaries=summaries,rows=rows,
        protocol=dict(housing_assumption='World-aligned50mmcube centered at optical origin; explicit evaluation assumption, not a real calibrated sensor housing.',
            reference='Exact copied visible robot triangle-mesh solids at five actual native endpoint postures.',
            limitations='No real mounting bracket/cables, table/other-asset mounting clearance or continuous robot swept-volume certification. Jis hypothetical and not rendered or attached.',
            runtime='No collision geometry is introduced into the live scene; these are offline evaluation boxes only.'))


def evaluate_layouts(layouts, poses, density=4):
    scenarios=fixtures();background=background_shapes()
    surface=[]
    for scenario in scenarios:
        for shape in scenario['objects']:
            points,normals,weights=shape_surface(shape,density)
            surface.append((scenario,shape,points,normals,weights))
    allpoints=np.concatenate([x[2] for x in surface])
    allnormals=np.concatenate([x[3] for x in surface])
    allweights=np.concatenate([x[4] for x in surface])
    offsets=np.cumsum([0]+[len(x[2]) for x in surface])
    # Robot visibility cached once per camera/posture; fixture occlusion is
    # scenario-specific and remains separate, avoiding invented object mixtures.
    cache={};rows=[];coverage_arrays={};pose_metadata=[]
    for pose in poses:
        points,groups=region_probes(pose)
        grid=workspace_grid()
        overlap={shape['id']:fixture_robot_clearance(pose,shape) for scenario in scenarios for shape in scenario['objects']}
        pose_metadata.append(dict(pose_id=pose.pose_id,source=pose.source,
            hand_position_world_m=pose.parts[0].world_from_local[:3,3],
            render_meshes=len(pose.parts),triangles=sum(len(p.mesh.triangles) for p in pose.parts),
            region_probe_count=len(points),part_names=pose.part_names,
            body_group_scope='Saved names when available;11-part V2b ordering otherwise, retained as explicit approximation.'))
        for layout in layouts:
            cameras=list(layout['cameras'])
            if layout['family']=='J':cameras[-1]=wrist_camera(layout,pose)
            masks=[];probe_masks=[];grid_masks=[];camera_metrics=[]
            for camera in cameras:
                key=(pose.pose_id,tuple(np.asarray(camera['world_from_camera_ros']).ravel()),tuple(camera['resolution']))
                if key not in cache:
                    mask,uv=visibility(camera,allpoints,pose.parts,background,allnormals)
                    for index,(scenario,shape,p,n,w) in enumerate(surface):
                        other=[s for s in scenario['objects'] if s['id']!=shape['id']]
                        if other:
                            additional,_=visibility(camera,p,(),other,n)
                            mask[offsets[index]:offsets[index+1]] &= additional
                    probes,_=visibility(camera,points,pose.parts,background)
                    grids,_=visibility(camera,grid,pose.parts,background)
                    cache[key]=(mask,uv,probes,grids)
                mask,uv,probes,grids=cache[key]
                masks.append(mask);probe_masks.append(probes);grid_masks.append(grids)
                camera_metrics.append(dict(camera_id=camera['camera_id'],actual_evaluated_camera=camera,
                    sampled_surface_visible_fraction=float(allweights[mask].sum()/allweights.sum()),
                    physical_interference_scope='Passive optical prim has no collision housing. Actual mounting/housing/cable clearance is unvalidated.'))
            counts=np.sum(masks,axis=0);probe_counts=np.sum(probe_masks,axis=0);grid_counts=np.sum(grid_masks,axis=0)
            object_rows=[]
            for index,(scenario,shape,p,n,w) in enumerate(surface):
                section=slice(offsets[index],offsets[index+1]);c=counts[section]
                object_rows.append(dict(scenario_id=scenario['scenario_id'],object_id=shape['id'],shape=shape,
                    visible_one_fraction=float(w[c>=1].sum()/w.sum()),visible_two_fraction=float(w[c>=2].sum()/w.sum()),
                    unobserved_fraction=float(w[c==0].sum()/w.sum()),surface_area_m2=float(w.sum()),
                    per_camera_visible_surface_samples=[int(mask[section].sum()) for mask in masks],
                    projected_visible_sample_pixels=[len(np.unique(np.floor(cache[(pose.pose_id,
                        tuple(np.asarray(camera['world_from_camera_ros']).ravel()),tuple(camera['resolution']))][1][section][mask[section]]).astype(int),axis=0))
                        for camera,mask in zip(cameras,masks)],
                    raw_depth_pixels=None,raw_depth_scope='Native rendered-depth validation required; projected surface samples are not pixel-count estimates.'))
                object_rows[-1]['robot_overlap_evaluation']=overlap[shape['id']]
            region={name:dict(probes=int(np.count_nonzero(groups==name)),
                visible_one_fraction=float(np.mean(probe_counts[groups==name]>=1)),
                visible_two_fraction=float(np.mean(probe_counts[groups==name]>=2))) for name in np.unique(groups)}
            new=float(allweights[(np.sum(masks[:-1],axis=0)==0)&masks[-1]].sum()/allweights.sum()) if len(cameras)>2 else 0.
            rows.append(dict(layout_id=layout['layout_id'],family=layout['family'],pose_id=pose.pose_id,
                cameras=camera_metrics,relevant_surface_visible_one_fraction=float(allweights[counts>=1].sum()/allweights.sum()),
                relevant_surface_visible_two_fraction=float(allweights[counts>=2].sum()/allweights.sum()),
                relevant_surface_unobserved_fraction=float(allweights[counts==0].sum()/allweights.sum()),
                third_camera_new_surface_fraction=new,region_probes=region,
                object_macro_visible_fraction=float(np.mean([r['visible_one_fraction'] for r in object_rows])),
                nonoverlap_object_macro_visible_fraction=float(np.mean([r['visible_one_fraction'] for r in object_rows
                    if not r['robot_overlap_evaluation']['potential_overlap']])),
                potential_overlap_object_ids=[r['object_id'] for r in object_rows if r['robot_overlap_evaluation']['potential_overlap']],
                blind_spot_bins=blind_spots(grid,grid_counts),
                relevant_path_visible_fraction=region['tcp_path']['visible_one_fraction'],
                full_bounds_probe_visible_fraction=float(np.mean(grid_counts>=1)),objects=object_rows))
            coverage_arrays[layout['layout_id']+'__'+pose.pose_id]=dict(points=grid,counts=grid_counts,
                relevant_points=points,relevant_counts=probe_counts,groups=groups,
                surface_points=allpoints,surface_counts=counts,surface_weights=allweights)
    summaries=[]
    for layout in layouts:
        selected=[r for r in rows if r['layout_id']==layout['layout_id']]
        summaries.append(dict(layout_id=layout['layout_id'],family=layout['family'],camera_count=len(layout['cameras']),
            relevant_surface_visible_one_mean=float(np.mean([r['relevant_surface_visible_one_fraction'] for r in selected])),
            relevant_surface_visible_one_min=float(min(r['relevant_surface_visible_one_fraction'] for r in selected)),
            relevant_surface_visible_two_mean=float(np.mean([r['relevant_surface_visible_two_fraction'] for r in selected])),
            tcp_path_visible_mean=float(np.mean([r['relevant_path_visible_fraction'] for r in selected])),
            full_bounds_probe_visible_mean=float(np.mean([r['full_bounds_probe_visible_fraction'] for r in selected])),
            third_camera_new_surface_mean=float(np.mean([r['third_camera_new_surface_fraction'] for r in selected])),
            object_macro_visible_mean=float(np.mean([r['object_macro_visible_fraction'] for r in selected])),
            nonoverlap_object_macro_visible_mean=float(np.mean([r['nonoverlap_object_macro_visible_fraction'] for r in selected])),
            arm_region_min_mean=float(np.mean([min(r['region_probes'][key]['visible_one_fraction']
                for key in ('shoulder','elbow','forearm','wrist')) for r in selected])),
            region_probes={name:float(np.mean([r['region_probes'][name]['visible_one_fraction'] for r in selected]))
                           for name in selected[0]['region_probes']}))
    return dict(status='PASS',layouts=layouts,pose_metadata=pose_metadata,rows=rows,summaries=summaries,
        pareto_two=pareto_rows(summaries,2),pareto_three=pareto_rows(summaries,3),
        protocol=dict(mapping_bounds=BOUNDS,surface_density=density,
            truth_scope='Evaluation-only analytic fixtures, saved background geometry, and exact copied visible robot triangle meshes. Never runtime RGB-D input.',
            surface_measure='Area-weighted convex analytic physical surfaces, front-facing and unoccluded along calibrated optical rays; full physical back surfaces remain denominator.',
            empty_scope='Empty means no additional fixtures; original final-scene visual objects and the robot remain.',
            depth_scope='No claim of native depth counts from this screen. Raw rendered pixel counts require native shortlist capture.',
            probe_scope='40mm external actual-link AABB face probes plus bounded hover/side path neighborhoods; sampled poses are not a continuous swept-volume proof.',
            probe_occlusion_scope='Link-neighborhood and workspace-bin probes include the robot and original background only; fixture-specific occlusion is evaluated in the separate physical surface rows.',
            swept_scope='Five captured endpoint postures plus a local hover/side corridor; complete interpolated full-arm swept-volume coverage is unmeasured.',
            physical_mount_scope='Optical viewpoints only. No camera housing/cable model or mount feasibility is inferred.',
            wrist_scope='Hypothetical hand-relative viewpoint evaluated from each copied robot pose. No wrist sensor attached, calibrated, synchronized or implemented.')),coverage_arrays


def figures(report, arrays, output):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/camera-observability-feasibility/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summaries=report['summaries']
    best_two=max((r for r in summaries if r['camera_count']==2),key=lambda r:r['relevant_surface_visible_one_mean'])
    best_three=max((r for r in summaries if r['camera_count']==3 and r['family']!='J'),key=lambda r:r['relevant_surface_visible_one_mean'])
    best_wrist=max((r for r in summaries if r['family']=='J'),key=lambda r:r['relevant_surface_visible_one_mean'])
    chosen=['A_current',best_two['layout_id'],best_three['layout_id'],best_wrist['layout_id']]
    layouts={r['layout_id']:r for r in report['layouts']}
    first_pose=report['pose_metadata'][0]['pose_id']
    representative_cameras={name:[item['actual_evaluated_camera'] for item in next(
        row for row in report['rows'] if row['layout_id']==name and row['pose_id']==first_pose)['cameras']]
        for name in chosen}
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,name in zip(axes.flat,chosen):
        for camera in representative_cameras[name]:
            eye=np.asarray(camera['position_world_m']);forward=np.asarray(camera['world_from_camera_ros'])[:3,2]
            ax.scatter(eye[0],eye[1],s=45)
            ax.arrow(eye[0],eye[1],forward[0]*.20,forward[1]*.20,width=.008,length_includes_head=True)
        ax.add_patch(plt.Rectangle(BOUNDS[0,:2],*(BOUNDS[1,:2]-BOUNDS[0,:2]),fill=False,color='gray'))
        ax.set_title(name,fontsize=9);ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    fig.tight_layout();fig.savefig(output/'camera_top_down.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in chosen:
        for i,camera in enumerate(representative_cameras[name]):
            eye=np.asarray(camera['position_world_m']);target=np.asarray(camera['aim_world_m'])
            distance=np.linalg.norm(eye[:2]-CENTER[:2]);targetdistance=np.linalg.norm(target[:2]-CENTER[:2])
            ax.plot([distance,targetdistance],[eye[2],target[2]],'-o',label=f'{name} / {i+1}')
    ax.set(xlabel='Horizontal distance to mapping-center (m)',ylabel='Height (m)',title='Pose, height and actual look direction')
    ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(output/'camera_height_angle.png',dpi=160);plt.close(fig)
    pose=report['pose_metadata'][0]['pose_id']
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True,sharey=True)
    for ax,name in zip(axes,chosen):
        values=arrays[name+'__'+pose]
        points=values['points'];counts=values['counts']
        xz=np.unique(points[:,[0,2]],axis=0)
        visible_fraction=np.array([np.mean(counts[np.all(np.isclose(points[:,[0,2]],point),axis=1)]>=1) for point in xz])
        im=ax.scatter(xz[:,0],xz[:,1],c=visible_fraction,cmap='viridis',vmin=0,vmax=1,s=90,marker='s')
        ax.set(title=name,xlabel='X (m)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Z (m)');fig.colorbar(im,ax=list(axes),label='Fraction of Y-layer probes visible to any camera')
    fig.savefig(output/'workspace_visibility.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for family in sorted({r['family'] for r in summaries}):
        selected=[r for r in summaries if r['family']==family]
        ax.scatter([r['full_bounds_probe_visible_mean']*100 for r in selected],
                   [r['relevant_surface_visible_one_mean']*100 for r in selected],label=family,s=35)
    ax.set(xlabel='Full mapping-bounds probes visible (%)',ylabel='Relevant obstacle surface visible (%)',
           title='Components remain separate: workspace coverage and hazard surfaces')
    ax.legend(title='Family',ncol=6,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/'visibility_components.png',dpi=160);plt.close(fig)
    comparisons=chosen[:3]
    fig,axes=plt.subplots(3,len(comparisons),figsize=(11,9),sharex=True,sharey=True)
    for column,name in enumerate(comparisons):
        record=arrays[name+'__'+first_pose];points=record['points'];counts=record['counts']
        for row,height in enumerate((.20,.50,.95)):
            z=points[np.argmin(abs(points[:,2]-height)),2]
            keep=np.isclose(points[:,2],z)
            ax=axes[row,column]
            image=ax.scatter(points[keep,0],points[keep,1],c=counts[keep],cmap='viridis',vmin=0,vmax=3,marker='s',s=100)
            ax.set_title(f'{name}\nZ={z:.3f}m',fontsize=8);ax.set_aspect('equal');ax.grid(alpha=.15)
            if row==2:ax.set_xlabel('World X (m)')
            if column==0:ax.set_ylabel('World Y (m)')
    fig.colorbar(image,ax=list(axes.flat),label='Visible camera count; zero = sampled blind bin')
    fig.savefig(output/'blind_spot_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return chosen


def export_compact(report, source_directory, destination, native=None):
    """Save reviewable components and selected per-object rows, excluding arrays."""
    source_directory,destination=Path(source_directory),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    selected=set(report['shortlist'])|{'C_asymmetric_1.15_0.65','H_third_side_-1.00',
        'K_C_asymmetric_1.15_0.65_rear','I_third_overhead_1.25'}
    result={key:report[key] for key in ('status','protocol','pose_metadata','summaries','pareto_two','pareto_three','shortlist')}
    result['pose_capture_source']=report.get('pose_capture_source')
    result['candidate_layouts']=len(report['layouts'])
    result['selected_layout_rows']=[dict(layout_id=row['layout_id'],pose_id=row['pose_id'],
        relevant_surface_visible_one_fraction=row['relevant_surface_visible_one_fraction'],
        relevant_surface_visible_two_fraction=row['relevant_surface_visible_two_fraction'],
        nonoverlap_object_macro_visible_fraction=row['nonoverlap_object_macro_visible_fraction'],
        region_probes=row['region_probes'],objects=row['objects'],blind_spot_bins=row['blind_spot_bins'],
        potential_overlap_object_ids=row['potential_overlap_object_ids'])
        for row in report['rows'] if row['layout_id'] in selected]
    result['wrist_evaluated_world_poses']=[dict(pose_id=row['pose_id'],layout_id=row['layout_id'],
        camera=row['cameras'][-1]['actual_evaluated_camera']) for row in report['rows'] if row['family']=='J']
    result['excluded_initial_attempts']=[dict(path='outputs/camera-observability-feasibility/capture-A-current/capture.json',
        status='FAIL',reason='Physical collision occluders interfered with posture setup; retain errors but exclude from layout recommendation.'),
        dict(path='outputs/camera-observability-feasibility/geometry-initial/geometry_screen.json',
        status='PRELIMINARY',reason='Historical three-posture screen, earlier sphere placement, and an initially incorrect scaled-mesh wrist offset; superseded by fresh five-posture geometry.')]
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(destination/'geometry_screen.json',result)
    write_json(destination/'camera_layouts.json',dict(layouts=report['layouts']))
    write_json(destination/'visibility_fixtures.json',dict(scenarios=fixtures()))
    if native is not None:
        compact=dict(native)
        compact['rows']=[{key:value for key,value in row.items() if key!='calibration'} for row in native['rows']]
        compact['actual_native_calibration']={identifier:next(
            row['calibration'] for row in native['rows'] if row['layout_id']==identifier)
            for identifier in {row['layout_id'] for row in native['rows']}}
        for calibration in compact['actual_native_calibration'].values():
            for camera in calibration:
                matrix=np.asarray(camera['world_from_camera_ros'])
                camera['actual_orientation_ros_wxyz']=quaternion_wxyz(matrix[:3,:3])
                camera['height_m']=matrix[2,3]
                camera['distance_to_mapping_center_m']=float(np.linalg.norm(matrix[:3,3]-CENTER))
                camera['horizontal_distance_to_mapping_center_m']=float(np.linalg.norm(matrix[:2,3]-CENTER[:2]))
                camera['optical_axis_unit_aim_world_m']=matrix[:3,3]+matrix[:3,2]
        write_json(destination/'native_visibility.json',compact)
    for path in (source_directory/'figures').glob('*.png'):
        shutil.copyfile(path,destination/path.name)


def reduce_native_captures(paths, output):
    """Pair predicted and rendered visibility on identical surface quadrature."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];sources=[];layouts={};representative={}
    overlap_reference={}
    geometry_path=output/'geometry_screen.json'
    if geometry_path.exists():
        geometry=json.loads(geometry_path.read_text())
        for row in geometry.get('rows',geometry.get('selected_layout_rows',[])):
            if row['layout_id']=='A_current':
                for item in row['objects']:
                    overlap_reference[(row['pose_id'],item['object_id'])]=item.get('robot_overlap_evaluation')
    for location in paths:
        path=Path(location)
        if path.is_dir():path=path/'capture.json'
        document=json.loads(path.read_text())
        identifier=document['layout_id'];layouts[identifier]=document['layout']
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            layout_id=identifier,status=document['status'],errors=document.get('errors',[]),
            motions=document.get('motions',[]),captured_postures=[item.get('id',item.get('pose_id')) for item in document['snapshots']],
            actual_postures=[dict(posture=item.get('id',item.get('pose_id')),tcp_position=item['tcp_position'],joints=item['joints'])
                             for item in document['snapshots']]))
        for record in document['records']:
            with np.load(path.parent/record['image_npz']) as data:
                if record['scenario_id']=='thin_post' and record['posture']=='hover':
                    representative[identifier]=(path.parent/record['image_npz'],record)
                for shape in record['objects']:
                    positions=[i for i,item in enumerate(record['scores']) if item['object_id']==shape['id']]
                    predicted=np.stack([data[f'score_{i}_predicted_mask'] for i in positions])
                    native=np.stack([data[f'score_{i}_native_mask'] for i in positions])
                    weights=data[f'score_{positions[0]}_surface_weights']
                    if not all(np.allclose(weights,data[f'score_{i}_surface_weights']) for i in positions):
                        raise ValueError('Native camera surface quadrature weights differ')
                    native_count=native.sum(0);predicted_count=predicted.sum(0)
                    area=weights.sum()
                    pixelrows=[record['scores'][i] for i in positions]
                    rows.append(dict(layout_id=identifier,posture=record['posture'],scenario_id=record['scenario_id'],
                        object_id=shape['id'],surface_area_m2=float(area),
                        native_visible_one_fraction=float(weights[native_count>=1].sum()/area),
                        native_visible_two_fraction=float(weights[native_count>=2].sum()/area),
                        native_unobserved_fraction=float(weights[native_count==0].sum()/area),
                        predicted_visible_one_fraction=float(weights[predicted_count>=1].sum()/area),
                        predicted_visible_two_fraction=float(weights[predicted_count>=2].sum()/area),
                        native_prediction_union_agreement=float(weights[(native_count>=1)==(predicted_count>=1)].sum()/area),
                        per_camera=[dict(camera_id=item['camera_id'],raw_projected_object_pixels=item['raw_projected_object_pixels'],
                            visible_return_pixels=item['native_surface_return_pixels'],valid_depth_pixels=item['valid_depth_pixels'],
                            invalid_depth_pixels=item['invalid_depth_pixels'],occluded_pixels=item['occluded_pixels'],
                            sampled_surface_agreement=item['sampled_visibility_agreement']) for item in pixelrows],
                        summed_visible_return_pixels=sum(item['native_surface_return_pixels'] for item in pixelrows),
                        robot_overlap_evaluation=overlap_reference.get((record['posture'],shape['id'])),
                        calibration=record['cameras']))
    conditions={identifier:{(r['posture'],r['scenario_id'],r['object_id']) for r in rows if r['layout_id']==identifier}
                for identifier in layouts}
    common=set.intersection(*conditions.values()) if conditions else set()
    summaries=[]
    for identifier in layouts:
        matched=[r for r in rows if r['layout_id']==identifier and (r['posture'],r['scenario_id'],r['object_id']) in common]
        area=sum(r['surface_area_m2'] for r in matched)
        summaries.append(dict(layout_id=identifier,matched_object_posture_conditions=len(matched),
            native_surface_visible_one_fraction=None if not area else sum(r['surface_area_m2']*r['native_visible_one_fraction'] for r in matched)/area,
            native_surface_visible_two_fraction=None if not area else sum(r['surface_area_m2']*r['native_visible_two_fraction'] for r in matched)/area,
            object_macro_visible_fraction=None if not matched else float(np.mean([r['native_visible_one_fraction'] for r in matched])),
            native_prediction_union_agreement=None if not area else sum(r['surface_area_m2']*r['native_prediction_union_agreement'] for r in matched)/area,
            visible_return_pixels=sum(r['summed_visible_return_pixels'] for r in matched),
            per_posture={posture:dict(object_conditions=len(selected),
                surface_visible_one_fraction=sum(r['surface_area_m2']*r['native_visible_one_fraction'] for r in selected)/sum(r['surface_area_m2'] for r in selected))
                for posture in sorted({r['posture'] for r in matched})
                if (selected:=[r for r in matched if r['posture']==posture])},
            per_object={obj:dict(mean_visible_fraction=float(np.mean([r['native_visible_one_fraction'] for r in selected])),
                mean_return_pixels=float(np.mean([r['summed_visible_return_pixels'] for r in selected])),
                worst_visible_fraction=float(min(r['native_visible_one_fraction'] for r in selected)))
                for obj in sorted({r['object_id'] for r in matched})
                if (selected:=[r for r in matched if r['object_id']==obj])}))
        eligible=[row for row in matched if row.get('robot_overlap_evaluation') is not None
                  and row['robot_overlap_evaluation']['potential_overlap'] is False]
        eligible_area=sum(row['surface_area_m2'] for row in eligible)
        summaries[-1]['primary_nonoverlap']=dict(matched_object_posture_conditions=len(eligible),
            native_surface_visible_one_fraction=None if not eligible_area else sum(row['surface_area_m2']*row['native_visible_one_fraction'] for row in eligible)/eligible_area,
            native_surface_visible_two_fraction=None if not eligible_area else sum(row['surface_area_m2']*row['native_visible_two_fraction'] for row in eligible)/eligible_area,
            object_macro_visible_fraction=None if not eligible else float(np.mean([row['native_visible_one_fraction'] for row in eligible])),
            native_prediction_union_agreement=None if not eligible_area else sum(row['surface_area_m2']*row['native_prediction_union_agreement'] for row in eligible)/eligible_area,
            visible_return_pixels=sum(row['summed_visible_return_pixels'] for row in eligible))
        summaries[-1]['per_object_nonoverlap']={obj:dict(conditions=len(selected),mean_visible_fraction=float(np.mean([
            row['native_visible_one_fraction'] for row in selected])),mean_return_pixels=float(np.mean([
            row['summed_visible_return_pixels'] for row in selected])),worst_visible_fraction=float(min([
            row['native_visible_one_fraction'] for row in selected])))
            for obj in sorted({row['object_id'] for row in eligible})
            if (selected:=[row for row in eligible if row['object_id']==obj])}
    reference=next((source for source in sources if source['layout_id']=='A_current'),sources[0])
    original={row['posture']:row for row in reference['actual_postures']}
    pose_matching=[]
    for source in sources:
        comparisons=[]
        for posture in source['actual_postures']:
            if posture['posture'] not in original:continue
            other=original[posture['posture']]
            comparisons.append(dict(posture=posture['posture'],
                tcp_difference_m=float(np.linalg.norm(np.asarray(posture['tcp_position'])-other['tcp_position'])),
                maximum_joint_difference_rad=float(np.max(abs(np.asarray(posture['joints'])-other['joints'])))))
        pose_matching.append(dict(layout_id=source['layout_id'],postures=comparisons,
            maximum_tcp_difference_m=max(row['tcp_difference_m'] for row in comparisons),
            maximum_joint_difference_rad=max(row['maximum_joint_difference_rad'] for row in comparisons)))
    if overlap_reference and any(row['maximum_tcp_difference_m']>1e-7 or row['maximum_joint_difference_rad']>1e-7 for row in pose_matching):
        raise ValueError('Reference overlap flags cannot be transferred to different robot postures; recompute actual mesh clearance for each layout')
    result=dict(status='MEASURED',sources=sources,summaries=summaries,rows=rows,
        pose_matching=dict(reference_layout=reference['layout_id'],comparisons=pose_matching,
            scope='Same requested direct targets and unchanged tolerances; actual endpoint differences retained, not forced or teleported.'),
        common_condition_count=len(common),common_conditions=sorted(common),
        protocol=dict(scope='Rendered-depth and geometric predictions compared on identical deterministic physical-surface quadrature; GT used only after capture for evaluation.',
            union='Per-object pointwise union of actual native masks across cameras, with true physical area weights.',
            pixel_scope='Exact per-camera pixel counts from independent first intersections matched to rendered depth within5mm; image-coordinate pixels across cameras are not physically deduplicated.',
            native_surface_tolerance_m=.008,depth_pixel_match_tolerance_m=.005,
            comparison='Only common posture/scenario/object conditions enter cross-layout summary; all source rows, missing-posture errors and failed movements remain retained.',
            primary_nonoverlap='Conservative actual robot-mesh/enclosing-fixture-box flags exclude potential intersections from primary metrics. All65conditions remain in stress-context metrics. Flags transfer only after actual robot endpoint poses match within1e-7m/rad.',
            timing='Full captures and scoring are excluded from runtime performance claims.'))
    write_json(output/'native_visibility.json',result)
    native_figures(result,representative,output/'figures')
    return result


def native_figures(report, representatives, output):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/camera-observability-feasibility/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=[row['layout_id'] for row in report['summaries']]
    objects=sorted({row['object_id'] for row in report['rows']})
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    x=np.arange(len(objects));width=.8/max(1,len(names))
    for index,row in enumerate(report['summaries']):
        offset=(index-(len(names)-1)/2)*width
        component=row.get('per_object_nonoverlap') or row['per_object']
        axes[0].bar(x+offset,[component.get(obj,{}).get('mean_visible_fraction',np.nan)*100 for obj in objects],width,label=row['layout_id'])
        axes[1].bar(x+offset,[component.get(obj,{}).get('mean_return_pixels',np.nan) for obj in objects],width,label=row['layout_id'])
    for ax in axes:
        ax.set_xticks(x,objects,rotation=70,ha='right',fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Physical surface visible to ≥1 camera (%)');axes[1].set_ylabel('Native matched depth pixels, summed views')
    axes[0].set_title('Conservatively non-overlapping fixture/posture conditions')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'native_obstacle_visibility.png',dpi=160);plt.close(fig)
    if representatives:
        columns=max(len(record['cameras']) for _,record in representatives.values())*2
        fig,axes=plt.subplots(len(representatives),columns,figsize=(3*columns,3*len(representatives)),squeeze=False)
        for row,(name,(path,record)) in enumerate(representatives.items()):
            with np.load(path) as data:
                for index in range(len(record['cameras'])):
                    rgb=data[f'rgb_{index}'];depth=data[f'depth_{index}']
                    axes[row,2*index].imshow(rgb);axes[row,2*index].set_title(f'{name} / view{index+1}',fontsize=8)
                    valid=np.where(np.isfinite(depth)&(depth>0),depth,np.nan)
                    axes[row,2*index+1].imshow(valid,vmin=0,vmax=3,cmap='viridis')
                    axes[row,2*index+1].set_title('Depth (0–3m)',fontsize=8)
            for ax in axes[row]:ax.axis('off')
        fig.tight_layout();fig.savefig(output/'native_rgb_depth_gallery.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/camera-observability-feasibility/geometry')
    parser.add_argument('--catalog-only',action='store_true')
    parser.add_argument('--geometry-manifest',type=Path)
    parser.add_argument('--density',type=int,default=4)
    parser.add_argument('--native-captures',nargs='+',type=Path)
    parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--compact-assets',type=Path)
    args=parser.parse_args()
    if not 2<=args.density<=10:parser.error('density must lie between2and10')
    layouts=catalog();args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'layouts.json',dict(layouts=layouts))
    write_json(args.output/'fixtures.json',dict(scenarios=fixtures()))
    if args.native_captures:
        native=reduce_native_captures(args.native_captures,args.output)
        print(json.dumps(dict(native_visibility='MEASURED',layouts=len(native['summaries']),common_conditions=native['common_condition_count'])),flush=True)
    if args.native_only:
        if args.compact_assets:
            if (args.output/'geometry_screen.json').exists():
                saved=json.loads((args.output/'geometry_screen.json').read_text())
                export_compact(saved,args.output,args.compact_assets,native)
            else:
                parser.error('--compact-assets with --native-only requires an existing geometry_screen.json in output')
        return
    if args.catalog_only:
        print(f'Wrote{len(layouts)}layout candidates to{args.output}',flush=True);return
    before=time.monotonic()
    report,arrays=evaluate_layouts(layouts,load_poses(args.geometry_manifest),args.density)
    if args.geometry_manifest:
        source=json.loads(args.geometry_manifest.read_text())
        report['pose_capture_source']=dict(path=str(args.geometry_manifest),status=source.get('status'),
            errors=source.get('errors',[]),motions=source.get('motions',[]),
            qualification='Successful actual pose snapshots are evaluated; failed/unmeasured requested postures remain explicit source errors.')
    report['offline_compute_seconds']=time.monotonic()-before
    report['shortlist']=figures(report,arrays,args.output/'figures')
    write_json(args.output/'geometry_screen.json',report)
    np.savez_compressed(args.output/'coverage_arrays.npz',**{
        key+'__'+name:values for key,record in arrays.items() for name,values in record.items()})
    if args.compact_assets:
        export_compact(report,args.output,args.compact_assets,native if args.native_captures else None)
    print(json.dumps(dict(status=report['status'],layouts=len(layouts),poses=len(report['pose_metadata']),
        seconds=report['offline_compute_seconds'],shortlist=report['shortlist'])),flush=True)


if __name__=='__main__':
    main()
