#!/usr/bin/env python3
"""Regenerate camera-study scene eligibility and optional semantic FOV audits.

No simulator, rendering, map processing, controller or benchmark is launched.
Inputs are saved native visibility rows and declared primitive geometry. The
42-condition restriction is derived from intersections, never hard-coded as an
eligibility mask. Existing54/all65 summaries are not modified by this reducer.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain(item) for item in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def source_name(path):
    path=Path(path).resolve()
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(shape):
    angle=shape.get('yaw',0.)
    return np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])


def half_extent(shape):
    if shape['kind']=='box':return np.asarray(shape['dimensions'])/2
    if shape['kind']=='sphere':return np.full(3,shape['radius'])
    return np.array([shape['radius'],shape['radius'],shape['height']/2])


def corners(shape):
    local=np.array(list(itertools.product((-1.,1.),repeat=3)))*half_extent(shape)
    return local@rotation(shape).T+shape['center']


def intersect(first,second):
    """Exact boxes/spheres; conservatively flag other enclosing-box candidates."""
    a,b=corners(first),corners(second)
    extent=np.minimum(a.max(0),b.max(0))-np.maximum(a.min(0),b.min(0))
    if np.any(extent<=1e-10):
        return False,'Separated or only touching enclosing world AABBs',extent
    if first['kind']==second['kind']=='box':
        ar,br=rotation(first),rotation(second)
        axes=[*ar.T,*br.T]+[np.cross(u,v) for u in ar.T for v in br.T]
        for axis in axes:
            length=np.linalg.norm(axis)
            if length<1e-12:continue
            axis=axis/length;ap,bp=a@axis,b@axis
            if min(ap.max(),bp.max())-max(ap.min(),bp.min())<=1e-10:
                return False,'Exact oriented-box SAT separation',extent
        return True,'Exact oriented-box SAT positive-volume intersection',extent
    if {first['kind'],second['kind']}=={'box','sphere'}:
        box,sphere=(first,second) if first['kind']=='box' else (second,first)
        local=(np.asarray(sphere['center'])-box['center'])@rotation(box)
        distance=np.linalg.norm(local-np.clip(local,-half_extent(box),half_extent(box)))
        return bool(distance<sphere['radius']-1e-10),'Exact sphere/oriented-box closest-point test',extent
    if first['kind']==second['kind']=='sphere':
        hit=np.linalg.norm(np.asarray(first['center'])-second['center'])<first['radius']+second['radius']-1e-10
        return bool(hit),'Exact sphere/sphere distance test',extent
    return True,'Conservative enclosing-world-AABB potential intersection; not an exact curved-primitive verdict',extent


def background_shapes(current):
    return [dict(id='background_'+item['name'],kind='box' if item['shape']=='cube' else item['shape'],
        center=item['center'],dimensions=item['size'],radius=item['radius'],yaw=item.get('yaw',0.))
        for item in current['objects']]


def aggregate(rows):
    area=sum(row['surface_area_m2'] for row in rows)
    if not area:raise ValueError('No surface area remains in the comparison')
    return dict(condition_count=len(rows),surface_area_denominator_m2=area,
        visible_surface_one_fraction=sum(row['surface_area_m2']*row['native_visible_one_fraction'] for row in rows)/area,
        visible_surface_two_fraction=sum(row['surface_area_m2']*row['native_visible_two_fraction'] for row in rows)/area,
        object_macro_visible_fraction=float(np.mean([row['native_visible_one_fraction'] for row in rows])))


def qualify(native,scenarios,current,sources):
    pairs=[];tested=0
    for scenario in scenarios:
        for fixture in scenario['objects']:
            for background in background_shapes(current):
                tested+=1;hit,method,extent=intersect(fixture,background)
                if hit:pairs.append(dict(scenario_id=scenario['scenario_id'],fixture_id=fixture['id'],
                    background_id=background['id'],method=method,world_aabb_overlap_extent_m=extent,
                    fixture=fixture,background=background))
    affected={(pair['scenario_id'],pair['fixture_id']) for pair in pairs}
    sensitivity=[]
    for layout in sorted({row['layout_id'] for row in native['rows']}):
        robot_clear=[row for row in native['rows'] if row['layout_id']==layout
            and row.get('robot_overlap_evaluation') is not None
            and not row['robot_overlap_evaluation']['potential_overlap']]
        disjoint=[row for row in robot_clear if (row['scenario_id'],row['object_id']) not in affected]
        sensitivity.append(dict(layout_id=layout,robot_fixture_nonoverlap=aggregate(robot_clear),
            robot_and_background_disjoint=aggregate(disjoint),
            excluded_background_intersection_conditions=[dict(posture=row['posture'],scenario_id=row['scenario_id'],object_id=row['object_id'])
                for row in robot_clear if (row['scenario_id'],row['object_id']) in affected]))
    rank_robot=[row['layout_id'] for row in sorted(sensitivity,key=lambda row:row['robot_fixture_nonoverlap']['visible_surface_one_fraction'],reverse=True)]
    rank_disjoint=[row['layout_id'] for row in sorted(sensitivity,key=lambda row:row['robot_and_background_disjoint']['visible_surface_one_fraction'],reverse=True)]
    return dict(schema_version=1,status='POSTHOC_SCENE_GEOMETRY_QUALIFICATION',fixture_background_pairs_tested=tested,
        intersecting_or_potential_pairs=pairs,affected_scenario_object_pairs=sorted(affected),sensitivity=sensitivity,
        rank_order_robot_fixture_nonoverlap=rank_robot,rank_order_also_background_disjoint=rank_disjoint,
        rank_order_unchanged=rank_robot==rank_disjoint,sources=sources,
        scope=dict(primary_54='The existing54condition label means robot/fixture-nonoverlap only; it does not certify mutually disjoint full scenes.',
            geometry='Research fixtures and original four tabletop primitives form static rendered volume unions. Some volumes intersect. Depth and occlusion remain actual rendered measurements; full declared primitive surfaces include buried/unobservable pieces.',
            sensitivity='Further restrict the same existing native measurements to fixture/background-disjoint conditions. No rerender, retuning, deletion, replacement of original metrics, or new sampling/performance measurement.',
            support_plane='Expected ground/table support contact is outside this inter-object check.',
            planner='Separate native physical planner fixtures deactivate original FinalPerceptionObjects and use prevalidated non-overlapping physical setups; their collision results are unaffected by this visibility-scene qualification.',
            limits='Exact tests cover boxes/spheres; curved-cylinder cases would use conservative enclosing bounds if any were potential intersections. No continuous motion/scene certification is implied.'))


def projected(camera,points):
    matrix=np.asarray(camera['world_from_camera_ros'])
    xyz=(np.asarray(points)-matrix[:3,3])@matrix[:3,:3]
    intrinsic=np.asarray(camera['intrinsics']);depth=xyz[:,2]
    pixels=np.full((len(xyz),2),np.nan);good=depth>1e-8
    pixels[good]=xyz[good,:2]/depth[good,None]*[intrinsic[0,0],intrinsic[1,1]]+[intrinsic[0,2],intrinsic[1,2]]
    width,height=camera['resolution']
    inside=good&(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    return xyz,pixels,inside


def semantic_projection(native,current,sources):
    identifier='C_asymmetric_1.15_0.65'
    camera=native['actual_native_calibration'][identifier][1]
    rows=[]
    for item in current['objects']:
        center=np.asarray(item['center'],float)
        shape=dict(kind='box',center=item['center'],dimensions=item['size'],yaw=item.get('yaw',0.))
        box=corners(shape)
        xyz,pixels,inside=projected(camera,np.vstack((center,box)))
        rows.append(dict(object_name=item['name'],shape=item['shape'],center_world_m=center,
            center_camera_xyz_m=xyz[0],center_uv=pixels[0],center_in_frame=bool(inside[0]),
            bounding_reference='Exact oriented box corners' if item['shape']=='cube' else 'Conservative axis-aligned enclosing sphere box corners',
            box_corner_world_m=box,box_corner_uv=pixels[1:],box_corner_camera_depth_m=xyz[1:,2],
            box_corner_in_frame=inside[1:],box_corner_in_frame_fraction=float(inside[1:].mean()),
            box_uv_bounds=[pixels[1:].min(0),pixels[1:].max(0)],
            whole_enclosing_box_below_image=bool((xyz[1:,2]>0).all() and (pixels[1:,1]>=camera['resolution'][1]).all())))
    return dict(schema_version=1,status='ANALYTIC_PROJECTION_OF_MEASURED_CALIBRATION',layout_id=identifier,
        camera=camera,objects=rows,sources=sources,
        equations=dict(camera_coordinates='p_camera = R_world_from_camera.T @ (p_world - camera_origin)',
            continuous_pixel='u = fx*x/z + cx; v = fy*y/z + cy',image_membership='z>0 and0<=u<width and0<=v<height',
            half_pixel='Acquisition rays use integer pixel indices plus0.5; a continuous projected point is assigned floor(u),floor(v). No half-pixel shift is added twice.'),
        scope='No new rendering or perception run. Center/corner projection is frustum evidence only; corner fractions are not visible-area fractions. No occlusion is inferred. This helps explain lost semantic-view redundancy but does not prove the sole cause of classification errors.')


def assert_numeric_equal(first,second,path='root'):
    """Compare a regenerated audit including all measured fractions and masks."""
    if isinstance(first,dict) and isinstance(second,dict):
        if set(first)!=set(second):raise AssertionError(path+': different keys')
        for key in first:assert_numeric_equal(first[key],second[key],path+'.'+key)
    elif isinstance(first,list) and isinstance(second,list):
        if len(first)!=len(second):raise AssertionError(path+': different lengths')
        for index,(a,b) in enumerate(zip(first,second)):assert_numeric_equal(a,b,f'{path}[{index}]')
    elif isinstance(first,(int,float)) and isinstance(second,(int,float)):
        if not np.isclose(first,second,rtol=1e-12,atol=1e-12):raise AssertionError(path+': different numbers')
    elif first!=second:
        raise AssertionError(path+': different values')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'docs/assets/camera_observability_feasibility')
    parser.add_argument('--current-layout',type=Path,default=ROOT/'docs/assets/multiview_final/selected-layout.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/camera_observability_feasibility/geometry_scene_qualifications.json')
    parser.add_argument('--semantic-output',type=Path)
    parser.add_argument('--verify-against',type=Path)
    parser.add_argument('--verify-semantic-against',type=Path)
    args=parser.parse_args()
    native_path=args.assets_dir/'native_visibility.json';geometry_path=args.assets_dir/'geometry_screen.json'
    fixture_path=args.assets_dir/'visibility_fixtures.json'
    native=json.loads(native_path.read_text());current=json.loads(args.current_layout.read_text())
    scenarios=json.loads(fixture_path.read_text())['scenarios']
    sources={source_name(path):sha(path) for path in (native_path,geometry_path,args.current_layout,fixture_path)}
    result=plain(qualify(native,scenarios,current,sources))
    if args.verify_against:
        original=json.loads(args.verify_against.read_text())
        assert_numeric_equal(result,original)
    write_json(args.output,result)
    if args.semantic_output:
        semantic_sources={source_name(path):sha(path) for path in (native_path,args.current_layout)}
        # The compact native evidence retains the capture hash, so regeneration
        # does not require ignored raw captures to exist in a clean clone.
        reference=next(row for row in native['sources'] if row['layout_id']=='C_asymmetric_1.15_0.65')
        semantic_sources[reference['path']]=reference['sha256']
        projection=plain(semantic_projection(native,current,semantic_sources))
        if args.verify_semantic_against:
            assert_numeric_equal(projection,json.loads(args.verify_semantic_against.read_text()))
        write_json(args.semantic_output,projection)
    print(json.dumps(dict(status='PASS',pairs_tested=result['fixture_background_pairs_tested'],
        overlap_pairs=len(result['intersecting_or_potential_pairs']),
        primary_condition_counts=[row['robot_and_background_disjoint']['condition_count'] for row in result['sensitivity']],
        verified=args.verify_against is not None,output=str(args.output))))


if __name__=='__main__':
    main()
