"""One-shot research bridge; original source checks, planner and map stay intact."""
from dataclasses import replace
import time

import numpy as np

from robot_skill_stack.integrations.isaac.manipulation.motion.static_world import StaticObstacle
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from . import reference as ref,visibility


def fail(code,message):
    raise PlanningFailure('research_'+code,message)


def identity(view):
    return view.generation,view.revision,view.timestamp


def same_frames(frames,records):
    if len(frames)!=2 or len(records)!=2:return False
    lookup={r['camera_id']:r for r in records}
    if len(lookup)!=2 or set(lookup)!={f.camera_id for f in frames}:return False
    return all(tuple(f.token)==tuple(lookup[f.camera_id]['token'])
        and f.acquired_at==lookup[f.camera_id]['acquired_at']
        and f.depth.shape==(480,640)==tuple(lookup[f.camera_id]['depth_shape'])
        and np.array_equal(f.intrinsics,lookup[f.camera_id]['intrinsics'])
        and np.array_equal(f.world_from_camera,lookup[f.camera_id]['world_from_camera']) for f in frames)


def associate(source,inputs,expected=None):
    view=source.snapshot()
    if source.generation!=view.generation or (expected is not None and identity(view)!=tuple(expected)):
        fail('source_mismatch','The first accepted publication changed')
    records=source.diagnostics.get('source_frames',())
    matches=[p for p in inputs if p.generation==view.generation and p.frames
             and min(f.acquired_at for f in p.frames)==view.timestamp and same_frames(p.frames,records)]
    if len(matches)!=1 or matches[0].ground_truth is not None:
        fail('source_mismatch','No unique camera-only input exactly matches the accepted publication')
    return view,matches[0]


def combine(world,frozen,models,support_z=0.):
    if not models or any(m['status']!='qualified' or m['geometry'] is None for m in models):
        fail('geometry_unqualified','Combination requires qualified inferred boxes')
    descriptors=[]
    for i,model in enumerate(models):
        g=model['geometry'];yaw=g['yaw_rad']
        c,s=np.cos(yaw),np.sin(yaw)
        if not np.allclose(g['rotation'],[[c,-s,0],[s,c,0],[0,0,1]],atol=1e-12,rtol=0.):
            fail('invalid_geometry','Containment rotation and registered yaw disagree')
        descriptors.append(StaticObstacle(f'/__research_rgbd_inferred__/{frozen.generation}/{frozen.revision}/{i}',
            'cuboid',tuple(g['center']),orientation=(np.cos(yaw/2),0.,0.,np.sin(yaw/2)),
            dimensions=tuple(g['dimensions']),source_kind='rgbd_inferred_box',
            approximation='single_convex_yaw_box_empirical_bounds'))
    retained=[];covered={d.path:[] for d in descriptors};floor_ids=[]
    exact={b.id:b for b in frozen.boxes}
    for obstacle in world.obstacles:
        if obstacle.source_kind=='Plane':
            retained.append(obstacle);continue
        if obstacle.source_kind!='rgbd_occupied_cells':fail('invalid_world','Unexpected non-camera obstacle')
        key=obstacle.path.removeprefix('/__rgbd_occupied__/')
        box=exact[key];hi=int(key.split(':')[-1])
        top=min(frozen.spec.bounds_min[2]+hi*frozen.spec.voxel_size,frozen.spec.bounds_max[2])
        floor=top<=support_z+frozen.spec.voxel_size+1e-12
        if floor:floor_ids.append(key)
        corners=np.asarray(box.center)+ref.CORNERS*np.asarray(box.dimensions)
        owner=None
        if not floor:
            for d,m in zip(descriptors,models):
                g=m['geometry'];q=(corners-np.asarray(g['center']))@np.asarray(g['rotation'])
                if np.all(abs(q)+1e-12<=np.asarray(g['dimensions'])/2):owner=d.path;break
        if owner is None:retained.append(obstacle)
        else:covered[owner].append(key)
    old_ids={b.path.removeprefix('/__rgbd_occupied__/') for b in retained if b.source_kind=='rgbd_occupied_cells'}
    replaced={x for ids in covered.values() for x in ids}
    if old_ids|replaced!=set(exact) or old_ids&replaced or not set(floor_ids)<=old_ids:
        fail('coverage','Selected occupied union or floor evidence was lost')
    count=len(old_ids)+len(descriptors)
    if count>64:fail('proxy_budget','Conservative combined world exceeds 64 camera proxies')
    audit=dict(original_selected=len(exact),combined_camera_proxies=count,protected_floor_ids=floor_ids,
               subsumed=covered,retained_selected_ids=sorted(old_ids),occupied_union_preserved=True)
    metadata=dict(world.source_metadata,research_inference=ref.plain(models),research_coverage=audit,
                  camera_obstacle_count=count,original_camera_obstacle_count=len(exact),
                  snapshot_policy='research_exact_occupied_union_plus_inferred_boxes')
    return replace(world,obstacles=tuple(retained+descriptors),source_metadata=metadata),audit


def capture(backend,source,inputs,expected,scene,record,base_capture):
    started=time.perf_counter()
    try:
        view,payload=associate(source,inputs,expected)
        record['publication']=identity(view)
        reconstruction,_,_=ref.reconstruct(payload.frames,scene,payload.robot_snapshots)
        models=[dict(component=c['component'],**visibility.qualify(c,payload.frames)) for c in reconstruction['components']]
        record.update(reconstruction=reconstruction,models=models,
                      source_age_after_reconstruction_s=source.clock()-view.timestamp)
        qualified=[]
        for c,m in zip(reconstruction['components'],models):
            if c.get('reason')=='possible_robot_component':
                if c['robot_envelope_points']!=c['points']:fail('geometry_ambiguous','Robot-connected component')
                continue
            if m['status']!='qualified':fail('geometry_unqualified',f"Component {c['component']} is {m['status']}: {m['reason']}")
            qualified.append(m)
        if not qualified:fail('geometry_unqualified','No qualified nonrobot obstacle')
        world,frozen=base_capture(backend,lambda:source,max_obstacles=64)
        record['final_capture']=dict(publication=identity(frozen),source_age_s=frozen.age_seconds,
                                    evaluated_at=frozen.evaluated_at,original_timestamp=frozen.timestamp)
        if identity(frozen)!=identity(view) or not same_frames(payload.frames,frozen.metadata['source_frames']):
            fail('source_mismatch','Final source differs from reconstruction exposures')
        combined,audit=combine(world,frozen,qualified,scene.support_plane_z)
        record['coverage']=audit
        return combined,frozen
    except Exception as e:
        record['failure']=dict(code=getattr(e,'code',type(e).__name__),message=str(e))
        raise
    finally:
        record['capture_wall_s']=time.perf_counter()-started
