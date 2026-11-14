"""Fixed sensor-derived negative controls; no GT or extra camera captures."""
from dataclasses import replace
import numpy as np
from .reference import _EDGE_POSITIONS, _RIBBON_OFFSETS, _unique_pixels


def missing_lower(frames, points, model):
    """Remove observed lower-edge cloud strip and its calibrated image ribbons."""
    center, r, size = model['center'], model['rotation'], model['dimensions']
    local = (points-center)@r
    cropped = points[local[:, 2] > -size[2]/2+.008]
    lines = []
    for axis in (0, 1):
        other = 1-axis
        for sign in (-1, 1):
            q = np.zeros((len(_EDGE_POSITIONS), 3))
            q[:, 2], q[:, axis], q[:, other] = -size[2]/2, sign*size[axis]/2, _EDGE_POSITIONS*size[other]
            lines.append(center+q@r.T)
    world = np.concatenate(lines)
    altered = []
    removed = 0
    for f in frames:
        camera = (world-f.world_from_camera[:3, 3])@f.world_from_camera[:3, :3]
        k = f.intrinsics
        projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
        pixels = _unique_pixels((np.rint(projected).astype(int)[:, None]+_RIBBON_OFFSETS).reshape(-1, 2))
        h, w = f.depth.shape
        pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
        depth = f.depth.copy()
        depth[pixels[:, 1], pixels[:, 0]] = np.nan
        removed += len(pixels)
        altered.append(replace(f, depth=depth))
    return tuple(altered), cropped, removed


NAMES=('missing_lower_boundary','robot_returns','robot_connected','merged_objects','contradictory_depth')


def make(frames,points,model,snapshots):
    missing_frames,missing_points,_=missing_lower(frames,points,model)
    snapshot=snapshots[frames[0].camera_id]
    robot=np.concatenate([p.mesh.vertices@p.world_from_local[:3,:3].T+p.world_from_local[:3,3]
                          for p in snapshot.parts])
    robot=robot[np.linspace(0,len(robot)-1,min(len(robot),1200),dtype=int)]
    # An actual <=4 mm chain joins the observed component to a known robot vertex.
    count=int(np.ceil(np.linalg.norm(robot[0]-points[0])/.004))+1
    bridge=np.linspace(points[0],robot[0],count)
    connected=np.concatenate((points,bridge))
    if len(connected)>12000:
        raise ValueError('Connected-observation control exceeds existing cloud cap')
    axis=int(np.argmax(model['dimensions'][:2]))
    shift=.6*model['dimensions'][axis]*model['rotation'][:,axis]
    merged=np.concatenate((points,points+shift))
    conflicting=tuple(replace(f,depth=np.where(np.isfinite(f.depth)&(f.depth>0),
                                              f.depth+.05,f.depth)) for f in frames)
    return dict(missing_lower_boundary=(missing_points,missing_frames),
                robot_returns=(robot,frames),robot_connected=(connected,frames),
                merged_objects=(merged,frames),contradictory_depth=(points,conflicting))
