"""Research-only organized-depth proposals and free-space split verification."""
import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import project_depth
from robot_skill_stack.world.perception_multiview.segmentation import _component_labels


def load_frames(directory, index=2):
    result = []
    for camera in range(2):
        path = directory/f'frame_{index:02d}_camera_{camera}.npz'
        if not path.exists():
            path = directory/f'camera_{camera}.npz'
        with np.load(path, allow_pickle=False) as data:
            token = data['token'] if 'token' in data else data['render_token']
            result.append(DepthFrame(str(camera), tuple(token.tolist()), data['depth'], data['intrinsics'],
                                    data['world_from_camera'], float(data['acquired_at'])))
    return result


def organized_segments(frame, scene, *, stride=2, jump=.008, max_distance=.012):
    points, pixels = project_depth(frame, stride=stride)
    valid = ((points >= scene.workspace_min).all(1) & (points <= scene.workspace_max).all(1) &
             (points[:, 2] > scene.support_plane_z+.004))
    points, pixels = points[valid], pixels[valid]
    width = (frame.depth.shape[1]+stride-1)//stride
    codes = pixels[:, 0]//stride*width+pixels[:, 1]//stride
    z = frame.depth[pixels[:, 0], pixels[:, 1]]
    pairs = []
    for offset in (1, width):
        target = codes+offset
        second = np.searchsorted(codes, target)
        first = np.flatnonzero(second < len(codes))
        first = first[codes[second[first]] == target[first]]
        if offset == 1:
            first = first[codes[first]//width == codes[second[first]]//width]
        allowed = ((abs(z[first]-z[second[first]]) <= jump) &
                   (np.linalg.norm(points[first]-points[second[first]], axis=1) <= max_distance))
        pairs.extend(zip(first[allowed].tolist(), second[first[allowed]].tolist()))
    labels = _component_labels(len(points), pairs)
    return [points[labels == value] for value in np.unique(labels) if np.count_nonzero(labels == value) >= 30]


def gap_witness(parent, children, frames, footprint, residual, *, plane=0.):
    """Count independent valid table pixels contradicting the one-solid model."""
    lo, hi = parent.center[:2]-parent.size[:2]/2, parent.center[:2]+parent.size[:2]/2
    x, y = np.meshgrid(*[np.linspace(a, b, min(64, max(3, int((b-a)/.002)+1))) for a, b in zip(lo, hi)])
    height = parent.center[2]+parent.size[2]/2-max(.0015, .03*parent.size[2])
    samples = np.c_[x.ravel(), y.ravel(), np.full(x.size, height)]
    samples = samples[residual(samples, parent, signed=True) < -.001]
    corridors = []
    for index, first in enumerate(children):
        for second in children[index+1:]:
            a, b = footprint(first), footprint(second)
            directions = []
            for polygon in (a, b):
                edges = np.roll(polygon, -1, axis=0)-polygon
                normals = np.c_[-edges[:, 1], edges[:, 0]]
                directions.extend(normals/np.linalg.norm(normals, axis=1)[:, None])
            best = None
            for direction in directions:
                pa, pb = a @ direction, b @ direction
                lo, hi = pa.max(), pb.min()
                if hi < lo:
                    lo, hi = pb.max(), pa.min()
                if best is None or hi-lo > best[0]:
                    best = hi-lo, direction, lo, hi
            separation, direction, lo, hi = best
            if separation <= .001:
                continue
            across = np.array([-direction[1], direction[0]])
            lower = max((a @ across).min(), (b @ across).min())+.001
            upper = min((a @ across).max(), (b @ across).max())-.001
            corridors.append((direction, across, lo+.0015, hi-.0015, lower, upper))
    def in_corridor(points):
        mask = np.zeros(len(points), bool)
        for direction, across, lo, hi, lower, upper in corridors:
            q, t = points[:, :2] @ direction, points[:, :2] @ across
            mask |= (q > lo) & (q < hi) & (t > lower) & (t < upper) & (points[:, 2] > plane+.008)
        return mask
    samples = samples[in_corridor(samples)]
    witnesses = []
    for frame in frames:
        camera = (samples-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        k = frame.intrinsics
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels = np.c_[k[0, 0]*camera[:, 0]/camera[:, 2]+k[0, 2]-.5,
                           k[1, 1]*camera[:, 1]/camera[:, 2]+k[1, 2]-.5]
        valid = np.isfinite(pixels).all(1) & (camera[:, 2] > 0)
        rounded = np.rint(np.where(np.isfinite(pixels), pixels, -1)).astype(int)
        x, y = rounded[:, 0], rounded[:, 1]
        valid &= (x >= 0) & (x < frame.depth.shape[1]) & (y >= 0) & (y < frame.depth.shape[0])
        selected = np.flatnonzero(valid)
        x, y = x[selected], y[selected]
        measured = frame.depth[y, x]
        ray = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))]
        world = (ray*measured[:, None]) @ frame.world_from_camera[:3, :3].T+frame.world_from_camera[:3, 3]
        agrees = (np.isfinite(measured) & (measured > camera[selected, 2]+.004) & (abs(world[:, 2]-plane) <= .004))
        occupied = (np.isfinite(measured) & in_corridor(world) & (residual(world, parent) <= .003))
        witnesses.append(dict(camera=frame.camera_id, pixels=len(set(zip(x[agrees].tolist(), y[agrees].tolist()))),
                              occupied_pixels=len(set(zip(x[occupied].tolist(), y[occupied].tolist())))))
    free = any(row['pixels'] >= 3 for row in witnesses)
    occupied = any(row['occupied_pixels'] >= 3 for row in witnesses)
    return dict(parent_samples=len(samples), cameras=witnesses,
                physical_gap_supported=free and not occupied,
                occupied_bridge_supported=occupied and not free)
