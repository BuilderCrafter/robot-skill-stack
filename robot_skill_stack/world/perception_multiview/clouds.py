"""Bounded, deterministic NumPy point-cloud operations in metres."""
from pathlib import Path
from itertools import product

import numpy as np


def project_depth(frame, *, stride=2, pixel_offset=.5, bounds=None):
    """Image-plane depth, ROS camera axes (+X right, +Y down, +Z forward).

    Half-pixel centres match Isaac 5.1 get_pointcloud's depth fallback. Return
    explicit image indices; never infer annotator point ordering from a mask.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError('stride must be a positive integer')
    if bounds is None:
        y, x = np.indices(frame.depth[::stride, ::stride].shape)
        y, x = y.ravel()*stride, x.ravel()*stride
    else:
        lo, hi = np.asarray(bounds, float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo > hi):
            raise ValueError('Projection bounds must be finite ordered XYZ corners')
        corners = np.array(list(product(*zip(lo, hi))))
        camera = (corners-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        if np.any(camera[:, 2] <= 0):
            # A box crossing the optical plane has no finite image rectangle.
            return project_depth(frame, stride=stride, pixel_offset=pixel_offset)
        k = frame.intrinsics
        pixels = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-pixel_offset
        first = np.floor(pixels.min(0)/stride).astype(int)*stride-stride
        last = np.ceil(pixels.max(0)/stride).astype(int)*stride+stride
        height, width = frame.depth.shape
        columns = np.arange(max(0, first[0]), min(width, last[0]+1), stride)
        rows = np.arange(max(0, first[1]), min(height, last[1]+1), stride)
        y, x = np.meshgrid(rows, columns, indexing='ij')
        y, x = y.ravel(), x.ravel()
    z = frame.depth[y, x]
    valid = np.isfinite(z) & (z > 0)
    y, x, z = y[valid], x[valid], z[valid]
    k = frame.intrinsics
    rays = np.column_stack(((x+pixel_offset-k[0, 2])/k[0, 0],
                            (y+pixel_offset-k[1, 2])/k[1, 1], np.ones(len(z))))
    points = (rays*z[:, None]) @ frame.world_from_camera[:3, :3].T + frame.world_from_camera[:3, 3]
    return points, np.column_stack((y, x))


def crop_workspace(points, scene, *, table_clearance=.004):
    p = np.asarray(points, float).reshape(-1, 3)
    keep = (np.isfinite(p).all(1) & (p >= scene.workspace_min).all(1) &
            (p <= scene.workspace_max).all(1) & (p[:, 2] > scene.support_plane_z+table_clearance))
    return p[keep]


def voxelize(points, voxel=.002, max_points=6000):
    if not np.isfinite(voxel) or voxel <= 0 or max_points < 1:
        raise ValueError('voxel and max_points must be positive')
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        return points
    keys, inverse = np.unique(np.floor(points/voxel+1e-9).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.column_stack([np.bincount(inverse, weights=points[:, i])/counts for i in range(3)])
    if len(result) > max_points:
        result = result[np.linspace(0, len(result)-1, max_points, dtype=int)]
    return result


def fuse_clouds(clouds, *, voxel=.002, max_points=6000):
    # First equalize density per camera, then suppress cross-camera duplicates.
    balanced = [voxelize(p, voxel, max_points) for p in clouds]
    return voxelize(np.concatenate(balanced) if balanced else np.empty((0, 3)), voxel, max_points)


def save_ply(path, points):
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                     'property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(stream, points, fmt='%.7f')
