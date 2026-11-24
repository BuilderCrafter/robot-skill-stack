"""Bounded, deterministic NumPy point-cloud operations in metres."""
from collections.abc import Sequence
from itertools import product
import math

import numpy as np

from .camera import DepthFrame


def project_depth(frame: DepthFrame, *, stride: int = 2, pixel_offset: float = .5,
                  bounds: Sequence | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Image-plane depth, ROS camera axes (+X right, +Y down, +Z forward).

    Half-pixel centres match Isaac 5.1 get_pointcloud's depth fallback. Return
    world points and their (row, column) indices. Bounds restrict the image
    window; callers still crop XYZ to the workspace. Configuration supplies
    ordered finite bounds and a positive integer stride.
    """
    intrinsics, transform = frame.intrinsics, frame.world_from_camera
    if bounds is None:
        y, x = np.indices(frame.depth[::stride, ::stride].shape)
        y, x = y.ravel() * stride, x.ravel() * stride
    else:
        lo, hi = np.asarray(bounds, float)
        corners = np.array(list(product(*zip(lo, hi))))
        camera_points = (corners - transform[:3, 3]) @ transform[:3, :3]
        if np.any(camera_points[:, 2] <= 0):
            # A box crossing the optical plane has no finite image rectangle.
            return project_depth(frame, stride=stride, pixel_offset=pixel_offset)
        pixels = (camera_points[:, :2] / camera_points[:, 2, None]
                  * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2] - pixel_offset)
        first = np.floor(pixels.min(0) / stride).astype(int) * stride - stride
        last = np.ceil(pixels.max(0) / stride).astype(int) * stride + stride
        height, width = frame.depth.shape
        columns = np.arange(max(0, first[0]), min(width, last[0] + 1), stride)
        rows = np.arange(max(0, first[1]), min(height, last[1] + 1), stride)
        y, x = np.meshgrid(rows, columns, indexing='ij')
        y, x = y.ravel(), x.ravel()
    z = frame.depth[y, x]
    valid = np.isfinite(z) & (z > 0)
    y, x, z = y[valid], x[valid], z[valid]
    rays = np.column_stack(((x + pixel_offset - intrinsics[0, 2]) / intrinsics[0, 0],
                            (y + pixel_offset - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(len(z))))
    points = (rays * z[:, None]) @ transform[:3, :3].T + transform[:3, 3]
    return points, np.column_stack((y, x))


def crop_workspace(points: np.ndarray, scene, *, table_clearance: float = .004) -> np.ndarray:
    """Keep finite world points inside the workspace and above the support plane."""
    points = np.asarray(points, float).reshape(-1, 3)
    keep = (np.isfinite(points).all(1) & (points >= scene.workspace_min).all(1) &
            (points <= scene.workspace_max).all(1) & (points[:, 2] > scene.support_plane_z + table_clearance))
    return points[keep]


def voxelize(points: np.ndarray, voxel: float = .002, max_points: int = 6000) -> np.ndarray:
    """Return finite-point voxel centroids in XYZ order, uniformly capped by index."""
    if not np.isfinite(voxel) or voxel <= 0 or max_points < 1:
        raise ValueError('voxel and max_points must be positive')
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        return points
    keys = np.floor(points / voxel + 1e-9).astype(np.int64)
    lower, upper = keys.min(0), keys.max(0)
    widths = tuple(int(hi) - int(lo) + 1 for lo, hi in zip(lower, upper))
    if math.prod(widths) <= np.iinfo(np.int64).max:
        # Checked mixed-radix keys preserve XYZ lexicographic group order.
        # Keep accumulation in original point order for identical centroids.
        codes = keys[:, 0] - lower[0]
        codes *= widths[1]
        codes += keys[:, 1] - lower[1]
        codes *= widths[2]
        codes += keys[:, 2] - lower[2]
        _, inverse = np.unique(codes, return_inverse=True)
    else:
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.column_stack([np.bincount(inverse, weights=points[:, i]) / counts for i in range(3)])
    if len(result) > max_points:
        result = result[np.linspace(0, len(result) - 1, max_points, dtype=int)]
    return result


def fuse_clouds(clouds: Sequence[np.ndarray], *, voxel: float = .002,
                max_points: int = 6000) -> np.ndarray:
    """Balance camera point densities, then merge redundant spatial samples."""
    # First equalize density per camera, then suppress cross-camera duplicates.
    balanced = [voxelize(p, voxel, max_points) for p in clouds]
    return voxelize(np.concatenate(balanced) if balanced else np.empty((0, 3)), voxel, max_points)
