"""Oriented box surface distances and batched yaw search."""
import numpy as np


def box_surface_distance(points: np.ndarray, center: np.ndarray, rotation: np.ndarray,
                         size: np.ndarray) -> np.ndarray:
    """Measure each point's distance from an oriented box surface."""
    q = abs((points-center) @ rotation)-size/2
    return abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))


def search_box_yaw(points: np.ndarray, side_mask: np.ndarray, angles: np.ndarray) -> tuple:
    """Transform each yaw once; score side samples within full-cloud bounds."""
    c, s = np.cos(angles), np.sin(angles)
    xy = np.stack((points[:, 0, None]*c+points[:, 1, None]*s,
                   -points[:, 0, None]*s+points[:, 1, None]*c), axis=2)
    lo, hi = np.percentile(xy, [.1, 99.9], axis=0)
    local_sides = xy[side_mask]
    error = np.percentile(np.minimum(abs(local_sides-lo), abs(local_sides-hi)).min(2), 75, axis=0)
    index = int(np.argmin(error))
    yaw = float(angles[index])
    r = np.array([[c[index], -s[index], 0.], [s[index], c[index], 0.], [0., 0., 1.]])
    return yaw, r, lo[index], hi[index]
