"""Global 3D connected components on a bounded voxel adjacency graph."""
from itertools import product

import numpy as np


def cluster_points(points, *, cell_size=.008, min_points=25):
    points = np.asarray(points, float).reshape(-1, 3)
    if not np.isfinite(cell_size) or cell_size <= 0 or min_points < 1:
        raise ValueError('Invalid clustering settings')
    if not len(points):
        return []
    keys, inverse = np.unique(np.floor(points/cell_size).astype(np.int64), axis=0, return_inverse=True)
    remaining = {tuple(key): index for index, key in enumerate(keys)}
    labels = np.full(len(keys), -1, int)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    group = 0
    while remaining:
        first, index = remaining.popitem()
        labels[index] = group
        stack = [first]
        while stack:
            current = stack.pop()
            for offset in offsets:
                neighbor = tuple(a+b for a, b in zip(current, offset))
                index = remaining.pop(neighbor, None)
                if index is not None:
                    labels[index] = group
                    stack.append(neighbor)
        group += 1
    membership = labels[inverse]
    groups = [points[membership == label] for label in range(group)
              if np.count_nonzero(membership == label) >= min_points]
    return sorted(groups, key=lambda p: tuple(np.mean(p, axis=0)))


def supported_object_cloud(points, support_plane_z):
    lo, hi = np.percentile(points, [1, 99], axis=0)
    extent = hi-lo
    if max(extent) > .18:
        return False, 'oversized_or_robot_connected'
    if lo[2] > support_plane_z+.014:
        return False, 'no_observed_support_contact'
    if max(extent[:2]) < .012 or hi[2]-support_plane_z < .010:
        return False, 'too_small'
    return True, None
