"""Geometric components with conservative primitive-supported refinement."""
from itertools import chain, product

import numpy as np


_NEIGHBORS = tuple(offset for offset in product((-1, 0, 1), repeat=3)
                   if offset > (0, 0, 0))


def _component_labels(count, edges):
    """Union a sparse graph using Python integer parents (no SDK/dependency)."""
    parent = list(range(count))
    for first, second in edges:
        while first != parent[first]:
            parent[first] = parent[parent[first]]
            first = parent[first]
        while second != parent[second]:
            parent[second] = parent[parent[second]]
            second = parent[second]
        if first != second:
            parent[second] = first
    for index in range(count):
        current = index
        while current != parent[current]:
            parent[current] = parent[parent[current]]
            current = parent[current]
        parent[index] = current
    return np.asarray(parent)


def _neighbor_cells(keys):
    """Vectorized 26-neighbor lookup, emitting each occupied cell pair once."""
    coordinates = keys-keys.min(0)+1
    dimensions = coordinates.max(0)+2
    strides = np.array([dimensions[1]*dimensions[2], dimensions[2], 1], dtype=np.int64)
    codes = coordinates @ strides
    for offset in _NEIGHBORS:
        targets = codes+np.asarray(offset) @ strides
        matches = np.searchsorted(codes, targets)
        first = np.flatnonzero(matches < len(codes))
        first = first[codes[matches[first]] == targets[first]]
        if len(first):
            yield first, matches[first]


def cluster_points(points, *, cell_size=.008, min_points=25):
    points = np.asarray(points, float).reshape(-1, 3)
    if not np.isfinite(cell_size) or cell_size <= 0 or min_points < 1:
        raise ValueError('Invalid clustering settings')
    if not len(points):
        return []
    keys, inverse = np.unique(np.floor(points/cell_size).astype(np.int64), axis=0, return_inverse=True)
    edges = ((int(first), int(second)) for left, right in _neighbor_cells(keys)
             for first, second in zip(left, right))
    labels = _component_labels(len(keys), edges)
    membership = labels[inverse]
    groups = [points[membership == label] for label in np.unique(membership)
              if np.count_nonzero(membership == label) >= min_points]
    return sorted(groups, key=lambda p: tuple(np.mean(p, axis=0)))


def _metric_components(points, radius=.004, min_points=25):
    """Exact radius graph; cell adjacency only proposes pairs, never an edge."""
    if not len(points):
        return []
    keys, inverse = np.unique(np.floor(points/radius).astype(np.int64), axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse)
    starts = np.r_[0, counts.cumsum()]
    # Fused clouds have bounded voxel density. Chunking also bounds temporary
    # work for unusually dense caller input without changing the radius graph.
    buckets = [order[starts[i]:starts[i+1]] for i in range(len(keys))]

    def edges():
        if counts.max() <= 32:
            padded = np.full((len(keys), counts.max()), -1, int)
            padded[inverse[order], np.arange(len(points))-np.repeat(starts[:-1], counts)] = order
            pairs = [(np.arange(len(keys)), np.arange(len(keys))), *_neighbor_cells(keys)]
            first = np.concatenate([left for left, _ in pairs])
            second = np.concatenate([right for _, right in pairs])
            for begin in range(0, len(first), 128):
                left, right = padded[first[begin:begin+128]], padded[second[begin:begin+128]]
                delta = points[left][:, :, None, :]-points[right][:, None, :, :]
                allowed = (left[:, :, None] >= 0) & (right[:, None, :] >= 0)
                allowed &= np.einsum('nijk,nijk->nij', delta, delta) <= radius*radius
                groups, rows, columns = np.nonzero(allowed)
                left_indices, right_indices = left[groups, rows], right[groups, columns]
                # Cell-pair direction is arbitrary; only self/duplicate edges
                # within the same cell need removing before sparse union.
                different = left_indices != right_indices
                yield from zip(left_indices[different].tolist(), right_indices[different].tolist())
            return
        adjacent = ((int(left), int(right)) for first, second in _neighbor_cells(keys)
                    for left, right in zip(first, second))
        for first_cell, second_cell in chain(zip(range(len(keys)), range(len(keys))), adjacent):
            first, second = buckets[first_cell], buckets[second_cell]
            for begin in range(0, len(first), 64):
                left = first[begin:begin+64]
                for end in range(0, len(second), 64):
                    right = second[end:end+64]
                    distances = ((points[left, None]-points[right])**2).sum(2)
                    rows, columns = np.nonzero(distances <= radius*radius)
                    for a, b in zip(left[rows].tolist(), right[columns].tolist()):
                        if first_cell != second_cell or a < b:
                            yield a, b
    membership = _component_labels(len(points), edges())
    groups = [points[membership == label] for label in np.unique(membership)
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


def refine_cluster(points, fitter, *, support_plane_z=0., parent_fit=None):
    """Accept metric cuts only when separate supported primitives explain them.

    Broad grouping preserves sparse single-view surfaces. A 4 mm metric graph
    merely proposes cuts: disconnected surface pieces are not automatically
    objects. Rejected cuts retain the entire original cloud and optional fit.
    An oversized background component may be partitioned from a supported
    primitive; otherwise at least two supported known primitives are required.
    Fits are returned with clouds so the caller never repeats accepted work.
    """
    original = [(points, parent_fit)]
    if len(points) < 50:
        return original
    fine = _metric_components(points)
    if not 2 <= len(fine) <= 16:
        return original
    _, parent_reason = supported_object_cloud(points, support_plane_z)
    potential, oversized = [], []
    for cloud in fine:
        supported, reason = supported_object_cloud(cloud, support_plane_z)
        if reason == 'oversized_or_robot_connected':
            oversized.append((cloud, None))
        elif supported or reason == 'no_observed_support_contact':
            potential.append((cloud, supported))
    potential_points = sum(len(cloud) for cloud, _ in potential)
    dense_points = potential_points+sum(len(cloud) for cloud, _ in oversized)
    possible_primitives = len(potential) >= 2 and potential_points >= .85*len(points)
    possible_background = (parent_reason == 'oversized_or_robot_connected' and potential
                           and oversized and dense_points >= .85*len(points))
    if not (possible_primitives or possible_background):
        return original
    valid = []
    for cloud, supported in potential:
        fit = fitter.fit(cloud, support_plane_z=support_plane_z)
        if fit.geometry.shape.value == 'unknown':
            continue
        inferred_contact = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                            abs(fit.center[2]-fit.size[2]/2-support_plane_z) <= .004)
        if supported or inferred_contact:
            valid.append((cloud, fit))
    coverage = sum(len(cloud) for cloud, _ in valid)/len(points)
    dense_coverage = (sum(len(cloud) for cloud, _ in valid+oversized)/len(points))
    primitive_partition = len(valid) >= 2 and coverage >= .85
    background_partition = (parent_reason == 'oversized_or_robot_connected' and
                            valid and oversized and dense_coverage >= .85)
    if not (primitive_partition or background_partition):
        return original
    # Nearly coincident fitted primitives indicate complementary surface pieces
    # of the same object, even when sampling disconnected their point graph.
    for index, (_, first) in enumerate(valid):
        for _, second in valid[index+1:]:
            separation = np.linalg.norm(first.center-second.center)
            if separation < .25*min(np.linalg.norm(first.size), np.linalg.norm(second.size)):
                return original
    return sorted(valid+oversized, key=lambda item: tuple(item[0].mean(0)))
