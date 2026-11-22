"""Geometric components with conservative primitive-supported refinement."""
from collections.abc import Iterable, Sequence
from itertools import chain, islice, product

import numpy as np

from .fitting import PrimitiveFit, unknown_fit


_NEIGHBORS = tuple(offset for offset in product((-1, 0, 1), repeat=3)
                   if offset > (0, 0, 0))


def _component_labels(count: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
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


def _neighbor_cells(keys: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Vectorized 26-neighbor lookup, emitting each occupied cell pair once."""
    coordinates = keys - keys.min(0) + 1
    dimensions = coordinates.max(0) + 2
    strides = np.array([dimensions[1] * dimensions[2], dimensions[2], 1], dtype=np.int64)
    codes = coordinates @ strides
    for offset in _NEIGHBORS:
        targets = codes + np.asarray(offset) @ strides
        matches = np.searchsorted(codes, targets)
        first = np.flatnonzero(matches < len(codes))
        first = first[codes[matches[first]] == targets[first]]
        if len(first):
            yield first, matches[first]


def cluster_points(points: np.ndarray, *, cell_size: float = .008, min_points: int = 25) -> list[np.ndarray]:
    """Group neighboring occupied 3D cells into candidate object clouds."""
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points):
        return []
    keys, inverse = np.unique(np.floor(points / cell_size).astype(np.int64), axis=0, return_inverse=True)
    edges = ((int(first), int(second)) for left, right in _neighbor_cells(keys)
             for first, second in zip(left, right))
    labels = _component_labels(len(keys), edges)
    membership = labels[inverse]
    labels, counts = np.unique(membership, return_counts=True)
    groups = [points[membership == label] for label in labels[counts >= min_points]]
    return sorted(groups, key=lambda p: tuple(np.mean(p, axis=0)))


def _metric_components(points: np.ndarray, radius: float = .004, min_points: int = 25) -> list[np.ndarray]:
    """Exact radius connectivity in fixed-size blocks, for sparse or dense cells."""
    if not len(points):
        return []
    keys, inverse, counts = np.unique(np.floor(points / radius).astype(np.int64),
                                     axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    width = min(32, counts.max())
    within_cell = np.arange(len(points)) - np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
    block_offsets = np.r_[0, np.cumsum((counts + width - 1) // width)]
    blocks = np.full((block_offsets[-1], width), -1, dtype=int)
    blocks[np.repeat(block_offsets[:-1], counts) + within_cell // width, within_cell % width] = order

    def edges():
        adjacent = ((int(a), int(b)) for left, right in _neighbor_cells(keys) for a, b in zip(left, right))
        cells = chain(zip(range(len(keys)), range(len(keys))), adjacent)
        pairs = ((a, b) for i, j in cells for a in range(block_offsets[i], block_offsets[i + 1])
                 for b in range(block_offsets[j], block_offsets[j + 1]) if i != j or a <= b)
        while packet := list(islice(pairs, 128)):
            left, right = blocks[np.asarray(packet).T]
            delta = points[left][:, :, None, :] - points[right][:, None, :, :]
            connected = (left[:, :, None] >= 0) & (right[:, None, :] >= 0)
            connected &= np.einsum('nijk,nijk->nij', delta, delta) <= radius * radius
            groups, rows, columns = np.nonzero(connected)
            first, second = left[groups, rows], right[groups, columns]
            distinct = first != second
            yield from zip(first[distinct].tolist(), second[distinct].tolist())

    membership = _component_labels(len(points), edges())
    labels, counts = np.unique(membership, return_counts=True)
    groups = [points[membership == label] for label in labels[counts >= min_points]]
    return sorted(groups, key=lambda cloud: tuple(cloud.mean(0)))


def supported_object_cloud(points: np.ndarray, support_plane_z: float) -> tuple[bool, str | None]:
    """Check whether a cluster has enough points, plausible size, and table contact."""
    if len(points) < 30:
        return False, 'unsupported_geometry'
    lo, hi = np.percentile(points, [1, 99], axis=0)
    extent = hi - lo
    if max(extent) > .18:
        return False, 'oversized_or_robot_connected'
    if lo[2] > support_plane_z + .014:
        return False, 'no_observed_support_contact'
    if max(extent[:2]) < .012 or hi[2] - support_plane_z < .010:
        return False, 'unsupported_geometry'
    return True, None


def _surface_distance(points: np.ndarray, fit: PrimitiveFit) -> np.ndarray:
    """Measure point distances to a fitted primitive's surface."""
    geometry = fit.geometry
    delta = points - fit.center
    if geometry.shape.value == 'sphere':
        value = np.linalg.norm(delta, axis=1) - geometry.radius
        return abs(value)
    if geometry.shape.value == 'cube':
        c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        q = abs(delta @ rotation) - geometry.box_size / 2
    elif geometry.shape.value == 'cylinder':
        along = delta @ geometry.axis
        q = np.c_[np.linalg.norm(delta - along[:, None] * geometry.axis, axis=1) - geometry.radius,
                  abs(along) - geometry.length / 2]
    else:
        return np.full(len(points), np.inf)
    value = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return abs(value)


def _distinct_models(fits: Sequence[PrimitiveFit]) -> bool:
    """Require separated sphere disks or conservative world-axis XY bounds.

    Overlapping AABBs of rotated boxes/cylinders stay unresolved, even when a
    more specialized intersection test could separate their hidden geometry.
    """
    for i, first in enumerate(fits):
        for second in fits[i + 1:]:
            if first.geometry.shape.value == second.geometry.shape.value == 'sphere':
                gap = np.linalg.norm(first.center[:2] - second.center[:2])
                separate = gap >= first.geometry.radius + second.geometry.radius - .001
            else:
                gap = abs(first.center[:2] - second.center[:2]) - (first.size[:2] + second.size[:2]) / 2
                separate = np.any(gap >= -.001)
            if not separate:
                return False
    return True


def refine_cluster(points: np.ndarray, fitter, *, support_plane_z: float = 0.) -> list[tuple[np.ndarray, PrimitiveFit | None]]:
    """Split only well-covered, supported metric components with distinct models.

    No missing samples are reassigned or used to infer FREE space. A competing
    single-solid explanation, fragmented split or insufficient fit budget is
    unknown. Without split candidates, ordinary single-object fitting remains.
    """
    supported, reason = supported_object_cloud(points, support_plane_z)
    if len(points) < 50 or not supported and reason != 'no_observed_support_contact':
        return [(points, None)]
    components = _metric_components(points)
    if len(components) <= 1:
        return [(points, None)]
    coverage = sum(map(len, components)) / len(points)
    unresolved = [(points, unknown_fit(points, reason='unresolved_components'))]
    if not 2 <= len(components) <= 8 or coverage < .90:
        return unresolved
    # A direct SurfaceFitter is useful offline; live calls share the provider's cap.
    if getattr(fitter, 'remaining', len(components) + 1) < len(components) + 1:
        return unresolved
    fits = []
    for cloud in components:
        supported, reason = supported_object_cloud(cloud, support_plane_z)
        if not supported and reason != 'no_observed_support_contact':
            return unresolved
        fit = fitter.fit(cloud, support_plane_z=support_plane_z)
        contact = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                   abs(fit.center[2] - fit.size[2] / 2 - support_plane_z) <= .004)
        if (fit.geometry.shape.value == 'unknown' or not (supported or contact) or
                np.percentile(_surface_distance(cloud, fit), 95) > .003):
            return unresolved
        fits.append(fit)
    if not _distinct_models(fits):
        return unresolved
    parent = fitter.fit(points, support_plane_z=support_plane_z)
    if parent.geometry.shape.value != 'unknown' and np.mean(_surface_distance(points, parent) <= .003) >= .95:
        return unresolved
    return list(zip(components, fits))
