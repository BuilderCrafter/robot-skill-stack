"""Deterministic cell-neighbor and exact-radius components of 3D clouds."""
from collections.abc import Iterable
from itertools import chain, islice, product

import numpy as np


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


def radius_components(points: np.ndarray, radius: float = .004, min_points: int = 25) -> list[np.ndarray]:
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
