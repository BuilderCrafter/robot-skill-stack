"""Geometric components with conservative primitive-supported refinement."""
from collections.abc import Iterable, Sequence
from itertools import chain, product

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry
from .acquisition import DepthFrame
from .fitting import PrimitiveFit


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
    """Exact radius graph; cell adjacency only proposes pairs, never an edge."""
    if not len(points):
        return []
    keys, inverse = np.unique(np.floor(points / radius).astype(np.int64), axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse)
    starts = np.r_[0, counts.cumsum()]

    def edges():
        # Pad sparse cells for batching; dense duplicates need bounded chunks.
        if counts.max() <= 32:
            padded = np.full((len(keys), counts.max()), -1, int)
            padded[inverse[order], np.arange(len(points)) - np.repeat(starts[:-1], counts)] = order
            pairs = [(np.arange(len(keys)), np.arange(len(keys))), *_neighbor_cells(keys)]
            first = np.concatenate([left for left, _ in pairs])
            second = np.concatenate([right for _, right in pairs])
            for begin in range(0, len(first), 128):
                left, right = padded[first[begin:begin + 128]], padded[second[begin:begin + 128]]
                delta = points[left][:, :, None, :] - points[right][:, None, :, :]
                allowed = (left[:, :, None] >= 0) & (right[:, None, :] >= 0)
                allowed &= np.einsum('nijk,nijk->nij', delta, delta) <= radius * radius
                groups, rows, columns = np.nonzero(allowed)
                left_indices, right_indices = left[groups, rows], right[groups, columns]
                # Cell-pair direction is arbitrary; only self/duplicate edges
                # within the same cell need removing before sparse union.
                different = left_indices != right_indices
                yield from zip(left_indices[different].tolist(), right_indices[different].tolist())
            return
        buckets = [order[starts[i]:starts[i + 1]] for i in range(len(keys))]
        adjacent = ((int(left), int(right)) for first, second in _neighbor_cells(keys)
                    for left, right in zip(first, second))
        for first_cell, second_cell in chain(zip(range(len(keys)), range(len(keys))), adjacent):
            first, second = buckets[first_cell], buckets[second_cell]
            for begin in range(0, len(first), 64):
                left = first[begin:begin + 64]
                for end in range(0, len(second), 64):
                    right = second[end:end + 64]
                    distances = ((points[left, None] - points[right]) ** 2).sum(2)
                    rows, columns = np.nonzero(distances <= radius * radius)
                    for a, b in zip(left[rows].tolist(), right[columns].tolist()):
                        if first_cell != second_cell or a < b:
                            yield a, b
    membership = _component_labels(len(points), edges())
    labels, counts = np.unique(membership, return_counts=True)
    groups = [points[membership == label] for label in labels[counts >= min_points]]
    return sorted(groups, key=lambda p: tuple(np.mean(p, axis=0)))


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


def _surface_distance(points: np.ndarray, fit: PrimitiveFit, *, signed: bool = False) -> np.ndarray:
    """Measure point distances to a fitted primitive's surface."""
    geometry = fit.geometry
    delta = points - fit.center
    if geometry.shape.value == 'sphere':
        value = np.linalg.norm(delta, axis=1) - geometry.radius
        return value if signed else abs(value)
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
    return value if signed else abs(value)


def _footprint(fit: PrimitiveFit) -> np.ndarray:
    """Approximate a fitted primitive's horizontal footprint as a polygon."""
    geometry = fit.geometry
    if geometry.shape.value == 'sphere' or (geometry.shape.value == 'cylinder' and abs(geometry.axis[2]) > .99):
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        return fit.center[:2] + geometry.radius / np.cos(np.pi / 32) * np.c_[np.cos(angles), np.sin(angles)]
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if geometry.shape.value == 'cube':
        c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
        rotation = np.array([[c, -s], [s, c]])
        local = corners * geometry.box_size[:2] / 2
    else:
        direction = geometry.axis[:2] / np.linalg.norm(geometry.axis[:2])
        rotation = np.c_[direction, [-direction[1], direction[0]]]
        local = corners * np.array([geometry.length / 2, geometry.radius])
    return fit.center[:2] + local @ rotation.T


def _separating_axis(a: np.ndarray, b: np.ndarray) -> tuple:
    """Return the largest signed gap and separating axis of two convex XY polygons."""
    best = None
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.c_[-edges[:, 1], edges[:, 0]]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        pa, pb = a @ normals.T, b @ normals.T
        forward = pb.min(0) - pa.max(0)
        backward = pa.min(0) - pb.max(0)
        separation = np.maximum(forward, backward)
        index = int(np.argmax(separation))
        direction = normals[index] if forward[index] >= backward[index] else -normals[index]
        lo, hi = float((a @ direction).max()), float((b @ direction).min())
        if best is None or hi - lo > best[0]:
            best = hi - lo, direction, lo, hi
    return best


def _distinct_models(fits: Sequence[PrimitiveFit]) -> bool:
    """Check whether fitted primitives have sufficiently separate footprints."""
    polygons = [_footprint(fit) for fit in fits]
    return all(_separating_axis(first, second)[0] >= -.001
               for index, first in enumerate(polygons) for second in polygons[index + 1:])


def _partition_view_evidence(parent: PrimitiveFit, children: Sequence[PrimitiveFit],
                             frames: Sequence[DepthFrame], plane: float) -> tuple[bool, bool]:
    """Return FREE-gap and OCCUPIED-bridge decisions from current calibrated rays.

    Samples lie inside the parent and between child footprints. Missing depth
    proves neither conclusion; conflicting observations also qualify neither.
    """
    lo, hi = parent.center[:2] - parent.size[:2] / 2, parent.center[:2] + parent.size[:2] / 2
    x, y = np.meshgrid(*[np.linspace(a, b, min(64, max(3, int((b - a) / .002) + 1)))
                        for a, b in zip(lo, hi)])
    height = parent.center[2] + parent.size[2] / 2 - max(.0015, .03 * parent.size[2])
    samples = np.c_[x.ravel(), y.ravel(), np.full(x.size, height)]
    samples = samples[_surface_distance(samples, parent, signed=True) < -.001]
    corridors = []
    polygons = [_footprint(fit) for fit in children]
    for index, first in enumerate(polygons):
        for second in polygons[index + 1:]:
            _, direction, lo, hi = _separating_axis(first, second)
            across = np.array([-direction[1], direction[0]])
            a, b = first @ across, second @ across
            corridors.append((direction, across, lo + .0015, hi - .0015,
                              max(a.min(), b.min()) + .001, min(a.max(), b.max()) - .001))

    def corridor_mask(points):
        keep = np.zeros(len(points), bool)
        for direction, across, lo, hi, lower, upper in corridors:
            q, t = points[:, :2] @ direction, points[:, :2] @ across
            keep |= ((q > lo) & (q < hi) & (t > lower) & (t < upper) & (points[:, 2] > plane + .008))
        return keep

    samples = samples[corridor_mask(samples)]
    free_seen = occupied_seen = False
    for frame in frames:
        camera = (samples - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        k = frame.intrinsics
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels = np.c_[k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2] - .5,
                           k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2] - .5]
        valid = np.isfinite(pixels).all(1) & (camera[:, 2] > 0)
        pixels = np.rint(np.where(np.isfinite(pixels), pixels, -1)).astype(int)
        x, y = pixels[:, 0], pixels[:, 1]
        valid &= (x >= 0) & (x < frame.depth.shape[1]) & (y >= 0) & (y < frame.depth.shape[0])
        selected = np.flatnonzero(valid)
        x, y = x[selected], y[selected]
        depth = frame.depth[y, x]
        rays = np.c_[(x + .5 - k[0, 2]) / k[0, 0], (y + .5 - k[1, 2]) / k[1, 1], np.ones(len(x))]
        world = (rays * depth[:, None]) @ frame.world_from_camera[:3, :3].T + frame.world_from_camera[:3, 3]
        finite = np.isfinite(depth) & (depth > 0)
        free = finite & (depth > camera[selected, 2] + .004) & (abs(world[:, 2] - plane) <= .004)
        occupied = finite & corridor_mask(world) & (_surface_distance(world, parent) <= .003)
        free_seen |= len(set(zip(x[free], y[free]))) >= 3
        occupied_seen |= len(set(zip(x[occupied], y[occupied]))) >= 3
    return free_seen and not occupied_seen, occupied_seen and not free_seen


def _refine_cluster_with_seeds(points: np.ndarray, fitter, fine: Sequence[np.ndarray], *,
                               support_plane_z: float, parent_fit: PrimitiveFit | None,
                               frames: Sequence[DepthFrame]) -> list[tuple[np.ndarray, PrimitiveFit | None]] | None:
    """Return a final partition, or None when this attempt is undecided.

    Original samples are assigned to supported child surfaces. A competing single
    solid needs current FREE-gap evidence; a bridge or ambiguity is final.
    """
    if not 2 <= len(fine) <= 8:
        return None
    _, parent_reason = supported_object_cloud(points, support_plane_z)
    potential, oversized = [], []
    for cloud in fine:
        supported, reason = supported_object_cloud(cloud, support_plane_z)
        if reason == 'oversized_or_robot_connected':
            oversized.append((cloud, None))
        elif supported or reason == 'no_observed_support_contact':
            potential.append((cloud, supported))
    potential_points = sum(len(cloud) for cloud, _ in potential)
    dense_points = potential_points + sum(len(cloud) for cloud, _ in oversized)
    possible_primitives = len(potential) >= 2 and potential_points >= .60 * len(points)
    possible_background = (parent_reason == 'oversized_or_robot_connected' and potential
                           and oversized and dense_points >= .85 * len(points))
    if not (possible_primitives or possible_background):
        return None
    required_fits = len(potential) + 1 if possible_background else 2 * len(potential) + 1
    if getattr(fitter, 'remaining', required_fits) < required_fits:
        return None
    valid = []
    for cloud, supported in potential:
        fit = fitter.fit(cloud, support_plane_z=support_plane_z)
        if fit.geometry.shape.value == 'unknown':
            continue
        inferred_contact = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                            abs(fit.center[2] - fit.size[2] / 2 - support_plane_z) <= .004)
        if supported or inferred_contact:
            valid.append((cloud, fit))
    dense_coverage = sum(len(cloud) for cloud, _ in valid + oversized) / len(points)
    background_partition = (parent_reason == 'oversized_or_robot_connected' and
                            valid and oversized and dense_coverage >= .85)
    if not _distinct_models([fit for _, fit in valid]):
        return None
    if background_partition:
        return sorted(valid + oversized, key=lambda item: tuple(item[0].mean(0)))
    if len(valid) < 2 or len(valid) != len(potential):
        return None
    distances = np.column_stack([_surface_distance(points, fit) for _, fit in valid])
    ranked = np.sort(distances, axis=1)
    labels = distances.argmin(1)
    labels[(ranked[:, 0] > .003) | (ranked[:, 1] - ranked[:, 0] < .0005)] = -1
    assigned = [points[labels == index] for index in range(len(valid))]
    if min(map(len, assigned)) < 30 or sum(map(len, assigned)) < .90 * len(points):
        return None
    refined = []
    for cloud, (seed, seed_fit) in zip(assigned, valid):
        fit = seed_fit if np.array_equal(cloud, seed) else fitter.fit(cloud, support_plane_z=support_plane_z)
        supported, _ = supported_object_cloud(cloud, support_plane_z)
        inferred = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                    abs(fit.center[2] - fit.size[2] / 2 - support_plane_z) <= .004)
        if (fit.geometry.shape != seed_fit.geometry.shape or not (supported or inferred) or
                np.percentile(_surface_distance(cloud, fit), 95) > .003):
            return None
        refined.append((cloud, fit))
    if not _distinct_models([fit for _, fit in refined]):
        return None
    parent_fit = parent_fit or fitter.fit(points, support_plane_z=support_plane_z)
    if (parent_fit.geometry.shape.value != 'unknown' and
            np.mean(_surface_distance(points, parent_fit) <= .003) >= .95):
        free_gap, occupied_bridge = _partition_view_evidence(
            parent_fit, [fit for _, fit in refined], frames, support_plane_z)
        if occupied_bridge:
            return [(points, parent_fit)]
        if not free_gap:
            geometry = PrimitiveGeometry('unknown', parent_fit.geometry.confidence,
                scores=dict(parent_fit.geometry.scores), metadata={'reason': 'ambiguous_single_or_multiple_surfaces'})
            return [(points, PrimitiveFit(geometry, parent_fit.center, parent_fit.size))]
    return sorted(refined, key=lambda item: tuple(item[0].mean(0)))


def refine_cluster(points: np.ndarray, fitter, *, support_plane_z: float = 0.,
                   parent_fit: PrimitiveFit | None = None,
                   frames: Sequence[DepthFrame] = ()) -> list[tuple[np.ndarray, PrimitiveFit | None]]:
    """Try 4 mm components, then one guarded 4.5 mm recovery on the same fit budget.

    Accepted splits, measured bridges and explicit ambiguities are final.
    """
    if len(points) >= 50:
        for radius in (.004, .0045):
            fine = _metric_components(points, radius=radius)
            result = _refine_cluster_with_seeds(points, fitter, fine, support_plane_z=support_plane_z,
                parent_fit=parent_fit, frames=frames)
            if result is not None:
                return result
            if (not 2 <= len(fine) <= 8 or getattr(fitter, 'remaining', 5) < 5 or
                    parent_fit is not None and parent_fit.geometry.metadata.get('reason') ==
                    'ambiguous_single_or_multiple_surfaces'):
                break
    return [(points, parent_fit)]
