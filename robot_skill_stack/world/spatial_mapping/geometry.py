"""SDK-free conservative voxel overlap for prepared affine scene geometry."""
from dataclasses import dataclass

import numpy as np

from .occupancy import GridSpec, _readonly


@dataclass(frozen=True)
class GeometryItem:
    """Owned worker geometry; a missing kind explicitly denotes a world AABB."""
    lower: np.ndarray
    upper: np.ndarray
    kind: str | None = None
    inverse: np.ndarray | None = None
    dimensions: np.ndarray | None = None
    radius: float | None = None
    height: float | None = None
    axis: int = 2

    def __post_init__(self) -> None:
        for name in ('lower', 'upper', 'inverse', 'dimensions'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))


def _primitive_item(kind: str, transform: np.ndarray, *, dimensions=None,
                    radius=None, height=None, axis: int = 2) -> GeometryItem:
    """Prepare world bounds and a cached inverse without retaining the pose twice."""
    linear, center = transform[:3, :3], transform[:3, 3]
    if kind == 'cube':
        dimensions = np.asarray(dimensions)
        extent = abs(linear) @ (dimensions / 2)
    elif kind == 'sphere':
        extent = radius * np.linalg.norm(linear, axis=1)
    elif kind == 'cylinder':
        radial = [i for i in range(3) if i != axis]
        extent = abs(linear[:, axis]) * height / 2 + radius * np.linalg.norm(linear[:, radial], axis=1)
    else:
        raise ValueError('Unsupported primitive: ' + kind)
    return GeometryItem(center - extent, center + extent, kind, np.linalg.inv(transform),
                        dimensions, radius, height, axis)


@dataclass(frozen=True)
class GroundTruthInput:
    """Original source time and independently owned scene geometry."""
    timestamp: float
    items: tuple[GeometryItem, ...]


def _rasterize(spec: GridSpec, item: GeometryItem, budget: int) -> tuple[np.ndarray, int]:
    """Filter bounded chunks of candidate cells by conservative local overlap."""
    if np.any(item.upper < spec.bounds_min) or np.any(item.lower > spec.bounds_max):
        return np.empty((0, 3), dtype=np.int64), 0
    # Include boundary-touching voxels, including shortened final cells.
    low = np.maximum(np.floor((item.lower - spec.bounds_min) / spec.voxel_size - 1e-9), 0).astype(int)
    high = np.minimum(np.floor((item.upper - spec.bounds_min) / spec.voxel_size + 1e-9),
                      np.asarray(spec.shape) - 1).astype(int)
    count = int(np.prod(high - low + 1, dtype=np.int64))
    if count > budget:
        raise RuntimeError('Ground-truth candidate voxel budget exceeded')
    if not count:
        return np.empty((0, 3), dtype=np.int64), 0
    chunks = []
    for start in range(0, count, 65536):
        indices = np.column_stack(np.unravel_index(np.arange(start, min(start + 65536, count)), high - low + 1)) + low
        if item.kind is not None:
            cell_lower = spec.bounds_min + indices * spec.voxel_size
            cell_upper = np.minimum(cell_lower + spec.voxel_size, spec.bounds_max)
            centers, half = (cell_lower + cell_upper) / 2, (cell_upper - cell_lower) / 2
            local = centers @ item.inverse[:3, :3].T + item.inverse[:3, 3]
            local_half = half @ abs(item.inverse[:3, :3]).T
            nearest = np.maximum(abs(local) - local_half, 0)
            if item.kind == 'cube':
                overlap = (nearest <= item.dimensions / 2 + 1e-12).all(1)
            elif item.kind == 'sphere':
                overlap = np.sum(nearest ** 2, axis=1) <= item.radius ** 2 + 1e-12
            else:
                radial = [i for i in range(3) if i != item.axis]
                overlap = ((nearest[:, item.axis] <= item.height / 2 + 1e-12) &
                           (np.sum(nearest[:, radial] ** 2, axis=1) <= item.radius ** 2 + 1e-12))
            indices = indices[overlap]
        chunks.append(indices)
    return np.concatenate(chunks), count


def rasterize_ground_truth(spec: GridSpec, prepared: GroundTruthInput,
                           max_candidate_voxels: int = 8_000_000) -> np.ndarray:
    """Rasterize the prepared scene into a deduplicated occupied-cell union."""
    occupied = np.zeros(spec.shape, bool)
    remaining = max_candidate_voxels
    for item in prepared.items:
        indices, count = _rasterize(spec, item, remaining)
        remaining -= count
        occupied[tuple(indices.T)] = True
    return np.argwhere(occupied)
