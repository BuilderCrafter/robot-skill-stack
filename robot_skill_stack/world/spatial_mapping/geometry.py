"""SDK-free conservative GT rasterization; affine primitive math retained unchanged."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .occupancy import _readonly


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


@dataclass(frozen=True)
class _Primitive:
    kind: str
    world_from_local: np.ndarray
    dimensions: np.ndarray | None = None
    radius: float | None = None
    height: float | None = None
    axis: int = 2

    def __post_init__(self):
        object.__setattr__(self, "world_from_local", _readonly(self.world_from_local))
        if self.dimensions is not None:
            object.__setattr__(self, "dimensions", _readonly(self.dimensions))


def _primitive_bounds(primitive):
    transform = np.asarray(primitive.world_from_local, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError('Geometry requires a finite affine world transform')
    linear, center = transform[:3, :3], transform[:3, 3]
    inverse = np.linalg.inv(transform)
    if primitive.kind == 'cube':
        dimensions = np.asarray(primitive.dimensions, dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('Cube dimensions must be finite and positive')
        extent = np.abs(linear) @ (dimensions/2)
    elif primitive.kind in ('sphere', 'cylinder'):
        if primitive.radius is None or not np.isfinite(primitive.radius) or primitive.radius <= 0:
            raise ValueError('Primitive radius must be finite and positive')
        if primitive.kind == 'sphere':
            extent = primitive.radius*np.linalg.norm(linear, axis=1)
        else:
            if (primitive.axis not in (0, 1, 2) or primitive.height is None or
                    not np.isfinite(primitive.height) or primitive.height <= 0):
                raise ValueError('Cylinder axis/height must be valid and positive')
            radial = [i for i in range(3) if i != primitive.axis]
            extent = (np.abs(linear[:, primitive.axis])*primitive.height/2 +
                      primitive.radius*np.linalg.norm(linear[:, radial], axis=1))
    else:
        raise ValueError(f'Unsupported analytic primitive: {primitive.kind}')
    return center-extent, center+extent, inverse


def _candidate_bounds(spec, lower, upper):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or
            not np.isfinite([lower, upper]).all() or np.any(lower > upper)):
        raise ValueError('Geometry has no finite world bounds')
    if np.any(upper < spec.bounds_min) or np.any(lower > spec.bounds_max):
        return None, None, 0
    # Include boundary-touching cells conservatively, including zero-thickness
    # unsupported geometry. The last map cell can be shorter than voxel_size.
    low = np.maximum(np.floor((lower-spec.bounds_min)/spec.voxel_size-1e-9), 0).astype(int)
    high = np.minimum(np.floor((upper-spec.bounds_min)/spec.voxel_size+1e-9),
                      np.asarray(spec.shape)-1).astype(int)
    count = int(np.prod(high-low+1, dtype=np.int64))
    return low, high, count


def _rasterize(spec, lower, upper, *, primitive=None, inverse=None, budget=8_000_000):
    low, high, count = _candidate_bounds(spec, lower, upper)
    if count > budget:
        raise RuntimeError(f'Ground-truth candidate voxel budget exceeded: {count} > {budget}')
    if not count:
        return np.empty((0, 3), dtype=np.int64), 0
    chunks = []
    for start in range(0, count, 65536):
        indices = (np.column_stack(np.unravel_index(
            np.arange(start, min(start+65536, count)), high-low+1))+low)
        if primitive is not None:
            cell_lower = spec.bounds_min+indices*spec.voxel_size
            cell_upper = np.minimum(cell_lower+spec.voxel_size, spec.bounds_max)
            centers, half = (cell_lower+cell_upper)/2, (cell_upper-cell_lower)/2
            local = centers @ inverse[:3, :3].T+inverse[:3, 3]
            local_half = half @ np.abs(inverse[:3, :3]).T
            nearest = np.maximum(np.abs(local)-local_half, 0)
            if primitive.kind == 'cube':
                overlap = np.all(nearest <= np.asarray(primitive.dimensions)/2+1e-12, axis=1)
            elif primitive.kind == 'sphere':
                overlap = np.sum(nearest**2, axis=1) <= primitive.radius**2+1e-12
            else:
                radial = [i for i in range(3) if i != primitive.axis]
                overlap = ((nearest[:, primitive.axis] <= primitive.height/2+1e-12) &
                           (np.sum(nearest[:, radial]**2, axis=1) <= primitive.radius**2+1e-12))
            indices = indices[overlap]
        chunks.append(indices)
    return np.concatenate(chunks), count


def primitive_voxel_indices(spec, kind, world_from_local, *, dimensions=None,
                            radius=None, height=None, axis='Z', max_candidates=8_000_000):
    """SDK-free conservative cell-volume rasterization of one affine primitive."""
    primitive = _Primitive(kind, np.asarray(world_from_local, dtype=float), dimensions,
                           radius, height, {'X': 0, 'Y': 1, 'Z': 2}.get(axis, -1))
    lower, upper, inverse = _primitive_bounds(primitive)
    return _rasterize(spec, lower, upper, primitive=primitive, inverse=inverse,
                      budget=max_candidates)[0]



@dataclass(frozen=True)
class GeometryItem:
    lower: np.ndarray
    upper: np.ndarray
    primitive: _Primitive | None = None
    inverse: np.ndarray | None = None

    def __post_init__(self):
        for name in ('lower', 'upper', 'inverse'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True)
class GroundTruthInput:
    timestamp: float
    items: tuple
    diagnostics: object

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('GT source timestamp must be finite and nonnegative')
        object.__setattr__(self, 'items', tuple(self.items))
        object.__setattr__(self, 'diagnostics', freeze(dict(self.diagnostics)))


def rasterize_ground_truth(spec, prepared, max_candidate_voxels=8_000_000):
    occupied = np.zeros(spec.shape, dtype=bool)
    stats = dict(prepared.diagnostics)
    stats['candidate_voxels'] = 0
    for item in prepared.items:
        indices, count = _rasterize(spec, item.lower, item.upper,
            primitive=item.primitive, inverse=item.inverse,
            budget=max_candidate_voxels-stats['candidate_voxels'])
        stats['candidate_voxels'] += count
        if len(indices):
            occupied[tuple(indices.T)] = True
    indices = np.argwhere(occupied)
    stats['occupied_voxels'] = len(indices)
    return indices, stats
