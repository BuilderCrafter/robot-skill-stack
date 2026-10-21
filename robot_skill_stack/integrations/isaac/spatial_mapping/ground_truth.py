"""Authoritative environment occupancy from active USD geometry, independent of semantics.

Primitive/voxel intersection is conservative: the inverse-transformed cell is
enclosed by local intervals. Supported primitives retain their curved/rotated
volume rather than occupying their entire world AABB. Unsupported Gprims use a
finite world AABB, explicitly counted in diagnostics. Invalid/unbounded geometry
fails the complete update. Invisible geometry remains included; guide geometry
is omitted only when it has no enabled CollisionAPI. Robot exclusion is an
explicit environment-layer policy, not evidence that the excluded region is safe
for a moving robot.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class _Primitive:
    kind: str
    world_from_local: np.ndarray
    dimensions: np.ndarray | None = None
    radius: float | None = None
    height: float | None = None
    axis: int = 2


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


def _usd_modules():
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


class IsaacGroundTruthMapper:
    def __init__(self, occupancy_map, *, excluded_paths=('/Franka',), max_candidate_voxels=8_000_000):
        if type(max_candidate_voxels) is not int or max_candidate_voxels <= 0:
            raise ValueError('max_candidate_voxels must be a positive integer')
        excluded_paths = tuple(excluded_paths)
        if any(not isinstance(path, str) or not path.startswith('/') or path == '/'
               for path in excluded_paths):
            raise ValueError('Excluded geometry paths must be absolute prim paths below the root')
        self.map = occupancy_map
        self.excluded_paths = tuple(path.rstrip('/') for path in excluded_paths)
        self.max_candidate_voxels = max_candidate_voxels
        self.diagnostics = {}
        self._closed = False

    def update(self, stage=None, *, timestamp=None, now=None):
        """Rebuild all nonexcluded geometry; other map cells become FREE atomically."""
        if self._closed:
            raise RuntimeError('Ground-truth mapper is closed')
        started = time.perf_counter()
        now = self.map.clock() if now is None else now
        timestamp = now if timestamp is None else timestamp
        stats = {'source': 'ground_truth', 'visited_prims': 0, 'supported_primitives': 0,
                 'excluded_prims': 0, 'guide_prims': 0, 'unsupported_aabb_prims': 0,
                 'approximations': [], 'candidate_voxels': 0}
        try:
            Usd, UsdGeom, UsdPhysics = _usd_modules()
            if stage is None:
                import omni.usd
                stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError('No active USD stage for ground-truth mapping')
            composition_errors = getattr(stage, 'GetCompositionErrors', lambda: ())()
            if composition_errors:
                raise RuntimeError(f'Unresolved USD scene composition: {composition_errors}')
            meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
            if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
                raise ValueError('USD metersPerUnit must be finite and positive')
            stats['meters_per_unit'] = meters_per_unit
            time_code = Usd.TimeCode.Default()
            xforms = UsdGeom.XformCache(time_code)
            bounds = UsdGeom.BBoxCache(time_code,
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
                 UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
            occupied = np.zeros(self.map.spec.shape, dtype=bool)
            for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
                stats['visited_prims'] += 1
                path = str(prim.GetPath())
                if any(path == prefix or path.startswith(prefix+'/') for prefix in self.excluded_paths):
                    stats['excluded_prims'] += 1
                    continue
                if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
                    continue
                if not prim.IsLoaded():
                    raise RuntimeError(f'Unloaded USD geometry payload: {path}')
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                collision = (prim.HasAPI(UsdPhysics.CollisionAPI) and
                             UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide and not collision:
                    stats['guide_prims'] += 1
                    continue
                # Gf matrices use row vectors; the math helpers use columns.
                transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=float).T
                transform = transform.copy()
                transform[:3, :] *= meters_per_unit
                if prim.IsA(UsdGeom.Cube):
                    size = UsdGeom.Cube(prim).GetSizeAttr().Get(time_code)
                    primitive = _Primitive('cube', transform, np.full(3, size, dtype=float))
                elif prim.IsA(UsdGeom.Sphere):
                    primitive = _Primitive('sphere', transform,
                        radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get(time_code))
                elif prim.IsA(UsdGeom.Cylinder):
                    cylinder = UsdGeom.Cylinder(prim)
                    primitive = _Primitive('cylinder', transform,
                        radius=cylinder.GetRadiusAttr().Get(time_code),
                        height=cylinder.GetHeightAttr().Get(time_code),
                        axis={'X': 0, 'Y': 1, 'Z': 2}.get(str(cylinder.GetAxisAttr().Get(time_code)), -1))
                else:
                    primitive = None
                if primitive is not None:
                    lower, upper, inverse = _primitive_bounds(primitive)
                    stats['supported_primitives'] += 1
                else:
                    world_bound = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
                    lower = np.asarray(world_bound.GetMin(), dtype=float)*meters_per_unit
                    upper = np.asarray(world_bound.GetMax(), dtype=float)*meters_per_unit
                    inverse = None
                    stats['unsupported_aabb_prims'] += 1
                    if len(stats['approximations']) < 64:
                        stats['approximations'].append({'path': path, 'type': prim.GetTypeName(),
                                                       'world_min': lower.tolist(), 'world_max': upper.tolist()})
                indices, count = _rasterize(self.map.spec, lower, upper,
                    primitive=primitive, inverse=inverse,
                    budget=self.max_candidate_voxels-stats['candidate_voxels'])
                stats['candidate_voxels'] += count
                if len(indices):
                    occupied[tuple(indices.T)] = True
            indices = np.argwhere(occupied)
            published = self.map.replace(indices, timestamp=timestamp, now=now)
            stats.update(published=published, occupied_voxels=len(indices),
                         update_seconds=time.perf_counter()-started)
            self.diagnostics = stats
            return published
        except Exception as exc:
            self.map.note_failure(str(exc), now=now)
            stats.update(published=False, error=str(exc), update_seconds=time.perf_counter()-started)
            self.diagnostics = stats
            raise

    def close(self):
        self._closed = True  # Owns no stage, camera, subscriptions or physics resources.
