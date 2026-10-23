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

import time
import numpy as np
from robot_skill_stack.world.spatial_mapping.geometry import (
    _Primitive, _primitive_bounds, _candidate_bounds, _rasterize,
    primitive_voxel_indices, GeometryItem, GroundTruthInput, rasterize_ground_truth,
)


def _usd_modules():
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def prepare_ground_truth(stage=None, *, excluded_paths=('/Franka',), timestamp=None,
                         clock=time.monotonic):
    """Collect immutable USD geometry on the SDK owner thread; never rasterize."""
    timestamp = clock() if timestamp is None else timestamp
    stats = {'source': 'ground_truth', 'visited_prims': 0, 'supported_primitives': 0,
             'excluded_prims': 0, 'guide_prims': 0, 'unsupported_aabb_prims': 0,
             'approximations': [], 'candidate_voxels': 0}
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
    items = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        stats['visited_prims'] += 1
        path = str(prim.GetPath())
        if any(path == prefix or path.startswith(prefix+'/') for prefix in excluded_paths):
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
        items.append(GeometryItem(lower, upper, primitive, inverse))
    return GroundTruthInput(timestamp, tuple(items), stats)


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

    def prepare(self, stage=None, *, timestamp=None):
        if self._closed:
            raise RuntimeError('Ground-truth mapper is closed')
        return prepare_ground_truth(stage, excluded_paths=self.excluded_paths,
                                    timestamp=timestamp, clock=self.map.clock)

    def update(self, stage=None, *, timestamp=None, now=None):
        """Synchronous compatibility path using the same preparation/rasterization."""
        if self._closed:
            raise RuntimeError('Ground-truth mapper is closed')
        started = time.perf_counter()
        now = self.map.clock() if now is None else now
        try:
            prepared = self.prepare(stage, timestamp=now if timestamp is None else timestamp)
            indices, stats = rasterize_ground_truth(self.map.spec, prepared, self.max_candidate_voxels)
            # Preserve the synchronous JSON diagnostics contract; prepared worker
            # metadata remains recursively immutable across the process boundary.
            stats['approximations'] = [dict(item, world_min=list(item['world_min']),
                world_max=list(item['world_max'])) for item in stats.get('approximations', ())]
            published = self.map.replace(indices, timestamp=prepared.timestamp, now=now)
            stats.update(published=published, update_seconds=time.perf_counter()-started)
            self.diagnostics = stats
            return published
        except Exception as exc:
            self.map.note_failure(str(exc), now=now)
            self.diagnostics = dict(source='ground_truth', published=False, error=str(exc),
                                    update_seconds=time.perf_counter()-started)
            raise

    def close(self):
        self._closed = True
