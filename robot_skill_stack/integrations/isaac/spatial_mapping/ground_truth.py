"""Authoritative environment occupancy from active USD geometry, independent of semantics.

Supported primitives use their affine shape; unsupported Gprims use a finite
world AABB. Invisible geometry remains included, while noncolliding guides and
the configured robot paths are excluded.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
import numpy as np
from robot_skill_stack.world.spatial_mapping.geometry import (
    _primitive_item, GeometryItem, GroundTruthInput,
)


def _usd_modules() -> tuple[object, object, object]:
    """Load USD APIs only when accessing an active Isaac scene."""
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def prepare_ground_truth(stage=None, *, excluded_paths: Sequence[str] = ('/Franka',),
                         timestamp: float | None = None,
                         clock: Callable[[], float] = time.monotonic) -> GroundTruthInput:
    """Collect immutable USD geometry on the SDK owner thread; never rasterize."""
    timestamp = clock() if timestamp is None else timestamp
    Usd, UsdGeom, UsdPhysics = _usd_modules()

    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('No active USD stage for ground-truth mapping')

    composition_errors = stage.GetCompositionErrors()
    if composition_errors:
        raise RuntimeError(f'Unresolved USD scene composition: {composition_errors}')

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('USD metersPerUnit must be finite and positive')

    time_code = Usd.TimeCode.Default()
    xforms = UsdGeom.XformCache(time_code)
    bbox_cache = UsdGeom.BBoxCache(time_code,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
         UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)

    geometry_items = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        path = str(prim.GetPath())
        if any(path == prefix or path.startswith(prefix+'/') for prefix in excluded_paths):
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
            continue

        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=float).T.copy()
        transform[:3, :] *= meters_per_unit
        if prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get(time_code)
            item = _primitive_item('cube', transform, dimensions=np.full(3, size, dtype=float))
        elif prim.IsA(UsdGeom.Sphere):
            item = _primitive_item('sphere', transform,
                radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get(time_code))
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            item = _primitive_item('cylinder', transform,
                radius=cylinder.GetRadiusAttr().Get(time_code),
                height=cylinder.GetHeightAttr().Get(time_code),
                axis={'X': 0, 'Y': 1, 'Z': 2}[str(cylinder.GetAxisAttr().Get(time_code))])
        else:
            world_bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(world_bound.GetMin(), dtype=float)*meters_per_unit
            upper = np.asarray(world_bound.GetMax(), dtype=float)*meters_per_unit
            if not np.isfinite([lower, upper]).all() or np.any(lower > upper):
                raise ValueError('Unsupported geometry needs finite world bounds: '+path)
            item = GeometryItem(lower, upper)
        geometry_items.append(item)
    return GroundTruthInput(timestamp, tuple(geometry_items))
