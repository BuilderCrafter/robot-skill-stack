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
from collections.abc import Callable, Sequence
import numpy as np
from robot_skill_stack.world.spatial_mapping.geometry import (
    _Primitive, _primitive_bounds, GeometryItem, GroundTruthInput,
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
    diagnostics = {'source': 'ground_truth', 'visited_prims': 0, 'supported_primitives': 0,
             'excluded_prims': 0, 'guide_prims': 0, 'unsupported_aabb_prims': 0,
             'approximations': [], 'candidate_voxels': 0}
    Usd, UsdGeom, UsdPhysics = _usd_modules()

    # Validate stage
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('No active USD stage for ground-truth mapping')

    # Check composition errors
    composition_errors = getattr(stage, 'GetCompositionErrors', lambda: ())()
    if composition_errors:
        raise RuntimeError(f'Unresolved USD scene composition: {composition_errors}')

    # Check unit scale
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('USD metersPerUnit must be finite and positive')
    diagnostics['meters_per_unit'] = meters_per_unit

    # Get objects and transform them into bounding boxes
    time_code = Usd.TimeCode.Default()
    xforms = UsdGeom.XformCache(time_code)
    bbox_cache = UsdGeom.BBoxCache(time_code,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
         UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)

    # Validate all objects
    geometry_items = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        diagnostics['visited_prims'] += 1
        path = str(prim.GetPath())
        if any(path == prefix or path.startswith(prefix+'/') for prefix in excluded_paths):
            diagnostics['excluded_prims'] += 1
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
            diagnostics['guide_prims'] += 1
            continue

        # Get objects geometry
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
            diagnostics['supported_primitives'] += 1
        else:
            # If object has unsupported geometry, estimate it with a bounding box
            world_bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(world_bound.GetMin(), dtype=float)*meters_per_unit
            upper = np.asarray(world_bound.GetMax(), dtype=float)*meters_per_unit
            inverse = None
            diagnostics['unsupported_aabb_prims'] += 1
            if len(diagnostics['approximations']) < 64:
                diagnostics['approximations'].append({'path': path, 'type': prim.GetTypeName(),
                                               'world_min': lower.tolist(), 'world_max': upper.tolist()})
        geometry_items.append(GeometryItem(lower, upper, primitive, inverse))
    return GroundTruthInput(timestamp, tuple(geometry_items), diagnostics)
