"""One camera-only global world plus explicitly allowlisted known support.

The bounded mapping worker product is copied by the SDK-free source boundary.
This adapter translates numeric descriptors into the existing Lula world
interface. Its only USD geometry read is the named physical support plane;
it never traverses the scene or imports obstacle/semantic GT geometry.
"""
from __future__ import annotations

import math

import numpy as np

from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from .static_world import StaticWorldSnapshot, StaticWorldError, _usd_modules, obstacle_from_primitive


SUPPORT_PLANE_PATHS = ('/World/GroundPlane/CollisionPlane',)


def capture_known_support(backend, *, stage=None, timestamp):
    """Read only exact allowlisted Plane prims and their ancestor transforms."""
    Usd, UsdGeom, UsdPhysics = _usd_modules()
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise StaticWorldError('Known support plane has no active USD stage')
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(units) or units <= 0:
        raise StaticWorldError('Known support stage units must be positive and finite')
    robot_position = np.asarray(backend.robot.get_world_pose()[0], float)*units
    if robot_position.shape != (3,) or not np.isfinite(robot_position).all():
        raise StaticWorldError('Known support domain requires a finite robot base position')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    obstacles, domains = [], []
    for path in SUPPORT_PLANE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if (not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsDefined()
                or prim.IsAbstract() or not prim.IsLoaded() or prim.GetPrimIndex().localErrors):
            raise StaticWorldError('Allowlisted support plane is unavailable: '+path)
        if str(prim.GetTypeName()) != 'Plane' or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise StaticWorldError('Allowlisted support must be an actual Plane collider: '+path)
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            raise StaticWorldError('Allowlisted support plane collider is disabled: '+path)
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            for attribute in ancestor.GetAttributes():
                if (attribute.GetName().startswith('xformOp:')
                        and attribute.ValueMightBeTimeVarying()):
                    raise StaticWorldError('Known support transform is animated: '+path)
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(ancestor)
                if (body.GetRigidBodyEnabledAttr().Get() is not False
                        and body.GetKinematicEnabledAttr().Get() is not True):
                    raise StaticWorldError('Known support plane has a dynamic body: '+path)
            ancestor = ancestor.GetParent()
        for attribute in prim.GetAttributes():
            if attribute.GetName() in ('axis','physics:collisionEnabled') and attribute.ValueMightBeTimeVarying():
                raise StaticWorldError('Known support geometry is animated: '+path)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), float).T.copy()
        transform[:3] *= units
        obstacle, domain = obstacle_from_primitive(path, 'Plane', transform,
            axis=prim.GetAttribute('axis').Get() or 'Z', plane_center=robot_position)
        obstacles.append(obstacle)
        domains.append(domain)
    return StaticWorldSnapshot(tuple(obstacles), timestamp, units, backend.robot.prim_path,
        plane_domains=tuple(domains))


def planning_world_from_frozen(frozen, support):
    """Translate exact worker boxes; no packing, expansion or cell scanning."""
    from .static_world import StaticObstacle
    paths = {item.path for item in support.obstacles}
    domains = {item.path for item in support.plane_domains}
    if (paths != set(SUPPORT_PLANE_PATHS) or domains != paths
            or len(support.obstacles) != len(SUPPORT_PLANE_PATHS)
            or len(support.plane_domains) != len(SUPPORT_PLANE_PATHS)
            or any(item.source_kind != 'Plane' or item.kind != 'cuboid'
                or item.approximation != 'plane_halfspace_clipped_to_certified_domain'
                for item in support.obstacles)):
        raise StaticWorldError('RGB-D support boundary cannot contain scene obstacle geometry')
    boxes = tuple(StaticObstacle('/__rgbd_occupied__/'+box.id, 'cuboid', box.center,
        orientation=box.orientation_wxyz, dimensions=box.dimensions, source_kind='rgbd_occupied_cells',
        approximation='exact_selected_occupied_cells', maximum_surface_error_m=0.) for box in frozen.boxes)
    metadata = dict(frozen.metadata, known_support_paths=tuple(item.path for item in support.obstacles),
        known_support_policy='Explicit exact Plane path allowlist; no scene traversal',
        camera_obstacle_count=len(boxes), known_support_count=len(support.obstacles),
        active_world_obstacle_refreshes=0)
    return StaticWorldSnapshot((*support.obstacles, *boxes), frozen.timestamp, support.meters_per_unit,
        support.robot_prim_path, plane_domains=support.plane_domains, source_metadata=metadata)


def capture_frozen_rgbd_world(backend, source_provider, *, max_obstacles=64, support_capture=capture_known_support):
    """Capture once; return frozen numeric world and fixed-time occupancy view."""
    from robot_skill_stack.world.spatial_mapping.frozen import capture_frozen_rgbd, FrozenRgbdSourceError
    try:
        source = None if source_provider is None else source_provider()
        frozen = capture_frozen_rgbd(source, max_obstacles=max_obstacles)
    except FrozenRgbdSourceError as error:
        raise PlanningFailure('planning_source_'+error.code, str(error), dict(error.details)) from error
    except Exception as error:
        raise PlanningFailure('planning_source_unavailable',
            f'Frozen RGB-D source access failed: {type(error).__name__}: {error}') from error
    try:
        support = support_capture(backend, timestamp=frozen.timestamp)
        snapshot = planning_world_from_frozen(frozen, support)
    except Exception as error:
        raise PlanningFailure('planning_source_support',
            f'Known static support cannot be captured: {type(error).__name__}: {error}') from error
    return snapshot, frozen
