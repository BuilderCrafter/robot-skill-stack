"""One camera-only global world plus explicitly allowlisted known support.

The SDK-free source boundary retains the mapping worker's fixed-time view.
This adapter translates numeric descriptors into the existing Lula world
interface. Its only USD geometry read is the named physical support plane;
it never traverses the scene or imports obstacle/semantic GT geometry.
"""
from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING

import numpy as np

from robot_skill_stack.manipulation.motion import MappingSourceProvider
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from robot_skill_stack.world.spatial_mapping import MappingReadView
from .static_world import StaticObstacle, StaticWorldError, StaticWorldSnapshot, obstacle_from_primitive

if TYPE_CHECKING:
    from pxr import Usd
    from ..franka_backend import IsaacFrankaBackend


SUPPORT_PLANE_PATHS = ('/World/GroundPlane/CollisionPlane',)


def capture_known_support(backend: IsaacFrankaBackend, *, stage: Usd.Stage | None = None,
                           timestamp: float) -> StaticWorldSnapshot:
    """Read only exact allowlisted Plane prims and their ancestor transforms."""
    from pxr import Usd, UsdGeom, UsdPhysics
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise StaticWorldError('Known support plane has no active USD stage')
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise StaticWorldError('Known support stage units must be positive and finite')
    robot_position = np.asarray(backend.robot.get_world_pose()[0], float) * meters_per_unit
    if robot_position.shape != (3,) or not np.isfinite(robot_position).all():
        raise StaticWorldError('Known support domain requires a finite robot base position')
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    obstacles, domains = [], []
    for path in SUPPORT_PLANE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if (not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsDefined()
                or prim.IsAbstract() or not prim.IsLoaded() or prim.GetPrimIndex().localErrors):
            raise StaticWorldError('Allowlisted support plane is unavailable: ' + path)
        if str(prim.GetTypeName()) != 'Plane' or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise StaticWorldError('Allowlisted support must be an actual Plane collider: ' + path)
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            raise StaticWorldError('Allowlisted support plane collider is disabled: ' + path)
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            for attribute in ancestor.GetAttributes():
                if (attribute.GetName().startswith('xformOp:')
                        and attribute.ValueMightBeTimeVarying()):
                    raise StaticWorldError('Known support transform is animated: ' + path)
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(ancestor)
                if (body.GetRigidBodyEnabledAttr().Get() is not False
                        and body.GetKinematicEnabledAttr().Get() is not True):
                    raise StaticWorldError('Known support plane has a dynamic body: ' + path)
            ancestor = ancestor.GetParent()
        for attribute in prim.GetAttributes():
            if (attribute.GetName() in ('axis', 'physics:collisionEnabled')
                    and attribute.ValueMightBeTimeVarying()):
                raise StaticWorldError('Known support geometry is animated: ' + path)
        # Transpose USD row vectors and convert the full affine map to meters.
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), float).T.copy()
        transform[:3] *= meters_per_unit
        obstacle, domain = obstacle_from_primitive(path, 'Plane', transform,
            axis=prim.GetAttribute('axis').Get() or 'Z', plane_center=robot_position)
        obstacles.append(obstacle)
        domains.append(domain)
    return StaticWorldSnapshot(tuple(obstacles), timestamp, meters_per_unit, backend.robot.prim_path,
        plane_domains=tuple(domains))


def planning_world_from_frozen(view: MappingReadView, support: StaticWorldSnapshot) -> StaticWorldSnapshot:
    """Combine allowlisted support and exact selected cells at the camera's source time."""
    support_paths = {item.path for item in support.obstacles}
    domain_paths = {item.path for item in support.plane_domains}
    if (support_paths != set(SUPPORT_PLANE_PATHS) or domain_paths != support_paths
            or len(support.obstacles) != len(SUPPORT_PLANE_PATHS)
            or len(support.plane_domains) != len(SUPPORT_PLANE_PATHS)
            or any(item.source_kind != 'Plane' or item.kind != 'cuboid'
                or item.approximation != 'plane_halfspace_clipped_to_certified_domain'
                for item in support.obstacles)):
        raise StaticWorldError('RGB-D support boundary cannot contain scene obstacle geometry')
    boxes = tuple(StaticObstacle('/__rgbd_occupied__/' + box.id, 'cuboid', box.center,
        orientation=(1., 0., 0., 0.), dimensions=box.dimensions, source_kind='rgbd_occupied_cells',
        approximation='exact_selected_occupied_cells', maximum_surface_error_m=0.) for box in view.obstacles.boxes)
    metadata = dict(source_type='rgbd', generation=view.generation, revision=view.revision,
        map_age_seconds=view.evaluated_at - view.timestamp,
        camera_ids=tuple(frame['camera_id'] for frame in view.source_frames),
        known_support_paths=tuple(item.path for item in support.obstacles),
        camera_obstacle_count=len(boxes))
    return StaticWorldSnapshot((*support.obstacles, *boxes), view.timestamp, support.meters_per_unit,
        support.robot_prim_path, plane_domains=support.plane_domains, source_metadata=metadata)


def capture_frozen_rgbd_world(
    backend: IsaacFrankaBackend, source_provider: MappingSourceProvider | None, *,
    max_obstacles: int = 64, support_capture: Callable[..., StaticWorldSnapshot] = capture_known_support,
    representation: str = 'occupied',
) -> tuple[StaticWorldSnapshot, MappingReadView]:
    """Capture once; return frozen numeric world and fixed-time occupancy view."""
    from robot_skill_stack.world.spatial_mapping.frozen import capture_frozen_rgbd, FrozenRgbdSourceError
    try:
        source = None if source_provider is None else source_provider()
        view = capture_frozen_rgbd(source, max_obstacles=max_obstacles)
    except FrozenRgbdSourceError as error:
        raise PlanningFailure('planning_source_' + error.code, str(error)) from error
    except Exception as error:
        raise PlanningFailure('planning_source_unavailable',
            f'Frozen RGB-D source access failed: {type(error).__name__}: {error}') from error
    try:
        support = support_capture(backend, timestamp=view.timestamp)
        # Validate occupied evidence even when reconstruction will subsume some of it.
        snapshot = planning_world_from_frozen(view, support)
    except Exception as error:
        raise PlanningFailure('planning_source_support',
            f'Known static support cannot be captured: {type(error).__name__}: {error}') from error
    if representation == 'reconstructed':
        from dataclasses import replace
        from robot_skill_stack.world.spatial_mapping.reconstruction import combine
        try:
            if max_obstacles != 64:
                raise ValueError('Reconstructed camera-proxy budget must be 64')
            descriptors = combine(view)
            camera_obstacles = tuple(StaticObstacle(item['path'], 'cuboid', item['position'],
                orientation=item['orientation'], dimensions=item['dimensions'], source_kind=item['source_kind'],
                approximation='single_convex_yaw_box_empirical_bounds' if item['source_kind'] == 'rgbd_inferred_box'
                else 'exact_selected_occupied_cells') for item in descriptors)
            metadata = dict(snapshot.source_metadata, snapshot_policy='frozen_rgbd_reconstructed',
                camera_obstacle_count=len(camera_obstacles))
            snapshot = replace(snapshot, obstacles=(*support.obstacles, *camera_obstacles), source_metadata=metadata)
        except Exception as error:
            raise PlanningFailure('planning_source_reconstruction', str(error)) from error
    elif representation != 'occupied':
        raise PlanningFailure('planning_source_representation', 'Unknown RGB-D representation')
    return snapshot, view
