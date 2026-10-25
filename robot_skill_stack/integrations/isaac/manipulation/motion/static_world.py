"""Frozen physical primitive worlds for static global planning.

USD is read once, on the simulator owner thread.  Numeric descriptors neither
create USD prims nor depend on mapping/perception.  Unsupported physical
geometry fails the complete request instead of disappearing from the world.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from robot_skill_stack.common.rotations import quat_matrix


class StaticWorldError(ValueError):
    """The physical scene cannot be represented by the static planner."""


def _usd_modules():
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def _tuple(value, count, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise StaticWorldError(f'{name} must contain {count} finite numbers')
    return tuple(map(float, array))


def _quaternion(rotation):
    """Scalar-first quaternion from a proper orthonormal matrix."""
    r = np.asarray(rotation, dtype=float)
    if np.linalg.det(r) < 0:
        r = r.copy()
        r[:, 0] *= -1
    trace = float(np.trace(r))
    if trace > 0:
        scale = 2*math.sqrt(trace+1)
        q = np.array([scale/4, (r[2, 1]-r[1, 2])/scale,
            (r[0, 2]-r[2, 0])/scale, (r[1, 0]-r[0, 1])/scale])
    else:
        axis = int(np.argmax(np.diag(r)))
        following, last = (axis+1) % 3, (axis+2) % 3
        scale = 2*math.sqrt(1+r[axis, axis]-r[following, following]-r[last, last])
        q = np.zeros(4)
        q[0] = (r[last, following]-r[following, last])/scale
        q[axis+1] = scale/4
        q[following+1] = (r[following, axis]+r[axis, following])/scale
        q[last+1] = (r[last, axis]+r[axis, last])/scale
    q /= np.linalg.norm(q)
    return tuple(map(float, q if q[0] >= 0 else -q))


def _affine(transform):
    matrix = np.array(transform, dtype=float, copy=True)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10)):
        raise StaticWorldError('Physical geometry requires a finite affine transform')
    scales = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scales <= 0):
        raise StaticWorldError('Degenerate physical geometry scale')
    rotation = matrix[:3, :3]/scales
    if not np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-7):
        raise StaticWorldError('Sheared physical geometry is unsupported')
    return matrix, rotation, scales


@dataclass(frozen=True)
class StaticObstacle:
    path: str
    kind: str
    position: tuple
    orientation: tuple = (1., 0., 0., 0.)
    dimensions: tuple = (0., 0., 0.)
    radius: float = 0.
    height: float = 0.
    source_kind: str = ''
    approximation: str = 'exact'
    maximum_surface_error_m: float = 0.

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path.startswith('/'):
            raise StaticWorldError('Physical obstacle path must be absolute')
        if self.kind not in ('cuboid', 'sphere', 'capsule'):
            raise StaticWorldError(f'Unsupported numeric obstacle kind: {self.kind}')
        object.__setattr__(self, 'position', _tuple(self.position, 3, 'position'))
        q = np.asarray(_tuple(self.orientation, 4, 'orientation'))
        if np.linalg.norm(q) <= 0:
            raise StaticWorldError('Physical obstacle orientation cannot be zero')
        object.__setattr__(self, 'orientation', tuple(map(float, q/np.linalg.norm(q))))
        object.__setattr__(self, 'dimensions', _tuple(self.dimensions, 3, 'dimensions'))
        if (not math.isfinite(self.radius) or self.radius < 0 or
                not math.isfinite(self.height) or self.height < 0):
            raise StaticWorldError('Physical primitive radius/height must be finite and nonnegative')
        if self.kind == 'cuboid' and min(self.dimensions) <= 0:
            raise StaticWorldError('Physical cuboid dimensions must be positive')
        if self.kind in ('sphere', 'capsule') and (not math.isfinite(self.radius) or self.radius <= 0):
            raise StaticWorldError('Physical sphere/capsule radius must be positive')
        if self.kind == 'capsule' and (not math.isfinite(self.height) or self.height <= 0):
            raise StaticWorldError('Physical capsule spine height must be positive')
        if not math.isfinite(self.maximum_surface_error_m) or self.maximum_surface_error_m < 0:
            raise StaticWorldError('Approximation error must be finite and nonnegative')


@dataclass(frozen=True)
class PlaneDomain:
    """Domain in which a finite cuboid exactly represents a plane half-space.

    The physical plane's positive local Z side is free.  Samples outside the
    lateral domain or below the represented depth are rejected, not certified.
    """
    path: str
    position: tuple
    orientation: tuple
    half_width_m: float = 25.
    depth_m: float = 50.

    def __post_init__(self):
        object.__setattr__(self, 'position', _tuple(self.position, 3, 'plane position'))
        q = np.asarray(_tuple(self.orientation, 4, 'plane orientation'))
        if np.linalg.norm(q) <= 0:
            raise StaticWorldError('Plane domain requires a nonzero orientation')
        object.__setattr__(self, 'orientation', tuple(map(float, q/np.linalg.norm(q))))
        if (not math.isfinite(self.half_width_m) or self.half_width_m <= 0 or
                not math.isfinite(self.depth_m) or self.depth_m <= 0):
            raise StaticWorldError('Certified plane domain must have positive finite dimensions')

    def assert_contains(self, centers, radii):
        points = np.asarray(centers, dtype=float)
        radius = np.asarray(radii, dtype=float).reshape(-1)
        if (points.ndim != 2 or points.shape[1:] != (3,) or radius.shape != (len(points),)
                or not np.isfinite(points).all() or not np.isfinite(radius).all() or np.any(radius < 0)):
            raise StaticWorldError('Finite world collision-sphere positions/radii are required')
        local = (points-np.asarray(self.position))@quat_matrix(self.orientation)
        if (np.any(np.abs(local[:, :2])+radius[:, None] > self.half_width_m) or
                np.any(local[:, 2]-radius < -self.depth_m)):
            raise StaticWorldError(f'Robot collision model leaves the certified plane domain: {self.path}')


@dataclass(frozen=True)
class StaticWorldSnapshot:
    obstacles: tuple
    timestamp: float
    meters_per_unit: float
    robot_prim_path: str
    excluded_colliders: int = 0
    disabled_colliders: int = 0
    plane_domains: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'obstacles', tuple(self.obstacles))
        object.__setattr__(self, 'plane_domains', tuple(self.plane_domains))
        if (not math.isfinite(self.timestamp) or self.timestamp < 0 or
                not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0):
            raise StaticWorldError('Static world needs valid capture time and stage units')
        if any(not isinstance(item, StaticObstacle) for item in self.obstacles):
            raise StaticWorldError('Static world requires immutable primitive descriptors')
        if len({item.path for item in self.obstacles}) != len(self.obstacles):
            raise StaticWorldError('Duplicate static obstacle paths')

    def assert_within_domains(self, world_sphere_centers, sphere_radii):
        for domain in self.plane_domains:
            domain.assert_contains(world_sphere_centers, sphere_radii)

    def as_dict(self):
        return asdict(self)


def obstacle_from_primitive(path, primitive_type, transform, *, size=None, radius=None,
                            height=None, axis='Z', plane_center=None):
    """Convert an authored primitive transform already expressed in meters.

    Return ``(descriptor, optional_plane_domain)``.  Cube scaling can be
    nonuniform; spheres and capsules require uniform absolute scaling because
    Lula has no ellipsoid or ellipsoidal capsule.  Flat cylinders are rejected.
    """
    matrix, rotation, scales = _affine(transform)
    position = matrix[:3, 3]
    if primitive_type == 'Cube':
        if size is None or not math.isfinite(size) or size <= 0:
            raise StaticWorldError(f'Invalid physical cube size: {path}')
        return StaticObstacle(path, 'cuboid', tuple(position), _quaternion(rotation),
            dimensions=tuple(size*scales), source_kind='Cube'), None
    if primitive_type in ('Sphere', 'Capsule'):
        if not np.allclose(scales, scales[0], rtol=1e-7, atol=1e-12):
            raise StaticWorldError(f'Nonuniform {primitive_type} scale is unsupported: {path}')
        if radius is None or not math.isfinite(radius) or radius <= 0:
            raise StaticWorldError(f'Invalid physical curved primitive radius: {path}')
        if primitive_type == 'Sphere':
            return StaticObstacle(path, 'sphere', tuple(position), _quaternion(rotation),
                radius=float(radius*scales[0]), source_kind='Sphere'), None
        if axis not in ('X', 'Y', 'Z') or height is None or not math.isfinite(height) or height <= 0:
            raise StaticWorldError(f'Invalid physical capsule axis/height: {path}')
        order = {'X': (1, 2, 0), 'Y': (2, 0, 1), 'Z': (0, 1, 2)}[axis]
        return StaticObstacle(path, 'capsule', tuple(position), _quaternion(rotation[:, order]),
            radius=float(radius*scales[0]), height=float(height*scales[0]), source_kind='Capsule'), None
    if primitive_type == 'Plane':
        if axis not in ('X', 'Y', 'Z'):
            raise StaticWorldError(f'Invalid physical plane axis: {path}')
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        normal = rotation[:, axis_index]
        tangent = rotation[:, (axis_index+1) % 3]
        basis = np.column_stack((tangent, np.cross(normal, tangent), normal))
        requested_center = position if plane_center is None else np.asarray(_tuple(plane_center, 3, 'plane_center'))
        position = requested_center-normal*float(np.dot(requested_center-position, normal))
        orientation = _quaternion(basis)
        domain = PlaneDomain(path, tuple(position), orientation)
        descriptor = StaticObstacle(path, 'cuboid', tuple(position-normal*domain.depth_m/2),
            orientation, dimensions=(2*domain.half_width_m, 2*domain.half_width_m, domain.depth_m),
            source_kind='Plane', approximation='plane_halfspace_clipped_to_certified_domain',
            maximum_surface_error_m=0.)
        return descriptor, domain
    raise StaticWorldError(f'Unsupported physical primitive {primitive_type}: {path}')


def box_from_convex_hull_points(path, points, transform):
    """Exact box recognition for a convex-hull mesh, never an AABB fallback."""
    vertices = np.asarray(points, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise StaticWorldError(f'Invalid physical convex-hull vertices: {path}')
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(high <= low):
        raise StaticWorldError(f'Degenerate physical convex hull: {path}')
    # Exact corner membership avoids silently enclosing slightly non-box data.
    if not np.all((vertices == low) | (vertices == high)):
        raise StaticWorldError(f'Convex-hull mesh is not an exact cuboid: {path}')
    corners = {tuple(row) for row in (vertices == high).astype(int)}
    if len(corners) != 8:
        raise StaticWorldError(f'Convex-hull mesh does not contain all eight box corners: {path}')
    matrix, rotation, scales = _affine(transform)
    center = matrix[:3, :3]@((low+high)/2)+matrix[:3, 3]
    return StaticObstacle(path, 'cuboid', tuple(center), _quaternion(rotation),
        dimensions=tuple((high-low)*scales), source_kind='Mesh:convexHull_exact_box')


def capture_static_world(stage=None, robot_prim_path='/Franka', *, timestamp=None):
    """Read all enabled external colliders once on the USD owner thread.

    Dynamic nonkinematic actors are unsupported even when momentarily at rest.
    Kinematic/static actors are frozen under the caller's stationarity contract.
    Visibility/purpose never removes an enabled physical collider.
    """
    Usd, UsdGeom, UsdPhysics = _usd_modules()
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise StaticWorldError('No active USD stage for static global planning')
    if (not isinstance(robot_prim_path, str) or not robot_prim_path.startswith('/') or
            robot_prim_path == '/'):
        raise StaticWorldError('Robot exclusion must be an absolute subtree below the stage root')
    robot_prim_path = robot_prim_path.rstrip('/')
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(units) or units <= 0:
        raise StaticWorldError('USD metersPerUnit must be finite and positive')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise StaticWorldError(f'Robot subtree is missing: {robot_prim_path}')
    robot_position = np.asarray(cache.GetLocalToWorldTransform(robot), dtype=float).T[:3, 3]*units
    obstacles, domains = [], []
    excluded = disabled = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        path = str(prim.GetPath())
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if path == robot_prim_path or path.startswith(robot_prim_path+'/'):
            excluded += int(collision)
            continue
        if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
            continue
        if not prim.IsLoaded():
            raise StaticWorldError(f'Unloaded external USD geometry: {path}')
        errors = prim.GetPrimIndex().localErrors
        if errors:
            raise StaticWorldError(f'Unresolved external USD scene composition: {path}: {errors}')
        if not collision:
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            disabled += 1
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            for attribute in ancestor.GetAttributes():
                if (attribute.GetName().startswith('xformOp:') and
                        attribute.ValueMightBeTimeVarying()):
                    raise StaticWorldError(f'Animated external collider transform is not static: {path}')
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(ancestor)
                if (body.GetRigidBodyEnabledAttr().Get() is not False and
                        body.GetKinematicEnabledAttr().Get() is not True):
                    raise StaticWorldError(f'Dynamic external body is not a static planning obstacle: {path}')
            ancestor = ancestor.GetParent()
        for attribute in prim.GetAttributes():
            if (attribute.GetName() in ('points', 'size', 'radius', 'height', 'axis',
                    'physics:collisionEnabled', 'physics:approximation') and
                    attribute.ValueMightBeTimeVarying()):
                raise StaticWorldError(f'Animated external collision geometry is not static: {path}')
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T.copy()
        transform[:3] *= units
        kind = str(prim.GetTypeName())
        if prim.IsA(UsdGeom.Mesh):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation != 'convexHull':
                raise StaticWorldError(f'Unsupported physical mesh collision approximation {approximation}: {path}')
            mesh = UsdGeom.Mesh(prim)
            if mesh.GetHoleIndicesAttr().Get():
                raise StaticWorldError(f'Physical mesh with holes is unsupported: {path}')
            obstacles.append(box_from_convex_hull_points(path, mesh.GetPointsAttr().Get(), transform))
            continue
        params = {}
        if kind == 'Cube':
            params['size'] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        elif kind == 'Sphere':
            params['radius'] = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        elif kind == 'Capsule':
            shape = UsdGeom.Capsule(prim)
            params.update(radius=float(shape.GetRadiusAttr().Get()), height=float(shape.GetHeightAttr().Get()),
                axis=str(shape.GetAxisAttr().Get()))
        elif kind == 'Plane':
            params.update(axis=str(prim.GetAttribute('axis').Get() or 'Z'), plane_center=robot_position)
        descriptor, domain = obstacle_from_primitive(path, kind, transform, **params)
        obstacles.append(descriptor)
        if domain is not None:
            domains.append(domain)
    return StaticWorldSnapshot(tuple(sorted(obstacles, key=lambda item: item.path)),
        time.monotonic() if timestamp is None else float(timestamp), units, robot_prim_path,
        excluded, disabled, tuple(domains))
