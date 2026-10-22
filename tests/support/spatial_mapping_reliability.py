"""SDK-free V4 references and metrics; never inputs to the production mapper.

Center membership is exact analytic solid membership (closed boundaries, with a
1e-12 local-coordinate numerical tolerance). Cell overlap is only a conservative
outer bound: an inverse-transformed voxel is enclosed by local intervals. These
two references deliberately have different meanings. Surface-event metrics use
only the caller's explicitly supplied measured endpoints and labels.

Callers must provide every physical shape relevant to their evaluated region,
or limit that region to a separately verified fixture neighborhood. Absence from
this reference list does not establish that an arbitrary workspace is free.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from robot_skill_stack.world.spatial_mapping import OccupancyState


_EPSILON = 1e-12
_CORNERS = np.asarray(list(product((0., 1.), repeat=3)))


def _readonly(values, dtype=float):
    values = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _points(values):
    values = np.asarray(values, float)
    if not values.size:
        return values.reshape(0, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected points with shape (N,3)')
    return values


def _convex_planes(vertices, triangles):
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or triangles.ndim != 2
            or triangles.shape[1] != 3 or len(triangles) < 4
            or triangles.dtype.kind not in 'iu' or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))):
        raise ValueError('Convex reference requires finite vertices and triangle indices')
    edges = np.sort(np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]])), axis=1)
    if np.any(np.unique(edges, axis=0, return_counts=True)[1] != 2):
        raise ValueError('Convex reference mesh must be closed and manifold')
    extent = max(float(np.ptp(vertices, axis=0).max()), _EPSILON)
    if np.linalg.svd(vertices-vertices.mean(0), compute_uv=False)[-1] <= extent*1e-10:
        raise ValueError('Convex reference must enclose a three-dimensional solid')
    first, second, third = vertices[triangles].transpose(1, 0, 2)
    normal = np.cross(second-first, third-first)
    magnitude = np.linalg.norm(normal, axis=1)
    if np.any(magnitude <= extent**2*1e-12):
        raise ValueError('Convex reference mesh has degenerate faces')
    normal /= magnitude[:, None]
    normal[np.sum((vertices.mean(0)-first)*normal, axis=1) > 0] *= -1
    offset = np.sum(normal*first, axis=1)
    if np.any(vertices @ normal.T-offset > extent*1e-9):
        raise ValueError('Reference mesh must be convex; use a union for nonconvex fixtures')
    return normal, offset


@dataclass(frozen=True, eq=False)
class ReferenceShape:
    """Owned local-meter geometry with a full affine world transform."""
    name: str
    kind: str
    world_from_local: np.ndarray
    dimensions: np.ndarray | None = None
    radius: float | None = None
    height: float | None = None
    axis: str = 'Z'
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    reference_role: str = 'physical_and_visual'
    note: str = ''

    def __post_init__(self):
        kind = 'box' if self.kind == 'cube' else self.kind
        transform = np.asarray(self.world_from_local, float)
        if (not isinstance(self.name, str) or not self.name or kind not in
                ('box', 'sphere', 'cylinder', 'convex_mesh') or transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                or self.reference_role not in ('physical_and_visual', 'visual_only', 'collision_only')):
            raise ValueError('Invalid reference name, kind, role or affine transform')
        try:
            inverse = np.linalg.inv(transform)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Reference transform must be invertible') from exc
        if not np.isfinite(inverse).all():
            raise ValueError('Reference transform needs a finite inverse')
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'world_from_local', _readonly(transform))
        object.__setattr__(self, '_inverse', _readonly(inverse))
        if kind == 'box':
            dimensions = np.asarray(self.dimensions, float)
            if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError('Box reference dimensions must be finite and positive')
            object.__setattr__(self, 'dimensions', _readonly(dimensions))
        if kind in ('sphere', 'cylinder'):
            if self.radius is None or not np.isfinite(self.radius) or self.radius <= 0:
                raise ValueError('Reference radius must be finite and positive')
            object.__setattr__(self, 'radius', float(self.radius))
        if kind == 'cylinder':
            if self.axis not in ('X', 'Y', 'Z') or self.height is None or not np.isfinite(self.height) or self.height <= 0:
                raise ValueError('Cylinder reference requires axis and positive finite height')
            object.__setattr__(self, 'height', float(self.height))
        if kind == 'convex_mesh':
            vertices, triangles = np.asarray(self.vertices, float), np.asarray(self.triangles)
            normal, offset = _convex_planes(vertices, triangles)
            for field, value, dtype in (('vertices', vertices, float), ('triangles', triangles, np.int64),
                    ('_normal', normal, float), ('_offset', offset, float)):
                object.__setattr__(self, field, _readonly(value, dtype))

    @classmethod
    def from_payload(cls, payload):
        return cls(**payload)

    def to_payload(self):
        result = dict(name=self.name, kind=self.kind, world_from_local=self.world_from_local.tolist(),
                      reference_role=self.reference_role, note=self.note)
        for name in ({'box': ('dimensions',), 'sphere': ('radius',),
                      'cylinder': ('radius', 'height', 'axis'),
                      'convex_mesh': ('vertices', 'triangles')}[self.kind]):
            value = getattr(self, name)
            result[name] = value.tolist() if isinstance(value, np.ndarray) else value
        return result

    def world_bounds(self):
        linear, center = self.world_from_local[:3, :3], self.world_from_local[:3, 3]
        if self.kind == 'box':
            extent = abs(linear) @ (self.dimensions/2)
        elif self.kind == 'sphere':
            extent = self.radius*np.linalg.norm(linear, axis=1)
        elif self.kind == 'cylinder':
            axis = 'XYZ'.index(self.axis)
            radial = [i for i in range(3) if i != axis]
            extent = abs(linear[:, axis])*self.height/2+self.radius*np.linalg.norm(linear[:, radial], axis=1)
        else:
            points = self.vertices @ linear.T+center
            return points.min(0), points.max(0)
        return center-extent, center+extent


def _shape_contains(points, shape):
    local = points @ shape._inverse[:3, :3].T+shape._inverse[:3, 3]
    if shape.kind == 'box':
        return (abs(local) <= shape.dimensions/2+_EPSILON).all(1)
    if shape.kind == 'sphere':
        return np.linalg.norm(local, axis=1) <= shape.radius+_EPSILON
    if shape.kind == 'cylinder':
        axis = 'XYZ'.index(shape.axis)
        return ((abs(local[:, axis]) <= shape.height/2+_EPSILON) &
                (np.linalg.norm(local[:, [i for i in range(3) if i != axis]], axis=1) <= shape.radius+_EPSILON))
    return (local @ shape._normal.T <= shape._offset+_EPSILON).all(1)


def center_membership(points, shapes):
    """Membership in the union of exact closed solids, never their world AABBs."""
    points = _points(points)
    if not np.isfinite(points).all():
        raise ValueError('Reference membership requires finite points')
    selected = np.zeros(len(points), bool)
    for shape in shapes:
        selected |= _shape_contains(points, shape)
    return selected


def conservative_cell_overlap(spec, indices, shapes):
    """Possible overlap, not exact occupied volume or positive collision proof."""
    indices = spec.validate_indices(indices)
    lower = np.asarray(spec.bounds_min)+indices*spec.voxel_size
    upper = np.minimum(lower+spec.voxel_size, spec.bounds_max)
    centers, half = (lower+upper)/2, (upper-lower)/2
    overlap = np.zeros(len(indices), bool)
    for shape in shapes:
        local = centers @ shape._inverse[:3, :3].T+shape._inverse[:3, 3]
        local_half = half @ abs(shape._inverse[:3, :3]).T
        nearest = np.maximum(abs(local)-local_half, 0)
        if shape.kind == 'box':
            selected = (nearest <= shape.dimensions/2+_EPSILON).all(1)
        elif shape.kind == 'sphere':
            selected = np.linalg.norm(nearest, axis=1) <= shape.radius+_EPSILON
        elif shape.kind == 'cylinder':
            axis = 'XYZ'.index(shape.axis)
            selected = ((nearest[:, axis] <= shape.height/2+_EPSILON) &
                (np.linalg.norm(nearest[:, [i for i in range(3) if i != axis]], axis=1) <= shape.radius+_EPSILON))
        else:
            selected = (local @ shape._normal.T-local_half @ abs(shape._normal).T <= shape._offset+_EPSILON).all(1)
        overlap |= selected
    return overlap


def conservative_cell_interior(spec, indices, shapes):
    """All eight corners inside one convex shape: sufficient whole-cell truth.

    A union can enclose a whole cell without any one part containing it; such
    cases are deliberately left outside this conservative interior stratum.
    Actual shortened boundary cells are used. Batches bound temporary arrays.
    """
    indices, shapes = spec.validate_indices(indices), tuple(shapes)
    inside = np.zeros(len(indices), bool)
    for begin in range(0, len(indices), 8192):
        selected = indices[begin:begin+8192]
        lower = np.asarray(spec.bounds_min)+selected*spec.voxel_size
        upper = np.minimum(lower+spec.voxel_size, spec.bounds_max)
        corners = (lower[:, None]+(upper-lower)[:, None]*_CORNERS).reshape(-1, 3)
        for shape in shapes:
            inside[begin:begin+len(selected)] |= _shape_contains(corners, shape).reshape(-1, 8).all(1)
    return inside


def neighborhood_indices(spec, shapes, *, margin=.02):
    """Unique map cells near reference AABBs; AABBs only select the scored domain."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Reference neighborhood margin must be finite and nonnegative')
    pieces = []
    for shape in shapes:
        lower, upper = shape.world_bounds()
        low = np.maximum(np.floor((lower-margin-spec.bounds_min)/spec.voxel_size), 0).astype(int)
        high = np.minimum(np.floor((upper+margin-spec.bounds_min)/spec.voxel_size), np.asarray(spec.shape)-1).astype(int)
        if np.all(high >= low):
            pieces.append(np.column_stack(np.unravel_index(np.arange(int(np.prod(high-low+1))), high-low+1))+low)
    return np.unique(np.concatenate(pieces), axis=0) if pieces else np.empty((0, 3), int)


def _states(states, selected):
    return dict(voxels=int(selected.sum()), **{state.name.lower(): int((selected & (states == state)).sum())
                for state in (OccupancyState.UNKNOWN, OccupancyState.FREE, OccupancyState.OCCUPIED)})


def _rate(count, denominator):
    return dict(count=int(count), denominator=int(denominator), rate=float(count/denominator) if denominator else None)


def _excluded(spec, indices, excluded_indices):
    excluded_indices = spec.validate_indices(excluded_indices)
    return np.isin(np.ravel_multi_index(indices.T, spec.shape),
                   np.ravel_multi_index(excluded_indices.T, spec.shape))


def score_voxel_centers(snapshot, shapes, indices, *, other_exposures=(), excluded_indices=()):
    """Score stable exposure truth; disagreement remains a separate stratum.

    ``indices`` declares the evaluated region, including its reference-completeness
    assumption. For motion, pass each other exposure's complete shape tuple. An
    occupied/free label then requires agreement across *all* supplied exposures.
    ``excluded_indices`` declares omitted/incomplete reference regions (for
    example conservative robot bounds). Omissions are counted, never GT solids.
    """
    shapes = tuple(shapes)
    groups = (shapes,)+tuple(tuple(group) for group in other_exposures)
    indices = np.unique(snapshot.spec.validate_indices(indices), axis=0)
    input_voxels = len(indices)
    excluded = _excluded(snapshot.spec, indices, excluded_indices)
    indices = indices[~excluded]
    points = snapshot.spec.indices_to_centers(indices)
    truth = np.array([center_membership(points, group) for group in groups])
    occupied, free = truth.all(0), ~truth.any(0)
    disagreement = ~(occupied | free)
    overlap = np.logical_or.reduce([conservative_cell_overlap(snapshot.spec, indices, group) for group in groups])
    interior = occupied & np.logical_and.reduce([
        conservative_cell_interior(snapshot.spec, indices, group) for group in groups])
    whole_clear, boundary = free & ~overlap, free & overlap
    query = snapshot.query_many(points)
    states = query.states
    active = query.observed & ~query.expired & ~disagreement
    classified = active & (states != OccupancyState.UNKNOWN)
    agree = ((occupied & (states == OccupancyState.OCCUPIED)) | (free & (states == OccupancyState.FREE)))
    result = dict(scope='voxel_centers', input_voxels=input_voxels, excluded_voxels=int(excluded.sum()),
        evaluated_voxels=len(indices), exposure_references=len(groups),
        boundary_policy='Closed solid membership; 1e-12 local-coordinate numerical tolerance.',
        reference_roles={shape.name: shape.reference_role for group in groups for shape in group},
        reference_occupied=_states(states, occupied), reference_free=_states(states, free),
        motion_disagreement=_states(states, disagreement),
        false_free=_rate((occupied & (states == OccupancyState.FREE)).sum(), occupied.sum()),
        conservative_whole_cell_occupied=_states(states, interior),
        center_occupied_partial_or_boundary=_states(states, occupied & ~interior),
        false_free_whole_cell=_rate((interior & (states == OccupancyState.FREE)).sum(), interior.sum()),
        false_occupied=_rate((free & (states == OccupancyState.OCCUPIED)).sum(), free.sum()),
        unknown=_rate((states == OccupancyState.UNKNOWN).sum(), len(states)),
        observed_region=dict(voxels=int(active.sum()), unknown=int((active & (states == OccupancyState.UNKNOWN)).sum()),
            agreement_all_observed=_rate((active & agree).sum(), active.sum()),
            agreement_classified=_rate((classified & agree).sum(), classified.sum())),
        center_free_possible_cell_overlap=_states(states, boundary),
        conservative_whole_cell_clear=_states(states, whole_clear),
        definite_occupied_excess=_rate((whole_clear & (states == OccupancyState.OCCUPIED)).sum(), whole_clear.sum()),
        per_object={})
    for name in sorted({shape.name for group in groups for shape in group}):
        inside = np.array([center_membership(points, [shape for shape in group if shape.name == name]) for group in groups])
        result['per_object'][name] = _states(states, inside.all(0))
        result['per_object'][name]['motion_disagreement_voxels'] = int((inside.any(0) & ~inside.all(0)).sum())
    return result


def score_surface_events(snapshot, points, labels, *, camera_ids=None):
    """Score supplied measured endpoints, keeping event and unique-cell counts.

    Call separately for dense, selected and retained rays. Native instance labels
    identify actual visible surfaces; they never enter production mapping. A
    depth return is an event, not an exact volumetric ground-truth voxel.
    """
    points = _points(points)
    labels = np.asarray(labels, str)
    cameras = np.full(len(points), 'combined') if camera_ids is None else np.asarray(camera_ids, str)
    if labels.shape != (len(points),) or cameras.shape != (len(points),):
        raise ValueError('Surface labels and camera IDs must match point count')
    valid = np.isfinite(points).all(1)
    query = snapshot.query_many(points[valid])
    labels, cameras = labels[valid], cameras[valid]

    def summarize(selected):
        states = query.states[selected]
        indices = query.indices[selected]
        indices = np.unique(indices[(indices >= 0).all(1)], axis=0)
        unique = snapshot.query_many(snapshot.spec.indices_to_centers(indices))
        count = lambda values: {state.name.lower(): int((values == state).sum()) for state in OccupancyState}
        return dict(events=len(states), event_states=count(states), unique_in_bounds_voxels=len(indices),
                    unique_voxel_states=count(unique.states))

    return dict(scope='supplied_surface_endpoint_events', input_events=len(points),
        invalid_events=int((~valid).sum()), finite_events=int(valid.sum()),
        overall=summarize(np.ones(valid.sum(), bool)),
        per_label={name: summarize(labels == name) for name in np.unique(labels)},
        per_camera={name: summarize(cameras == name) for name in np.unique(cameras)})


def score_clear_region(snapshot, points, shapes, *, excluded_indices=()):
    """Known-clear sample preflight and occupancy; no trajectory or planner.

    Only sampled points are certified by the provided analytic references. A
    narrow physically clear gap may contain zero wholly clear map cells at the
    tested voxel resolution; possible wall-overlap cells remain separate.
    """
    points, shapes = _points(points), tuple(shapes)
    finite = np.isfinite(points).all(1)
    query = snapshot.query_many(points)
    in_bounds = (query.indices >= 0).all(1)
    inside = np.zeros(len(points), bool)
    inside[finite] = center_membership(points[finite], shapes)
    errors = []
    if not len(points): errors.append('empty_probe_set')
    if not finite.all(): errors.append('nonfinite_probe_points')
    if not in_bounds.all(): errors.append('out_of_bounds_probe_points')
    if inside.any(): errors.append('probe_intersects_reference_solid')
    excluded_points = _excluded(snapshot.spec, query.indices[in_bounds], excluded_indices)
    if excluded_points.any(): errors.append('probe_in_excluded_reference_region')
    indices = np.unique(query.indices[in_bounds], axis=0)
    centers = snapshot.spec.indices_to_centers(indices)
    cells = snapshot.query_many(centers)
    overlap = conservative_cell_overlap(snapshot.spec, indices, shapes)
    omitted = _excluded(snapshot.spec, indices, excluded_indices)
    return dict(scope='sampled_clear_region', valid=not errors, errors=errors,
        input_points=len(points), solid_points=int(inside.sum()),
        excluded_points=int(excluded_points.sum()),
        in_bounds_points=int(in_bounds.sum()), unique_in_bounds_voxels=len(indices),
        map_states=_states(cells.states, np.ones(len(indices), bool)),
        conservative_whole_cell_clear=_states(cells.states, ~overlap & ~omitted),
        possible_cell_overlap=_states(cells.states, overlap),
        excluded_reference_cells=_states(cells.states, omitted),
        limitation='Samples do not establish clearance of an unsampled continuous volume or a robot path.')
