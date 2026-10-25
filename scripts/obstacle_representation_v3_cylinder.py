"""Independent cooked-convex reference for the research cylindrical fixture.

The native fixture is an explicit regular32-sided convex prism. Solid convex
hulls of its actual PhysX-cooked vertices and the cooked robot vertices are
scored after motion. Rounded cooked polygon fans can differ by micrometers
from their vertex-set convex hull; the numerical certificate describes the
vertex-hull query, not physical model fidelity or contact offsets.
No reference geometry enters RGB-D extraction, selection or control. GJK uses
support vertices and a closest-simplex calculation, with distance upper/lower
bounds; it never substitutes the fixture's enclosing box for clearance.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np


DISTANCE_TOLERANCE_M = 1e-9


def unit_prism(sides=32):
    """Return vertices/ordered faces for diameter1,height1,axisZ,center0."""
    if type(sides) is not int or not 8 <= sides <= 256:
        raise ValueError('Prism sides must be an integer within [8,256]')
    angles = 2*np.pi*np.arange(sides)/sides
    ring = np.column_stack((.5*np.cos(angles), .5*np.sin(angles)))
    vertices = np.vstack([np.column_stack((ring, np.full(sides, z))) for z in (-.5, .5)])
    faces = [list(range(sides-1, -1, -1)), list(range(sides, 2*sides))]
    faces += [[i, (i+1) % sides, (i+1) % sides+sides, i+sides] for i in range(sides)]
    return vertices, np.array([len(face) for face in faces], np.int64), np.array(
        [value for face in faces for value in face], np.int64)


def _vertices(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Convex reference requires nonempty finite Nx3 vertices')
    return result


def _closest_simplex(points):
    """Closest point in a <=5-point hull by bounded face enumeration.

    Caratheodory's theorem bounds a3D convex combination to four points. Each
    enumerated affine projection is accepted only with nonnegative weights;
    vertices and lower-dimensional faces cover singular/degenerate simplices.
    """
    points = _vertices(points)
    best, selected = math.inf, None
    for size in range(1, min(4, len(points))+1):
        for indices in combinations(range(len(points)), size):
            face = points[list(indices)]
            if size == 1:
                weights = np.ones(1)
            else:
                values = np.linalg.lstsq((face[1:]-face[0]).T, -face[0], rcond=None)[0]
                weights = np.r_[1-values.sum(), values]
                if np.min(weights) < -1e-12:
                    continue
                weights = np.maximum(weights, 0.)
                weights /= weights.sum()
            point = weights@face
            squared = float(point@point)
            if squared < best-1e-26:
                chosen = weights > 1e-13
                reduced_weights = weights[chosen]/weights[chosen].sum()
                reduced = face[chosen]
                best, selected = squared, (reduced_weights@reduced, reduced)
    if selected is None:
        raise RuntimeError('No feasible closest convex-simplex point')
    return selected


def convex_distance_details(first, second, *, tolerance=DISTANCE_TOLERANCE_M, max_iterations=128):
    """GJK solid distance with an absolute primal/dual separation certificate."""
    first, second = _vertices(first), _vertices(second)
    if not np.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('GJK requires positive finite tolerance and iteration bound')

    def support(direction):
        return first[np.argmax(first@direction)]-second[np.argmin(second@direction)]

    direction = second.mean(0)-first.mean(0)
    if np.linalg.norm(direction) < tolerance:
        direction = np.array([1., 0., 0.])
    simplex = np.array([support(direction)])
    closest = simplex[0]
    lower = 0.
    for iteration in range(1, max_iterations+1):
        upper = float(np.linalg.norm(closest))
        if upper <= tolerance:
            return dict(distance_m=0., lower_bound_m=0., upper_bound_m=upper,
                        iterations=iteration, converged=True)
        point = support(-closest)
        lower = max(lower, 0., float(closest@point)/upper)
        # Floating support/projection arithmetic may differ below machine eps.
        if upper-lower <= tolerance:
            return dict(distance_m=upper, lower_bound_m=min(lower, upper), upper_bound_m=upper,
                        iterations=iteration, converged=True)
        if any(np.array_equal(point, value) for value in simplex):
            raise RuntimeError('GJK repeated support without a valid distance certificate')
        closest, simplex = _closest_simplex(np.vstack((simplex, point)))
    raise RuntimeError(f'GJK did not converge within {max_iterations} iterations; last lower bound {lower}')


def convex_distance(first, second, **kwargs):
    return convex_distance_details(first, second, **kwargs)['distance_m']


def reduce_cylinder_clearances(robotmeshes, body_paths, body_transforms, boxes,
                                cylinder_hulls, cylinder_body_path):
    """Match the existing post-motion reducer shape using actual convex hulls."""
    if not robotmeshes or any(not part.approximation.startswith('physx_cooked:') for part in robotmeshes):
        raise ValueError('Cylinder comparison requires actual cooked convex robot hulls; no authored fallback')
    hulls = [_vertices(hull[0]) for hull in cylinder_hulls]
    if not hulls:
        raise ValueError('Actual cooked cylinder hulls are required')
    indices = {path: index for index, path in enumerate(body_paths)}
    if cylinder_body_path not in indices or len(body_transforms) != len(boxes):
        raise ValueError('Cylinder reference/body pose samples do not match')
    if any(len(row) != 1 or row[0][0] != cylinder_body_path for row in boxes):
        raise ValueError('Cylinder reducer only supports the explicitly isolated cylinder fixture')
    per_link = {part.body_path: math.inf for part in robotmeshes}
    per_group = {part.group: math.inf for part in robotmeshes}
    seen, closest, minimum = {}, None, math.inf
    evaluations = iterations = duplicate_skips = zero_skips = bound_skips = 0
    largest_certificate_width = 0.
    bounds = [(hull.min(0), hull.max(0)) for hull in hulls]
    for sample, poses in enumerate(body_transforms):
        poses = np.asarray(poses, float)
        cylinder_from_world = np.linalg.inv(poses[indices[cylinder_body_path]])
        for part in robotmeshes:
            if per_link[part.body_path] == 0.:
                zero_skips += len(hulls)
                continue
            transform = cylinder_from_world@poses[indices[part.body_path]]@part.body_from_mesh
            key = transform.tobytes()
            previous = seen.setdefault(part.path, set())
            if key in previous:
                duplicate_skips += len(hulls)
                continue
            previous.add(key)
            vertices = part.mesh.vertices@transform[:3, :3].T+transform[:3, 3]
            lo, hi = vertices.min(0), vertices.max(0)
            for hull, (hlo, hhi) in zip(hulls, bounds):
                bound = float(np.linalg.norm(np.maximum(np.maximum(lo-hhi, hlo-hi), 0.)))
                if bound >= per_link[part.body_path]:
                    bound_skips += 1
                    continue
                measured = convex_distance_details(vertices, hull)
                distance = measured['distance_m']
                largest_certificate_width = max(largest_certificate_width,
                    measured['upper_bound_m']-measured['lower_bound_m'])
                evaluations += 1
                iterations += measured['iterations']
                per_link[part.body_path] = min(per_link[part.body_path], distance)
                per_group[part.group] = min(per_group[part.group], distance)
                if distance < minimum:
                    minimum = distance
                    closest = dict(sample=sample, robot_link=part.body_path, mesh=part.path,
                                   obstacle=cylinder_body_path, clearance_m=distance)
    finite = lambda value: float(value) if np.isfinite(value) else None
    return dict(minimum_m=finite(minimum), by_group_m={key:finite(value) for key,value in per_group.items()},
        by_link_m={key:finite(value) for key,value in per_link.items()}, closest_pair=closest,
        mesh_box_evaluations=0, convex_convex_evaluations=evaluations,
        sampled_poses=len(body_transforms), exact_duplicate_pose_skips=duplicate_skips,
        zero_minimum_skips=zero_skips, aabb_lower_bound_skips=bound_skips,
        gjk_iterations=iterations, maximum_distance_certificate_width_m=largest_certificate_width,
        distance_tolerance_m=DISTANCE_TOLERANCE_M,
        reference='Solid convex hulls of actual PhysX-cooked robot and32-sided cylindrical-fixture vertices',
        limitation='Sampled unsigned solid distance with1nm numerical certificate; excludes contact/rest offsets and does not certify continuous motion. No enclosing-box clearance substitute.')


def _planes(hull):
    vertices, counts, indices = hull
    vertices = _vertices(vertices)
    counts, indices = np.asarray(counts), np.asarray(indices)
    if (counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 3)
            or indices.ndim != 1 or indices.dtype.kind not in 'iu' or counts.sum() != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(vertices))):
        raise ValueError('Invalid cooked convex face topology')
    center, result, cursor = vertices.mean(0), [], 0
    for count in counts:
        face = vertices[indices[cursor:cursor+count]]
        cursor += count
        vectors = face[1:]-face[0]
        candidates = np.cross(vectors[:-1], vectors[1:])
        normal = candidates[np.argmax(np.linalg.norm(candidates, axis=1))]
        length = float(np.linalg.norm(normal))
        if length < 1e-20:
            raise ValueError('Degenerate cooked convex face')
        normal /= length
        if normal@(face.mean(0)-center) < 0:
            normal *= -1
        result.append((normal, float(np.max(vertices@normal))))
    return np.array([row[0] for row in result]), np.array([row[1] for row in result])


def depth_reference_scores_convex(frames, hulls, center):
    """Score stride-grid optical depths against the actual static cooked prism."""
    world_from_body = np.asarray(center, float)
    if world_from_body.shape == (3,):
        position = world_from_body
        world_from_body = np.eye(4)
        world_from_body[:3, 3] = position
    if world_from_body.shape != (4, 4) or not np.isfinite(world_from_body).all():
        raise ValueError('Expected actual static cylinder XYZ or world pose')
    planes = [_planes(hull) for hull in hulls]
    if not planes:
        raise ValueError('Cooked cylinder topology required')
    body_from_world = np.linalg.inv(world_from_body)
    rows = []
    for frame in frames:
        y, x = np.indices(frame.depth[::8, ::8].shape)
        y, x = y.ravel()*8, x.ravel()*8
        k = frame.intrinsics
        camera_rays = np.column_stack(((x+.5-k[0, 2])/k[0, 0],
                                      (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))))
        body_from_camera = body_from_world@frame.world_from_camera
        directions = camera_rays@body_from_camera[:3, :3].T
        origin = body_from_camera[:3, 3]
        first = np.full(len(x), np.inf)
        for normals, offsets in planes:
            denominator = directions@normals.T
            numerator = offsets-origin@normals.T
            parallel = np.abs(denominator) < 1e-15
            valid = ~np.any(parallel & (numerator < -1e-12), axis=1)
            ratios = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=~parallel)
            near = np.where(denominator < -1e-15, ratios, -np.inf).max(1)
            far = np.where(denominator > 1e-15, ratios, np.inf).min(1)
            valid &= (far >= near) & (near > 0)
            first = np.minimum(first, np.where(valid, near, np.inf))
        potential = np.isfinite(first)
        measured = frame.depth[y, x]
        valid = np.isfinite(measured) & (measured > 0)
        rows.append(dict(camera_id=frame.camera_id, obstacle='ThinCylinder',
            original_timestamp=frame.acquired_at, frame_token=frame.token,
            potential_stride_grid_rays=int(potential.sum()),
            visible_stride_grid_returns=int(np.count_nonzero(potential & valid & (np.abs(measured-first) <= .015))),
            occluded_stride_grid_returns=int(np.count_nonzero(potential & valid & (measured < first-.015))),
            invalid_stride_grid_returns=int(np.count_nonzero(potential & ~valid))))
    return dict(rows=rows, stride=8, depth_match_tolerance_m=.015,
        reference='Actual PhysX-cooked static cylindrical prism; evaluation-only analytic convex halfspace intersections.',
        meaning='Stride-grid surface visibility; not mapping-ray allocation, completed occupancy or a clearance proof.')
