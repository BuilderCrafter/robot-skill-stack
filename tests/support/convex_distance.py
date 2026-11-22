"""Independent native regression reference; extracted from preserved source cdd2be9.

Numeric algorithms and safety allowances are retained unchanged; no planning input.
"""
from itertools import combinations
import math
import numpy as np
DISTANCE_TOLERANCE_M = 1e-9


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
