"""NumPy surface fits; inputs are visible samples, not complete solids."""
from __future__ import annotations
import numpy as np


def hull2d(points):
    points = np.unique(np.asarray(points, float), axis=0)
    if len(points) < 3:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(sequence):
        result = []
        for p in sequence:
            while len(result) >= 2:
                a, b = result[-1] - result[-2], p - result[-1]
                if a[0]*b[1] - a[1]*b[0] > 0:
                    break
                result.pop()
            result.append(p)
        return result

    return np.array(half(points)[:-1] + half(points[::-1])[:-1])


def area(hull):
    if len(hull) < 3:
        return 0.
    p = hull - hull.mean(0)
    return float(abs(p[:, 0] @ np.roll(p[:, 1], -1) - p[:, 1] @ np.roll(p[:, 0], -1))/2)


def rectangle(hull):
    if len(hull) < 3 or area(hull) < 1e-10:
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi/2))
    best = None
    for yaw in angles:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        uv = hull @ R
        lo, hi = uv.min(0), uv.max(0)
        size = hi - lo
        volume = float(np.prod(size))
        if best is None or volume < best[0]:
            best = (volume, float(yaw), size, ((hi+lo)/2) @ R.T)
    volume, yaw, size, center = best
    return yaw, size, center, float(area(hull)/max(volume, 1e-12))


def box_footprint(points, top_z, tolerance):
    sides = points[points[:, 2] < top_z-3*tolerance, :2]
    if len(sides) < 12:
        return rectangle(hull2d(points[:, :2]))
    best = None
    for yaw in np.linspace(0, np.pi/2, 181, endpoint=False):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        uv, walls = points[:, :2] @ R, sides @ R
        lo, hi = uv.min(0), uv.max(0)
        distances = np.minimum(abs(walls-lo), abs(walls-hi)).min(1)
        error = float(np.percentile(distances, 75))
        if best is None or error < best[0]:
            best = error, yaw, hi-lo, ((lo+hi)/2) @ R.T
    _, yaw, size, center = best
    return float(yaw), size, center, float(area(hull2d(points[:, :2]))/np.prod(size))


def radial_fit(points):
    """Fit circle (N,2) or sphere (N,3), solving for center as well as radius."""
    p = np.asarray(points, float)
    if len(p) < 12:
        return None
    origin = p.mean(0)
    scale = max(float(np.max(np.ptp(p, axis=0))), 1e-9)
    q = (p-origin)/scale
    A = np.column_stack((2*q, np.ones(len(q))))
    b = (q*q).sum(1)
    weights = np.ones(len(q))
    for _ in range(4):
        root = np.sqrt(weights)
        x, _, rank, singular = np.linalg.lstsq(A*root[:, None], b*root, rcond=None)
        if rank != A.shape[1] or singular[-1] < singular[0]*1e-4:
            return None
        radius_sq = x[-1] + x[:-1] @ x[:-1]
        if radius_sq <= 0:
            return None
        radius = np.sqrt(radius_sq)
        residual = abs(np.linalg.norm(q-x[:-1], axis=1)-radius)
        cutoff = max(2.5*float(np.median(residual)), .002)
        weights = np.minimum(1., cutoff/np.maximum(residual, 1e-12))
    return origin+scale*x[:-1], float(scale*radius)


def quality(residual, tolerance):
    residual = np.asarray(residual, float)
    if not len(residual) or not np.isfinite(residual).all():
        return 0.
    error = float(np.percentile(abs(residual), 80))
    return float(np.exp(-.5*(error/tolerance)**2))


def cylinder_residual(points, center, axis, radius, length):
    delta = points-center
    axial = delta @ axis
    radial = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
    q = np.column_stack((radial-radius, abs(axial)-length/2))
    return abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))


def circle_boundary(hull):
    if len(hull) < 6:
        return None
    # Mid-edge samples distinguish a rectangle from a circle through its corners.
    nxt = np.roll(hull, -1, axis=0)
    samples = np.concatenate([(1-t)*hull+t*nxt for t in (0., .25, .5, .75)])
    fit = radial_fit(samples)
    if fit is None:
        return None
    center, radius = fit
    residual = abs(np.linalg.norm(samples-center, axis=1)-radius)
    return center, radius, residual
