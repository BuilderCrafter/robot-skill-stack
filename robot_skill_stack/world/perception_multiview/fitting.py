"""Independent surface hypotheses on unorganized per-object XYZ clouds.

Table-supported boxes use yaw; cylinders may be upright or lying. Arbitrary
tilted boxes are outside this experiment. Scores measure residual/coverage,
not statistical probabilities or the number of overlapping camera samples.
Internal clouds are finite Nx3 arrays; settings and support height come from
the scene configuration.
"""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.common.point_clouds import voxelize


@dataclass
class PrimitiveFit:
    """A fitted primitive's geometry and estimated 3D center and dimensions."""
    geometry: PrimitiveGeometry
    center: np.ndarray
    size: np.ndarray


def unknown_fit(points, *, reason='insufficient_or_ambiguous_surface', support_plane_z=None,
                confidence=0.) -> PrimitiveFit:
    """Bound the current finite XYZ cloud without guessing a primitive."""
    p = np.asarray(points, float).reshape(-1, 3)
    lo, hi = (p.min(0), p.max(0)) if len(p) else (np.zeros(3), np.zeros(3))
    if support_plane_z is not None:
        lo[2] = min(lo[2], support_plane_z)
    geometry = PrimitiveGeometry(PrimitiveShape.UNKNOWN, confidence, metadata={'reason': reason})
    return PrimitiveFit(geometry, (lo+hi)/2, hi-lo)


def _radial(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Robustly fit a circle or sphere center and radius to surface samples."""
    origin = points.mean(0)
    span = float(np.max(np.ptp(points, axis=0)))
    if span < .005:
        return None
    q = (points-origin)/span
    a = np.column_stack((2*q, np.ones(len(q))))
    squared_radius = (q*q).sum(1)
    weights = np.ones(len(q))
    for _ in range(4):
        root = np.sqrt(weights)
        fit, _, rank, singular = np.linalg.lstsq(a*root[:, None], squared_radius*root, rcond=None)
        radius2 = fit[-1] + fit[:-1] @ fit[:-1]
        if rank < a.shape[1] or singular[-1] < singular[0]*1e-4 or radius2 <= 0:
            return None
        radius = np.sqrt(radius2)
        error = abs(np.linalg.norm(q-fit[:-1], axis=1)-radius)
        weights = np.minimum(1., max(.002, 2.5*np.median(error))/np.maximum(error, 1e-12))
    return origin+span*fit[:-1], float(span*radius)


def _hull_area(points: np.ndarray) -> float:
    """Calculate the area of the convex hull of 2D points."""
    # unique also supplies the lexicographic order for the monotone-chain hull.
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return 0.
    def boundary(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2:
                a, b = result[-1]-result[-2], point-result[-1]
                if a[0]*b[1]-a[1]*b[0] > 0:
                    break
                result.pop()
            result.append(point)
        return result
    h = np.asarray(boundary(points)[:-1]+boundary(points[::-1])[:-1])
    return float(abs(h[:, 0] @ np.roll(h[:, 1], -1)-h[:, 1] @ np.roll(h[:, 0], -1))/2)


def _box_distance(points: np.ndarray, center: np.ndarray, rotation: np.ndarray,
                  size: np.ndarray) -> np.ndarray:
    """Measure each point's distance from an oriented box surface."""
    q = abs((points-center) @ rotation)-size/2
    return abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))


def _box_search(points: np.ndarray, side_mask: np.ndarray, angles: np.ndarray) -> tuple:
    """Transform each yaw once; score side samples within full-cloud bounds."""
    c, s = np.cos(angles), np.sin(angles)
    xy = np.stack((points[:, 0, None]*c+points[:, 1, None]*s,
                   -points[:, 0, None]*s+points[:, 1, None]*c), axis=2)
    lo, hi = np.percentile(xy, [.1, 99.9], axis=0)
    local_sides = xy[side_mask]
    error = np.percentile(np.minimum(abs(local_sides-lo), abs(local_sides-hi)).min(2), 75, axis=0)
    index = int(np.argmin(error))
    yaw = float(angles[index])
    r = np.array([[c[index], -s[index], 0.], [s[index], c[index], 0.], [0., 0., 1.]])
    return yaw, r, lo[index], hi[index]


def _radius_evidence(points: np.ndarray, fit: PrimitiveFit, plane: float) -> np.ndarray | None:
    """Measured radii under the supported shape's existing contact prior."""
    g = fit.geometry
    if g.shape == PrimitiveShape.SPHERE:
        upper = points[points[:, 2]-plane >= g.radius]
        if len(upper) < 20:
            return None
        z = upper[:, 2]-plane
        return (((upper[:, :2]-fit.center[:2])**2).sum(1)+z*z)/(2*z)
    axis = g.axis
    along = (points-fit.center) @ axis
    core = points[abs(along) < .35*g.length]
    if len(core) < 20:
        return None
    if abs(axis[2]) > .999:
        return np.linalg.norm(core[:, :2]-fit.center[:2], axis=1)
    upper = core[core[:, 2]-plane >= g.radius]
    if len(upper) < 20:
        return None
    side = np.cross([0., 0., 1.], axis)
    horizontal = (upper-fit.center) @ side
    z = upper[:, 2]-plane
    return (horizontal*horizontal+z*z)/(2*z)


def _cylinder_evidence_error(points: np.ndarray, fit: PrimitiveFit, tolerance: float) -> str | None:
    """Require curvature and measured cap interiors before using dimensions."""
    g = fit.geometry
    delta = points-fit.center
    along = delta @ g.axis
    core = delta[abs(along) < .35*g.length]
    if len(core) < 20:
        return 'insufficient_curved_surface'
    vertical = abs(g.axis[2]) > .999
    cross = core[:, :2] if vertical else np.c_[core @ [-g.axis[1], g.axis[0], 0.], core[:, 2]]
    angles = np.sort(np.arctan2(cross[:, 1], cross[:, 0]))
    coverage = float(2*np.pi-np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
    # Upright cross sections have no measured centre prior: less than a half
    # circle may fit a small-radius arc deceptively well. Horizontal cylinders
    # additionally have the configured support-contact constraint.
    required = np.pi if vertical else np.pi/2
    if coverage < required:
        return 'insufficient_curvature_extent'
    radial = np.linalg.norm(delta-along[:, None]*g.axis, axis=1)
    interior = radial < .85*g.radius
    ends = [1.] if vertical else [-1., 1.]
    caps = [int(np.count_nonzero(interior & (abs(along-end*g.length/2) < 2*tolerance))) for end in ends]
    if min(caps) < 5:
        return 'unobserved_axial_endpoint'
    return None


class SurfaceFitter:
    """Compare primitive surface fits on unorganized 3D object clouds.

    Return UNKNOWN when no shape is sufficiently confident or distinct.
    """
    def __init__(self, *, tolerance=.0015, threshold=.65, margin=.08, max_points=1200):
        self.tolerance, self.threshold, self.margin = tolerance, threshold, margin
        self.max_points = max_points

    def _candidate(self, shape: str, center: np.ndarray, size: np.ndarray, residual: np.ndarray,
                   coverage=1., **parameters) -> PrimitiveFit:
        """Build a primitive result scored by surface residual and coverage."""
        error = float(np.percentile(residual, 80))
        score = float(np.exp(-.5*(error/self.tolerance)**2)*np.clip(coverage, 0, 1))
        geometry = PrimitiveGeometry(shape, score, **parameters)
        return PrimitiveFit(geometry, center, size)

    def _sphere(self, p: np.ndarray, plane: float) -> PrimitiveFit | None:
        """Fit a sphere using sampled surface curvature and support-plane checks."""
        fit = _radial(p)
        if fit is None:
            return None
        center, radius = fit
        if not .005 <= radius <= .10 or center[2]-radius < plane-.004:
            return None
        directions = (p-center)/radius
        coverage = min(1., float(np.linalg.norm(np.ptp(directions, axis=0)))/1.8)
        return self._candidate('sphere', center, np.full(3, 2*radius),
                               abs(np.linalg.norm(p-center, axis=1)-radius), coverage, radius=radius)

    def _box(self, p: np.ndarray, plane: float) -> PrimitiveFit | None:
        """Fit an upright box using top-plane and side-wall evidence."""
        high = float(np.percentile(p[:, 2], 99.8))
        upper = p[p[:, 2] > high-.010]
        bins = np.floor(upper[:, 2]/self.tolerance).astype(int)
        values, counts = np.unique(bins, return_counts=True)
        level = values[np.argmax(counts)]
        top_z = float(np.median(upper[bins == level, 2]))
        top = p[abs(p[:, 2]-top_z) < self.tolerance]
        sides = p[:, 2] < top_z-3*self.tolerance
        height = top_z-plane
        if len(top) < 15 or np.count_nonzero(sides) < 15 or height < .008:
            return None
        # Surface distances select yaw, rather than PCA of biased partial faces.
        # Coarse coverage spans the full supported interval. Refinement uses all
        # samples; sparse curved surfaces must still compete against box fits.
        yaw, r, lo, hi = _box_search(p, sides, np.deg2rad(np.arange(0, 90, 5.)))
        for width, step in ((5., .5), (.5, .05)):
            angles = yaw+np.deg2rad(np.arange(-width, width+step/2, step))
            yaw, r, lo, hi = _box_search(p, sides, angles)
        local_size = np.r_[hi-lo, height]
        if min(local_size) < .007 or max(local_size) > .20:
            return None
        center = np.r_[((lo+hi)/2) @ r[:2, :2].T, plane+height/2]
        # Circular caps and isolated planar patches must not become boxes.
        fill = _hull_area(top[:, :2])/max(float(np.prod(hi-lo)), 1e-12)
        coverage = np.clip((fill-.78)/.16, 0, 1)
        return self._candidate('cube', center, abs(r) @ local_size,
                               _box_distance(p, center, r, local_size), coverage,
                               box_size=local_size, yaw=float(yaw))

    def _cylinder_axis(self, p: np.ndarray, plane: float, axis) -> PrimitiveFit | None:
        """Fit a circle cross-section for an upright or horizontal unit axis."""
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        vertical = abs(axis[2]) > .999
        if vertical:
            lo, hi = plane, float(np.percentile(p[:, 2], 99.8))
            axial = p[:, 2]
            cross_points = p[:, :2]
        else:
            side = np.array([-axis[1], axis[0], 0.])
            axial = p @ axis
            lo, hi = np.percentile(axial, [.1, 99.9])
            cross_points = np.c_[p @ side, p[:, 2]]
        length = float(hi-lo)
        if length < .01:
            return None
        core_mask = (axial > lo+.15*length) & (axial < hi-.15*length)
        core = cross_points[core_mask]
        if len(core) < 20:
            return None
        fit = _radial(core)
        if fit is None:
            return None
        cross, radius = fit
        if not .005 <= radius <= .10 or (not vertical and length < 1.5*radius):
            return None
        center = (np.r_[cross, (lo+hi)/2] if vertical else
                  axis*((lo+hi)/2)+side*cross[0]+[0., 0., cross[1]])
        extent = abs(axis)*length/2+radius*np.sqrt(np.maximum(1-axis*axis, 0))
        if center[2]-extent[2] < plane-.004:
            return None
        radial = np.linalg.norm(cross_points-cross, axis=1)
        q = np.column_stack((radial-radius, abs(axial-(lo+hi)/2)-length/2))
        residual = abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))
        side_error = abs(radial[core_mask]-radius)
        side_quality = np.exp(-.5*(np.percentile(side_error, 80)/self.tolerance)**2)
        return self._candidate('cylinder', center, 2*extent, residual, side_quality,
                               radius=radius, axis=axis, length=length)

    def _lying_cylinder(self, p: np.ndarray, plane: float) -> PrimitiveFit | None:
        """Cover all horizontal axes, then refine the strongest surface fit."""
        best = None
        visited = set()
        for width, step in ((None, None), (10., 2.), (2., .25)):
            if width is None:
                angles = np.deg2rad(np.arange(0, 180, 10.))
            elif best is None:
                break
            else:
                angle = np.arctan2(best.geometry.axis[1], best.geometry.axis[0])
                angles = angle+np.deg2rad(np.arange(-width, width+step/2, step))
            for angle in angles:
                token = round(float(angle % np.pi), 10)
                if token in visited:
                    continue
                visited.add(token)
                fit = self._cylinder_axis(p, plane, [np.cos(angle), np.sin(angle), 0.])
                if fit is not None and (best is None or fit.geometry.confidence > best.geometry.confidence):
                    best = fit
        return best

    def finalize(self, fit: PrimitiveFit, points: np.ndarray, *, support_plane_z=0.,
                 view_clouds=()) -> PrimitiveFit:
        """Qualify measured outer dimensions without changing the cached raw fit.

        Extents include measured residual spread, not a calibrated confidence
        interval. Graspability requires supported dimensions and, for boxes,
        the provider's separate camera-boundary assessment.
        """
        result = PrimitiveFit(deepcopy(fit.geometry), fit.center.copy(), fit.size.copy())
        g = result.geometry
        p = points
        g.metadata['parameter_observable'] = False
        if g.shape == PrimitiveShape.UNKNOWN or len(p) < 30:
            g.metadata['parameter_reason'] = 'unknown_or_insufficient_current_surface'
            return result
        if g.shape == PrimitiveShape.CUBE:
            g.metadata['fitted_top_z'] = float(fit.center[2]+g.box_size[2]/2)
            c, s = np.cos(g.yaw), np.sin(g.yaw)
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            local = p @ rotation
            half = g.box_size/2
            old_center = fit.center @ rotation
            padding = float(np.percentile(_box_distance(p, fit.center, rotation, g.box_size), 95))
            lo = np.minimum(local.min(0), old_center-half)-padding
            hi = np.maximum(local.max(0), old_center+half)+padding
            lo[2] = support_plane_z
            dimensions = hi-lo
            if padding > self.tolerance or np.max(dimensions-g.box_size) > 2*self.tolerance:
                g.metadata['parameter_reason'] = 'inconsistent_box_surface_extent'
                return result
            result.center = ((lo+hi)/2) @ rotation.T
            g.box_size = dimensions
            result.size = abs(rotation) @ dimensions
        else:
            contact = g.shape == PrimitiveShape.SPHERE or abs(g.axis[2]) < .001
            if contact and abs(fit.center[2]-g.radius-support_plane_z) > .004:
                g.metadata['parameter_reason'] = 'unobserved_support_contact'
                return result
            if g.shape == PrimitiveShape.CYLINDER:
                reason = _cylinder_evidence_error(p, fit, self.tolerance)
                if reason is not None:
                    g.metadata['parameter_reason'] = reason
                    return result
            samples = _radius_evidence(p, fit, support_plane_z)
            if samples is None:
                g.metadata['parameter_reason'] = 'insufficient_radius_evidence'
                return result
            median, high = np.percentile(samples, [50, 95])
            outer = float(max(g.radius, 2*high-median))
            view_extents = []
            for cloud in view_clouds:
                values = _radius_evidence(cloud, fit, support_plane_z)
                if values is not None:
                    middle, upper = np.percentile(values, [50, 95])
                    view_extents.append(float(2*upper-middle))
            if view_extents:
                outer = max(outer, max(view_extents))
            if (outer-g.radius > .004 or
                    (view_extents and np.ptp(view_extents) > 2*self.tolerance)):
                g.metadata['parameter_reason'] = 'inconsistent_radius_evidence'
                return result
            g.radius = outer
            if contact:
                result.center[2] = support_plane_z+outer
            if g.shape == PrimitiveShape.SPHERE:
                result.size = np.full(3, 2*outer)
            else:
                result.size = abs(g.axis)*g.length+2*outer*np.sqrt(np.maximum(1-g.axis*g.axis, 0))
        g.metadata['parameter_observable'] = True
        return result

    def fit(self, points: np.ndarray, *, support_plane_z=0.) -> PrimitiveFit:
        """Fit candidate primitives and choose the best supported shape."""
        p = voxelize(points, .0015, self.max_points)
        scores = dict(cube=0., sphere=0., cylinder=0.)
        fits = []
        if len(p) >= 30:
            fits = [f for f in (self._sphere(p, support_plane_z), self._box(p, support_plane_z),
                                self._cylinder_axis(p, support_plane_z, [0., 0., 1.])) if f is not None]
            best = self._lying_cylinder(p, support_plane_z)
            if best is not None:
                fits.append(best)
            for fit in fits:
                shape = fit.geometry.shape.value
                scores[shape] = max(scores[shape], fit.geometry.confidence)
        ranked = sorted(scores.values(), reverse=True)
        if fits and ranked[0] >= self.threshold and ranked[0]-ranked[1] >= self.margin:
            result = max(fits, key=lambda f: f.geometry.confidence)
        else:
            result = unknown_fit(p, support_plane_z=support_plane_z, confidence=ranked[0])
        result.geometry.scores = scores
        return result
