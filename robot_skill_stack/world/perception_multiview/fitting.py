"""Independent surface hypotheses on unorganized per-object XYZ clouds.

Table-supported boxes use yaw; cylinders may be upright or lying. Arbitrary
tilted boxes are outside this experiment. Scores measure residual/coverage,
not statistical probabilities or the number of overlapping camera samples.
"""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.perception_multiview.clouds import voxelize


@dataclass
class PrimitiveFit:
    geometry: PrimitiveGeometry
    center: np.ndarray
    size: np.ndarray


def unknown_fit(points, *, reason='insufficient_or_ambiguous_surface', support_plane_z=None,
                confidence=0., scores=None):
    """Describe current finite samples without model search or stored geometry."""
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    lo, hi = (p.min(0), p.max(0)) if len(p) else (np.zeros(3), np.zeros(3))
    if support_plane_z is not None:
        if not np.isfinite(support_plane_z):
            raise ValueError('Support plane must be finite')
        lo[2] = min(lo[2], support_plane_z)
    geometry = PrimitiveGeometry(PrimitiveShape.UNKNOWN, confidence, scores=dict(scores or {}),
                                 metadata={'reason': str(reason), 'fit_points': len(p)})
    return PrimitiveFit(geometry, (lo+hi)/2, hi-lo)


def orientation_metadata(fit, points):
    """Current empirical side intervals, without a hard near-square threshold."""
    geometry = fit.geometry
    if geometry.shape.value == 'sphere':
        return dict(face_grid_observable=False, major_axis_observable=False,
                    reason='featureless_sphere_has_no_geometric_orientation', yaw_hypothesis_rad=None)
    if geometry.shape.value == 'cylinder':
        axis = geometry.axis
        observed = bool(geometry.metadata.get('parameter_observable', False))
        return dict(face_grid_observable=False, major_axis_observable=False,
            axis_observable=observed, axis_hypothesis=None if axis is None else axis.tolist(),
            axis_direction_period_degrees=180, rotation_about_axis_observable=False,
            yaw_hypothesis_rad=None, reason='existing_cylinder_axis_and_parameter_evidence',
            scope='Existing upright/lying cylinder axis support unchanged; no angular confidence interval')
    if geometry.shape.value != 'cube':
        return dict(face_grid_observable=False, major_axis_observable=False, reason='not_a_known_box')
    evidence = geometry.metadata.get('geometric_observability', {})
    bounded = bool(geometry.metadata.get('parameter_observable', False) and evidence.get('observable', False))
    c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    bounded = bool(bounded and len(p) >= 30)
    result = dict(face_grid_observable=bounded, face_grid_period_degrees=90,
        major_axis_observable=False, major_axis_period_degrees=180,
        yaw_hypothesis_rad=float(geometry.yaw), major_axis_yaw_rad=None,
        reason='unbounded_current_footprint', side_intervals_m=None,
        scope='Empirical side-order evidence under fitted supported-box/ray model; no angular confidence interval')
    if len(p) < 30:
        result['reason'] = 'insufficient_current_points'
        return result
    span = np.ptp((p-fit.center) @ rotation, axis=0)[:2]
    padding = float(geometry.metadata.get('parameter_uncertainty', {}).get('surface_padding_m', 0.))
    # Observed span, with the existing measured residual margin, supplies the
    # lower side evidence. Published current outward extents supply the upper.
    # These are empirical intervals, not certified uncertainty on hidden shapes.
    lower = np.maximum(0., span-2*padding)
    upper = np.maximum(geometry.box_size[:2], span+2*padding)
    result['side_intervals_m'] = np.c_[lower, upper].tolist()
    numerical = 16*np.finfo(float).eps*max(upper)
    major = 0 if lower[0]-upper[1] > numerical else 1 if lower[1]-upper[0] > numerical else None
    if bounded and major is not None:
        result.update(major_axis_observable=True, major_axis_yaw_rad=float(geometry.yaw+major*np.pi/2),
                      reason='distinct_current_side_intervals')
    elif bounded:
        result['reason'] = 'overlapping_current_side_intervals'
    return result


def _radial(points):
    origin = points.mean(0)
    span = float(np.max(np.ptp(points, axis=0)))
    if len(points) < 12 or span < .005:
        return None
    q = (points-origin)/span
    a = np.column_stack((2*q, np.ones(len(q))))
    weights = np.ones(len(q))
    for _ in range(4):
        root = np.sqrt(weights)
        fit, _, rank, singular = np.linalg.lstsq(a*root[:, None], (q*q).sum(1)*root, rcond=None)
        radius2 = fit[-1] + fit[:-1] @ fit[:-1]
        if rank < a.shape[1] or singular[-1] < singular[0]*1e-4 or radius2 <= 0:
            return None
        radius = np.sqrt(radius2)
        error = abs(np.linalg.norm(q-fit[:-1], axis=1)-radius)
        weights = np.minimum(1., max(.002, 2.5*np.median(error))/np.maximum(error, 1e-12))
    return origin+span*fit[:-1], float(span*radius)


def _hull_area(points):
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return 0.
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
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


def _box_distance(points, center, rotation, size):
    q = abs((points-center) @ rotation)-size/2
    return abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))


def _box_search(points, sides, angles):
    """Batch yaw hypotheses while retaining full-cloud percentile bounds."""
    c, s = np.cos(angles), np.sin(angles)
    xy = np.stack((points[:, 0, None]*c+points[:, 1, None]*s,
                   -points[:, 0, None]*s+points[:, 1, None]*c), axis=2)
    lo, hi = np.percentile(xy, [.1, 99.9], axis=0)
    local_sides = np.stack((sides[:, 0, None]*c+sides[:, 1, None]*s,
                            -sides[:, 0, None]*s+sides[:, 1, None]*c), axis=2)
    error = np.percentile(np.minimum(abs(local_sides-lo), abs(local_sides-hi)).min(2), 75, axis=0)
    index = int(np.argmin(error))
    yaw = float(angles[index])
    r = np.array([[c[index], -s[index], 0.], [s[index], c[index], 0.], [0., 0., 1.]])
    return yaw, r, lo[index], hi[index]


def _radius_evidence(points, fit, plane):
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


def _cylinder_observability(points, fit, tolerance):
    """Require curvature and measured cap interiors before using dimensions."""
    g = fit.geometry
    delta = points-fit.center
    along = delta @ g.axis
    core = delta[abs(along) < .35*g.length]
    if len(core) < 20:
        return False, {'reason': 'insufficient_curved_surface'}
    reference = np.eye(3)[np.argmin(abs(g.axis))]
    side = np.cross(g.axis, reference)
    side /= np.linalg.norm(side)
    other = np.cross(g.axis, side)
    angles = np.sort(np.arctan2(core @ other, core @ side))
    coverage = float(2*np.pi-np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
    # Upright cross sections have no measured centre prior: less than a half
    # circle may fit a small-radius arc deceptively well. Horizontal cylinders
    # additionally have the configured support-contact constraint.
    required = np.pi if abs(g.axis[2]) > .999 else np.pi/2
    radial = np.linalg.norm(delta-along[:, None]*g.axis, axis=1)
    interior = radial < .85*g.radius
    ends = [1.] if abs(g.axis[2]) > .999 else [-1., 1.]
    caps = [int(np.count_nonzero(interior & (abs(along-end*g.length/2) < 2*tolerance))) for end in ends]
    evidence = {'curvature_span_rad': coverage, 'cap_interior_points': caps}
    if coverage < required:
        return False, dict(evidence, reason='insufficient_curvature_extent')
    if min(caps) < 5:
        return False, dict(evidence, reason='unobserved_axial_endpoint')
    return True, evidence


class SurfaceFitter:
    def __init__(self, *, tolerance=.0015, threshold=.65, margin=.08, max_points=1200):
        if (not np.isfinite(tolerance) or tolerance <= 0 or not 0 < threshold <= 1 or
                not 0 <= margin < 1 or max_points < 30):
            raise ValueError('Invalid surface fitting settings')
        self.tolerance, self.threshold, self.margin = tolerance, threshold, margin
        self.max_points = max_points

    def _candidate(self, shape, center, size, residual, coverage=1., **parameters):
        error = float(np.percentile(residual, 80))
        score = float(np.exp(-.5*(error/self.tolerance)**2)*np.clip(coverage, 0, 1))
        geometry = PrimitiveGeometry(shape, score, metadata={
            'center': center.tolist(), 'size_world': size.tolist(),
            'residual_p80_m': error, 'coverage': float(coverage)}, **parameters)
        return PrimitiveFit(geometry, center, size)

    def _sphere(self, p, plane):
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

    def _box(self, p, plane):
        high = float(np.percentile(p[:, 2], 99.8))
        upper = p[p[:, 2] > high-.010]
        bins = np.floor(upper[:, 2]/self.tolerance).astype(int)
        values, counts = np.unique(bins, return_counts=True)
        level = values[np.argmax(counts)]
        top_z = float(np.median(upper[bins == level, 2]))
        top = p[abs(p[:, 2]-top_z) < self.tolerance]
        sides = p[p[:, 2] < top_z-3*self.tolerance]
        height = top_z-plane
        if len(top) < 15 or len(sides) < 15 or height < .008:
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

    def _cylinder_axis(self, p, plane, axis):
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        reference = np.eye(3)[np.argmin(abs(axis))]
        side = np.cross(axis, reference)
        side /= np.linalg.norm(side)
        other = np.cross(axis, side)
        axial = p @ axis
        lo, hi = np.percentile(axial, [.1, 99.9])
        vertical = abs(axis[2]) > .999
        if vertical:
            axis = np.array([0., 0., 1.])
            lo, hi = plane, float(np.percentile(p[:, 2], 99.8))
            axial = p[:, 2]
        length = float(hi-lo)
        if length < .01:
            return None
        core_mask = (axial > lo+.15*length) & (axial < hi-.15*length)
        cross_points = np.column_stack((p @ side, p @ other))
        core = cross_points[core_mask]
        if len(core) < 20:
            return None
        fit = _radial(core)
        if fit is None:
            return None
        cross, radius = fit
        if not .005 <= radius <= .10 or (not vertical and length < 1.5*radius):
            return None
        center = axis*((lo+hi)/2)+side*cross[0]+other*cross[1]
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

    def _lying_cylinder(self, p, plane):
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

    def finalize(self, fit, points, *, support_plane_z=0., view_clouds=()):
        """Report conservative current extents without repeating classification.

        The surface fit remains the classification evidence. Empirical outer
        dimensions describe measured residual spread, not a calibrated confidence
        interval. Missing geometric observability is explicit for the source's
        graspable flag; it never changes held/tracking state or invents samples.
        """
        if not np.isfinite(support_plane_z):
            raise ValueError('Support plane must be finite')
        result = PrimitiveFit(deepcopy(fit.geometry), fit.center.copy(), fit.size.copy())
        g = result.geometry
        p = np.asarray(points, float).reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        evidence = {'method': 'measured_outer_envelope', 'current_points': len(p)}
        g.metadata.update(parameter_observable=False, parameter_uncertainty=evidence)
        if g.shape == PrimitiveShape.UNKNOWN or len(p) < 30:
            evidence['reason'] = 'unknown_or_insufficient_current_surface'
            return result
        g.metadata['surface_fit'] = dict(center=fit.center.tolist(), size_world=fit.size.tolist(),
            radius=g.radius, length=g.length, box_size=None if g.box_size is None else g.box_size.tolist())
        if g.shape == PrimitiveShape.CUBE:
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
            evidence.update(surface_padding_m=padding, dimension_envelope_m=dimensions.tolist())
            if padding > self.tolerance or np.max(dimensions-g.box_size) > 2*self.tolerance:
                evidence['reason'] = 'inconsistent_box_surface_extent'
                return result
            result.center = ((lo+hi)/2) @ rotation.T
            g.box_size = dimensions
            result.size = abs(rotation) @ dimensions
        else:
            contact = g.shape == PrimitiveShape.SPHERE or abs(g.axis[2]) < .001
            if contact and abs(fit.center[2]-g.radius-support_plane_z) > .004:
                evidence['reason'] = 'unobserved_support_contact'
                return result
            if g.shape == PrimitiveShape.CYLINDER:
                observable, details = _cylinder_observability(p, fit, self.tolerance)
                evidence.update(details)
                if not observable:
                    return result
            samples = _radius_evidence(p, fit, support_plane_z)
            if samples is None:
                evidence['reason'] = 'insufficient_radius_evidence'
                return result
            low, median, high = np.percentile(samples, [5, 50, 95])
            outer = float(max(g.radius, 2*high-median))
            view_extents = []
            for cloud in view_clouds:
                q = np.asarray(cloud, float).reshape(-1, 3)
                q = q[np.isfinite(q).all(1)]
                values = _radius_evidence(q, fit, support_plane_z)
                if values is not None:
                    middle, upper = np.percentile(values, [50, 95])
                    view_extents.append(float(2*upper-middle))
            if view_extents:
                outer = max(outer, max(view_extents))
            evidence.update(radius_samples=len(samples), radius_envelope_m=[float(low), outer],
                            contact_radius_m=float(median), view_outer_radii_m=view_extents)
            if (outer-g.radius > .004 or
                    (view_extents and np.ptp(view_extents) > 2*self.tolerance)):
                evidence['reason'] = 'inconsistent_radius_evidence'
                return result
            g.radius = outer
            if contact:
                result.center[2] = support_plane_z+outer
            if g.shape == PrimitiveShape.SPHERE:
                result.size = np.full(3, 2*outer)
            else:
                result.size = abs(g.axis)*g.length+2*outer*np.sqrt(np.maximum(1-g.axis*g.axis, 0))
        evidence['reason'] = 'current_surface_envelope'
        g.metadata.update(parameter_observable=True, center=result.center.tolist(), size_world=result.size.tolist())
        return result

    def fit(self, points, *, support_plane_z=0.):
        p = voxelize(points, .0015, self.max_points)
        if not np.isfinite(support_plane_z):
            raise ValueError('Support plane must be finite')
        scores = dict(cube=0., sphere=0., cylinder=0.)
        residuals = {}
        fits = []
        if len(p) >= 30:
            fits = [f for f in (self._sphere(p, support_plane_z), self._box(p, support_plane_z),
                                self._cylinder_axis(p, support_plane_z, [0., 0., 1.])) if f is not None]
            best = self._lying_cylinder(p, support_plane_z)
            if best is not None:
                fits.append(best)
            for fit in fits:
                shape = fit.geometry.shape.value
                if fit.geometry.confidence >= scores[shape]:
                    scores[shape] = fit.geometry.confidence
                    residuals[shape] = fit.geometry.metadata['residual_p80_m']
        ranked = sorted(scores.values(), reverse=True)
        if fits and ranked[0] >= self.threshold and ranked[0]-ranked[1] >= self.margin:
            result = max(fits, key=lambda f: f.geometry.confidence)
        else:
            result = unknown_fit(p, support_plane_z=support_plane_z, confidence=ranked[0])
        result.geometry.scores = scores
        result.geometry.metadata.update(residuals_m=residuals, fit_points=len(p))
        return result
