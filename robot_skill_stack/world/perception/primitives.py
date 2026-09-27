from __future__ import annotations
import numpy as np

from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.perception.primitive_fits import (
    area, box_footprint, circle_boundary, cylinder_residual, hull2d, quality, radial_fit, rectangle,
)


class PrimitiveEstimator:
    def __init__(self, classification_threshold=.58, ambiguity_margin=.07, top_band=.15,
                 fit_tolerance=.0008, max_fit_points=1600):
        self.threshold = float(classification_threshold)
        self.margin = float(ambiguity_margin)
        self.top_band = float(top_band)
        self.tolerance = float(fit_tolerance)
        self.max_points = int(max_fit_points)
        if not (0 < self.threshold <= 1 and 0 <= self.margin < 1 and 0 < self.top_band < 1):
            raise ValueError('invalid primitive classification thresholds')
        if self.tolerance <= 0 or self.max_points < 30:
            raise ValueError('fit_tolerance must be positive; max_fit_points must be >= 30')

    def _top(self, p, height):
        upper = float(np.percentile(p[:, 2], 99.5))
        band = p[p[:, 2] >= upper-max(height*self.top_band, 4*self.tolerance)]
        bins = max(3, int(np.ptp(band[:, 2])/self.tolerance)+1)
        counts, edges = np.histogram(band[:, 2], bins=bins)
        i = int(np.argmax(counts))
        z = float(np.median(band[(band[:, 2] >= edges[i]) & (band[:, 2] <= edges[i+1]), 2]))
        top = p[abs(p[:, 2]-z) <= self.tolerance]
        flatness = min(1., len(top)/max(len(band)*.65, 1))
        return z, top, flatness

    @staticmethod
    def _result(shape, score, center, size, **kwargs):
        return PrimitiveGeometry(shape, float(np.clip(score, 0, 1)),
                                 metadata={'center': center.tolist(), 'size_world': size.tolist()}, **kwargs)

    def _box(self, p, top, z, flatness, plane, all_area):
        rect = rectangle(hull2d(top[:, :2]))
        if rect is None or len(top) < 12:
            return None
        fill = rect[3]
        yaw, xy_size, xy_center, _ = box_footprint(p, z, self.tolerance)
        cover = min(1., area(hull2d(top[:, :2]))/max(all_area, 1e-12))
        height = z-plane
        if min(*xy_size, height) < .005:
            return None
        size = np.r_[xy_size, height]
        center = np.r_[xy_center, plane+height/2]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        q = abs((p-center) @ R)-size/2
        residual = abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))
        # A round planar cap is not a box, regardless of similar XYZ dimensions.
        score = flatness * np.clip((fill-.78)/.16, 0, 1) * min(1., cover/.75)
        score *= quality(residual, self.tolerance*1.5)
        return self._result(PrimitiveShape.CUBE, score, center, abs(R) @ size, box_size=size, yaw=yaw)

    def _sphere(self, p, plane):
        fit = radial_fit(p)
        if fit is None:
            return None
        center, radius = fit
        span = float(np.max(np.ptp(p, axis=0)))
        if not .005 <= radius <= span*1.1 or center[2]-radius < plane-3*self.tolerance:
            return None
        residual = abs(np.linalg.norm(p-center, axis=1)-radius)
        score = quality(residual, self.tolerance)
        return self._result(PrimitiveShape.SPHERE, score, center, np.full(3, 2*radius), radius=radius)

    def _vertical(self, p, top, z, flatness, plane, all_area):
        side = p[p[:, 2] < z-2*self.tolerance]
        fit = radial_fit(side[:, :2]) if len(side) >= 20 else None
        cap_only = fit is None
        if cap_only:
            boundary = circle_boundary(hull2d(top[:, :2]))
            if boundary is None or flatness < .8:
                return None
            xy, radius, boundary_res = boundary
        else:
            xy, radius = fit
        length = z-plane if flatness > .8 else float(np.percentile(p[:, 2], 99.5)-plane)
        if not (.005 <= radius <= .10 and length >= .008):
            return None
        center = np.r_[xy, plane+length/2]
        axis = np.array([0., 0., 1.])
        score = quality(cylinder_residual(p, center, axis, radius, length), self.tolerance)
        if cap_only:
            cover = min(1., area(hull2d(top[:, :2]))/max(all_area, 1e-12))
            score *= quality(boundary_res, self.tolerance) * min(1., cover/.75)
        else:
            score *= quality(abs(np.linalg.norm(side[:, :2]-xy, axis=1)-radius), self.tolerance)
        return self._result(PrimitiveShape.CYLINDER, score, center, np.array([2*radius, 2*radius, length]),
                            axis=axis, radius=radius, length=length)

    def _horizontal(self, p, plane):
        # PCA alone follows sampling bias. Search axis directions against a circle fit.
        sample = p[np.linspace(0, len(p)-1, min(500, len(p)), dtype=int)]
        best = None

        def fit_angle(yaw):
            axis = np.array([np.cos(yaw), np.sin(yaw), 0.])
            side = np.array([-axis[1], axis[0], 0.])
            u = sample @ axis
            lo, hi = np.percentile(u, [.5, 99.5])
            length = hi-lo
            core = sample[(u > lo+.15*length) & (u < hi-.15*length)]
            fit = radial_fit(np.column_stack((core @ side, core[:, 2])))
            if fit is None:
                return None
            cross, radius = fit
            if not (.005 <= radius <= .10 and length >= 2.4*radius):
                return None
            if cross[1]-radius < plane-3*self.tolerance:
                return None
            center = axis*((hi+lo)/2) + side*cross[0] + np.array([0., 0., cross[1]])
            residual = cylinder_residual(sample, center, axis, radius, length)
            score = quality(residual, self.tolerance)
            return score, yaw, center, axis, radius, length

        for yaw in np.linspace(0, np.pi, 36, endpoint=False):
            fit = fit_angle(yaw)
            if fit is not None and (best is None or fit[0] > best[0]):
                best = fit
        if best is None:
            return None
        for yaw in best[1] + np.deg2rad(np.linspace(-5, 5, 21)):
            fit = fit_angle(yaw)
            if fit is not None and fit[0] > best[0]:
                best = fit
        score, _, center, axis, radius, length = best
        score = quality(cylinder_residual(p, center, axis, radius, length), self.tolerance)
        size = abs(axis)*length + 2*radius*np.sqrt(np.maximum(1-axis*axis, 0))
        return self._result(PrimitiveShape.CYLINDER, score, center, size, axis=axis, radius=radius, length=length)

    def estimate(self, candidate):
        p = np.asarray(candidate.metadata.get('points', []), float)
        if p.ndim != 2 or p.shape[1] != 3:
            return PrimitiveGeometry(PrimitiveShape.UNKNOWN, metadata={'reason': 'no_point_cloud'})
        p = p[np.isfinite(p).all(1)]
        if len(p) < 30:
            return PrimitiveGeometry(PrimitiveShape.UNKNOWN, metadata={'reason': 'too_few_points'})
        if len(p) > self.max_points:
            p = p[np.linspace(0, len(p)-1, self.max_points, dtype=int)]
        plane = float(candidate.metadata.get('support_plane_z', min(0., p[:, 2].min())))
        height = float(np.percentile(p[:, 2], 99.5)-plane)
        z, top, flatness = self._top(p, height)
        all_area = area(hull2d(p[:, :2]))
        fits = [fit for fit in (
            self._box(p, top, z, flatness, plane, all_area), self._sphere(p, plane),
            self._vertical(p, top, z, flatness, plane, all_area), self._horizontal(p, plane),
        ) if fit is not None]
        scores = {shape: 0. for shape in ('cube', 'sphere', 'cylinder')}
        for fit in fits:
            scores[fit.shape.value] = max(scores[fit.shape.value], fit.confidence)
        ranked = sorted(scores.values(), reverse=True)
        if not fits or ranked[0] < self.threshold or ranked[0]-ranked[1] < self.margin:
            return PrimitiveGeometry(PrimitiveShape.UNKNOWN, ranked[0], scores=scores,
                                     metadata={'reason': 'weak_or_ambiguous_fit'})
        best = max(fits, key=lambda fit: fit.confidence)
        best.scores = scores
        return best
