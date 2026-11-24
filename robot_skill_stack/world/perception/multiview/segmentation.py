"""Geometric components with conservative primitive-supported refinement."""
from collections.abc import Sequence

import numpy as np

from robot_skill_stack.world.common.components import radius_components

from .fitting import PrimitiveFit, unknown_fit


def supported_object_cloud(points: np.ndarray, support_plane_z: float) -> tuple[bool, str | None]:
    """Check whether a cluster has enough points, plausible size, and table contact."""
    if len(points) < 30:
        return False, 'unsupported_geometry'
    lo, hi = np.percentile(points, [1, 99], axis=0)
    extent = hi - lo
    if max(extent) > .18:
        return False, 'oversized_or_robot_connected'
    if lo[2] > support_plane_z + .014:
        return False, 'no_observed_support_contact'
    if max(extent[:2]) < .012 or hi[2] - support_plane_z < .010:
        return False, 'unsupported_geometry'
    return True, None


def _surface_distance(points: np.ndarray, fit: PrimitiveFit) -> np.ndarray:
    """Measure point distances to a fitted primitive's surface."""
    geometry = fit.geometry
    delta = points - fit.center
    if geometry.shape.value == 'sphere':
        value = np.linalg.norm(delta, axis=1) - geometry.radius
        return abs(value)
    if geometry.shape.value == 'cube':
        c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        q = abs(delta @ rotation) - geometry.box_size / 2
    elif geometry.shape.value == 'cylinder':
        along = delta @ geometry.axis
        q = np.c_[np.linalg.norm(delta - along[:, None] * geometry.axis, axis=1) - geometry.radius,
                  abs(along) - geometry.length / 2]
    else:
        return np.full(len(points), np.inf)
    value = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return abs(value)


def _distinct_models(fits: Sequence[PrimitiveFit]) -> bool:
    """Require separated sphere disks or conservative world-axis XY bounds.

    Overlapping AABBs of rotated boxes/cylinders stay unresolved, even when a
    more specialized intersection test could separate their hidden geometry.
    """
    for i, first in enumerate(fits):
        for second in fits[i + 1:]:
            if first.geometry.shape.value == second.geometry.shape.value == 'sphere':
                gap = np.linalg.norm(first.center[:2] - second.center[:2])
                separate = gap >= first.geometry.radius + second.geometry.radius - .001
            else:
                gap = abs(first.center[:2] - second.center[:2]) - (first.size[:2] + second.size[:2]) / 2
                separate = np.any(gap >= -.001)
            if not separate:
                return False
    return True


def refine_cluster(points: np.ndarray, fitter, *, support_plane_z: float = 0.) -> list[tuple[np.ndarray, PrimitiveFit | None]]:
    """Split only well-covered, supported metric components with distinct models.

    No missing samples are reassigned or used to infer FREE space. A competing
    single-solid explanation or insufficient fit budget is unknown. Sparse
    fragments may retain a whole sphere or a supported box with one connected
    measured footprint. A separate oversized component may be discarded
    downstream without losing a qualified target.
    """
    supported, reason = supported_object_cloud(points, support_plane_z)
    oversized = reason == 'oversized_or_robot_connected'
    if len(points) < 50 or not supported and reason != 'no_observed_support_contact' and not oversized:
        return [(points, None)]
    components = radius_components(points)
    if len(components) <= 1:
        return [(points, None)]
    coverage = sum(map(len, components)) / len(points)
    unresolved = [(points, unknown_fit(points, reason='unresolved_components'))]
    if not 2 <= len(components) <= 8:
        return unresolved
    if coverage < .90:
        # Vertical sampling gaps need not separate objects. A recovered box must
        # also have one measured XY footprint, not two disconnected box tops.
        if not oversized:
            parent = fitter.fit(points, support_plane_z=support_plane_z)
            shape = parent.geometry.shape.value
            if shape in ('sphere', 'cube') and np.percentile(_surface_distance(points, parent), 95) <= .003:
                if shape == 'sphere':
                    return [(points, parent)]
                if supported:
                    footprint = points.copy()
                    footprint[:, 2] = 0.
                    projected = radius_components(footprint, min_points=1)
                    if max(map(len, projected)) >= .95 * len(points):
                        return [(points, parent)]
        return unresolved
    candidates, background = [], []
    for cloud in components:
        if oversized and supported_object_cloud(cloud, support_plane_z)[1] == 'oversized_or_robot_connected':
            background.append(cloud)
        else:
            candidates.append(cloud)
    if oversized and (not background or not candidates):
        return unresolved
    required = len(candidates) + (not background)
    # A direct SurfaceFitter is useful offline; live calls share the provider's cap.
    if getattr(fitter, 'remaining', required) < required:
        return unresolved
    fits = []
    for cloud in candidates:
        supported, reason = supported_object_cloud(cloud, support_plane_z)
        if not supported and reason != 'no_observed_support_contact':
            return unresolved
        fit = fitter.fit(cloud, support_plane_z=support_plane_z)
        contact = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                   abs(fit.center[2] - fit.size[2] / 2 - support_plane_z) <= .004)
        if (fit.geometry.shape.value == 'unknown' or not (supported or contact) or
                np.percentile(_surface_distance(cloud, fit), 95) > .003):
            return unresolved
        fits.append(fit)
    if not _distinct_models(fits):
        return unresolved
    if background:
        return sorted(list(zip(candidates, fits)) + [(cloud, None) for cloud in background],
                      key=lambda item: tuple(item[0].mean(0)))
    parent = fitter.fit(points, support_plane_z=support_plane_z)
    if parent.geometry.shape.value != 'unknown' and np.mean(_surface_distance(points, parent) <= .003) >= .95:
        return unresolved
    return list(zip(components, fits))
