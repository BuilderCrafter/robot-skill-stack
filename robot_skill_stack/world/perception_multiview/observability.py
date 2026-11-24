"""Qualify measured dimensions of table-supported yaw boxes.

A rectangular surface patch does not establish a physical edge. Each horizontal
side needs adjacent occupied/free pixel transitions along three separated edge
segments. Invalid, occluded and out-of-frame depth cannot establish a bound.
Measured side planes may tighten these empirical bounds; hidden geometry is
qualified only under the fitter's single supported-box assumption.
"""
from collections.abc import Sequence

import numpy as np

from robot_skill_stack.world.common.camera import DepthFrame
from robot_skill_stack.world.perception_multiview.fitting import PrimitiveFit


_DEPTH_TOLERANCE = .002
_PIXEL_RADIUS = 3
_EDGE_SAMPLES = 48
_RIBBON_OFFSETS = np.array([(x, y) for x in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)
                                for y in range(-_PIXEL_RADIUS, _PIXEL_RADIUS+1)])
_PIXEL_NEIGHBORS = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])
_EDGE_POSITIONS = np.linspace(-.4, .4, _EDGE_SAMPLES)


def _unique_pixels(raw: np.ndarray) -> np.ndarray:
    """Deduplicate bounded image-coordinate ribbons in row-major order."""
    low = raw.min(0)
    width = int(raw[:, 0].max()-low[0])+1
    codes = np.unique((raw[:, 1]-low[1])*width+raw[:, 0]-low[0])
    return np.column_stack((codes % width+low[0], codes // width+low[1]))


def _view_boundary(center: np.ndarray, size: np.ndarray, rotation: np.ndarray,
                   top_z: float, top_spread: float, frame: DepthFrame,
                   axis: int, sign: int) -> float | None:
    """Bound one side using adjacent occupied/free pairs in a pixel ribbon."""
    k, transform, depth = frame.intrinsics, frame.world_from_camera, frame.depth
    height, width = depth.shape
    other = 1-axis
    local = np.zeros((_EDGE_SAMPLES, 3))
    local[:, axis] = sign*size[axis]/2
    local[:, other] = _EDGE_POSITIONS*size[other]
    boundary = center+local @ rotation.T
    boundary[:, 2] = top_z
    camera = (boundary-transform[:3, 3]) @ transform[:3, :3]
    if transform[2, 3] <= top_z or np.any(camera[:, 2] <= 0):
        return None
    projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    pixels = _unique_pixels((np.rint(projected).astype(int)[:, None]+_RIBBON_OFFSETS).reshape(-1, 2))
    pixels = pixels[((pixels >= 0) & (pixels < [width, height])).all(1)]
    rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                 (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
    world_rays = rays @ transform[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        expected_depth = (top_z-transform[2, 3])/world_rays[:, 2]
    world = transform[:3, 3]+world_rays*expected_depth[:, None]
    q = (world-center) @ rotation
    keep = ((expected_depth > 0) & np.isfinite(q).all(1) &
            (abs(q[:, other]) <= .38*size[other]))
    pixels, q, world_rays, expected_depth = (value[keep] for value in
                                             (pixels, q, world_rays, expected_depth))
    if not len(pixels):
        return None
    measured = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    free = valid & (measured > expected_depth+_DEPTH_TOLERANCE)
    # The occupied anchor must be on the top plane and inside the candidate.
    occupied = (valid & ~free & (measured >= expected_depth-_DEPTH_TOLERANCE) &
                (sign*q[:, axis] <= size[axis]/2+top_spread))
    free_ids = np.flatnonzero(free)
    if not len(free_ids) or not occupied.any():
        return None
    # Deduplication and filtering preserve row-major order, so image IDs remain
    # sorted. Search only the eight image neighbors; reject image-border wrap.
    ids = pixels[:, 1]*width+pixels[:, 0]
    nearby = pixels[free_ids, None]+_PIXEL_NEIGHBORS
    nearby_ids = nearby[..., 1]*width+nearby[..., 0]
    positions = np.searchsorted(ids, nearby_ids)
    selected = np.minimum(positions, len(ids)-1)
    paired = ((positions < len(ids)) & (ids[selected] == nearby_ids) & occupied[selected] &
              (sign*(q[selected, axis]-q[free_ids, None, axis]) < 0) &
              ((nearby >= 0) & (nearby < [width, height])).all(2))
    witnesses = free_ids[paired.any(1)]
    # Plane-height uncertainty changes lateral intersections of oblique rays.
    local_rays = world_rays[witnesses] @ rotation
    plane_margin = top_spread*abs(local_rays[:, axis]/world_rays[witnesses, 2])
    upper = sign*q[witnesses, axis]+plane_margin
    bins = []
    for lower, higher in ((-.38, -.1267), (-.1267, .1267), (.1267, .38)):
        chosen = ((q[witnesses, other] >= lower*size[other]) &
                  (q[witnesses, other] <= higher*size[other]))
        if not chosen.any():
            return None
        bins.append(float(upper[chosen].min()))
    return float(max(size[axis]/2, max(bins)))


def _face_plane(local: np.ndarray, size: np.ndarray, axis: int,
                sign: int) -> tuple[float, float, float] | None:
    """Fit side slope/intercept/spread, excluding top and adjacent corners."""
    other = 1-axis
    selected = ((abs(local[:, axis]-sign*size[axis]/2) <= .0015) &
                (abs(local[:, other]) < .35*size[other]) &
                (abs(local[:, 2]) < .35*size[2]))
    points = local[selected]
    if (len(points) < 6 or np.ptp(points[:, 2]) < .25*size[2] or
            np.ptp(points[:, other]) < .3*size[other]):
        return None
    design = np.c_[points[:, other], np.ones(len(points))]
    slope, intercept = np.linalg.lstsq(design, sign*points[:, axis], rcond=None)[0]
    spread = float(np.percentile(abs(design @ [slope, intercept]-sign*points[:, axis]), 95))
    return (float(slope), float(intercept), spread) if spread <= _DEPTH_TOLERANCE else None


def assess(fit: PrimitiveFit, points: np.ndarray, frames: Sequence[DepthFrame], *,
           support_plane_z=0.) -> tuple[np.ndarray, np.ndarray] | None:
    """Return qualified box center/local dimensions, or None without a bound.

    Inputs are finite current XYZ samples and valid fitted geometry. The caller
    must also retain the fitter's parameter qualification. Curves use their own
    finalization; this function only qualifies supported yaw boxes.
    """
    geometry = fit.geometry
    if geometry.shape.value != 'cube' or len(points) < 30 or not frames:
        return None
    size, center = geometry.box_size, fit.center
    c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = (points-center) @ rotation
    top_z = geometry.metadata.get('fitted_top_z', float(center[2]+size[2]/2))
    top_mask = abs(points[:, 2]-top_z) <= _DEPTH_TOLERANCE
    top, top_local = points[top_mask], local[top_mask]
    if (len(top) < 12 or np.any(np.ptp(top_local, axis=0)[:2] < .25*size[:2]) or
            points[:, 2].min() > support_plane_z+.014 or
            abs(center[2]-size[2]/2-support_plane_z) > .004):
        return None
    # Edge voxels mix top/side samples; interior height spread estimates plane
    # uncertainty without treating that mixing as top-plane measurement noise.
    core = top[(abs(top_local[:, :2]) < .35*size[:2]).all(1)]
    plane_samples = core if len(core) >= 6 else top
    top_spread = float(np.percentile(abs(plane_samples[:, 2]-top_z), 95))
    limits = np.c_[-size/2, size/2]
    faces = []
    for axis in (0, 1):
        for sign in (-1, 1):
            bounds = [_view_boundary(center, size, rotation, top_z, top_spread, frame, axis, sign)
                      for frame in frames]
            bounds = [bound for bound in bounds if bound is not None]
            if not bounds:
                return None
            limits[axis, int(sign > 0)] = sign*min(bounds)
            faces.append((axis, sign, _face_plane(local, size, axis, sign)))
    # First establish all camera bounds, then tighten with measured side planes.
    # Preserve the slope contribution across the other axis's current interval.
    for axis, sign, plane in faces:
        if plane is not None:
            slope, intercept, spread = plane
            upper = intercept+max(slope*limits[1-axis])+2*spread
            outer = min(abs(limits[axis, int(sign > 0)]), max(size[axis]/2, upper))
            limits[axis, int(sign > 0)] = sign*outer
    return center+limits.mean(1) @ rotation.T, limits[:, 1]-limits[:, 0]
