"""Measured yaw boxes qualified by face patches and occupied/free boundaries."""
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np

from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.fitting import _box_distance, _box_search
from robot_skill_stack.world.perception_multiview.observability import (
    _DEPTH_TOLERANCE, _EDGE_POSITIONS, _EDGE_SAMPLES, _PIXEL_NEIGHBORS,
    _RIBBON_OFFSETS, _unique_pixels,
)
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points

TOL = .0015
DEPTH = _DEPTH_TOLERANCE
FACE_KEYS = tuple(product(range(3), (-1, 1)))
CORNERS = np.array(list(product((-1., 1.), repeat=3)))/2
PATCH_GRID = np.array(list(product((-.25, 0., .25), repeat=2)))


def _hypothesis(points: np.ndarray) -> dict | None:
    """Fit yaw and propose extents from surfaces, without a support-height prior."""
    p = points[np.linspace(0, len(points)-1, min(len(points), 1200), dtype=int)]
    upper = p[p[:, 2] > np.percentile(p[:, 2], 99.8)-.010]
    levels, counts = np.unique(np.floor(upper[:, 2]/TOL).astype(int), return_counts=True)
    top = np.median(upper[np.floor(upper[:, 2]/TOL).astype(int) == levels[np.argmax(counts)], 2])
    side = p[p[:, 2] < top-3*TOL]
    if len(side) < 15 or np.count_nonzero(abs(p[:, 2]-top) < TOL) < 15:
        return None
    yaw, rotation, _, _ = _box_search(p, side, np.deg2rad(np.arange(0, 90, 5.)))
    for width, step in ((5., .5), (.5, .05)):
        yaw, rotation, _, _ = _box_search(p, side, yaw+np.deg2rad(np.arange(-width, width+step/2, step)))
    local = points@rotation
    lo, hi = local.min(0), local.max(0)
    size = hi-lo
    if min(size) < .007:
        return None
    center = ((lo+hi)/2)@rotation.T
    residual = np.percentile(_box_distance(p, center, rotation, size), 80)
    if np.exp(-.5*(residual/TOL)**2) < .65:
        return None
    return dict(center=center, dimensions=size, yaw_rad=yaw, rotation=rotation)


def _face_support(local: np.ndarray, size: np.ndarray, axis: int, sign: int) -> tuple | None:
    """Return a directly measured face's plane and depth-banded interval."""
    other = [a for a in range(3) if a != axis]
    p = local[(abs(local[:, axis]-sign*size[axis]/2) <= TOL) &
              (abs(local[:, other]) < .35*size[other]).all(1)]
    if len(p) < 6 or np.any(np.ptp(p[:, other], axis=0) < .25*size[other]):
        return None
    design = np.c_[p[:, other], np.ones(len(p))]
    plane, _, rank, _ = np.linalg.lstsq(design, p[:, axis], rcond=None)
    spread = np.percentile(abs(design@plane-p[:, axis]), 95)
    if rank != 3 or spread > DEPTH or np.sum(abs(plane[:2])*size[other]) > 2*DEPTH:
        return None
    return float(plane[-1]), (p[:, axis].min()-DEPTH, p[:, axis].max()+DEPTH)


def _boundary(model: dict, frame, target: tuple, carrier: tuple, plane: float) -> float | None:
    """Bound a hidden face using three separated occupied/free transition bins."""
    a, sign = target
    b, carrier_sign = carrier
    c = next(x for x in range(3) if x not in (a, b))
    size, center, r = model['dimensions'], model['center'], model['rotation']
    origin = (frame.world_from_camera[:3, 3]-center)@r
    if carrier_sign*(origin[b]-plane) <= 0:
        return None
    edge = np.zeros((_EDGE_SAMPLES, 3))
    edge[:, a], edge[:, b], edge[:, c] = sign*size[a]/2, plane, _EDGE_POSITIONS*size[c]
    camera = (center+edge@r.T-frame.world_from_camera[:3, 3])@frame.world_from_camera[:3, :3]
    if np.any(camera[:, 2] <= 0):
        return None
    k = frame.intrinsics
    projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    pixels = _unique_pixels((np.rint(projected).astype(int)[:, None]+_RIBBON_OFFSETS).reshape(-1, 2))
    h, w = frame.depth.shape
    pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
    rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                 (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
    directions = rays@frame.world_from_camera[:3, :3].T@r
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (plane-origin[b])/directions[:, b]
        q = origin+directions*expected[:, None]
    keep = (expected > 0) & np.isfinite(q).all(1) & (abs(q[:, c]) <= .38*size[c])
    pixels, q, directions, expected = [v[keep] for v in (pixels, q, directions, expected)]
    if not len(pixels):
        return None
    measured = frame.depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    free = valid & (measured > expected+DEPTH)
    occupied = valid & (measured >= expected-DEPTH) & ~free & (sign*q[:, a] <= size[a]/2+DEPTH)
    free_ids = np.flatnonzero(free)
    if not len(free_ids) or not occupied.any():
        return None
    ids = pixels[:, 1]*w+pixels[:, 0]
    order = np.argsort(ids)
    neighbors = pixels[free_ids, None]+_PIXEL_NEIGHBORS
    neighbor_ids = neighbors[..., 1]*w+neighbors[..., 0]
    positions = np.searchsorted(ids[order], neighbor_ids)
    selected = order[np.minimum(positions, len(order)-1)]
    pairs = ((positions < len(order)) & (ids[selected] == neighbor_ids) & occupied[selected] &
             (sign*(q[selected, a]-q[free_ids, None, a]) < 0) &
             ((neighbors >= 0) & (neighbors < [w, h])).all(2))
    witness = free_ids[pairs.any(1)]
    outward = sign*q[witness, a]+DEPTH*(1+abs(directions[witness, a]/directions[witness, b]))
    bins = []
    for low, high in ((-.38, -.1267), (-.1267, .1267), (.1267, .38)):
        selected = (q[witness, c] >= low*size[c]) & (q[witness, c] <= high*size[c])
        if not selected.any():
            return None
        bins.append(outward[selected].min())
    return float(sign*max(size[a]/2, max(bins)))


def _free_conflict(model: dict, frames: Sequence) -> bool:
    """Reject current depth rays that traverse the strict box interior."""
    size, center, r = model['dimensions'], model['center'], model['rotation']
    samples = []
    for a, sign in FACE_KEYS:
        other = [x for x in range(3) if x != a]
        for u, v in PATCH_GRID:
            q = np.zeros(3)
            q[a], q[other[0]], q[other[1]] = sign*size[a]/2, u*size[other[0]], v*size[other[1]]
            samples.append(q)
    world = center+np.array(samples)@r.T
    half = size/2-DEPTH
    for f in frames:
        camera = (world-f.world_from_camera[:3, 3])@f.world_from_camera[:3, :3]
        camera = camera[camera[:, 2] > 0]
        k = f.intrinsics
        pixels = np.unique(np.rint(camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5).astype(int), axis=0)
        h, w = f.depth.shape
        pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
        rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                     (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(len(pixels))]
        origin = (f.world_from_camera[:3, 3]-center)@r
        directions = rays@f.world_from_camera[:3, :3].T@r
        with np.errstate(divide='ignore', invalid='ignore'):
            first, second = (-half-origin)/directions, (half-origin)/directions
        enter, leave = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
        measured = f.depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(measured) & (measured > 0) & (leave > np.maximum(enter, 0))
        if np.any(valid & (measured > enter+DEPTH)):
            return True
    return False


def _face_patch(model: dict, plane: float, axis: int, sign: int, frame) -> tuple[bool, bool]:
    """Return patch support and FREE contradiction from nine current pixels."""
    r, center, size = (model[k] for k in ('rotation', 'center', 'dimensions'))
    origin = (frame.world_from_camera[:3, 3]-center)@r
    if sign*(origin[axis]-plane) <= 0:
        return False, False
    other = [a for a in range(3) if a != axis]
    local = np.zeros((9, 3))
    local[:, axis], local[:, other] = plane, PATCH_GRID*size[other]
    camera = (center+local@r.T-frame.world_from_camera[:3, 3])@frame.world_from_camera[:3, :3]
    k = frame.intrinsics
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = camera[:, :2]/camera[:, 2, None]*[k[0, 0], k[1, 1]]+k[:2, 2]-.5
    finite = np.isfinite(projected).all(1)
    pixels = np.zeros((9, 2), dtype=int)
    pixels[finite] = np.rint(projected[finite]).astype(int)
    h, w = frame.depth.shape
    inside = finite & ((pixels >= 0) & (pixels < [w, h])).all(1) & (camera[:, 2] > 0)
    rays = np.c_[(pixels[:, 0]+.5-k[0, 2])/k[0, 0],
                 (pixels[:, 1]+.5-k[1, 2])/k[1, 1], np.ones(9)]
    directions = rays@frame.world_from_camera[:3, :3].T@r
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (plane-origin[axis])/directions[:, axis]
    measured = np.full(9, np.nan)
    measured[inside] = frame.depth[pixels[inside, 1], pixels[inside, 0]]
    valid = inside & np.isfinite(expected) & (expected > 0) & np.isfinite(measured) & (measured > 0)
    if np.any(valid & (measured > expected+DEPTH)):
        return False, True
    supported = valid & (measured >= expected-DEPTH) & (measured <= expected+DEPTH)
    # Six distinct pixels on a 3x3 grid already span both grid axes.
    return len(np.unique(pixels[supported], axis=0)) >= 6, False


def _qualify(points: np.ndarray, frames: Sequence) -> tuple[dict | None, str | None]:
    """Require six measured bounds and visible patches in all three axes."""
    if len(points) < 30:
        return None, 'insufficient_points'
    model = _hypothesis(points)
    if model is None:
        return None, 'insufficient_box_surfaces'
    local = (points-model['center'])@model['rotation']
    supports = {key: _face_support(local, model['dimensions'], *key) for key in FACE_KEYS}
    if _free_conflict(model, frames):
        return None, 'contradictory_current_free_rays'
    bounds = np.empty((3, 2))
    for axis, sign in FACE_KEYS:
        support = supports[axis, sign]
        if support is not None:
            bounds[axis, int(sign > 0)] = support[1][int(sign > 0)]
            continue
        edges = [_boundary(model, frame, (axis, sign), carrier, value[0])
                 for carrier, value in supports.items() if carrier[0] != axis and value is not None
                 for frame in frames]
        edges = [edge for edge in edges if edge is not None]
        if not edges:
            return None, 'unqualified_hidden_bound'
        bounds[axis, int(sign > 0)] = sign*min(sign*edge for edge in edges)
    axes = [False]*3
    for (axis, sign), support in supports.items():
        if support is not None:
            for frame in frames:
                passed, conflict = _face_patch(model, support[0], axis, sign, frame)
                if conflict:
                    return None, 'contradictory_visible_plane_pixels'
                axes[axis] |= passed
    if not all(axes):
        return None, 'insufficient_observed_three_axis_patch'
    return dict(center=model['center']+bounds.mean(1)@model['rotation'].T,
                dimensions=np.diff(bounds, axis=1).ravel(), rotation=model['rotation'],
                yaw_rad=model['yaw_rad']), None


def reconstruct(frames: Sequence, scene, snapshots: Mapping) -> tuple[tuple[dict, ...], str | None]:
    """Project, cluster and qualify boxes; reject a mixed or unsupported world."""
    envelopes = []
    for frame in frames:
        for part in snapshots[frame.camera_id].parts:
            p = part.mesh.vertices@part.world_from_local[:3, :3].T+part.world_from_local[:3, 3]
            envelopes.append((p.min(0)-.005, p.max(0)+.005))
    clouds = [crop_workspace(project_depth(frame, stride=2,
              bounds=(scene.workspace_min, scene.workspace_max))[0], scene) for frame in frames]
    clusters = cluster_points(fuse_clouds(clouds, max_points=12000))
    models, fits = [], 0
    for points in sorted(clusters, key=len, reverse=True)[:16]:
        overlap = np.logical_or.reduce([((points >= lo) & (points <= hi)).all(1)
                                       for lo, hi in envelopes]) if envelopes else np.zeros(len(points), bool)
        if overlap.all():
            continue
        if overlap.any():
            return (), 'robot_connected_component'
        if fits == 4:
            return (), 'hypothesis_budget_exhausted'
        fits += 1
        geometry, reason = _qualify(points, frames)
        if reason is not None:
            return (), reason
        models.append(geometry)
    return tuple(models), None
