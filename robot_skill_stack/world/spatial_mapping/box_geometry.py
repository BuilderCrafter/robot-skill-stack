"""Measured yaw boxes qualified by face patches and occupied/free boundaries."""
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np

from robot_skill_stack.world.common.camera import DepthFrame
from robot_skill_stack.world.common.point_clouds import crop_workspace, fuse_clouds, project_depth
from robot_skill_stack.world.perception_multiview.fitting import _box_distance, _box_search
from robot_skill_stack.world.perception_multiview.observability import (
    _DEPTH_TOLERANCE, _EDGE_POSITIONS, _EDGE_SAMPLES, _PIXEL_NEIGHBORS,
    _RIBBON_OFFSETS, _unique_pixels,
)
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points
from .robot_filter import RobotSnapshot

TOL = .0015
DEPTH = _DEPTH_TOLERANCE
FACE_KEYS = tuple(product(range(3), (-1, 1)))
CORNERS = np.array(list(product((-1., 1.), repeat=3))) / 2
PATCH_GRID = np.array(list(product((-.25, 0., .25), repeat=2)))


def _hypothesis(points: np.ndarray) -> dict | None:
    """Fit yaw and propose extents from surfaces, without a support-height prior."""
    sampled_points = points[np.linspace(0, len(points) - 1, min(len(points), 1200), dtype=int)]
    upper = sampled_points[sampled_points[:, 2] > np.percentile(sampled_points[:, 2], 99.8) - .010]
    levels, counts = np.unique(np.floor(upper[:, 2] / TOL).astype(int), return_counts=True)
    top = np.median(upper[np.floor(upper[:, 2] / TOL).astype(int) == levels[np.argmax(counts)], 2])
    side = sampled_points[:, 2] < top - 3 * TOL
    if np.count_nonzero(side) < 15 or np.count_nonzero(abs(sampled_points[:, 2] - top) < TOL) < 15:
        return None
    yaw, rotation, _, _ = _box_search(sampled_points, side, np.deg2rad(np.arange(0, 90, 5.)))
    for width, step in ((5., .5), (.5, .05)):
        yaw, rotation, _, _ = _box_search(sampled_points, side,
            yaw + np.deg2rad(np.arange(-width, width + step / 2, step)))
    local = points @ rotation
    lo, hi = local.min(0), local.max(0)
    size = hi - lo
    if min(size) < .007:
        return None
    center = ((lo + hi) / 2) @ rotation.T
    residual = np.percentile(_box_distance(sampled_points, center, rotation, size), 80)
    if np.exp(-.5 * (residual / TOL) ** 2) < .65:
        return None
    return dict(center=center, dimensions=size, yaw_rad=yaw, rotation=rotation)


def _face_support(local: np.ndarray, size: np.ndarray, axis: int, sign: int) -> tuple | None:
    """Return a directly measured face's plane and depth-banded interval."""
    other = [a for a in range(3) if a != axis]
    face_points = local[(abs(local[:, axis] - sign * size[axis] / 2) <= TOL) &
                        (abs(local[:, other]) < .35 * size[other]).all(1)]
    if len(face_points) < 6 or np.any(np.ptp(face_points[:, other], axis=0) < .25 * size[other]):
        return None
    design = np.c_[face_points[:, other], np.ones(len(face_points))]
    plane, _, rank, _ = np.linalg.lstsq(design, face_points[:, axis], rcond=None)
    spread = np.percentile(abs(design @ plane - face_points[:, axis]), 95)
    if rank != 3 or spread > DEPTH or np.sum(abs(plane[:2]) * size[other]) > 2 * DEPTH:
        return None
    return float(plane[-1]), (face_points[:, axis].min() - DEPTH, face_points[:, axis].max() + DEPTH)


def _boundary(model: dict, frame: DepthFrame, target: tuple[int, int],
              carrier: tuple[int, int], plane: float) -> float | None:
    """Bound a hidden face using three separated occupied/free transition bins."""
    a, sign = target
    b, carrier_sign = carrier
    c = next(x for x in range(3) if x not in (a, b))
    size, center, rotation = model['dimensions'], model['center'], model['rotation']
    origin = (frame.world_from_camera[:3, 3] - center) @ rotation
    if carrier_sign * (origin[b] - plane) <= 0:
        return None
    edge = np.zeros((_EDGE_SAMPLES, 3))
    edge[:, a], edge[:, b], edge[:, c] = sign * size[a] / 2, plane, _EDGE_POSITIONS * size[c]
    camera_points = ((center + edge @ rotation.T - frame.world_from_camera[:3, 3])
                     @ frame.world_from_camera[:3, :3])
    if np.any(camera_points[:, 2] <= 0):
        return None
    intrinsics = frame.intrinsics
    projected = (camera_points[:, :2] / camera_points[:, 2, None]
                 * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2] - .5)
    pixels = _unique_pixels((np.rint(projected).astype(int)[:, None] + _RIBBON_OFFSETS).reshape(-1, 2))
    h, w = frame.depth.shape
    pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
    rays = np.c_[(pixels[:, 0] + .5 - intrinsics[0, 2]) / intrinsics[0, 0],
                 (pixels[:, 1] + .5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(len(pixels))]
    directions = rays @ frame.world_from_camera[:3, :3].T @ rotation
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (plane - origin[b]) / directions[:, b]
        intersections = origin + directions * expected[:, None]
    keep = (expected > 0) & np.isfinite(intersections).all(1) & (abs(intersections[:, c]) <= .38 * size[c])
    pixels, intersections, directions, expected = [
        v[keep] for v in (pixels, intersections, directions, expected)]
    if not len(pixels):
        return None
    measured = frame.depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    free = valid & (measured > expected + DEPTH)
    occupied = (valid & (measured >= expected - DEPTH) & ~free &
                (sign * intersections[:, a] <= size[a] / 2 + DEPTH))
    free_ids = np.flatnonzero(free)
    if not len(free_ids) or not occupied.any():
        return None
    ids = pixels[:, 1] * w + pixels[:, 0]
    order = np.argsort(ids)
    neighbors = pixels[free_ids, None] + _PIXEL_NEIGHBORS
    neighbor_ids = neighbors[..., 1] * w + neighbors[..., 0]
    positions = np.searchsorted(ids[order], neighbor_ids)
    selected = order[np.minimum(positions, len(order) - 1)]
    pairs = ((positions < len(order)) & (ids[selected] == neighbor_ids) & occupied[selected] &
             (sign * (intersections[selected, a] - intersections[free_ids, None, a]) < 0) &
             ((neighbors >= 0) & (neighbors < [w, h])).all(2))
    witness = free_ids[pairs.any(1)]
    outward = sign * intersections[witness, a] + DEPTH * (1 + abs(directions[witness, a] / directions[witness, b]))
    bins = []
    for low, high in ((-.38, -.1267), (-.1267, .1267), (.1267, .38)):
        selected = (intersections[witness, c] >= low * size[c]) & (intersections[witness, c] <= high * size[c])
        if not selected.any():
            return None
        bins.append(outward[selected].min())
    return float(sign * max(size[a] / 2, max(bins)))


def _free_conflict(model: dict, frames: Sequence[DepthFrame]) -> bool:
    """Reject current depth rays that traverse the strict box interior."""
    size, center, rotation = model['dimensions'], model['center'], model['rotation']
    samples = []
    for a, sign in FACE_KEYS:
        other = [x for x in range(3) if x != a]
        for u, v in PATCH_GRID:
            local_point = np.zeros(3)
            local_point[a], local_point[other[0]], local_point[other[1]] = (
                sign * size[a] / 2, u * size[other[0]], v * size[other[1]])
            samples.append(local_point)
    world = center + np.array(samples) @ rotation.T
    half = size / 2 - DEPTH
    for frame in frames:
        camera_points = (world - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        camera_points = camera_points[camera_points[:, 2] > 0]
        intrinsics = frame.intrinsics
        pixels = np.unique(np.rint(camera_points[:, :2] / camera_points[:, 2, None]
            * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2] - .5).astype(int), axis=0)
        h, w = frame.depth.shape
        pixels = pixels[((pixels >= 0) & (pixels < [w, h])).all(1)]
        rays = np.c_[(pixels[:, 0] + .5 - intrinsics[0, 2]) / intrinsics[0, 0],
                     (pixels[:, 1] + .5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(len(pixels))]
        origin = (frame.world_from_camera[:3, 3] - center) @ rotation
        directions = rays @ frame.world_from_camera[:3, :3].T @ rotation
        with np.errstate(divide='ignore', invalid='ignore'):
            first, second = (-half - origin) / directions, (half - origin) / directions
        enter, leave = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
        measured = frame.depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(measured) & (measured > 0) & (leave > np.maximum(enter, 0))
        if np.any(valid & (measured > enter + DEPTH)):
            return True
    return False


def _face_patch(model: dict, plane: float, axis: int, sign: int,
                frame: DepthFrame) -> tuple[bool, bool]:
    """Return patch support and FREE contradiction from nine current pixels."""
    rotation, center, size = (model[key] for key in ('rotation', 'center', 'dimensions'))
    origin = (frame.world_from_camera[:3, 3] - center) @ rotation
    if sign * (origin[axis] - plane) <= 0:
        return False, False
    other = [a for a in range(3) if a != axis]
    local = np.zeros((9, 3))
    local[:, axis], local[:, other] = plane, PATCH_GRID * size[other]
    camera_points = ((center + local @ rotation.T - frame.world_from_camera[:3, 3])
                     @ frame.world_from_camera[:3, :3])
    intrinsics = frame.intrinsics
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = (camera_points[:, :2] / camera_points[:, 2, None]
                     * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2] - .5)
    finite = np.isfinite(projected).all(1)
    pixels = np.zeros((9, 2), dtype=int)
    pixels[finite] = np.rint(projected[finite]).astype(int)
    h, w = frame.depth.shape
    inside = finite & ((pixels >= 0) & (pixels < [w, h])).all(1) & (camera_points[:, 2] > 0)
    rays = np.c_[(pixels[:, 0] + .5 - intrinsics[0, 2]) / intrinsics[0, 0],
                 (pixels[:, 1] + .5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(9)]
    directions = rays @ frame.world_from_camera[:3, :3].T @ rotation
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = (plane - origin[axis]) / directions[:, axis]
    measured = np.full(9, np.nan)
    measured[inside] = frame.depth[pixels[inside, 1], pixels[inside, 0]]
    valid = inside & np.isfinite(expected) & (expected > 0) & np.isfinite(measured) & (measured > 0)
    if np.any(valid & (measured > expected + DEPTH)):
        return False, True
    supported = valid & (measured >= expected - DEPTH) & (measured <= expected + DEPTH)
    # Six distinct pixels on a 3x3 grid already span both grid axes.
    return len(np.unique(pixels[supported], axis=0)) >= 6, False


def _qualify(points: np.ndarray, frames: Sequence[DepthFrame]) -> tuple[dict | None, str | None]:
    """Require six measured bounds and visible patches in all three axes."""
    if len(points) < 30:
        return None, 'insufficient_points'
    model = _hypothesis(points)
    if model is None:
        return None, 'insufficient_box_surfaces'
    local = (points - model['center']) @ model['rotation']
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
        bounds[axis, int(sign > 0)] = sign * min(sign * edge for edge in edges)
    axes = [False] * 3
    for (axis, sign), support in supports.items():
        if support is not None:
            for frame in frames:
                passed, conflict = _face_patch(model, support[0], axis, sign, frame)
                if conflict:
                    return None, 'contradictory_visible_plane_pixels'
                axes[axis] |= passed
    if not all(axes):
        return None, 'insufficient_observed_three_axis_patch'
    return dict(center=model['center'] + bounds.mean(1) @ model['rotation'].T,
                dimensions=np.diff(bounds, axis=1).ravel(), rotation=model['rotation'],
                yaw_rad=model['yaw_rad']), None


def reconstruct(frames: Sequence[DepthFrame], scene,
                snapshots: Mapping[str, RobotSnapshot]) -> tuple[tuple[dict, ...], str | None]:
    """Project, cluster and qualify boxes; reject a mixed or unsupported world."""
    envelopes = []
    for frame in frames:
        for part in snapshots[frame.camera_id].parts:
            world_vertices = part.mesh.vertices @ part.world_from_local[:3, :3].T + part.world_from_local[:3, 3]
            envelopes.append((world_vertices.min(0) - .005, world_vertices.max(0) + .005))
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
