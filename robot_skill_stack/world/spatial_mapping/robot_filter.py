"""Selected-pixel robot depth exclusion using immutable meshes and poses."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from .occupancy import _readonly


@dataclass(frozen=True, eq=False)
class TriangleMesh:
    """Own local geometry and an eight-triangle-leaf BVH shared across poses."""
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices, triangles = np.asarray(self.vertices, float), np.asarray(self.triangles, np.int64)
        points = vertices[triangles]
        low, high = points.min(1), points.max(1)
        centers = (low+high)/2
        nodes, permutation = [], []

        def build(indices: np.ndarray) -> int:
            node = len(nodes)
            nodes.append(None)
            lower, upper = low[indices].min(0), high[indices].max(0)
            if len(indices) <= 8:
                start = len(permutation)
                permutation.extend(indices.tolist())
                nodes[node] = (lower, upper, -1, -1, start, len(permutation))
            else:
                axis = int(np.argmax(np.ptp(centers[indices], axis=0)))
                ordered = indices[np.argsort(centers[indices, axis], kind='stable')]
                middle = len(ordered)//2
                nodes[node] = (lower, upper, build(ordered[:middle]), build(ordered[middle:]), 0, 0)
            return node

        build(np.arange(len(triangles)))
        for name, value in (('vertices', vertices), ('triangles', triangles),
                ('_first', points[:, 0]), ('_edge1', points[:, 1]-points[:, 0]), ('_edge2', points[:, 2]-points[:, 0]),
                ('_lower', np.array([node[0] for node in nodes])),
                ('_upper', np.array([node[1] for node in nodes])),
                ('_children', np.array([node[2:4] for node in nodes], np.int64)),
                ('_ranges', np.array([node[4:6] for node in nodes], np.int64)),
                ('_permutation', np.asarray(permutation, np.int64))):
            object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True, eq=False)
class MeshInstance:
    """Own a part's world pose and its cached inverse."""
    mesh: TriangleMesh
    world_from_local: np.ndarray

    def __post_init__(self) -> None:
        transform = _readonly(np.asarray(self.world_from_local, float))
        object.__setattr__(self, 'world_from_local', transform)
        object.__setattr__(self, '_local_from_world', _readonly(np.linalg.inv(transform)))


@dataclass(frozen=True)
class RobotSnapshot:
    """Prepared mesh poses at a camera's simulation time and render token."""
    parts: tuple[MeshInstance, ...]
    simulation_time: float
    frame_token: tuple | None = None


def _box_hits(origin: np.ndarray, directions: np.ndarray, lower: np.ndarray,
              upper: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Ray/AABB broadphase only; touching a box is never a robot mask."""
    moving = directions != 0
    first, second = np.full(directions.shape, -np.inf), np.full(directions.shape, np.inf)
    np.divide(lower-origin, directions, out=first, where=moving)
    np.divide(upper-origin, directions, out=second, where=moving)
    near, far = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
    parallel_outside = (~moving & ((origin < lower) | (origin > upper))).any(1)
    return ~parallel_outside & (far >= np.maximum(near, 0.)) & (near <= limit)


def _mesh_depth(mesh: TriangleMesh, origin: np.ndarray, directions: np.ndarray,
                limit: np.ndarray) -> np.ndarray:
    """Nearest positive triangle parameter using bounded BVH ray/node packets.

    Median splitting bounds stack depth. Roots contain at most 64 rays, node
    packets 1024 pairs, and leaves eight triangles.
    """
    best = np.asarray(limit, float).copy()
    selected = np.flatnonzero(_box_hits(origin, directions,
        mesh._lower[0], mesh._upper[0], best))
    tolerance = 32 * np.finfo(float).eps
    for start in range(0, len(selected), 64):
        ray_ids = selected[start:start + 64]
        stack = [(np.zeros(len(ray_ids), dtype=np.int64), ray_ids)]
        while stack:
            nodes, ray_ids = stack.pop()
            hit = _box_hits(origin, directions[ray_ids],
                mesh._lower[nodes], mesh._upper[nodes], best[ray_ids])
            nodes, ray_ids = nodes[hit], ray_ids[hit]
            if not len(nodes):
                continue
            children = mesh._children[nodes]
            leaf = children[:, 0] < 0
            if np.any(leaf):
                ranges = mesh._ranges[nodes[leaf]]
                # TriangleMesh builds leaves with at most eight triangles.
                offsets = ranges[:, 0, None] + np.arange(8)[None]
                valid = offsets < ranges[:, 1, None]
                triangles = mesh._permutation[offsets[valid]]
                pair_rays = np.broadcast_to(ray_ids[leaf, None], offsets.shape)[valid]
                for begin in range(0, len(triangles), 1024):
                    triangle_ids, ray_indices = triangles[begin:begin + 1024], pair_rays[begin:begin + 1024]
                    rays = directions[ray_indices]
                    first, edge1, edge2 = (mesh._first[triangle_ids], mesh._edge1[triangle_ids],
                                          mesh._edge2[triangle_ids])
                    cross = np.cross(rays, edge2)
                    # Moller-Trumbore intersection: u and v are triangle barycentrics.
                    determinant = np.einsum('nk,nk->n', cross, edge1)
                    scale = np.linalg.norm(rays, axis=1) * (
                        np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1))
                    usable = abs(determinant) > tolerance * scale
                    inverse = np.zeros_like(determinant)
                    np.divide(1., determinant, out=inverse, where=usable)
                    offset = origin - first
                    u = np.einsum('nk,nk->n', cross, offset) * inverse
                    other = np.cross(offset, edge1)
                    v = np.einsum('nk,nk->n', rays, other) * inverse
                    depth = np.einsum('nk,nk->n', edge2, other) * inverse
                    usable &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (depth > 0)
                    np.minimum.at(best, ray_indices, np.where(usable, depth, np.inf))
            branch = ~leaf
            if np.any(branch):
                next_nodes, next_rays = children[branch].ravel(), np.repeat(ray_ids[branch], 2)
                for begin in range(0, len(next_nodes), 1024):
                    node_packet, ray_packet = next_nodes[begin:begin + 1024], next_rays[begin:begin + 1024]
                    stack.append((node_packet, ray_packet))
    return best


class RobotSelfFilter:
    """Exclude robot surfaces and hidden returns while retaining foreground."""
    def __init__(self, *, depth_tolerance: float = .005, pose_tolerance: float = .01) -> None:
        self.depth_tolerance, self.pose_tolerance = depth_tolerance, pose_tolerance

    def masks(self, frames: Sequence[DepthFrame], snapshots_by_camera: Mapping[str, RobotSnapshot],
              *, pixels_by_camera: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
        """Mask only mapper-selected pixels using exposure-matched robot poses."""
        masks = {}
        for frame in frames:
            robot = snapshots_by_camera[frame.camera_id]
            if (abs(robot.simulation_time - frame.token[1]) > self.pose_tolerance
                    or robot.frame_token is not None and robot.frame_token != frame.token):
                raise ValueError('Robot pose does not match camera exposure')
            mask = masks[frame.camera_id] = np.zeros(frame.depth.shape, bool)
            y, x = pixels_by_camera[frame.camera_id]
            if not len(x):
                continue
            measured = frame.depth[y, x]
            intrinsics, transform = frame.intrinsics, frame.world_from_camera
            rays = np.c_[(x + .5 - intrinsics[0, 2]) / intrinsics[0, 0],
                         (y + .5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(len(x))]
            world_rays = rays @ transform[:3, :3].T
            origin = transform[:3, 3]
            robot_depth = np.full(len(x), np.inf)
            for part in robot.parts:
                inverse = part._local_from_world
                local_origin = origin @ inverse[:3, :3].T + inverse[:3, 3]
                robot_depth = _mesh_depth(part.mesh, local_origin, world_rays @ inverse[:3, :3].T, robot_depth)
            mask[y, x] = (np.isfinite(measured) & (measured > 0) & np.isfinite(robot_depth) &
                          (measured >= robot_depth - self.depth_tolerance))
        return masks
