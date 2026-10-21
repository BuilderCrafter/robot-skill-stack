"""Model-based depth exclusion using immutable meshes; no simulator dependencies."""
from dataclasses import dataclass
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


def _readonly(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _token(value):
    try:
        token = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected a finite render-number/simulation-time token') from exc
    if len(token) != 2 or not np.isfinite(token).all() or token[0] <= 0 or token[1] < 0:
        raise ValueError('Expected a positive render number and nonnegative simulation time')
    return token


@dataclass(frozen=True, eq=False)
class TriangleMesh:
    """Owned immutable local geometry and a BVH built once, independent of pose."""
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        vertices, triangles = np.asarray(self.vertices, float), np.asarray(self.triangles)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or triangles.ndim != 2
                or triangles.shape[1] != 3 or not len(triangles)
                or triangles.dtype.kind not in 'iu' or np.any(triangles < 0)
                or np.any(triangles >= len(vertices))):
            raise ValueError('Robot meshes require finite vertices and valid triangle indices')
        points = vertices[triangles]
        first, edge1, edge2 = points[:, 0], points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
        if not np.any(np.linalg.norm(np.cross(edge1, edge2), axis=1) > 0):
            raise ValueError('Robot mesh contains no nondegenerate surface triangles')
        low, high = points.min(1), points.max(1)
        centers = (low+high)/2
        nodes, permutation = [], []

        def build(indices):
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
                left, right = build(ordered[:middle]), build(ordered[middle:])
                nodes[node] = (lower, upper, left, right, 0, 0)
            return node

        build(np.arange(len(triangles)))
        for name, value, dtype in (('vertices', vertices, float), ('triangles', triangles, np.int64),
                ('_first', first, float), ('_edge1', edge1, float), ('_edge2', edge2, float),
                ('_lower', np.array([node[0] for node in nodes]), float),
                ('_upper', np.array([node[1] for node in nodes]), float),
                ('_children', np.array([node[2:4] for node in nodes]), np.int64),
                ('_ranges', np.array([node[4:6] for node in nodes]), np.int64),
                ('_permutation', permutation, np.int64)):
            object.__setattr__(self, name, _readonly(value, dtype))


@dataclass(frozen=True, eq=False)
class MeshInstance:
    mesh: TriangleMesh
    world_from_local: np.ndarray

    def __post_init__(self):
        transform = np.asarray(self.world_from_local, float)
        if (not isinstance(self.mesh, TriangleMesh) or transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
            raise ValueError('Robot mesh instance needs finite affine geometry and transform')
        try:
            inverse = np.linalg.inv(transform)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Robot mesh transform must be invertible') from exc
        if not np.isfinite(inverse).all():
            raise ValueError('Robot mesh transform must have a finite inverse')
        object.__setattr__(self, 'world_from_local', _readonly(transform))
        object.__setattr__(self, '_local_from_world', _readonly(inverse))


@dataclass(frozen=True)
class RobotSnapshot:
    parts: tuple[MeshInstance, ...]
    simulation_time: float
    frame_token: tuple | None = None

    def __post_init__(self):
        parts = tuple(self.parts)
        if not parts or any(not isinstance(part, MeshInstance) for part in parts):
            raise ValueError('Robot snapshot requires nonempty validated mesh instances')
        stamp = float(self.simulation_time)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError('Robot snapshot simulation time must be finite and nonnegative')
        object.__setattr__(self, 'parts', parts)
        object.__setattr__(self, 'simulation_time', stamp)
        if self.frame_token is not None:
            object.__setattr__(self, 'frame_token', _token(self.frame_token))


def _box_hits(origin, directions, lower, upper, limit):
    """Ray/AABB broadphase only; touching a box is never a robot mask."""
    moving = directions != 0
    first, second = np.full(directions.shape, -np.inf), np.full(directions.shape, np.inf)
    np.divide(lower-origin, directions, out=first, where=moving)
    np.divide(upper-origin, directions, out=second, where=moving)
    near, far = np.minimum(first, second).max(1), np.maximum(first, second).min(1)
    parallel_outside = (~moving & ((origin < lower) | (origin > upper))).any(1)
    return ~parallel_outside & (far >= np.maximum(near, 0.)) & (near <= limit)


def _mesh_depth(mesh, origin, directions, limit):
    """Nearest positive triangle parameter for each unnormalized local ray."""
    best = np.asarray(limit, float).copy()
    stack = [(0, np.arange(len(directions)))]
    while stack:
        node, selected = stack.pop()
        selected = selected[_box_hits(origin, directions[selected],
            mesh._lower[node], mesh._upper[node], best[selected])]
        if not len(selected):
            continue
        left, right = mesh._children[node]
        if left >= 0:
            stack.extend(((int(right), selected), (int(left), selected)))
            continue
        start, stop = mesh._ranges[node]
        triangles = mesh._permutation[start:stop]
        first, edge1, edge2 = mesh._first[triangles], mesh._edge1[triangles], mesh._edge2[triangles]
        # Leaves have at most eight triangles. Chunk rays to bound temporaries
        # even for caller-selected dense sampling or very large camera images.
        for begin in range(0, len(selected), 256):
            ids = selected[begin:begin+256]
            rays = directions[ids]
            cross = np.cross(rays[:, None], edge2[None])
            determinant = np.einsum('ntk,tk->nt', cross, edge1)
            scale = np.linalg.norm(rays, axis=1)[:, None]*(
                np.linalg.norm(edge1, axis=1)*np.linalg.norm(edge2, axis=1))[None]
            usable = abs(determinant) > 32*np.finfo(float).eps*scale
            inverse = np.zeros_like(determinant)
            np.divide(1., determinant, out=inverse, where=usable)
            offset = origin-first
            u = np.einsum('ntk,tk->nt', cross, offset)*inverse
            other = np.cross(offset, edge1)
            v = np.einsum('nk,tk->nt', rays, other)*inverse
            depth = np.einsum('tk,tk->t', edge2, other)[None]*inverse
            hit = usable & (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10) & (depth > 0)
            best[ids] = np.minimum(best[ids], np.where(hit, depth, np.inf).min(1))
    return best


class RobotSelfFilter:
    """Withhold robot matches/occluded returns; retain clear foreground objects.

    Only stride-grid pixels are evaluated. Consumers must use the same stride
    as their mapper and may restrict evaluation to its selected ray pixels.
    Masks skip whole rays; they neither add occupied robot volumes nor prove
    any space in front of or behind the robot free.
    """
    def __init__(self, *, depth_tolerance=.005, pose_tolerance=.01, stride=8):
        if any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
               for value in (depth_tolerance, pose_tolerance)):
            raise ValueError('Robot depth/pose tolerances must be finite and positive')
        if type(stride) is not int or stride < 1:
            raise ValueError('Robot filter stride must be a positive integer')
        self.depth_tolerance, self.pose_tolerance, self.stride = depth_tolerance, pose_tolerance, stride
        self.diagnostics = {'state': 'waiting'}

    def masks(self, frames, snapshots_by_camera, *, pixels_by_camera=None):
        """Return depth-shaped masks for all stride pixels or an explicit subset.

        Explicit pixel selections are complete camera-ID mappings to ``(ys,xs)``
        integer vectors. Each pair must be unique, in bounds and on this filter's
        stride grid. Empty vectors are valid for views given no ray budget;
        unselected pixels remain False rather than being labeled robot pixels.
        """
        started = time.perf_counter()
        self.diagnostics = dict(state='validating', cameras={})
        try:
            frames, snapshots = tuple(frames), dict(snapshots_by_camera)
            ids = tuple(frame.camera_id for frame in frames)
            if (not frames or any(not isinstance(frame, DepthFrame) for frame in frames)
                    or any(not isinstance(key, str) or not key for key in ids)
                    or len(set(ids)) != len(ids) or set(snapshots) != set(ids)):
                raise ValueError('Robot snapshots must match every distinct depth camera')
            selections = None if pixels_by_camera is None else dict(pixels_by_camera)
            if selections is not None and set(selections) != set(ids):
                raise ValueError('Robot pixel selections must match every depth camera')
            for frame in frames:
                key, token = frame.camera_id, _token(frame.token)
                DepthFrame(key, frame.token, frame.depth, frame.intrinsics,
                           frame.world_from_camera, frame.acquired_at)
                snapshot = snapshots[key]
                if not isinstance(snapshot, RobotSnapshot):
                    raise ValueError(f'Missing validated robot snapshot: {key}')
                if (abs(snapshot.simulation_time-token[1]) > self.pose_tolerance
                        or snapshot.frame_token is not None and snapshot.frame_token != token):
                    raise ValueError(f'Robot pose does not match camera render time/token: {key}')
                if selections is not None:
                    pair = selections[key]
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                        raise ValueError(f'Robot pixel selections need (ys,xs) vectors: {key}')
                    y, x = (np.asarray(values) for values in pair)
                    if (y.ndim != 1 or x.ndim != 1 or y.shape != x.shape
                            or y.dtype.kind not in 'iu' or x.dtype.kind not in 'iu'
                            or np.any(y < 0) or np.any(y >= frame.depth.shape[0])
                            or np.any(x < 0) or np.any(x >= frame.depth.shape[1])
                            or np.any(y % self.stride) or np.any(x % self.stride)):
                        raise ValueError(f'Robot pixels must be integer stride-grid image coordinates: {key}')
                    if len(np.unique(y*frame.depth.shape[1]+x)) != len(y):
                        raise ValueError(f'Robot pixel selections must be unique: {key}')
                    selections[key] = (y.copy(), x.copy())
            masks = {}
            for frame in frames:
                key, snapshot = frame.camera_id, snapshots[frame.camera_id]
                mask = np.zeros(frame.depth.shape, bool)
                if selections is None:
                    y, x = np.indices(frame.depth[::self.stride, ::self.stride].shape)
                    y, x = y.ravel()*self.stride, x.ravel()*self.stride
                else:
                    y, x = selections[key]
                measured = frame.depth[y, x]
                valid = np.isfinite(measured) & (measured > 0)
                y, x, measured = y[valid], x[valid], measured[valid]
                k, transform = frame.intrinsics, frame.world_from_camera
                camera_rays = np.c_[(x+.5-k[0, 2])/k[0, 0],
                                    (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))]
                world_rays = camera_rays @ transform[:3, :3].T
                origin = transform[:3, 3]
                expected = np.full(len(x), np.inf)
                for part in snapshot.parts:
                    inverse = part._local_from_world
                    local_origin = origin @ inverse[:3, :3].T+inverse[:3, 3]
                    local_rays = world_rays @ inverse[:3, :3].T
                    expected = _mesh_depth(part.mesh, local_origin, local_rays, expected)
                hit = np.isfinite(expected)
                self_match = hit & (abs(measured-expected) <= self.depth_tolerance)
                behind = hit & (measured > expected+self.depth_tolerance)
                foreground = hit & (measured < expected-self.depth_tolerance)
                mask[y, x] = self_match | behind
                masks[key] = mask
                self.diagnostics['cameras'][key] = dict(sampled_pixels=valid.size,
                    valid_pixels=int(valid.sum()), self_match=int(self_match.sum()),
                    behind_withheld=int(behind.sum()), foreground_kept=int(foreground.sum()),
                    no_hit=int((~hit).sum()), masked_pixels=int(mask.sum()),
                    pose_skew_seconds=abs(snapshot.simulation_time-frame.token[1]))
            self.diagnostics.update(state='ready', elapsed_seconds=time.perf_counter()-started)
            return masks
        except Exception as exc:
            self.diagnostics.update(state='error', reason=str(exc), elapsed_seconds=time.perf_counter()-started)
            raise
