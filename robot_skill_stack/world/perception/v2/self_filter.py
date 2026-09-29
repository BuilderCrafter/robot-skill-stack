"""Immutable robot-only surface model, with no simulator or environment-object APIs."""
from dataclasses import dataclass, field
from itertools import product
import hashlib
import time
import numpy as np
from robot_skill_stack.world.perception.v2.mesh_rays import TriangleBVH
from robot_skill_stack.world.perception.v2.mesh_raster import raster_depth


def rigid_matrices(matrices):
    t = np.asarray(matrices, float)
    if (t.ndim != 3 or t.shape[1:] != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[:, 3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(t[:, :3, :3].transpose(0, 2, 1) @ t[:, :3, :3], np.eye(3), atol=1e-4) or
            np.any(np.linalg.det(t[:, :3, :3]) < .999)):
        raise ValueError('Robot link poses must be finite rigid world transforms')
    return t


@dataclass(frozen=True)
class RobotSnapshot:
    model: object
    world_from_links: np.ndarray
    camera_time: float
    pose_time: float
    token: object = None

    def __post_init__(self):
        t = rigid_matrices(self.world_from_links).copy()
        if len(t) != len(self.model.names) or not np.isfinite([self.camera_time, self.pose_time]).all():
            raise ValueError('Invalid robot snapshot link count/time')
        t.setflags(write=False)
        object.__setattr__(self, 'world_from_links', t)


@dataclass
class SelfFilterResult:
    mask: np.ndarray
    model_depth: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self, shape):
        if (self.mask.shape != shape or self.mask.dtype != np.bool_ or self.model_depth.shape != shape or
                np.any(np.isfinite(self.model_depth) & (self.model_depth <= 0))):
            raise ValueError('Invalid same-frame robot exclusion data')
        return self


class RobotSurfaceModel:
    def __init__(self, meshes):
        if not meshes or len({name for name, _ in meshes}) != len(meshes):
            raise ValueError('Robot model needs unique nonempty link meshes')
        self.names = tuple(str(name) for name, _ in meshes)
        self.bvhs = tuple(TriangleBVH(triangles) for _, triangles in meshes)
        digest = hashlib.sha256()
        for name, bvh in zip(self.names, self.bvhs):
            digest.update(name.encode()); digest.update(bvh.triangles.astype('<f8').tobytes())
        self.sha256 = digest.hexdigest()
        self.triangle_count = sum(len(b.triangles) for b in self.bvhs)

    def depth_reference(self, frame, poses):
        poses = rigid_matrices(poses)
        if len(poses) != len(self.names):
            raise ValueError('Robot pose/model mismatch')
        y, x = np.indices(frame.depth.shape)
        k = frame.intrinsics
        rays = np.column_stack(((x.ravel()-k[0, 2])/k[0, 0], (y.ravel()-k[1, 2])/k[1, 1], np.ones(x.size)))
        camera = frame.world_from_camera
        world_rays = rays @ camera[:3, :3].T
        nearest = np.full(len(rays), np.inf)
        h, w = frame.depth.shape
        for bvh, pose in zip(self.bvhs, poses):
            lo, hi = bvh.nodes[0][:2]
            corners = np.where(np.asarray(list(product((False, True), repeat=3))), hi, lo)
            vertices = (corners @ pose[:3, :3].T+pose[:3, 3]-camera[:3, 3]) @ camera[:3, :3]
            if np.max(vertices[:, 2]) <= 1e-7:
                continue
            x0, y0, x1, y1 = 0, 0, w, h
            if np.min(vertices[:, 2]) > 1e-7:
                uv = (vertices @ k.T)[:, :2] / vertices[:, 2, None]
                x0, y0 = np.maximum(np.floor(uv.min(0)).astype(int)-1, [0, 0])
                x1, y1 = np.minimum(np.ceil(uv.max(0)).astype(int)+2, [w, h])
            if x1 <= x0 or y1 <= y0:
                continue
            ids = (np.arange(y0, y1)[:, None]*w+np.arange(x0, x1)).ravel()
            origin = (camera[:3, 3]-pose[:3, 3]) @ pose[:3, :3]
            directions = world_rays[ids] @ pose[:3, :3]
            nearest[ids] = np.minimum(nearest[ids], bvh.intersect(origin, directions))
        return nearest.reshape(frame.depth.shape).astype(np.float32)


    def depth(self, frame, poses):
        poses = rigid_matrices(poses)
        if len(poses) != len(self.names):
            raise ValueError('Robot pose/model mismatch')
        camera = frame.world_from_camera
        nearest = np.full(frame.depth.shape, np.inf)
        crossing = []
        for bvh, pose in zip(self.bvhs, poses):
            t = ((bvh.triangles @ pose[:3, :3].T + pose[:3, 3]) - camera[:3, 3]) @ camera[:3, :3]
            front = (t[..., 2] > 1e-7).all(1)
            nearest = np.minimum(nearest, raster_depth(t[front], frame.intrinsics, frame.depth.shape))
            cross = ~front & (t[..., 2] > 1e-7).any(1)
            if cross.any():
                crossing.append(t[cross])
        if crossing:
            # Rare near-plane intersections keep exact ray semantics, not dropped faces.
            y, x = np.indices(frame.depth.shape)
            k = frame.intrinsics
            rays = np.column_stack(((x.ravel()-k[0, 2])/k[0, 0], (y.ravel()-k[1, 2])/k[1, 1], np.ones(x.size)))
            depth = TriangleBVH(np.concatenate(crossing)).intersect(np.zeros(3), rays)
            nearest = np.minimum(nearest, depth.reshape(frame.depth.shape))
        return nearest.astype(np.float32)


def filter_robot(frame, snapshot, tolerance, time_tolerance):
    if abs(snapshot.camera_time-snapshot.pose_time) > time_tolerance:
        raise ValueError('Robot pose does not match the camera acquisition time')
    start = time.perf_counter()
    model_depth = snapshot.model.depth(frame, snapshot.world_from_links)
    valid = np.isfinite(frame.depth) & (frame.depth > 0) & np.isfinite(model_depth)
    difference = np.full(frame.depth.shape, np.inf)
    np.subtract(frame.depth, model_depth, out=difference, where=valid)
    mask = valid & (np.abs(difference) <= tolerance)
    return SelfFilterResult(mask, model_depth, dict(
        enabled=True, renderer='batched_triangle_raster', frame_id=frame.frame_id, source='known_robot_meshes_and_measured_link_poses', model_sha256=snapshot.model.sha256,
        link_names=list(snapshot.model.names), triangles=snapshot.model.triangle_count,
        world_from_links=snapshot.world_from_links.tolist(), camera_time=snapshot.camera_time,
        pose_time=snapshot.pose_time, time_error_s=abs(snapshot.camera_time-snapshot.pose_time),
        frame_token=str(snapshot.token), surface_tolerance_m=tolerance, removed_pixels=int(mask.sum()),
        predicted_robot_pixels=int(np.isfinite(model_depth).sum()),
        preserved_foreground_pixels=int((valid & (difference < -tolerance)).sum()),
        render_s=time.perf_counter()-start,
    ))
