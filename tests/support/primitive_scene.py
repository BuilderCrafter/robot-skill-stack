"""Small analytic pinhole renderer for depth-only, single-view regression tests."""
from dataclasses import dataclass
import numpy as np

from robot_skill_stack.world.perception.frame import PerceptionFrame


@dataclass
class Solid:
    shape: str
    center: tuple = (.45, 0., .025)
    size: tuple = (.05, .05, .05)
    yaw: float = 0.
    axis: tuple = (0., 0., 1.)
    radius: float = .025
    length: float = .10


def camera(eye=(1.05, .55, .90), target=(.45, 0., .025), resolution=(320, 240)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((right, down, forward))
    T[:3, 3] = eye
    w, h = resolution
    K = np.array([[.95*w, 0., w/2], [0., .95*w, h/2], [0., 0., 1.]])
    y, x = np.indices((h, w))
    rays = np.c_[(x.ravel()-K[0, 2])/K[0, 0], (y.ravel()-K[1, 2])/K[1, 1], np.ones(w*h)]
    return K, T, rays @ T[:3, :3].T


def _quadratic(a, b, c):
    discriminant = b*b - 4*a*c
    root = np.sqrt(np.maximum(discriminant, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        near, far = (-b-root)/(2*a), (-b+root)/(2*a)
    valid = (discriminant >= 0) & (a > 1e-12)
    return np.where(valid & (near > 0), near, np.inf), np.where(valid & (far > 0), far, np.inf)


def intersect(solid, origin, rays):
    p = origin - np.asarray(solid.center)
    if solid.shape in ('sphere', 'ellipsoid'):
        scale = np.repeat(solid.radius, 3) if solid.shape == 'sphere' else np.asarray(solid.size)/2
        d, q = rays/scale, p/scale
        near, far = _quadratic((d*d).sum(1), 2*(d*q).sum(1), (q*q).sum()-1)
        return np.minimum(near, far)
    if solid.shape == 'cube':
        c, s = np.cos(solid.yaw), np.sin(solid.yaw)
        R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        d, q = rays @ R, p @ R
        half = np.asarray(solid.size)/2
        with np.errstate(divide='ignore', invalid='ignore'):
            t0, t1 = (-half-q)/d, (half-q)/d
        near, far = np.minimum(t0, t1).max(1), np.maximum(t0, t1).min(1)
        return np.where((far >= near) & (far > 0), np.where(near > 0, near, far), np.inf)
    if solid.shape != 'cylinder':
        raise ValueError(solid.shape)
    a = np.asarray(solid.axis, float)
    a /= np.linalg.norm(a)
    dz, pz = rays @ a, p @ a
    d, q = rays - dz[:, None]*a, p - pz*a
    near, far = _quadratic((d*d).sum(1), 2*(d*q).sum(1), q@q-solid.radius**2)
    hits = []
    for t in (near, far):
        with np.errstate(invalid='ignore'):
            hits.append(np.where(abs(pz+t*dz) <= solid.length/2, t, np.inf))
    for end in (-solid.length/2, solid.length/2):
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (end-pz)/dz
            radial = q + t[:, None]*d
        hits.append(np.where((t > 0) & ((radial*radial).sum(1) <= solid.radius**2), t, np.inf))
    return np.minimum.reduce(hits)


def render(solids=(), *, eye=(1.05, .55, .90), resolution=(320, 240), noise=0., dropout=0., seed=1, plane_z=0.):
    K, T, rays = camera(eye=eye, resolution=resolution)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (plane_z-T[2, 3])/rays[:, 2]
    depth = np.where(depth > 0, depth, np.inf)
    for solid in solids:
        depth = np.minimum(depth, intersect(solid, T[:3, 3], rays))
    rng = np.random.default_rng(seed)
    depth += rng.normal(0., noise, depth.shape)
    depth[rng.random(len(depth)) < dropout] = np.nan
    w, h = resolution
    return PerceptionFrame(1, np.zeros((h, w, 3), np.uint8), depth.reshape(h, w).astype(np.float32), K, T, 1.)


class FrameCamera:
    def __init__(self, frame):
        self.frame, self.token = frame, 1
    def get_rgb(self): return self.frame.rgb
    def get_depth(self): return self.frame.depth
    def get_intrinsics(self): return self.frame.intrinsics
    def get_world_from_camera_transform(self): return self.frame.world_from_camera
    def get_frame_token(self): return self.token
    def push(self, frame):
        self.frame = frame
        self.token += 1
