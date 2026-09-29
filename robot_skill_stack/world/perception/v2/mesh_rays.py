"""Small NumPy triangle BVH. Ray parameters are camera-Z depth, not ray length."""
import numpy as np


class TriangleBVH:
    def __init__(self, triangles, leaf_size=8):
        t = np.asarray(triangles, dtype=float)
        if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.isfinite(t).all():
            raise ValueError('Expected finite Nx3x3 robot triangles')
        area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
        self.triangles = t[area > 1e-12].copy()
        if not len(self.triangles):
            raise ValueError('Robot mesh has no nondegenerate triangles')
        self.triangles.setflags(write=False)
        centers = self.triangles.mean(1)
        self.nodes = []

        def build(ids):
            pts = self.triangles[ids]
            index = len(self.nodes)
            self.nodes.append(None)
            lo, hi = pts.min((0, 1)), pts.max((0, 1))
            if len(ids) <= leaf_size:
                self.nodes[index] = (lo, hi, ids, None)
            else:
                axis = np.argmax(np.ptp(centers[ids], axis=0))
                ids = ids[np.argsort(centers[ids, axis], kind='stable')]
                cut = len(ids)//2
                self.nodes[index] = (lo, hi, None, (build(ids[:cut]), build(ids[cut:])))
            return index
        build(np.arange(len(self.triangles)))

    @staticmethod
    def _box(origin, directions, lo, hi, limits):
        parallel = np.abs(directions) < 1e-14
        outside = ((origin < lo-1e-9) | (origin > hi+1e-9)) & parallel
        safe = np.where(parallel, 1., directions)
        a, b = (lo-origin)/safe, (hi-origin)/safe
        near = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=1)
        far = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=1)
        return ~outside.any(1) & (far >= np.maximum(near, 1e-7)) & (near <= limits)

    def intersect(self, origin, directions):
        o, d = np.asarray(origin, float), np.asarray(directions, float)
        if o.shape != (3,) or d.ndim != 2 or d.shape[1] != 3 or not np.isfinite(o).all() or not np.isfinite(d).all():
            raise ValueError('Invalid ray origin/directions')
        nearest = np.full(len(d), np.inf)
        stack = [(0, np.arange(len(d)))]
        while stack:
            ni, rays = stack.pop()
            lo, hi, ids, children = self.nodes[ni]
            rays = rays[self._box(o, d[rays], lo, hi, nearest[rays])]
            if not len(rays):
                continue
            if children is not None:
                stack.extend((ci, rays) for ci in children)
                continue
            # Bounded leaf batches avoid a pixels x complete-mesh allocation.
            t = self.triangles[ids]
            e1, e2 = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
            s = o-t[:, 0]
            q = np.cross(s, e1)
            for start in range(0, len(rays), 4096):
                ri = rays[start:start+4096]
                dr = d[ri, None, :]
                h = np.cross(dr, e2[None, :, :])
                det = np.einsum('mti,ti->mt', h, e1)
                ok = np.abs(det) > 1e-12
                inv = np.divide(1., det, out=np.zeros_like(det), where=ok)
                u = np.einsum('mti,ti->mt', h, s)*inv
                v = np.einsum('mti,ti->mt', dr, q)*inv
                depth = np.sum(e2*q, axis=1)[None, :]*inv
                hit = ok & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8) & (depth > 1e-7)
                nearest[ri] = np.minimum(nearest[ri], np.min(np.where(hit, depth, np.inf), axis=1))
        return nearest


def triangulate(points, counts, indices, holes=()):
    """USD face-vertex conversion; robot meshes must have triangle/convex-quad faces."""
    points, counts, indices = np.asarray(points, float), np.asarray(counts, int), np.asarray(indices, int)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
            counts.ndim != 1 or indices.ndim != 1 or counts.sum() != len(indices) or
            np.any((counts < 3) | (counts > 4)) or (len(indices) and (indices.min() < 0 or indices.max() >= len(points)))):
        raise ValueError('Robot mesh must contain finite points and valid triangle/convex-quad faces')
    holes = set(map(int, holes))
    out, start = [], 0
    for face, n in enumerate(counts):
        f = indices[start:start+n]
        start += n
        if face in holes:
            continue
        if n == 4:
            p = points[f]
            normals = np.cross(np.roll(p, -1, axis=0)-p, np.roll(p, -2, axis=0)-np.roll(p, -1, axis=0))
            if np.any(normals @ normals[0] < -1e-10):
                raise ValueError('Concave quad in robot mesh; triangulate the robot asset first')
        out.extend(points[[f[0], f[j], f[j+1]]] for j in range(1, n-1))
    return np.asarray(out, float).reshape(-1, 3, 3)
