"""Tessellate analytic robot geometry only (not estimated objects)."""
import numpy as np


def box_triangles(size):
    s = np.broadcast_to(np.asarray(size, float), (3,))
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('Invalid box size')
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*s/2
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([v[[a, b, c]] for a, b, c, d in faces]+[v[[a, c, d]] for a, b, c, d in faces])


def round_triangles(kind, radius, height=0., axis='Z', segments=48):
    if kind not in {'Sphere', 'Cylinder', 'Capsule', 'Cone'} or radius <= 0 or height < 0 or axis not in 'XYZ':
        raise ValueError('Unsupported robot round primitive')
    if not np.isfinite([radius, height]).all():
        raise ValueError('Nonfinite robot primitive')
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    if kind in {'Sphere', 'Capsule'}:
        phi = np.linspace(-np.pi/2, np.pi/2, segments//2+1)
        z, r = radius*np.sin(phi), radius*np.cos(phi)
        if kind == 'Capsule':
            z += np.where(phi < 0, -height/2, height/2)
            i = len(phi)//2
            z, r = np.insert(z, i, -height/2), np.insert(r, i, radius)
    else:
        z = np.array([-height/2, -height/2, height/2, height/2])
        r = np.array([0., radius, radius if kind == 'Cylinder' else 0., 0.])
    rings = np.stack([r[:, None]*np.cos(theta), r[:, None]*np.sin(theta), np.broadcast_to(z[:, None], (len(z), segments))], axis=2)
    triangles = []
    for i in range(len(z)-1):
        for j in range(segments):
            k = (j+1)%segments
            triangles.extend(([rings[i, j], rings[i+1, j], rings[i+1, k]], [rings[i, j], rings[i+1, k], rings[i, k]]))
    t = np.asarray(triangles)
    return t[..., {'Z': [0, 1, 2], 'X': [2, 0, 1], 'Y': [0, 2, 1]}[axis]]
