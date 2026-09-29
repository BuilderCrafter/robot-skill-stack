"""Batched pinhole triangle rasterization at integer sensor-pixel centers."""
import numpy as np


def raster_depth(triangles, intrinsics, shape, budget=65536):
    """Return camera-Z depth. Triangles crossing the camera plane need ray fallback."""
    h, w = shape
    t = np.asarray(triangles, float)
    if not len(t):
        return np.full(shape, np.inf)
    if np.any(t[..., 2] <= 1e-7):
        raise ValueError('Raster triangles must be in front of the camera plane')
    uv = (t @ np.asarray(intrinsics).T)[..., :2] / t[..., 2, None]
    # Pixel coordinates match back-projection: (u-cx)/fx, not (u+0.5-cx)/fx.
    low = np.maximum(np.ceil(uv.min(1)-1e-7), [0, 0])
    high = np.minimum(np.floor(uv.max(1)+1e-7), [w-1, h-1])
    ok = (high >= low).all(1)
    uv, t, low, high = uv[ok], t[ok], low[ok].astype(int), high[ok].astype(int)
    result = np.full(h*w, np.inf)
    if not len(t):
        return result.reshape(shape)
    sizes = high-low+1
    buckets = np.minimum(2**np.ceil(np.log2(sizes)).astype(int), [w, h])
    keys = buckets[:, 0]*(h+1)+buckets[:, 1]
    for key in np.unique(keys):
        ids = np.flatnonzero(keys == key)
        bw, bh = buckets[ids[0]]
        area = int(bw*bh)
        step = max(1, budget//area)
        for start in range(0, len(ids), step):
            ii = ids[start:start+step]
            a, b, c = uv[ii, 0], uv[ii, 1], uv[ii, 2]
            den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
            nonzero = abs(den) > 1e-12
            inv = np.divide(1., den, out=np.zeros_like(den), where=nonzero)
            for offset in range(0, area, max(1, budget//len(ii))):
                offsets = np.arange(offset, min(area, offset+max(1, budget//len(ii))))
                x = low[ii, 0, None] + offsets[None, :] % bw
                y = low[ii, 1, None] + offsets[None, :] // bw
                dx, dy = x-c[:, 0, None], y-c[:, 1, None]
                l0 = ((b[:, 1]-c[:, 1])[:, None]*dx + (c[:, 0]-b[:, 0])[:, None]*dy)*inv[:, None]
                l1 = ((c[:, 1]-a[:, 1])[:, None]*dx + (a[:, 0]-c[:, 0])[:, None]*dy)*inv[:, None]
                l2 = 1.-l0-l1
                inside = (nonzero[:, None] & (x <= high[ii, 0, None]) & (y <= high[ii, 1, None])
                          & (l0 >= -1e-8) & (l1 >= -1e-8) & (l2 >= -1e-8))
                reciprocal = l0/t[ii, 0, 2, None] + l1/t[ii, 1, 2, None] + l2/t[ii, 2, 2, None]
                inside &= reciprocal > 0
                ri, pi = np.nonzero(inside)
                np.minimum.at(result, y[ri, pi]*w+x[ri, pi], 1./reciprocal[ri, pi])
    return result.reshape(shape)
