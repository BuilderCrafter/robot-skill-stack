"""Synchronous snapshots of distinct render frames; no simulator imports."""
from collections.abc import Mapping
from dataclasses import dataclass
import time

import numpy as np


def _render_token(value: tuple[float, float] | None) -> tuple[float, float] | None:
    """Normalize a render number and simulation time, or report no exposure."""
    if value is None:
        return None
    number, stamp = map(float, value)
    if not np.isfinite([number, stamp]).all() or number <= 0 or stamp < 0:
        return None
    return number, stamp


@dataclass(frozen=True)
class DepthFrame:
    """One camera's depth frame, calibration, render token, and capture time."""
    camera_id: str
    token: tuple[float, float]
    depth: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    acquired_at: float
    rgb: np.ndarray | None = None

    def __post_init__(self):
        # Projection assumes zero-skew pinhole intrinsics and a right-handed
        # rigid camera-to-world transform; malformed calibration corrupts XYZ.
        k, t = self.intrinsics, self.world_from_camera
        if (k.shape != (3, 3) or t.shape != (4, 4) or
                not np.isfinite(k).all() or not np.isfinite(t).all() or
                min(k[0, 0], k[1, 1]) <= 0 or
                not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) + abs(k[1, 0]) > 1e-8 or
                not np.allclose(t[3], [0, 0, 0, 1]) or
                np.linalg.det(t[:3, :3]) < 0 or
                not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)):
            raise ValueError('Invalid pinhole calibration or rigid transform')


class FrameSynchronizer:
    """Own configured cameras and admit complete, distinct synchronized exposures."""
    def __init__(self, cameras: Mapping, *, max_skew: float = .025):
        self.cameras, self.max_skew = dict(cameras), max_skew
        self._consumed = {}
        self._closed = False

    def capture(self) -> tuple[DepthFrame, ...] | None:
        """Consume one synchronized attempt, copy its buffers, and verify exposure."""
        if self._closed:
            raise RuntimeError('Multi-view cameras are closed')
        acquired = time.monotonic()
        tokens = {key: _render_token(camera.get_frame_token()) for key, camera in self.cameras.items()}
        if any(token is None for token in tokens.values()):
            raise RuntimeError('Missing multi-view frames')
        repeated = sum(token == self._consumed.get(key) for key, token in tokens.items())
        if repeated == len(tokens):
            return None
        if repeated:
            raise RuntimeError('Multi-view camera frozen')
        if any(key in self._consumed and
               (token[0] <= self._consumed[key][0] or token[1] <= self._consumed[key][1])
               for key, token in tokens.items()):
            raise RuntimeError('Camera render clock regressed; reinitialize the source')
        skew = np.ptp([token[1] for token in tokens.values()])
        if skew > self.max_skew:
            raise RuntimeError(f'Multi-view render skew {skew:.6f}s exceeds {self.max_skew}s')
        # Consume before reading buffers: a failed attempt cannot be repaired
        # and replayed later as a fresh observation.
        self._consumed = tokens
        frames = []
        for key, camera in self.cameras.items():
            depth = camera.get_depth()
            if depth is None or not np.any(np.isfinite(depth) & (depth > 0)):
                raise RuntimeError(f'Missing or invalid multi-view depth: {key}')
            rgb = camera.get_rgb()
            frames.append(DepthFrame(
                key, tokens[key], np.array(depth, dtype=float, copy=True),
                np.array(camera.get_intrinsics(), dtype=float, copy=True),
                np.array(camera.get_world_from_camera_transform(), dtype=float, copy=True),
                acquired, None if rgb is None else np.array(rgb, copy=True)))
        if any(_render_token(camera.get_frame_token()) != tokens[key] for key, camera in self.cameras.items()):
            raise RuntimeError('Camera updated during multi-view acquisition')
        return tuple(frames)

    def close(self) -> None:
        """Close every owned camera once, retaining the first cleanup failure."""
        if self._closed:
            return
        self._closed = True
        failure = None
        for camera in reversed(tuple(self.cameras.values())):
            try:
                camera.close()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
