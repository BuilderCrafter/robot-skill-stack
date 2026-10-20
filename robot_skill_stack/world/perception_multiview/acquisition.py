"""Synchronous snapshots of distinct render frames; no simulator imports."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    camera_id: str
    token: tuple[float, float]
    depth: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    acquired_at: float
    rgb: np.ndarray | None = None

    def __post_init__(self):
        if self.depth.ndim != 2 or not self.depth.size:
            raise ValueError('Depth must be a nonempty image')
        k, t = self.intrinsics, self.world_from_camera
        if (k.shape != (3, 3) or t.shape != (4, 4) or
                not np.isfinite(k).all() or not np.isfinite(t).all() or
                min(k[0, 0], k[1, 1]) <= 0 or
                not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-8 or
                not np.allclose(t[3], [0, 0, 0, 1]) or
                np.linalg.det(t[:3, :3]) < 0 or
                not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)):
            raise ValueError('Invalid pinhole calibration or rigid transform')


class FrameSynchronizer:
    def __init__(self, cameras, *, max_skew=.025):
        if len(cameras) < 2 or len({id(c) for c in cameras.values()}) != len(cameras):
            raise ValueError('At least two independent cameras are required')
        if not np.isfinite(max_skew) or max_skew <= 0:
            raise ValueError('max_skew must be finite and positive')
        self.cameras, self.max_skew = dict(cameras), float(max_skew)
        self._consumed = {}
        self.status = {'state': 'waiting'}
        self._closed = False

    @staticmethod
    def _token(camera):
        value = camera.get_frame_token()
        if value is None:
            return None
        try:
            number, stamp = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError('Camera must expose render number and simulation timestamp') from exc
        if not np.isfinite(number) or number <= 0 or not np.isfinite(stamp) or stamp < 0:
            return None
        return number, stamp

    def capture(self):
        if self._closed:
            raise RuntimeError('Multi-view cameras are closed')
        acquired = time.monotonic()
        tokens = {key: self._token(camera) for key, camera in self.cameras.items()}
        missing = [key for key, token in tokens.items() if token is None]
        if missing:
            self.status = {'state': 'missing', 'cameras': missing}
            raise RuntimeError(f'Missing multi-view frames: {missing}')
        repeated = [key for key, token in tokens.items() if token == self._consumed.get(key)]
        if repeated:
            self.status = {'state': 'unchanged' if len(repeated) == len(tokens) else 'frozen',
                           'cameras': repeated}
            if len(repeated) != len(tokens):
                raise RuntimeError(f'Multi-view camera frozen: {repeated}')
            return None
        if any(key in self._consumed and
               (token[0] <= self._consumed[key][0] or token[1] <= self._consumed[key][1])
               for key, token in tokens.items()):
            self.status = {'state': 'clock_regression'}
            raise RuntimeError('Camera render clock regressed; reinitialize the source')
        skew = float(np.ptp([token[1] for token in tokens.values()]))
        if skew > self.max_skew:
            self.status = {'state': 'unsynchronized', 'skew_seconds': skew}
            raise RuntimeError(f'Multi-view render skew {skew:.6f}s exceeds {self.max_skew}s')
        # Consume attempted tokens, including frames whose buffers are invalid.
        # An old buffer cannot later be relabelled as a fresh acquisition.
        self._consumed = tokens
        self.status = {'state': 'acquiring', 'skew_seconds': skew}
        frames = []
        for key, camera in self.cameras.items():
            depth = camera.get_depth()
            if depth is None:
                self.status = {'state': 'missing_depth', 'cameras': [key]}
                raise RuntimeError(f'Missing multi-view depth: {key}')
            if not np.any(np.isfinite(depth) & (np.asarray(depth) > 0)):
                self.status = {'state': 'invalid_depth', 'cameras': [key]}
                raise RuntimeError(f'Invalid multi-view depth: {key}')
            rgb = camera.get_rgb()
            frames.append(DepthFrame(
                key, tokens[key], np.array(depth, dtype=float, copy=True),
                np.array(camera.get_intrinsics(), dtype=float, copy=True),
                np.array(camera.get_world_from_camera_transform(), dtype=float, copy=True),
                acquired, None if rgb is None else np.array(rgb, copy=True)))
        if any(self._token(camera) != tokens[key] for key, camera in self.cameras.items()):
            self.status = {'state': 'changed_during_acquisition'}
            raise RuntimeError('Camera updated during multi-view acquisition')
        self.status = {'state': 'synchronized', 'skew_seconds': skew,
                       'tokens': tokens}
        return tuple(frames)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for camera in reversed(tuple(self.cameras.values())):
            try:
                camera.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('Multi-view camera cleanup failed: ' + '; '.join(map(str, errors))) from errors[0]
