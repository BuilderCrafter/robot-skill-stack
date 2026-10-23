"""Small owning-context frame cache with independent complete-batch admission.

Camera objects are borrowed. One consumer's frozen extra camera cannot block a
different subset. Copied buffers retain their first acquisition time and token.
"""
import time

import numpy as np

from .acquisition import DepthFrame, FrameSynchronizer


def _owned(values, dtype=None):
    values = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


class SharedFramePool:
    def __init__(self, cameras, *, token_reader=FrameSynchronizer._token, clock=time.monotonic):
        if not cameras or len({id(camera) for camera in cameras.values()}) != len(cameras):
            raise ValueError('Shared cameras must have distinct identities')
        self.cameras = dict(cameras)
        self._token_reader, self._clock = token_reader, clock
        self._consumed, self._cache, self.status = {}, {}, {}
        self._blocked_tokens = {}
        self._quarantined_tokens = {}
        self._acquired_at = None
        self.acquisitions = self.shared_reads = self.failures = 0
        self._closed = False

    def begin_update(self):
        self._acquired_at = None

    def _token(self, camera):
        value = self._token_reader(camera)
        if value is None:
            return None
        try:
            number, stamp = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError('Camera requires a render number and simulation timestamp') from exc
        if not np.isfinite(number) or number <= 0 or not np.isfinite(stamp) or stamp < 0:
            return None
        return number, stamp

    def reset(self):
        # A render buffer retained through Stop/Play is still the old exposure.
        self._blocked_tokens.update({key: value[0] for key, value in self._cache.items()})
        self._consumed.clear()
        self._cache.clear()
        self.status.clear()
        self.begin_update()

    def batch(self, consumer, paths, *, max_skew=.025, simulation_time=None):
        if self._closed:
            raise RuntimeError('Shared frame pool is closed')
        paths = tuple(paths)
        if not paths or len(set(paths)) != len(paths) or set(paths)-self.cameras.keys():
            raise ValueError('A consumer requires distinct configured camera IDs')
        if not np.isfinite(max_skew) or max_skew <= 0:
            raise ValueError('max_skew must be finite and positive')
        try:
            tokens = {key: self._token(self.cameras[key]) for key in paths}
            if any(token is None for token in tokens.values()):
                raise RuntimeError('Missing camera render token')
            if simulation_time is not None:
                if not np.isfinite(simulation_time) or simulation_time < 0:
                    raise ValueError('Simulation time must be finite and nonnegative')
                future = [key for key, token in tokens.items() if token[1] > simulation_time+max_skew]
                if future:
                    # A renderer can deliver an in-flight pre-Stop exposure
                    # after Play, ahead of the newly reset simulation clock.
                    # Do not establish this as the new generation's first token.
                    # Bounded per-camera quarantine: a frozen rejected exposure
                    # stays invalid even when the restarted clock catches up.
                    self._quarantined_tokens.update({key: tokens[key] for key in future})
                    raise RuntimeError('Waiting for camera exposure in the current simulation generation')
            blocked = [key for key in paths if tokens[key] in
                       (self._blocked_tokens.get(key), self._quarantined_tokens.get(key))]
            if blocked:
                if len(blocked) == len(paths):
                    self.status[consumer] = dict(state='unchanged_after_restart', cameras=blocked)
                    return None
                raise RuntimeError(f'Camera retained a pre-restart exposure: {blocked}')
            previous = self._consumed.get(consumer, {})
            repeated = [key for key in paths if previous.get(key) == tokens[key]]
            if repeated:
                if len(repeated) == len(paths):
                    self.status[consumer] = dict(state='unchanged', cameras=repeated)
                    return None
                raise RuntimeError(f'Camera frozen: {repeated}')
            if any(key in previous and (token[0] <= previous[key][0] or
                    token[1] <= previous[key][1]) for key, token in tokens.items()):
                raise RuntimeError('Camera render clock regressed; reinitialize the runtime')
            skew = float(np.ptp([token[1] for token in tokens.values()]))
            if skew > max_skew:
                raise RuntimeError(f'Camera render skew {skew:.6f}s exceeds {max_skew}s')
            # Admit the complete attempted batch before any buffers/robot pose
            # preparation can fail. It cannot be repaired and replayed later.
            self._consumed[consumer] = tokens
            if self._acquired_at is None:
                self._acquired_at = self._clock()
            frames = tuple(self._frame(key, tokens[key]) for key in paths)
            if any(self._token(self.cameras[key]) != tokens[key] for key in paths):
                for key in paths:
                    self._cache[key] = (tokens[key], None, 'Camera updated during acquisition')
                raise RuntimeError('Camera updated during acquisition')
            self.status[consumer] = dict(state='synchronized', tokens=tokens,
                                         skew_seconds=skew)
            return frames
        except Exception as exc:
            self.failures += 1
            self.status[consumer] = dict(state='error', reason=str(exc))
            raise

    def _frame(self, key, token):
        cached = self._cache.get(key)
        if cached is not None and cached[0] == token:
            if cached[2] is not None:
                raise RuntimeError(cached[2])
            self.shared_reads += 1
            return cached[1]
        camera = self.cameras[key]
        try:
            depth = camera.get_depth()
            if depth is None or not np.any(np.isfinite(depth) & (np.asarray(depth) > 0)):
                raise RuntimeError(f'Missing or invalid depth: {key}')
            rgb = camera.get_rgb()
            frame = DepthFrame(key, token, _owned(depth, float),
                _owned(camera.get_intrinsics(), float),
                _owned(camera.get_world_from_camera_transform(), float),
                self._acquired_at, None if rgb is None else _owned(rgb))
            self._cache[key] = (token, frame, None)
            self.acquisitions += 1
            return frame
        except Exception as exc:
            self._cache[key] = (token, None, str(exc))
            raise

    def close(self):
        self._closed = True
        self._cache.clear()
        self._consumed.clear()
