"""Bounded calibrated depth rays; independent of semantic perception and SDKs."""
from collections.abc import Callable, Mapping, Sequence
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from .occupancy import GridSpec, OccupancyMap


def _traverse_rays(origin: np.ndarray, endpoints: np.ndarray, spec: GridSpec,
                   max_visits: int) -> tuple[np.ndarray, int]:
    """Return flat cells and visit count for bounded batches of finite rays.

    Visits include duplicates and occupied endpoint cells. On budget exhaustion
    return exactly ``max_visits+1`` visits; the caller rejects the whole update.
    Ray parameters run from zero at the camera to one at the measured endpoint.
    Workspace is bounded by 256 rays and the remaining visit budget, rather than
    allocating a ray-by-grid array. Emission order across rays is immaterial to
    the deduplicated evidence; each ray retains the scalar DDA boundary rules.
    """
    # Clip segments to the grid, then advance all tied voxel-boundary crossings.
    origin, endpoints = np.asarray(origin, float), np.asarray(endpoints, float)
    lower, upper, shape = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max), np.asarray(spec.shape)
    steps = sum(spec.shape) + 3
    capacity = min(max_visits + 1, len(endpoints) * steps)
    output, used = np.empty(capacity, dtype=np.int64), 0
    tolerance = 32 * np.finfo(float).eps
    for begin in range(0, len(endpoints), 256):
        direction = endpoints[begin:begin + 256] - origin
        enter, leave = np.zeros(len(direction)), np.ones(len(direction))
        valid = np.ones(len(direction), bool)
        for axis in range(3):
            moving = direction[:, axis] != 0
            valid &= moving | ((lower[axis] <= origin[axis]) & (origin[axis] < upper[axis]))
            first, second = np.zeros(len(direction)), np.zeros(len(direction))
            np.divide(lower[axis] - origin[axis], direction[:, axis], out=first, where=moving)
            np.divide(upper[axis] - origin[axis], direction[:, axis], out=second, where=moving)
            enter = np.where(moving, np.maximum(enter, np.minimum(first, second)), enter)
            leave = np.where(moving, np.minimum(leave, np.maximum(first, second)), leave)
        valid &= leave > enter
        direction, enter, leave = direction[valid], enter[valid], leave[valid]
        if not len(direction):
            continue
        start = origin + enter[:, None] * direction
        probe = np.where(direction == 0, start,
                         np.nextafter(start, np.where(direction > 0, np.inf, -np.inf)))
        probe = np.maximum(lower, np.minimum(probe, np.nextafter(upper, lower)))
        cell = np.floor((probe - lower) / spec.voxel_size).astype(np.int64)
        step = np.sign(direction).astype(np.int64)
        crossing, interval = np.full(direction.shape, np.inf), np.full(direction.shape, np.inf)
        moving = direction != 0
        boundary = lower + (cell + (step > 0)) * spec.voxel_size
        np.divide(boundary - origin, direction, out=crossing, where=moving)
        np.divide(spec.voxel_size, abs(direction), out=interval, where=moving)
        current_t, active = enter.copy(), np.ones(len(direction), bool)
        for _ in range(steps):
            active &= (cell >= 0).all(1) & (cell < shape).all(1)
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            next_t = crossing[ids].min(1)
            emitted = ids[np.minimum(next_t, leave[ids]) > current_t[ids] + tolerance]
            if len(emitted):
                positions = cell[emitted]
                flat = (positions[:, 0] * shape[1] + positions[:, 1]) * shape[2] + positions[:, 2]
                take = min(len(flat), capacity - used)
                output[used:used + take] = flat[:take]
                used += take
                if used > max_visits:
                    return output[:used], used
            done = next_t >= leave[ids] - tolerance
            active[ids[done]] = False
            advance, next_t = ids[~done], next_t[~done]
            crossed = crossing[advance] <= next_t[:, None] + 1e-12
            cell[advance] += crossed * step[advance]
            values = crossing[advance]
            values[crossed] += interval[advance][crossed]
            crossing[advance], current_t[advance] = values, next_t
    return output[:used], used


class RgbdMapper:
    """Integrate complete calibrated batches supplied by acquisition.

    Masked and over-range depths contribute no evidence. Robot masks are applied
    after bounded pixel selection, without replacing excluded rays.
    """
    def __init__(self, occupancy_map: OccupancyMap, *, stride: int = 8, max_rays: int = 1024,
                 max_voxel_visits: int = 500_000, max_depth: float = 5.,
                 max_skew: float = .025, max_frame_age: float = 1.,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.map, self.stride, self.max_rays = occupancy_map, stride, max_rays
        self.max_voxel_visits = max_voxel_visits
        self.max_depth, self.max_skew, self.max_frame_age = max_depth, max_skew, max_frame_age
        self.clock, self._consumed = clock, {}

    def _failure(self, reason: str) -> bool:
        self.map.note_failure(reason)
        return False

    def _allocate(self, counts: Sequence[int]) -> list[int]:
        """Share rays equally, redistributing unused shares in camera-ID order."""
        allocation = [0]*len(counts)
        remaining = min(self.max_rays, sum(counts))
        while remaining:
            active = [i for i, count in enumerate(counts) if allocation[i] < count]
            share = max(1, remaining//len(active))
            for i in active:
                take = min(share, counts[i]-allocation[i], remaining)
                allocation[i] += take
                remaining -= take
        return allocation

    def update(self, frames: Sequence[DepthFrame] | None, *, now: float | None = None,
               mask_provider: Callable[[tuple[DepthFrame, ...], dict[str, tuple[np.ndarray, np.ndarray]]],
                                       Mapping[str, np.ndarray]] | None = None) -> bool:
        """Select pixels, mask robots, traverse rays and integrate one whole batch."""
        supplied_time = now is not None
        now = self.clock() if now is None else now
        self.map.error = None
        if not frames:
            return False
        frames = tuple(sorted(frames, key=lambda frame: frame.camera_id))
        tokens = {f.camera_id: f.token for f in frames}
        if tokens == self._consumed:
            return False
        if any(key in self._consumed and any(value <= before for value, before in zip(token, self._consumed[key]))
               for key, token in tokens.items()):
            return self._failure('frozen_or_regressed_frames')
        # An attempted exposure cannot be repaired and replayed as new input.
        self._consumed = tokens
        if np.ptp([f.token[1] for f in frames]) > self.max_skew:
            return self._failure('unsynchronized_frames')
        acquired = np.array([f.acquired_at for f in frames])
        if (not np.isfinite(acquired).all() or np.any(acquired > now) or
                np.any(now - acquired >= self.max_frame_age)):
            return self._failure('invalid_acquisition_time')
        pixels = {}
        for frame in frames:
            y, x = np.indices(frame.depth[::self.stride, ::self.stride].shape)
            y, x = y.ravel() * self.stride, x.ravel() * self.stride
            depth = frame.depth[y, x]
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                return self._failure('invalid_depth')
            keep = valid & (depth <= self.max_depth)
            pixels[frame.camera_id] = y[keep], x[keep]
        allocation = self._allocate([len(x) for y, x in pixels.values()])
        for key, count in zip(pixels, allocation):
            y, x = pixels[key]
            selected = ((np.arange(count) + .5) * len(x) / count).astype(int) if count else np.empty(0, int)
            pixels[key] = y[selected], x[selected]
        masks = mask_provider(frames, pixels) if mask_provider is not None and any(allocation) else None
        free_parts, occupied_parts = [], []
        shape, remaining = self.map.spec.shape, self.max_voxel_visits
        for frame in frames:
            y, x = pixels[frame.camera_id]
            if masks is not None:
                keep = ~masks[frame.camera_id][y, x]
                y, x = y[keep], x[keep]
            if not len(x):
                continue
            intrinsics, transform = frame.intrinsics, frame.world_from_camera
            rays = np.c_[(x + .5 - intrinsics[0, 2]) / intrinsics[0, 0],
                         (y + .5 - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(len(x))]
            endpoints = (rays * frame.depth[y, x, None]) @ transform[:3, :3].T + transform[:3, 3]
            hits = self.map.spec.world_to_indices(endpoints)
            hits = hits[(hits >= 0).all(1)]
            occupied_parts.append(np.unique(np.ravel_multi_index(hits.T, shape)))
            visited, visits = _traverse_rays(transform[:3, 3], endpoints, self.map.spec, remaining)
            remaining -= visits
            if remaining < 0:
                return self._failure('voxel_budget_exceeded')
            free_parts.append(np.unique(visited))
        occupied = np.unique(np.concatenate(occupied_parts)) if occupied_parts else np.empty(0, np.int64)
        free = np.unique(np.concatenate(free_parts)) if free_parts else np.empty(0, np.int64)
        free = np.setdiff1d(free, occupied, assume_unique=True)
        if not len(free) and not len(occupied):
            return False
        completed = now if supplied_time else self.clock()
        if completed < now:
            return self._failure('clock_regressed')
        if np.any(completed - acquired >= self.max_frame_age):
            return self._failure('stale_after_traversal')
        return bool(self.map.integrate(np.column_stack(np.unravel_index(free, shape)),
            np.column_stack(np.unravel_index(occupied, shape)), timestamp=float(acquired.min()), now=completed))
