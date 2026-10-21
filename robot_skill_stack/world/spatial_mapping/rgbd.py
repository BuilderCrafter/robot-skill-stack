"""Bounded calibrated depth rays; independent of semantic perception and SDKs."""
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


def _ray_voxels(origin, endpoint, spec):
    """Yield cells with a positive-length intersection before the measured hit.

    Clip the actual finite segment, never an infinite camera ray. Simultaneous
    boundary crossings advance together, so corner/edge touches do not clear
    neighboring voxels that the ray never traversed.
    """
    origin, endpoint = np.asarray(origin, float), np.asarray(endpoint, float)
    direction = endpoint-origin
    lower, upper = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max)
    enter, leave = 0., 1.
    for axis in range(3):
        if direction[axis] == 0:
            if not lower[axis] <= origin[axis] < upper[axis]:
                return
            continue
        near, far = sorted(((lower[axis]-origin[axis])/direction[axis],
                            (upper[axis]-origin[axis])/direction[axis]))
        enter, leave = max(enter, near), min(leave, far)
    if leave <= enter:
        return
    start = origin+enter*direction
    # Select the voxel just inside the segment, including negative directions
    # beginning on a cell boundary and entries through the exclusive upper face.
    probe = np.where(direction == 0, start,
                     np.nextafter(start, np.where(direction > 0, np.inf, -np.inf)))
    probe = np.maximum(lower, np.minimum(probe, np.nextafter(upper, lower)))
    cell = np.floor((probe-lower)/spec.voxel_size).astype(int)
    step = np.sign(direction).astype(int)
    crossing, interval = np.full(3, np.inf), np.full(3, np.inf)
    moving = direction != 0
    boundary = lower+(cell+(step > 0))*spec.voxel_size
    crossing[moving] = (boundary[moving]-origin[moving])/direction[moving]
    interval[moving] = spec.voxel_size/abs(direction[moving])
    current = enter
    tolerance = 32*np.finfo(float).eps
    # A finite line can cross each grid plane at most once.
    for _ in range(sum(spec.shape)+3):
        if np.any(cell < 0) or np.any(cell >= spec.shape):
            return
        next_crossing = float(crossing.min())
        if min(next_crossing, leave) > current+tolerance:
            yield tuple(int(value) for value in cell)
        if next_crossing >= leave-tolerance:
            return
        crossed = crossing <= next_crossing+1e-12
        cell[crossed] += step[crossed]
        crossing[crossed] += interval[crossed]
        current = next_crossing


class RgbdMapper:
    """Integrate complete fresh snapshots from one or arbitrarily many cameras.

    ``exclusion_masks`` maps camera IDs to boolean depth-sized images. True
    skips the entire ray, including space in front of/behind a robot return.
    Automatic robot masking is the caller's responsibility. Measurements past
    ``max_depth`` are ignored, never converted into artificial free-space rays.
    """
    def __init__(self, occupancy_map, *, stride=8, max_rays=1024, max_voxel_visits=500_000, max_depth=5.,
                 max_skew=.025, max_frame_age=1., expected_camera_ids=None,
                 clock=time.monotonic):
        if type(stride) is not int or stride < 1 or type(max_rays) is not int or max_rays < 1:
            raise ValueError('stride and max_rays must be positive integers')
        if type(max_voxel_visits) is not int or max_voxel_visits < 1:
            raise ValueError('max_voxel_visits must be a positive integer')
        for value in (max_depth, max_skew, max_frame_age):
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError('Depth, skew and frame-age limits must be finite and positive')
        ids = None if expected_camera_ids is None else tuple(expected_camera_ids)
        if ids is not None and (not ids or any(not isinstance(key, str) or not key for key in ids)
                                or len(set(ids)) != len(ids)):
            raise ValueError('Expected camera IDs must be nonempty distinct strings')
        if not callable(clock):
            raise ValueError('clock must be callable')
        self.map, self.stride, self.max_rays = occupancy_map, stride, max_rays
        self.max_voxel_visits = max_voxel_visits
        self.max_depth, self.max_skew, self.max_frame_age = max_depth, max_skew, max_frame_age
        self.expected_camera_ids = None if ids is None else tuple(sorted(ids))
        self.clock, self._consumed, self._rotation = clock, {}, 0
        self.diagnostics = {'state': 'waiting'}

    def _failure(self, reason, now):
        self.diagnostics.update(state='error', reason=reason)
        self.map.note_failure(reason, now=now)
        return False

    def _no_frame(self, reason, now):
        self.diagnostics.update(state='no_frame', reason=reason)
        self.map.note_no_frame(reason, now=now)
        return False

    def _allocate(self, counts):
        """Equal shares, redistributing spare rays; rotate remainder fairly."""
        names = sorted(key for key, count in counts.items() if count)
        if names:
            offset = self._rotation % len(names)
            names = names[offset:]+names[:offset]
        allocation = dict.fromkeys(counts, 0)
        remaining = min(self.max_rays, sum(counts.values()))
        selected = remaining
        while remaining and names:
            share = remaining//len(names)
            if not share:
                for key in names[:remaining]:
                    allocation[key] += 1
                break
            for key in names:
                take = min(share, counts[key]-allocation[key])
                allocation[key] += take
                remaining -= take
            names = [key for key in names if allocation[key] < counts[key]]
        self._rotation += max(1, selected)
        return allocation

    def update(self, frames, exclusion_masks=None, *, now=None):
        supplied_time = now is not None
        now = float(self.clock() if now is None else now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('Current time must be finite and nonnegative')
        self.diagnostics = dict(state='validating', rays=0, cameras={}, voxel_visits=0,
                                voxel_visit_budget=self.max_voxel_visits)
        if frames is None:
            return self._no_frame('no_frames', now)
        try:
            frames = tuple(frames)
            if not frames:
                if self.expected_camera_ids:
                    return self._failure('Missing configured cameras', now)
                return self._no_frame('no_frames', now)
            if any(not isinstance(frame, DepthFrame) for frame in frames):
                return self._failure('Expected calibrated DepthFrame snapshots', now)
            ids = tuple(frame.camera_id for frame in frames)
            if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
                return self._failure('Camera IDs must be nonempty and distinct', now)
            if self.expected_camera_ids is None:
                self.expected_camera_ids = tuple(sorted(ids))
            if set(ids) != set(self.expected_camera_ids):
                return self._failure('Missing or unexpected cameras: expected '
                    f'{self.expected_camera_ids}, received {tuple(sorted(ids))}', now)
            tokens = {}
            for frame in frames:
                try:
                    token = tuple(float(value) for value in frame.token)
                    valid = len(token) == 2 and np.isfinite(token).all() and token[0] > 0 and token[1] >= 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    return self._failure(f'Invalid frame token: {frame.camera_id}', now)
                tokens[frame.camera_id] = token
            repeated = [key for key in ids if tokens[key] == self._consumed.get(key)]
            if repeated:
                if len(repeated) == len(ids):
                    return self._no_frame('repeated_frames', now)
                return self._failure(f'Frozen cameras: {tuple(sorted(repeated))}', now)
            regressed = [key for key in ids if key in self._consumed and any(
                value <= before for value, before in zip(tokens[key], self._consumed[key]))]
            if regressed:
                return self._failure(f'Regressed camera clocks: {tuple(sorted(regressed))}', now)
            # Consume complete attempted snapshots before buffer/age validation.
            # Repairing a bad buffer cannot relabel the same tokens as fresh.
            self._consumed = tokens
            if np.ptp([token[1] for token in tokens.values()]) > self.max_skew:
                return self._failure('Camera synchronization skew exceeds limit', now)
            acquired = np.array([frame.acquired_at for frame in frames], float)
            if (not np.isfinite(acquired).all() or np.any(acquired > now) or
                    np.any(now-acquired >= self.max_frame_age)):
                return self._failure('Stale, future or invalid acquisition timestamp', now)
            masks = {} if exclusion_masks is None else dict(exclusion_masks)
            if set(masks)-set(ids):
                return self._failure('Exclusion mask has an unknown camera ID', now)
            candidates, by_id = {}, {frame.camera_id: frame for frame in frames}
            for key in sorted(ids):
                frame = by_id[key]
                # Revalidate mutable calibration arrays at the mapper boundary.
                DepthFrame(key, frame.token, frame.depth, frame.intrinsics,
                           frame.world_from_camera, frame.acquired_at)
                valid_depth = np.isfinite(frame.depth) & (frame.depth > 0)
                if not valid_depth.any():
                    return self._failure(f'All depth pixels invalid: {key}', now)
                mask = masks.get(key)
                if mask is not None and (np.asarray(mask).dtype != np.bool_ or np.shape(mask) != frame.depth.shape):
                    return self._failure(f'Exclusion mask must match boolean depth image: {key}', now)
                y, x = np.indices(frame.depth[::self.stride, ::self.stride].shape)
                y, x = y.ravel()*self.stride, x.ravel()*self.stride
                depth = frame.depth[y, x]
                keep = valid_depth[y, x] & (depth <= self.max_depth)
                masked = np.zeros(len(x), bool) if mask is None else np.asarray(mask)[y, x]
                keep &= ~masked
                candidates[key] = (y[keep], x[keep], depth[keep])
                self.diagnostics['cameras'][key] = dict(sampled_pixels=len(x),
                    masked_pixels=int(masked.sum()), candidate_rays=int(keep.sum()), rays=0,
                    free_voxels=0, occupied_voxels=0)
            allocation = self._allocate({key: len(value[0]) for key, value in candidates.items()})
            self.diagnostics['rays'] = sum(allocation.values())
            free, occupied = set(), set()
            for key in sorted(ids):
                count = allocation[key]
                if not count:
                    continue
                y, x, depth = candidates[key]
                selected = ((np.arange(count)+.5)*len(x)/count).astype(int)
                x, y, depth = x[selected], y[selected], depth[selected]
                frame = by_id[key]
                k, transform = frame.intrinsics, frame.world_from_camera
                rays = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(count)]
                endpoints = (rays*depth[:, None]) @ transform[:3, :3].T+transform[:3, 3]
                hit_indices = self.map.world_to_indices(endpoints)
                view_occupied = {tuple(int(v) for v in index) for index in hit_indices if np.all(index >= 0)}
                view_free = set()
                for endpoint in endpoints:
                    for index in _ray_voxels(transform[:3, 3], endpoint, self.map.spec):
                        self.diagnostics['voxel_visits'] += 1
                        if self.diagnostics['voxel_visits'] > self.max_voxel_visits:
                            return self._failure('Voxel traversal budget exceeded', now)
                        view_free.add(index)
                free.update(view_free)
                occupied.update(view_occupied)
                self.diagnostics['cameras'][key].update(rays=count,
                    free_voxels=len(view_free-view_occupied), occupied_voxels=len(view_occupied))
            free.difference_update(occupied)
            self.diagnostics.update(rays=sum(allocation.values()), free_voxels=len(free), occupied_voxels=len(occupied))
            if not free and not occupied:
                return self._no_frame('no_intersecting_evidence', now)
            indices = lambda values: np.asarray(sorted(values), dtype=int).reshape(-1, 3)
            free_indices, occupied_indices = indices(free), indices(occupied)
            # Production clocks include traversal time. Explicit ``now`` is a
            # deterministic caller-supplied evaluation time (e.g. offline tests).
            completed = now if supplied_time else float(self.clock())
            if not np.isfinite(completed) or completed < now:
                return self._failure('Clock regressed during RGB-D integration', now)
            if np.any(completed-acquired >= self.max_frame_age):
                return self._failure('RGB-D snapshot became stale during integration', completed)
            changed = self.map.integrate(free_indices, occupied_indices,
                                         timestamp=float(acquired.min()), now=completed)
            self.diagnostics['state'] = 'updated' if changed else 'no_frame'
            return bool(changed)
        except (ValueError, TypeError, AttributeError, IndexError) as exc:
            return self._failure(f'Invalid RGB-D snapshot: {exc}', now)
