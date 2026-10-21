"""Bounded calibrated depth rays; independent of semantic perception and SDKs."""
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


def _traverse_rays(origin, endpoints, spec, max_visits):
    """Return flat cells and visit count for bounded batches of finite rays.

    Visits include duplicates and occupied endpoint cells. On budget exhaustion
    return exactly ``max_visits+1`` visits; the caller rejects the whole update.
    Workspace is bounded by 256 rays and the remaining visit budget, rather than
    allocating a ray-by-grid array. Emission order across rays is immaterial to
    the deduplicated evidence; each ray retains the scalar DDA boundary rules.
    """
    origin, endpoints = np.asarray(origin, float), np.asarray(endpoints, float)
    if (origin.shape != (3,) or endpoints.ndim != 2 or endpoints.shape[1] != 3 or
            not np.isfinite(origin).all() or not np.isfinite(endpoints).all()):
        raise ValueError('Traversal requires finite XYZ origin and endpoints')
    lower, upper, shape = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max), np.asarray(spec.shape)
    steps = sum(spec.shape)+3
    capacity = min(max_visits+1, len(endpoints)*steps)
    output, used = np.empty(capacity, dtype=np.int64), 0
    tolerance = 32*np.finfo(float).eps
    for begin in range(0, len(endpoints), 256):
        direction = endpoints[begin:begin+256]-origin
        enter, leave = np.zeros(len(direction)), np.ones(len(direction))
        valid = np.ones(len(direction), bool)
        for axis in range(3):
            moving = direction[:, axis] != 0
            valid &= moving | ((lower[axis] <= origin[axis]) & (origin[axis] < upper[axis]))
            first, second = np.zeros(len(direction)), np.zeros(len(direction))
            np.divide(lower[axis]-origin[axis], direction[:, axis], out=first, where=moving)
            np.divide(upper[axis]-origin[axis], direction[:, axis], out=second, where=moving)
            enter = np.where(moving, np.maximum(enter, np.minimum(first, second)), enter)
            leave = np.where(moving, np.minimum(leave, np.maximum(first, second)), leave)
        valid &= leave > enter
        direction, enter, leave = direction[valid], enter[valid], leave[valid]
        if not len(direction):
            continue
        start = origin+enter[:, None]*direction
        probe = np.where(direction == 0, start,
                         np.nextafter(start, np.where(direction > 0, np.inf, -np.inf)))
        probe = np.maximum(lower, np.minimum(probe, np.nextafter(upper, lower)))
        cell = np.floor((probe-lower)/spec.voxel_size).astype(np.int64)
        step = np.sign(direction).astype(np.int64)
        crossing, interval = np.full(direction.shape, np.inf), np.full(direction.shape, np.inf)
        moving = direction != 0
        boundary = lower+(cell+(step > 0))*spec.voxel_size
        np.divide(boundary-origin, direction, out=crossing, where=moving)
        np.divide(spec.voxel_size, abs(direction), out=interval, where=moving)
        current, active = enter.copy(), np.ones(len(direction), bool)
        for _ in range(steps):
            active &= (cell >= 0).all(1) & (cell < shape).all(1)
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            following = crossing[ids].min(1)
            emitted = ids[np.minimum(following, leave[ids]) > current[ids]+tolerance]
            if len(emitted):
                positions = cell[emitted]
                flat = (positions[:, 0]*shape[1]+positions[:, 1])*shape[2]+positions[:, 2]
                take = min(len(flat), capacity-used)
                output[used:used+take] = flat[:take]
                used += take
                if used > max_visits:
                    return output[:used], used
            done = following >= leave[ids]-tolerance
            active[ids[done]] = False
            advance, following = ids[~done], following[~done]
            crossed = crossing[advance] <= following[:, None]+1e-12
            cell[advance] += crossed*step[advance]
            values = crossing[advance]
            values[crossed] += interval[advance][crossed]
            crossing[advance], current[advance] = values, following
    return output[:used], used


def _ray_voxels(origin, endpoint, spec):
    """Yield one finite segment's cells, preserving the scalar helper contract."""
    flat, _ = _traverse_rays(origin, np.asarray(endpoint, float).reshape(1, 3),
                             spec, sum(spec.shape)+3)
    for cell in np.column_stack(np.unravel_index(flat, spec.shape)):
        yield tuple(int(value) for value in cell)


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

    def update(self, frames, exclusion_masks=None, *, now=None, mask_provider=None):
        """Integrate one complete batch; an optional provider supplies its masks.

        After frame validation, token consumption and fair ray selection, the
        provider receives ``(frames, {camera_id: (y_pixels, x_pixels)})``. It may
        exclude selected rays, which are not backfilled with other candidates.
        Failed work cannot be repaired and replayed using the same tokens.
        """
        supplied_time = now is not None
        now = float(self.clock() if now is None else now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('Current time must be finite and nonnegative')
        self.diagnostics = dict(state='validating', rays=0, allocated_rays=0, retained_rays=0,
                                automatic_masked_selected=0, cameras={}, voxel_visits=0,
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
            by_id, valid_depths = {frame.camera_id: frame for frame in frames}, {}
            for key in sorted(ids):
                frame = by_id[key]
                # Revalidate mutable calibration arrays before requesting masks.
                DepthFrame(key, frame.token, frame.depth, frame.intrinsics,
                           frame.world_from_camera, frame.acquired_at)
                valid_depths[key] = np.isfinite(frame.depth) & (frame.depth > 0)
                if not valid_depths[key].any():
                    return self._failure(f'All depth pixels invalid: {key}', now)
            masks = {} if exclusion_masks is None else dict(exclusion_masks)
            if set(masks)-set(ids):
                return self._failure('Exclusion mask has an unknown camera ID', now)
            candidates = {}
            for key in sorted(ids):
                frame = by_id[key]
                valid_depth = valid_depths[key]
                mask = masks.get(key)
                if ((mask is not None and (np.asarray(mask).dtype != np.bool_ or
                        np.shape(mask) != frame.depth.shape)) or
                        (mask_provider is not None and key in masks and mask is None)):
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
                    allocated_rays=0, retained_rays=0, automatic_masked_selected=0,
                    free_voxels=0, occupied_voxels=0)
            allocation = self._allocate({key: len(value[0]) for key, value in candidates.items()})
            self.diagnostics['allocated_rays'] = sum(allocation.values())
            selected_rays, pixels = {}, {}
            for key in sorted(ids):
                y, x, depth = candidates[key]
                count = allocation[key]
                selected = ((np.arange(count)+.5)*len(x)/count).astype(int) if count else np.empty(0, int)
                selected_rays[key] = (y[selected], x[selected], depth[selected])
                pixels[key] = selected_rays[key][:2]
                self.diagnostics['cameras'][key]['allocated_rays'] = count
            if mask_provider is not None and self.diagnostics['allocated_rays']:
                try:
                    automatic = dict(mask_provider(frames, pixels))
                except Exception as exc:
                    return self._failure(f'Invalid RGB-D snapshot: mask provider failed: {exc}', now)
                if set(automatic) != set(ids):
                    return self._failure('Automatic mask camera IDs must match every depth frame', now)
                for key in sorted(ids):
                    mask = np.asarray(automatic[key])
                    if mask.dtype != np.bool_ or mask.shape != by_id[key].depth.shape:
                        return self._failure(f'Automatic mask must match boolean depth image: {key}', now)
                    y, x, depth = selected_rays[key]
                    keep = ~mask[y, x]
                    removed = int((~keep).sum())
                    self.diagnostics['cameras'][key]['automatic_masked_selected'] = removed
                    self.diagnostics['automatic_masked_selected'] += removed
                    selected_rays[key] = (y[keep], x[keep], depth[keep])
            retained = sum(len(value[0]) for value in selected_rays.values())
            self.diagnostics.update(rays=retained, retained_rays=retained)
            free_parts, occupied_parts = [], []
            shape = self.map.spec.shape
            for key in sorted(ids):
                y, x, depth = selected_rays[key]
                count = len(x)
                self.diagnostics['cameras'][key]['retained_rays'] = count
                if not count:
                    continue
                frame = by_id[key]
                k, transform = frame.intrinsics, frame.world_from_camera
                rays = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(count)]
                endpoints = (rays*depth[:, None]) @ transform[:3, :3].T+transform[:3, 3]
                hit_indices = self.map.world_to_indices(endpoints)
                valid_hits = hit_indices[(hit_indices >= 0).all(1)]
                view_occupied = np.unique(np.ravel_multi_index(valid_hits.T, shape))
                remaining = self.max_voxel_visits-self.diagnostics['voxel_visits']
                visited, count_visits = _traverse_rays(transform[:3, 3], endpoints, self.map.spec, remaining)
                self.diagnostics['voxel_visits'] += count_visits
                if count_visits > remaining:
                    return self._failure('Voxel traversal budget exceeded', now)
                view_free = np.unique(visited)
                free_parts.append(view_free)
                occupied_parts.append(view_occupied)
                self.diagnostics['cameras'][key].update(rays=count,
                    free_voxels=len(np.setdiff1d(view_free, view_occupied, assume_unique=True)),
                    occupied_voxels=len(view_occupied))
            occupied = np.unique(np.concatenate(occupied_parts)) if occupied_parts else np.empty(0, np.int64)
            free = np.unique(np.concatenate(free_parts)) if free_parts else np.empty(0, np.int64)
            free = np.setdiff1d(free, occupied, assume_unique=True)
            self.diagnostics.update(free_voxels=len(free), occupied_voxels=len(occupied))
            if not len(free) and not len(occupied):
                return self._no_frame('no_intersecting_evidence', now)
            indices = lambda values: np.column_stack(np.unravel_index(values, shape))
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
