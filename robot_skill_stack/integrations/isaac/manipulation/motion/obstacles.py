"""Bounded numeric cuboids for the installed Isaac 5.1 public Lula interface.

``RmpFlow.add_cuboid`` consumes three numeric getters; unlike ``add_obstacle``
it does not require a USD wrapper. These planning objects never create a prim,
render product, collider, material or semantic annotation. All SDK calls occur
on their creating thread. The normal MotionPolicyController.forward refreshes
the Lula world each step using these cheap numeric getters.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


class _NumericCuboid:
    def __init__(self, registration_id, center, dimensions, meters_per_unit):
        self.registration_id = registration_id
        self.center = np.array(center, dtype=float, copy=True)
        self.dimensions = np.array(dimensions, dtype=float, copy=True)
        self._meters_per_unit = meters_per_unit

    def get_size(self):
        return 1.0

    def get_local_scale(self):
        return self.dimensions / self._meters_per_unit

    def get_world_pose(self):
        # Lula's installed utility scales the returned position in place.
        # Returning a new array prevents repeated world updates mutating us.
        return self.center / self._meters_per_unit, np.array([1., 0., 0., 0.])


def _equal(first, second):
    return bool(np.allclose(first, second, rtol=0., atol=1e-12))


class RmpflowObstacleRegistry:
    """Own only this strategy's registrations in one backend-owned policy.

    A box's voxel-derived ID can change as it moves across grid boundaries.
    Deterministic overlap matching reuses a compatible numeric object; matching
    changes identity only, never the desired geometry or obstacle count.

    Call ``reset_tracking`` immediately AFTER the backend resets RMPflow,
    because the SDK reset has already erased every registration. ``clear`` and
    ``close`` remove our registrations without resetting the shared controller.
    """
    def __init__(self, policy, *, max_obstacles=64, meters_per_unit=None):
        if type(max_obstacles) is not int or not 1 <= max_obstacles <= 128:
            raise ValueError('max_obstacles must be an integer from 1 through 128')
        if meters_per_unit is None:
            from isaacsim.core.utils.stage import get_stage_units
            meters_per_unit = get_stage_units()
        if (type(meters_per_unit) not in (int, float) or
                not math.isfinite(meters_per_unit) or meters_per_unit <= 0):
            raise ValueError('Stage meters_per_unit must be finite and positive')
        self.policy, self.max_obstacles = policy, max_obstacles
        self._meters_per_unit = float(meters_per_unit)
        self._owner = threading.get_ident()
        self._proxies = {}
        self._generation = self._revision = self._timestamp = None
        self._signature = None
        self._packet = None
        self._next_expiry = math.inf
        self._next_registration = 0
        self._closed = False
        self._counts = dict(additions=0, removals=0, pose_updates=0,
            dimension_updates=0, identity_reuses=0, world_updates=0, resets=0, skipped=0)
        self._last_apply_seconds = 0.

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('RMPflow obstacles must be accessed on their Kit owning thread')

    @property
    def diagnostics(self):
        self._assert_owner()
        return dict(self._counts, registered=len(self._proxies), proxy_count=len(self._proxies),
            added=self._counts['additions'], removed=self._counts['removals'],
            updated=self._counts['pose_updates']+self._counts['dimension_updates'],
            last_update_seconds=self._last_apply_seconds, closed=self._closed,
            generation=self._generation, revision=self._revision,
            timestamp=self._timestamp, apply_seconds=self._last_apply_seconds,
            registrations={key: proxy.registration_id for key, proxy in sorted(self._proxies.items())})

    def reset_tracking(self):
        self._assert_owner()
        self._proxies.clear()
        self._generation = self._revision = self._timestamp = self._signature = None
        self._packet = None
        self._next_expiry = math.inf
        self._counts['resets'] += 1

    def _remove(self, proxy):
        # False means the policy already removed/reset this registration.
        self.policy.remove_obstacle(proxy)
        self._counts['removals'] += 1

    def clear(self):
        self._assert_owner()
        self._packet = None
        errors = []
        removed = False
        for key, proxy in tuple(self._proxies.items()):
            try:
                self._remove(proxy)
                del self._proxies[key]
                removed = True
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('Failed removing owned RMPflow obstacles: '+
                               '; '.join(map(str, errors))) from errors[0]
        self._generation = self._revision = self._timestamp = self._signature = None
        self._next_expiry = math.inf
        if removed:
            self.policy.update_world([])
            self._counts['world_updates'] += 1

    def close(self):
        self._assert_owner()
        self._closed = True
        self.clear()

    def _register(self, proxy):
        try:
            if not self.policy.add_cuboid(proxy, static=False):
                raise RuntimeError('RMPflow rejected numeric cuboid registration')
        except Exception as exc:
            # A wrapper may throw after its SDK registration took effect.
            try:
                self.policy.remove_obstacle(proxy)
            except Exception as cleanup:
                exc.add_note(f'RMPflow registration cleanup also failed: {cleanup}')
            raise
        self._counts['additions'] += 1

    def sync(self, packet, *, now=None):
        """Apply prepared boxes and their original expiry; never inspect voxels.

        Repeated revisions produce no SDK calls. Expiry is still checked for the
        bounded descriptors, so old voxel evidence cannot survive a quiet map.
        All input validation precedes SDK mutation.
        """
        self._assert_owner()
        if self._closed:
            raise RuntimeError('RMPflow obstacle registry is closed')
        started = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        generation, revision, timestamp = packet.generation, packet.revision, float(packet.timestamp)
        if (type(generation) is not int or generation < 0 or
                type(revision) is not int or revision < 0 or
                not math.isfinite(timestamp) or timestamp < 0 or
                not math.isfinite(now) or timestamp > now):
            raise ValueError('Obstacle packet generation, revision and source time must be valid')
        if (packet is self._packet and generation == self._generation and
                revision == self._revision and timestamp == self._timestamp and
                now < self._next_expiry):
            self._last_apply_seconds = time.perf_counter()-started
            self._counts['skipped'] += 1
            return False
        if len(packet.boxes) > self.max_obstacles:
            raise ValueError('Obstacle packet exceeds configured proxy count')
        raw, desired = [], {}
        for box in packet.boxes:
            key = box.id
            center, dimensions = np.asarray(box.center, dtype=float), np.asarray(box.dimensions, dtype=float)
            expires = float(box.expires_at)
            if (not isinstance(key, str) or not key or
                    center.shape != (3,) or dimensions.shape != (3,) or
                    not np.isfinite(center).all() or not np.isfinite(dimensions).all() or
                    np.any(dimensions <= 0) or math.isnan(expires) or expires < 0):
                raise ValueError('Obstacle boxes require unique IDs, finite centers and positive dimensions')
            raw.append((key, tuple(center), tuple(dimensions), expires))
            if expires > now:
                if key in desired:
                    raise ValueError('Obstacle box IDs must be unique')
                desired[key] = (center, dimensions)
        signature = tuple(sorted(raw))
        if len({row[0] for row in raw}) != len(raw):
            raise ValueError('Obstacle box IDs must be unique')
        generation_cleared = False
        if self._generation is not None:
            if generation < self._generation:
                raise ValueError('Obsolete obstacle generation')
            if generation == self._generation:
                if revision < self._revision or timestamp < self._timestamp:
                    raise ValueError('Obsolete obstacle revision or timestamp')
                if revision == self._revision and (signature != self._signature or timestamp != self._timestamp):
                    raise ValueError('Conflicting geometry or source time for the same obstacle revision')
            else:
                generation_cleared = bool(self._proxies)
                self.clear()

        self._packet = None
        before = self._proxies
        following = {}
        changed = generation_cleared
        # Exact source IDs reserve their own registrations before overlap reuse.
        for key in sorted(set(before) & set(desired)):
            following[key] = before[key]
        available = {key: proxy for key, proxy in before.items() if key not in following}
        unmatched = sorted(set(desired)-set(following))
        candidates = []
        if unmatched and available:
            old_keys = sorted(available)
            centers = np.array([desired[key][0] for key in unmatched])
            sizes = np.array([desired[key][1] for key in unmatched])
            old_centers = np.array([available[key].center for key in old_keys])
            old_sizes = np.array([available[key].dimensions for key in old_keys])
            displacement = np.abs(centers[:, None, :]-old_centers[None, :, :])
            overlaps = np.all(displacement < (sizes[:, None, :]+old_sizes[None, :, :])/2+1e-12, axis=2)
            resized = np.any(np.abs(sizes[:, None, :]-old_sizes[None, :, :]) > 1e-12, axis=2)
            distance = np.linalg.norm(displacement, axis=2)
            for new_index, old_index in np.argwhere(overlaps):
                candidates.append((bool(resized[new_index, old_index]), float(distance[new_index, old_index]),
                                   unmatched[new_index], old_keys[old_index]))
        for _, _, key, old_key in sorted(candidates):
            if key in following or old_key not in available:
                continue
            following[key] = available.pop(old_key)
            self._counts['identity_reuses'] += 1
        for proxy in available.values():
            self._remove(proxy)
            changed = True
        # Preserve references to successfully registered objects if a later SDK
        # call fails; clear()/close() can still remove every owned registration.
        self._proxies = following
        for key in sorted(desired):
            center, dimensions = desired[key]
            proxy = following.get(key)
            if proxy is None:
                proxy = _NumericCuboid(self._next_registration, center, dimensions, self._meters_per_unit)
                self._next_registration += 1
                following[key] = proxy
                self._register(proxy)
                changed = True
            elif not _equal(proxy.dimensions, dimensions):
                self._remove(proxy)
                proxy.center, proxy.dimensions = center.copy(), dimensions.copy()
                self._register(proxy)
                self._counts['dimension_updates'] += 1
                changed = True
            elif not _equal(proxy.center, center):
                proxy.center = center.copy()
                self._counts['pose_updates'] += 1
                changed = True
        if changed:
            # New poses/registrations are visible immediately, even before a
            # subsequent controller forward. forward() also refreshes each step.
            self.policy.update_world()
            self._counts['world_updates'] += 1
        self._generation, self._revision, self._timestamp = generation, revision, timestamp
        self._signature = signature
        self._packet = packet
        self._next_expiry = min((row[3] for row in raw if row[3] > now), default=math.inf)
        self._last_apply_seconds = time.perf_counter()-started
        self._counts['skipped'] += int(not changed)
        return changed
