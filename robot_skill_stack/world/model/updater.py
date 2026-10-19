from __future__ import annotations

import time
import math

from robot_skill_stack.world.model.provider import WorldObservationProvider, ObservationHealth
from robot_skill_stack.world.model.world_model import WorldModel


class WorldModelUpdater:
    def __init__(
        self,
        world_model: WorldModel,
        provider: WorldObservationProvider,
        update_hz: float | None = None,
        *, stale_object_ttl=15.0, stale_after=1.0,
    ):
        if update_hz is not None and update_hz <= 0:
            raise ValueError("update_hz must be > 0 or None")

        if not all(math.isfinite(v) and v > 0 for v in (stale_object_ttl, stale_after)):
            raise ValueError("freshness and object TTL must be finite and > 0")
        self.stale_object_ttl = float(stale_object_ttl)
        self.health = world_model.source_health = ObservationHealth(stale_after=stale_after)
        self.world_model = world_model
        self.provider = provider
        self.update_hz = update_hz
        self._period = None if update_hz is None else 1.0 / update_hz
        self._elapsed = 0.0
        self._callback_name = None
        self._scheduler = None
        self._closed = False
        self.update_count = 0

    @property
    def last_error(self):
        return self.health.error

    def update(self):
        if self._closed:
            return []
        # Expire before association: an overdue identity cannot be resurrected
        # by a new frame unless a held attachment or active placement hint protects it.
        self.cleanup()
        try:
            batch = self.provider.observe(context=self.world_model.observation_context())
            timestamp = batch.timestamp
            if timestamp is None:
                return []
            now = time.monotonic()
            if not math.isfinite(timestamp) or timestamp > now:
                raise ValueError('Observation timestamp must use the local monotonic clock')
            if self.health.last_frame_at is not None and timestamp <= self.health.last_frame_at:
                return []
            if now-timestamp >= self.health.stale_after:
                return []
            for observation in batch.observations:
                observation.timestamp = timestamp
            self.world_model.apply_observations(batch.observations, mark_missing_invisible=True)
            self.health.last_frame_at = timestamp
            self.health.error = None
            self.update_count += 1
            return batch.observations
        except Exception as exc:
            self.health.error = f'{type(exc).__name__}: {exc}'
            # Source failure invalidates visibility immediately, retaining IDs.
            for obj in self.world_model.objects():
                obj.visible = False
            return []
        finally:
            self.cleanup()

    def cleanup(self, now=None):
        if self._closed:
            return ()
        now = time.monotonic() if now is None else float(now)
        model = self.world_model
        model.expire_hints(now)
        removed = []
        for obj in model.objects():
            age = math.inf if obj.last_seen is None else now-obj.last_seen
            if age >= self.health.stale_after:
                obj.visible = False
            if not obj.visible and age >= self.stale_object_ttl and model.remove(obj.object_id):
                self.provider.forget(obj.object_id)
                removed.append(obj.object_id)
        return tuple(removed)

    def clear_lost(self):
        if self._closed:
            return ()
        removed = list(self.cleanup())
        for obj in self.world_model.objects():
            if not obj.visible and self.world_model.remove(obj.object_id):
                self.provider.forget(obj.object_id)
                removed.append(obj.object_id)
        return tuple(removed)

    def tick(self, step_size: float):
        if self._closed:
            return None
        if self._period is None:
            return self.update()

        self._elapsed += float(step_size)
        if self._elapsed < self._period - 1e-9:
            self.cleanup()
            return None

        self._elapsed = max(0.0, self._elapsed - self._period)
        return self.update()

    def _on_physics(self, step_size):
        if self._closed or self._callback_name is None:
            return
        try:
            self.tick(step_size)
        except Exception as exc:
            self.health.error = f'{type(exc).__name__}: {exc}'

    def start(self, world, callback_name="world_model_updater"):
        if self._closed:
            raise RuntimeError('A closed updater cannot be restarted; construct a new runtime')
        if self._callback_name is not None:
            return
        exists = getattr(world, 'physics_callback_exists', None)
        if exists is not None and exists(callback_name):
            raise RuntimeError(f'Physics callback {callback_name!r} already exists')
        world.add_physics_callback(callback_name, callback_fn=self._on_physics)
        self._scheduler, self._callback_name = world, callback_name

    def stop(self, world=None):
        if self._closed:
            return
        scheduler = self._scheduler if self._scheduler is not None else world
        name, self._callback_name = self._callback_name, None
        self._scheduler, self._closed, self._elapsed = None, True, 0.0
        self.health.error = "Observation source closed"
        try:
            if name is not None and scheduler is not None:
                exists = getattr(scheduler, 'physics_callback_exists', None)
                if exists is None or exists(name):
                    scheduler.remove_physics_callback(name)
        finally:
            self.provider.close()
