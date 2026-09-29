from __future__ import annotations

from robot_skill_stack.world.model.provider import WorldObservationProvider
from robot_skill_stack.world.model.world_model import WorldModel


class WorldModelUpdater:
    def __init__(
        self,
        world_model: WorldModel,
        provider: WorldObservationProvider,
        update_hz: float | None = None,
    ):
        if update_hz is not None and update_hz <= 0:
            raise ValueError("update_hz must be > 0 or None")

        self.world_model = world_model
        self.provider = provider
        self.update_hz = update_hz
        self._period = None if update_hz is None else 1.0 / update_hz
        self._elapsed = 0.0
        self._callback_name = None
        self._scheduler = None
        self._closed = False
        self.update_count = 0
        self.last_error = None

    def update(self):
        if self._closed:
            return []
        context = self.world_model.observation_context()
        observations = self.provider.observe(context=context)
        self.world_model.apply_observations(
            observations,
            mark_missing_invisible=True,
        )
        self.cleanup()
        self.update_count += 1
        self.last_error = None
        return observations

    def cleanup(self, now=None):
        if self._closed:
            return ()
        age = getattr(self.provider, "age", None)
        if age is not None:
            for object_id in age(now=now, context=self.world_model.observation_context()):
                obj = self.world_model.get(object_id)
                if obj is not None:
                    obj.visible = False
        return self.world_model.expire_stale(now=now, forget=getattr(self.provider, "forget", None))

    def clear_lost(self):
        forget = getattr(self.provider, "forget", None)
        if forget is None:
            return ()
        self.cleanup()
        return self.world_model.clear_lost(forget)

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
            self.last_error = f'{type(exc).__name__}: {exc}'

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
        try:
            if name is not None and scheduler is not None:
                exists = getattr(scheduler, 'physics_callback_exists', None)
                if exists is None or exists(name):
                    scheduler.remove_physics_callback(name)
        finally:
            close = getattr(self.provider, "close", None)
            if close is not None:
                close()
