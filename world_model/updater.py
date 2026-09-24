from __future__ import annotations


class WorldModelUpdater:
    def __init__(self, world_model, provider, update_hz: float | None = None):
        if update_hz is not None and update_hz <= 0:
            raise ValueError("update_hz must be > 0 or None")

        self.world_model = world_model
        self.provider = provider
        self.update_hz = update_hz
        self._period = None if update_hz is None else 1.0 / update_hz
        self._elapsed = 0.0
        self._callback_name = None

    def update(self):
        observations = self.provider.observe()
        self.world_model.apply_observations(
            observations,
            mark_missing_invisible=True,
        )
        return observations

    def tick(self, step_size: float):
        if self._period is None:
            return self.update()

        self._elapsed += float(step_size)

        if self._elapsed < self._period - 1e-9:
            return None

        self._elapsed = max(
            0.0,
            self._elapsed - self._period,
        )

        return self.update()

    def start(self, world, callback_name="world_model_updater"):
        if self._callback_name is not None:
            return

        self._callback_name = callback_name

        world.add_physics_callback(
            callback_name,
            callback_fn=self.tick,
        )

    def stop(self, world):
        if self._callback_name is None:
            return

        world.remove_physics_callback(
            self._callback_name
        )

        self._callback_name = None
        self._elapsed = 0.0