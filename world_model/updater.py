from __future__ import annotations


class WorldModelUpdater:
    def __init__(self, world_model, provider):
        self.world_model = world_model
        self.provider = provider
        self._callback_name = None

    def update(self):
        observations = self.provider.observe()
        self.world_model.apply_observations(
            observations,
            mark_missing_invisible=True,
        )
        return observations

    def start(self, world, callback_name="world_model_updater"):
        if self._callback_name is not None:
            return

        self._callback_name = callback_name

        def on_physics_step(_step_size):
            self.update()

        world.add_physics_callback(
            callback_name,
            callback_fn=on_physics_step,
        )

    def stop(self, world):
        if self._callback_name is None:
            return

        world.remove_physics_callback(self._callback_name)
        self._callback_name = None