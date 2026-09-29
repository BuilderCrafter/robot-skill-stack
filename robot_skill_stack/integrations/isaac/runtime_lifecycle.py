"""Ownership and the Isaac 5.1 Stop/Play callback-retention compatibility boundary."""
import builtins


class PhysicsCallbackScope:
    def __init__(self, world):
        self.world, self.owned = world, {}

    def _retained(self):
        # 5.1 Stop clears subscriptions, but retains callables for PHYSICS_READY.
        value = getattr(self.world, '_physics_functions', None)
        return value if isinstance(value, dict) else {}

    def physics_callback_exists(self, name):
        return self.world.physics_callback_exists(name) or name in self._retained()

    def add_physics_callback(self, name, callback_fn):
        if self.physics_callback_exists(name):
            raise RuntimeError(f'Physics callback {name!r} is still owned by another runtime; restart Isaac')
        self.world.add_physics_callback(name, callback_fn=callback_fn)
        if not self.world.physics_callback_exists(name):
            raise RuntimeError(f'Physics callback {name!r} was not registered')
        self.owned[name] = callback_fn

    def remove_physics_callback(self, name):
        fn = self.owned.pop(name, None)
        if fn is None:
            return
        retained = self._retained()
        if name in retained and retained[name] is not fn:
            raise RuntimeError(f'Refusing to remove foreign callback {name!r}')
        if self.world.physics_callback_exists(name):
            self.world.remove_physics_callback(name)
        # Only our exact callable; never clear_all_callbacks or foreign subscriptions.
        if retained.get(name) is fn:
            del retained[name]


def construct_world_deferred(world_cls, **kwargs):
    """Select 5.1's deferred constructor; restore the SDK flag before any await."""
    marker = object()
    previous = getattr(builtins, 'ISAAC_LAUNCHED_FROM_TERMINAL', marker)
    try:
        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
        return world_cls(**kwargs)
    finally:
        if previous is marker:
            del builtins.ISAAC_LAUNCHED_FROM_TERMINAL
        else:
            builtins.ISAAC_LAUNCHED_FROM_TERMINAL = previous
