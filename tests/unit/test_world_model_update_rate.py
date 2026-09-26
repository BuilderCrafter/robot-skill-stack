from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel


class CountingProvider:
    def __init__(self):
        self.calls = 0
        self.contexts = []

    def observe(self, context=None):
        self.calls += 1
        self.contexts.append(context)
        return [
            ObjectObservation(
                object_id="cube",
                class_name="cube",
                pose=Pose([self.calls, 0, 0]),
                source="test",
            )
        ]


print("[1] Testing every-step updater...")

provider = CountingProvider()
model = WorldModel()
updater = WorldModelUpdater(model, provider)

for _ in range(120):
    updater.tick(1 / 60)

assert provider.calls == 120
assert all(context is not None for context in provider.contexts)
print("Calls after 120 ticks:", provider.calls)


print("\n[2] Testing 5 Hz updater...")

provider = CountingProvider()
model = WorldModel()
updater = WorldModelUpdater(model, provider, update_hz=5)

for _ in range(120):
    updater.tick(1 / 60)

print("Calls after 2 simulated seconds:", provider.calls)
assert provider.calls == 10
assert model.require("cube").pose.position[0] == 10


print("\n[3] Testing 10 Hz updater...")

provider = CountingProvider()
model = WorldModel()
updater = WorldModelUpdater(model, provider, update_hz=10)

for _ in range(120):
    updater.tick(1 / 60)

print("Calls after 2 simulated seconds:", provider.calls)
assert provider.calls == 20


print("\n=== WORLD MODEL UPDATE RATE ===")
print("Every step: PASS")
print("5 Hz:       PASS")
print("10 Hz:      PASS")
print("Context:    PASS")
print("PASS")
print("===============================")
