from core.types import Pose
from world_model.observations import ObjectObservation
from world_model.updater import WorldModelUpdater
from world_model.world_model import WorldModel


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def observe(self):
        self.calls += 1
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

print("Calls after 120 ticks:", provider.calls)


print("\n[2] Testing 5 Hz updater...")

provider = CountingProvider()
model = WorldModel()
updater = WorldModelUpdater(
    model,
    provider,
    update_hz=5,
)

for _ in range(120):
    updater.tick(1 / 60)

print("Calls after 2 simulated seconds:", provider.calls)

assert provider.calls == 10
assert model.require("cube").pose.position[0] == 10


print("\n[3] Testing 10 Hz updater...")

provider = CountingProvider()
model = WorldModel()
updater = WorldModelUpdater(
    model,
    provider,
    update_hz=10,
)

for _ in range(120):
    updater.tick(1 / 60)

print("Calls after 2 simulated seconds:", provider.calls)

assert provider.calls == 20


print("\n=== WORLD MODEL UPDATE RATE ===")
print("Every step: PASS")
print("5 Hz:       PASS")
print("10 Hz:      PASS")
print("PASS")
print("===============================")