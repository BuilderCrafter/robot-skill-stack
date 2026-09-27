# Isaac GUI

The Isaac extension is a thin client of the generic lower-level stack. It reads
cached state from `WorldModel` and sends commands through `RobotRuntime`; it does
not query USD, the camera, or perception directly.

## Launch

Use the repository wrapper so `.deps` is ahead of Isaac's bundled Python paths:

```bash
./run_isaac_sim.sh
```

The project pins `numpy==1.26.4`. When the world provider is `perception`, the
Isaac bootstrap rejects NumPy 2.x with a clear error instead of allowing the
camera/SyntheticData stack to fail later.

Open `scenes/playground.usd` if it is not already open, then use the **Robot
Skill Runtime** window and press **Initialize Runtime**.

## WorldModel panel

The panel displays every persistent WorldModel object, including currently lost
objects:

- persistent ID
- semantic class
- current/last position
- estimated size
- visibility state
- held state

Click an object row to select it. `Pick Selected` and the Pick+Place Behavior
Trees use that selected persistent ID. Place acts on the currently held object.

The same UI works with `[world].provider = "perception"` and
`[world].provider = "ground_truth"` because it only consumes `WorldModel`.

The panel refreshes from cached WorldModel state at roughly 5 Hz when no skill
is executing. It never starts a second perception loop.

A ready-made ground-truth profile is available at `config/scenes/playground_ground_truth.toml`.
