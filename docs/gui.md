# Isaac GUI

The Isaac extension reads cached state from `WorldModel` and sends manipulation
commands through `RobotRuntime`. It checks generic backend readiness, displays
source health, and offers frame capture when the bundle provides that callback.
Refresh is read-only; Clear Lost explicitly requests removal through the
updater. Stale or failed source state
disables Pick, Place and observation-dependent tasks. Home/Move remain available
subject to existing readiness and held-state checks. Clear Held remains an
explicit, confirmed software-only reset through the runtime, rejected while a
command is active. Clear Lost works with either source and preserves held/hinted
objects. No controls were added or rearranged.

## Launch

Use the repository wrapper so `.deps` is ahead of Isaac's bundled Python paths:

```bash
./run_isaac_sim.sh
```

Set `ISAAC_SIM` if the installation differs from `/home/etfrobotics/isaacsim`.
The GUI launcher imports and requires `numpy==1.26.4` before `SimulationApp`.
The bootstrap additionally rejects NumPy 2.x for perception. The full Isaac
experience and ROS bridge startup overrides are retained. Do not pass
`--no-ros-env`: Isaac Sim 5.1.0-rc.19's Python launcher does not support it.

Open `scenes/playground.usd` if it is not already open, then use the **Robot
Skill Stack** window and press **Initialize Runtime**.

## WorldModel panel

The panel displays every persistent WorldModel object, including currently lost
objects:

- persistent ID
- semantic class
- current/last position
- estimated size
- visibility state
- held state

Click an object row to select it. `Pick` and the Pick+Place Behavior
Trees use that selected persistent ID. Place acts on the currently held object.

The same object display and manipulation controls work with
`[world].provider = "perception"` and `[world].provider = "ground_truth"`.

The panel refreshes from cached WorldModel state at roughly 5 Hz, including
during async execution. It never starts a second perception loop.

Use a complete setup file to select the source; the
[configuration guide](configuration.md#required-settings-and-source-selection)
provides the ground-truth copy command, schema and migration notes. Restart
Isaac before switching setups; the GUI does not switch an initialized robot.

Capture writes an on-demand RGB-D `.npz` under `outputs/perception/`. To replay
one without starting Isaac, use the existing wrapper (which uses Isaac's Python
and repo-local NumPy, but constructs no `SimulationApp`):

```bash
python3 scripts/replay_perception_capture.py outputs/perception/capture_TIMESTAMP.npz
```

This is diagnostic replay of one frame, not a separate observation runtime.
