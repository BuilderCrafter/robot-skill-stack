# Isaac GUI

The Isaac extension reads cached state from `WorldModel` and sends manipulation
commands through `RobotRuntime`. It checks generic backend readiness, displays
source health, and offers frame capture when the bundle provides that callback.
Refresh is read-only; Clear Lost
explicitly requests removal through the updater. Stale or failed source state
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

Click an object row to select it. `Pick Selected` and the Pick+Place Behavior
Trees use that selected persistent ID. Place acts on the currently held object.

The same object display and manipulation controls work with
`[world].provider = "perception"` and `[world].provider = "ground_truth"`.

The panel refreshes from cached WorldModel state at roughly 5 Hz, including
during async execution. It never starts a second perception loop.

There is no checked-in ground-truth profile. From the repository root, copy
`config/scenes/playground.toml` to `outputs/playground_ground_truth.toml` (create
`outputs/` first), and change only `[world].provider` to `"ground_truth"` in the
copy. Restart Isaac and select this profile before initialization; the current
GUI does not switch profiles on an initialized robot. The USD scene is unchanged.

Each file is a complete setup, without inheritance. `[scene]` contains the
support-plane height and workspace bounds. `[perception]` contains camera
references; `[ground_truth]` contains oracle references and optional object
overrides. RGB-D can omit the entire ground-truth section, and ground truth can
omit the camera section. Existing custom profiles need the
[Phase 4B key migration](phase_4b_config_adapters_ui.md#configuration-and-migration).
