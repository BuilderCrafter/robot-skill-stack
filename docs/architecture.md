# Architecture

```text
GUI / Behavior Trees / standalone Kit bridge
                  ↓
       await RobotRuntime.execute_async(...)
                  ↓
      Home / MoveToPose / Pick / Place
         ↙                      ↘
grasp / placement checks    ManipulationBackend → Isaac Franka
                  ↑
              WorldModel ← WorldModelUpdater ← ObservationBatch
                  ↓               ↑                  ↑
          read-only GUI view   held / hints      RGB-D V1 or ground truth
```

## Components and boundaries

`runtime`, `manipulation`, `world`, `presentation` and `orchestration` are
simulator-independent. Isaac/Omniverse imports belong under
`integrations/isaac` and the extension. Skills use the cached WorldModel and
`ManipulationBackend`; they never query SDK scene objects or a concrete source.

`RobotRuntime(registry, world_model)` shares the skills' model. It reserves a
command before its first await, rejects overlap with `BUSY` and releases
ownership on result, exception or cancellation. Skill classes declare a simple
`name` and `execute_async`; the small registry prevents duplicate names.
There is no metadata interpreter, command queue or worker-thread controller.

`ManipulationBackend` provides async move/open/close/Home, readiness,
pose/reachability queries, gripper feedback and grasp-width limits. Commands
cooperate with Kit; queries do not advance simulation. Carrying commands call
the passed grasp guard after updates and stop through their existing hold path
on failure. `HeldState` owns attachment bookkeeping, evidence resolution and
grip-loss policy; the backend has no WorldModel dependency.

`WorldObservationProvider` implements `observe(context) -> ObservationBatch`,
`forget(id)` and `close()`. A fresh sample, including an empty one, has an
increasing monotonic timestamp. No new sample has no timestamp; errors raise.
`WorldModelUpdater` owns sampling rate, health, visibility, expiration and
coordinated forgetting. Sources own their camera/resources. WorldModel owns
cached objects, attachment state and placement hints. See [perception](perception.md).

Opt-in [spatial mapping](spatial_mapping.md) is a separate occupancy map, with
simulator-independent voxel evidence/depth rays and Isaac geometry/camera
adapters. Callers explicitly update/query it; WorldModel, skills, planners and
runtime callbacks do not own or consume its grid. Mapping can borrow the same
acquired RGB-D frames without initializing another camera pipeline.

## Construction

`integrations/isaac/bootstrap.py` loads one [setup TOML](configuration.md), checks
the robot and selected source's prims, and constructs:

1. Existing/deferred World and physics context, then robot factory's SDK wrapper.
2. Reset/play and 30 Kit updates.
3. Source factory: ground truth, or camera initialization plus 60 Kit updates.
4. WorldModel/updater and robot factory's backend plus gripper spec.
5. Existing grasp/placement planners, four skills, registry and runtime.
6. Initial observation and the existing owned physics callback, then the bundle.

`manipulation/factory.py` explicitly selects Franka, the only implemented robot.
`world/factory.py` explicitly selects RGB-D V1 or ground truth. Unknown selectors
fail; RGB-D does not require oracle objects or wrappers. Native test oracle
handles belong to `tests/support/isaac_helpers.py`, not the production bundle.
There is no plugin registry, setup inheritance or live adapter switching.

A new robot implements the backend contract and supplies its gripper spec in
the factory, with its selector allowed in config. A new source implements the
observation contract and adds explicit construction/reference validation. Its
observation logic can remain Isaac-free. Skills, WorldModel and GUI do not need
robot/source branches. Future ROS or thesis-level planning should use these
boundaries; neither integration is implemented.

## Orchestration and presentation

One `SkillNode` submits runtime coroutines; `BTExecutor.run_async` yields between
ticks. The recovery selector remembers its running branch. Home/retry is allowed
only for eligible pre-closure Pick failures, never after engagement, grip loss,
uncertainty, source failure, busy or internal errors. Place failure stops either
task before final Home. Cancellation/tick exhaustion drains child tasks; this
is coroutine cleanup, not an emergency stop or automatic physical recovery.

The extension reads `WorldModelViewModel`, backend readiness, runtime busy and
source health. Refresh is read-only. Confirmed Clear Held calls the runtime's
software reset, which rejects active commands or changed IDs. Clear Lost uses
the common updater with held/hint protection. Optional bundle
`capture_frame(path)` isolates RGB-D diagnostics; GUI cleanup calls bundle
`close()` instead of managing sources.

## Isaac execution and lifecycle constraints

All commands follow one cooperative async path. Standalone native tests use
`tests/support/isaac_async.py` to pump `SimulationApp.update()` until a coroutine
finishes. Live code must not use `asyncio.run()` or add blocking `world.step()`
loops inside commands. See [validation](validation.md) for launcher invariants.

Deferred World construction avoids draw/button callback reentrancy. Owned
physics callback scopes preserve registration/removal across Stop/Play; bundle
close and updater stop own source cleanup. Initialization failure/cancellation
must close acquired resources, and repeated initialization closes the prior
bundle. UI window disposal remains deferred. These SDK workarounds are retained,
covered by doubles and require native revalidation when touched.

The Franka feedback/arm-hold corrections are explained in
[control diagnostics](control_diagnostic.md). They must not be replaced with raw
instantaneous velocities merely to shorten the adapter. Home and transport are
not collision-free guarantees; see [manipulation](manipulation.md).
