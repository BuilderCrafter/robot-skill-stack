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

The Franka backend delegates Cartesian motion to one selected SDK-free
`MotionStrategy` contract. Its Isaac-specific direct and reactive strategies
borrow the existing controller, IK and robot; the backend keeps all SDK resource ownership.
Home uses the same delegation. Selection is validated and injected once during
construction, and overlapping backend arm requests are rejected. An optional
late-bound provider exposes V5's read-only mapping service to reactive; direct
never queries it. Reactive requires fresh prepared obstacle evidence and returns
explicit unavailable-world-state failures on staleness or source failure.
Free-space MoveToPose/Home use the common arm-command guard. A narrow backend
capability query makes Pick/Place return `MOTION_UNSUPPORTED` before side effects
for reactive; direct manipulation remains intact. See [motion strategies](motion_planning.md).

`WorldObservationProvider` implements `observe(context) -> ObservationBatch`,
`forget(id)` and `close()`. A fresh sample, including an empty one, has an
increasing monotonic timestamp. No new sample has no timestamp; errors raise.
`WorldModelUpdater` owns health, visibility, expiration and coordinated
forgetting. In normal RGB-D runtime, `ObservationRuntime` schedules acquisition
and publication on Kit, while bounded SDK-free workers compute geometry.
Tracking association and WorldModel mutation stay on the owning thread.
Synchronous `observe()`/updater diagnostics remain available. Sources own their camera/resources. WorldModel owns
cached objects, attachment state and placement hints. See [perception](perception.md).

Opt-in [spatial mapping](spatial_mapping.md) is a separate occupancy map, with
simulator-independent voxel evidence/depth rays and Isaac geometry/camera
adapters. Enabling mapping in the setup starts its independent worker service;
`bundle.spatial_mapping.snapshot()` returns a lightweight immutable read view.
Its worker owns the mutable grid. Reactive selection enables optional SDK-free
occupied-cell cuboid extraction in that same worker. Its immutable packet is
published with the matching map generation, revision and source timestamp;
direct mapping does not compute this product. The Kit reactive strategy reads
only the small packet, validates freshness and updates bounded numeric cuboids
through public RMPflow APIs. These planning objects have no USD prims, colliders
or rendered geometry. Skills and grasp/placement planners do not inspect voxels.
Sparse support uses a radius-two neighborhood without geometric dilation.
RGB-D additionally retains already OCCUPIED, unexpired cells with temporal
probability at least 0.95; ground-truth extraction keeps the previous rule.
Floor-only cuboids follow above-plane hazards within the bounded budget using
the existing scene support plane. Filtering and budget omissions stay visible;
no FREE or UNKNOWN geometry is filled. Robot self-filter ray intersections use
bounded batches over the same mesh BVHs and exact intersection tests, with the
previous traversal retained as a fallback. This changes computation cost, not
the selected rays, robot masks, timestamps or occupancy probability model.
Overlapping camera sources share copied frames and render products; mapping-only
cameras have explicit lifecycle ownership.

Each worker has one running job and one replaceable pending job. Camera and
exposure-matched PhysX snapshot reads stay on Kit; projection, fitting, ray/mesh
intersection, occupancy integration and full snapshot copying run in workers.
The Kit subscription never waits for completion. Generation/failure barriers
discard obsolete results, and monotonic WorldModel context revisions reject
geometry captured before held/hint/forget changes before tracker association.
Normal runtime uses a separate SDK-free Python interpreter for each processor;
bounded parent I/O threads start it, serialize observations and receive results.
Children import neither simulator SDKs nor the native application's main module.
Standalone bounded threads remain available, but the measured combined thread
prototype failed the unchanged observation freshness/throughput limits.

## Construction

`integrations/isaac/bootstrap.py` loads one [setup TOML](configuration.md), checks
the robot and selected source's prims, and constructs:

1. Existing/deferred World and physics context, then robot factory's SDK wrapper.
2. Reset/play and 30 Kit updates.
3. Source factory: ground truth, or camera initialization plus 60 Kit updates.
4. WorldModel/updater and robot factory's backend, selected motion strategy and gripper spec.
5. Existing grasp/placement planners, four skills, registry and runtime.
6. RGB-D observation coordinator and optional mapping service, then the bundle.
   Lightweight ground-truth semantics retain their initial synchronous sample
   and owned physics callback.

`manipulation/factory.py` explicitly selects Franka, the only implemented robot.
`world/factory.py` explicitly selects RGB-D V1, multi-view or ground truth. Unknown selectors
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
close first releases the selected strategy's owned obstacle registrations,
then observation resources and source cleanup. Initialization failure/cancellation
must close acquired resources, and repeated initialization closes the prior
bundle. UI window disposal remains deferred. These SDK workarounds are retained,
covered by doubles and require native revalidation when touched.

Observation restart resets camera sampling clocks and blocks retained exposures.
After Stop/Play, the owner rebinds cached robot geometry to a validated current
PhysX view, then waits for newly captured exposure-matched poses. Model/topology
edits or a broken subprocess transport require clean runtime reinitialization.
No view or model is rebuilt inside a physics callback. The supported Python
launchers provide the worker interpreter; an embedded Kit executable is rejected
with launcher guidance.

The Franka feedback/arm-hold corrections are explained in
[control diagnostics](control_diagnostic.md). They must not be replaced with raw
instantaneous velocities merely to shorten the adapter. Home and transport are
not collision-free guarantees; see [manipulation](manipulation.md).
