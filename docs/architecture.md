# Architecture

The production composition is:

```text
GUI / Behavior Trees
  → RobotRuntime → registered skills → ManipulationBackend → MotionStrategy
Semantic observations → WorldModelUpdater → WorldModel → skills / GUI
Independent spatial observations → MappingService → immutable MappingReadView
```

`common`, `runtime`, `world`, `manipulation`, `orchestration` and `presentation`
remain SDK-free. Isaac adapters assemble the existing contracts in
`integrations/isaac/bootstrap.py`; the GUI invokes that same asynchronous builder.
There is no plugin framework or alternate motion executor.

## Commands and manipulation

RobotRuntime admits one asynchronous command. Cancellation propagates through
skills and backend ownership. Direct RMPflow is the default strategy and retains
Pick, Place, Home, grasp/loss checks, held-state policy, release and recovery.
Franka-specific joint mapping lives in the Isaac motion adapter; generic route
validation accepts explicit named joints.

Static global motion uses `GlobalRrtStrategy` and `LulaGlobalPlanner`, with ordinary
Lula IK, static-world capture, enhanced/stock model selection, joint/route/spline
validation, tracking checks, terminal settling and final hold. Planner resources
close in the existing cleanup path. GT planning does not require Spatial Mapping.
The optional reactive obstacle strategy and registry are retired. Direct RMPflow
continues to use its ordinary controller.

## Acquisition and workers

`ObservationRuntime` owns Kit-thread camera acquisition and publication. A
`SharedFramePool` shares original exposures between semantic perception and
mapping. SDK-free processors run in bounded persistent child interpreters through
`ProcessWorker`: one running job and one replaceable pending job. Immutable arrays
and explicit generation/revision barriers prevent mutable aliasing or publication
of invalidated results. No worker owns SDK handles.

Semantic association and WorldModel mutation remain on the owner thread. Held
identity, hints and observation-context revisions invalidate obsolete association
results. Spatial Mapping publishes separate occupancy; it does not replace the
semantic WorldModel or automatically make UNKNOWN space FREE.

RGB-D mapping prepares synchronized robot exclusion from actual robot poses and
cooked geometry. Projection, FREE/OCCUPIED evidence, bounded occupied-cuboid
extraction and optional measured-box qualification run in the existing mapping
worker. No standalone synchronous mapping factory or second mapping pipeline is
used by the GUI/runtime.

## Frozen RGB-D global source

`frozen_rgbd` requires the enhanced model and enabled RGB-D mapping with automatic
robot filtering. Reconstructed mode waits cooperatively, within the existing
five-second admission deadline, for the first new accepted publication. It then
captures once with the original strict one-second freshness check. It does not
search for a favorably qualified exposure or retry a rejected source.

The immutable frozen world retains generation, revision, original timestamps,
calibration, occupied evidence and uncertainty. The occupied representation uses
selected cell cuboids. Reconstructed mode adds qualified measured boxes and
subsumes original cuboids only with the existing all-eight-corner proof. Protected
floor evidence and the 64-camera-proxy limit remain. Known support is restricted
to the existing explicit Plane allowlist; scene GT obstacles never enter this source.
Invalid, stale, contradictory or unqualified sources fail explicitly, with no
GT/direct fallback. See [motion planning](motion_planning.md).

## Lifecycle

Bootstrap initializes the selected source, backend, workers and shared cameras,
then registers the completed bundle. Partial startup closes constructed resources.
Stop/Play invalidates generations and refreshes synchronized robot information;
late completions cannot revive old data. Shutdown removes subscriptions/callbacks,
closes mapping/semantic workers and releases motion resources. Reinitialization
uses distinct owners. GUI controls delegate to these existing lifecycle paths.

Run `scripts/check_architecture.py` and the [retained regressions](validation.md)
when changing composition, source publication or lifecycle handling.
