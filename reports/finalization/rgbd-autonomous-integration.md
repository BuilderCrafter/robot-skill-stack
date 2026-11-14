# Autonomous reconstructed RGB-D global RRT integration

Running engineering record, 2026-11-14 UTC. Starting V5 commit:
`71c07f815410a308dc7ea176148c017055c7fd7a`. Development branch:
`feature/rgbd-rrt-autonomous`. The original V5 branch and its successful research
proof remain unchanged. Sprint budget: approximately two hours, starting 16:20
UTC; implementation/testing stops by 18:20 UTC, followed only by handover.

## Checkpoint A — architecture decision and baseline

Initial worktree was clean; fetch confirmed the V5 tip above. Read `AGENTS.md`,
the requested finalization reports, V5 documentation and current source. The
known-good partial proof remains immutable: two qualified boxes, 24 camera
proxies, fresh final capture at 0.945622 s, independent physical acceptance and
contact-free native execution. This is a research baseline, not production
readiness evidence.

**Selected approach A:** optionally run the unchanged numerical reconstruction
and visibility qualification inside the existing mapping subprocess, alongside
its occupancy/extraction computation. Publish one immutable inferred-geometry
product bound to that exact completion's generation, revision, acquisition time
and calibrated frames. The existing owner-thread publication barrier accepts or
rejects the whole observation; no independently refreshed geometry is allowed.
Capture validates the ordinary frozen source first, then assembles a separate
numeric planning snapshot from its original occupied cells and qualified boxes.

Rejected B: semantic perception has different scheduling/context/segmentation;
reusing its tracked products would require another cross-worker synchronization
contract and cannot supply the required exact raw robot/frame association.
Rejected C: owner-thread on-demand inference repeats the research blocking and
54 ms freshness margin; adding another asynchronous worker is outside scope.

The frozen research reference/visibility files remain hash-identical. Numerical
code will be extracted into an SDK-free shared module; production will never
import `research`. No algorithm, fit/sample thresholds or fixture inputs change.

Opt-in plan: keep `global_obstacle_source = "frozen_rgbd"` and add
`global_rgbd_representation = "reconstructed"`, default `"occupied"`. Preserve
static GT/direct/reactive defaults and all existing planner/executor contracts.
Only the reconstructed mode enables worker inference; normal Lula derives its
own validated joint goal from the ordinary request, without GT/control input.

All selected occupied IDs must be retained or conservatively enclosed at all
eight corners; protected floor cells and support Plane remain. The camera
budget stays 64. Empty inference never deletes observed cells or proves UNKNOWN
space free. Missing/invalid/ambiguous products fail closed; original acquisition
age remains below one second at capture. Reset, generation and failure barriers
invalidate geometry together with occupancy.

**Measured baseline/uncertainty:** mapping computation/transport plus roughly
410 ms inference may exceed the one-second acquisition limit or reduce mapping
below the 1.8 Hz target. This is an explicit reliability risk to measure, not a
reason to raise limits, add resources or select old qualified geometry.

## Chronological decision log

- 16:20–16:35: inspected source/worker/capture/factory boundaries. Chose optional
  same-worker publication because the existing process already owns immutable
  synchronized camera/robot payloads. Change is opt-in and reversible. No native
  process started; performance and general qualification remain unverified.

## Validation plan and status

Offline: extraction equivalence, qualification negative controls, source identity
and lifecycle barriers, conservative proxy accounting and existing motion/runtime
regressions. Native: prioritize autonomous partial, then empty/single-obstacle,
different geometry and floor/base cases. Continue past scene-specific rejection.
Native research instrumentation may independently veto motion, but it must not
provide geometry or an IK goal to the selected production planner.

## Checkpoint B — opt-in worker/capture integration

Implemented an optional immutable `InferredGeometry` attached to the existing
`MappingCompleted` and `MappingReadView`. The service publishes it only with the
same accepted completion and clears it at reset. It never pairs an old qualified
result with a newer occupied world. Source capture checks generation, revision,
timestamp and every camera token/time/calibration field before assembly.

The numerical reference/visibility modules were copied into SDK-free
`world/spatial_mapping/box_geometry`; AST regression confirms all code after the
module docstring is identical to the frozen research originals. No production
module imports research. The worker receives only whitelisted workspace/support
values, raw frames and synchronized robot snapshots, never fixture identities.

Configuration example (existing RGB-D mapping/self-filter configuration required):

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "frozen_rgbd"
global_rgbd_representation = "reconstructed"
max_obstacles = 64
```

Default representation remains `occupied`. GT/direct/reactive defaults are
unchanged. Global strategy changes only pass the opt-in representation into the
existing capture boundary; Lula, IK, executor and motion skills are untouched.
Empty inference retains all observed cuboids; zero selected occupancy still
receives the existing explicit no-usable-evidence refusal rather than treating
UNKNOWN as free. Unqualified or mixed robot-connected components reject capture.

103 targeted offline tests pass, covering initial geometry/source assembly,
mapping defaults, global strategy, observer and configuration regressions.
Further lifecycle/source-adapter and trusted archive checks follow at checkpoint
C. Native behavior and inference throughput are not yet measured.

- 16:28–16:35 decision: preserve the existing empty-source and floor guards.
  Evidence: their contracts require observed selected cuboids; inference alone
  cannot certify free space. Implemented explicit empty-inference status while
  retaining occupancy. Reversible opt-in; empty/floor scenes may safely refuse.

Current status: `PARTIALLY_INTEGRATED`; no production readiness claim.

## Checkpoint C — source/lifecycle regressions and native test boundary

Package-qualified full unit discovery passed **1,394 tests in 106.380 s**.
The first discovery invocation omitted `-t .`, importing subprocess test fixtures
as unresolvable top-level modules; its eight failures/one error were command
configuration errors. Correcting the invocation resolved them without changing
the process worker or historical tests. Original logs remain retained.

New coverage includes immutable geometry transport, exact source/calibration
association, stale/reset/late-completion barriers, empty inference retaining all
cells, floor/union/budget checks, invalid geometry, default occupied-only behavior
and known-support preservation. Trusted archive replay reproduces both models
from the successful partial capture to 1e-12 numerical tolerance without reading
GT. Ten existing measured negative controls and a non-box cloud control reject
geometry; missing rear/lower/top, invalid and unobservable patches are covered.

The native research harness uses the normal configured production capture and
normal Lula planner/IK. A scoped planner observer independently vetoes unsafe
execution. Only AFTER the selected ordinary plan exists does it derive a common
joint goal for GT, same-packet occupied-only and support-only controls. That goal
never flows back into the selected planner. Tests cover goal-flow direction,
selected failure before controls and mandatory physical veto. This instrumentation
is not required by application/runtime configuration and adds no controller.

- 16:38–16:46: independent test boundary added; retained the original benchmark
  path by default. Opt-in autonomous testing skips its old capture replacement
  and records that the production capture is unchanged. Decision is reversible;
  native performance and safety results remain to be measured.

## Checkpoint D — native evidence (running)

The Isaac-container offline subset passed **65 tests in 5.269 s**; architecture
and syntax checks passed. Container startup reused its saved configuration;
exclusive Kit/GPU preflight was clear. Each run uses a new directory below
`outputs/motion-planning-v5/autonomous-integration-20261114`, original protocol
fixtures, two cameras, unchanged rates/rays/voxel/budget, and a 600 s deadline.
Commands and exit status are retained separately; SDK exit zero is not success.

- 16:51–16:52, partial attempt 1: both boxes qualified; inference 318.510 ms.
  First new publication age at readiness was 952.692 ms. Authoritative ordinary
  capture correctly refused it at **1,028.465 ms**, `planning_source_stale`.
  Selected-request action count zero; all runtime/scene/source contracts passed.
  This is `SOURCE_FAILURE`, not a planning or geometry failure. Recent accepted
  publication ages were 835–939 ms, compute 495–567 ms, transport 127–167 ms,
  queue 47–248 ms. Hypothesis: inference plus existing map/transport/queue leaves
  insufficient margin, worsened by test instrumentation between readiness and
  request. No threshold, timestamp or scheduling change made. Continue matrix.
- 16:52–16:53, empty attempt 1: empty inference retained all 64 selected floor
  cuboids. Source age **789.946 ms** passed unchanged capture. Lula base preflight
  rejected protected floor evidence with `base_geometry_rejected`; selected
  action count zero, no physics advancement during selected request. This is
  `EXPECTED_SAFE_REJECTION`; no floor removal or UNKNOWN-to-free conversion.
- 16:53–16:54, static attempt 1: measured box qualified in 186.548 ms and
  authoritative capture passed at **978.309 ms**. Protected floor evidence again
  caused `base_geometry_rejected`, before IK/RRT/execution. No correction: this
  is the established conservative base-guard limitation, not evidence to remove.
- 16:55 decision: retain the unmodified implementation while finishing the scene
  matrix. Run exactly three total `partial` trials at the same fixed planner seed
  to measure short-run variability (attempt 1 already failed). No seed search,
  exposure-age selection, fixture change or success-dependent stopping. Report
  every attempt. Performance work requires measured avoidable duplication;
  inference itself and existing transport/queue cost cannot be wished away.
- 16:54–16:55, large attempt 1: authoritative source expired before assembly.
  Independently retained inference was already rejected: the large component
  qualified, but additional nonrobot components of 55 and 28 points lacked
  sufficient box surfaces/points. Discarding these measured components would
  hide unsupported geometry; no such change made. Inference 353.781 ms.
- 16:55–16:57, partial attempt 2: both boxes qualified in 286.663 ms. Readiness
  age 911.821 ms; authoritative capture age **1,006.755 ms**. Rejected stale,
  zero selected actions. Readiness-to-capture overhead was 94.935 ms; attempt 1
  overhead was 75.773 ms. This includes research observation and GT test-world
  hashing before the ordinary request; it is not reconstruction owner blocking.
- 16:57–16:58, narrow attempt 1: one component had an unqualified hidden bound.
  Ordinary source assembly propagated `planning_source_reconstruction` with
  `component_1: unqualified_hidden_bound`. Inference 258.550 ms; zero actions.
- 16:58–16:59, v43 attempt 1: one box qualified in 154.247 ms; authoritative
  source age **836.134 ms**. Existing fixed-base guard rejected retained floor
  cuboids, before IK, RRT or execution; zero actions. No safety correction.
- 17:00, sparse attempt 1: sensor box qualified in 167.793 ms, but the inferred
  box plus every retained observed/floor proxy exceeded 64. Assembly correctly
  rejected `planning_source_reconstruction` / `Combined inferred/observed world
  exceeds 64 camera proxies`, classified `COLLISION_WORLD_FAILURE`; zero actions.
  Preserve original selected evidence and budget; no packing/removal workaround.
- 17:01–17:03, verified-occlusion attempt 1 rejected the occluder's insufficient
  observed three-axis patch/residual; 295.528 ms inference, zero actions. The
  separate native depth diagnostic did confirm its declared partial occlusion.
- 17:02–17:03, partial attempt 3: both boxes qualified in 302.249 ms; readiness
  age 933.064 ms, capture age **1,005.905 ms**, zero actions. Original three-trial
  series is **0/3 authoritative capture successes**. Timing-only instrumentation
  measured 13.232 ms between readiness recognition and callback return (including
  JSON conversion), 8.990 ms test GT-world hashing and 30.255 ms from observer
  operation start to capture. It did not change source/goal/physics behavior.
- 17:04 bounded correction decision: defer immutable inference serialization to
  post-experiment output and hash the test GT world before source readiness.
  These are research instrumentation costs, absent from ordinary application
  requests; they needlessly consume the source's freshness window. No production
  change, new source selection rule, extra wait, threshold change or timestamp
  refresh. Original three failures remain counted. Run one separately labeled
  corrected partial trial at the same fixed seed, then stop this root-cause
  iteration regardless of outcome. This correction is reversible and does not
  establish runtime publication reliability even if the next trial succeeds.
