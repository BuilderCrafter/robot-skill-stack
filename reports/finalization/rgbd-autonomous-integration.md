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
