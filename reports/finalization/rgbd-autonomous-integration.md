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

Current status: `NOT_IMPLEMENTED` (architecture checkpoint only). No production
readiness claim. Subsequent checkpoints append measured outcomes and remaining
limitations here.
