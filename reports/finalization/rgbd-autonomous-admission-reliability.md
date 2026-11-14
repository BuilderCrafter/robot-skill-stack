# Autonomous RGB-D ordinary-request admission reliability

Running engineering record, 2026-11-14 UTC. Starting fetched clean HEAD
`acf331d2b04aeb038e21ac711292c2b96a9eed6c`, continuing only on
`feature/rgbd-rrt-autonomous`. Start 18:48 UTC; useful work budget 30–60 minutes,
stop development by 19:48 UTC. Original V5 and every preceding measurement remain
unchanged. Container initially stopped/exit 137, no GPU compute process/stream.

## A — ordinary request versus research preparation

Read AGENTS, requested reports and actual runtime, skill/backend, global strategy,
frozen source, mapping service, research entry/preparation and existing tests.
The successful performance checkpoint remains: 3/3 physically accepted executed
partial trials, capture ages 906.868/766.312/748.799 ms, 24 camera proxies, 0.875 mm
TCP error, zero reports, 1401 passing unit tests; mapping still .87–1.19 Hz.

Actual sequence: RobotRuntime reserves busy ownership; MoveToPose delegates to
FrankaBackend, which reserves motion ownership; GlobalRrtStrategy checks target,
lifecycle/payload/finger contract, synchronously captures the latest source, then
runs ordinary Lula IK/planning and the existing executor. There is no cooperative
source wait in the ordinary request. Capture accepts or refuses whichever current
publication exists at that instant. After freezing, no active-world refresh occurs.

The old research callback explicitly awaited a first new publication before the
request, then captured actual start state. Thus it performed preparation absent
from the application request. This establishes a timing risk, not yet a measured
ordinary-request failure. The asynchronous strategy can already await
`backend._next_update()` while retaining runtime/backend ownership. Existing
`_live(epoch)` checks reject stop, close or articulation invalidation. A source
wait would additionally need generation/failure barriers and a strict deadline;
it must not search for qualified boxes or change authoritative age checks.

## B — predeclared ungated native baseline

Declare exactly three independent original partial trials: **0.0, 0.3, 0.6 seconds**
after the end of physical-reference preparation (including its static-world hash).
One fresh Kit per trial, fixed seed/scene/cameras/budgets. No adjustment based on
mapping health. The fixed delay uses cooperative Kit updates and may overshoot
its deadline by one update. Existing request-observer setup follows; actual request
time relative to preparation is recorded rather than pretending exact wall timing.
The old publication-wait callback is disabled in all these trials.

Research-only option `--request-delay {0.0,0.3,0.6}` leaves the historical gated
entry behavior unchanged when omitted. Record request timestamp, latest immutable
mapping identity/age and worker health immediately before the normal request;
never inject a source/GT goal. Record selected actual planning-entry time and exact
frozen inference product. Independent evaluation resamples actual robot/body start
state at planner entry, because any future production wait advances physics;
this refresh affects only the test evaluator, never the selected source or IK.
All original physical geometry, original warmup and veto remain.

No production correction before the three measured baseline outcomes. If failure
is material, evaluate a reconstructed-mode-only deterministic policy: always await
the first new accepted publication (even if the current source is fresh), within
five seconds, then capture once with the unchanged one-second guard. This avoids
an arbitrary near-stale threshold. Qualification failure of that publication must
fail the request; no further exposure search. GT and occupied-only remain unchanged.

Chronological decisions:

- 18:48–18:56: trace confirms synchronous current-source capture, separate research
  preparation and reusable cooperative/lifecycle primitives. Add fixed-time-only
  research mode and actual-start evaluator refresh. No geometry/performance change.
  Run focused research tests before three predeclared baseline native cases.

Research timing/goal-flow and existing paired-veto subset: **14 tests PASS in
.045 s**; architecture and syntax PASS before native baseline. The initial test
command also named a nonexistent benchmark-test module; correcting that command
resolved the import error without changing existing tests or implementation.
