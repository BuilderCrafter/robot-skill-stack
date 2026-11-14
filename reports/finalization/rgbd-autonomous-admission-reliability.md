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

Baseline outcomes, no production change:

| Declared delay s | Actual fixed delay s | Request after preparation s | Age at request ms | Authoritative age ms | Selected result |
|---:|---:|---:|---:|---:|---|
| 0.0 | .000002 | .028314 | 1081.359 | 1126.462 | planning_source_stale; zero actions |
| 0.3 | .322129 | .350246 | 1376.343 | 1428.375 | planning_source_stale; zero actions |
| 0.6 | .602857 | .615931 | 919.904 | 939.904 | physically accepted native execution; 110 actions |

All cases completed and preserved; failures had no internal runtime error.
The third timing succeeded without publication gating, proving the old callback
is not required for every success; two other natural timings rejected safely.
The material 1/3 ordinary success rate justifies production admission correction.

## C — justified readiness correction

Choose reconstructed-mode-only **always first new accepted publication**, with a
five-second cooperative deadline. Current freshness does not change this policy:
even an initially usable source waits for the next accepted observation. This is
an explicit latency tradeoff, avoiding an arbitrary scene-tuned headroom threshold.
Check lifecycle and source generation/failure barriers throughout; do not poll
workers or sensors directly. Qualification is never inspected by the wait.

After selection, pin the same service and exact generation/revision/timestamp,
then perform one ordinary authoritative capture with the unchanged <1 s rule.
If it has expired, is inconsistent or has unqualified geometry, reject once;
never continue looking for another exposure. Recheck current payload/fingers after
cooperative waiting. No arm command or solver activation precedes a valid plan.
GT and occupied-only paths do not enter this wait.

- 18:55–18:58: completed all predeclared independent baseline timings; ordinary
  freshness is materially phase-dependent (two failures, one safe success).
  Implement the existing coroutine's small admission wait, not another service,
  worker, source or timestamp rule. Native post-correction uses exactly the same
  three ungated timing conditions and independent evaluator.

Offline correction coverage: real MappingService publications and original source
adapter in unit tests; immediately usable/stale current views, new publication,
timeout/stalled update, cancellation, Stop/Play/reset/close/articulation changes,
source/worker failure and recovery barriers, late rejected completions, identity/
service replacement, skipped publications, expired selected source, rejected/empty
inference, no geometry search, finger changes and zero pre-plan commands. GT and
occupied-only bypass admission. Existing full union/floor/collision/robot-model
and FREE/UNKNOWN tests remain intact.

The wait uses Python 3.11 `asyncio.timeout` around the existing update await,
so a deadline does not create an extra asyncio task for each Kit update. External
cancellation propagates; timeouts and lifecycle failures refuse before capture.
An initial full run passed 1445 tests but included 21 imported helper-fixture
cases twice. Importing that helper through its module removes duplicate discovery;
no fixture assertion was changed. Final **1424 tests PASS in 111.276 s**;
**85 focused tests PASS in 2.266 s**. Architecture/syntax pass. Production diff
is **71 added / 3 removed lines**, only GlobalRrtStrategy; no new configuration,
worker, source service, sensor reader, dependency, planner or executor change.
