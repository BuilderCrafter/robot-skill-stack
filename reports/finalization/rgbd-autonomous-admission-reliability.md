# Autonomous RGB-D ordinary-request admission reliability

Running engineering record, 2026-11-14 UTC. Starting fetched clean HEAD
`acf331d2b04aeb038e21ac711292c2b96a9eed6c`, continuing only on
`feature/rgbd-rrt-autonomous`. Start 18:48 UTC; useful work budget 30–60 minutes,
stop development by 19:48 UTC. Original V5 and every preceding measurement remain
unchanged. Container initially stopped/exit 137, no GPU compute process/stream.

**Final decision: `RGBD_DEMO_CANDIDATE`, for the tested nominal partial scene.**
Ungated baseline requests succeeded at only one of three declared timings. After
the compact production admission wait, all three timings obtained fresh sources,
passed the independent physical veto and executed with .875 mm TCP error and zero
reported contacts. The research publication wait was disabled throughout. This
does not establish universal scene support, long-run reliability or production
readiness: mapping remains about 1 Hz, and minimum measured capture headroom was
only 32.106 ms. Review this bounded candidate before a thesis demonstration.

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

Native Python 3.11 / NumPy 1.26.4: the same **85 focused tests PASS in 2.440 s**.
Correction checkpoint is `a248c63`; research timing baseline is `250e89c`.
Both checkpoints pushed normally. Post-correction runs keep the same declared
0/.3/.6 second source-independent delays, and still never call the original
research publication-readiness callback. Only the production strategy now waits.

- 19:18–19:32 final review: all six native results/source identities and original
  timestamps retained, corrected first-publication counters independently checked,
  physical veto/actual motion/cleanup gates passed. Inspected required occupancy
  packet fields without changing transport. Restored the stopped container;
  verified every earlier artifact set and report. Bound 131 new raw files and
  checked report links. No further implementation or native run after publication.

## D — native correction results

The same predeclared source-independent timings ran sequentially, one fresh Kit
per trial, with original nominal partial geometry/configuration, original two
640×480 cameras, semantic 5 Hz / mapping 2 Hz settings, 1024 shared rays, .01 m
voxels, 64 camera proxies, enhanced model, START/GOAL, seed 123456 and 10,000
iterations. No retry, threshold/fixture/seed change or second correction was used.

| Delay s | Source age at request ms | Production wait s | Authoritative age ms | Headroom ms | Selected outcome |
|---:|---:|---:|---:|---:|---|
| 0.0 | 1093.744 | 1.793539 | 852.472 | 147.528 | physical acceptance + native execution |
| 0.3 | 1348.383 | .979594 | 967.894 | 32.106 | physical acceptance + native execution |
| 0.6 | 978.123 | 1.300324 | 903.281 | 96.719 | physical acceptance + native execution |

All three selected **exactly the first new accepted mapping publication**:
accepted counters 11→12, 13→14, 11→12. Revisions can jump across rejected/stale
worker completions; this is not skipping accepted observations. The wait does
not inspect inferred status, scores, dimensions or identities. The initially
fresh .978-second source in the third case also waited, as the deterministic
policy explicitly requires. No older qualified model was substituted.

Each accepted identity equals the frozen generation/revision/original timestamp
and its inference/camera records. Capture occurs once, after admission; timestamp
age remains below the unchanged one-second authoritative limit. Subsequent source
changes do not refresh the active world. A selected source expiring before capture
still fails safely without another wait, as directly tested offline.

All three ordinary requests used normal Lula IK and RRT, with no GT selected
geometry or GT-derived joint goal. Independent controls take their joint goal
from the successful selected plan, never the reverse. The test evaluator resampled
actual robot/body start after cooperative waiting; this does not change the plan,
source or goal. Matched starts/endpoints and the unadvanced planning/evaluation
physics instant were verified. Selected and GT controls passed independent
sampling; occupied-only/support-only counterplans were physically vetoed and
never executed. Selected detour was three waypoints, **1.057749 rad** from the
independently obstructed direct segment.

Each selected world retained the original 64 selected occupied cuboids through
42 conservative eight-corner subsumptions and 22 unchanged cells, plus two
qualified inferred boxes: **24 camera proxies**, with separate known support
Plane. No selected floor existed in partial; the floor-preservation rule and base
guard are unchanged. All six measured/edge-qualified model bounds, robot exclusion
and FREE contradiction checks remain. Both inferred solids contained their
post-hoc physical references; no complete FREE cell lay strictly inside them.

Every corrected motion had **110 actions**, **1.107708 rad** maximum observed
joint displacement, **.874674 mm** TCP position / **.00242493 rad** orientation
error, **.036278 rad** worst joint tracking error and **41.143 mm** minimum sampled
moving-arm cooked-solid clearance. Zero reported contact/proximity points or
penetrating pairs; no buffer saturation. Complete physics-step coverage, final
hold, zero final applied arm velocity, solver session-opinion restoration and
runtime closure passed. Sampled evaluation is not continuous swept-volume or
real-robot certification.

Chronological native decisions:

- 19:13–19:15, corrected immediate request: started with stale source age 1093.744
  ms, waited 1.794 s and executed from a new 852.472 ms source. No harness gate.
- 19:15–19:16, corrected .3-second request: initial source age 1348.383 ms, wait
  .980 s, new capture 967.894 ms, successful physical acceptance/execution.
- 19:16–19:18, corrected .6-second request: initial source already fresh at
  978.123 ms; policy still waited 1.300 s and captured a new 903.281 ms source.
  Successful physical acceptance/execution. All three declared corrected cases
  complete; no further native run or production iteration.

## Request latency and unchanged mapping limitations

| Trial | User request → planner entry s | User request → first command s | Inference ms | RRT ms | Accepted mapping Hz | Semantic published / audit s |
|---|---:|---:|---:|---:|---:|---:|
| baseline 0 | — | — | 218.364 | — | .566 | 1.872 |
| baseline .3 | — | — | 223.373 | — | .871 | 2.005 |
| baseline .6 | .084236 | 4.494729 | 214.872 | 5.701 | .807 | 1.997 |
| corrected 0 | 1.883970 | 6.314449 | 230.361 | 5.767 | .806 | 1.988 |
| corrected .3 | 1.070936 | 5.390266 | 219.027 | 12.290 | 1.215 | 2.084 |
| corrected .6 | 1.383262 | 5.822142 | 213.784 | 5.927 | .666 | 1.991 |

Mapping rates use the published-counter delta over the pre-request audit window;
semantic normalization uses the full camera-audit interval including setup/warmup.
These are small independent samples, not steady-state or statistical guarantees.
Higher source acceptance **is not a throughput improvement**: no worker scheduling
or reconstruction compute changed. The prior approximately .98 Hz mapping limit
remains; none of these samples establishes the 1.8 Hz target. Semantic publication
normalization also remains below 4.5 Hz. Existing acquisition settings are not
misreported as accepted publication rates.

The added request latency is explicit: .980–1.794 s source admission before
planning. First-command times also include existing matched controls and
independent physical-evaluator waits; they are research measurement latency,
not isolated production planner time. Per-request controller interval p50/p95/
p99/worst ms: corrected immediate 79.403/117.902/139.572/147.805; corrected .3 s
75.536/115.162/131.002/163.345; corrected .6 s 80.318/126.017/144.307/178.678.
These distributions use actual execution updates, excluding readiness waits.

## Optional transport inspection

After primary validation, inspected the saved fixed-time occupancy fields and
existing ProcessWorker serializer. The grid has 1,573,200 cells. States, observed
and expired use 1,573,200 bytes each; probabilities and last-observed times use
12,585,600 each. Total **29,890,800 bytes**, explaining almost all the prior
29.925 MB result. These five arrays are distinct, with different consumer/TTL
semantics; no redundant field was demonstrated that could safely be removed.

The serializer does make immutable-buffer copies, but this task did not establish
a measured safe cross-process copy-elimination correction. No field, precision,
immutable output, IPC protocol, dependency or source barrier was changed. Leave
transport code unchanged rather than make it a second architecture project.

## Review, preservation and reproduction

Production modification is solely
`robot_skill_stack/integrations/isaac/manipulation/motion/global_rrt.py`:
**71 added / 3 removed lines**. It adds bounded cooperative reconstructed-only
admission, pins service/publication identity at capture and rechecks payload/finger
contracts after waiting. Public motion/backend/skill/BT interfaces remain unchanged.
No manipulation executor, Lula, collision model/resource, mapping worker, geometry
algorithm, factory, shipped configuration, scene, camera or launcher changed.
Reconstructed representation remains explicitly opt-in; GT and occupied-only
have no new wait or source reads.

Research changes are limited to fixed ungated timing, request/source telemetry,
correctly archived selected inference and actual-start physical-veto measurement
in `autonomous_trial.py` and `benchmark_global_rgbd_v5.py`. Historical research preparation
remains available when the timing option is omitted; production admission is then
additional in that diagnostic mode. Application requests do not need it, and it
was never used by these six native trials. The production
strategy imports no research code and does not depend on this instrumentation.

Tests: **1424 full host tests PASS / 111.276 s**; **85 focused host tests PASS /
2.266 s** and the same **85 native-Python tests PASS / 2.440 s**. Architecture,
syntax and whitespace checks pass. Source/cancellation tests include stalled
updates, timeout, normal cancellation propagation, Stop/Play/reset/close, genuine
source/worker failure, failure recovered within one update, late rejected result,
service/revision mismatch, skipped accepted publications, selected-source expiry,
rejected/empty geometry with no exposure search, finger changes, zero pre-plan
commands and unchanged GT/occupied behavior. Existing floor/union/proxy, collision,
robot-model, skill/BT and source timestamp tests remain unchanged.

Published evidence:
[summary](../../research/rgbd_box_geometry/evidence/autonomous_admission/summary.json),
[validation](../../research/rgbd_box_geometry/evidence/autonomous_admission/validation.json),
[raw manifest](../../research/rgbd_box_geometry/evidence/autonomous_admission/raw-manifest.json),
[prediction seal](../../research/rgbd_box_geometry/evidence/autonomous_admission/predictions-seal.json),
[packet inspection](../../research/rgbd_box_geometry/evidence/autonomous_admission/packet-inspection.json)
and [preservation checks](../../research/rgbd_box_geometry/evidence/autonomous_admission/preservation.json).
Exact request timestamps use the existing monotonic clock; per-run receipts bind
UTC process start, command, commit and exclusive Kit/GPU preflight. Raw cameras,
robot snapshots, all request/worker/publication states, plan/evaluation curves,
logs and reduction scripts remain under
`outputs/motion-planning-v5/autonomous-admission-20261114`; large raw files stay
outside Git. All six native processes exited normally with all harness contracts.

Reproduction, fresh outputs and one Kit at a time:

```sh
uv run --offline --no-project --python 3.12 --with numpy==1.26.4 \
  --with py_trees==2.6.0 --with scipy==1.15.3 \
  python -m unittest discover -s tests/unit -t . -q
python3 scripts/check_architecture.py

docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.autonomous_trial --scenario partial \
  --request-delay 0.3 \
  --output outputs/motion-planning-v5/autonomous-admission-20261114/NEW/attempt-1
```

Original `isaac51` stopped/exit-137 state restored, no GPU compute process;
Config/HostConfig/Mounts hash remains
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`. No active
stream was interrupted, and no SDK/network/VM/container configuration changed.
Preservation confirms 48 original published research files, 16 static/v43
input/reference files, old native sets (24/21/21/29 files), 242 autonomous-integration
raw files, 112 performance raw files and all three prior reports unchanged.
Original V5 remains `71c07f815410a308dc7ea176148c017055c7fd7a`; recovery reference
remains `edc1b467c6bf03c192d0d9dccce2b538309a024f`.

## Demo-readiness recommendation

`RGBD_DEMO_CANDIDATE` under the tested original nominal partial conditions.
Normal requests no longer need research-specific publication timing: the strategy
owns bounded admission, and three independent ungated timing cases executed safely
under the existing physical-veto test. Baseline 1/3 improved to corrected 3/3.

Keep the opt-in status, timeout/refusal messages and validated GT option. These
are six independent warmed-up Kit runs, not prolonged GUI operation, cold-source
startup, all scenes or repeated requests in one persistent session. Minimum
32.106 ms freshness margin remains narrow, mapping throughput is unresolved,
and established floor/bound/budget scene limitations remain. No general payload,
dynamic-obstacle, continuous safety or real-robot claim follows. Review the
recorded nominal candidate before thesis use; stop after publication, without
GUI work or another optimization/reconstruction/worker direction.
