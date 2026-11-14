# Autonomous reconstructed RGB-D global RRT integration

Running engineering record, 2026-11-14 UTC. Starting V5 commit:
`71c07f815410a308dc7ea176148c017055c7fd7a`. Development branch:
`feature/rgbd-rrt-autonomous`. The original V5 branch and its successful research
proof remain unchanged. Sprint budget: approximately two hours, starting 16:20
UTC; implementation/testing stops by 18:20 UTC, followed only by handover.

**Final status: `BLOCKED_BY_RELIABILITY`.** The opt-in normal-runtime integration
is implemented and regression-tested. All eleven selected reconstructed RGB-D
native requests safely refused; none executed or demonstrated autonomous RRT
avoidance. All four partial captures expired despite qualifying both boxes.
The original GT native control succeeded. Keep the validated GT demonstration
path; this feature remains experimental and unsuitable for the demonstration.

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

Checkpoint B status was `PARTIALLY_INTEGRATED`; no production readiness claim.

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
- 17:04–17:05, corrected partial attempt 4: both boxes qualified in 284.450 ms,
  but readiness age was already **997.227 ms**. Ordinary capture rejected stale
  and issued zero actions. Deferring test serialization/hash work cannot fix a
  source delivered with 2.773 ms initial margin. Correction did not establish
  success; stop this root cause. All four partial runs remain failures. Production
  remains unchanged from checkpoint B. No additional algorithm/worker proposed.
- 17:06: all eight native scenes have retained results. Begin one original static
  GT native regression with reconstruction disabled and independent physical
  veto. This checks fallback compatibility and observes the original worker load;
  it is not a substitute for a successful selected RGB-D run. Freeze hashes of
  the eleven completed autonomous prediction JSONs before any new independent
  reference scoring (eight initial scenes plus three partial repeats).
- 17:06–17:07, original static GT regression: ordinary native motion succeeded,
  independent route/spline evaluation allowed it; **164 commands**, **0.941662 mm**
  TCP position error, **0.00249680 rad** orientation error, **1.351180 rad** actual
  joint displacement, **28.213 mm** minimum sampled moving-arm clearance, zero
  contact/proximity reports, zero final applied arm velocity, final hold and
  cleanup. Direct START setup also succeeded in all applicable matrix scenes.
- 17:07–17:14 final regression environment corrections: full discovery inside the
  unchanged Isaac image ran 1,372 tests with four errors because the image lacks
  `git`, needed by historical baseline-reading tests. Host Python 3.12 ran all
  1,397 but 13 research collision-evaluation tests lacked SciPy. These are explicit
  environment errors, not assertions or altered fixtures. Rerun using the cached
  pre-existing test dependencies NumPy 1.26.4, py_trees 2.6.0 and SciPy 1.15.3,
  offline. No production requirement, SDK or container installation is changed.

- 17:15–17:22 final review: complete cached-dependency host suite passed all
  1,397 tests; architecture/syntax passed after adding the evidence reducer.
  Verified every native result, command count, source-capture count and original
  11 cooked-body fingerprints. Restored the original stopped container and
  verified its configuration hash and exit state. Historical package, raw captures,
  V5 branch and recovery ref checks passed. Freeze 242 new raw files into the
  publication manifest and publish the reviewed integration branch. No further
  native or production changes. Final status remains `BLOCKED_BY_RELIABILITY`.

## Final native matrix

Eleven reconstructed requests, eight original scenes, one fixed seed (123456),
no fixture/camera/threshold changes. Every run retained all five harness contracts
(runtime closed, source unchanged during run, no internal runtime error, all 11
cooked robot bodies available, static scene unchanged). Harness `PASS` denotes a
valid retained measurement; the outcome below is the selected motion outcome.
No selected RGB-D request issued an arm command or advanced its physics instant.

| Scene / attempt | Selected inference | Inference ms | Authoritative age ms | Camera proxies* | Outcome / first failure |
|---|---|---:|---:|---:|---|
| partial / 1 | 2 qualified boxes | 318.510 | 1028.465 | 24* | SOURCE_FAILURE: stale |
| partial / 2 | 2 qualified boxes | 286.663 | 1006.755 | 24* | SOURCE_FAILURE: stale |
| partial / 3, timing probe | 2 qualified boxes | 302.249 | 1005.905 | 24* | SOURCE_FAILURE: stale |
| partial / 4, corrected harness | 2 qualified boxes | 284.450 | 1062.561 | 24* | SOURCE_FAILURE: stale |
| empty / 1 | empty, robot components excluded | 124.546 | 789.946 | 64 | EXPECTED_SAFE_REJECTION: base geometry |
| static / 1 | 1 qualified box | 186.548 | 978.309 | 46 | EXPECTED_SAFE_REJECTION: base geometry |
| large / 1 | rejected extra unsupported components | 353.781 | 1010.031 | — | SOURCE_FAILURE: stale; inference also rejected |
| narrow / 1 | rejected hidden bound | 258.550 | passed <1000; not retained | — | RECONSTRUCTION_FAILURE |
| v43 / 1 | 1 qualified box | 154.247 | 836.134 | 63 | EXPECTED_SAFE_REJECTION: base geometry |
| sparse / 1 | 1 qualified box | 167.793 | passed <1000; not retained | 65 required | COLLISION_WORLD_FAILURE: proxy budget |
| verified_occlusion / 1 | rejected visible patch/residual | 295.528 | passed <1000; not retained | — | RECONSTRUCTION_FAILURE |

Asterisk: partial's 24-proxy assembly was checked **offline after rejection**,
not admitted as a live source. Original 64 selected cells partition into 42 fully
subsumed cells plus 22 exact retained cells; add two inferred boxes. Static
retained 45 cells (28 protected floor), subsumed 19, added one box. V43 retained
62 cells (57 protected floor), subsumed two, added one box. Empty retained all 64
floor proxies. Sparse needs all 64 original proxies plus its box. The separately
allowlisted support Plane is additional to the camera budget in every world.

Failed reconstruction assembly does not return its partial frozen capture, so
exact authoritative ages for narrow/sparse/verified-occlusion are unavailable;
reaching that stage proves the existing initial source check passed. The age
column is the unchanged authoritative frozen-source check, not an assertion of
age at planner completion. Adapter wall time for admitted worlds was 31–51 ms,
including known-support conversion and metadata construction. No timestamps were
refreshed. No stale archive was offered as a fresh source.

No selected RGB-D route exists: ordinary IK/RRT success, paired route improvement,
detour, physical route/spline acceptance, execution latency, tracking and endpoint
accuracy remain **unestablished**, not zero-error successes. The three base
rejections occurred before IK. The selected planner receives no GT goal; matched
controls would obtain their joint goal from a successful selected route. They
were consequently not run for failed selected plans. The separate successful GT
regression is a compatibility control, not an RGB-D improvement measurement.

## Geometry and source evidence

[Numeric evidence](../../research/rgbd_box_geometry/evidence/autonomous_integration/summary.json)
contains original camera tokens, timestamps, intrinsics/transforms, model centers,
yaw/rotation, dimensions, all six face bounds, calibrated transition witnesses,
qualification reasons, union audits, raw result categories and timing summaries.
[Prediction seal](../../research/rgbd_box_geometry/evidence/autonomous_integration/predictions-seal.json)
fixes all eleven sensor-only prediction files before independent numeric scoring.

Post-hoc physical-reference names below are evaluation labels only. Production
saw raw calibrated camera observations, synchronized robot geometry and whitelisted
workspace/support bounds; no fixture identities, dimensions, physical references
or tracked semantic objects entered reconstruction or selected planning.

| Evaluation match | Center m (x,y,z) | Dimensions m (x,y,z) | Max extent error mm | Contains physical reference |
|---|---|---|---:|---|
| partial Occluder | .787552, -.300554, .529049 | .088897, .405109, .405904 | 6.897 | yes, without allowance |
| partial Post | .450000, -.000413, .349335 | .094001, .124827, .265330 | 3.330 | yes, without allowance |
| static Post | .450000, -.000413, .349335 | .094001, .124827, .265330 | 3.330 | yes, without allowance |
| sparse ThinPost | .449999, -.000498, .349387 | .039001, .064998, .205227 | 3.227 | yes, without allowance |
| v43 box | .550000, -.225314, .299528 | .064001, .064630, .124947 | 2.946 | yes, without allowance |

All reported yaw estimates are effectively zero (-1.94e-18 rad). All four partial
runs produced exactly identical numeric box geometry. For partial's Occluder,
+x,+y,+z have direct patches; -x,-y,-z use calibrated occupied/free boundary
witnesses. The Post, static, sparse and v43 boxes have direct ±x,+y,+z patches;
-y and -z use boundary witnesses. All six bounds qualified under the unchanged
single static convex yaw-box model. This empirical model is not a general shape
or hidden-volume guarantee. Rejected scenes received no guessed hidden bound.

Every accepted model had zero measured FREE-ray contradiction, and independent
archived grid scoring found **zero complete FREE cells strictly inside** any
accepted model. Near-boundary FREE cell centers are separately recorded as grid
alias diagnostics. Robot-envelope components were excluded; mixed robot-connected
components are rejected by tests. The runtime map's original FREE/OCCUPIED/UNKNOWN
arrays and bounded extractor's selected evidence remain unchanged. This preserves
all originally **selected** cells, not every supported cell omitted by the
pre-existing 64-proxy extractor.

## Performance and reliability

Rates below use pre-request observation windows. Mapping rate is the published
counter delta over the recorded mapping-audit interval. Semantic rate is published
count divided by its whole camera-audit interval (including setup/warmup), not a
steady-state 5 Hz claim. Timing percentiles use matched accepted completions;
rejected/old completions are excluded from those percentiles.

| Scene / attempt | Accepted mapping Hz | Semantic published / audit s | Approx semantic Hz | Combined compute p50/p95 ms | Publication age p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| partial / 1 | .570 | 35 / 20.007 | 1.749 | 559.9 / 597.1 | 896.0 / 955.0 |
| partial / 2 | .442 | see numeric evidence | — | 573.7 / 597.7 | 907.4 / 946.6 |
| partial / 3 | .576 | see numeric evidence | — | 571.2 / 640.2 | 904.8 / 958.1 |
| partial / 4 | .538 | see numeric evidence | — | 564.0 / 609.0 | 871.1 / 982.4 |
| empty | 1.566 | 43 / 13.358 | 3.219 | 403.5 / 437.5 | 714.0 / 917.8 |
| static | 1.018 | 33 / 14.948 | 2.208 | 451.6 / 476.2 | 884.8 / 992.2 |
| large | .497 | 38 / 22.609 | 1.681 | 576.6 / 666.1 | 931.8 / 977.0 |
| narrow | .630 | 34 / 18.826 | 1.806 | 524.6 / 577.6 | 868.8 / 963.6 |
| v43 | 1.325 | 15 / 9.425 | 1.592 | 376.2 / 499.5 | 796.8 / 958.7 |
| sparse | 1.166 | 31 / 12.946 | 2.395 | 438.3 / 503.5 | 834.3 / 907.2 |
| verified occlusion | .589 | 25 / 19.546 | 1.279 | 552.9 / 620.3 | 889.2 / 972.8 |
| static GT control, inference disabled | 2.158 | 28 / 12.260 | 2.284 | 276.8 / 319.2 | 552.2 / 878.0 |

Partial's selected inference cost was 284–319 ms. Mapping alone typically cost
about 250–280 ms; transport and queued work consumed further margin. One measured
partial decomposition was 198 ms projection/fusion/clustering, 105 ms numerical
qualification and 11 ms robot envelopes. Work runs in the existing mapping
subprocess, not on the Kit owner. No obvious duplicate reconstruction or cheap
correction to this combined throughput was demonstrated. No broad transport or
mapping redesign was attempted to rescue the sprint.

Reconstruction significantly reduces accepted mapping throughput in these tests;
none of the reconstructed cases reached the 1.8 Hz target. The same-scene static
GT control's pre-request rate was 2.158 Hz without inference versus 1.018 Hz with
inference. This is one observational comparison, not a statistical performance
study. Semantic publications also miss the 4.5 Hz target in the GT control, so
that shortfall cannot be attributed solely to the new inference work. Acquisition
configuration and advancing camera exposure budgets remain 5 Hz semantic / 2 Hz
mapping with two 640×480 cameras, 1024 shared mapping rays and .01 m voxels.

RGB-D controller distributions are unavailable because execution never started.
GT control's controller wait p50/p95/p99/worst was 74.759/126.056/155.728/175.372 ms.
Request-to-first-command was 2.194 s, including about 1.260 s independent research
physical-evaluator wait. Its Kit interval worst 2.284 s includes research planning
and evaluator blocking; it is not production inference latency. The RGB-D feature
has no demonstrated execution responsiveness or long-run reliability claim.

## Final validation, boundaries and reproduction

Final complete host discovery: **1,397 tests PASS in 107.221 s**. The unchanged
Isaac Python integration subset: **65 PASS in 5.269 s**; corrected-harness geometry
and goal-flow subset: **20 PASS in 1.429 s**. Architecture and Python syntax pass;
no historical fixture/hash expectation was modified. Earlier environment failures
and corrected commands/logs remain retained and hashed.

The production diff is **666 added / 14 removed lines across ten files**, including
453 lines copied unchanged from frozen numerical reference/visibility code.
The remaining 213 added lines provide the opt-in product and existing seams:

- `world/spatial_mapping/box_geometry/{reference,visibility}.py` and its package
  initializer: SDK-free numerics, AST-equivalent after module docstring.
- `world/spatial_mapping/reconstruction.py`: immutable product, qualification,
  source identity and conservative observed-union assembly.
- `world/spatial_mapping/runtime.py`: optional same-completion worker product,
  owner publication and reset invalidation.
- `integrations/isaac/{config,observation_runtime}.py`: opt-in validation and reuse
  of the existing mapping process; default inference work remains zero.
- `integrations/isaac/manipulation/{factory,motion/global_rrt,motion/frozen_rgbd}.py`:
  pass the representation choice and translate into existing numeric obstacles.

No Lula implementation, collision resources, trajectory executor, manipulation
skill semantics, scene/config fixture, GUI or launcher was modified. No new
persistent service/worker, model dependency or resource allocation was added.
Existing cancellation, Stop/Play/reset, ownership, failure and shutdown tests
pass; old source/geometry cannot be selected after invalidation. Observer USD
getter exclusions remain intact; real robot-model changes still invalidate the
source through the existing robot-filter contract.

Production is independent of research. The new experimental native entry point
is `research.rgbd_box_geometry.autonomous_trial`; its planner observer only
records comparisons and vetoes unsafe/unresolved plans. It never injects a source
or selected joint goal. `reduce_autonomous.py` reduces the fixed retained sprint
artifacts with the original independent scoring utilities; it is not runtime code.

Commands (from repository root; each native output must be new):

```sh
# Native, sequential, after clear container/Kit/GPU preflight:
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.autonomous_trial --scenario partial \
  --output outputs/motion-planning-v5/autonomous-integration-20261114/partial/NEW-ATTEMPT

# Complete offline suite; cached existing research-test dependencies:
uv run --offline --no-project --python 3.12 --with numpy==1.26.4 \
  --with py_trees==2.6.0 --with scipy==1.15.3 \
  python -m unittest discover -s tests/unit -t . -q
python3 scripts/check_architecture.py

# Offline artifact reduction in the unchanged native Python, without Kit:
./run_isaac_python.sh -m research.rgbd_box_geometry.reduce_autonomous
```

Exact per-run commands, exits, original timestamps/input hashes, full camera
arrays, robot snapshots, complete diagnostics and rejected outcomes remain under
`outputs/motion-planning-v5/autonomous-integration-20261114`. The published
[raw manifest](../../research/rgbd_box_geometry/evidence/autonomous_integration/raw-manifest.json)
binds those files. All twelve native processes exited normally and all runtime
closures succeeded. The original stopped `isaac51` state (exit 137) was restored,
with unchanged Config/HostConfig/Mounts SHA-256
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918` and no GPU
compute process. No active stream existed or was interrupted.

[Preservation checks](../../research/rgbd_box_geometry/evidence/autonomous_integration/preservation.json)
confirm 48 original published package files, 16 original static/v43 input/reference
files, and previous raw artifact sets (24/21/21/29 files) unchanged. Local and
remote V5 remain `71c07f815410a308dc7ea176148c017055c7fd7a`; recovery ref remains
`edc1b467c6bf03c192d0d9dccce2b538309a024f`. Raw arrays and pickles are retained
locally rather than committed; compact numeric evidence and reproducible research
code are published on the dedicated branch.

## Handover

`BLOCKED_BY_RELIABILITY`, with a useful tested opt-in integration checkpoint.
Geometry reconstruction and observed-union accounting work on the qualified
captures; live delivery cannot reliably meet freshness/throughput in partial,
and other scenes retain existing floor, unsupported-bound or proxy-budget limits.
There is no newly demonstrated autonomous RGB-D route, detour or execution.

Use the validated static GT path for thesis demonstration and proceed with GUI
finalization after review. Any further RGB-D performance engineering should be a
separate explicitly reviewed task with a measured budget; do not weaken source,
geometry or collision criteria. No further implementation, native tests, merge,
GUI work or alternative planning architecture was started after this handover.
