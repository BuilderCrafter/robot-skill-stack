# Autonomous RGB-D performance and freshness recovery

Running engineering record, 2026-11-14 UTC. Start 18:09 UTC; bounded budget
45–90 minutes, stop development by 19:39 UTC at the latest. Continue on
`feature/rgbd-rrt-autonomous` from clean, fetched remote/local HEAD
`99be1540b98986d8836c9238247f060b4216a74b`. Original V5 remains unchanged.

**Final status: `FUNCTIONAL_BUT_PERFORMANCE_LIMITED`.** A 15-line production
optimization recovered fresh autonomous partial planning and execution in **3/3
identical native trials**, versus the previous **0/4 fresh captures**. Every motion
passed the independent physical veto, reached the pose with 0.875 mm TCP error
and zero reported contacts. Mean accepted mapping rate rose from .531 to .984 Hz;
that remains below 1.8 Hz. This is a working experimental path, not demonstrated
production readiness or long-run reliability.

## A — profile and optimization decision

Reviewed AGENTS, the integration report and twelve retained native measurements,
the successful research postfix proof, V5 documentation and actual worker,
publication, capture, numerical and harness code. Container `isaac51` initially
stopped/exit 137, original official 5.1.0 image, no GPU compute process or stream.
No native matrix replay before profiling.

Baseline partial: 0/4 fresh authoritative captures, 1005.905–1062.561 ms original
source ages; accepted mapping .442–.576 Hz. Selected inference 284–319 ms,
combined compute approximately 560 ms, transport roughly 145–190 ms and queued
work consume most freshness margin. Shared mapping camera-batch access typically
.18 ms (already acquired semantic buffers); semantic batch acquisition about
20 ms. These are consumer timings, not camera hardware exposure latency.
Publication-to-readiness/request adds existing owner/update and harness delay.
The prior harness already deferred large report serialization and GT hashing;
its corrected partial still failed, initially reaching readiness at age 997 ms.

New bounded offline profile: use only hash-verified partial attempt-4 camera and
robot payload (20,411,990 bytes), original scene workspace/support settings and
original algorithm. Three fresh worker computations with cProfile: 507–552 ms.
One decomposition: inference 301 ms, occupancy/ray work 146 ms, extraction 29 ms,
immutable snapshot 26 ms; inference includes voxel grouping 106 ms, clustering
62 ms, robot envelopes ~10 ms and numerical qualification ~65 ms. Result payload
29,924,959 bytes. Input encode 9–26 ms/decode 5–6 ms; result encode 39–78 ms/decode
6–19 ms. These local serialization microtimings exclude pipe wait/scheduling;
previous native worker-minus-compute is the combined transport residual, not a
separately measured pure wire latency. Full timings and CPU profile are retained.

**Choose Option A, one numerical implementation optimization:** voxel grouping
currently performs expensive structured three-column integer sorting via
`np.unique(axis=0)`. Encode the same three integer coordinates into a checked,
collision-free mixed-radix int64 key, preserving lexicographic group order and
input-order bincount accumulation. Fall back to the original path if the key
range would overflow. No voxel size, points, cap, bins, thresholds, model or
qualification predicate changes. This is representation/ordering optimization,
not removal of geometric work or new shape inference.

An isolated scratch prototype produced exactly equal voxelized arrays for five
cloud sizes through 100,000 points and exactly equal centers/dimensions/rotations/
yaw for both partial models. Four baseline inference runs had median 281.256 ms;
four candidate runs median 220.263 ms (21.7% lower). Projection/fusion/clustering
fell from ~199–210 to ~117–131 ms. This is promising, not native evidence.

Estimated production diff: approximately 20 lines in the existing SDK-free cloud
voxel utility, plus focused equivalence/overflow/archived regressions. The shared
semantic path calls the same utility, so exact output equivalence and existing
semantic regression tests are required. Production research/visibility files
stay unchanged; original frozen research files/evidence remain immutable.

Rejected request-scoped Option B for this checkpoint: normal GlobalRrtStrategy
capture is synchronous. Correct per-request use of the existing bounded worker
would require asynchronous admission/waiting, a request token through replaceable
pending work, cancellation/reset/rejection barriers, and exactly-once inference
in the child to prevent two queued jobs reconstructing one request. Binding the
accepted product to its corresponding immutable occupancy remains possible, but
is materially larger than this measured local sorting correction. Do not combine
that architecture with Option A. Scheduling/worker/process contracts remain
unchanged and continuous inference still runs once per accepted mapping compute.

Chronological decisions:

- 18:09–18:13: verified clean branch/remote and original stopped environment;
  profiled archived payload before any Kit run. Identified structured voxel
  sorting as avoidable cost rather than blaming all time on the box model.
- 18:13: select reversible Option A after exact numeric scratch comparisons.
  Native throughput/freshness improvement remains to be established. No threshold,
  resource, worker or source-selection policy change is authorized by this choice.

## Validation plan

Before native: exact voxel grouping/cap/order equivalence, negative/extreme
coordinates and overflow fallback; archived box/face/robot/FREE evidence
comparison; existing mapping/frozen-source/lifecycle/cancellation/configuration,
ordinary GT/direct/skills/BT and semantic regressions; architecture checks.
Then original partial through normal autonomous runtime plus independent physical
veto. If a valid planning world is reached, run up to two identical repeats,
recording every result. One GT native control next; secondary geometry only if
primary path works and budget remains. No seed/fixture/threshold search.

## B — minimal implementation and offline coverage

Implemented checked integer-key grouping inside the existing `voxelize` function:
**15 added / one removed production line, one file**. Ordinary mapping/source
capture, continuous-inference scheduling, immutable publications, generation and
camera association, failure propagation, cancellation, reset, Stop/Play and
shutdown contracts are unchanged. No extra reader or worker is involved. Existing
inference and all six-face/visibility/robot/FREE guards execute unchanged.

New focused tests compare exact output arrays over 45 randomized size/voxel/cap
combinations (including duplicates/nonfinite observations), negative offsets,
voxel boundaries, degenerate axes and checked overflow fallback. Eight original
archived scene inputs are hash-verified and replayed under both old structured
and new integer grouping: every status, reason, numeric model, face witness,
FREE contradiction check and robot diagnostic is exactly equal (timing fields
excluded). No physical reference is used by these comparisons. The existing
trusted successful partial numeric test also still passes.

**24 focused tests PASS in 6.674 s** before native work. Full suite and architecture
checks follow; exact commands and logs are retained under
`outputs/motion-planning-v5/autonomous-performance-20261114`.

- 18:13–18:16: implemented Option A, kept old structured sorting only for integer
  range overflow, and validated exact archived evidence rather than geometry
  tolerance alone. No numerical qualification rule changed. Container startup
  reuses the saved configuration after verifying it was stopped with no GPU job.

Full package-qualified host discovery: **1,401 tests PASS in 110.922 s**.
Architecture boundary and Python syntax checks PASS. Existing historical fixture
expectations were unchanged, including GT/direct, semantic old/new equivalence,
source and robot-model invalidation, mapping worker transport, frozen publication,
reset/cancellation/shutdown and skills/BT tests. The four new tests add exhaustive
integer-key and eight-scene numeric/evidence equivalence, rather than changing
existing baselines.

Native Python 3.11 / NumPy 1.26.4 subset: **69 tests PASS in 10.758 s** before
Kit startup, including archived evidence equivalence, actual subprocess worker
transport and frozen-source/global configuration regressions. One owned Kit/GPU
instance only; immutable per-attempt directories, internal 600 s timeout plus
650 s host bound and per-run preflight/command receipts. All selected native
source/IK capture paths remain the committed production paths.

## C — native validation (running)

- 18:18–18:19 partial attempt 1: **PASS_PLANNING_AND_EXECUTION** through normal
  runtime capture and ordinary selected Lula IK. Two boxes qualified; inference
  **216.892 ms**. Original acquisition age at authoritative capture **906.868 ms**,
  headroom **93.132 ms**. Combined camera world 24 proxies plus support Plane.
  Three-waypoint selected route, 1.057749 rad deviation from matched direct joint
  segment; independent selected route/spline accepted. Same-packet occupied-only
  and support-only controls independently vetoed physical collisions; GT control
  accepted. Actual selected execution: **110 actions**, **1.107708 rad** maximum
  joint displacement, **0.874674 mm** final TCP error / **0.00242493 rad** orientation
  error, zero contact/proximity reports, strict success. No source or GT goal
  injection. Proceed with exactly two identical repeats, preserving failures.
- 18:19–18:21 partial attempt 2: same unchanged conditions, **PASS_PLANNING_AND_EXECUTION**.
  Two qualified boxes, inference **232.169 ms**, original capture age **766.312 ms**,
  headroom **233.688 ms**. Same 24 camera proxies, independent selected physical
  acceptance, genuine detour and 110 executed actions; zero reported contacts.
  No correction or source-selection change between attempts. First two requests
  both fresh and successful; finish the third declared repeat before assessing
  repeatability. Checkpoints A/B pushed normally to the existing branch.
- 18:21–18:22 partial attempt 3: **PASS_PLANNING_AND_EXECUTION**, source age
  **748.799 ms**, headroom **251.201 ms**, inference **215.962 ms**. Same qualified
  geometry, 24 camera proxies, independent selected acceptance, genuine detour,
  110 actions, 0.874674 mm / 0.00242493 rad endpoint error and zero contact reports.
  Selected tracking worst **0.036278 rad**, minimum sampled moving-arm solid
  clearance **41.143 mm**, final hold and solver opinion restoration passed.
  Declared primary repeat series complete: **3/3 fresh successful native motions**,
  compared with the previous **0/4 fresh captures**. No retries or corrections.
- 18:22: begin one original static GT native regression. Then perform only sparse
  as a different-obstacle guard check: its original selected union previously
  required 65 proxies with inference. Preserve that refusal if still necessary;
  do not drop evidence or attempt to rescue it. No broad matrix rerun.
- 18:22–18:23 original static GT native regression: **success**, all harness
  contracts, independent physical acceptance, 164 actions, 0.941662 mm TCP error,
  0.00249680 rad orientation error and zero contact/proximity reports. Ordinary
  direct START setup succeeded as well. No GT/direct behavior regression found.
- 18:23–18:25 sparse attempt 1: box qualified in **154.619 ms**, but combined
  selected evidence still required 65 camera proxies. Explicit
  `planning_source_reconstruction` / `Combined inferred/observed world exceeds
  64 camera proxies`, **COLLISION_WORLD_FAILURE**, zero selected actions. This is
  the preserved budget guard; no retry or evidence removal.
- 18:25–18:27: sealed four sensor-only prediction files before numeric reference
  scoring. Reduced all five native measurements. Follow-up identical offline
  worker profile: median 452.561 ms versus 510.448 ms before (11.3% lower);
  input/result serialization sizes unchanged at 20.412/29.925 MB. Integer grouping
  uses 12.802 MB peak traced temporary allocation versus 14.504 MB structured,
  with exactly equal output for 100,000 points. This is utility allocator tracing,
  not whole-process RSS. Restored original stopped container and configuration.
  No further production implementation or native run is needed for this option.

- 18:30–18:38 final review: validated all three selected route/spline and native
  execution gates, matched controls, unchanged capture/goal flow, physical vetoes,
  cleanup and remaining source/resource limits. Verified original package/V5
  artifacts plus all 242 preceding autonomous raw files unchanged. Bound 112 new
  raw files and checked report links. Publish the reviewed checkpoint; no further
  native or production work after publication.

## Native results and comparison

| Original partial trial | Inference ms | Authoritative source age ms | Headroom ms | Accepted mapping Hz | Selected outcome |
|---|---:|---:|---:|---:|---|
| 1 | 216.892 | 906.868 | 93.132 | 1.191 | planning + physical veto + execution passed |
| 2 | 232.169 | 766.312 | 233.688 | .894 | planning + physical veto + execution passed |
| 3 | 215.962 | 748.799 | 251.201 | .867 | planning + physical veto + execution passed |

All three used the same original START/GOAL, Post/Occluder, two 640×480 cameras,
5 Hz semantic / 2 Hz mapping acquisition settings, 1024 shared rays, .01 m voxels,
64 camera proxies, enhanced Franka model, seed 123456 and 10,000 iteration limit.
The first new accepted publication was selected without inspecting qualification;
there was no exposure search, retry, GT goal transfer or source callback injection.
Each request captured one immutable source and performed zero active-world
refreshes. Original generation, revision, timestamp, camera token/calibration and
synchronized robot snapshots match the retained input and inference product.

Each selected world contained two qualified boxes plus 22 exact occupied-cell
proxies (24 camera proxies), preserving all original 64 selected cuboids: 42
fully subsumed by conservative eight-corner containment, 22 retained verbatim.
There were no selected floor cells in partial; the allowlisted Plane remained.
The unchanged inferred model contained both independently copied physical
references and had no complete measured FREE cell strictly inside its interior.
No geometry, thresholds, uncertainty bands or collision-model resource changed.

Each selected ordinary Lula plan had three waypoints, 183 route samples, 334
trajectory samples and 1.274674 s trajectory duration. Matched controls received
the joint goal **from the selected ordinary IK result**, after it existed.
Selected/GT controls passed independent sampled physical evaluation. Same-packet
occupied-only and support-only plans were independently vetoed for panda-hand /
Post intersections; neither executed. In trial 1, verified positive intersection
ball radii were .023432 mm (occupied-only) and 1.491023 mm (support-only).

The selected route deviated **1.057749 rad** from the direct joint segment with
matched start/end arrays, and that reference segment was independently obstructed.
Thus the detour is supported by physical obstruction and acceptance, rather than
route length alone. This establishes improvement over the same-packet occupied
control under this sampled test, without claiming general unseen-volume safety.

Actual selected motion in every trial: 110 recorded actions (109 timed updates
plus hold), **1.107708 rad** maximum joint displacement, **.874674 mm** final TCP
position error, **.00242493 rad** orientation error, **.036278 rad** worst tracking
error, **41.143 mm** minimum sampled moving-arm cooked-solid clearance. Zero
contact/proximity points or penetrating pairs, no contact-buffer saturation,
complete physics-step sampling, final hold, zero final applied arm velocity and
solver session-opinion restoration. Independent checks are sampled numeric
checks, not continuous swept-volume or real-robot safety certification.

GT static regression: successful ordinary selected GT execution, 164 actions,
.941662 mm / .00249680 rad endpoint error, zero reports, physical acceptance and
normal cleanup. Sparse: one qualified inferred box, but **65 camera proxies
required**; explicit budget rejection with zero selected commands. Exact capture
age is not returned on this assembly error, although the unchanged initial
source check passed before assembly. No other native scenes or seeds were run.

| Partial comparison | Previous autonomous (4 trials) | Optimized (3 trials) |
|---|---:|---:|
| Fresh source / selected execution | 0/4 / 0/4 | 3/3 / 3/3 |
| Authoritative source-age range ms | 1005.905–1062.561 | 748.799–906.868 |
| Median source age ms | 1017.610 | 766.312 |
| Median selected inference ms | 294.456 | 216.892 |
| Median of per-run accepted-compute p50 ms | 567.601 | 467.356 |
| Accepted mapping range Hz | .442–.576 | .867–1.191 |
| Mean accepted mapping Hz | .531 | .984 |
| Mean semantic published / audit-window second | 1.707 | 1.998 |

Median selected inference fell **26.3%**; mean accepted mapping rate increased
**85.3%**. Rates use published-counter delta over the recorded pre-request mapping
window. Timing percentiles use accepted completions whose timestamps match their
views. Semantic normalization uses the whole camera-audit window including
setup/warmup; it is not a steady-state publication-rate guarantee. These are
small observational samples. The old baseline includes three original-harness
runs and one corrected-harness run; all new runs use the unchanged corrected
harness from the starting commit. No harness optimization was added in this task.

## Critical path after optimization and remaining limits

The follow-up profile used exactly the same archived payload. Whole worker median
fell **510.448 → 452.561 ms**, while input/result payloads remain **20.412/29.925 MB**.
The optimization changes sorting work, not diagnostic omission, geometry frequency
or transport semantics. Full result grids dominate byte volume; inferred-box
metadata is only a small fraction. Required per-camera density balancing and
fusion still run; no result is cached across different exposures.

New native measurements:

| Stage | Measured result |
|---|---|
| Shared mapping / semantic camera batch p50 | .182–.203 ms / 19.151–19.618 ms |
| Accepted combined compute p50 | 465.5–467.8 ms |
| Accepted reconstruction p50 | 216.0–225.1 ms |
| Transport residual p50 | 135.5–179.1 ms |
| Worker queue p50 | 118.4–153.8 ms |
| Worker completion to owner publication p50 | 28.7–43.4 ms |
| Chosen publication admission age | 812.124 / 701.216 / 676.937 ms |
| Chosen readiness age | 832.419 / 720.072 / 689.125 ms |
| Readiness to authoritative capture | 74.449 / 46.240 / 59.675 ms |
| Normal source adapter wall time | 32.9–43.6 ms |
| Selected ordinary planner wall time | 402.9–423.9 ms |
| Native RRT alone | 5.750–12.853 ms |
| Selected physical-evaluator wait | .962–1.038 s |
| Request to first command | 4.329–4.472 s, including all research comparisons |

Source age is the unchanged authoritative check; source timestamps were never
refreshed. Local input/result encode/decode profiles separate serialization from
other work, but native transport residual includes pipe waiting, copies and
scheduling. It is not a pure wire-time measurement. No new protocol instrumentation
or persistent worker was introduced to obtain a finer attribution.

Owner inference blocking was not added: the same existing mapping subprocess
runs the numerical work. The research observer still performs matched controls
and physical veto while preserving one simulation instant; its multi-second
pre-command wait is not production inference latency. Controller interval p50 /
p95 / worst ms was 76.195/124.742/147.550, 76.724/135.146/207.142 and
79.682/118.568/155.574. No long-run real-time responsiveness claim follows.

Mapping remains below 1.8 Hz, and semantic publication normalization remains below
4.5 Hz. The GT control without inference measured 2.117 Hz pre-request mapping
versus the previous 2.158 Hz control, with the same exact executed endpoint and
command count; no GT regression was found. Continuous inference plus roughly
30 MB result transport still costs more than the half-second mapping period.
A new asynchronous request lifecycle or broader transport optimization would be
larger work than this selected correction and was not combined with it.

## Validation, evidence and handover

Production changes: only
`robot_skill_stack/world/perception_multiview/clouds.py`, **15 added / 1 removed
line**. Test addition: `tests/unit/test_voxel_grouping_performance.py`. No change
to runtime/factories/configuration, mapping/inference/capture contracts, Lula,
executor, MoveToPose/Home/Pick/Place/BT interfaces, geometry reference/visibility
files, collision resources, scene/camera configuration or launchers. Default
occupied-only behavior remains numerically equivalent; shared semantic voxel
outputs also remain exactly equivalent. No new dependency, worker, camera,
resource budget or cloud assumption.

Final verification: **1,401 full host tests passed in 110.922 s**, **69 tests under
native Python 3.11 / NumPy 1.26.4 passed in 10.758 s**, architecture/syntax and
whitespace checks passed. Tests cover existing source age, exposure/calibration,
generation/reset/Stop/Play/cancellation/late-completion/worker-shutdown barriers,
empty/rejected/multiple geometry, FREE/UNKNOWN immutability, floor/union/budget
rules and real robot-model invalidation. No historical fixture/hash expectation
was changed. Exact array/geometry/evidence equivalence is additionally checked on
all eight prior sensor captures, including previously rejected scenes.

[Comparison](../../research/rgbd_box_geometry/evidence/autonomous_performance/comparison.json),
[numeric native evidence](../../research/rgbd_box_geometry/evidence/autonomous_performance/summary.json),
[validation](../../research/rgbd_box_geometry/evidence/autonomous_performance/validation.json),
[before CPU profile](../../research/rgbd_box_geometry/evidence/autonomous_performance/cpu-profile.txt),
[after CPU profile](../../research/rgbd_box_geometry/evidence/autonomous_performance/cpu-profile-after.txt)
and [raw manifest](../../research/rgbd_box_geometry/evidence/autonomous_performance/raw-manifest.json)
bind the measurements. Raw frames, robot inputs, exact native commands, source
hashes, all controls/physical curves and profiling/reduction scripts remain in
`outputs/motion-planning-v5/autonomous-performance-20261114`; large raw files are
not committed. The original research/autonomous reports and evidence are intact.

Reproduction, sequentially from repository root after exclusive Kit/GPU preflight:

```sh
uv run --offline --no-project --python 3.12 --with numpy==1.26.4 \
  --with py_trees==2.6.0 --with scipy==1.15.3 \
  python -m unittest discover -s tests/unit -t . -q
python3 scripts/check_architecture.py

docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.autonomous_trial --scenario partial \
  --output outputs/motion-planning-v5/autonomous-performance-20261114/partial/NEW-ATTEMPT
```

All five native measurement contracts passed and processes exited normally.
The original stopped/exit-137 `isaac51` state was restored with identical
Config/HostConfig/Mounts hash
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918` and no GPU
compute process. No active stream was interrupted. Preservation checks confirm
48 original published research files, 16 archived static/v43 inputs/reference
files, previous raw artifact sets (24/21/21/29), and **242 previous autonomous
raw files** unchanged. The original V5 branch remains at
`71c07f815410a308dc7ea176148c017055c7fd7a`; recovery ref remains
`edc1b467c6bf03c192d0d9dccce2b538309a024f`.

**Recommendation:** retain this small optimization and the opt-in autonomous
feature as `FUNCTIONAL_BUT_PERFORMANCE_LIMITED`. It answers fresh qualified
normal-runtime planning positively in three fixed nominal trials, with genuine
physical obstacle avoidance and successful execution. It does not meet the full
mapping/semantic performance targets or establish broad-scene/long-run demo
reliability. Keep the validated GT path available for thesis work. Review these
results before any separate request-scoped/transport task; no additional
architecture, GUI work or optimization campaign is started here. Stop after
publishing this checkpoint, earlier than the maximum budget because primary
freshness recovery and its bounded repeatability check are complete.
