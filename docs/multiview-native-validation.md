# Native multiview end-to-end validation

**Native Isaac execution occurred. Geometry passes; continuous perception
availability fails. The current configuration is not yet reliable for the
planned demonstration.** Production behavior remains unchanged.

| Validation | Status | Result |
|---|---|---|
| Native four-object geometry, graspability and IDs | PASS | Four correct; maximum center/dimension errors .286/.542 mm |
| Static availability / asynchronous runtime | FAIL | About 2 Hz publication; objects periodically become stale and invisible |
| Controlled arm approach and return | PASS for motion | All four moves reached within 10 mm |
| Target availability during approach | FAIL | Geometry/ID stay correct in publications, but stale intervals hide the target |
| Native freshness and lifecycle controls | PASS | Freeze/resume, mixed-frame rejection, in-flight Stop/Play and cleanup |
| GUI operation / complete Pick-and-Place | NOT RUN | Headless; approach only |

## Environment and method

Production checkout: clean `refactor/thesis-cleanup` at
`f2cfa9203edf3a9ce7bd837dc9444f5b1df547e7`, matching origin. Main remained
`5b61d4442d74e3afd76420ffc11e8e3ab0b21937`.

| Item | Actual environment |
|---|---|
| Installation | Existing `isaac51` container, `/isaac-sim`; no rebuild or installation |
| Isaac | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| GPU / driver | NVIDIA L4, 23,034 MiB / 580.178.04 |
| Python / NumPy | 3.11.13 / 1.26.4 from existing repository `.deps` |
| Launch | Unchanged `./run_isaac_python.sh`, `ISAAC_SIM=/isaac-sim`, headless |
| Scene | `scenes/multiview_perception_final.usda`, original sublayers |
| Configuration | `config/scenes/multiview_perception_final.toml` |
| Budgets | Two 640×480 cameras, stride 2, 2 mm voxels, 12,000-point cap, 24 fits, 5 Hz requested |

The container was stopped and GPU idle. Its repository mount points to the
research archive, so `git archive` of the requested HEAD was extracted into
`/tmp/thesis-multiview-native-f2cfa92`; only `.deps` was borrowed by symlink.
No archived project code was imported. The stage edit target was the session
layer, with no custom scene/camera edits. Authored file hashes stayed unchanged.
No launcher, SDK, dependencies or system configuration changed.

`tests/multiview/test_final_native.py` uses normal `build_runtime`: actual Isaac
cameras → SharedFramePool → ObservationRuntime → ProcessWorker → processor →
association → WorldModelUpdater → WorldModel. Test-only hooks count captures,
publication admission and visibility. SDK-free `native_probe.py` times original
compute stages in the actual child; its result is unwrapped before normal
publication. No synthetic inputs, source substitution, freshness extension,
sampling reduction or detector threshold adjustment was used. GT enters only
one-to-one outcome scoring and independent saved-depth analysis.

Two sessions ran sequentially. Run 1 exposed intermittent staleness, then a
**test-only** kinematics query after Stop interrupted lifecycle reporting. Its
cleanup succeeded. Run 2 fixed that stopped-state measurement, used established
asynchronous Stop/Play calls, confirmed the same freshness issue, and completed
arm/lifecycle checks. No third session or broad regression suite ran.

## Actual calibration and four-object results

Both SDK camera intrinsics have zero skew, principal point (320,240), bottom row
(0,0,1). Saved native records retain full ROS camera-to-world transforms:

| Camera | fx | fy | World position (m) |
|---|---:|---:|---|
| `/World/PerceptionCamera` | 554.256226 | 554.256287 | (1.658960, −.052813, 1.103581) |
| `/World/PerceptionCameraSide` | 732.999268 | 732.999268 | (.180000, .800000, .650000) |

These are native measurements, not the previous analytic lens approximation.
Normal camera initialization reported a vertical-aperture adjustment for square
pixels. Full intrinsics/transforms, actual render tokens and original monotonic
acquisition times are saved in result JSON and depth NPZs. Both token clocks
advance together; no artificial exposure times were supplied.

All five static published batches in run 2 consistently contain exactly four
objects, with no extras. Both cameras contribute to every object. All are fresh
at publication and retain their IDs across the static observations:

| Reference | Class | Center error mm | Max dimension error mm | Confidence | Graspable | ID |
|---|---|---:|---:|---:|---|---|
| Square | cube | .135 | .273 | 1.000000 | yes | `multiview_2` |
| Rectangle | cube | .116 | .234 | 1.000000 | yes | `multiview_1` |
| SmallSphere | sphere | .176 | .314 | .970753 | yes | `multiview_4` |
| LargeSphere | sphere | .286 | .542 | .948450 | yes | `multiview_3` |

Run 1's final static snapshot happened during a stale interval and showed no
visible objects, although its five published static batches all contained four
correct objects. The table scores actual fresh publications; it does not conceal
the availability failure: run 2 has **4/14 static samples with zero visible
objects**, at source ages 1.105–1.196 s.

## Asynchronous runtime measurements

The observation window spans 12 simulation seconds. Scheduling uses wall time;
published rate counts successful updater applications, not ticks. Camera counts
are captured buffer snapshots, not every exposure rendered internally by Isaac.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Wall / simulation seconds | 17.824 / 12.000 | 17.919 / 12.000 |
| Requested acquisitions per wall second | 5 | 5 |
| Successful camera snapshots | 178 | 180 (90 per camera) |
| Fresh synchronized batches / submitted jobs | 89 / 89 | 90 / 90 |
| Jobs started / completed, counter deltas | 35 / 35 | 36 / 36 |
| Pending jobs replaced | 54 | 54 |
| Completed results discarded by worker | 0 | 0 |
| Worker exceptions | 0 | 0 |
| Delivered results / successful publications | 36 / 36 | 37 / 36 |
| Published batches per wall second | **2.020** | **2.009** |
| Published batches per simulation second | 3.000 | 3.000 |
| Rejected publication preflights | 0 | 1, age 1.060 s |
| Camera admission errors in window | 0 | 0 |
| Visibility samples with zero / four objects | 19 / 65 | 18 / 66 |
| Source health stale samples | 22 / 84 | 21 / 84 |
| Worker duration p50 / p95 ms | 505.21 / 582.49 | 500.82 / 562.58 |
| Actual compute p50 / p95 ms | 495.23 / 573.03 | 490.99 / 549.69 |
| Transport/IO residual p50 / p95 ms | 9.70 / 11.66 | 9.87 / 12.32 |
| Publication frame age p50 / p95 ms | 648.48 / 770.38 | 633.43 / 707.07 |
| Maximum publication frame age ms | 903.04 | 713.21 |

Started/completed and delivered counts can differ by one because a job crosses
a measurement boundary. Run 2's stale rejection is an inherited job at window
entry after saving static artifacts paused the owner. Run 1 reproduces visibility
gaps without a rejected publication. A warm-up result was also correctly rejected
after initial discovery changed context revision 0→4; it is outside both windows.
Pending replacement is the intended bounded-worker policy, not itself a failure.

Run 2 stage timings, milliseconds:

| Stage | p50 | p95 |
|---|---:|---:|
| Projection, cropping and provider voxelization | 17.49 | 21.83 |
| Initial clustering | 10.03 | 11.08 |
| Refinement, inclusive | 44.24 | 51.17 |
| Metric graph, within refinement | 43.17 | 50.01 |
| Primitive fitting | **386.48** | **433.11** |
| Parameter finalization | 2.90 | 3.39 |
| Geometric observability | 19.93 | 27.32 |

Timings overlap where one stage calls another; percentiles are not additive.
Transport/IO is worker duration minus child compute, including serialization,
pipes and scheduling, rather than a pure transfer measurement. Updater application
has median .107 ms, maximum .203 ms. Association was not separately profiled.
Fitting accounts for around 79% of the static compute median and is the dominant
measured cost. No performance optimization was attempted.

The freshness failure follows original timestamps: results arrive about .63 s
after capture, then the next publication takes roughly .5 s. The previous
observation crosses the existing 1 s limit before replacement; cleanup hides all
four objects together. Every empty static/throughput/arm sample in run 2 has
stale source health and age >1 s. No worker exception or mixed-camera admission
caused those gaps. Maximum sampled throughput age was 1.458 s, including the
artifact boundary. This is a system-level demo blocker, not evidence that the
large sphere vanished from segmentation.

## Arm approach and native foreground observations

Before moving, the harness waits for all four confirmed native observations.
It selects the observed large sphere and uses the registered Pick skill's
existing grasp planner. Only its pre-grasp pose is executed, preceded by an
intermediate pose 120 mm higher. The normal backend RMPflow API and simulation
stepping execute every move. No joint teleport, grasp closure or object movement
is used. Initial TCP: (.597491,−.086039,.364492) m.

| Phase | Actual final TCP (m) | TCP error mm | Target missing / sampled states | Published batches |
|---|---|---:|---:|---:|
| Intermediate | (.534399,−.152441,.253597) | 9.66 | 4 / 30 | 13 |
| Pre-grasp | (.536435,−.156485,.143494) | 9.42 | 4 / 18 | 7 |
| Retreat | (.538084,−.157105,.246560) | 9.65 | 6 / 19 | 8 |
| Return to initial pose | (.591398,−.091308,.360000) | 9.22 | 5 / 21 | 10 |

Across all 38 published arm-phase batches, the target retains `multiview_3`,
sphere classification, graspability, .286 mm center error and .542 mm dimension
error. No additional confirmed robot object appears. All **19/88 target-missing
samples** are stale-source intervals (age 1.029–1.287 s), not target-specific
UNKNOWN fits. For example, a pre-grasp loss occurs at TCP
(.530247,−.148770,.230408), frame age 1.081 s, when all four objects disappear.
Full sampled joint vectors and TCP positions are retained.

Independent analysis of saved native static/pre-grasp/home depth finds the large
sphere in a separate **337-point coarse component**. All 337 samples match the
reference sphere within the 2.5 mm evaluation tolerance, with identical component
bounds in these captures. Native RGB also shows the target visible. This bounded
approach does not demonstrate target/robot coarse merging. It demonstrates stale
hiding while the target remains visible; fresh publication restores the same ID.

There is a separate **Square occlusion/fragmentation limitation**. Near pre-grasp,
the main-camera gripper obscures part of the square box. Square becomes UNKNOWN
and ungraspable in 5/7 pre-grasp publications and 3/8 retreat publications, then
returns to cube in all 10 home publications with its original ID. Its worst
UNKNOWN measured-bound dimension error is 17.84 mm. The first UNKNOWN is fresh
at publication (age .658 s), with reason `unresolved_components`. Its nearest
recorded acquisition-time TCP is (.528743,−.149248,.196275) m, sampled 88 ms
earlier; the full joint vector is saved.

Saved pre-grasp depth has 487 Square samples versus 570 statically; all match
box surfaces within 2.5 mm. The 4 mm graph retains [291,37,38] points across
components: **75.15%** coverage. Segmentation rejects this fragmented box. A
diagnostic whole-cloud fit returns cube, but that alone does not establish
hidden bounds, observability or grasp eligibility. No robot returns were found
in that captured coarse component. The evidence supports occlusion-driven
segmentation qualification loss, not robot merging. No native old-version
comparison ran, so this is **not a proven baseline regression**. No repair was
attempted during validation.

## Freshness, lifecycle and cleanup

Run 2 passes all bounded controls:

- Pausing both native cameras does not republish unchanged exposures; objects
  become invisible after the normal freshness limit. Resume publishes fresh data.
- With runtime ticks detached, a real Kit render advances between native camera
  reads. The pool rejects `Camera updated during acquisition`. No fabricated
  frame or token is used.
- Stop with worker work in flight increments its epoch and permits no stopped
  publication. Play resets camera clocks and recovers with new acquisition times.
  One late old-generation camera exposure is correctly refused before recovery.
- Camera callbacks, render products and ReferenceTime annotators close. The worker
  thread stops, child exits, and authored files remain unchanged.
- External inspection after shutdown finds only the original idle `sleep` process
  in the container and 0 MiB GPU usage. The container is returned to its original
  stopped state. No retained Isaac or worker subprocess was found.

## Evidence and reproduction

Ignored artifacts: `outputs/multiview-native-20261122/session-1/` (6.3 MiB) and
`session-2/` (16 MiB): result JSON, native logs, capture/publication events, states,
a few RGB/depth images and original native depth NPZs. Exact run-specific test
files are copied beside each session's evidence. Run 2 additionally contains
`native-depth-connectivity.json`, `square-fragmentation.json`, and
`failure-examples.json`. Large generated files are not committed. No synthetic
camera input was used for execution or diagnosis.

Run this branch with an existing installation and unused output paths:

```bash
ISAAC_SIM=/actual/isaac-sim ./run_isaac_python.sh \
  tests/multiview/test_final_native.py --headless \
  --result outputs/NEW-native-final/result.json \
  --artifacts-dir outputs/NEW-native-final
```

On this VM, that unchanged launcher ran via `docker exec` with working directory
`/tmp/thesis-multiview-native-f2cfa92` and `ISAAC_SIM=/isaac-sim`, under a 480 s
external timeout. Sessions finished in about 57/70 s including startup/shutdown.
Both result JSON files say FAIL; the SDK launcher returning exit code 0 does not
establish success. Validation comprised syntax checks and these bounded native
sessions, without a full suite.

## Recommended next action

Review a targeted performance/freshness recovery task before claiming a reliable
live demonstration. Preserve original acquisition timestamps and meaningful
freshness checks; do not simply increase their limit or reduce resolution to
hide gaps. Fitting is the measured primary cost. Separately review the captured
occluded-box fragmentation case before choosing any narrow recovery. Geometry
and lifecycle are good in this test; continuous availability is not. Production
code remains unchanged. Stop for review.
