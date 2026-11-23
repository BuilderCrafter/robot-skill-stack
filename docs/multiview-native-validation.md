# Native multiview runtime measurements

The native Xeon/L4 study found accurate four-object geometry and working bounded
motion/lifecycle controls, but **continuous perception availability failed**.
It measured about 2 Hz publication against 5 Hz requested, with stale intervals
hiding all objects. It did not validate GUI operation or complete Pick/Place.

These are historical native results at
`f2cfa9203edf3a9ce7bd837dc9444f5b1df547e7`. Subsequent fixed-input work recovered
the occluded Square and preserved static geometry; see the
[geometry/performance experiments](multiview-performance-ab.md#native-occluded-box-qualification).
Those offline results do not constitute a new live availability test. Current
behavior/setup is defined by the [multiview guide](multiview_rgbd.md) and
[demonstration guide](multiview_perception_final.md).

## Environment and method

| Item | Native environment |
|---|---|
| Installation | Existing `isaac51` container, `/isaac-sim` |
| Isaac | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| GPU / driver | NVIDIA L4, 23,034 MiB / 580.178.04 |
| Python / NumPy | SDK 3.11.13 / repository `.deps` 1.26.4 |
| Launcher | Unchanged `run_isaac_python.sh`, `ISAAC_SIM=/isaac-sim`, headless |
| Scene / configuration | `scenes/multiview_perception_final.usda` / matching TOML |
| Budgets | Two 640×480 cameras, stride 2, 2 mm voxels, 12,000 points, 24 fits, 5 Hz requested |

Two sessions ran sequentially using a clean `git archive` extraction at
`/tmp/thesis-multiview-native-f2cfa92`, borrowing only `.deps`. Normal bootstrap
owns acquisition → shared frames → observation runtime → process worker →
association → updater → WorldModel. Test hooks record capture, publication and
visibility; `native_probe.py` times original child computation and unwraps its
result before normal publication. No detector/source/freshness adjustment,
synthetic camera input, custom camera edit or dependency change was made.
GT enters only scoring and independent saved-depth analysis. Run 2 completes
the arm/lifecycle controls; a stopped-state test query interrupted run 1's lifecycle
reporting, with successful cleanup. Authored scene/configuration hashes are unchanged.

Native pinhole calibration has zero skew, principal point (320,240), bottom row
(0,0,1); full ROS transforms, tokens and acquisition times are saved:

| Camera | fx | fy | World position, m |
|---|---:|---:|---|
| `/World/PerceptionCamera` | 554.256226 | 554.256287 | (1.658960, −.052813, 1.103581) |
| `/World/PerceptionCameraSide` | 732.999268 | 732.999268 | (.180000, .800000, .650000) |

These measured focal lengths supersede the analytic main-lens approximation.
The SDK reported vertical-aperture adjustment for square pixels; both render
clocks advance together without artificial exposure times.

## Geometry versus availability

All five static publications/session contain four correct objects with both
cameras contributing and stable IDs. Run 2 fresh-publication errors:

| Reference | Class | Center / max dimension error, mm | Confidence | Graspable | ID |
|---|---|---|---:|---|---|
| Square | cube | .135 / .273 | 1.000000 | yes | `multiview_2` |
| Rectangle | cube | .116 / .234 | 1.000000 | yes | `multiview_1` |
| SmallSphere | sphere | .176 / .314 | .970753 | yes | `multiview_4` |
| LargeSphere | sphere | .286 / .542 | .948450 | yes | `multiview_3` |

Despite that accuracy, run 2 has **4/14 static samples with zero visible objects**
at source ages 1.105–1.196 s. Run 1's final static snapshot also falls in a stale
interval. Fresh-publication geometry and continuous availability are separate results.

The throughput window spans 12 simulation seconds; scheduling uses wall time.
Camera counts are captured snapshots, not every internally rendered exposure.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Wall / simulation seconds | 17.824 / 12.000 | 17.919 / 12.000 |
| Requested acquisitions/wall second | 5 | 5 |
| Camera snapshots | 178 | 180 (90/camera) |
| Fresh synchronized batches / submitted jobs | 89 / 89 | 90 / 90 |
| Jobs started / completed, window deltas | 35 / 35 | 36 / 36 |
| Pending replacements | 54 | 54 |
| Discarded completions / worker exceptions | 0 / 0 | 0 / 0 |
| Delivered results / successful publications | 36 / 36 | 37 / 36 |
| Publications/wall second | 2.020 | 2.009 |
| Publications/simulation second | 3.000 | 3.000 |
| Rejected publication preflights | 0 | 1 (age 1.060 s) |
| Camera admission errors | 0 | 0 |
| Visibility samples with zero / four objects | 19 / 65 | 18 / 66 |
| Source-health stale samples | 22 / 84 | 21 / 84 |
| Worker p50 / p95, ms | 505.21 / 582.49 | 500.82 / 562.58 |
| Actual compute p50 / p95, ms | 495.23 / 573.03 | 490.99 / 549.69 |
| Transport/IO residual p50 / p95, ms | 9.70 / 11.66 | 9.87 / 12.32 |
| Publication frame-age p50 / p95, ms | 648.48 / 770.38 | 633.43 / 707.07 |
| Maximum publication frame age, ms | 903.04 | 713.21 |

A job crossing the window boundary explains ±1 count differences. Run 2's rejected
publication is an inherited job delayed by artifact capture; run 1 reproduces
visibility gaps without such a rejection. A discovery-context revision change
also correctly rejects a warm-up result outside these windows. Pending replacement
is the bounded worker policy, not itself a failure.

Run 2 child-stage measurements, ms; overlapping stages/percentiles are not additive:

| Stage | p50 | p95 |
|---|---:|---:|
| Projection/cropping/voxelization | 17.49 | 21.83 |
| Initial clustering | 10.03 | 11.08 |
| Refinement inclusive | 44.24 | 51.17 |
| Metric graph within refinement | 43.17 | 50.01 |
| Fitting | 386.48 | 433.11 |
| Parameter finalization | 2.90 | 3.39 |
| Geometric observability | 19.93 | 27.32 |

Updater application median/max is .107/.203 ms. Transport/IO is worker duration
minus compute, including serialization, pipes and scheduling; association was
not separately profiled. Fitting accounts for roughly 79% of static compute.
Results arrive about .63 s after capture, with about .5 s to the next publication:
the previous observation exceeds the unchanged one-second freshness limit.
Every empty run-2 static/throughput/arm sample has stale health and age >1 s;
maximum sampled age is 1.458 s, including the artifact boundary. Neither worker
exceptions nor mixed-camera admission caused these gaps.

## Controlled arm approach and occlusion

After four confirmed observations, the existing Pick grasp planner selects the
large sphere's pre-grasp pose. Normal backend RMPflow executes an intermediate
pose 120 mm higher, pre-grasp, retreat and return; no joint teleport, finger
closure or object transport occurs. Initial TCP (.597491,−.086039,.364492) m.

| Phase | Actual final TCP, m | Error, mm | Target missing / samples | Publications |
|---|---|---:|---:|---:|
| Intermediate | (.534399,−.152441,.253597) | 9.66 | 4 / 30 | 13 |
| Pre-grasp | (.536435,−.156485,.143494) | 9.42 | 4 / 18 | 7 |
| Retreat | (.538084,−.157105,.246560) | 9.65 | 6 / 19 | 8 |
| Return | (.591398,−.091308,.360000) | 9.22 | 5 / 21 | 10 |

All 38 arm-phase publications retain target `multiview_3`, sphere classification,
graspability and .286/.542 mm errors; no extra confirmed robot object appears.
All **19/88 target-missing samples** are stale intervals, ages 1.029–1.287 s.
One occurs at TCP (.530247,−.148770,.230408), age 1.081 s, hiding all four objects.
Saved static/pre-grasp/home depths retain the same separate 337-point sphere
component; all samples match the reference surface within 2.5 mm, with identical
bounds and visible native RGB. This approach demonstrates stale hiding, not
sphere/robot coarse merging.

The Square has a different historical failure: main-camera gripper occlusion
causes UNKNOWN/ungraspable in 5/7 pre-grasp and 3/8 retreat publications; all 10
home publications restore its cube and ID. Worst UNKNOWN bound error is 17.84 mm.
The first failure is fresh (age .658 s), reason `unresolved_components`; nearest
acquisition-time TCP (.528743,−.149248,.196275) was sampled 88 ms earlier, with
joints saved. The Square cloud shrinks 570→487 points; all are within 2.5 mm of
box surfaces and no robot returns are found. Retained 4 mm components [291,37,38]
cover only 75.15%. A whole-cloud cube fit alone cannot prove hidden dimensions.
Subsequent [matched-input comparison and recovery](multiview-performance-ab.md#native-occluded-box-qualification)
proved the baseline regression and restored this saved geometry; the native
session itself did not test that repair.

## Freshness and lifecycle controls

Run 2 verifies that frozen camera exposures are not republished, visibility expires
normally, and resume requires fresh frames. A real Kit render between camera reads
is rejected as `Camera updated during acquisition`. In-flight Stop advances the
worker epoch and prevents stopped publication; Play resets admission and recovers
with new timestamps, rejecting one late old-generation exposure first.

Shutdown releases camera callbacks, render products, ReferenceTime annotators,
worker thread and child process. External inspection finds only the original idle
`sleep` process and 0 MiB GPU usage; the container returns to stopped. No scene,
SDK, launcher or global configuration changes are involved.

## Evidence and reproduction

`outputs/multiview-native-20261122/session-{1,2}/` retains result JSON, native logs,
capture/publication/state records, RGB/depth images and original NPZs, with exact
run-specific test files. Session 2 also contains `native-depth-connectivity.json`,
`square-fragmentation.json` and `failure-examples.json`. Large generated buffers
remain outside Git; the [fixed-input study](multiview-performance-ab.md) records
the replay manifest and calibration/input hashes.

Use an existing installation and an unused output path:

```bash
ISAAC_SIM=/actual/isaac-sim ./run_isaac_python.sh \
  tests/multiview/test_final_native.py --headless \
  --result outputs/NEW-native-final/result.json \
  --artifacts-dir outputs/NEW-native-final
```

On the VM, the unchanged launcher ran through `docker exec` in the extraction
above with `ISAAC_SIM=/isaac-sim`, externally bounded to 480 s. Sessions finished
in about 57/70 s including startup/shutdown. Both result JSONs say **FAIL**;
SDK launcher exit 0 does not establish success. For exact historical repetition,
use the recorded source/test copies; running today's source is a new validation.

The live availability failure remains a demo limitation. Do not extend freshness
limits, retimestamp archived observations or reduce camera evidence to conceal
it. The headless approach does not establish full manipulation readiness or GUI
quality. See [validation](validation.md) for supported checks.
