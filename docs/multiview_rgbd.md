# Experimental multi-view RGB-D perception

Follow-up: [robustness and performance, iteration two](multiview_robustness_performance.md).
The first-iteration measurements below remain the preserved comparison baseline.

This experiment tests whether complementary depth surfaces improve geometric
primitive fitting before solving automatic segmentation. V1 is the unchanged
reference, never a runtime dependency of the experimental implementation.

## Baseline and acquisition checkpoint

Branch `research/multiview-rgbd-fusion` starts at merged main `6929856` (PR #7).
The workstation wrapper selects Isaac Python 3.11.13, repository-local NumPy
1.26.4 and py_trees 2.6.0. No dependencies or workstation settings were changed.
The existing offline baseline passed: 297 discovered tests, eight standalone
suites and architecture/syntax. Artifacts:
`outputs/multiview-baseline-unit/run-dsx3wpeg/summary.json`.

Python 3.11's syntax scan initially failed while traversing a protected `.codex`
directory. The checker now prunes editor, dependency and generated-output
directories before traversing, while still compiling repository Python sources
and enforcing all existing architecture rules.

`scenes/multiview_playground.usda` sublayers the original binary scene. It
preserves Camera A, disables the baseline cube only in this experimental layer,
adds four static analytic shapes, and adds Camera B at (0.25, -0.85, 0.65) m
looking at (0.45, 0, 0.04) m. A is at approximately
(1.65896, -0.05281, 1.10358) m; exact matrices and intrinsics are saved in captures.
Both cameras use 640 × 480 pixels; no robot commands are issued.

Native acquisition passed at `outputs/multiview-capture-02/result.json`.
Camera A/B produced 341/1592 workspace points at stride 2, zero render-time skew,
and maximum SDK projection errors 0.000778/0.000263 mm. Unchanged tokens return
no sample. The initial failed capture is retained at `outputs/multiview-capture/`:
Isaac 5.1 exposes rational ReferenceTime dictionaries rather than integer frame
numbers. Only the new camera adapter normalizes these references.

Projection uses image-plane depth, ROS optical axes and half-pixel centres.
The SDK comparison uses explicit pixel coordinates and unordered nearest-point
checks, never annotator point order. See the versioned
[Isaac 5.1 camera API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.camera/docs/index.html).
The capture exports RGB, calibrated depth NPZs and separate/fused ASCII PLYs.

Run with the established wrappers:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/multiview_capture.py \
  --result outputs/multiview-capture-new/result.json \
  --artifacts-dir outputs/multiview-capture-new
```

Checkpoint B measures oracle-segmented object surfaces in a research utility.
Automatic detection and manipulation readiness are not established by capture
or analytic unit tests. Quantitative fitting and production-source results follow.

## Isolated surface fitting

`SurfaceFitter` consumes unorganized world-coordinate XYZ samples, without V1
imports. It fits robust algebraic spheres, table-supported yaw boxes using
surface-distance search and planar footprint coverage, and capped cylinders
using radial fits and bounded upright/lying axis search. All candidates compete
using 80th-percentile surface residuals and coverage; weak or ambiguous evidence
stays unknown. Two-stage voxelization suppresses within-view and across-view
duplicate density. Sample counts never directly increase fit confidence.

Initial native benchmark: `outputs/multiview-geometry-01/result.json`, 8 objects ×
2 camera/position arrangements × 2 repeated captures = 32 matched trials per
method. Oracle surface masks exist only in `tests/support/multiview_research.py`.
V1 A/B are evaluated unchanged on the same images; surface A/B/fused use the
independent experimental fitter. Every trial, including any rejection, is saved.

| Method | Correct / trials | Mean center error (mm) | Mean dimension error (mm) | Mean processing (ms) |
| --- | ---: | ---: | ---: | ---: |
| V1 A | 32/32 | 1.932 | 1.509 | 93.2 |
| V1 B | 32/32 | 1.013 | 1.164 | 133.9 |
| Surface A | 32/32 | 0.295 | 1.364 | 58.9 |
| Surface B | 32/32 | 0.192 | 1.054 | 65.3 |
| Fused surfaces | 32/32 | 0.146 | 1.167 | 63.9 |

This first run establishes improved center fitting, **no classification gain**
in these isolated scenes, and a dimension-error regression relative to B alone.
V1 processing includes discovery; surface-fit timing excludes oracle extraction,
projection and rendering, so those timing columns are not equivalent pipelines.
The next benchmark includes a fixed occluder between A and the object, retaining
the same objects and a complementary B viewpoint. Two cameras do not observe
surfaces hidden from both views. These measurements justify continuing the
automatic-segmentation experiment, not declaring manipulation readiness.

The completed extended benchmark is retained at
`outputs/multiview-geometry-02/`: 48 trials per method, including 16 occluded
trials. All 240 method/trial rows are in JSON and CSV, with 72 object PLYs,
representative RGBs and separate render/acquisition/fusion timings.

| Method | Correct | Unknown | Center mean / p95 (mm) | Dimension mean (mm) |
| --- | ---: | ---: | ---: | ---: |
| V1 A | 44/48 | 4 | 2.950 / 8.826 | 2.050 |
| V1 B | 48/48 | 0 | 1.005 / 1.375 | 1.151 |
| Surface A | 44/48 | 4 | 1.252 / 6.868 | 1.892 |
| Surface B | 48/48 | 0 | 0.199 / 0.483 | 1.036 |
| Fused surfaces | 48/48 | 0 | 0.167 / 0.475 | 1.115 |

All methods detected oracle-matched points in all 48 trials. Under occlusion,
A's two fitters each classified 12/16; B and fused classified 16/16. Mean center
errors in those 16 trials were 4.987 mm (V1 A), 3.166 mm (surface A), 0.215 mm
(surface B) and 0.210 mm (fused). Dimension/axis errors are reported only where
the corresponding primitive parameters exist; sample counts are in result JSON.
Unknown trials remain in the accuracy denominator and center metrics. B's
dimension error remains better than fusion, and V1's box/axis search is slightly
more precise than the new bounded yaw search. Repeated captures of identical
static geometry are not independent random draws; no significance claim is made.

The final export run at `outputs/multiview-geometry-final/` repeated the same
48 trials per method and retained the same classification counts. Its 240 JSON
rows additionally record canonical estimated dimensions, estimated axis/yaw,
candidate residuals in metres and confidence. Box dimension errors compare
sorted side lengths; cylinder dimensions are diameter/diameter/length. These
exports and their PLYs are the inputs to the final figures.

V1 retains its original integer-pixel projection convention. The experimental
projector uses the SDK's half-pixel centres, and the fitter is independent.
Consequently, V1-versus-new localization gains cannot be attributed solely to
fusion; surface A/B versus fused is the controlled comparison for added geometry.

```bash
./run_isaac_python.sh scripts/benchmark_multiview_geometry.py \
  --result outputs/multiview-geometry-new/result.json \
  --artifacts-dir outputs/multiview-geometry-new --repeats 2
```

## Automatic source and integration

Select `world.provider = "multi_view"` with
`config/scenes/multiview_playground.toml` and the experimental USD layer. The
ordinary playground still selects `perception`; `ground_truth` remains available.
No V1 discovery, fitting, tracking, provider, baseline scene, manipulation,
Behavior Tree or WorldModel implementation was changed.

The new package has separate acquisition, clouds, segmentation, fitting, tracking
and provider modules. Runtime imports include common Pose, PrimitiveGeometry,
ObjectObservation and ObservationBatch, with no V1 detector/estimator or research
oracle imports. Shared changes only extend configuration, source construction,
default update scheduling and test discovery. No registry or inheritance schema
was introduced.

Both proposed grouping strategies were investigated on the calibrated four-object
capture (`outputs/multiview-grouping/result.json`). A research-only probe used
V1 image regions as candidate references and associated their world centroids:
4 regions per camera, 4 unambiguous pairs, 0 unmatched. Global 3D grouping also
gave 4 clusters. That probe took 177 ms for region extraction/association and
35 ms for projection/fusion/global grouping, with differing sampling and code;
it is not a controlled performance comparison. Global grouping was selected
because it operates directly on fused surfaces and needs no second production
association pipeline. The probe's V1 imports are confined to a research script.

The automatic path projects valid image-plane depths, crops to the configured
workspace, removes samples within 4 mm of the known support plane, and voxelizes
each view before global duplicate suppression. It bounds the workspace cloud to
12,000 points. A world-coordinate 8 mm cell adjacency graph separates components;
small disconnected components are discarded. At most 16 candidate components
and 8 accepted objects are processed, with 1,200 fitting points per object.
Robust residuals and percentile bounds limit isolated depth errors. Large arm-
connected components and elevated components without support evidence are
rejected. Spheres/cylinders may infer a hidden contact from an accepted surface
fit; boxes require observed low side samples. RGB, identities, masks, object
prim paths and semantic labels are not production segmentation inputs.

Primitive fits run once per fused object cluster. Tracking confirms three
consecutive distinct samples and assigns one identity per detection. Placement
hints take priority for long-distance association. Held/uncertain identities
cannot be stolen for tabletop detections. The unchanged WorldModelUpdater owns
source health, expiration, missing visibility and `forget` calls. The source
retains confirmed tracks until the updater forgets them.

Frames must have increasing render reference and simulation time, and timestamps
within 25 ms across cameras. The batch keeps the local monotonic acquisition
time recorded before processing. Repeated frames publish no batch; one frozen,
missing or all-invalid camera fails explicitly. There is no silent one-camera
source fallback. An object visible to only one of two healthy cameras is allowed.
Diagnostics distinguish acquisition, projection, fusion, grouping and fitting.

The source owns both cameras through one rig. Partial construction/warmup failures
close every allocated camera. Shutdown clears tracks, removes camera callbacks,
detaches annotators and destroys render products once. Native contract testing
also exercised failed second-camera construction and two complete reinitializations.

## Automatic evaluation and latency

`outputs/multiview-detection-01/` contains 24 native scene trials: two camera
arrangements × six conditions × two seeded millimetre position perturbations.
Conditions include clear scenes, partial/strong A occlusion, B occlusion, nearby
separated spheres, and an elevated link-shaped primitive moved between trials.
The last fixture is a surrogate, not exhaustive validation of moving Franka
geometry. Four targets per scene (two in the nearby condition) yield 88 expected
object observations per method. All sources operate without oracle inputs;
truth is consulted only afterward for matching and errors. Matching is one-to-one
within 40 mm. A mixed cloud has at least 10% surface support from two truth objects.

| Method | Detected | Correct / expected | Unknown | Missing | Center mean / p95 (mm) | Dimension mean (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 A | 88/88 | 82/88 | 6 | 0 | 2.491 / 8.323 | 1.478 |
| V1 B | 80/88 | 80/88 | 0 | 8 | 1.025 / 1.488 | 1.193 |
| Multi-view | 84/88 | 84/88 | 0 | 4 | 0.221 / 0.458 | 1.136 |

There were no extra detections, duplicate IDs or mixed accepted clouds in these
trials (88/80/84 clouds checked respectively). Missing detections remain in the
accuracy denominator; localization metrics are conditional on detection. The
multi-view method therefore improves correct classification over either fixed
view on this set, but **has a lower detection rate than A**. Every miss occurred
when the upright cylinder's approximately 4.5 mm gap to the large B-side occluder
was bridged by 3D adjacency. The combined component was conservatively rejected
as oversized. No failed trial was removed or relabelled successful.

Mean/p95 native processing times (ms), excluding render: acquisition 5.0/6.3,
projection plus per-view filtering 15.2/19.8, fusion 4.9/12.0, grouping 40.4/71.2,
fitting 175.3/251.1, total 241.0/308.8. Eight actual render steps between samples
took 80.5/95.1 ms. The measured loop rate was **3.11 Hz**, not 5 Hz. This loop
times the multi-view source alone; the separate V1 comparisons are excluded.
Default runtime scheduling requests 5 Hz in simulation time, so wall-clock
throughput may be lower when synchronous processing slows Kit. A 3 Hz setting
is more appropriate for these four-object scenes until fitting/grouping are
optimized. Peak process RSS was 5,212 MiB for the whole Isaac/evaluation process,
not isolated provider memory. The native log identifies an RTX 4070 SUPER.

The saved-capture sampling probe (`outputs/multiview-sampling/result.json`)
measured 342 ms at stride 1, 189 ms at stride 2, 159 ms at stride 3, and 186 ms
at stride 2 with 3 mm voxels/600 fit points. All four isolated objects classified
correctly; mean center errors were 0.142, 0.186, 0.162 and 0.181 mm respectively.
These are three-repeat offline replays of one native capture, excluding live
acquisition/rendering. They support investigating stride 3, but do not establish
its native occlusion accuracy. The conservative default remains stride 2.

Three standalone figures are generated under `outputs/multiview-figures/`:
`geometry_comparison.png`, `partial_and_fused_clouds.png`, and
`automatic_detection_and_latency.png`. Per-object PLYs and row-level estimated
and true dimensions accompany the JSON/CSV. Captures and figures stay ignored;
the commands reproduce them rather than committing large artifacts.

```bash
./run_isaac_python.sh scripts/benchmark_multiview_detection.py \
  --result outputs/multiview-detection-new/result.json \
  --artifacts-dir outputs/multiview-detection-new --repeats 2
./run_isaac_python.sh scripts/compare_multiview_grouping.py \
  outputs/multiview-capture-new --result outputs/multiview-grouping-new/result.json
./run_isaac_python.sh scripts/probe_multiview_sampling.py \
  outputs/multiview-capture-new --result outputs/multiview-sampling-new/result.json
./run_isaac_python.sh scripts/plot_multiview_results.py \
  --geometry outputs/multiview-geometry-new \
  --detection outputs/multiview-detection-new --output outputs/multiview-figures-new
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview \
  --artifacts-dir outputs/multiview-native-new
```

The benchmark result status PASS means the complete experiment ran and saved all
trials; classification failures are in its metrics. In contrast, the native
contract scenario's PASS requires its explicit geometry, identity, freshness and
cleanup assertions to succeed.

## Validation and readiness

Executed on this workstation:

| Check | Result / artifacts |
| --- | --- |
| Before production changes | 297 tests + 8 suites + architecture PASS, baseline path above |
| Final offline | 320 tests + 8 suites + architecture PASS, `outputs/multiview-unit-final/run-ic2azsjt/` |
| Native acquisition | PASS, `outputs/multiview-capture-02/` |
| Isolated native geometry | Completed 48 trials per method, `outputs/multiview-geometry-02/` |
| Automatic native evaluation | Completed 24 scenes, `outputs/multiview-detection-01/` |
| Native source contract | PASS initially and after final freshness/cache changes, `outputs/multiview-native-final/run-edphbmp7/` |
| Unchanged V1 integration | Both PASS, `outputs/multiview-v1-isaac/run-fgkfqfjd/` |
| Unchanged six robotics regressions | All PASS, `outputs/multiview-v1-regression/run-3i85xmhb/` |

Existing native acceptance gates, including all four benchmark trials and the
25 mm maximum physical placement error, are unchanged. Run native scenarios
sequentially through the existing launcher and close conflicting Isaac instances.
The original benchmark completed 4/4 successful Pick/Place trials, maximum
physical target error 10.798 mm. The experiment's own tests commanded no robot
motion. The 23 added offline tests cover transforms, synchronization, invalid
depth, duplicate filtering, grouping, primitive fits/unknowns, one-view visibility,
identity/placement hints, held/uncertain protection, expiration, configuration,
failed initialization and resource cleanup. All original 297 tests remain.
The native contract also passed with artifacts disabled; its runner log is
`outputs/multiview-native-no-artifacts.log`.

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/multiview-v1-isaac-new
./run_isaac_python.sh scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/multiview-v1-regression-new
```

The new source is an opt-in experimental perception alternative and has native
observation-contract evidence. It is **not ready to claim reliable manipulation**:
no multi-view Pick/Place was commanded. Narrow gaps, touching/stacked objects,
robot links near the support plane, arbitrary clutter, arbitrary tilted boxes,
oblique cylinder axes, calibration errors, time offsets and realistic sensor
noise remain limits or pending native experiments. Table support is a strong
prior, not a reconstruction of hidden surfaces. Yaw boxes and upright/lying
cylinder axes are supported; general 6D shape recovery is not claimed. The scores
are geometric consistency scores, not calibrated probabilities. Refine grouping
and reduce latency, then perform supervised application tests before relying on
this source for grasping. V1 remains the default working baseline.
