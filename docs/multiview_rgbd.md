# Experimental multi-view RGB-D perception

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
or analytic unit tests. Quantitative fitting and production-source results will
be recorded below after execution.

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

```bash
./run_isaac_python.sh scripts/benchmark_multiview_geometry.py \
  --result outputs/multiview-geometry-new/result.json \
  --artifacts-dir outputs/multiview-geometry-new --repeats 2
```
