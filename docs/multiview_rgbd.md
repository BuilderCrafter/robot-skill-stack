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
