# Multi-view geometric observability and grasp safety

This focused continuation starts from merged research parent
`3314eafcfd8199ad711172b955a373d63350b1eb` (PR #10), on
`research/multiview-geometric-observability`. V1, ground truth, WorldModel,
updater, planning/manipulation, robot control, baseline scenes, launchers and
dependencies remain outside the production change scope. No multi-view Pick,
Place, gripper action or contact trajectory is authorized or exercised.

## A. Reproduced failure and measured boundaries

All seven merged package modules are preserved in an isolated import namespace
under `outputs/multiview-observability/baseline_snapshot/`. Its manifest records
the original Git commit/tree and original/exported SHA256 hashes. The fresh
baseline offline suite passed 374 discovered units, eight standalone checks
and architecture, at
`outputs/multiview-observability-baseline-unit/run-wxa6n20t/`.

Before fixture or detector changes, the existing native partial-box benchmark
reproduced its **FAIL** at
`outputs/multiview-observability/baseline-partial/`: five targets, four correct
classifications, one false-feasible grasp and one actual width violation, with
zero source/runtime errors or robot commands. Its original two invalid B-control
audit rows remain preserved. They do not invalidate the separately valid
both-camera unsafe case.

The actual box is 90 × 90 × 50 mm, centered at `[445, -10, 25]` mm. Its measured
source component has 850 current points. The baseline reports
90.002 × 43.401 × 50.001 mm centered at approximately
`[445.000, -33.301, 25.001]` mm: 23.301 mm center error, cube confidence
0.999999923, `parameter_observable=True`, `graspable=True`. The unchanged planner
accepts a 43.401 mm closing width; its actual required width is 90 mm.

The combined views constrain both X limits and the negative-Y limit. The true
positive-Y physical boundary is at +35 mm world Y; the fitted apparent boundary
is approximately −11.601 mm, leaving 46.601 mm of actual extent hidden. Neither
camera measures the physical positive-Y boundary. Raw target-associated depth
samples are 235 in A and 3463 in B at stride one. Under an evaluation-only
±0.8 mm physical-plane check, positive-Y has zero samples in both views, while
negative-Y has 54/1892 and positive-X 70/28. Such plane proximity includes edge
samples and is not proof of a full visible face. Top height is measured;
the lower-Z support plane is a configured prior, not a reconstructed hidden face.

![True geometry, per-camera points, actual component, fitted extent and unsupported boundary](assets/multiview_observability/hidden_box_failure.png)

This figure replays the preserved source to recover its actual accepted
component and verifies its center/dimensions against the native result. Ground
truth is used only to draw and score physical geometry. Original raw frames,
camera poses/intrinsics, full camera/fused clouds, fitting scores, estimated
geometry and planner decisions remain in the baseline artifacts. Figure data
and large exports are under `outputs/multiview-observability/figures/`.

Small residuals explain the visible rectangular patch but do not establish that
its measured limits are object edges. A smaller complete box and a larger box
hidden behind the same foreground surface can explain these measurements.
More frames from unchanged obstructed viewpoints do not resolve that ambiguity.
The source must distinguish occupied surface, observed free space, nearer
occluders, invalid depth and field-of-view truncation before establishing action
eligibility. A free ray beyond an edge bounds an interval; it does not establish
the fitted sample minimum/maximum as an exact physical boundary.

## Reproduction environment

Use the unchanged workstation wrapper, Isaac Sim 5.1.0-rc.19/Python 3.11,
repository NumPy 1.26.4 before SimulationApp, and py_trees 2.6.0. Native
processes run sequentially. No dependency/environment changes or `--no-ros-env`.

```bash
./run_isaac_python.sh scripts/preserve_multiview_baseline.py \
  --ref 3314eafcfd8199ad711172b955a373d63350b1eb \
  --output outputs/multiview-observability/baseline_snapshot
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/multiview-observability-baseline-unit
# This command was run before repairing the B-only control fixture.
./run_isaac_python.sh scripts/benchmark_multiview_partial_grasp.py --no-ik --strict \
  --result outputs/multiview-observability/baseline-partial/result.json \
  --artifacts-dir outputs/multiview-observability/baseline-partial
./run_isaac_python.sh scripts/plot_multiview_observability.py \
  --capture outputs/multiview-observability/baseline-partial \
  --baseline-pythonpath outputs/multiview-observability/baseline_snapshot \
  --output outputs/multiview-observability/figures \
  --docs-dir docs/assets/multiview_observability
```

## B. Current physical-boundary evidence

The corrected B-only roof ends at world X = 420 mm, rather than 450 mm. It
still blocks B, but no longer removes 6.548 mm of the target's extent from the
intended clear A view. The valid both-camera 90 mm fixture is unchanged. A fresh
native baseline run with the corrected controls detects/classifies all five
targets, passes all fixture audits, and still has one false-feasible/width
violation. See `outputs/multiview-observability/corrected-partial-baseline/`.
The original failed controls and captures remain available separately.

Four methods were compared on 51 frozen baseline candidate clouds, including
18 box candidates, extracted without oracle masks from the old partial and
selection captures. This is a single-frame functional comparison; it does not
establish publication, current freshness or live performance.

| Method | Bounded boxes / 18 | Reason retained or rejected |
| --- | ---: | --- |
| Require direct support for every vertical face | 0 | Rejects the usual unseen opposite faces despite physical silhouette evidence. |
| Anchored top-plane free-space strip 4 mm outward | 14 | Conservative, but sparse free samples and a fixed outward offset lose useful observations. |
| Contradict an extra 4 mm box extension on each side | 17 | Stronger coverage, but a unilateral 4 mm limit costs up to 8 mm across a dimension. |
| Adjacent measured-top/free-depth transitions, with side-plane tightening | 16 | Retains 16/17 candidates excluding the mandatory unsafe box; rejects that hidden dimension and one close-neighbor ambiguity. |

The comparison script, candidate arrays, rows, parameters and source hashes are
saved in `outputs/multiview-observability/method-comparison-initial/`. The selected
method's accepted boxes grow by a mean 0.811 mm on their most affected axis,
worst 3.241 mm in a close-pair case. Clear 80/90 mm reference boxes grow only
0.226/0.249 mm. These are resolution-aware empirical bounds under the fitted
single-box model, not certified physical bounds for arbitrary hidden shapes.

The selected helper inspects a bounded ribbon around each fitted top edge in
each current calibrated depth image. It distinguishes measured occupied top,
valid depth farther than the proposed top-plane intersection, nearer occluder,
missing depth, and pixels outside the image. A physical edge needs adjacent
occupied/free pixels with free evidence outward from the measured interior in
three separated intervals along the edge. Foreground, invalid and out-of-view
space never count as free. An edge can be supported by either camera; one
occluded camera does not veto the other's actual physical-boundary evidence.

Each accepted ray transition supplies an outward interval, rather than an
assertion that the last cloud sample is the physical edge. Current measured
side planes can tighten that interval only when a top/free transition also
exists. Top-plane spread contributes a lateral ray margin; top/side edge voxels
are excluded from the plane-noise estimate. Both local X and Y signs must be
bounded, the top must be measured, and current low-side samples must support
the configured table-contact prior. Z support remains an explicit prior.

`geometric_observability` records per-axis/sign status, interval, per-camera
occupied/free/occluded/missing/out-of-frame counts, transitions and side-plane
evidence. The source combines this decision with its existing surface-parameter
check. It preserves classification scores and measured geometry; eligible boxes
publish the current outward geometry, with the pre-bound estimate retained in
metadata. An ambiguous known box remains a known box with `graspable=False`.
Sphere/cylinder finalization, point clouds, fit budgets, association, timestamps
and held/uncertain state retain their existing paths. There is no gripper-width
constant, history cache or simulator import in the helper.

Repeated obstructed fixed views cannot reveal a hidden boundary. The temporal
experiment therefore requires a new complete view to restore eligibility;
merely repeating frozen input cannot do so. The method assumes one supported
yaw box, finite depth and useful calibration. Correlated calibration/top-plane
error, thin hidden extensions and unsupported box shapes remain limitations;
the perturbation experiments measure their effect without claiming a universal
safety certificate.

```bash
./run_isaac_python.sh scripts/probe_multiview_observability.py \
  --capture outputs/multiview-readiness/partial-boxes \
            outputs/multiview-iteration2/selection-capture-v2 \
  --baseline-pythonpath outputs/multiview-observability/baseline_snapshot \
  --output outputs/multiview-observability/method-comparison-initial
```
