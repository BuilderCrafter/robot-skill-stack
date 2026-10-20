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

The following checkpoints record the corrected fixtures, method selection,
source freeze, independent holdout, native performance and explicit readiness
decision. Baseline failure analysis alone does not establish a safe solution.
