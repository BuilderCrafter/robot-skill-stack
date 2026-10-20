# Multi-view manipulation readiness

This research branch starts at `7d1eb65e26354ac9bfc5a6b19664005729a6491d`
on `research/multiview-rgbd-fusion`. The original observation interface and the
V1, WorldModel, manipulation, controller and planner implementations are
unchanged. This experiment does not execute multi-view Pick/Place.

## A. Reproducible baseline and failure analysis

The previous implementation is exported, with original and import-isolated
SHA256 hashes, to `outputs/multiview-readiness/baseline_snapshot/manifest.json`.
The export contains all seven package modules and the instrumented original
benchmark. Its only changes are import isolation and research capture/timing
instrumentation. The original package tree is
`1a611c419dd07117759073b932ea43eabbaf1334`.

The complete baseline suite passed before production edits:
`outputs/multiview-readiness-baseline/run-gplmi46n/`. This includes 340 discovered
unit tests, eight standalone checks, architecture checks, two V1 native Isaac
tests, six robotics regressions and the two existing multi-view native tests.
The original accuracy gates and launch environment were preserved.

A fresh two-repeat original benchmark reproduced **87/88 correct**, one miss,
zero unknown, false-positive, duplicate or mixed observations. Mean center error
was **0.196 mm**, dimension error **1.150 mm**, and axis error **0.0169 degrees**.
Mean processing was **162.807 ms**, rendering **81.359 ms**, and the combined
loop **244.179 ms / 4.095 Hz**. Artifacts:
`outputs/multiview-readiness-baseline-detection/`. The earlier report's 4.15 Hz,
44/52 clean holdout and 85/103 perturbed holdout remain historical until their
matched replays below; they are not fresh results from this run.

| Failure | Selection evidence | Consequence |
| --- | --- | --- |
| Sphere pairs at 2/4/6/10 mm | Dense metric seeds contain only 72–81% of measured points; sparse islands cause the old coverage rejection | Measurements support two surfaces; retain exclusive model-consistent assignments and refit the changed clouds |
| Single object with an artificial missing-depth band | Old hierarchy splits one box or lying cylinder into two known primitives | Mathematical fit quality alone cannot establish a physical gap; check original depth rays |
| Cylinder beside the original occluder | Actual separation is approximately 1.671 mm | Preserve the failure capture; distinguish visibility/resolution limits from the sphere coverage issue |
| Close box/cylinder under sparse sampling | Competing models remain insufficiently distinct | Retain conservative rejection rather than a confident merged primitive |
| Small panel fragment | 26 measured points, below the fitter's 30-point minimum | Reject unsupported sample counts before emitting an object |
| Curved dimension bias | Native USD curves versus explicit high-resolution meshes with identical nominal dimensions | Investigate the rendered surface and observable outer envelope; do not apply a nominal-radius multiplier |
| Incomplete curved arc | High class confidence can coexist with an underestimated radius and displaced center | Geometric observability must govern grasp eligibility independently of primitive class |

Complete selection evidence, seed proposals, assigned clouds and model scores
are retained under `outputs/readiness-segmentation/`. Twenty controlled native
geometry captures, including two camera arrangements and explicit tessellation
ablations, are under
`outputs/multiview-readiness/geometry/controlled-capture/`.

The first native planner-selection run is preserved at
`outputs/multiview-readiness/planner-baseline-selection/`: its JSON result is
**FAIL** because the research serializer encountered a NumPy boolean. Its process
exited zero because the diagnostic writes failures to its result file. Native
results in this report must therefore be checked from the result JSON, not only
the process exit code. This is invalid fixture evidence, not a detector score.

## Reproduction environment

Always initialize repository NumPy 1.26.4 before SimulationApp and use the
unchanged wrapper. Isaac Sim is 5.1.0-rc.19 with Python 3.11 and py_trees 2.6.0.
Native processes run sequentially.

```bash
./run_isaac_python.sh scripts/preserve_multiview_baseline.py \
  --ref 7d1eb65e26354ac9bfc5a6b19664005729a6491d \
  --output outputs/multiview-readiness/baseline_snapshot
./run_isaac_python.sh scripts/run_all_regressions.py --group all \
  --artifacts-dir outputs/multiview-readiness-baseline
./run_isaac_python.sh outputs/multiview-readiness/baseline_snapshot/benchmark_original.py \
  --result outputs/multiview-readiness-baseline-detection/result.json \
  --artifacts-dir outputs/multiview-readiness-baseline-detection --repeats 2
./run_isaac_python.sh scripts/benchmark_multiview_dimensions.py \
  --output outputs/multiview-readiness/geometry/controlled-capture
```

Subsequent checkpoints append the selected algorithm, frozen holdout results,
real-Franka measurements, planner compatibility, performance and explicit
readiness decisions here. No final readiness decision is implied by this
baseline checkpoint.
