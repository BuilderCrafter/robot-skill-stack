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

## B. Separation method selected before final holdout

The retained hierarchy uses metric dense seeds, exclusive assignment of sparse
islands to primitive surfaces, refits of changed measured clouds, and conservative
non-overlapping footprints. If one whole primitive also explains the cloud, raw
depth rays through the proposed separating corridor decide whether the gap is
observed free space, occupied foreground, or unresolved missing measurements.
Occupied foreground keeps one model; missing/conflicting evidence produces one
unknown. The algorithm never adds surface points or uses object truth.

Nearest-seed assignment and model-surface assignment both recovered four sphere
pairs in selection. Nearest assignment retained fewer samples and, without the
whole-parent/depth check, both methods falsely split objects with a missing band.
Organized depth-discontinuity proposals added no selection gain and were not
retained as a second production pipeline. New normal/curvature clustering or a
third production camera was unnecessary to resolve the demonstrated sphere
coverage failure. Conservative model surface residuals and current per-camera
evidence were sufficient for that case.

Holding the merged-baseline fitter fixed, the last captured frame of each of the
27 old selection scenes improved from **37/47 to 45/47 correct**, mixed clouds
from five to one, missing targets from five to one, and false positives from one
to zero. Known mixed clouds and false splits remained zero. All four sphere
pairs became eight known spheres. These 47 targets are a segmentation isolation
experiment, not the full temporal benchmark denominator. The remaining 2 mm
box/cylinder proposals already contain 92.9% of the measurements, but both seed
fits are unknown: this is model/partition ambiguity rather than sparse-island
coverage. The 26-point panel fragment is rejected below the existing fitter's
30-point minimum.

The negative missing-band control retains one sphere and conservatively returns
one unknown for a box or lying cylinder; the previous implementation returned
two false known objects for the latter cases. Eight old and nine new focused
segmentation/budget tests passed before the final source freeze.

## C. Dimensions and grasp eligibility

Controlled captures compare native USD curved primitives with explicit 32/128
segment meshes at identical nominal dimensions in two camera arrangements.
For a 71 mm sphere, mean baseline radius error was **1.123 mm native versus
0.0123 mm with the 128-segment mesh**. Corresponding values were 1.223/0.0134 mm
for a 76 mm sphere, 1.133/0.0072 mm for an upright cylinder and 1.158/0.0068 mm
for a lying cylinder. Each pair includes Camera A, Camera B and fused clouds
from both arrangements; these are correlated measurements, not independent
statistical trials. This isolates rendered surface tessellation as the dominant
source of inward bias in these controlled captures.

The finalizer preserves the original classification fit and scores. It derives
an empirical outer extent from current measurements (`2*P95-P50` of measured
radii), including separate view estimates. Sphere and lying-cylinder contact
constraints are used only when the original fit already infers support within
the existing 4 mm tolerance. Upright cylinders require at least semicircular
curvature; lying cylinders require quarter-circle curvature and measured cap
interiors. Inconsistent views or unobserved parameters retain the class but
publish `graspable=False`. Boxes use observed extents and measured residual
spread. No gripper-width constants or simulator tessellation correction enter
the production estimator.

The controlled clean envelope study covered all 60 evaluated radii and reduced
radius MAE from 0.553 to 0.234 mm. Sparse and noisy variants also covered all 60
in that selection study. An incomplete upright arc still exposed a displaced
center and underestimated radius despite high class confidence; this motivated
the explicit observability rejection. These are empirical envelopes, **not
certified upper bounds on hidden dimensions**. Opposite box faces in particular
may remain unobserved. Independent partial-occlusion and width tests below
therefore determine readiness separately from classification confidence.

The provider emits an explicit current boolean for graspability. This fixes the
old known-observation `None` convention retaining a previous `False` in the
unchanged WorldModel. Both no-new-frame retention and same-ID recovery from a
new real measurement are tested.

The first improved native planner-selection run passed its strict gate:
40/40 detected, 37/40 correctly classified, zero false-feasible, width or
clearance violations, and six conservative false-infeasible decisions. All 18
queried endpoint IK checks succeeded and all freshness/held/hint contracts
passed. The baseline had three false-feasible and three false-infeasible
decisions and four clearance shortfalls of approximately 0.5–0.96 mm.
These initial native runs used separate captures; the exact-frame comparison
and final holdout are reported separately. Artifacts:
`outputs/multiview-readiness/planner-improved-selection/`.

The selected production algorithm is frozen before evaluating the new
independent final holdouts. Subsequent checkpoints record hashes and all
outcomes, including failures; readiness is not inferred from selection results.
