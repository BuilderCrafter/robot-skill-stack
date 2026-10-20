# Multi-view robustness and performance, iteration two

This research continues `research/multiview-rgbd-fusion` at
`42e3258ac5a5c79bba766262357e77f80d158c14`. The new branch is
`research/multiview-robustness-performance`; its PR targets the experimental
parent. The [first report](multiview_rgbd.md), original benchmark scenarios and
previous artifacts remain the comparison baseline. V1 remains the default.

The retained hierarchy and fitter improve original automatic results from 84/88
to 87/88 correct and live loop rate from 3.13 to 4.15 Hz. Clean unseen holdout
outcomes improve from 22/52 to 44/52, including the expected unknown; known-shape
accuracy improves from 21/51 to 43/51. Sparse sphere pairs, one original narrow
gap and a small dimension-error regression remain. This supports continued
controlled perception research, without establishing autonomous manipulation
readiness.

## Checkpoint A: preserve and diagnose

Before algorithm changes, the established wrapper passed 320 discovered unit
tests, eight standalone suites and architecture checks
(`outputs/multiview-iteration2-baseline-unit/run-kw3e885w/`). The unchanged native
source-contract test passed
(`outputs/multiview-iteration2-baseline-native/run-nfp_32pv/`): four objects,
persistent identities, frozen-camera failure, stale repeated frames, protected
uncertain identities, placement-hint reacquisition, partial-initialization
cleanup and two reinitializations. No robot commands were issued.

`scripts/preserve_multiview_baseline.py` exports the seven original package files
from Git into an independent `baseline_multiview` import namespace under ignored
`outputs/multiview-iteration2/baseline_snapshot/`. Its manifest records original
commit/tree and original/exported SHA256 hashes. The only package transformation
is the internal import prefix, so baseline and improved implementations can
consume identical saved frames in one replay without module-cache collisions.
The original experimental TOML and report are copied alongside it.

The generated `benchmark_original.py` retains the original 24 native scenes,
88 expected observations, V1 comparisons and live timing procedure. It retains
the latest fresh frames in memory and exports full camera/fused clouds only
after the timed samples. Thus rejected components, including occluders, are
available for diagnosis; the old accepted-object-only exports omitted them.
The generated script's hash and instrumentation are recorded in the manifest.

The original 8 mm occupied-cell graph connects all 26 adjacent cells without a
sample-distance check. Points in adjoining cells may be separated by nearly
27.7 mm in three dimensions. It can bridge a real 4.5 mm gap and reject the
combined object/occluder as oversized. Merely decreasing a global threshold has
a competing failure: sparse surfaces from one object become disconnected.
Initial saved-cloud probes retained every case, including those false splits.

The frozen fitter was profiled on 114 saved native object clouds (72 isolated
geometry clouds plus 42 accepted automatic clouds). Instrumented CPU replay
took 6.855 s: cylinder hypotheses accounted for 5.037 s cumulative (73.5%) and
box search 1.673 s (24.4%). There were 37,290 percentile calls (2.274 s) and
25,467 least-squares calls (0.674 s). Repeated orientation/residual work dominates
more than least squares alone. The profile is retained at
`outputs/iteration2-fitting/baseline-profile.txt` and `baseline.prof`; its
60.2 ms/cloud is profiled offline replay, not live Isaac throughput.

The previous automatic benchmark remains 84/88 correct, four missing, zero
extra/duplicate/mixed accepted observations, 0.221 mm mean center error,
1.136 mm mean dimension error, 241 ms processing and 3.11 Hz measured loop rate.
Those historical numbers are kept separate from this iteration's fresh runs.

Reproduce the preserved baseline with the existing environment:

```bash
./run_isaac_python.sh scripts/preserve_multiview_baseline.py
./run_isaac_python.sh outputs/multiview-iteration2/baseline_snapshot/benchmark_original.py \
  --result outputs/multiview-iteration2-baseline-detection/result.json \
  --artifacts-dir outputs/multiview-iteration2-baseline-detection --repeats 2
```

Measurements use the university workstation: AMD Ryzen 7 7800X3D, RTX 4070 SUPER
(12,282 MiB), NVIDIA driver 580.167.08, Isaac Sim 5.1.0-rc.19, repository-local
NumPy 1.26.4, py_trees 2.6.0 and the established Isaac Python wrapper. Dependencies,
launcher, baseline scene, V1 and the robotics architecture are unchanged.

## Checkpoint B: separation alternatives

The reproducible research probe is `scripts/probe_multiview_segmentation.py`.
It tested smaller occupied cells; exact 3/4/5 mm distance graphs; density-adaptive
connectivity; local PCA normal-angle and tangent-plane gates; and primitive-
supported metric partitions. Competing prototypes remain in this research
utility, not in production dispatch. Translated isolated native clouds test
geometry/density in isolation, but omit mutual occlusion. New native captures
provide the occlusion-sensitive selection comparison.

On the 42 previously accepted native single-object clouds, 3 mm occupied cells
lost two sparse objects. A 4 mm metric graph falsely split four and lost two;
5 mm avoided those splits but still lost two. Adaptive density and normal-angle
gates also lost two, increased computation, and left 16 and 17 mixed pairs out
of 32 respectively, versus 16 for plain radius 5 mm and 32 for the original graph.
Tangent gates added nine false splits on the 32 translated pairs.
These alternatives were rejected as global production replacements. Full rows,
including every negative outcome, are in
`outputs/iteration2-segmentation/result-initial.json`.

The retained hierarchy preserves broad connectivity, with a faster vectorized
cell-neighbor lookup and integer union graph. An exact 4 mm point-distance graph
proposes cuts within each broad component. A cut needs at least two individually
known, supported primitive fits explaining at least 85% of the original cloud.
An oversized component may instead yield a known supported object beside a
dense, separately oversized remainder. Nearly coincident child fits are vetoed.
Implausible or overly fragmented cuts retain the complete parent; missing
samples are never reconstructed. Fitting work is cached for accepted children.
No identities, object counts, expected positions or truth labels guide the cut.
There are at most 16 broad components, 16 proposed fine components per parent
and eight accepted detections. This hierarchy can perform more than the old
16 total candidate fits in fragmented clutter; worst-case latency remains a
limitation despite the bounded workspace cloud.

The selected rule preserved all 42 singles and reduced mixed translated pairs
from 32/32 to 14/32, with zero false splits or target-component losses
(`selected-original-clouds.json`). On 27 valid native selection workspaces,
coarse grouping produced 14 mixed components; plain radius grouping eliminated
mixes but caused one false split and two losses. The hierarchy reduced mixed
components to five, with zero false splits and zero target-component losses
(`native-selection-v2.json`). These are segmentation diagnostics, not automatic
classification counts. Remaining mixes were four sphere pairs at 2/4/6/10 mm
and a 2 mm box/cylinder pair. Sparse sphere samples left only 72–81% of the
cloud in dense seeds, so the conservative coverage requirement refused a cut.
Sparse-island model assignment was not retained without further evidence.

The original hidden-B failure is now directly observable: its 5,040-point merged
component spans approximately 150 × 74 × 196 mm. Exact metric partition gives a
4,315-point oversized occluder and a 517-point supported cylinder, covering
95.9% of the parent together. The remaining sparse samples are not invented or
assigned by truth. Native automatic recovery is evaluated after combining the
fitter changes.

The new selection/holdout fixture uses disjoint deterministic seeds and camera
viewpoints, 2/4/6/10 mm gaps, varied dimensions/yaws, isolated supported shapes,
unknown ellipsoid, empty workspace, physical A/B occluders, true one-view cases,
an elevated link surrogate and temporal disappearance/reacquisition. The first
selection capture reused primitive paths across different USD types; twelve
gap scenes rendered invalid empty workspace clouds. That failed run and its
invalid probe remain under `outputs/multiview-iteration2/selection-capture/`.
Unique per-scene paths repaired the captures; visibility prerequisites now fail
the fixture explicitly. The second capture's 27 scenes and 52 nonempty oracle
clouds passed an independent audit at `selection-capture-v2/fixture_audit.json`.
Its original script exited during SimulationApp shutdown before final result
writing; the audit records that missing finalization. The launcher was unchanged;
the new benchmark now closes the app after writing results/provenance.

Selection one-view coverage was verified from saved depth, not assumed from
scene names: A-hidden had 0/468 target points (A/B); B-hidden had 94/0. All
acquisition timestamps and tokens survive replay. Oracle masks are used only
for diagnostics, matching and isolated fitting experiments, never provider input.

```bash
./run_isaac_python.sh scripts/benchmark_multiview_robustness.py --capture-only \
  --split selection --result outputs/selection-new/result.json \
  --artifacts-dir outputs/selection-new
./run_isaac_python.sh scripts/probe_multiview_segmentation.py \
  --methods voxel8,radius4,selected \
  --capture-manifest outputs/selection-new/capture_manifest.json \
  --fitter-package-dir outputs/multiview-iteration2/baseline_snapshot \
  --result outputs/selection-grouping-new.json
```

## Checkpoint C: fitter optimization

The retained fitter batches box transforms/quantiles, searches the complete
supported yaw interval at 5° spacing, then refines at 0.5° and 0.05°. Horizontal
cylinders cover the full 180° interval at 10°, then refine at 2° and 0.25°.
Repeated axes are skipped only within that fitting operation. Cross-section
projections and radial residuals are reused. Every hypothesis still uses the
same bounded cloud, with the original 1,200-point limit, robust radial solver,
confidence/coverage requirements, competing shapes and unknown margin. No
historical fit, simulator geometry or decisive-winner shortcut replaces current
evidence. There are no new dependencies or worker threads.

The serialized numerical comparison at
`outputs/multiview-iteration2/fitting-original-final/` replayed 114 saved clouds
three times per alternative. Mean/p95/worst fitting times were
44.16/62.72/78.88 ms originally and 27.22/41.29/51.85 ms retained: a 38.35%
mean reduction, with zero class changes across 342 paired rows. Box-only
optimization gave 37.71 ms. A 600-point cap gave 23.26 ms and preserved those
classes, but did not speed up the lower-density selection set or improve its
dimensions. A 300-point cap gave 20.22 ms but rejected a previously correct
rotated box, so reduced sampling was not retained. All alternatives and failed
rows are exported by `scripts/probe_multiview_fitting.py`.

Separate sparse/noisy stress replay retained all 297 outcomes: 288 perturbed
known shapes plus nine explicit unknown fixtures. Both versions correctly
classified 224 known shapes, conservatively rejected 64, and correctly rejected
all nine unknown fixtures. There were zero wrong primitive labels and zero
changed labels. This demonstrates preserved conservative rejection on this
corpus, not universal noise robustness. Artifacts:
`outputs/iteration2-fitting/production-stress/`.

On the 52 valid selection oracle clouds, both versions classified 52/52 correctly
(including the expected unknown). Retained mean fitting time was 20.41 versus
35.63 ms. Mean dimension error changed 1.064 to 1.056 mm, center error 0.248 to
0.236 mm and supported box/cylinder yaw/axis error 0.103° to 0.009°. These are
isolated oracle fitting results, not automatic segmentation or live throughput.
Tiny localization changes are secondary to preserved classification. Artifacts:
`outputs/multiview-iteration2/fitting-selection-v2/`.

The curved-dimension weakness remains. Saved sphere surfaces already lie inside
their nominal analytic geometry: mean per-cloud median radial errors were
−0.692/−0.768/−0.771 mm for A/B/fused. Raw fitted radius bias was
−0.791/−0.746/−0.822 mm; final fitter voxelization changed mean radius by at most
about 0.0023 mm. This is consistent with faceted rendered surfaces and viewpoint
sampling, without proving a simulator defect. No arbitrary radius inflation or
simulator-specific correction was added. Full radius diagnostics are retained.

```bash
./run_isaac_python.sh scripts/probe_multiview_fitting.py \
  --baseline-pythonpath outputs/multiview-iteration2/baseline_snapshot \
  --geometry outputs/multiview-geometry-final \
  --detection outputs/multiview-detection-01 \
  --variants baseline box_only improved sample600 sample300 --repeats 3 \
  --profile --radius-diagnostics --output outputs/fitting-comparison-new
```

The production method was frozen before accessing holdout captures. Fitting and
segmentation selection used only original and selection data. Subsequent holdout
failures will be retained without tuning the algorithms on them.

## Combined evaluation method

`scripts/benchmark_multiview_robustness.py` feeds the same three distinct native
frames to unchanged V1 A/B, the preserved baseline and the frozen improved
provider. Temporal scenes include six frames: confirmation, motion, one-frame
absence and reacquisition. Replay keeps native acquisition times; it does not
claim current WorldModel freshness from old buffers. It checks pre-third-frame
confirmation, unchanged-frame publication and exact timestamp preservation.
The original native contract separately exercises updater health and lifecycle.

Each split has 27 scenes and 52 scored target observations in the clean case.
Six representative scenes also receive deterministic 0.5 mm depth noise,
Camera B calibration offsets of (+0.5, −0.3, +0.2) mm (0.616 mm translation norm)
and 0.05° world-Z rotation, and 35% pixel removal. Temporal
continuations are included in those denominators. Thus clean plus perturbations
have 103 expected observations per method and 45 scene/variant combinations.
These perturbations are numerical stress tests, not a physical sensor model.

Scoring uses one-to-one center matching within 40 mm. Exclusive nearest-surface
membership within 1.5 mm avoids counting close-gap points twice. At least eight
points and 10% cloud support define a mixed accepted cloud or a false split.
Oracle component diagnostics, accepted observations and isolated oracle fitting
are exported separately. Missing and unknown observations stay in accuracy
denominators; localization/dimension metrics are conditional on available fits.
Geometry scores are not calibrated probabilities. Physical one-view coverage
is reported only when the hidden camera has zero target points and the other
at least eight. Every per-scenario row, confidence/margin, candidate residual,
radius/length/yaw/axis error, rejected component and runtime error is retained.

The matched replay timings are offline CPU measurements, never live Isaac
throughput. The original 24-scene benchmark measures the source alone with real
acquisition and eight rendered steps between each of 72 samples, excluding V1
comparison calls. A second live experiment compares baseline/improved separately
on five harder holdout scenes after three warmup samples per source. No Kit
worker threads are introduced, and no two native Isaac processes run together.

## Checkpoint D: measured combined results

The unchanged original 24-scene automatic benchmark completed with 87/88 correct
versus the reproduced baseline's 84/88. **Three of the four old misses were
recovered; one remains** (`1_hidden_B_1`, upright cylinder). There were zero new
unknowns, false positives, duplicate or mixed accepted observations in these
original scenes. V1 A/B retained 82/88 and 80/88 correct respectively. Artifacts:
`outputs/multiview-iteration2-baseline-detection/` and
`outputs/multiview-iteration2-detection/`.

The old report's approximately 4.5 mm cylinder/occluder gap is nominal. Its
seeded target translation changes the actual gap while the occluder stays fixed.
Exact replay of the retained failure frames shows the last case has a 1.671 mm
physical gap and 2.755 mm nearest measured sample spacing. The 4 mm proposal
graph therefore retains one dense 7,075-point component with 568 cylinder and
6,331 occluder surface samples; the cut is refused before any child fitting.
Resolved examples have 4.021 and 7.750 mm physical gaps. This is an unresolved
near-2 mm case in this arrangement, not a fitting/classification regression or
evidence of a universal resolution limit. Other 2 mm pairs do separate. Full
diagnosis: `outputs/iteration2-segmentation/remaining-native-failure-diagnosis.json`.

Mean center error changed 0.221 to 0.196 mm; dimension error regressed slightly,
1.136 to 1.150 mm. On the 84 jointly correct targets, centers changed 0.221 to
0.202 mm and dimensions 1.136 to 1.151 mm, so the dimension regression cannot be
explained away by newly recovered objects. Mean supported yaw/axis error changed
0.137° to 0.017°. Classification/detection gains come from separation, not from
new claims of a better primitive classifier: the frozen-fitter replay preserved
all original class decisions. In its controlled 24-cloud-per-view subset, A was
22/24 correct and B/fused 24/24 before and after optimization. Complementary
surfaces retain their occlusion benefit; they do not guarantee better dimensions.
V1 retains its different projection convention, so V1 comparison alone does not
isolate the causal contribution of fusion.

Live original-scene timing, **mean / p95 / worst in ms**, 72 samples per version:

| Phase | Reproduced baseline | Improved |
| --- | ---: | ---: |
| Acquisition | 5.38 / 6.72 / 10.45 | 5.41 / 6.58 / 7.06 |
| Projection/filtering | 15.25 / 19.76 / 20.29 | 15.33 / 19.85 / 20.27 |
| Fusion | 4.92 / 12.12 / 12.45 | 2.86 / 7.08 / 7.25 |
| Segmentation | 39.64 / 69.99 / 70.28 | 25.89 / 57.50 / 58.44 |
| Fitting phase | 170.88 / 244.99 / 252.84 | 111.56 / 149.91 / 153.24 |
| Tracking | 0.149 / 0.196 / 0.217 | 0.151 / 0.182 / 0.194 |
| Total processing | 236.23 / 296.72 / 312.32 | 161.20 / 216.08 / 222.29 |
| Eight rendered steps | 83.23 / 102.23 / 129.62 | 79.96 / 93.88 / 97.06 |
| End-to-end loop | 319.47 / 377.84 / 407.54 | 241.17 / 292.91 / 299.73 |

Effective measured rate rose **3.13 to 4.15 Hz**: 31.8% less mean processing
time and 32.5% higher loop rate. The original workload still does not reach
5 Hz. The fitting phase includes fitting-related support/view loop overhead,
as in the old measurement; child model-fit time is counted there, while pure
partition work is counted under segmentation. Per-view voxelization is no
longer redundantly repeated during fusion. Peak RSS was 5,392 versus 5,257 MiB
for the whole Isaac benchmark process, not isolated provider memory.
The original worst loop was elevated clutter (407.5 ms); the improved worst
was strong A occlusion (299.7 ms). Repeated static captures are not independent
random trials and no statistical significance claim is made.

The separate live five-scene holdout experiment retained 25 samples per version
after three warmups per scene/source. Mean/p95/worst processing was
135.6/261.0/262.5 ms baseline and 92.7/171.2/177.1 ms improved; corresponding
render-plus-processing loop times were 221.4/350.3/357.5 and
178.0/247.9/261.3 ms, or **4.52 to 5.62 Hz**. The isolated four-object scene
remained slower: mean loop 346.5 to 247.8 ms. This is a different scene mix,
not evidence that the original workload or arbitrary clutter meets 5 Hz.
All nine timing phases, per-scene distributions and raw CSV are retained at
`outputs/multiview-iteration2/figures/live_holdout_comparison.json` and
`live_holdout_timings.csv`; source data are in
`outputs/multiview-iteration2-live-holdout/`.

Matched clean automatic replay (52 expected per split, including one expected
unknown; mixed means published clouds including explicit unknowns):

| Split | Method | Correct | Missing | Mixed | False positives |
| --- | --- | ---: | ---: | ---: | ---: |
| Selection | V1 A | 33 | 10 | 8 | 0 |
| Selection | V1 B | 39 | 7 | 5 | 0 |
| Selection | Original multi-view | 24 | 14 | 14 | 1 |
| Selection | Improved multi-view | 42 | 5 | 5 | 1 |
| Holdout | V1 A | 35 | 9 | 7 | 0 |
| Holdout | V1 B | 40 | 7 | 5 | 0 |
| Holdout | Original multi-view | 22 | 15 | 15 | 0 |
| Holdout | Improved multi-view | 44 | 4 | 4 | 0 |

Known-primitive-only accuracy was 23/51 to 41/51 on selection and 21/51 to 43/51
on holdout. Clean-plus-stress totals below include four expected unknowns per split.

The one selection false positive is the same 26-point off-target panel strip
in both multi-view implementations. It is an unknown; there is no new false
positive relative to the baseline. There were zero false splits, duplicate IDs,
identity switches or runtime errors across either complete replay. All mixed
published clouds were explicit unknowns, never known primitives. Remaining
holdout clean failures are all four sphere pairs: one unknown merged cloud and
one missing sphere each. All 24 close box/cylinder targets were correct on
holdout; selection retained the 2 mm box/cylinder failure.

Per gap, correct close targets out of eight (four pair types):

| Gap | Selection old → new | Holdout old → new |
| --- | ---: | ---: |
| 2 mm | 0 → 4 | 0 → 6 |
| 4 mm | 0 → 6 | 0 → 6 |
| 6 mm | 2 → 6 | 2 → 6 |
| 10 mm | 2 → 6 | 0 → 6 |

With all perturbations included, selection improved 63/103 to 85/103 and holdout
61/103 to 85/103. No row previously correct under the old multi-view source
became incorrect. Holdout perturbation subsets each have 17 expected targets:
baseline was 13/17 for noise, calibration and sparse variants; improved was
13/17, 15/17 and 13/17 respectively. Thus noise/sparsity remove the clean
4 mm box/cylinder separation gain. These failures remain in the totals.
Both multi-view versions kept all four expected unknown observations per split
unknown. V1 B missed one of those selection fixtures. Empty,
isolated, elevated-link and temporal improved cases were correct. Both multi-view versions
correctly classified the true one-view targets; holdout visibility was
0/487 and 115/0 points for A-hidden/B-hidden cases.

On the 21 holdout known targets correctly classified by both versions, mean
center error changed 0.258 to 0.240 mm, dimension error 1.221 to 1.193 mm and
box yaw 0.321° to 0.020° (seven boxes). Cylinder radius/length/axis were unchanged
in that paired set. The correctly classified holdout known shapes had mean
consistency score 0.971 and score margin 0.957; remaining clean unknown merges
had confidence at most 0.001464. Scores remain geometric consistency measures,
not probabilities; these diagnostics do not establish real-world calibration.

Each split passed 294 source replay checks: zero early confirmations, stale
observation timestamps or repeated-frame publications. Native timestamp/update
health, uncertain/held identity protection, placement hints, reinitialization and
cleanup are additionally tested by the unchanged native source contract.
The new native recovery test requires all four correct objects in the three
resolved old failure fixtures and permits only the upright cylinder's omission
in the fourth. It asserts three-frame confirmation, stable IDs, no invented
known geometry/mixed cloud, unchanged-frame silence and center errors below
the original 10 mm source-contract bound. It passed 12 confirmed samples with
maximum center error 0.721 mm and exports exact failure frames. No original
acceptance threshold was changed to obtain this result.

All automatic rows, per-scenario summaries, confidence/fit residuals and geometry
parameters are in `outputs/multiview-iteration2/{selection,holdout}-evaluation/`
(JSON and CSV). Start/end source hashes stayed identical and record the clean
freeze commit `bf066b6`. Paired geometry and remaining-failure joins are in
`evaluation_analysis.json` and `correct_only_geometry.json` at the iteration-two
root. Reproduction and standalone figures:

```bash
./run_isaac_python.sh scripts/benchmark_multiview_detection.py \
  --result outputs/detection-improved-new/result.json \
  --artifacts-dir outputs/detection-improved-new --repeats 2
./run_isaac_python.sh scripts/benchmark_multiview_robustness.py --capture-only \
  --split holdout --result outputs/holdout-new/result.json --artifacts-dir outputs/holdout-new
./run_isaac_python.sh scripts/benchmark_multiview_robustness.py \
  --replay outputs/holdout-new \
  --baseline-pythonpath outputs/multiview-iteration2/baseline_snapshot \
  --result outputs/holdout-evaluation-new/result.json --artifacts-dir outputs/holdout-evaluation-new
./run_isaac_python.sh scripts/benchmark_multiview_robustness.py --capture-only --split holdout \
  --scenario gap_4_box_cylinder --scenario gap_2_sphere_sphere --scenario isolated \
  --scenario unknown --scenario strong_A --native-timing-samples 5 \
  --baseline-pythonpath outputs/multiview-iteration2/baseline_snapshot \
  --result outputs/live-holdout-new/result.json --artifacts-dir outputs/live-holdout-new
./run_isaac_python.sh scripts/summarize_multiview_iteration2.py \
  --baseline outputs/multiview-iteration2-baseline-detection \
  --improved outputs/multiview-iteration2-detection \
  --baseline-pythonpath outputs/multiview-iteration2/baseline_snapshot \
  --robustness outputs/multiview-iteration2/selection-evaluation \
  --robustness outputs/multiview-iteration2/holdout-evaluation \
  --live-robustness outputs/multiview-iteration2-live-holdout --output outputs/iteration2-figures-new
```

`outputs/multiview-iteration2/figures/` contains original accuracy/latency,
selection/holdout accuracy and before/after hidden-B cloud PNGs, plus matched
original-scene CSV and JSON. Artifact directories stay ignored. Benchmark PASS
means complete experiment data were written; all classification failures are
still present. Source-contract/regression PASS requires its explicit assertions.

## Checkpoint E: validation and remaining application blockers

The chosen method is one conservative production hierarchy, with numerical
orientation-search optimization. Smaller fixed thresholds, density/normal gates,
the known-parent veto and aggressive sampling were investigated independently
and rejected where they lost sparse objects, split surfaces or lost correct
classification. No heavy learned system, alternate runtime pipeline, dependency
change, unsafe Kit thread, simulator oracle or tracked-geometry substitution was
introduced. Only experimental `fitting.py`, `segmentation.py` and `provider.py`
production files changed. New utilities/tests/docs are separate. V1, ground
truth, WorldModel/updater, planning, manipulation, backend, BT/runtime, GUI,
existing USD/TOML, launchers and environment pins remain unchanged. Existing
untracked `models/` was not modified. Experimental benchmarks command no robot
motion; the unchanged V1 robotics regressions exercise their original motion.

The remaining blockers before relying on this source in an application are
specific: unresolved sparse sphere-pair segmentation even at 10 mm in these
viewpoints; the 1.67 mm original miss; noise/density sensitivity of a 4 mm
box/cylinder cut; the existing unknown panel-fragment false positive; curved
dimension bias and the original-scene dimension regression; and scene-dependent
latency that remains below 5 Hz on the original workload. More fragmented
clutter can require extra model fits. There is no universal gap-resolution or
worst-case real-time guarantee.

Table support is a configured prior, not measured hidden geometry. Supported
orientation remains yaw-only boxes and upright/horizontal cylinders. Touching,
intersecting, stacked and arbitrarily tilted objects, real moving robot links
near the support plane, broader calibration error, real sensor noise, fast
motion and long occlusion have not been established. The elevated surrogate and
small synthetic perturbations are bounded tests. Further perception-only
experiments are justified; supervised application tests should first address
those failure modes and dimension tolerances. No multi-view Pick/Place or GUI
integration was performed or claimed.

All experiments used the established workstation wrappers, NumPy 1.26.4 before
SimulationApp, Isaac Sim 5.1.0-rc.19 and py_trees 2.6.0. No `--no-ros-env`, pip
install, dependency/environment mutation or competing native process was used.
The source freeze is local commit `bf066b6`, tree
`11fdd9baf828fdf5bf51917bf2acfde37aba2520`; recorded per-file hashes permit
content verification independently of commit metadata.

Executed validation (all native processes sequential; logs, JSON and exported
outputs inspected):

| Check | Result | Artifact directory under `outputs/` |
| --- | --- | --- |
| Before changes | 320 units + eight suites + architecture PASS | `multiview-iteration2-baseline-unit/run-kw3e885w/` |
| Original native source contract before changes | PASS | `multiview-iteration2-baseline-native/run-nfp_32pv/` |
| Final offline | 340 units + eight suites + architecture/syntax PASS | `multiview-iteration2-unit/run-e4b67lni/` |
| Native recovery and unchanged source contract | Both PASS | `multiview-iteration2-native/run-68mnt35r/` |
| Unchanged V1 discovery/manipulation integration | Both PASS | `multiview-iteration2-v1-isaac/run-romoiv1m/` |
| Unchanged benchmark/grasp/multi-object/placement/reactive/recovery | All six PASS | `multiview-iteration2-v1-regression/run-wepe89gj/` |

The original robotics benchmark completed 12 perception poses and 4/4 successful
Pick/Place trials. Maximum physical target error was 10.797745 mm against the
unchanged 25 mm gate. Oversized grasp and occupied placement rejection still
issued no forbidden commands; reactive pick replanned once and succeeded;
recovery rejected an unavailable pose, then succeeded after restoration.
No acceptance thresholds, controller settings or V1 algorithms were changed.
The added 20 unit tests cover exact broad/metric graph connectivity, conservative
splits, fitting orientation and unknowns, reproducible research fixtures/scoring,
render validation and replay timestamps. The unchanged provider tests cover both
held/uncertain branches; the native contract exercises uncertain held identity.

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview \
  --artifacts-dir outputs/multiview-iteration2-native-new
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/multiview-iteration2-v1-isaac-new
./run_isaac_python.sh scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/multiview-iteration2-v1-regression-new
```
