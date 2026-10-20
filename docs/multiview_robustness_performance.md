# Multi-view robustness and performance, iteration two

This research continues `research/multiview-rgbd-fusion` at
`42e3258ac5a5c79bba766262357e77f80d158c14`. The new branch is
`research/multiview-robustness-performance`; its PR targets the experimental
parent. The [first report](multiview_rgbd.md), original benchmark scenarios and
previous artifacts remain the comparison baseline. V1 remains the default.

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
gates also lost two, increased computation, and did not improve close-pair
separation. Tangent gates added nine false splits on the 32 translated pairs.
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
Resource limits bound processing, as in the original provider.

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
