# Occluded-box recognition recovery

The saved native pre-grasp Square is now recognized as an accurate, graspable
cube, with **exactly the same detection geometry as original main**. The static
four-object results remain identical. Production change: **213 → 222 lines** in
`segmentation.py`, with no provider or fitter changes.

## Starting point and reproduction

Started clean on `refactor/thesis-cleanup` at
`58859c32cfd788bef55cd61d185ce196f389704d`, matching origin. Original comparison:
`5b61d4442d74e3afd76420ffc11e8e3ab0b21937`, using its existing detached worktree.
Main is unchanged; no branch was switched or merged.

Before editing, the existing `tests/multiview/performance_ab.py` reproduced the
failure on the original `session-2/pregrasp-{0,1}.npz` files: UNKNOWN,
`unresolved_components`, 2.758 mm center error, 17.697 mm dimension error,
ungraspable. Original main returned the known box with .408/.773 mm errors.
Fitting and observability source are byte-identical to original main.

The input manifest remains
`outputs/multiview-performance-ab-20261122/input.json`, SHA-256
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.
Complete depth buffers, native calibration, camera identities, tokens and original
acquisition times were reused unchanged. No synthetic replacement frames were
used for the primary before/after comparison.

## Root cause and recovery condition

The 487-point native box has retained 4 mm components [291,37,38]: only 75.15%
coverage after the 25-point component cutoff. The sparse fallback already fits
the entire cloud once, but previously retained only a sphere. Thus the valid
box fit was discarded before downstream geometric qualification.

High confidence and low residual cannot resolve this alone: nearby boxes also
produce an excellent but incorrect single-box fit. Their **measured horizontal
connectivity** distinguishes these cases:

| Candidate | 3D retained coverage | Largest XY component / all samples | Whole fit | Decision |
|---|---:|---:|---|---|
| Native occluded Square | 75.15% | 477/487 = **97.95%** | cube | Recover cached fit |
| Nearby rotated boxes | 71.22% | 531/1011 = **52.52%** | cube | UNKNOWN |
| Nearby axis-aligned boxes | 67.44% | 584/1041 = **56.10%** | cube | UNKNOWN |
| Missing-band box | 100% | 1156/2312 = **50%** | cube | Existing ambiguity rejection remains |

The extension applies only to the existing 2–8-component, below-90%-coverage
fallback. It requires:

1. A whole-cloud primitive selected by the unchanged fitter, which already
   compares competing shapes and applies its confidence/margin requirements.
2. The existing 95th-percentile surface residual ≤3 mm on the **original cloud**.
3. For a box, the existing supported-cloud test must pass, including observed
   low-side support evidence and plausible extent.
4. Project the original cloud onto XY, and reuse the exact-radius graph with
   the same 4 mm radius and minimum component size 1. At least 95% of **all
   samples** must belong to one projected component.

The 95% criterion follows the existing single-model consistency fraction; neither
it nor the radius was fitted to GT dimensions. XY projection tests measured
connectivity without requiring vertical sampling patches to connect in 3D.
It does not add FREE evidence or certify hidden occupancy. The nearby-box gap
survives projection, so its confident merged hypothesis is rejected.

The returned cloud is the same original array; no points are reassigned or
removed. The copied projected array is used only for qualification. The whole
fit is cached and reused by the provider. Sphere recovery is unchanged, and a
continuous one-component box still takes the ordinary fitting path.

## Existing geometric uncertainty remains authoritative

The box continues through unchanged `SurfaceFitter.finalize()` and
`observability.assess()` with its actual per-camera samples and depth buffers.
No confidence, `parameter_observable`, or graspability value is manually set.

The recovered native pre-grasp box passes current measured-envelope finalization
and all six existing bound checks: the four horizontal sides have calibrated
boundary evidence, the upper bound has measured top evidence, and the lower bound
uses the configured support plane with current low-side samples. All six bounds
are recorded as bounded; final parameter observability is true. Graspability
therefore comes from the normal provider rule, not a test exception.

No 4.5 mm retry, child reassignment/refitting, SAT, camera-gap subsystem, additional
worker, new interface or dependency was restored. The provider's 24-fit cap,
fitter thresholds, resolution, stride, voxels, timestamps and one-second
freshness limit are unchanged.

## Before/after native processor replay

CPython 3.11.18 and existing NumPy 1.26.4 on the same Xeon/L4 host. Each timing
uses two warm-ups and five measured repetitions through the existing comparison
harness. These are bounded descriptive samples, not a new benchmark campaign.

| Metric | Before | After |
|---|---:|---:|
| Square classification | UNKNOWN | cube |
| Square center error | 2.758 mm | **.408 mm** |
| Square maximum dimension error | 17.697 mm | **.773 mm** |
| Square confidence | 0 | .999999864 |
| Square graspable | no | **yes** |
| Pre-grasp compute p50 | 431.72 ms | **452.88 ms** |
| Pre-grasp compute p95 | 448.63 ms | 454.43 ms |
| Numerical fits per pre-grasp compute | 4 | **4** |
| Static compute p50 | 423.35 ms | 427.16 ms |
| Static results | four correct, graspable | **identical** |
| Segmentation lines | 213 | **222** |

The restored Square center is (.360041,−.160406,.027501) m; local dimensions are
(55.273,55.773,55.001) mm. Every pre-grasp detection, including the other three
objects, exactly matches original main's output. Original main took 718.37 ms
and 10 fitting calls in this reproduction.

Incremental pre-grasp cost is **21.16 ms (+4.90%)**, with no additional fits.
Metric-graph median increases 70.08→83.44 ms due to the footprint check;
observability increases 8.55→15.50 ms because the box now receives its normal
geometric assessment. The static path adds no graph or fit work and produces
identical geometry; its .90% timing difference is sample/context variation.
No general performance optimization was attempted.

## Negative controls and regression tests

| Control | Result |
|---|---|
| Missing-band box | UNKNOWN; finalized parameters remain unobservable |
| Nearby rotated / axis-aligned boxes, frozen 640×480 buffers | One UNKNOWN, ungraspable; no restored confident merged box |
| Nearby rotated boxes, closer viewpoints | Remains UNKNOWN and ungraspable |
| Disconnected sparse patches / excessive fragments | UNKNOWN |
| 20% missing-depth sphere | Known, accurate, graspable; one observation |
| Separate oversized foreground, coarse-connected or tangent captures | Supported sphere recovered; no additional confirmed foreground object |
| Truly metric-connected oversized background | Rejected as a whole |
| Continuous box | Ordinary fitting path; refinement performs no numerical fit |
| Shared fitting budget | Existing enforcement passes; native recovery caches one fit even with budget 1 |

The unit fixture `tests/fixtures/multiview_occluded_box.npz` is approximately
30 KiB: the complete native 487-point coarse cloud and a 1011-point analytic
nearby-box negative control. Its JSON sidecar records source hashes, extraction
and per-array hashes. The reference center selects which complete coarse cloud
to preserve for the unit test only; there is no oracle point mask, altered native
sample, or production reference-geometry input. Primary acceptance still uses
the complete original native images.

The new focused test recovers the native cloud, verifies cached single-fit use
and unchanged samples, repeats with rotated/translated geometry and a shifted
support plane, and rejects the separate-footprint negative. Existing tests
preserve ambiguity, sphere/foreground behavior and the global fit budget.

Final requested unit selection:

```bash
python -m unittest tests.unit.test_multiview_segmentation \
  tests.unit.test_multiview_segmentation_readiness tests.unit.test_multiview_provider
# 26 tests, 9.100 s, OK
```

Six selected frozen analytic controls were additionally replayed through the
processor/provider/WorldModel harness: sparse sphere, rotated/axis/closer box
pairs, and both oversized-foreground cases. This was a focused subset, not the
historical scenario matrix. All required outcomes above passed.

## Four-object association and scope of validation

`tests/multiview/occluded_box_replay.py` replays the actual archived static,
pre-grasp and home exposures once each, in their original chronological order.
Using the third saved native exposure permits normal three-hit confirmation
without duplicating an exposure or inventing new camera timestamps. All three
processor results contain two qualified cubes and two qualified spheres.

The offline test clock supplies each original acquisition time plus measured
compute duration. It exercises association and the unchanged updater in that
historical clock domain; it does **not** model queue time or claim live source
freshness. Original timestamps/tokens stay unchanged. Published object counts
are [0,0,4], with the same four tracker IDs throughout and four final graspable
WorldModel objects. No additional confirmed foreground object is produced.

**Live Isaac session: NOT RUN for this repair.** Validation uses real saved
Isaac observations offline. Worker scheduling, sustained throughput, Stop/Play,
and the known live stale-visibility problem were neither changed nor revalidated.
The existing container stayed stopped.

## Evidence and reproduction

Ignored output: `outputs/occluded-box-recovery-20261123/` retains before/current,
original-main and after replay JSON, qualification analysis, the six-control
subset and results, and historical-clock WorldModel/face-evidence results.
Original capture hashes were verified before and after replay. Final segmentation
SHA-256: `4e314d3a3fa802cdb12b2b7bdfba05788889fee9b5d0f4785463f03136368344`.

Run the existing comparison harness as documented in
[multiview-performance-ab.md](multiview-performance-ab.md), selecting `static`
and `pregrasp` from the unchanged input manifest. The association check is:

```bash
PYTHONPATH=.:/absolute/existing/.deps python3.11 \
  tests/multiview/occluded_box_replay.py \
  --data outputs/multiview-performance-ab-20261122/input.json \
  --result outputs/NEW-occluded-box/world-model.json
```

Only segmentation behavior, direct regression tests/fixture, the opt-in replay
check and documentation changed. Fitting, observability, provider, acquisition,
tracking, mapping, motion, workers and freshness policies are unchanged.

## Remaining limits

A connected measured footprint is an empirical single-object qualification,
not a proof that every hidden region is filled or every touching arrangement is
one object. Ambiguous layouts outside this narrow recovery remain subject to
existing conservative checks; general close-object separation is not added.
Unbounded geometry can still be ungraspable downstream. The live throughput and
freshness failures remain unresolved, and this replay does not establish overall
demo readiness. Stop for review; no fitting/observability work follows automatically.
