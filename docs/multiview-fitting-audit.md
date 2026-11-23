# Multiview surface fitting: algorithm-value audit and refactor

The fitter is smaller, with the demonstrated geometry and qualification
preserved. `fitting.py` falls from **441 to 366 lines** (−75, 17.0%). The four
affected production files total **1,140 → 1,056 lines**. Detailed research
metadata is removed; robust fitting, shape competition, conservative dimensions
and cylinder evidence remain.

Started clean on `refactor/thesis-cleanup` at
`41034d236789dd1595c5999d7e67ca0676ec5579`, matching origin. A detached checkout
at that commit supplied the baseline. Main remains
`5b61d4442d74e3afd76420ffc11e8e3ab0b21937`. No merge, old-cleanup code import,
dependency change, Isaac launch or worker/freshness change was made;
`isaac51` remained stopped.

## Audit and decisions

Callers, downstream metadata consumers and fixed-input alternatives were examined
before production edits. “Essential” describes the mathematical function;
it does not require retaining its original implementation.

| Mechanism | Classification | Decision and reason |
|---|---|---|
| `PrimitiveFit` | Essential | Keep geometry, center and world-axis size: tracking uses world bounds; manipulation also needs local box dimensions and yaw or cylinder axis. |
| `unknown_fit()` | Essential | Keep current bounds, empty-cloud behavior, reason and optional support prior. Remove unused scores argument, point-count reporting and repeated input validation. |
| `orientation_metadata()` | Optional | Delete the entire explanatory report and provider assignment. GUI and grasp planning read actual yaw, axis and dimensions directly. |
| `_radial()` | Essential | Keep normalized four-pass robust least squares and conditioning checks; compute the squared-radius right-hand side once. Partial-arc outliers justify reweighting. |
| `_hull_area()` | Essential | Keep circular-cap rejection. Remove the redundant sort after lexicographically sorted `unique`. |
| `_box_distance()` | Essential | Keep shared surface distance, also used by segmentation and mapping; no duplicate multiview implementation to remove. |
| `_box_search()` | Useful but replaceable | Keep batched full-cloud bounds and yaw scoring; reuse those transformed samples for side scoring instead of transforming them twice. |
| `_radius_evidence()` | Essential | Keep support-contact and per-view radius evidence. Remove the unused fifth-percentile diagnostic from finalization. |
| Cylinder observability | Essential | Keep curvature and cap-interior requirements; return one failure reason instead of a boolean plus measurement dictionary. |
| `_sphere()` | Essential | Keep curvature fit, plausible radius, support-plane check and residual/coverage score. |
| `_box()` | Essential | Keep top/side selection, hull coverage, dimensions and all three yaw stages. Off-grid angles benefit from the last stage. |
| `_cylinder_axis()` | Useful but replaceable | Specialize cross-sections to supported upright/horizontal axes, eliminating arbitrary 3D reference-frame construction and unused upright axial percentiles. |
| `_lying_cylinder()` | Essential | Keep full horizontal search, deduplication and both refinements. The final stage improves axis/dimension accuracy. |
| `finalize()` | Essential | Keep raw-fit copying, measured envelopes, support contact, view disagreement and parameter qualification. Remove measurement-report construction. |
| `fit()` / candidate competition | Essential | Keep all hypotheses, per-class maximum scores, confidence threshold, runner-up margin and UNKNOWN. Remove residual dictionaries and metadata readback. |
| Detailed uncertainty and duplicate geometry metadata | Redundant | Delete the producer/consumer chains; preserve actual geometry and the final qualification boolean. |

## Bounded algorithm experiments

The local audit froze 12 clouds: four native static objects, the committed
487-point occluded Square, off-grid/wrap boxes, noisy sphere/upright cylinder,
two lying cylinders and a circular-cap negative. Alternative functions were
patched only inside the offline audit process; production was still unchanged.

| Alternative | Measured consequence | Decision |
|---|---|---|
| Replace hull area with rectangular footprint area | Circular cap plus 16 wall points changes UNKNOWN (confidence .0259) to cube (1.0), with finalized parameters accepted. | Reject simplification. |
| Remove box .05° refinement | Off-grid/wrap yaw changes .25°; centers shift up to .136 mm, dimensions .244 mm, confidence drops up to .00778. Native authored angles conceal this loss. | Keep refinement. |
| Remove lying-cylinder .25° refinement | Two axes change .25°/.50°; maximum world-size changes .282/.737 mm; confidence drops .00556/.00966. | Keep refinement. |
| Use one or two radial passes | Native/noisy controls change little, but partial-arc outlier control below loses accuracy. | Keep four passes. |

Radial control: 64 points on a 120° arc of a 30 mm circle, 0.1 mm Gaussian
noise (seed 113), with four points shifted 3 mm in X. Center/radius errors:
**1.111/1.014 mm** with one pass, **.553/.517 mm** with two,
**.243/.236 mm** with four. The retained regression checks geometric accuracy
under this contamination, rather than an iteration count. A second new test
checks that the circular cap cannot become a box.

No thresholds or angular steps were changed. Circle/sphere estimation, Gaussian
residual scoring, support priors, outer-radius calculation and all geometric
rejection tolerances remain. The cylinder frame change is a coordinate rotation
or reflection of the same circle problem for the supported axes, not a new fit.

## Dependency cleanup and scope

Removed candidate `center`, `size_world`, `residual_p80_m`, `coverage`, final
`residuals_m`/`fit_points`, and nested `parameter_uncertainty` measurements
(`surface_padding_m`, `dimension_envelope_m`, radius counts/envelopes,
contact/per-view radii, curvature span and cap counts). Those numerical quantities
still take part in decisions where needed; their reporting dictionaries do not.
`parameter_observable` remains authoritative; a flat `parameter_reason` explains
failed finalization. Successful finalization needs no duplicate success string.

Removed provider `orientation_observability`, `surface_parameters_observable`,
`measured_box_geometry` and duplicate geometry coordinates. Tracking, WorldModel,
GUI and primitive grasp planning use the retained geometric fields and graspable
flag. V1 perception's independent coordinate metadata is untouched.

The only necessary `surface_fit` consumer was box observability's measured top
height. Replace the whole raw-fit snapshot with scalar `fitted_top_z`, recorded
before outward expansion. The small `assess()` read change preserves its raw-fit
fallback and all camera-boundary mathematics. Mapping's shared box-search caller
changes two lines to pass a side mask. There is no broader mapping, provider,
segmentation or observability refactor.

Removed repeated finite-cloud/support-height/settings checks inside fitting.
Internal finalization and budget fallback now require finite Nx3 arrays and
valid scene settings, as supplied by preprocessing/configuration. Arbitrary
nonfinite direct-helper inputs and friendly invalid-setting exceptions are no
longer supported. Numerical degeneracy checks, minimum meaningful sample counts,
size plausibility, UNKNOWN and incomplete-geometry rejection remain.

Obsolete orientation metadata assertions and reporting consumers were removed or
updated. The frozen semantic-provider fixture remains byte-for-byte unchanged;
its existing in-memory adaptation skips retired metadata, while still comparing
exact geometry, confidence, identities, graspability and retained metadata.

## Fixed native-input comparison

Both versions use the original saved 640×480 `session-2` static/pre-grasp depths,
calibration, camera identities, tokens and acquisition times. Manifest:
`outputs/multiview-performance-ab-20261122/input.json`, SHA-256
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.

All native detections match **exactly** before/after: shape, center, local
dimensions, yaw, radius, confidence, graspability and both contributing cameras.
Coarse clouds, numerical fit inputs, four fit calls and timestamps also match.
All four objects remain graspable. Selected published geometry below is identical
in both versions; coordinates are meters, dimensions millimeters.

| Capture / object | Center XYZ | Dimensions | Yaw | Confidence |
|---|---|---|---|---:|
| Static Rectangle | (.539937, .159903, .024001) | (42.234, 68.003, 48.002) | 57° | .999999925 |
| Static Square | (.359878, −.160057, .027501) | (55.273, 55.003, 55.001) | 25° | .999999879 |
| Static LargeSphere | (.539973, −.160087, .030271) | (60.542, 60.542, 60.542) | — | .948450102 |
| Static SmallSphere | (.360037, .159931, .022157) | (44.314, 44.314, 44.314) | — | .970752962 |
| Pre-grasp occluded Square | (.360041, −.160406, .027501) | (55.273, 55.773, 55.001) | 25° | .999999864 |

The other three pre-grasp objects equal their static outputs. The occluded
Square retains **.408 mm center / .773 mm maximum dimension error** and all
six qualified bounds under the existing measured-boundary/support model.

Across all 12 frozen clouds, classifications and parameter qualification agree.
Native, box, sphere and lying-cylinder fitted geometry is exact; the noisy
upright cylinder differs only at floating-point roundoff (maximum final size
delta 6.94e−17 m). Maximum per-class score delta is 3.75e−16.

Historical WorldModel replay uses three distinct original static/pre-grasp/home
exposures. It passes with four persistent IDs, publication counts **[0, 0, 4]**,
all detections qualified and original capture hashes unchanged. Its clock is
archived acquisition time plus measured computation; this does not establish
live worker freshness or repair the previously recorded L4 scheduling issue.

## Validation and timing

- Required fitting selection: **16 tests, 3.760 s**, passed. Covers rotated
  boxes, four lying axes, noisy/sparse curves, unsupported surfaces, robust arc,
  circular-cap rejection, support contact and incomplete/full cylinder caps.
- Segmentation/readiness/provider selection plus frozen multiview equivalence
  and mapping `NumericalTests`: **29 tests, 10.649 s**, passed. Shared 24-fit
  budget/caching, missing-band UNKNOWN and native Square recovery remain.
- Direct clear-box observability check: **1 test, .135 s**, passed.
- Existing six frozen negative/control scenes, four observations each: outputs,
  confirmation counts and fit counts match the prior recovery result. Nearby
  rotated/axis-aligned boxes remain UNKNOWN and ungraspable; 20% missing-depth
  sphere and separately connected/tangent foreground controls retain one correct,
  graspable sphere. No new confident merged box appears.

CPython 3.11.18, NumPy 1.26.4, same interpreter/environment and 12-thread OpenBLAS.
One sequential before/after processor comparison, two warm-ups and five samples
per capture/version, using the same updated harness:

| Metric | Before | After |
|---|---:|---:|
| Static p50 / p95 | 422.67 / 422.81 ms | 406.90 / 408.76 ms |
| Pre-grasp p50 / p95 | 450.49 / 461.04 ms | 426.37 / 429.76 ms |
| Static fitting p50 | 329.74 ms | 312.52 ms |
| Pre-grasp fitting p50 | 306.73 ms | 285.26 ms |
| Fits per compute, either scene | 4 | 4 |

The small sample shows no slowdown, with medians 3.73%/5.35% lower. This is
descriptive evidence, not a hardware throughput or live-rate claim. No full
suite, new native capture or historical benchmark matrix was run.

## Reproduction and next cleanup

Local preserved evidence: `outputs/fitting-audit-20261123/` contains the detached
baseline, audit script, 12 frozen clouds, algorithm variants, radial outlier
samples, four timing JSONs, after-cloud comparison, WorldModel replay and six
control results. `clouds.npz` SHA-256 is
`bd18e2c7723b63253861ee40543613e35aff767eccb0218e5debf4496113c803`.
The existing tracked performance harness adds yaw, axis, radius and length to its
output for both checkouts. No new benchmark framework was added.

From each checkout, with the interpreter above and
`PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps`, invoke the absolute
path to this checkout's `tests/multiview/performance_ab.py compute`, passing the
same absolute `--data` manifest, `--scenario static` or `pregrasp`, the checkout's
`--expected-head`, `--warmup 2 --repeats 5` and a new `--output`. Original inputs
are hash-checked and output overwrite is refused. The retained two new tests
reproduce the decisive radial/hull controls without local audit artifacts.

Observability's large `geometric_observability` dictionary remains for the next
authorized task. Its meaningful outputs are the qualification decision and
conservative box geometry; counters, prose and nested per-camera diagnostics
should receive their own end-consumer audit. `fitted_top_z` currently exists
solely for its necessary top-plane measurement. Keep that distinction when
cleaning the remaining report structure. Stop here for review.
