# Multiview geometric observability audit and refactor

`observability.py` is **238 → 162 lines** (−76, 31.9%). Together with its provider
consumer, production is **429 → 349 lines** (−80). The nested measurement report
is gone; supported box geometry and graspability are preserved on fixed inputs.

Started clean on `refactor/thesis-cleanup` at
`2006046f1cd036078b6e74c83bc4a903d0940a01`, matching origin. A detached baseline
is preserved locally. Main remains `5b61d4442d74e3afd76420ffc11e8e3ab0b21937`.
No merge, new dependency, Isaac launch, worker change or freshness adjustment.
The existing `isaac51` container remained stopped.

## Algorithm-value audit

Callers and fixed-input expected results were examined before production edits.

| Mechanism | Classification | Decision and reason |
|---|---|---|
| `_unique_pixels()` | Replaceable | Keep scalar encoding after measurement; remove arbitrary-int64 overflow fallback, unused height calculation and its artificial test. Row-major order also makes the later ID sort redundant. |
| `_view_boundary()` ray geometry | Essential | Keep calibrated top-plane intersections, valid-depth checks and outward boundary estimates; return one bound or `None`. |
| Pixel ribbon | Essential | Retain 48 edge samples and the ±3-pixel ribbon; they provide measured coverage around projected boundaries. |
| Adjacent occupied/free transitions | Essential | Keep eight-neighbor pairs, inward occupied anchors, positive finite depth and image-border checks. Distant free pixels or foreground cannot establish an exposed edge. |
| Three-bin edge coverage | Essential | Require independent transitions in all three separated edge segments; an isolated local transition is insufficient. |
| Top-plane spread compensation | Essential | Keep interior top-height spread and the oblique-ray lateral margin; uncertain plane height must not manufacture FREE evidence. |
| `_face_plane()` | Essential | Keep six-point minimum, horizontal/vertical spread, least squares and residual tolerance. Return slope/intercept/spread, used to tighten already witnessed bounds. |
| `_unbounded_status()` | Optional | Delete status aggregation and its pixel counters. Grasping needs unresolved geometry, not a ranked explanation of missing versus occluded pixels. |
| `assess()` qualification | Essential | Keep top/support evidence and independent camera bounds, then measured-face tightening. Replace the report with `(center, local_dimensions)` or `None`. |
| Nested result and metadata publication | Redundant | Remove methods, assumptions, resolution, axes/faces/cameras, counters, status strings, duplicated yaw and geometry serialization. Final numeric geometry and qualification already serve application consumers. |

## Result and dependency changes

The provider now calls `assess()` only for a cube and computes:

```python
parameter_observable = fitter_qualified and bounds is not None
```

Qualified bounds still update the actual center, local box dimensions and world
size, even when the fitter's independent qualification is false, matching the
previous behavior. Non-boxes retain their existing fitter/finalization decision.
An unresolved known cube remains known and ungraspable; UNKNOWN is not promoted.

`geometry.metadata['geometric_observability']` is no longer constructed or
published. WorldModel, GUI and grasp planning use actual geometry and graspability;
none used its nested measurements. Native replay/snapshot utilities now retain
numeric geometry and the final qualification instead. Observation camera identities,
contributing views, calibration and original timestamps remain unchanged.

The `assess(view_clouds=...)` argument existed only for diagnostic counts and is
removed. Per-view clouds remain in the provider where radius finalization and
contributing-camera detection actually use them. No replacement result class,
failure-report schema or production compatibility wrapper was added.

`fitted_top_z` remains: boundary rays use the measured top before outward geometry
expansion. Direct raw-fit calls retain the fitted center/height fallback. Fitting,
segmentation, tracking and spatial mapping production code are untouched.

The frozen semantic-provider fixture remains byte-identical. Its existing
test-only source adaptation translates the new bounds contract and excludes the
retired report from comparison, while exact geometry, confidence, identities,
graspability and retained metadata remain checked.

## Numerical simplifications and retained checks

- Reuse the already transformed full cloud for top-local samples instead of
  transforming those samples twice again.
- Deduplicated pixels are row-major and remain ordered after filtering. Neighbor
  lookup can search image IDs directly without sorting and remapping them.
- Remove foreground/counting dictionaries; the occupied predicate still excludes
  foreground, invalid and FREE measurements using the same depth thresholds.
- Return unresolved immediately when top/support evidence or any horizontal
  boundary is missing. All available cameras are still evaluated for a side;
  an occluded view cannot veto a valid companion view.
- Preserve the two-pass order: establish all camera intervals first, then tighten
  using side-plane slope across the other axis's current interval and twice the
  measured spread. Tightening order, tolerances and all sampling constants remain.

Repeated validation of internal finite XYZ arrays, positive fitted dimensions
and valid scene settings is removed. These are preprocessing/fitter/configuration
contracts. Real insufficient samples/frames, missing top/support, invalid camera
depth and unobserved physical edges still return unresolved. Enormous artificial
integer-coordinate extents are outside the pixel helper's supported image domain.

## Fixed-input results

Primary inputs are the original saved native 640×480 static/pre-grasp camera
buffers, with unchanged calibration, camera identities, tokens and acquisition
times. Manifest `outputs/multiview-performance-ab-20261122/input.json` SHA-256:
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.

All native detections match **exactly**: class, center, dimensions, yaw/radius,
confidence, graspability and contributing cameras. The four directly captured
box-assessment results also match exactly, independently of the provider's final
graspable flag. Coarse/fit clouds, four numerical calls and timestamps match.

| Native result | Before and after |
|---|---|
| Static scene | Two accurate, graspable cubes and two accurate, graspable spheres |
| Pre-grasp scene | Same four qualified objects; repaired Square remains graspable |
| Occluded Square center, m | (.360040528, −.160406280, .027500704) |
| Occluded Square local dimensions, mm | (55.272921, 55.773421, 55.001408) |
| Square center / maximum dimension error | .408 / .773 mm |
| Square yaw / confidence | 25° / .999999864 |
| Three- and five-camera controls | Center, dimensions, world size and graspability exactly match their baselines |

The historical static/pre-grasp/home WorldModel replay passes with four stable
IDs and publication counts **[0, 0, 4]**. Its clock is original acquisition time
plus measured computation. This is offline validation, not a live queue/freshness
claim; the earlier live throughput issue remains outside this task.

Sixteen fixed numerical cases were captured before editing and compared after.
Input hashes and supported/unresolved outcomes match; all supported conservative
centers/dimensions match exactly.

| Control | Preserved result |
|---|---|
| Clear box; first or second camera blocked | Qualified; one valid view suffices |
| Both cameras replaced by foreground | Unresolved |
| NaN, zero or negative depth | Unresolved; never FREE |
| Out-of-frame boundaries | Unresolved |
| Missing frames, sparse cloud, missing top or support samples | Unresolved |
| Small hidden extension | Qualified geometry contains the extension; unchanged conservative bound |
| Physically covered partial box | Unresolved despite a confident cube fit |
| Same partial box with 1.5 mm correlated top-height error | Unresolved |
| Committed 487-point native Square with original pre-grasp depth | Qualified; identical conservative geometry |

## Focused checks and timing

**24 tests passed in 9.563 s:** the requested observability, variable-camera and
provider modules, plus the pixel-order regression, frozen multiview observation
comparison and mapping's two numerical tests for the shared pixel helper.
No fitting interface changed, so fitting tests were not repeated. No full suite,
historical benchmark matrix, new camera capture or live Isaac run was needed.

```bash
python -m unittest tests.unit.test_multiview_observability \
  tests.unit.test_multiview_variable_cameras tests.unit.test_multiview_provider \
  tests.unit.test_multiview_final_refinements \
  tests.unit.test_semantic_processing.SemanticProcessingTests.test_multiview_split_preserves_frozen_observations_geometry_ids_and_clouds \
  tests.unit.test_rgbd_reconstruction.NumericalTests
```

Variable-camera tests now assert actual graspability with three/five views,
third-only sphere visibility, stable identities after reordering, exact fused
clouds and unchanged geometry/graspability when adding a valid empty camera.
The single-clear-view test additionally checks that the blocked camera alone
cannot qualify and reversing camera order preserves geometry. Invalid-depth
checks cover NaN, zero and negative depth instead of diagnostic count fields.

Pixel comparison: 32 real native ribbons, two warm-ups and five measured passes
over the same frozen arrays. Exact pixel coordinates were checked for all methods.

| Deduplication method | Median for all 32 ribbons |
|---|---:|
| Previous scalar encoding | 7.16 ms |
| `np.unique(raw, axis=0)` | 48.84 ms |
| Retained bounded scalar encoding | 4.47 ms |

The axis-based version would add about 21 ms per two-box scene relative to the
old method, a material cost for this small operation. Scalar encoding remains.
The pixel-order test now checks row-major coordinates needed by neighbor lookup;
the artificial overflow fallback test is retired.

One sequential processor before/after comparison used the existing harness,
two warm-ups and five samples per scene/version, CPython 3.11.18, NumPy 1.26.4
and the same 12-thread OpenBLAS environment:

| Metric | Before | After |
|---|---:|---:|
| Static p50 / p95 | 402.70 / 406.55 ms | 402.33 / 403.88 ms |
| Pre-grasp p50 / p95 | 426.87 / 430.22 ms | 427.38 / 428.70 ms |
| Static observability p50 | 16.91 ms | 15.21 ms |
| Pre-grasp observability p50 | 15.46 ms | 14.02 ms |
| Numerical fits, both scenes | 4 | 4 |

Overall differences are −.09% / +.12%, with no meaningful slowdown in this small
sample. These measurements do not establish sustained live performance.

## Evidence and remaining cleanup

`outputs/observability-audit-20261123/` preserves the detached baseline, local
probe, before/after controls and three/five-view outputs, frozen pixel arrays,
deduplication measurements, direct native bounds, four processor timing JSONs,
WorldModel replay and an evidence hash manifest. The tracked native performance
harness is unchanged; reproduce it with the original input manifest and distinct
output paths as described in [the fitting audit](multiview-fitting-audit.md).

For the final multiview audit, inspect whether `fitted_top_z` needs to travel
beyond qualification and whether the already-combined `parameter_observable`
metadata is the clearest remaining provider-to-observation representation. Its
decision is necessary; do not discard it as reporting. Shared pixel constants and
deduplication remain used by mapping; no generic geometry abstraction was added.
No further module refactor is started here. Stop for review.
