# RGB-D static obstacle reconstruction: design checkpoint

14 November 2026. Repository `BuilderCrafter/robot-skill-stack`, branch
`feature/global-rgbd-planning-v5`. Investigation baseline and independently
advertised remote tip: `f08a41204dd66cb39498fc8e68c557f7c15cf1d4`.
`git fetch origin` and `git ls-remote --heads origin
refs/heads/feature/global-rgbd-planning-v5` agreed with local HEAD. The starting
worktree was clean. Read [AGENTS.md](../../AGENTS.md) and all three existing
[finalization reports](v5-realignment.md), including the excluded base-clearance
candidate. This task changes only this report.

## Decision

**Select approach 3: a minimal measured-box model alongside unchanged occupancy.**
Reuse the existing multi-view worker's dense clouds, clustering, box-surface
search and image-boundary evidence. Do not feed tracked semantic objects directly
to Lula, complete occupancy cells, or build a second reconstruction pipeline.

The justified next implementation is **one bounded offline box-observability
prototype**, followed by a review gate. There is **NO-GO for runtime integration
or native execution on the present evidence**. Two archived sensor pairs contain
useful obstacle extent information, but this investigation has not qualified all
hidden bounds. Raw `large` and `partial` observations are unavailable locally.
The unchanged selected floor evidence and base guard also prevent this overlay
from resolving the recorded static rejection; source freshness is independently
unresolved. A requirement to improve all four native cases immediately under
unchanged checks therefore receives NO-GO.

This is a conditional implementation recommendation with explicit early exits,
not a claim that complete geometry or reliable RGB-D planning has been achieved.
If the bounded prototype cannot qualify geometry from current pixels within
existing budgets, stop this final RGB-D attempt and retain the validated GT
demonstration. Do not start implementation as part of this design task.

## 1. Current pipeline and architectural boundaries

Paths below are relative to `robot_skill_stack/` unless prefixed otherwise.

| Boundary | Actual implementation and relevant contract |
| --- | --- |
| Acquisition/calibration | `integrations/isaac/sensors/rgbd_camera.py` reads image-plane depth, actual intrinsics and the ROS optical camera transform. `integrations/isaac/world/multiview.py` supplies exposure tokens, including rational ReferenceTime. Two distinct 640×480 cameras remain the resource envelope. |
| Synchronization/shared exposure | `world/perception_multiview/acquisition.py:DepthFrame` retains depth, K, world-from-camera, token and original monotonic acquisition time. `FrameSynchronizer` requires complete, fresh, nonrepeated batches with at most 25 ms skew. `shared_frames.py:SharedFramePool` copies each exposure once and serves both consumers without refreshing timestamps. |
| Runtime ownership | `integrations/isaac/observation_runtime.py` owns SDK acquisition/publication. The existing semantic and mapping subprocesses do heavy work. Semantic acquisition is scheduled at 5 Hz, mapping at 2 Hz; shared-camera mapping acquisition follows semantic acquisition. Worker results are generation/context checked. Neither shared images nor similar completion times alone prove an exact exposure match. |
| Dense geometric perception | `world/perception_multiview/provider.py:MultiViewProcessor.compute` projects stride-2 depth, crops workspace/support, voxelizes each view and their union at 2 mm, capped at 12,000 points. `segmentation.py` clusters in 8 mm cells, minimum 25 points. Processing considers at most 16 components, eight detections and 24 numerical fits; each fitter uses at most 1,200 points. Identity/tracking is applied later by the provider. |
| Geometric inference | `fitting.py:SurfaceFitter`, `segmentation.py:refine_cluster`, and `observability.py:assess` already perform primitive fitting, surface refinement and current calibrated-ray boundary checks. Their box model assumes a single table-supported yaw box. They are not arbitrary-obstacle solid reconstruction. |
| Temporal occupancy | `world/spatial_mapping/runtime.py:MappingProcessor` uses frozen `MappingInput` frames and exposure-matched robot snapshots. `rgbd.py:RgbdMapper` allocates a total 1,024 rays across cameras, stride 8, without backfilling masked rays. Endpoints provide occupied evidence, preceding ray segments free evidence, and space behind returns remains unknown. Evidence TTL is 5 s; source freshness is separately 1 s. |
| Robot filtering | `world/spatial_mapping/robot_filter.py:RobotSelfFilter` uses permitted known robot geometry and exposure-matched robot snapshots. It filters the mapper-selected rays. The dense semantic cloud is not equivalently filtered. Robot returns must not become inferred external solids. |
| Extraction | `world/spatial_mapping/obstacles.py:ObstacleExtractor` selects supported occupied evidence, including strong sparse RGB-D cells, and merges exact Z/Y/X grid cuboids. It does not bridge unknown cells. Hazard-first selection is deterministic; normal budget is 64, with 128 diagnostic only. |
| Frozen source | `world/spatial_mapping/frozen.py:capture_frozen_rgbd` validates health, freshness, source/filter provenance, generation/revision, calibration, tokens, expiry and exact grid geometry. IDs, dimensions and voxel counts must agree. The immutable read view fixes query time; no dense reconstruction occurs during capture. |
| Planner | `integrations/isaac/manipulation/motion/frozen_rgbd.py` converts exact numeric cuboids to `StaticWorldSnapshot`, tagged `rgbd_occupied_cells` / `exact_selected_occupied_cells`. Its only external USD geometry read is the allowlisted known support Plane. `global_rrt.py` captures once; `lula_global.py` already registers oriented cuboids, spheres and capsules. Lula, route/spline validation and the timed executor share that frozen world. |
| Independent evaluation | `scripts/global_rgbd_v5_pair.py:MatchedSourcePlanner` controls the common start and validated joint goal. `scripts/evaluate_global_rgbd_v5.py:PhysicalMotionEvaluator` uses independently copied cooked solids and URDF FK at declared route/spline samples, vetoing unsafe or unresolved results before actions. This research evaluator is neither a reconstruction input nor a general production safety certificate. |

There is no missing planner interface for a numeric box. The difficult work is
qualifying the box and transporting its provenance without changing occupancy,
runtime lifecycle, or the frozen-source contract.

## 2. Root causes established by V5 evidence

The [V5 study](../../docs/motion_planning_v5_rgbd_global.md) and tracked
[`primary64/summary.json`](../../docs/assets/motion_planning_v5/primary64/summary.json)
report GT 7/7 completed motions. RGB-D returned three plans: `partial` and
`large` were physically unsafe and vetoed; `narrow` executed the exact empty-GT
route. Four cases were rejected at the base guard. No obstacle-induced RGB-D
detour was demonstrated.

| Primary V5 case | Observed eligible target cells retained | Extraction candidates | Selected floor boxes | Interpretation |
| --- | --- | ---: | ---: | --- |
| `static` Post | 36/36 | 526 | 28 | Complete retention of a sparse visible surface still does not represent the solid. Base rejection also applies. |
| `v43` box | 7/7 | 505 | 57 | Seven retained surface cells cannot substitute for a complete obstacle. Base rejection also applies. |
| `partial` | Post 36/36; Occluder 28/77 | 578 | 0 | Both missing solid volume and selection loss matter. Nominal partial placement did not actually occlude the Post; the separately verified occlusion condition was unsafe too. |
| `large` Wall | 67/218 | 608 | 0 | Only 30.7% of eligible wall evidence was selected, in addition to unseen-volume loss. |

These are observed-cell counts, not solid recall. A 100% retention percentage
does not establish physical obstacle completeness. Increasing the proxy budget
does not repair unseen volume. The [V3 representation study](../../docs/obstacle_representation_v3.md)
also found the existing exact packer attained its component-based lower bound
on 66/66 archived maps. Further exact packing searches are a poor final attempt.

The [excluded base-clearance experiment](rgbd-bounded-improvement.md) provides
two separate stop signals. Its fresh static RGB-D request was rejected at
1.060005840 s source age against 1.000 s, before planning. A geometry-only audit
of that packet retained 64 boxes and resolved **zero** of its one original base
rejection. The selected floor cell centered at approximately
(0.025, 0.095, -0.005) m remained problematic. Retaining that selected cell and
the unchanged guard means adding inferred solids cannot unblock the same world.
Do not rerun that experiment, remove floor evidence, or modify collision
resources to give reconstruction a favorable outcome.

Freshness, floor/base rejection, missing volume and omitted observed evidence
are distinct causes. Reconstruction addresses only the latter two, and only
if its additional work preserves original-source freshness.

## 3. Reusable measured information and algorithms

The dense multi-view branch contains far more surface detail than the mapper's
1,024 selected rays. Reuse its already projected clouds before manipulation
qualification and tracking. Do not consume `WorldModel` estimates: persistent
identity, stale estimates, attachment hints and grasp observability are not an
immutable collision-source contract.

Useful existing pieces are calibrated projection with pixel-center convention,
bounded voxel fusion, deterministic clustering, box yaw/residual search,
side-plane fitting, and bounded occupied/free image ribbons. The existing box
fitter uses a 1.5 mm residual tolerance and 0.65 score threshold; current boundary
assessment uses 2 mm depth tolerance, radius-three pixel ribbons and 48 edge
samples. These are empirical measurement rules, not calibrated probabilities
or universal physical containment certificates.

Direct reuse as complete obstacles is invalid:

- `supported_object_cloud` rejects components over 0.18 m and requires observed
  support proximity. The existing box fitter limits dimensions to 0.20 m and
  sets the lower face to configured support height.
- `assess` qualifies four horizontal bounds through top-plane transitions,
  plus top/support evidence. It does not establish a suspended box's lower
  face or arbitrary hidden back face.
- In the evaluation fixtures, the Post bottom is 0.22 m, v43 bottom 0.24 m,
  partial Occluder bottom 0.33 m, and large Wall bottom 0.05 m. Those numbers
  come from independent fixture scoring only. Substituting them as support
  heights, or setting support to minimum observed Z, would fabricate evidence.
- The older `world/perception/primitives.py:PrimitiveEstimator` is also a
  supported-small-object estimator. It is not a more general alternative.

A separate obstacle-box qualification helper is justified because its support
assumptions differ from manipulation. Reuse numerical helpers while preserving
the current fitter defaults and semantic outputs.

## 4. Comparison of the three directions

| Direction | Inputs → outputs; location | Advantages | Occlusion, inference and collision risks | Cost and required changes |
| --- | --- | --- | --- | --- |
| **1. Existing geometric perception → complete primitive estimates** | Current synchronized dense clouds and fits → numeric primitive hypotheses in the semantic worker, before tracking. | Most reusable geometry; far fewer proxies for supported, fully qualified small objects. | Existing supported-box assumptions reject or misdescribe these suspended/large fixtures. Residual/confidence does not bound hidden dimensions. Filling from an incomplete fit risks false negatives; guessed support or conservative whole-cloud boxes risk false positives, including robot geometry. Every filled interior must be labeled inferred. | Reusing emitted detections is cheap but ineffective here. Generalizing box, sphere and cylinder observability plus immutable source transport is a substantial change. Reject direct semantic-object-to-Lula wiring. |
| **2. Occupancy/surface reconstruction** | Existing map, selected endpoints, ray-free evidence and supported components → fitted boxes/hulls in the mapping worker. | Natural robot filtering and map provenance; no cross-worker join. | Sparse disconnected returns do not identify hidden thickness. Hulls, closing, dilation and extrusion can bridge unrelated objects or free space, while missing unseen extensions. Writing their interiors as occupied would violate unknown semantics; a separate inferred product would still be necessary. Temporal accumulation can mix old evidence. | Clustering existing endpoints is cheap but produced no minimum-size components in the two replays. Denser ray integration is forbidden. A new TSDF/mesh/learned pipeline, temporal shape tracker or packing campaign is disproportionate. |
| **3. Dense measured box + unchanged occupancy** | Existing dense worker clouds and current pixels → qualified box models; exact mapping publication supplies retained occupied evidence and source checks. Join only matching exposures. | Uses available detail without more cameras/rays; preserves observed evidence; separates modeling assumptions from map truth; existing Lula accepts the result. | Bounds can remain unobservable. Dense robot returns, cross-worker skew, false merges and proxy-budget pressure require explicit rejection. The inferred interior stays outside occupancy. It cannot fix the existing floor guard. | One box-only helper plus bounded provenance/join/adapter work. No new worker, acquisition, dependency or planner. Greatest implementation risk is trustworthy qualification and publication, so test those offline before runtime wiring. **Selected, conditionally.** |

No measured workstation cost exists for a new reconstruction algorithm. The
single-call host replay times below are a warning against assuming spare CPU,
not a throughput prediction. Duplicating the dense fitter in the mapping worker
would waste the most valuable reuse opportunity.

On the university workstation, approach 3 adds CPU-only bounded searches on
already allocated clouds: at most four hypotheses, each using the existing
1,200-point numerical cap and fixed image ribbons. It adds small numeric result
records, with no new dense map, image acquisition or GPU allocation. Work scales
with candidate points and fixed search/ribbon counts, rather than map volume.
The exact-exposure join copies only small metadata. This bounds resource growth
but does not establish acceptable milliseconds; incremental timing and original
publication-age/rate gates must decide feasibility on that workstation.

## 5. Scratch offline feasibility analysis

Only two recent raw sensor archives were found:

```text
outputs/motion-planning-v5/rgbd-base-clearance-20261113/static/frozen_rgbd/
outputs/motion-planning-v5/rgbd-base-clearance-20261113/v43/static/
```

Each has `accepted_mapping_input.pickle`, camera arrays/calibration, runtime
profile, occupancy and independent geometry reference. These are the recent
captures described in the excluded-candidate report, **not** the original
seven-case primary cohort. The v43 archive was recorded during a GT-selected
request; its independently acquired camera pair still supports sensor-only
offline analysis, not a successful RGB-D request claim. No raw `large` or
`partial` archive was found; tracked summaries/figures cannot establish new fits.

Scratch work is outside Git at `/tmp/rgbd-reconstruction-design-20261114/`.
`replay_inputs.py` loaded the trusted local `MappingInput`, asserted two 480×640
frames and `ground_truth is None`, and ran existing `MultiViewProcessor` and
projection/fusion/clustering with their current parameters. It also replayed
the existing mapper allocation and robot masks without additional rays. Inputs
were depth, calibration, workspace/support configuration and captured robot
snapshots; fixture names, dimensions and reference solids were not passed into
the geometric algorithms. Raw cluster minima/maxima were recorded before
manipulation qualification; **no new closed-solid fitting algorithm was run**.

For archived-ray replay only, the mapper clock was set to the maximum original
acquisition time plus 1 ms. Original frame times were unchanged. This makes the
saved pixels replayable; it does **not** validate live freshness, authorize the
previously rejected source, or produce a planning/execution measurement.

Only after blind outputs were saved and hashed did `score_outputs.py` read
`geometry_reference.json`. It associated the nearest already-produced component
with each fixture for scoring, then verified that prediction hashes remained
unchanged. There was no GT cropping, fixture-specific reconstruction threshold
or fitted dimension selection.

| Offline measurement | `static` | `v43` |
| --- | ---: | ---: |
| Fused dense workspace points / blind components | 9,977 / 6 | 5,363 / 8 |
| Fixture-associated dense points, chosen during scoring only | 4,929 | 941 |
| Measured XYZ span, m | (0.0900011, 0.1199706, 0.2599695) | (0.0600006, 0.0599452, 0.1199339) |
| Maximum absolute AABB boundary error against GT | 0.030942 mm | 0.067389 mm |
| Fixture-associated points within GT box ±2 mm | 100% | 100% |
| Existing semantic output includes the fixture | No: oversized component | No: no observed support contact |
| Mapper rays allocated / retained after robot mask | 1,024 / 989 | 1,024 / 990 |
| Mapper nonfloor workspace endpoints | 36 | 7 |
| Existing 25-point clustering on those endpoints | 0 components | 0 components |
| Existing complete semantic call, host wall time | 376.352 ms | 475.756 ms |
| Existing mapper replay, host wall time | 154.480 ms | 156.645 ms |
| Existing numerical fit calls / cap | 2 / 24 | 5 / 24 |

Measured centers were (0.4500000, 0.0000151, 0.3500157) m and
(0.5500004, -0.2249714, 0.3000343) m. This is strong evidence that current
dense pixels contain useful extents for these two ideal simulated captures.
It is **not** evidence that a raw cluster envelope conservatively encloses a
physical object: unseen extensions remain possible and observed extrema can
undershoot. Independent truth-plane diagnostics found no face-interior samples
on the -Y rear or -Z bottom face in either case. Edge visibility must be
qualified from pixels before accepting either bound. Raw dense clouds also
include robot returns; successful fixture association during scoring does not
solve runtime robot rejection.

The calls were single host Python 3.12.16 runs, NumPy 1.26.4, py_trees 2.6.0,
SciPy 1.15.3 and PyYAML 6.0.2. They are neither native Python 3.11/workstation
timings, incremental reconstruction cost, nor a frequency/latency distribution.
Do not infer 5 Hz headroom from two calls. No Isaac/container/stream operation
was performed.

For provenance, SHA256 bindings are:

| Artifact | `static` | `v43` |
| --- | --- | --- |
| `accepted_mapping_input.pickle` | `36b0d55abec6b335573f76d39f35a30cd2cd5752cd3b29f1a601f06af57ce555` | `00dc6e5c38a1976aaad9ddf7db1c77efde4a52acdfe46d1fa068c72a3a5dfff5` |
| `camera_frames.npz` | `957e2ef505fb8445ef780057ed7c4faa571c20ba560d842ea5c517c5fb6fc73f` | `95c9a4d386ec9e8abe7d7c30e9d93d4bf788e25eb4934da1ea747616356d0da1` |
| `runtime_profile.toml` | `48a7610283a2746b83b1981c50bdf5877c516eaed258087566fbffebdd7f9283` | `0e3f0121d45c1988714e63e98f8049410d4a2d9a7ef0e2fdddc1decdc23a7e5c` |
| Blind `*-replay.json` | `55e377777d5e3f285909b138bc8dfe1722fd9e84c6492bbb7768f474533d973b` | `e750afcf6b91cb8d899305eb4296a048b20cc38f67d7aaf2bc8344de7db2d6d3` |

Scratch script hashes: `replay_inputs.py`
`1fb70fd38df0961bc424ce88dec7de4a1da3cbc76d901d4ed9ab418cbf0ba5d3`;
`score_outputs.py`
`7eeb9af4b302a25a979e969b1a565cb8c4c0d827c79fd736077725b2903a8240`;
`scoring.json`
`d57a295c83c5536a4ba24fd5c380f490d31a11932cb87ec1bcac07f3a7966391`.
`input-manifest.json` additionally binds scoring references and original result
files. Scratch scripts/results are local supporting material, not committed
fixtures or portable historical benchmarks. The table retains the substantive
results for remote review.

## 6. Recommended design and precise data flow

Scope is static, single yaw-box components only. Exclude arbitrary six-degree
boxes, meshes, spheres/cylinders, learned completion and temporal solid growth.
No object dimensions, identity or fixture thresholds may enter the algorithm.

```text
Existing shared exposure pair + original K/T/token/time
  -> existing semantic worker projection/fusion/clustering
  -> optional bounded obstacle-box qualification on existing clouds
  -> small immutable geometry/provenance result
                         +
Same exposure's existing mapping-worker exact occupied-cell packet
  -> exact-token/calibration/generation join of bounded publications
  -> frozen exact occupancy + separate qualified inferred-box records
  -> <=64 numeric cuboids preserving the original selected occupied union
  -> existing StaticWorldSnapshot -> Lula RRT -> route/spline checks -> executor
```

1. Keep manipulation processing and its candidate order unchanged. Reuse its
   cached geometry where valid. Any extra box hypothesis must consume only the
   **remaining** existing 24-fit allowance, retain the 1,200-point fit cap and
   use at most four hypotheses from the existing capped component list. Do not
   repeat projection, clustering or a second full fitter. Budget exhaustion is
   a rejected reconstruction, not an increase or manipulation regression.
2. Fit a yaw/plane hypothesis using reusable box-surface helpers, without the
   tabletop lower-face assumption. Qualify all six extents with current direct
   face evidence or calibrated occupied/free boundary transitions. Extend the
   existing bounded ribbon approach for lower-face evidence only if it can be
   justified compactly. Missing, occluded, out-of-frame or ambiguous evidence
   leaves a bound unqualified. Minimum observed Z is never a support plane.
3. Reject components that could contain the robot using conservative envelopes
   from already captured, exposure-matched robot snapshots. The offline gate
   must demonstrate that this rejects robot/connected returns without losing
   the target. Do not expand mapper masks into full-resolution mesh ray tests,
   reacquire robot poses, or claim that clustering alone identifies the robot.
   If available snapshots cannot support bounded conservative rejection, stop.
4. Produce a separate immutable `ReconstructedBox`-style numeric record:
   center, yaw/quaternion, dimensions, per-face interval/evidence status,
   residual/measurement bands, explicit single-box assumptions, contributing
   cluster/view IDs and bounded boundary-pixel witnesses (not full clouds),
   original exposure/calibration identity,
   oldest acquisition time, expiry and lifecycle identity. Quality is not a
   probability of safety. Inferred interior is labeled `rgbd_inferred_box`.
   Raw mutable `PrimitiveFit`/tracked objects must not cross this boundary.
5. Join only identical camera IDs/tokens, acquisition times and K/T, with valid
   runtime generation/context revision and mapping generation/revision. At most
   one unmatched result from each worker is retained. No nearest-time join,
   use of an arbitrary latest semantic result, blocking wait on the SDK owner,
   timestamp refresh or cross-reset reuse. Failed/late/mismatched results make
   the requested inferred source unavailable. Associate the accepted small
   geometry product with one mapping publication before frozen capture.
6. Preserve `ObstaclePacket.boxes` and the current exact-cell validators as-is.
   Store inferred records separately; never give a model a grid-cell ID or
   invented voxel count. Capture validates/copies at most four model records
   plus the existing bounded packet. Map queries continue to return their
   original occupied/free/unknown states at the original fixed time.
7. Respect **64 total camera-derived collision proxies**. A model may replace
   a subset of original proxies only if it geometrically contains each whole
   replaced cuboid, including all corners with outward numerical allowance.
   Keep original IDs and coverage associations in provenance. Retain every
   uncovered proxy and every selected floor box. Verify
   `union(original selected cuboids) ⊆ union(planning cuboids)`; testing centers
   is insufficient. If this cannot fit within 64, reject the inferred source.
   Do not reserve slots by dropping observed boxes or use 128 as the solution.
   This preserves selected evidence; it does not claim to recover previously
   omitted cells. Report that omission independently.
8. Translate the combined numeric geometry through the existing static-world
   adapter with distinct approximation tags. Capture once and retain the same
   RRT, validation, executor, known support, collision resources and thresholds.
   In inferred mode, failed qualification/coverage/source checks return an
   explicit planning-source failure and zero actions. The existing explicitly
   selected occupied-cell mode remains the default; there is no silent mode,
   GT, direct or empty-world substitution.

The single-box assumption can infer occupied physical volume in areas whose
**occupancy state remains UNKNOWN**. That distinction must be visible in source
metadata, diagnostics and research figures. Also reject models contradicted by
current calibrated free-ray evidence beyond existing measurement/quantization
bands. Finite consistency checks do not prove all unseen space is correct.

## 7. Exact proposed file changes, for a later authorized task

These are proposals only; none is implemented by this report.

| File under `robot_skill_stack/` | Proposed responsibility |
| --- | --- |
| **New** `world/perception_multiview/obstacle_geometry.py` | SDK-free immutable box/evidence result and bounded free-box qualification. This small separation is necessary to avoid changing table-supported manipulation assumptions. Reuse helpers; no new architecture. |
| `world/perception_multiview/fitting.py` | Expose/reuse the existing yaw/box residual helper where necessary. Keep `SurfaceFitter` defaults, primitive policies and manipulation finalization unchanged. |
| `world/perception_multiview/observability.py` | Reuse calibrated ray/face helpers and add a separate free-box bound assessment if needed. Preserve `assess` semantics and established numerical thresholds. |
| `world/perception_multiview/provider.py` | Optional small result field built from existing raw components/cached fits and remaining fit budget. Preserve detections, tracking, public observation batches and default behavior. |
| `world/spatial_mapping/runtime.py` | Optional separate immutable geometry attachment to an accepted publication/read view, with generation/revision binding. Do not change `RgbdMapper`, occupancy updates or exact extraction. |
| `integrations/isaac/observation_runtime.py` | Opt-in bounded exact-exposure join, generation/reset rejection, and access to already captured robot snapshots. Heavy inference stays in the existing worker; SDK owner handles only small metadata. |
| `world/spatial_mapping/frozen.py` | Validate/copy the separate model attachment while preserving existing exact packet checks and occupancy queries. Original freshness, TTL and lifecycle rules still apply. |
| `integrations/isaac/manipulation/motion/frozen_rgbd.py` | Construct the conservative <=64 combined numeric geometry, verify coverage and add separate source tags. Preserve known support and explicit source failures. |
| `integrations/isaac/config.py`, `integrations/isaac/manipulation/factory.py`, `integrations/isaac/manipulation/motion/global_rrt.py` | Only if the runtime gate is later approved: validate/wire an explicit optional geometry policy, e.g. `occupied_cells` / `measured_boxes`, defaulting to existing behavior. Require the existing compatible multi-view setup; no automatic cameras or extra acquisition. No planner/executor algorithm edits. |

Proposed focused tests belong beside existing multiview, shared-frame,
observation-runtime and frozen-source tests. No production change is needed in
`lula_global.py`, `collision_model.py`, motion resources, launchers, skills,
Behavior Trees or existing public manipulation/runtime APIs.

## 8. Safety, uncertainty and failure behavior

An accurate visible envelope is insufficient: hidden extensions can cause false
negative collisions; merging nearby objects or robot returns can block valid
routes. Require explicit per-face qualification and conservative numerical
coverage, rather than a residual-only acceptance rule or a guessed confidence
cutoff. Reject ambiguous yaw, merges, missing lower bounds, contradictory free
evidence, invalid calibration, failed robot rejection and budget overflow.

Observed evidence never depends on model acceptance. The exact packet and read
view remain available for audit. No rejected hypothesis erases a selected cell;
no successful hypothesis relabels unknown cells. In inferred-source mode, a
required model that is unqualified or unpublishable fails the request. Do not
execute a partly reconstructed scene while presenting it as complete. Requiring
qualification of all eligible nonfloor, nonrobot obstacle components may itself
produce NO-GO; more than four such hypotheses exceeds this bounded scope.

Use the oldest contributing original acquisition time as the model timestamp.
Its expiry cannot exceed the earliest contributor/evidence expiry, and capture
must enforce the original one-second freshness limit. Freeze all planning
geometry together once.
Static-scene assumptions, sampled collision checks, payload limitations and
the existing finger contract remain unchanged. The independent physical veto
must still reject unsafe/unresolved plans; a veto does not count as avoidance.
No result here establishes continuous or real-robot collision safety.

## 9. Estimated complexity and bounded implementation plan

This is not a three-line adapter change. An engineering estimate is roughly
300–500 lines of geometry/evidence logic, 150–250 lines of optional publication
and adapter changes, plus focused tests: approximately 2–4 engineering days if
the observability gate succeeds. These are planning estimates, not a budget
authorization or measured performance. No new dependency is justified.

The next focused task should contain **only the first stage**:

1. Implement a scratch/offline-callable free-box qualifier in the proposed
   module, extracting minimal helpers as needed. Replay the same two blind
   sensor pairs under existing cloud/fit limits. Include robot rejection and
   per-face evidence; do not wire runtime/config/planner yet.
2. Produce frozen predictions before independent scoring. Run the controls in
   section 10. Report incremental cost, fit counts, bound failures and whether
   exact selected-cell coverage can fit within 64. Stop after one fixed design;
   no fixture-specific threshold sweep or new reconstruction family.
3. Review GO/NO-GO. Failure to qualify a hidden bound, reject robot returns or
   retain evidence within the budget ends the attempt. Successful offline
   geometry alone does not authorize native runs. Only a separately approved
   follow-up may implement the bounded join/frozen adapter and evaluate runtime
   freshness. The known floor rejection remains a blocking independent gate.

## 10. Specific acceptance criteria

**Offline geometry gate:** preserve the input hashes and publish blind outputs
before GT association. Both static and v43 must produce finite, nondegenerate
qualified bounds from current pixels, with all six face statuses and their
witnesses. Scoring must demonstrate conservative physical-box containment
within the evaluator's unchanged 10 µm numeric-frame allowance; the existing
2 mm depth band is a diagnostic measurement tolerance, not permission for
2 mm of collision undershoot. Report excess volume/extent and any disagreement
with free rays. The raw AABB errors measured here are not this acceptance test.

Use a fixed small set of sensor-only negative controls: crop a required edge,
remove lower-bound depth, insert nearer foreground/invalid pixels in boundary
ribbons, present robot-only/robot-connected returns, and present two nearby
objects or contradictory free-ray evidence. Missing evidence must cause explicit
unqualified/rejected results; an oversized merged box is not an accepted fix.
No GT dimensions may tune the qualifier. Directly preserve and compare existing
semantic outputs. Additional computation must remain inside the existing point,
fit, worker and queue limits.

**Representation/source gate:** whole-cuboid coverage of all original selected
evidence, <=64 total proxies, selected floor unchanged, and separate observed
and inferred tags. Fixed-time occupancy queries must be identical before/after
adding the model product. Exercise stale/missing batches, mismatched tokens or
calibration, late worker completion, Stop/reset/close and generation changes.
Each invalid inferred request fails with zero actions, no fallback and no
mutation of prior frozen worlds. Confirm default occupied-cell, GT and direct
paths remain unchanged with existing focused regressions.

**Unmeasured-case gate:** `large` and `partial` need actual raw synchronized
observations before either receives a reconstruction claim. First recover an
existing archive if available elsewhere. If it is unavailable, record those
cases as unverified; a later authorized acquisition must preserve the original
camera/ray/scene settings. No fit can be inferred from tracked figures or GT
fixture dimensions. Failure on either stops any four-case improvement claim.

**Runtime gate, only after separate authorization:** retain the original
[workstation/workload protocol](../../docs/assets/spatial_mapping_v5/acceptance_protocol.json):
semantic publication >=4.5 Hz, mapping >=1.8 Hz, original-source age <1 s,
one running/one pending job per processor, movement apply p99 <=43.7803 ms,
worst apply/controller wait <100 ms and zero waits over 100 ms, alongside its
unchanged skill/tracking requirements. Compare matching workloads and record
raw distributions. Host single-call timings cannot satisfy these gates. If
sharing geometry cannot meet them, return NO-GO rather than add resources or
relax freshness. The previous 1.060005840 s capture remains rejected.

**Native gate, not authorized by this task:** predeclare at most four serial
native jobs after reviewing offline results and fresh-source/base eligibility:
one GT v43 preservation/control job; one primary RGB-D target (verified-occlusion
`partial` if qualified, otherwise `large`); its empty reference at the exact same
start and validated joint goal; and the other qualified case as a safety
regression. Use the existing paired planner, seed, enhanced model, physical veto
and execution checks. A detour claim applies only to a case with its own matched
empty reference; do not reuse an empty route with a different IK branch. Static
and v43 RGB-D remain NO-GO while their unchanged selected floor/base check
rejects; do not spend native jobs repeating that failure. Any change to this
small declaration requires review before execution, not an automatic wider
matrix or seed search.

A useful success requires at least one independently safe, actually executed
obstacle-induced RGB-D detour in a qualified case, plus preservation controls.
Establish empty-route obstruction by the physical obstacle, changed RRT
geometry at the same joint goal, and the existing V4.3 criterion: at least three
native waypoints and maximum seven-joint Euclidean distance from the original
direct joint segment >0.05 rad. Path length alone is
insufficient. Require physical evaluator acceptance, actual target reached,
zero contact reports/penetration/impulse and positive reported sampled cooked
clearance. State sampling limits. A predicted-collision or unresolved veto,
source rejection, unchanged empty route or incomplete actual motion fails the
native improvement claim. Stop on unsafe output; never execute it or compensate
with 128 proxies, altered safety thresholds or GT obstacle substitution.

## 11. Unresolved risks and checkpoint verification

The leading unresolved risks are lower/back bound observability, uncalibrated
noise/model assumptions, robot contamination, large-object segmentation,
strict cross-worker exposure matching and insufficient runtime headroom.
Current crop/point caps may also truncate a large wall. The missing large and
partial raw archives prevent a four-case feasibility conclusion. More selected
surface retention cannot establish a complete unseen object. Retaining the
floor/base guard means this recommendation is not a solution to all historical
V5 failures.

During this investigation, the existing focused fitting, observability,
frozen-source, shared-frame and paired-planner tests passed: **49 tests, 3.142 s,
exit 0**. Command from the repository root:

```bash
/tmp/robot-v5-realignment-py312/bin/python -m unittest \
  tests.unit.test_multiview_fitting tests.unit.test_multiview_observability \
  tests.unit.test_frozen_rgbd_world tests.unit.test_shared_frames \
  tests.unit.test_global_rgbd_v5_pair
```

The log is `/tmp/rgbd-reconstruction-design-20261114/focused-tests.log`.
These unchanged tests substantiate the inspected baseline contracts; they do
not test the unimplemented design. No full-suite/native rerun was necessary
for a report-only change.
Historical evidence and baseline hashes were not changed. No production code,
tracked test, fixture, existing documentation, SDK installation, container or
stream was changed. This report is the only publication artifact. After its
commit and normal V5 push, stop and await the separate implementation task.
