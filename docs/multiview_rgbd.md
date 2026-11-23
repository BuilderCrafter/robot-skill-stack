# Multi-view RGB-D perception

`[world].provider = "multi_view"` selects geometry-based multiview perception;
the ordinary playground defaults to single-camera [Perception V1](perception.md).
Use the [final demonstration setup](multiview_perception_final.md) for the two
existing 640×480 cameras. [Configuration](configuration.md) describes adding
camera prims and selecting a complete setup file.

## Data flow and ownership

```text
Distinct synchronized exposures → world XYZ → per-view voxel centroids
  → fused cloud → candidate components → competing primitive fits
  → measured parameter/boundary qualification → tracking → WorldModel
```

Camera axes are ROS (+X right, +Y down, +Z forward); depth is image-plane Z,
with half-pixel centers and calibrated camera-to-world transforms. Workspace
cropping removes the configured support plane. Default processing uses stride 2,
2 mm voxels, at most 12,000 fused points, 16 coarse candidates, eight detections
and 24 shared numerical fitting calls. Internal clouds are finite Nx3 arrays;
scene configuration supplies valid bounds and support height.

Each view is voxelized before fusion to balance camera density. The provider
merges those already balanced clouds directly. `fuse_clouds()` performs both
stages for mapping's dense box-reconstruction input, which is not pre-balanced.
Neither operation fills missing surfaces or creates FREE evidence.

`DepthFrame` retains camera identity, calibration, render token and original
monotonic acquisition time. Every configured camera must provide a complete,
distinct exposure within the synchronization skew; no partial-camera fallback
is selected. N-camera operation is supported, with bounded fused-cloud/fitting
budgets rather than a promise of equal throughput as views are added.

The [observation runtime](architecture.md#acquisition-and-workers) acquires on the
Isaac owner thread. `SharedFramePool` shares immutable original exposures with
mapping, admitting complete batches independently for each consumer. SDK-free
processors run in the existing bounded worker; association and WorldModel writes
remain on the owner. Original timestamps, context revisions and timeline epochs
prevent stale or invalidated computation from becoming a new observation.

## Segmentation

Coarse candidates use neighboring 8 mm cells. Refinement uses exact 4 mm radius
connectivity, with bounded 32-point blocks and packets of at most 128 block pairs.
It splits well-covered components only when supported primitive fits have
separate conservative XY bounds and no competing single-solid explanation.
Unresolved multi-component explanations remain UNKNOWN.

For 2–8 retained components covering less than 90% of a non-oversized candidate,
a single cached whole-surface sphere or supported box can survive if its 95th
percentile residual is at most 3 mm. A box additionally requires at least 95%
of **all** samples in one measured XY component at the same 4 mm radius.
This accepts vertical sampling gaps without merging separate footprints.
Subsequent parameter and camera-boundary qualification still decide graspability.

A qualified target can survive beside a separate oversized component when the
retained components cover at least 90% of the input. Truly connected oversized
geometry remains rejected. No surface reassignment, 4.5 mm retry or camera-gap
inference is used. Close rotated boxes/cylinders with overlapping world-axis
bounds may remain unresolved. The [fixed-input experiments](multiview-performance-ab.md)
record sparse-sphere, foreground and occluded-box controls, including failures.

## Primitive fitting

`SurfaceFitter` compares sphere, supported yaw-box, upright-cylinder and
horizontal-cylinder hypotheses. Confidence scores combine measured residuals
and coverage; they are not calibrated probabilities. Selection requires the
best class score ≥.65 and a runner-up margin ≥.08; otherwise it returns UNKNOWN.
Actual yaw, local dimensions, radius, length and axis serve manipulation and GUI
consumers. Classification never substitutes scene object IDs or reference sizes.

Circle/sphere estimation uses normalized four-pass robust least squares.
Boxes require top/side samples and convex-hull footprint coverage; yaw uses
5° coarse, .5° and .05° refinements. Horizontal cylinder axes use 10°, 2° and
.25° stages. These remain justified by partial-arc outlier, circular-cap and
off-grid orientation [experiments](multiview-performance-ab.md#fitting-and-pixel-processing-controls).
Arbitrarily tilted boxes/cylinders are outside the supported model.

Finalization copies the cached raw fit, then qualifies measured outer extents
without repeating classification. It preserves support contact, per-view radius
agreement and conservative spread. Upright cylinders need at least half-circle
curvature and measured top-cap interior; lying cylinders need quarter-circle
curvature and both measured endpoints. A short arc or missing cap is not
qualified merely because the numerical residual is small.

## Box dimensions and graspability

A convincing surface fit does not prove hidden physical bounds. `assess()`
returns qualified center/local dimensions or `None`. Its inputs include the
original measured top height before outward geometry expansion. For each
horizontal side it:

- Projects 48 edge samples into calibrated camera depth with a ±3-pixel ribbon.
- Requires adjacent occupied/FREE pixel pairs across three separated edge bins.
- Excludes invalid/nonpositive depth, foreground and out-of-frame pixels.
- Compensates lateral intersections for top-height spread at oblique angles.
- Combines independent camera bounds, then tightens them with supported side
  planes, including slope and residual spread.

Measured top extent and current low-side samples supporting the configured
support plane qualify the vertical bounds. One clear camera may establish an
edge hidden in another; every camera need not see every edge. These are empirical
bounds under the single supported yaw-box model, not universal hidden-volume
certificates. Mapping reconstruction has its own six-face/FREE checks and is
[separate from semantic grasp qualification](spatial_mapping.md).

The provider combines these bounds with the fitter's parameter qualification.
Known but unsupported boxes remain ungraspable. Curves use their existing
finalization path. Temporary measured top-height metadata ends at qualification;
published geometry retains the meaningful parameter decision and a short fitter
failure reason when applicable, without nested camera reports.

## Identity, availability and limits

Three consecutive distinct detections confirm an ID. Greedy one-to-one
association gives eligible placement hints priority; expected poses are priors,
not replacements for measured geometry. Held/uncertain IDs are retained. The
provider exposes current visible clouds; WorldModel stores geometry rather than
clouds. `association_hint_match` permits WorldModel to consume a matched hint;
`contributing_views` records actual camera support.

The updater retains the one-second freshness limit and configured expiry.
Detection, correct geometry, persistent identity, graspability and verified
physical manipulation are separate outcomes. Semantic perception does not mask
robot returns generally, reliably separate touching objects, or certify arbitrary
6D geometry. More cameras cannot establish every hidden bound.

Saved native captures retain four accurate, graspable objects, including the
occluded Square. The [native runtime study](multiview-native-validation.md)
measured about 2 Hz publication and stale visibility gaps on the Xeon/L4 host;
later offline geometry recovery does not establish live availability. GUI review
and a complete multiview Pick/Place demonstration remain unvalidated there.
Use [validation](validation.md) for current commands, not historical test totals.

Earlier detailed [observability](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_geometric_observability.md),
[robustness/performance](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_robustness_performance.md)
and [manipulation-readiness](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_manipulation_readiness.md)
studies remain at the preserved research checkpoint. Their retired drivers are
not part of the supported regression inventory.
