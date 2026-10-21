# Spatial mapping

Spatial mapping stores volumetric environment evidence independently of semantic
objects. `world/spatial_mapping` contains the NumPy grid, occupancy evidence and
calibrated depth-ray integration; it imports no Isaac/Omniverse SDK. Isaac stage
geometry, existing camera access and explicit construction belong under
`integrations/isaac/spatial_mapping`. Neither source uses object labels, primitive
fitting, tracked identities or WorldModel to decide occupancy.

Mapping is opt-in and explicitly updated by its caller. Parsing its TOML section
does not add a runtime callback, initialize cameras or change application
behavior. No planner, skill, backend or GUI consumes the map in this version.

## Common map contract

`GridSpec` defines world-coordinate bounds in meters, voxel size and maximum
voxel allocation. Bounds are lower-inclusive/upper-exclusive, with XYZ array
indices. A final boundary cell can be shorter than the voxel size. Out-of-bounds
queries return `OUT_OF_BOUNDS` with unknown probability, never `FREE`.

```python
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap

occupancy = OccupancyMap(GridSpec((.10, -.45, -.05), (.85, .45, .65), .01))
query = occupancy.query((.45, 0., .10))
batch = occupancy.query_many([[.45, 0., .10], [.55, 0., .10]])
snapshot = occupancy.snapshot()
print(query.state.name, query.probability, snapshot.counts(), dict(snapshot.health))
centers = snapshot.voxel_centers()  # Optional export; full-grid allocation.
occupancy.clear()                  # UNKNOWN again; generation/revision advance.
```

The states are `UNKNOWN`, `FREE` and `OCCUPIED`. `observed=False` distinguishes
never-observed space from uncertain evidence near probability 0.5. Expired
evidence becomes UNKNOWN/0.5 with `observed=True`, `expired=True` and its historical
last-observed timestamp retained. Position/index conversions are exposed by
`world_to_indices` and `indices_to_centers`.

Snapshots own immutable arrays and retain one consistent revision, timestamp and
health record. An existing snapshot does not age in place; request another to
see current expiry/freshness. Map queries, snapshots and updates apply expiry;
there is no background aging worker. `revision` counts evidence/expiry/reset
changes; `generation` counts resets. Source health distinguishes waiting, fresh,
no new frame, error and stale. Failures do not refresh the acquisition timestamp
or add evidence; expiry can still change the revision while a source fails.

## Input sources and uncertainty

Ground truth traverses active scene Gprims, including instance proxies and
invisible geometry. Cubes, spheres and cylinders use actual schema dimensions,
axis and complete affine parent/world transforms, including nonuniform scale.
USD `metersPerUnit` converts all geometry to map meters. Conservative cell-volume
overlap preserves curved/rotated volumes instead of filling each primitive's
world AABB. Inverse local interval bounds and boundary touching can mark extra
cells occupied; this is not exact collision geometry.

Unsupported finite Gprims, including meshes/points/curves, use conservative world
AABBs and are counted in source diagnostics. Invalid/unbounded geometry, invalid
units, unloaded nonexcluded payloads, reported scene composition errors or work
budget failures reject the complete update. Fully resolved scene assets are
required. Successful snapshots rebuild occupancy, so moved/removed objects leave
FREE cells. Other map cells are authoritative FREE for this environment layer.
Cameras/lights are not geometry; inactive/undefined/abstract prims are skipped.
Guide geometry is skipped unless an enabled CollisionAPI makes it physical.
`/Franka` is excluded by default and can be changed explicitly; its omission
does not make the robot's swept volume safe. Composition errors are checked
globally when exposed by USD, including errors outside the mapping bounds.

RGB-D consumes existing calibrated `DepthFrame` snapshots from one or more
cameras, using their original IDs, tokens, acquisition times and transforms.
Pinhole rays use optical +Z forward and meter-valued depth. Valid rays add FREE
evidence before measured surfaces and OCCUPIED evidence at their endpoints;
unobserved space beyond a hit receives no evidence. Another view may observe it.
Invalid pixels and measurements beyond `max_depth` are ignored, without free
rays. An all-invalid depth image is a source failure.

Per batch, each voxel receives at most one hit/miss update, with occupied evidence
winning conflicts. Default log-odds increments are +0.85/-0.4, bounded to ±4;
FREE requires probability ≤0.35 and OCCUPIED ≥0.65. Consequently one miss can
remain uncertain. Repeated valid free observations can clear previous occupied
evidence; unrefreshed cells expire toward UNKNOWN after five seconds by default.
Source freshness defaults to one second and is independent of cell expiry.

Every configured camera must be present and valid. Missing/frozen/regressed,
skewed or stale frames do not publish fresh evidence; there is no partial-camera
fallback. Repeated buffers produce no new map timestamp. If camera IDs are not
configured, the first complete batch establishes the required set.
The default clock checks frame age again after ray processing; over-budget or
aged-out work is discarded atomically. Explicit `now=` supports deterministic
offline callers. Clearing a map preserves the mapper's consumed-token history;
reconstruct the source after a renderer clock reset.

`exclusion_masks={camera_id: boolean_depth_image}` skips each true pixel's entire
ray. A masked robot hit therefore clears neither the robot nor space behind it.
V1 accepts caller-supplied masks; V2 adds the optional automatic robot filter
described below. With automatic filtering disabled, callers must provide reliable
robot masks or robot returns can enter the environmental map. Ground-truth robot
exclusion does not automatically mask depth images.

## Configuration and explicit use

Add this section to a local complete setup copy; semantic `[world].provider` and
its workspace remain independent:

```toml
[spatial_mapping]
enabled = true
source = "rgbd"                    # Or "ground_truth".
bounds_min = [0.10, -0.45, -0.05]
bounds_max = [0.85, 0.45, 0.65]
voxel_size = 0.01
max_voxels = 2000000
camera_prim_paths = ["/World/PerceptionCamera", "/World/PerceptionCameraSide"]
stride = 8
max_rays = 1024
max_voxel_visits = 500000
max_depth = 5.0
max_skew = 0.025
max_frame_age = 1.0
stale_after = 1.0
evidence_ttl = 5.0
excluded_prim_paths = ["/Franka"]   # Ground truth only.
max_candidate_voxels = 8000000     # Ground-truth candidate work across all prims.
automatic_robot_filter = false    # V2, RGB-D only; explicit opt-in.
robot_depth_tolerance = 0.005     # Meters, optical-depth comparison.
robot_pose_tolerance = 0.01       # Seconds, pose/render timestamp matching.
# update_hz = 5.0                  # Optional limit on explicit update calls.
```

This example enables RGB-D with explicit camera paths; the default source is
ground truth and camera paths default to empty. Omitted `enabled` is false.
The other values shown are defaults. Bounds/voxel budgets are validated
without allocating a map. RGB-D camera paths identify expected input frames;
they do not construct cameras. One camera is valid for mapping. Depth sampling
and ray/voxel-visit budgets bound work independently of the semantic 2 mm
point-cloud settings. More cameras/geometry and finer grids cost time/memory;
equal throughput is not promised. `update_hz` limits caller-driven attempts,
without scheduling them. GT ignores RGB-D settings; RGB-D ignores GT exclusions.

```python
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping.factory import create_spatial_mapping

config = load_scene_config("outputs/my_setup.toml")
session = create_spatial_mapping(
    config.spatial_mapping,
    robot_prim_path=config.robot.prim_path,  # Required for automatic RGB-D filtering.
)  # None when disabled.
if session is not None:
    try:
        session.update(frames)     # RGB-D: already-acquired DepthFrames.
        print(session.map.query((.45, 0., .10)))
    finally:
        session.close()
```

For ground truth, pass `stage=stage` at construction or use the current Isaac
stage, then call `session.update()` without frames. This never calls the semantic
ground-truth provider.

To reuse exact multi-view captures, install `SharedFrames(existing_provider.rig)`
before the owner acquires a new sample; pass that tap as `frame_source` and call
mapping `update()` after the owner's observation. The tap stores those same
DepthFrame objects and timestamps, without capturing again. `own_frame_source=True`
lets session cleanup restore the owner's capture method; the tap never closes
the owner's cameras. For an already initialized single V1 camera,
`CameraSnapshots({camera_path: existing_camera})` borrows its buffers/calibration;
it initializes no cameras and closes only its reader. Close taps/readers before
replacing their acquisition owner. Caller-supplied frames need no adapter.

## V1 validation history and limits

```bash
./run_isaac_python.sh -m unittest \
  tests.unit.test_spatial_mapping_occupancy \
  tests.unit.test_spatial_mapping_rgbd \
  tests.unit.test_spatial_mapping_ground_truth \
  tests.unit.test_spatial_mapping_config_factory
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/spatial-mapping-unit-final
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping \
  --artifacts-dir outputs/spatial-mapping-native
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview \
  --artifacts-dir outputs/spatial-mapping-perception
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/spatial-mapping-v1
./run_isaac_python.sh scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/spatial-mapping-regression
```

The V1 final offline run passed all 528 discovered tests (72 new mapping tests),
eight standalone checks and the architecture/syntax checks. Both mapping native
scenarios passed, as did the six existing multi-view and two V1 scenarios. The
legacy regression group initially passed five of six: its benchmark retained one
stale perceived pose across twelve teleports and exceeded its 15 mm error limit.
The unchanged `main` benchmark and a feature-branch retry both passed, with maximum
position error 2.01 mm and all four Pick/Place trials successful. No source or test
threshold changed for the retry. The intermittent failure's cause remains
unisolated; the initial FAIL is retained alongside successful comparisons.
Thus all sixteen native scenarios have passing results, with that retry caveat.
The durable
[validation summary](assets/spatial_mapping/validation.json) records inspected
scenario results and workstation timings. The GT test uses an isolated USD stage,
exports its fixture/occupancy and checks transformed primitives, invisible and
unsupported geometry, instance proxies, move/remove, units and failed updates.
The RGB-D test uses the existing acquisition path and compares representative
GT/free/surface/occluded queries, not complete grid equality. Inspect result JSON
as well as exit status. Use the established launcher/NumPy environment; do not
use `--no-ros-env`.

The failed scenario was repeated with:

```bash
./run_isaac_python.sh tests/regression/test_benchmark.py \
  --result outputs/spatial-mapping-benchmark-retry/result.json \
  --artifacts-dir outputs/spatial-mapping-benchmark-retry/artifacts
```

The RGB-D test is also the standalone demonstration:

```bash
./run_isaac_python.sh tests/spatial_mapping/test_rgbd.py \
  --result outputs/spatial-mapping-demo/result.json \
  --artifacts-dir outputs/spatial-mapping-demo
```

It saves calibrated capture diagnostics, representative queries, NPZ state and
probability grids, and `ground_truth_xy.ppm`/`rgbd_xy.ppm` slices through z=0.025 m
(unknown gray, free white, occupied red). NPZ arrays use XYZ axes and include
bounds/resolution; array-only exports are diagnostics, while consumers should
prefer query/snapshot methods. Reinitialization borrows the same camera resources.

![Native GT and RGB-D occupancy slice](assets/spatial_mapping/occupancy_slice.png)

The cyan cross marks the queried cube interior: occupied in GT, unobserved in RGB-D.

Measured results on the workstation, with 1 cm voxels:

| Source/scenario | Dimensions | FREE | OCCUPIED | UNKNOWN | Update time |
| --- | --- | ---: | ---: | ---: | ---: |
| GT isolated fixture, final scene | 120×120×70 | 1,002,644 | 5,356 | 0 | 5.9–24.9 ms |
| GT existing demo scene | 68×80×32 | 162,504 | 11,576 | 0 | 5.7 ms |
| RGB-D existing demo, four fresh batches | 68×80×32 | 24,007 | 356 | 149,717 | 373.7–406.8 ms |

The demo uses both existing 640×480 cameras, stride 8, 1,024 total sampled rays,
two-second frame/freshness limits and a 30-second evidence TTL. Times exclude
rendering and semantic processing; source health was fresh at the captured
snapshots. GT and RGB-D agreed on representative visible occupied/free cells;
the cube interior was GT OCCUPIED and RGB-D never-observed UNKNOWN. Frozen and
duplicate camera data retained the prior timestamp and failed freshness checks;
resume restored acquisition. These are bounded functional checks, not claims of
complete obstacle coverage or collision-safe robot motion.

V1 is a bounded dense grid with sampled rays, finite evidence memory and explicit
updates. Thin geometry between sampled rays, calibration/depth noise, occlusion
and unmodeled robot returns limit RGB-D completeness. GT approximations can
overoccupy. Unknown space is not certified free. V2 adds model-derived robot
masks; uncertainty-aware ray evidence and sparse storage remain future work.
This stage supplies occupancy evidence; it implements no collision checking,
avoidance or motion planning.

## V2 automatic robot filtering

Set `automatic_robot_filter = true` for RGB-D mapping and pass
`robot_prim_path=config.robot.prim_path` to `create_spatial_mapping`. There is no
automatic-filter fallback to `/Franka`; the configured robot can live elsewhere.
The default remains off, and GT mapping continues to use its independent
`excluded_prim_paths`. The two positive finite tolerances above are the only new
tuning settings. `robot_pose_tolerance` is a time tolerance, not robot inflation.

The Isaac adapter reads the robot's authored visual meshes, associates them with
their kinematic rigid bodies and caches the local triangle geometry and fixed
mesh-to-body transforms. This requires static visual meshes and static transforms
relative to each rigid body; the standard Franka's arm and finger bodies can
still move through their current PhysX poses. Skinning, deforming meshes and
animated mesh-to-body transforms are not supported. The generic
`world/spatial_mapping/robot_filter.py` builds reusable bounding-volume
hierarchies (BVHs) and intersects calibrated camera rays with the actual
triangles. Bounding boxes accelerate traversal; they do not define the final
mask. There is no silhouette dilation, broad occupied robot envelope, new camera,
extra render pipeline or production segmentation annotation.

The mapper first validates caller masks and fairly selects at most `max_rays`
from valid stride-grid candidates across cameras. The filter predicts the
nearest positive robot surface only for those selected pixels, in optical-depth
coordinates. A measured return
more than `robot_depth_tolerance` in front of it is retained as possible external
geometry. Returns consistent with the robot, and ambiguous returns behind that
expected occluding surface, are withheld. Rays without a robot intersection are
retained. Automatic exclusions are dropped without backfilling other rays.
The automatic boolean masks have the full depth-image shape; unqueried pixels
remain false and are not classified as robot. These are mapping masks rather
than dense robot segmentation images. Factory construction keeps both strides
equal. Diagnostics distinguish allocated, retained and automatically excluded
selected rays, so a robot-dominated view cannot silently consume extra work.

Automatic and caller-supplied `exclusion_masks` are combined with OR without
changing either input. All configured cameras still participate in one complete
batch. Every automatic mask must have the corresponding camera ID, boolean dtype
and depth-image shape. The existing mapper skips each excluded ray completely:
filtering contributes neither occupied robot voxels nor free evidence in front
of or behind that return. Another unmasked camera can still observe the region.
Nothing turns UNKNOWN into OCCUPIED.

### Robot poses and frame time

`DepthFrame.acquired_at` is the local monotonic time when existing camera buffers
were read. Its `token[1]` identifies simulation time for the rendered exposure.
Those clocks serve different purposes: sampling the robot at mapping-processing
time can pair old depth with a newer arm pose. Native investigation found
roughly 33–67 ms of render lag, and USD transforms read in a post-physics callback
could still be one physics step old. These observations motivated pose history;
they are not a fixed lag correction.

Only the enabled Isaac filter owns a post-physics subscription and a bounded
256-sample history of raw PhysX rigid-body poses. For the inspected Panda, the callback
copies 11 body poses; it does no camera reads or mesh intersections. Camera-specific
geometry uses the nearest history sample to each frame's original render
timestamp, subject to `robot_pose_tolerance`. This is a bounded time approximation,
not an exact pose or motion-error guarantee. It does not substitute the latest USD
pose. Cached mesh data are reused as the arm and fingers move. The adapter
borrows the global physics view and owns its subscription and body-view handle.
Backward simulation time clears pose history. Callback errors are reported by
the next mask request; they are not silently ignored. The geometry cache does
not monitor runtime USD mesh, visibility, topology, mesh-to-body transform or
stage edits. Callers must close and manually reinitialize the filter after such
changes, once the model is static again. These edits do not automatically fail
the next update or refresh the cached model.

Construct the mapping session before acquiring the frames to be mapped so its
pose history can collect matching samples. Missing or out-of-tolerance pose
history, invalid poses and detected model-construction failures are explicit
errors, with no unfiltered fallback. This does not imply detection of later
changes to cached geometry. The
RGB-D mapper validates complete camera tokens, age and calibration before
requesting automatic masks. Attempted tokens are consumed before filtering, so
repairing a failed filter cannot relabel the same buffers as fresh evidence.
The normal completion-age check includes filter processing time; explicit
`now=` retains its deterministic offline semantics.

### Explicit use and lifecycle

Existing `SharedFrames` and `CameraSnapshots` adapters continue to borrow camera
resources. Install the mapping session while the simulator is initialized, then
feed new observations through the existing acquisition owner:

```python
from robot_skill_stack.integrations.isaac.spatial_mapping import SharedFrames

tap = SharedFrames(existing_provider.rig)
session = create_spatial_mapping(
    config.spatial_mapping,
    stage=stage,
    frame_source=tap,
    own_frame_source=True,
    robot_prim_path=config.robot.prim_path,
)
try:
    # After the owner acquires a fresh batch and matching pose history exists:
    existing_provider.observe()
    updated = session.update()  # Automatic masks; no manual robot masks needed.
    print(updated, session.robot_filter.diagnostics)
finally:
    session.close()             # Removes the filter subscription and restores the tap.
```

This example requires enabled RGB-D mapping with automatic filtering enabled.
It illustrates explicit calls; bootstrap, semantic perception, the controller,
skills and planner do not start or consume mapping. Any supported camera count
uses those same calibrated snapshots. The filter creates no replacement cameras.

Closing a session is idempotent. It attempts source, filter/history and owned
frame-adapter cleanup even if an earlier close fails, preserving cleanup errors.
Borrowed cameras, the robot, stage and global physics view remain owned by their
existing components. Constructor failures clean resources whose ownership was
acquired; the filter also owns cleanup of its partial initialization.

Enabling filtering does not delete prior unfiltered occupancy. Start a fresh map
with `session.map.clear()` when comparing modes, or allow existing evidence to
clear through later valid free rays or its ordinary TTL. The implementation does
not erase robot-shaped volumes, which could also remove external obstacles.

### V2 verification and remaining limits

SDK-free geometry tests cover calibrated half-pixel projection, actual triangle
intersections, optical depth, foreground/self/behind distinctions, open/closed
finger gaps, multiple views and moving snapshots. Integration tests cover the
opt-in configuration, arbitrary robot path, mask composition, failed/repeated
buffers, processing age and resource ownership:

```bash
./run_isaac_python.sh -m unittest \
  tests.unit.test_spatial_mapping_robot_filter \
  tests.unit.test_spatial_mapping_robot_adapter \
  tests.unit.test_spatial_mapping_self_filter_integration
./run_isaac_python.sh tests/spatial_mapping/test_robot_self_filter.py \
  --result outputs/spatial-mapping-v2/result.json \
  --artifacts-dir outputs/spatial-mapping-v2
```

The final V2 run on 22 October 2026 passed all 573 offline tests, eight standalone
checks and architecture/syntax checks. All 17 native scenarios passed: three
mapping, six multi-camera perception, two V1 perception/manipulation and six
legacy manipulation regressions. The mapping suite includes the unchanged
GT/RGB-D scenarios. The self-filter fixture used seven
configurations, both existing 640×480 cameras and independent instance-ID
annotations on those same render products. Production does not use those labels.
The full [evaluation](assets/spatial_mapping_v2/evaluation.json) records per-pose
reference denominators, state counts, calibration timing and test artifacts.
The [verification record](assets/spatial_mapping_v2/verification.json) includes
all suite results and confirms that all 19 frozen perception/configuration files
are unchanged.

| Configuration | Selected robot returns excluded/observed | Robot-reference OCCUPIED before → after | Selected external returns withheld/observed |
| --- | ---: | ---: | ---: |
| Home, closed | 164/164 | 164 → 0 | 0/12 |
| Extended, open | 167/167 | 167 → 0 | 0/7 |
| Rotated, closed | 154/154 | 154 → 0 | 0/11 |
| Side extended, open | 170/170 | 170 → 0 | 0/11 |
| Moving, sample 4 | 166/166 | 166 → 0 | 0/10 |
| Moving, sample 8 | 160/160 | 147 → 0 | 0/12 |
| Moving, sample 12 | 164/165 | 149 → 1 | 1/10 |

These are exact mapping-selected rays and per-configuration reference cells,
summed across configurations, not unique global voxels. All 62 selected clear
forearm/wrist/gripper/foreground returns were retained. One of 11 selected
occlusion-edge returns was withheld. A separate dense stride-8 validation found
8,253/8,255 robot returns excluded, zero exclusions among the 424 clear adjacent
and foreground returns, and four withheld among 97 occlusion-edge returns.
Dense validation is evaluated separately and does not backfill production rays.

At Home, the 115×120×114 grid (1,573,200 voxels, 1 cm) changed as follows:

| Mode | FREE | OCCUPIED | UNKNOWN |
| --- | ---: | ---: | ---: |
| Unfiltered | 141,940 | 505 | 1,430,755 |
| Filtered | 124,573 | 341 | 1,448,286 |

The filter creates no occupied model envelope. The additional UNKNOWN cells
reflect withheld whole rays, including their formerly cleared prefixes.
Over historical robot-reference cells that GT classifies environment-FREE,
occupied counts fell from 409 to one. This set includes current robot volume;
it is not a measurement of exclusively vacated-space ghosts. Existing unfiltered
ghosts still require valid free evidence, expiration or a deliberate map reset.

No FREE classifications occurred at the 35 GT-occupied probe-center checks.
There were two FREE classifications at selected robot surface-reference cells;
the dense gripper reference also contained one FREE cell in both modes. These
surface-cell checks expose voxel-boundary/sampling uncertainty and are not
complete continuous-volume ground truth. One robot return remained unmasked
while moving, and occlusion-edge external returns can be withheld. These residual
errors remain visible in the report; neither robot exclusion nor FREE is a
geometric collision certificate.

The probe behind the arm had a robot-labeled foreground occluder in the first
camera in every tested pose, and could be seen from the second camera. Its
interior remained UNKNOWN rather than falsely cleared. Delayed-frame masks were
unchanged after advancing the arm to another configuration. Frozen/repeated
sources did not refresh map timestamps; resume, reinitialization and cleanup
passed. Pose-history matches had zero measured simulation-time skew in this run,
within the configured 10 ms limit.

![Native self-filter accuracy and occupancy](assets/spatial_mapping_v2/self_filter_evaluation.png)

Median model lookup/filtering was **227 ms** (207–238 ms) for 1,024 selected rays.
One-time geometry/BVH initialization was 507 ms. The 11-body post-physics history
capture averaged 0.130 ms, with a 0.202 ms maximum over 240 callbacks. The larger
full-arm grid took 2.165 s unfiltered and 2.135 s filtered at the median: dropped
robot rays offset the extra filtering work. These mapping times should not be
compared directly with V1's smaller tabletop ROI. They exclude rendering and the
separate dense validation pass. The comparison uses five-second acquisition and
freshness limits because it runs both maps sequentially on the same original
snapshot; timestamps are never rewritten. An initial run alongside CPU-heavy
units correctly expired, so final native timing runs were isolated.

Reproduce the final verification in sequence, with each native run isolated:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/spatial-mapping-v2-unit-final
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping \
  --artifacts-dir outputs/spatial-mapping-v2-final
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview \
  --artifacts-dir outputs/spatial-mapping-v2-perception
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/spatial-mapping-v2-v1
./run_isaac_python.sh scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/spatial-mapping-v2-regression
```

V2 is ready for broader spatial-mapping validation on additional layouts, speeds,
thin obstacles and close-contact gaps. The results support practical robot
exclusion with clear foreground preservation, while the remaining edge errors
need continued evaluation. Runtime integration and motion planning remain
separate tasks.

This is model-based evidence filtering for collision-avoidance support, not a
certified safety mechanism. Geometry/calibration error, render/pose mismatch and
stride sampling can leave robot returns or withhold nearby external returns.
Camera calibration is trusted as supplied: native validation uses fixed world
cameras, and because existing acquisition reads K/T with the buffers rather than
recovering historical extrinsics, moving cameras require caller-supplied frames
calibrated at exposure.
An obstacle within the depth tolerance can be ambiguous; withholding it does not
mean the space is free. Missing views and occlusion still limit map coverage, and
old evidence can persist until cleared or expired. Practical readiness depends
on the native obstacle-preservation and latency results; it does not establish
collision-safe motion or authorize runtime/planning integration.

## V3 CPU performance

V3 continues from `d260377` without changing mapping bounds, resolution,
sampling, camera configuration, ray budgets, probability settings or the V2
robot-depth rules. The unchanged seven-configuration native fixture first
reproduced 2,133 ms median filtered updates, 2,190 ms unfiltered and 223 ms
filter computation. Its 15 fresh updates are a reproduction check; the
controlled before/after benchmark uses additional warm-up and repeated identical
recorded observations. The original
[profile](assets/spatial_mapping_v3/baseline_profile.json) retains this distinction.

A focused SDK-free profile of the original Home observation found scalar voxel
traversal consumed about 92% of `RgbdMapper.update`: 147,009 visits generated
over 3.4 million function calls. Occupancy integration took about 29 ms, expiration
8 ms and a separately measured full immutable snapshot 17 ms. These measurements
support optimizing traversal rather than replacing the dense grid.

`RgbdMapper` now advances finite rays in bounded batches of at most 256. It
retains slab clipping, half-open grid bounds, `nextafter` entry selection,
positive-length cells, simultaneous crossings and the original endpoint rules.
Traversal emits flat integer voxel IDs; NumPy deduplicates them per view and
across views, with occupied endpoints winning over free evidence. Every visit
still counts before deduplication. Exceeding the configured global budget
rejects the complete update without integrating partial evidence. The visit
buffer is bounded by the remaining budget plus one, not the grid volume times
the ray count.

The occupancy engine, dense-grid representation, immutable snapshots and query
API remain unchanged. Repeated flat-index validation inside integration is now
a small cost and does not justify another internal update API. New differential
tests retain a frozen V2 scalar reference and compare complete map states,
probabilities, observation/expiration arrays, timestamps and diagnostics,
including boundary, budget and multi-camera cases. Two existing clock-delay
tests inject their delay at the new traversal entry point; their freshness and
original-timestamp assertions are preserved.

The accurate mesh filter and Isaac pose-history adapter are unchanged. Its
separate seven-configuration profile attributed about 48% of cumulative time to
BVH box tests and 22% to triangle cross products; repeated norm calculations
accounted for only 4%. A research-only norm/leaf-array cache preserved all masks
but changed median lookup from 178.48 to 175.28 ms, just 1.8%. This was too small
to justify another cached layout, so the prototype was discarded. Neither
triangle geometry nor tolerances were weakened for performance.

The reproducible benchmark captures the exact V2 fixture from baseline Git
objects into an isolated package namespace, including original depth/RGB,
calibration, tokens, acquisition timestamps and exposure-matched mesh poses.
Capture I/O is excluded from performance claims. CPU replay builds each version's
mesh/BVH once, uses the same complete fifteen-update sequence and map resets,
and alternates baseline/current pass order. It times updates separately from
snapshots, queries, equality comparisons and disk writes. Explicit replay time
is the original observation timeline; replay is not evidence of live freshness.
The original capture manifest is retained, with a documented correction to its
reset annotation before the moving sequence.

Benchmark-only line attribution and cProfile runs provide phase evidence;
their instrumented times are excluded from headline medians. Scalar traversal
includes per-visit set insertion, whereas batched traversal and array dedup have
separate costs. Expiration nested inside integration is reported as overlapping
work rather than added twice. No profiling hooks run in normal mapping.

Reproduce the performance comparison on the Isaac workstation, with no other
CPU-heavy work or simulator running:

```bash
./run_isaac_python.sh scripts/benchmark_spatial_mapping_performance.py \
  --capture-only --baseline-ref d260377 \
  --artifacts-dir outputs/spatial-mapping-v3-capture \
  --result outputs/spatial-mapping-v3-capture/result.json
./run_isaac_python.sh scripts/benchmark_spatial_mapping_performance.py \
  --replay outputs/spatial-mapping-v3-capture --warmup 2 --repeats 5 \
  --semantic --profile --memory \
  --artifacts-dir outputs/spatial-mapping-v3-performance \
  --result outputs/spatial-mapping-v3-performance/result.json
./run_isaac_python.sh scripts/check_spatial_mapping_live_freshness.py \
  --artifacts-dir outputs/spatial-mapping-v3-live-final \
  --result outputs/spatial-mapping-v3-live-final/result.json
```

The baseline commit must exist in the local Git history. Capture/replay validates
input file hashes and records source and script hashes; benchmark PASS requires
those sources to stay unchanged during execution. Raw captures, profiler files
and verbose logs stay under ignored `outputs/`. The separate live check uses
real production clocks and the original acquisition timestamps, with
`max_frame_age=stale_after=1` for its independent filtered maps. Only its nested
`live_default_freshness` measurements are timing evidence: the original fixture's
unfiltered timers include the injected freshness check.

### Controlled mapping results

On the Ryzen 7 7800X3D workstation, two warm-up passes preceded five measured
passes with alternating baseline/current order. Each pass processed all fifteen
captured batches, retaining mapper rotation/token history and the native map
resets. The table uses the last static batch and each moving configuration:
35 representative updates per mode/version across seven configurations.

| Same-input CPU update | Baseline median / p95 / max, ms | V3 median / p95 / max, ms | Median reduction |
| --- | ---: | ---: | ---: |
| Full-arm, robot filtering enabled | 1,763.9 / 1,812.2 / 1,822.5 | 252.8 / 266.7 / 287.8 | 1,511.1 ms, 85.7% |
| Full-arm, filtering disabled | 1,785.5 / 1,839.9 / 1,853.4 | 77.4 / 81.3 / 82.2 | 1,708.1 ms, 95.7% |
| Filter with recorded-pose assembly | 182.1 / 201.5 / 206.0 | 182.9 / 197.5 / 220.0 | Unchanged implementation |

Every measured batch, including predecessors omitted from the timing summary,
matched the baseline's complete state, probability, observation/expiration and
last-observed arrays, map metadata and fixed spatial queries. Production-selected
pixels and robot masks also matched the original capture exactly. There were no
runtime errors or output mismatches. This preserves V2's residual errors as well
as its successful exclusions; it does not improve or certify geometric accuracy.

The separate native real-clock check passed fifteen fresh filtered updates:
median/p95/max 300.9/311.2/313.0 ms, with a maximum original-observation age of
335.6 ms at completion. Paired camera acquisition took 5.6/7.6/8.2 ms. The unchanged
full-arm ground-truth mapper took 11.2/17.3/20.4 ms across eight native reference
updates. These native measurements include stage/camera access and differ from
SDK-free replay; they are not interchangeable timing distributions.

The required uninstrumented native self-filter fixture also passed. Its fifteen
fresh filtered updates changed from the reproduced baseline's
2,132.7/2,189.7/2,207.3 ms median/p95/max to 303.6/314.8/321.8 ms, an 85.8% median
reduction. Unfiltered updates took 90.2/95.9/96.5 ms. All native selected-reference
quality counts matched the V2 reproduction: 1,145/1,146 robot returns excluded,
robot-reference occupied cells 1,117 → one, all 62 clear nearby-obstacle returns
retained and one of eleven occlusion-edge returns withheld. The same two robot
surface-reference FREE cells remained; no FREE appeared at the 35 GT-occupied
probe-center checks. The measured speedup preserves the documented residual
errors and accuracy limits.

No default timing limits were increased. The measured isolated full-arm workload
now fits the original one-second acquisition/freshness defaults. The map keeps
the oldest contributing acquisition timestamp, so processing and earlier
consumers still consume the same freshness budget. Slow or delayed inputs remain
explicit failures; the deterministic replay clock is never substituted in live
mapping. TTL and repeated/frozen-frame behavior are unchanged.

### Semantic perception and scheduling scope

The semantic pipeline is unchanged. Its existing diagnostics split acquisition,
projection/cropping/per-view voxelization, fusion, segmentation/refinement,
fitting/finalization, observability and tracking/publication. The combined
benchmark processes each original camera batch through semantic perception and
then a filtered map, directly measuring both baseline and V3 schedules. It
compares semantic class, persistent ID, geometry, confidence, grasp readiness,
metadata and original timestamps across schedules and repeats. Timing on these
full-arm views is distinct from the broader frozen tabletop research protocol.

Existing `SharedFrames` can reuse the same acquired batch without another camera
read. Sharing the semantic cloud would change mapping evidence: semantic
projection uses stride 2, workspace/support cropping and voxel centroids, whereas
mapping uses the selected stride-8 depth rays and needs their entire free-space
segments. Calibration arithmetic is small relative to fitting and self-filter
intersection work. No new shared preprocessing framework is justified here.

Both APIs remain synchronous. Calling a roughly 300 ms native map update inside
a simulation/control stepping loop would delay that loop for its processing
duration, even though the pose-history callback itself remains small. Combining
consumers adds their sequential latency and consumes more observation age.
The replay benchmark does not demonstrate asynchronous execution or controller
deadlines. Runtime scheduling and planner/controller integration remain future
work; no worker threads, control callbacks or motion behavior were changed.

The measured semantic-only median/p95/max was 103.0/105.8/109.1 ms. Fitting took
86.9 ms at the median, compared with projection 4.1 ms, fusion 0.5 ms,
segmentation 5.0 ms, observability 6.3 ms and tracking/publication 0.1 ms. No small
projection/copy optimization would materially change this fitting-dominated
workload, so all frozen perception code and its limits were retained. All 35
representative samples contained four confirmed objects; class/identity,
geometry, confidence and grasp-readiness signatures were identical across the
semantic-only and both combined schedules.

| Sequential same-input workload | Median / p95 / max, ms |
| --- | ---: |
| Perception plus baseline filtered mapping | 1,873.7 / 1,916.3 / 1,917.6 |
| Perception plus V3 filtered mapping | 360.8 / 387.0 / 406.5 |

The measured combined median reduction is 1,512.9 ms (80.7%); this is a directly
timed sequence, not a sum of separate medians. Original observation and map
timestamps were preserved. Its roughly 0.4 s tail is a CPU replay result, not a
live deadline guarantee across other layouts, speeds or fitting workloads.

### Phase and memory evidence

Separate line-attributed Home profiling explains the gain. These instrumented
single-update values are not substitutes for the warm latency distributions:

| Filtered Home phase | Baseline ms | V3 ms |
| --- | ---: | ---: |
| Recorded-frame validation | 0.66 | 0.61 |
| Candidate/selected rays | 0.21 | 0.20 |
| Robot geometry/masks, with tracing overhead | 256.0 | 270.6 |
| World-coordinate projection | 0.27 | 0.24 |
| Traversal; scalar also includes per-visit insertion | 2,192.1 | 52.4 |
| Remaining accumulation/deduplication | 64.9 | 9.9 |
| Probabilistic integration, including nested expiry | 8.18 | 8.29 |
| Expiry across session/integration, overlapping the above | 0.84 | 0.94 |

Snapshot/query work was timed separately without the line profiler. Full
immutable snapshots had median/p95/max 6.22/17.58/18.22 ms before and
6.00/6.63/12.53 ms after; 1,000-position queries took 0.251/0.262/0.269 ms and
0.236/0.284/0.319 ms. Those implementations were not changed, and their timing
variation is not claimed as an optimization. Physical camera acquisition and
native stage access are reported separately above.

Separate memory workers measured traced peak update allocations of
25,791,774 → 15,732,616 bytes, a 39.0% reduction. Resident memory after the update
was 264.2 → 237.2 MB. These processes deliberately retained all fifteen captured
batches (101.4 MB of frame buffers), cached geometry and a map; they are not
production application RSS measurements. Process high-water RSS was already
398.2 MB before each probe and stayed there, so no process-peak reduction is
claimed. Tracemalloc was excluded from timing runs. The dense map still occupies
the same storage; the reduction comes from traversal/accumulation temporaries.

![Controlled V3 latency and allocation comparison](assets/spatial_mapping_v3/performance.png)

The compact [evaluation](assets/spatial_mapping_v3/evaluation.json) records full
distributions, configuration scopes, 30 baseline/current full-array/query
comparisons, 150 exact captured-mask/selection checks and 300 recorded consistency
checks, of which 240 compare later repetitions with their first measured result.
The remaining 60 establish those initial signatures. Semantic equivalence and
timestamp checks are recorded separately. Detailed replay inputs, samples and
profiles remain local artifacts rather than committed bulk data.

### Completed V3 validation

On 22 October 2026, all 582 discovered offline tests (nine new), eight standalone
checks and architecture/syntax checks passed. All 17 native scenarios passed:
three spatial mapping, six multi-camera perception, two V1 perception/manipulation
and six legacy manipulation regressions. All 19 frozen perception/configuration
hashes remained unchanged. The [verification record](assets/spatial_mapping_v3/verification.json)
contains exact commands, artifact paths, result hashes and protected-source
checks. Reproduce the final groups sequentially:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/spatial-mapping-v3-unit-final
for mapping_group in spatial_mapping multiview isaac regression; do
  ./run_isaac_python.sh scripts/run_all_regressions.py --group "$mapping_group" \
    --artifacts-dir "outputs/spatial-mapping-v3-final-$mapping_group" || break
done
```

V3 meets the 500 ms full-arm stretch target on this tested configuration with
equivalent occupancy quality. Accurate robot intersection is now the largest
mapping cost; semantic fitting dominates the other consumer. Larger ray budgets,
regions, complex scenes and longer observation queues still need measurement.
The validated subsystem remains independent and explicitly updated; runtime
integration, motion planning and controller scheduling have not begun.
