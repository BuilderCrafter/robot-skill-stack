# Spatial mapping V1

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
Automatic geometry-aware robot masks are **not implemented**. Callers must supply
reliable masks; otherwise robot returns can enter the environmental map. Ground
truth robot exclusion does not automatically mask depth images.

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
session = create_spatial_mapping(config.spatial_mapping)  # None when disabled.
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

## Verification and limits

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

The final offline run passed all 528 discovered tests (72 new mapping tests),
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
overoccupy. Unknown space is not certified free. Future work includes reliable
robot masks, uncertainty-aware ray evidence and sparse storage, without changing
the consumer contract. This stage supplies occupancy evidence; it implements no
collision checking, avoidance or motion planning.
