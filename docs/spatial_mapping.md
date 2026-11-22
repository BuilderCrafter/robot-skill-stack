# Spatial Mapping

Spatial Mapping is optional occupancy mapping, independent of the semantic
WorldModel. Bootstrap creates it through the observation runtime when
`[spatial_mapping].enabled = true`. The supported runtime has no standalone
synchronous mapping session.

## Sources and ownership

RGB-D frames retain their original calibration, render token and monotonic
acquisition time. The token's timestamp is simulation time: robot poses must
match that exposure, while source freshness uses acquisition time. A shared
frame pool reuses camera buffers when semantic perception and mapping overlap.
SDK acquisition and publication stay on the Isaac owner thread.

The robot adapter caches the configured robot's visible, unsubdivided triangle
meshes once. It records NumPy PhysX transforms after physics steps in a bounded
history, then prepares mesh instances for each camera exposure. Missing history
or an authored model change refuses the observation. Preparation preserves USD
stage units and static mesh scaling; the numerical worker performs intersections
and depth masking.

GT mapping instead prepares active USD geometry, excluding configured prim paths
and noncolliding guides. Invisible geometry remains included. Cubes, spheres and
cylinders retain their affine shapes; other Gprims use conservative world AABBs.
This source is separate from RGB-D evidence and semantic object identities.

The existing ProcessWorker owns mutable computation: ray integration or GT
replacement, robot filtering, obstacle extraction and optional reconstruction.
Each completion groups occupancy and its optional geometry for one input
generation. Acquisition and process transport own independent numeric buffers;
published occupancy arrays are read-only and can outlive their worker or service.

## Configuration and limits

```toml
[spatial_mapping]
enabled = true
source = "rgbd"
automatic_robot_filter = true
update_hz = 2.0
voxel_size = 0.01
max_rays = 1024
stride = 8
max_frame_age = 1.0
stale_after = 1.0
evidence_ttl = 5.0
```

Supply scene-appropriate bounds in the full TOML. Omitted RGB-D camera paths
inherit selected semantic cameras; GT semantics require explicit mapping camera
paths. The nominal global fixture uses two 640×480 cameras, 5 Hz semantics and
a shared 1,024-ray mapping budget. Default application profiles use direct motion.

Configuration checks source choices, prim paths and positive sampling/time
limits. Grid allocation is capped by `max_voxels`; RGB-D integration enforces
shared ray and voxel-visit budgets, and GT rasterization caps candidate cells.
Sampled depths contribute only when finite and positive. Source age is checked
before and after computation and again when a result is published or selected.
These checks do not provide general validation of arbitrary array layouts,
robot models or USD configurations. Numerical code trusts internally prepared
data for the fixed scene. Keep the declared experimental budgets unchanged.

FREE, OCCUPIED, UNKNOWN and OUT_OF_BOUNDS remain distinct. Occupancy accumulates
log-odds evidence with one vote per cell per update; occupied hits override
same-update misses. Expiry is derived from each cell's last observation time.
Queries expose expired evidence as UNKNOWN without changing its history. A new
measurement discards an expired prior. Evidence TTL and source freshness are
separate: reading or serializing a result never renews its timestamp.

## Read views and global planning

`bundle.spatial_mapping.snapshot()` returns a `MappingReadView` evaluated at one
fixed time. Its sparse queries age the selected occupancy without mutating it.
Frozen RGB-D capture validates and returns that view directly, retaining the
original camera records, timestamp, generation and revision.

When frozen RGB-D global motion is selected, the worker also prepares bounded
occupied cuboids. Sparse support retains original occupied cells; repeated
RGB-D evidence can retain confident isolated hits. Packing fills no FREE or
UNKNOWN gaps, although the proxy budget can omit occupied detail. An
`ObstaclePacket` contains selected boxes and support height; its enclosing
completion supplies publication identity. Each box expires with its oldest cell.

Reconstructed mode qualifies measured yaw boxes in the same completion. Observed
cuboids remain unless all eight corners lie inside inferred geometry. Floor
cuboids are retained and combined camera geometry is limited to 64 proxies.
Missing six-face evidence and contradictory measured FREE rays refuse inference.
No GT obstacle geometry or fixture identity enters reconstruction. Known support
is captured separately through the planner's explicit support-plane allowlist.

The RGB-D global source requires enhanced collision geometry and automatic robot
filtering. Selection enforces current generation, healthy fresh occupancy and
nonempty, unexpired cuboids within budget. The nominal freshness limits are one
second. Reconstructed requests admit the first new accepted publication and
refuse source changes or failures instead of searching later exposures. Floor/base
collisions still refuse planning. Occupied mode remains available without hidden
volume inference. See [motion planning](motion_planning.md).

## Lifecycle and validation

The coordinator owns update/timeline subscriptions and worker scheduling. It
borrows semantic cameras and closes only additional mapping cameras. Stop/Play
resets camera admission, robot pose history and worker publication barriers;
Play rebinds a replaced PhysX view and requires fresh observations. Source or
worker failures invalidate pending results. Close releases owned resources and
prevents late publication; retained read views keep their original data and time.

Use focused offline tests for the changed boundary. Planned native checks and
their existing launch commands are described in [validation](validation.md).
Inspect original timestamps, generation, motion/contact results and cleanup.
Nominal partial RGB-D detours have executed, but freshness, hidden-surface,
floor/base and proxy-budget limits remain; persistent-session requests can refuse.
Sampled checks are not a continuous-safety certificate. Historical large
experiments remain on the preserved
[development branch](https://github.com/BuilderCrafter/robot-skill-stack/tree/feature/rgbd-rrt-autonomous).
