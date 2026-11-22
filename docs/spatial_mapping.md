# Spatial Mapping

Spatial Mapping is an optional occupancy subsystem, independent of the semantic
WorldModel. Bootstrap constructs it through the shared asynchronous observation
runtime when `[spatial_mapping].enabled = true`. There is no standalone synchronous
session/factory in the supported runtime.

## Sources and ownership

RGB-D input contains original camera exposures, calibration and synchronized robot
information. A shared frame pool reuses semantic camera buffers when the selected
paths overlap. Owner-thread acquisition never gives SDK camera/articulation
handles to the worker. GT mapping instead copies explicitly selected collider
geometry; it is separate from sensor-derived RGB-D input.

The existing bounded ProcessWorker owns mutable computation. Mapping updates,
robot exclusion, obstacle extraction and optional reconstruction finish in that
same worker. Publication is an immutable completion bound to its input generation,
revision and timestamp. Read-only arrays backed by mutable storage are copied;
transport never shares mutable parent/child memory. Diagnostics are immutable
mappings and use the existing transport/publication contracts.

## Configuration and budgets

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

Supply scene-appropriate bounds in the full TOML. Omitted RGB-D mapping camera
paths inherit selected semantic cameras; GT semantics require explicit mapping
camera paths. The nominal global fixture keeps the original two 640×480 cameras,
semantic 5 Hz and shared 1,024-ray mapping budget. All configured resolution,
ray, voxel-visit, cell and memory bounds remain validated; do not raise them to
work around a rejected scene. Default application profiles retain direct motion.

FREE, OCCUPIED, UNKNOWN and OUT_OF_BOUNDS remain distinct. Sparse support retains
only original occupied cells. Fixed temporal support can retain strongly accumulated
occupied evidence; it does not fill UNKNOWN gaps or expand cuboids. Evidence TTL
and map freshness are separate checks. Capture timestamps are never renewed just
because a result is read or serialized.

## Read views and global planning

Use `bundle.spatial_mapping` to access the existing service. `snapshot()` returns
an immutable `MappingReadView`; occupancy queries and diagnostic reads do not
mutate the map. Stop/Play, reset, robot-model invalidation and shutdown prevent
late or wrong-generation results from becoming accepted publications.

When frozen RGB-D global motion is selected, the worker also prepares bounded
occupied cuboids. Extraction can omit detail when the budget binds; diagnostics
retain represented/omitted counts. Reconstructed mode qualifies measured static
yaw boxes in the same completion and preserves all originally selected cuboids
through retention or the existing eight-corner containment proof. Floor evidence
is protected and combined camera geometry must remain within 64 proxies. Known
support is separate. No GT obstacle geometry or fixture identity enters inference.

Enhanced collision geometry and automatic robot filtering are required for this
global source. The selected source must pass the original strict one-second
freshness and identity checks. Incomplete six-face/visibility evidence, FREE
contradictions, floor/base collisions and over-budget worlds fail explicitly.
The occupied representation remains selectable and does not infer hidden volume.
See [motion planning](motion_planning.md).

## Lifecycle and validation

The observation coordinator owns subscriptions, cameras, robot snapshots and
worker scheduling. Stop invalidates freshness; Play uses a new generation and
reacquires synchronized information. Source/worker failures cannot silently
publish stale geometry. Close is idempotent, releases owners and ignores late
completions; reinitialization uses distinct resources.

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping --artifacts-dir outputs/mapping
./run_isaac_python.sh scripts/run_all_regressions.py --group global --artifacts-dir outputs/global
```

Inspect actual source timestamps, generation, robot-filter health, motion/contact
results and cleanup. Nominal partial RGB-D global detours have executed, but
throughput/freshness, hidden-surface, floor/base and proxy-budget limits remain.
A later persistent-session observation can refuse a return. Mapping is not a
continuous-safety certificate or a replacement for collision validation.
Historical large mapping/performance experiments remain on the preserved
[development branch](https://github.com/BuilderCrafter/robot-skill-stack/tree/feature/rgbd-rrt-autonomous).
