# Motion planning

Direct RMPflow is the default and preserves ordinary manipulation. Static GT
Lula RRT is the validated global option. Frozen RGB-D RRT is experimental and
explicitly selected at initialization; it uses the same planner and executor.

## Configuration

For static GT, set `[motion]` to `strategy = "global"`,
`global_collision_model = "enhanced"`, `global_obstacle_source = "static"`.
This option works independently of mapping. The stock model remains available;
the enhanced YAML/JSON resources are under `config/motion/` and retain their
validated Franka geometry/finger contract.

For RGB-D:

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "frozen_rgbd"
global_rgbd_representation = "reconstructed"
max_obstacles = 64

[spatial_mapping]
enabled = true
source = "rgbd"
automatic_robot_filter = true
update_hz = 2.0
stale_after = 1.0
max_frame_age = 1.0
evidence_ttl = 5.0
```

Use the existing two-camera configuration and mapping bounds; the nominal
acceptance fixture retains 640×480, 1,024 shared rays, 10 mm voxels, semantic 5 Hz,
mapping 2 Hz, seed 123456 and 10,000 iterations. Camera paths may be inherited
from the selected semantic RGB-D source. GT semantics require explicit mapping
camera paths. See [configuration](configuration.md).

`occupied` remains the default RGB-D representation and uses the original
selected occupied cuboids. It does not complete hidden obstacle volume.
`reconstructed` requires global frozen RGB-D, the enhanced model and exactly the
64-proxy budget. It uses measured static yaw boxes only when all six face bounds,
calibrated visibility/edge witnesses, robot exclusion and FREE contradictions
pass the existing qualification rules. Unsupported hidden bounds are refused.

Every original selected cuboid must be retained or proven contained by all eight
corners in a qualified box. Floor cuboids are protected. The combined camera
world must remain within 64 proxies, plus separately allowlisted known support.
This preserves the selected occupied union; bounded extraction may still omit
other observed occupied detail and reports that limitation.

## Execution and limitations

The selected planner obtains its own ordinary IK. Joint limits, self-collision,
fixed-base guard, route/spline checks, trajectory limits, tracking bounds,
settling, cancellation and solver restoration remain active. Planning uses one
fixed-time snapshot, with zero active-world refreshes. It is not dynamic obstacle
avoidance. Intended-contact and carrying restrictions remain; direct is the
ordinary Pick/Place baseline. Home is not intrinsically a collision-safe target.

Reconstructed RGB-D uses bounded cooperative admission and the original strict
<1 s source freshness. Original acquisition timestamps are never refreshed to
make an archived or stale world eligible. Failure does not select GT, direct or
another exposure silently. The retired reactive obstacle strategy is rejected
by MotionConfig.

The original nominal partial scene has demonstrated genuine RGB-D obstacle-induced
RRT detours and actual motion. This is not general scene reliability:

- Floor observations may conservatively reject the fixed robot base.
- Incomplete/contradictory surfaces or unsupported hidden bounds reject geometry.
- Sparse evidence can require 65 proxies and exceed the unchanged 64 limit.
- Mapping throughput and freshness headroom remain limited.
- Later observations in a persistent session may refuse a return movement.

Use GT as the primary demonstration. RGB-D remains a secondary experimental
candidate. Native acceptance retains an independent sampled cooked-geometry veto
before execution; GT reference geometry is confined to that test evaluator.
No sampled result certifies continuous swept safety, arbitrary obstacles or payloads.

Run the sequential native checks with
`./run_isaac_python.sh scripts/run_all_regressions.py --group global --artifacts-dir outputs/global`.
The source branch preserves the full historical research record; it is not a
runtime dependency. See [validation](validation.md) and [integration](integration-rgbd-global-rrt.md).
