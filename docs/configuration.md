# Setup configuration

Use one complete TOML file per scene/setup, based on
`config/scenes/playground.toml`. Open the matching USD scene in Isaac, select the
file in the GUI and initialize. There is no inheritance, live reload or dynamic
switching. Restart Isaac before changing setups or robot prims.

```toml
[robot]
backend = "franka"
prim_path = "/Franka"

[world]
provider = "perception"

[scene]
support_plane_z = 0.0
workspace_min = [0.20, -0.40, 0.0]
workspace_max = [0.70, 0.40, 0.22]

[perception]
camera_prim_path = "/World/PerceptionCamera"

# Optional oracle references, used only by ground truth.
[ground_truth]
objects_root = "/World/Objects"

[ground_truth.objects.cube]
prim_path = "/World/Objects/Cube"
size = [0.05, 0.05, 0.05]
```

## Required settings and source selection

| Section | Meaning |
| --- | --- |
| `robot` | Backend selector and existing absolute robot prim path; only `franka` is implemented |
| `world` | `provider` is `perception` (geometry V1) or `ground_truth`; unknown names fail |
| `scene` | Finite support height and increasing three-dimensional world-coordinate workspace bounds, in meters; also used by grasp planning |
| `perception` | Existing camera prim for RGB-D; optional resolution defaults to `[640, 480]` |
| `ground_truth` | Existing object-root prim for ground truth; optional named object overrides |

RGB-D may omit the entire ground-truth section; it needs no oracle object IDs,
sizes, paths or SDK wrappers. Ground truth may omit the camera section. Both
need robot and scene settings. Only selected-source prims are validated.

To use ground truth without changing core code, copy the complete setup:

```bash
mkdir -p outputs
cp config/scenes/playground.toml outputs/playground_ground_truth.toml
```

Edit only `[world].provider` to `"ground_truth"` in the copy, restart Isaac and
select that file. The existing ground-truth section supports the playground.
This is a local setup copy, not a configuration layer. Automated regression
helpers make their own complete source-specific copies under their artifacts.

Ground truth discovers children of `objects_root`; named entries optionally
override `size` and `graspable` (default true). Keys must match the source's
object IDs (custom `object_id` or lowercase prim name). `prim_path` is validated
for each configured entry. Omitted size is measured from USD bounds.

## Deliberately small tuning surface

`world.update_hz` optionally sets a positive finite sampling rate. Omitted:
RGB-D runs at 5 Hz, ground truth at every physics callback.
`world.stale_object_ttl` optionally sets a positive finite expiry duration
(default 15 s). The updater's existing freshness budget is 1 s. Resolution must
contain two positive integers. Invalid references/bounds, unknown selectors,
fields and sections are rejected instead of selecting a fallback.

Discovery/tracking/fitting and planner settings use existing Python defaults;
they are not duplicated in TOML. The robot factory supplies gripper geometry.
Franka retains 5–75 mm usable grasp width, 80 mm open width, and the validated
palm/finger geometry. The production setup keeps its original V1 thresholds,
grasp offsets, clearances and controller tolerances.

For a new supported scene, change prim references, support height and workspace
bounds in one copied file. A different robot requires a backend implementation
and explicit factory/config selection first; configuration alone cannot create
an unimplemented robot. See [architecture](architecture.md).

## Migration from earlier profiles

Phase 4B replaced `robot.id`/`type` with `robot.backend`, moved
`world.objects_root` and top-level `objects` under `ground_truth`, and moved
support-plane/workspace fields from `perception.discovery` into `scene`.
Algorithm tuning sections, `perception.type` and `grasping` overrides were
removed. Camera resolution, graspability, expiry and rate may be omitted when
using defaults. Old custom profiles must be migrated; no compatibility parser
silently accepts retired settings. The checked-in playground is the reference.
