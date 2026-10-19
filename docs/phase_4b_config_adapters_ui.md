# Phase 4B — Configuration, adapters and GUI

Branch: `refactor/phase-4b-config-adapters-ui`, based on main `1d9141f`
(Phase 4A merge, PR #5). Before work started, the operator confirmed Phase 4A's
automated tests, both-source GUI checks, pause/resume, repeated initialization
and shutdown, and injected-loss diagnostic passed. Those checks were not rerun
in the cloud. Phase 4B remains unmerged and needs its own workstation validation.

## Configuration and migration

One complete TOML file describes a setup. The playground file shrank from 46 to
22 lines; it retains the same effective robot, camera, workspace and algorithm
settings. No inheritance, registry, overrides layer or live switching was added.

Before, the file duplicated robot `id`/`type`, default camera resolution,
graspability, expiry and all discovery/tracking/fitting thresholds. Object
references lived at top level and support geometry under perception. After:

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

# Optional oracle references: used only when world.provider = "ground_truth".
[ground_truth]
objects_root = "/World/Objects"

[ground_truth.objects.cube]
prim_path = "/World/Objects/Cube"
size = [0.05, 0.05, 0.05]
```

For RGB-D, the ground-truth section may be omitted entirely. For ground truth,
set `provider = "ground_truth"`; the camera section may be omitted. Ground truth
still discovers children of its configured root; optional named object entries
provide size/graspability overrides as before. Scene geometry remains required
for grasp planning regardless of source. Configure a new setup by copying this
single file and changing the prim references and scene geometry, then restarting
Isaac before initializing it. This phase does not add scene loading to the GUI.

| Previous setting | Current location or behavior |
| --- | --- |
| `robot.id`, `robot.type` | `robot.backend`; SDK scene name is internal to the adapter |
| `world.objects_root` | `ground_truth.objects_root` |
| `objects.<id>` | `ground_truth.objects.<id>`; optional `size`, `graspable=true` default |
| `perception.discovery` support height/bounds | Required `[scene]` properties |
| Other discovery, tracking and primitive settings | Existing algorithm constructor defaults; no TOML tuning sections |
| `perception.type` | Removed; RGB-D means geometry V1 |
| `grasping` and nested safety/gripper overrides | Existing planner defaults and adapter-owned gripper construction |
| `perception.resolution` | Optional, default `[640, 480]` |
| `world.update_hz` | Optional; default RGB-D 5 Hz, ground truth every physics callback |
| `world.stale_object_ttl` | Optional, default 15 s; source freshness budget unchanged |

Unknown selectors, obsolete sections/fields, invalid bounds, nonpositive
resolution/rates/TTL and missing selected-source references are rejected. The
selected robot/source prims must exist before World construction. RGB-D does not
validate oracle object paths or construct their wrappers. Existing local custom
TOMLs must be migrated; no compatibility parser silently accepts old settings.

Preserved effective values include workspace `[.20, -.40, 0]`–`[.70, .40, .22]`,
support height 0, camera 640×480, discovery height .008–.22 m, components
30–30000 pixels, candidate extent .18 m, spawn extent .02–.12 m, compactness 3.5,
support tolerance .015 m and neighbor distance .02 m. Tracker remains .18 m,
size ratio 1.5 and three confirmations. Fitting remains threshold .58, margin
.07, top band .15, tolerance .0008 m and 1600 points.

Grasp approach/lift remain .10/.12 m, z offset 0, safety clearance .005 m,
contact overlap .008 m, maximum height .20 m and approach clearance .02 m.
Franka keeps .005–.075 m usable grasp width and .080 m open width, with the
same palm/finger geometry. Backend tolerances remain .01 m and .05 rad with
1000 motion steps. Tests check these values; no benchmark success or placement
accuracy threshold was relaxed.

## Construction and adapter boundaries

```text
one setup TOML → config validation → robot + selected-source prim validation
    → existing World/context → robot factory's SDK wrapper → reset/play
    → 30 Kit updates → source factory (RGB-D initialization + 60 Kit updates)
    → WorldModel/updater → robot factory's backend + gripper spec
    → planners + skills → RobotRuntime(registry, same WorldModel)
    → initial observation + existing owned physics callback → runtime bundle
```

`integrations/isaac/manipulation/factory.py` has explicit Franka selection;
`integrations/isaac/world/factory.py` has explicit RGB-D/ground-truth selection.
Only these factories instantiate concrete adapters. Bootstrap wires interfaces
and keeps the existing construction order, callback ownership and cleanup.
An existing SDK robot is reused at the same prim; a changed robot prim requires
restarting Isaac rather than silently reusing the previous robot.

A future robot implements `ManipulationBackend` and supplies its gripper spec
through the explicit factory, with its selector allowed by configuration. A
future source implements `observe(context) -> ObservationBatch`, `forget(id)`
and `close()`, and adds its explicit construction and required-reference branch.
Its observation logic can remain Isaac-free; simulator wiring stays in the
integration. Neither addition requires changing skills, WorldModel or GUI.
There is no new capability hierarchy, plugin registry or controller layer.

`IsaacRuntimeBundle` no longer exposes `objects` or `state_provider`.
Native tests and the existing loss diagnostic acquire their playground oracle
handle through `tests/support/isaac_helpers.py`. The helper checks that the cube
prim already exists and does not register an object in the production World.
All eight Isaac scenarios and their acceptance assertions are retained.

The bundle's optional `capture_frame(path)` callback isolates RGB-D debugging.
GUI refresh reads the presentation model, backend readiness and source health;
it never queries a concrete source or mutates cached state. Bundle `close()` is
the GUI cleanup boundary. The layout, dialogs and actions remain in place.
Clear Lost now uses the common updater for either source, retaining held/hint
protection. Clear Held uses `runtime.clear_held_state(expected_id)`, retaining
confirmation and rejecting active commands or a changed held ID. The reset is
software-only and issues no robot commands. This closes the existing gap where
a command started outside the GUI could overlap a GUI reset.

Public Python callers must pass the skills' WorldModel when constructing
`RobotRuntime(registry, world_model)`. Commands and backend interfaces are
otherwise unchanged. Test-local benchmark trace functions now receive the
oracle cube explicitly instead of reading it from the production bundle.

## Changes and complexity

| Area | Files and change |
| --- | --- |
| Setup | `config/scenes/playground.toml`, `integrations/isaac/config.py`: smaller explicit schema; remove `DiscoveryConfig`, `TrackingConfig`, `PrimitiveConfig`, `GraspConfig` |
| Assembly | `integrations/isaac/bootstrap.py`: remove oracle wrapper construction and inline adapter creation |
| Factories added | `integrations/isaac/manipulation/factory.py`, `integrations/isaac/world/factory.py` |
| Perception assembly | `world/perception/factory.py`: use scene geometry and existing defaults; remove `build_tracker` forwarding helper |
| Runtime/UI | `runtime/runtime.py`, GUI `extension.py` and `panel.py`: explicit reset/capture/close interfaces; remove reflective close fallback and source-specific button gating |
| Native fixtures | `tests/support/isaac_helpers.py`, both `tests/isaac` scenarios, all six regressions, `scripts/diagnose_held_state.py`: test-owned oracle access |
| Offline coverage | New `test_setup_config.py`; update bootstrap, UI, runtime, diagnostic and affected constructor fixtures |
| Documentation | README, architecture, GUI, perception, manipulation, validation, Phase 4A status and this report |

No files were renamed or deleted. Concrete construction was extracted into two
small factories; configuration forwarding and obsolete helpers were removed.
Line counts below include comments/blank lines and compare against `1d9141f`.

| Measure | Before | After | Change |
| --- | ---: | ---: | ---: |
| Playground TOML | 46 | 22 | −24 (52%) |
| Config + bootstrap + perception factory + both new adapter factories | 507 | 372 | −135 (27%) |
| All production Python (`robot_skill_stack/` + `exts/`) | 5456 | 5324 | −132 (2.4%) |
| Production Python files | 74 | 76 | +2 small explicit factories |
| Discovered unit tests | 266 | 286 | +20 boundary/regression checks |

Tests and documentation grow, so the repository's total line count increases.
The maintenance benefit is fewer configuration knobs and source-specific
assembly/GUI paths, rather than a large reduction across untouched algorithms.
No accuracy or robustness reduction is intended. Legacy custom tuning now needs
an explicit Python change; there is deliberately no generic override mechanism.

Control/feedback, skills, gripper verification, automatic held-state policy,
grasp/placement mathematics, fitting/tracking, WorldModel/updater lifetime,
Behavior Trees, SDK camera/lifecycle, launchers and dependency pins are byte-for-
byte unchanged from main. Only perception construction changed in that package.

## Cloud validation

Baseline: 266 discovered tests, eight standalone offline modules, architecture
boundaries and syntax checks passed before edits. Each major implementation
step received focused checks. Final command:

```bash
.venv/bin/python scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/phase4b-final
```

Result: **PASS**, 286 discovered tests, all eight standalone modules, architecture
and syntax. Local artifacts: `outputs/phase4b-final/run-wsp5ua8i/summary.json`
and neighboring logs (ignored, not committed).

Coverage includes effective values, invalid config, explicit adapter selection,
RGB-D without oracle references, ground truth without a camera, missing prims,
oracle fixture ownership, both-source UI controls/capture, source health,
busy-state resets, cancellation, repeated initialization and failure cleanup.
SDK/UI doubles cannot prove real Isaac initialization, camera timing, physics
or rendering behavior.

## Workstation validation

Run from the normal repository checkout, with the established `.deps` and Isaac
installation unchanged:

```bash
git fetch origin
git switch refactor/phase-4b-config-adapters-ui
git pull --ff-only
python3 scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/phase4b-isaac
python3 scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/phase4b-regression
```

The two integration scenarios and six regression scenarios retain the existing
gates, including benchmark success and 25 mm physical placement accuracy. Review
logs/JSON from every scenario, not just successful startup. The runner still
uses direct scripts through `run_isaac_python.sh`; `.deps` precedence and NumPy
1.26.4 verification before `SimulationApp` are unchanged. py_trees remains
2.6.0. No environment or dependency changes are needed.

Launch `./run_isaac_sim.sh` and run the [manual smoke checklist](validation.md#manual-gui-smoke-test)
with RGB-D, then with a complete ground-truth copy of the same TOML as described
in [gui.md](gui.md). Check initialization, selection, Home/Move/Pick/Place, both
BT tasks, source health on pause/resume, held/uncertain controls, Clear Lost,
frame capture where available, repeated initialization and shutdown. Restart
Isaac before changing setups. Confirm an RGB-D setup with its optional
ground-truth section removed still initializes and manipulates normally.

Re-running the existing injected-loss diagnostic is recommended because its
oracle fixture handle moved out of bootstrap:

```bash
mkdir -p outputs/phase4b-held-state
./run_isaac_python.sh scripts/diagnose_held_state.py \
  --result outputs/phase4b-held-state/loss.json \
  --artifacts-dir outputs/phase4b-held-state/loss \
  > outputs/phase4b-held-state/loss.log 2>&1
```

No Phase 4B native tests ran in the cloud. Remaining risks are SDK wrapper reuse
without World registration for test oracle handles, camera/source construction
in live Kit, local profile migration and real GUI lifecycle behavior. Stop if
the prior shaking/contortion/contact behavior returns; offline checks do not
validate robot motion. Do not merge until the workstation checks pass.

## Deferred

Home remains the TCP pose captured at backend construction. Place keeps its
current `retreat_ok` reporting. These semantics are reserved for Phase 4C or a
separately reviewed correctness fix; neither was changed here. No additional
robot/source, force sensing, live configuration switching or GUI redesign was
implemented. Phase 4C has not started.
