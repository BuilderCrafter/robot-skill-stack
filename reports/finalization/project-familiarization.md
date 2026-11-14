# Robot Skill Stack: reconnaissance and finalization handover

> Historical snapshot from 13 November 2026, published on V5 during the
> [14 November realignment](v5-realignment.md). Branch,
> worktree and environment statements below describe the reconnaissance task;
> consult the realignment report for the current publication decision.

Investigated 13 November 2026 (Europe/Budapest). Checkout:
`/home/ubuntu/projects/robot-skill-stack`; container mount:
`/workspace/robot-skill-stack`. This report follows the later **reconnaissance-only**
request: no implementation, native simulation, commit, push, infrastructure
change, or interruption of the stream was performed. The earlier finalization
request was superseded in those respects.

## 1. Executive assessment

The project is close to a defensible thesis prototype. It already combines
four manipulation skills, persistent semantic observations, independent
occupancy mapping, bounded observation workers, Behavior Trees, a GUI, and
direct/reactive/global motion behind established interfaces. Its strongest
planning result is **static ground-truth obstacle-induced RRT detouring**;
its strongest perception result is controlled multi-view cube/sphere recognition.
These are complementary demonstrations, not evidence of autonomous collision-safe
RGB-D manipulation. See sections 3–7 for source and measured evidence.

The main completion risk is reproducibility and presentation, rather than a
missing planning architecture. This VM has the published V5 code but lacks the
historical raw evidence needed by strict V4.2/V4.3 replay preflights. The running
stream is a standard embedded Kit experience rather than the project Python
launcher. README/configuration summaries lag the implemented global strategy.
No fresh native robot acceptance was conducted here. These gaps prevent calling
the **current cloud checkout** a freshly validated final release.

Frozen RGB-D global planning is implemented, source-isolated and deliberately
opt-in, but is **not a reliable obstacle avoidance demonstration**. Primary
RGB-D planning returns 3/7 routes; two are physically vetoed and one completes
without establishing an obstacle-induced detour. Increasing the proxy budget
did not repair the unsafe routes. Freeze it as experimental for this release
unless a single clearly justified representation fix becomes available; there
is no demonstrated small change that resolves all its failure mechanisms.
Evidence: [V5 primary summary](../../docs/assets/motion_planning_v5/primary64/summary.json),
[native execution receipt](../../docs/assets/motion_planning_v5/native_execution.json),
[128 diagnostic](../../docs/assets/motion_planning_v5/diagnostic128/summary.json).

## 2. Verified repository state and history

### Checkout and local material

| Item | Verified finding |
| --- | --- |
| Branch | `finish/cloud-final` |
| HEAD | `e11a1fd2635d6b06a0de46feab07fb7b20cd4550` |
| Remote | `https://github.com/BuilderCrafter/robot-skill-stack.git` |
| V5 relationship | HEAD equals local `feature/global-rgbd-planning-v5` and cached `origin/feature/global-rgbd-planning-v5`; no new cloud implementation commits |
| V4.3 | Ancestor `aef5f2fe0af0229a1c69c64035b7de0b07173b0b`, cached `origin/research/global-rrt-detour-v4-3` |
| Main | Cached `origin/main` is `34baab2bb2028274c525d8ca9f34164fb65a9c77`; HEAD has 49 subsequent commits; no local `main` branch was listed |
| Initial worktree | Not clean: pre-existing **staged addition `AGENTS.md`**, 158 lines. No other nonignored changes/untracked files initially reported |
| This task's write | This single report, left uncommitted; existing staged work is preserved |
| Ignored local material | `.deps/`, bytecode and `outputs/`; current outputs held startup/stream logs and the interrupted-turn offline run, not the historical native datasets |

These relationships come from `git branch -avv`, `git rev-parse`,
`git log --all`, `git status --porcelain=v1 --untracked-files=all`, and
`git diff origin/feature/global-rgbd-planning-v5`. No fetch was needed; remote
branch statements describe **local remote-tracking refs**, not a live GitHub
query. Nothing in the inspected worktree indicates additional unpublished
functional work. Historical preservation receipts mention untracked `models/`;
it is absent in this checkout and is not required by the supported runtime.

[AGENTS.md](../../AGENTS.md) requires portability to Isaac 5.1.0-rc.19/Python
3.11, NumPy 1.26.4 and py_trees 2.6.0; unchanged launcher behavior; no
`--no-ros-env`; preserved command/lifecycle/safety contracts; 64 ordinary
proxies; bounded experiments; no GT leakage, UNKNOWN completion or failed-source
fallback. Its ordinary workflow recommends validated commits, but the user's
specific reconnaissance instruction explicitly prohibits committing now.

### Development chronology reconstructed from Git

| Stage / reference | Completed work and reason it matters |
| --- | --- |
| Phase 1 `0f735c5` | Established actual architecture/baseline before pruning; archived experimental code remains recoverable through `archive/perception-v2-pre-phase-2` |
| Phase 2 `22114ac` | Removed learned V2/legacy duplicate paths; restored direct-script native startup after `runpy` dispatch problems; did not add a new runtime |
| Phase 3A `6f696d0`, controller correction `b574cf8` | One cooperative async command path, overlap rejection, BT cancellation draining, measured frame-velocity feedback and zero-velocity final arm targets; raw contact-sensitive velocity feedback had caused unstable control |
| Phase 3B `30e7420` | Centralized held/uncertain/loss handling and read physical finger DOF limits through supported articulation properties |
| Phase 4A `1d9141f` / `f87b008` | Updater owns health, aging and forgetting; cached WorldModel and association hints have distinct roles |
| Phase 4B `1f53ed8` / `a8fe32c` | Explicit robot/source factories, complete scene TOML, shared GUI and source contracts |
| Phase 4C `8b3973d`, subsequently merged | Removed unused skill metadata, clarified captured-TCP Home, preserved release state while reporting retreat failures honestly |
| Multi-view through `82d952c`, variable cameras `00dcd14`, main `34baab2` | Separate geometry source, conservative segmentation/observability, editable selected two-camera scene; controlled perception success, manipulation readiness still limited |
| Mapping `45bfe39` → `e8f6d64` | Independent occupancy, exposure-matched robot filtering, contradiction-aware FREE evidence, then process workers; combined thread prototype failed existing responsiveness/freshness gates |
| Direct strategy `e1aa4b7` / `04190a6` | Extracted working Cartesian RMPflow loop without changing public skill semantics |
| Reactive `cc03869` → V2b `8e56b16` | Worker-prepared occupied-cell cuboids, bounded numeric RMPflow registration; stronger sparse RGB-D evidence retention; local minima and partial-visibility limits retained |
| Representation V3 `4f64b35` | Exact packing/selection experiments: fragmented observations often already meet a packing lower bound; more packing searches cannot invent missing surface/volume evidence |
| Global V4 `5fd3ca2` → `8725b48` | Static Lula RRT + sampled route/spline checks + physical-time joint executor; Home tracking remained a real blocker; ineffective Home-branch preference was rejected (`408d538`) |
| V4.1 `4a76487` → `5ce1b70` | Scoped TGS position iterations plus terminal velocity dwell fixed measured Home execution; no gain/effort/tolerance relaxation; five-second hold evidence remains bounded |
| V4.2 `c3b4db2`, `8a710b7`, `83a1197` | Project-owned 102-sphere description and guarded finger envelope; 13 measured round trips, but identical two-waypoint global routes did not prove obstacle-induced RRT detours |
| V4.3 `2fa12c4` → `aef5f2f` | Independent frozen obstruction, fixed validated IK candidate, five matched free/obstacle comparisons; closed that route-selection gap |
| V5 `cf865c1`, `decb953`, `67c4bdb`, `e11a1fd` | Frozen RGB-D source boundary, paired native evaluation and physical execution veto, genuine-occlusion secondary, candid final limitations |

Source history is corroborated by [history](../../docs/history.md), historical
`git show 1f53ed8:docs/phase_2_pruning.md`, `:docs/phase_3a_execution.md`,
`:docs/phase_4a_world_model.md`, and current motion/mapping reports. Earlier
phase/native counts describe their checkpoints, not a new test of HEAD.

## 3. Architecture and ownership

### Commands, skills and physics

```text
GUI callback / BT SkillNode / standalone Kit coroutine bridge
  -> RobotRuntime.execute_async(name, **arguments)
  -> registered skill.execute_async
  -> HeldState + grasp/placement policy + ManipulationBackend
  -> IsaacFrankaBackend arm-ownership guard
  -> selected MotionStrategy
     direct: Cartesian RMPflow action loop
     reactive: fresh numeric occupancy cuboids + RMPflow loop
     global: one world -> IK/RRT/validation -> timed joint action loop
  -> Franka.apply_action -> PhysX -> measured joint/TCP feedback
  -> final position target with zero requested velocity -> cleanup
```

`RobotRuntime` reserves `_busy` before its first await, returns `BUSY` for
overlap, converts ordinary exceptions to `INTERNAL_ERROR`, and releases in
`finally`; cancellation propagates. Backend arm ownership also protects calls
that bypass the runtime. Skills do not call USD or cameras. `HeldState` coordinates
software attachment and grasp evidence independently of the concrete backend.
References: [runtime.py](../../robot_skill_stack/runtime/runtime.py),
[backend contract](../../robot_skill_stack/manipulation/backend.py),
[held_state.py](../../robot_skill_stack/manipulation/held_state.py),
[franka_backend.py](../../robot_skill_stack/integrations/isaac/manipulation/franka_backend.py).

`SkillNode` submits the same runtime coroutine. `BTExecutor.run_async` yields
between ticks and cancels/awaits pending children on exit. Recovery's remembered
selector permits Home/retry only for eligible pre-closure Pick failures;
gripper engagement, loss, uncertainty, source failure and busy/internal failures
block that recovery. Place failure stops the sequence before final Home.
References: [BT executor](../../robot_skill_stack/orchestration/behavior_trees/executor.py),
[recovery task](../../robot_skill_stack/orchestration/behavior_trees/tasks/pick_and_place_recovery.py).

Generic `common`, `runtime`, `world`, `manipulation`, `presentation` and
`orchestration` code is separated from Isaac imports by protocols including
`ManipulationBackend`, `MotionStrategy`, `WorldObservationProvider`,
`ObservationBatch` and `MappingSourceProvider`. Bootstrap/factories explicitly
select Franka and the source; no plugin framework or second executor is needed.
See [bootstrap.py:build_runtime](../../robot_skill_stack/integrations/isaac/bootstrap.py),
[motion/strategy.py](../../robot_skill_stack/manipulation/motion/strategy.py),
[provider.py](../../robot_skill_stack/world/model/provider.py), and
[architecture check](../../scripts/check_architecture.py).

### Observations and WorldModel

```text
Kit: calibrated camera buffers + original tokens/time + robot pose snapshots
  -> owned immutable input -> bounded worker transport
  -> SDK-free geometry discovery / primitive fitting / multi-view fusion
  -> Kit: reject old generation/context/time -> tracker association
  -> WorldModelUpdater.publish(ObservationBatch)
  -> cached object IDs/poses/visibility/health/attachment/hints
  -> GUI, skills and BTs

Independent branch from shared camera exposures:
  -> worker: exact robot-ray filtering -> occupancy hit/miss integration
  -> immutable map + optional bounded ObstaclePacket
  -> MappingService publication / fixed-time MappingReadView
  -> reactive updates OR one frozen RGB-D global capture
```

WorldModel is semantic object state, **not the voxel map**. Neither semantic
classes nor tracked object dimensions determine RGB-D occupancy. A placement
hint is an association prior, not a fabricated measured pose. Updater health
ages with monotonic time, including while physics is paused; old/repeated/future
samples cannot refresh it. Default semantic freshness is 1 s and object TTL
15 s. Held/hinted IDs are protected from expiration and Clear Lost.
References: [updater.py](../../robot_skill_stack/world/model/updater.py),
[world_model.py](../../robot_skill_stack/world/model/world_model.py),
[perception documentation](../../docs/perception.md).

The Kit thread owns SDK reads, commands, tracker mutation and publication.
Heavy semantic and mapping computations run in separate SDK-free Python child
processors, with parent I/O threads. Each bounded worker has one running job
and one replaceable pending job. Generation/failure/context barriers reject
late work; timestamps retain acquisition time. Camera sharing avoids duplicate
render products; mapping-only camera ownership is explicit. These are distinct
from synchronous global Lula planning, which **does block its owner loop**.
References: [observation_runtime.py](../../robot_skill_stack/integrations/isaac/observation_runtime.py),
[process_worker.py](../../robot_skill_stack/runtime/process_worker.py),
[mapping runtime](../../robot_skill_stack/world/spatial_mapping/runtime.py),
[shared_frames.py](../../robot_skill_stack/world/perception_multiview/shared_frames.py).

## 4. Capabilities matrix

“Historical native” below means inspected tracked result receipts, not a fresh
native run on this VM. Current offline status is recorded separately in §8.

| Capability | Status and evidence | Boundary / limitation |
| --- | --- | --- |
| Direct MoveToPose and Home | Implemented; V5 native preservation `direct` case | RMPflow has no generic external obstacle-world registration here; endpoint IK is not path safety |
| Pick and stable Place | Implemented; direct native case checks lift, held identity, release and same-ID reacquisition | Conservative simple primitives, known-object placement occupancy; no arbitrary payload collision planner |
| Intentional release API | Implemented, offline failure/outcome tests | GUI uses stable mode only; release skips landing verification, not motion/open/retreat failure handling |
| Held/lost/uncertain state | Implemented; `test_held_state`, `test_place_modes`, `test_execution_runtime` and native direct lifecycle receipt | Finger position evidence cannot prove identity or detect every slip; software Clear Held does not physically release |
| Pick/Place BTs and recovery | Implemented; shared runtime and unit/native preservation | Recovery constrained before closure; no automatic retry after Place failure |
| V1 RGB-D | Supported default, primitive tests and legacy native receipts | Separated boxes/cubes, spheres, upright/lying cylinders; touching/stacking/arbitrary 6D objects not generally supported |
| GT semantic provider | Implemented, same ObservationBatch/WorldModel interface | Oracle mode must be explicitly identified; semantic GT is separate from static planning-world capture |
| Multi-view perception | Implemented, opt-in; variable camera-count tests; final controlled cube/sphere checkpoint | Both/all configured cameras must remain healthy; one-view object evidence is allowed, camera-failure fallback is not; autonomous clutter manipulation not established |
| Occupancy mapping | Implemented GT/RGB-D, immutable publications, source health/expiry, automatic robot filtering | OCCUPIED/FREE/UNKNOWN/OOB remain distinct; observed surfaces are not complete solids |
| Reactive RMPflow | Implemented opt-in, ordinary move/Home/lifecycle native preservation | Partial success in obstruction studies; local minima/timeouts/contact cases remain; Pick/Place/carrying rejected |
| Static GT global RRT | Implemented opt-in; strongest measured result is V4.3 | Static supported colliders only; empty gripper, sampled collision checks, no dynamic repair |
| Frozen RGB-D global RRT | Implemented, experimental; one primary completed motion | No reliable obstacle-induced detour; unknown space is traversable by incomplete collision representation, never certified free |
| GUI | Implemented profile selection, initialization, cached object table, target XYZ, Move/Home/Pick/Place, both BTs, health, Clear Lost/Held and V1 capture | No dedicated global-detour preset/route display/full orientation entry; current stream integration unverified |
| Additional robot / ROS runtime / learned V2 | Unsupported current runtime | Historical removed code is not an available production feature |
| Hybrid, LLM deliberation, arbitrary-object planning | Deferred | No implementation needed for thesis completion |

Primary references: [manipulation](../../docs/manipulation.md),
[V5 regressions](../../docs/assets/motion_planning_v5/regressions.json),
[multi-view final checkpoint](../../docs/multiview_perception_final.md),
[GUI panel](../../exts/robot_skill_stack.ui/robot_skill_stack_ui/ui/panel.py),
[GUI controller](../../exts/robot_skill_stack.ui/robot_skill_stack_ui/ui/extension.py).
The controlled final perception report records 212/212 scored correct selected-layout
observations (66/66 static, 80/80 temporal plus other scored observations),
approximately 4.13 Hz mean complete native loop, and explicitly preserves
multi-view manipulation NO-GO. That accuracy is not a current HEAD live-cloud
measurement or a general occlusion/robotics guarantee.

## 5. Motion planning assessment

### Selection, direct and reactive

`MotionConfig` actually accepts `direct`, `reactive`, and `global`; default is
`direct`. `hybrid` explicitly fails. Selection occurs once during backend
construction; Home delegates to the selected strategy. `max_obstacles` defaults
to 64; validation permits 1–128, but 128 remains a diagnostic condition.
`rrt_seed=123456`, `rrt_max_iterations=10000`, `global_collision_model='stock'`,
`global_obstacle_source='static'` are defaults. RGB-D global requires enhanced,
enabled RGB-D mapping and automatic robot filtering. References:
[config.py:MotionConfig/SceneConfig](../../robot_skill_stack/integrations/isaac/config.py),
[backend factory](../../robot_skill_stack/integrations/isaac/manipulation/factory.py).

Direct retains its controller reset, reachability, cooperative action loop,
grasp guard and final hold. It never queries mapping.
Reactive reads only prepared cuboids, validates health/time/product identity,
updates numeric RMPflow obstacles and clears its registrations on exit.
Neither creates physical collision proxies in USD. No source failure becomes
direct motion. References: [direct.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/direct.py),
[reactive.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/reactive.py).

### Global request from world capture to cleanup

1. `GlobalRrtStrategy.move_to_pose_async` checks lifecycle/readiness, full finite
   world pose/tolerances, empty/free gripper and the selected collision contract.
2. Capture **one** static USD world or frozen RGB-D world. Static capture excludes
   the robot and reads enabled colliders including invisible ones; rejects
   nonkinematic dynamics, animated/unresolved/unsupported geometry. Supported
   primitives include cuboids, spheres, capsules and a finite-domain plane slab;
   convexHull meshes must qualify as exact boxes. Ordinary dynamic tabletop
   manipulation objects are not automatically admissible static obstacles.
3. `LulaGlobalPlanner` uses installed Franka URDF/RRT resources, seven named
   active arm joints, seed forwarded into native C-space RRT, 0.05 rad search
   step and bounded iterations. Up to four warm-start IK candidates are checked
   for limits, full-pose FK, duplicate solution, external/self/domain validity.
4. RRT produces an actual joint route. Endpoints must equal validated measured
   start/goal; route interpolation checks ≤0.01 rad maximum joint spacing.
5. Lula C-space trajectory generation preserves endpoints. Spline samples are
   spaced at most `min(1/240 s, .01/max_velocity)` with ≤0.01001 rad step,
   bounded to 100,000 samples/120 s; joint, velocity, acceleration, jerk and
   collision/domain constraints are checked. This is sampled model validation.
6. Before commands, reject changed start joints, simulation instant/order/model
   or lifecycle. Scope the TGS iteration lease only for a validated moving plan.
7. Execute trajectory positions/velocities at actual elapsed physics time,
   using named arm indices and `apply_action`, never joint teleportation or
   Cartesian waypoint substitution. Monitor base/lifecycle/physics progression,
   contract validity, tracking and watchdogs.
8. Preserve 0.15 rad sustained tracking bound with 0.25 s grace, 0.40 rad
   immediate abort, final joint gap ≤0.05 rad, requested pose tolerances,
   measured arm speed ≤0.01 rad/s for 0.25 simulated seconds within original
   three-second settling budget. Standard skill pose tolerances are 25 mm/0.10 rad;
   backend defaults are 10 mm/0.05 rad. No tolerance was changed here.
9. Hold final positions with zero requested velocity if still live; remove
   registrations, close planner, restore exact prior solver session opinion;
   no stale command after Stop/reset/close. Failed cleanup remains explicit.

Source: [global_rrt.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/global_rrt.py),
[lula_global.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/lula_global.py),
[global_solver.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/global_solver.py),
[static_world.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/static_world.py).
TGS requests 255 position iterations without changing velocity iterations,
physical gains, effort limits or saved assets. Effective native island counts
are not independently measured by the USD getter. Planning is synchronous on
Kit; cooperative execution does not imply nonblocking search.

### Robot collision models

Stock has 55 arm/hand spheres and nominal fixed 25 mm fingers, with measured
physical coverage gaps. Enhanced adds 47 spheres (102 total), including swept
finger envelopes, binds descriptor and installed URDF hashes, and separately
preflights the mounted-base enclosure. Each measured finger and every exposed
native finger target must stay within **[1,5] mm**, nominal 3 mm. Ordinary
open-gripper manipulation is not a supported enhanced-global configuration.
An explicit physically certified finger-pair exclusion supplements structural
URDF adjacency exclusions; it does not remove finger/forearm checking.
References: [collision_model.py](../../robot_skill_stack/integrations/isaac/manipulation/motion/collision_model.py),
[model certificate](../../config/motion/franka_enhanced_v4_2.json),
[V4.2 geometry report](../../docs/motion_planning_v4_2_collision_model.md).

Current local descriptor SHA256 is
`6801f0c053dba1944157385ae484feb661b3edc14197bf7ba5cf01e568e04a5d`.
Measured 100% sampled moving-surface coverage and source-bound solid-cell
containment are stronger than stock, but do not certify arbitrary execution.
The approximately 24 nm mounted-base/support separation is not a useful hazard
clearance margin. Finite floor cuboids can conservatively overlap the enclosing
base sphere even without proven physical collision.

### Exact RGB-D isolation boundary and limitations

`world/spatial_mapping/frozen.py:capture_frozen_rgbd` validates enabled source
type, automatic filtering, accepted camera identities/calibration, clock,
freshness, generation/revision/timestamp match, extraction policy/budget,
unexpired selected evidence, exact disjoint grid cuboids and cell accounting.
It copies bounded descriptors, not dense cells or new images. Sparse queries
retain capture time, uncertainty and original acquisition timestamps.

`integrations/isaac/manipulation/motion/frozen_rgbd.py` translates these exact
numeric cuboids. Its only external geometry access is the explicit static Plane
allowlist `/World/GroundPlane/CollisionPlane`; it never traverses environmental
fixtures or imports semantic objects. Support is separately labeled, not used
to erase sensed floor evidence. Global strategy never calls static capture on
RGB-D source failure. Later maps cannot refresh or mutate the active world.
Source checks: [frozen.py](../../robot_skill_stack/world/spatial_mapping/frozen.py),
[Isaac adapter](../../robot_skill_stack/integrations/isaac/manipulation/motion/frozen_rgbd.py),
[source tests inventory](../../docs/assets/motion_planning_v5/source_validation.json).

The worker integrates calibrated depth rays (FREE before valid hits, OCCUPIED
endpoints, no inference behind surfaces), applies exposure-matched exact robot
ray masks, retains supported OCCUPIED components and strong sparse RGB-D evidence
(probability ≥0.95), then packs exact cells with no gap filling/padding. Floor
priority affects selection, not truth labels. V5 uses two authored 640×480
cameras, 5 Hz semantics, 2 Hz mapping, stride 8, shared 1,024-ray budget,
10 mm voxels, 5 s evidence TTL and 1 s freshness. References:
[obstacles.py](../../robot_skill_stack/world/spatial_mapping/obstacles.py),
[V5 protocol](../../scripts/global_rgbd_v5_protocol.py),
[mapping documentation](../../docs/spatial_mapping.md).

UNKNOWN is **not registered as occupied**, so a plan can traverse unobserved
space. The retained query API does not relabel it FREE, but that distinction
does not itself prevent unsafe trajectories. The independent GT physical veto
in `scripts/global_rgbd_v5_pair.py:MatchedSourcePlanner` is **research tooling**,
not an automatic production RGB-D safety interlock. The paired harness uses a
common validated joint goal for comparison; fixture solids are used for GT
control/evaluation, not RGB-D obstacle construction. Do not enable ordinary
RGB-D execution on the assumption that its production route checks include the
research cooked-geometry veto.

## 6. Inspected physical results

### V4.3: strongest obstacle-induced route-selection evidence

The inspected [detour_results.json](../../docs/assets/motion_planning_v4_3/detour_results.json)
contains 13 comparison/control cases, five matched pairs, accepted-pair counts,
route coordinates, operation outcomes, physical acceptance and hash references.
[validation.json](../../docs/assets/motion_planning_v4_3/validation.json) records
16 native processes including preservation checks. All five declared obstacle
round trips have strict zero reports, genuine outward and Home motion and
meaningful deviation from the corresponding unobstructed joint chord.

| Seed / repetitions | Free → obstacle waypoints | Maximum chord deviation | Minimum outward / Home box clearance |
| --- | --- | --- | --- |
| 123456 × 3 | 2 → 3 | 0.080928 rad | 15.922 / 17.494 mm |
| 234567 × 1 | 2 → 3 | 0.218259 rad | 16.289 / 15.403 mm |
| 345678 × 1 | 2 → 3 | 0.600000 rad | 18.265 / 25.964 mm |

The unchanged meaningful-detour threshold is >0.05 rad with at least three
waypoints, across all seven joints. Box center `(0.55,-0.225,0.30)` m and
dimensions `(0.06,0.06,0.12)` m are independently frozen; full target is
`(0.45,-0.25,0.35)` m, quaternion wxyz `[0,0,1,0]`. Local fixture hash matches
`d5c78c58343f40f559ff174ded89814493ac147d62aad1e00327657efd08680f`;
fixed-target hash matches
`c38daa00d06958076cba111dc45ba0af8aa80c4890beadc08e2dc30d52ef83da`.

Independent positive-volume intersection witnesses invalidate the original
route with the box present (planned finger ball radius 3.091 mm, actual
corroboration 3.742 mm). An independent continuous numeric witness establishes
feasibility without supplying waypoints to RRT. It is **not** continuous
certification of the returned spline or actual execution. Maximum final TCP
error across B runs is about 1.832 mm; primary outward/Home errors are
1.060/1.831 mm. Off-route and blocked-goal controls retain their separate
outcomes; blocked goal issues zero arm commands. References:
[frozen fixture](../../docs/assets/motion_planning_v4_3/frozen_fixture.json),
[model receipt](../../docs/assets/motion_planning_v4_3/reference_model_check.json),
[V4.3 report](../../docs/motion_planning_v4_3_detour_verification.md).

### V5 preservation of GT

[regressions.json](../../docs/assets/motion_planning_v5/regressions.json) records
ten native fixtures: direct, reactive, global roundtrip/large/cancel/Stop/reset/
close, detour free and detour obstacle. Their fixture contracts pass. The V4.3
free route, validated trajectory and command stream exactly match the original
obstructed reference; obstacle replay returns three waypoints, 0.080928 rad
deviation and strict outward/Home completion. Global-large reaches the goal
with four positive-separation reports and remains `strict_success=false`.
Interruption operations intentionally do not return ordinary motion success;
the lifecycle fixture's acceptance checks are the correct gate.

The V5-to-V4.3 source diff has eight intended original-file modifications and
two new production boundary files; direct/reactive implementations,
`lula_global.py`, `global_solver.py`, launchers and collision resources are
unchanged. See [preservation.json](../../docs/assets/motion_planning_v5/preservation.json)
(684/692 baseline files unchanged, eight intended changes) and
`git diff aef5f2f..HEAD`. This is preservation evidence, not a new five-seed
cloud replay.

### V5 primary64 native execution, inspected from scalar receipts

| Scene | GT route points / completion / reports | RGB-D planning → execution | GT final position error / moving-arm clearance |
| --- | --- | --- | --- |
| empty | 2 / yes / 0 | base rejection → no commands | 0.990 / 140.965 mm |
| static | 3 / yes / 0 | base rejection → no commands | 0.942 / 28.213 mm |
| sparse | 3 / yes / 15 | base rejection → no commands | 0.970 / 13.566 mm |
| partial | 3 / yes / 0 | 3-point plan → physical veto, no commands | 0.942 / 28.213 mm |
| narrow | 2 / yes / 44 | 2-point plan → completes, 44 reports | 0.990 / 14.436 mm |
| large | 3 / yes / 4 | 5-point plan → physical veto, no commands | 0.845 / 23.148 mm |
| v43 | 3 / yes / 0 | base rejection → no commands | 1.060 / 15.922 mm |

Source: [primary64 summary](../../docs/assets/motion_planning_v5/primary64/summary.json)
and [native_execution.json rows](../../docs/assets/motion_planning_v5/native_execution.json).
All seven GT motions physically complete within original full-pose limits;
all have zero observed physical-contact steps. Only **4/7** meet strict
zero-report acceptance (empty/static/partial/v43). Positive-separation reports
in sparse/narrow/large are not hidden or counted as strict success.

Among **seven selected RGB-D requests**, four fail base preflight, three return
plans, two returned plans are physically vetoed, and one executes. The completed
narrow move has 86 trajectory updates/87 captured actions, 1.107 rad maximum
actual displacement, 0.990 mm/0.002204 rad endpoint errors, 14.436 mm sampled
moving-arm clearance, zero observed physical-contact steps/normal impulse and
44 positive-separation proximity reports. It remains strict-false and follows
the empty-GT route. Measurement PASS does not turn these into seven successful
avoidance runs.

The 14 primary processes each compare both sources before selecting one for
execution. Do not count counterplans as additional independently executed trials.
The original primary+128 cohorts have 20 processes and 11 moving trials (seven
primary GT, one primary RGB-D, three diagnostic GT), all closed/ownership released
with source captured once and zero active-world refreshes. Source:
`native_execution.json:summary`.

At 128, partial and large still return unsafe RGB-D routes; narrow adds floor
boxes and fails base preflight. GT completes all three diagnostic moves. The
separately declared verified-occlusion pair adds one strict GT completion and
another RGB-D route vetoed before any arm action. Thus across the **12 selected
RGB-D requests** in these three cohorts: six planner rejections, six accepted
plans, five subsequent physical vetoes and one executed motion. The secondary
is not retroactively part of the original seven-scene denominator.
References: [diagnostic128](../../docs/assets/motion_planning_v5/diagnostic128/summary.json),
[verified occlusion](../../docs/assets/motion_planning_v5/verified_occlusion/summary.json).

### Failure attribution and viable RGB-D scope

| Mechanism | Evidence / interpretation | Finalization decision |
| --- | --- | --- |
| Floor finite boxes conflict with conservative fixed-base sphere | Four primary preflight rejections; `collision_model.py:preflight_base` explicitly says conservative rejection, physical collision not proven | Potential model/representation conservatism, not evidence that safety should be disabled. Any replacement needs an equally conservative geometric proof |
| Partial/unobserved solid volume | RGB-D plans collide in independent cooked geometry even after 128 retains more observed cells | Observation limit; not repairable by treating UNKNOWN as FREE or adding hidden GT solids |
| Proxy-budget omissions | Only 30.7% observed Wall and 36.4% original partial Occluder evidence selected at 64 in the V5 report | Representation limitation; 128 remains diagnostic, not a production fix |
| Fragmentation | Hundreds of small candidates; V3 exact-packing lower-bound analysis rules out simple compression gains on those measured masks | Do not start another packing/tuning campaign |
| Original “partial” scene lacked intended actual Post occlusion | Explicitly corrected by a separate verified-occlusion condition, which also yields unsafe RGB-D plan | Protocol limitation preserved, not a hidden successful case |
| Planning/evaluation latency | Narrow request-to-first-command 642.57 ms, worst Kit gap 669.80 ms includes paired plans and blocking research veto waits | Do not present this as production-only planner latency or asynchronous search performance |

References: [V5 analysis](../../docs/motion_planning_v5_rgbd_global.md),
[collision witness occupancy](../../docs/assets/motion_planning_v5/collision_witness_occupancy.json),
[V3 representation report](../../docs/obstacle_representation_v3.md),
[performance.json](../../docs/assets/motion_planning_v5/performance.json).
No demonstrated runtime defect presently justifies an RGB-D implementation
change in preference to finishing the GT demonstration.

## 7. Evidence inventory and confidence

Confidence distinguishes review of a tracked receipt from possession of the raw
data or a fresh physical run. All native figures/results below are historical
Isaac simulation, not real-robot evidence.

| Claim | Source/checkpoint and exact artifact | Type/result | Confidence and limitation |
| --- | --- | --- | --- |
| Five meaningful GT detours | `aef5f2f`; `docs/assets/motion_planning_v4_3/detour_results.json` | Matched native + independent geometry; 5/5 strict outward/Home | Strong retained scalar/route evidence; raw NPZ/traces absent here; no new native replay |
| Original route genuinely obstructed | V4.3 `frozen_fixture.json`, `reference_model_check.json` | Positive-volume physical witness and native model rejection of samples 33–44 | Source-bound numeric proof retained; not continuous certification of new executed route |
| Enhanced model coverage | `83a1197`; V4.2 `collision_coverage.json`, `self_and_rejected_ik.json` | 100% sampled moving surface; 30/30 stock misses detected by enhanced | Numeric geometry plus samples, bounded finger contract; base has separate preflight |
| Home execution correction | `5ce1b70`; V4.1 `home_tracking_before_after.json`, `home_hold_five_seconds.json` | Scoped solver/settling; bounded five-second hold | Hold error can reach about 22.17 mm; not indefinite millimeter stationarity |
| V5 preserves direct/reactive/GT/lifecycle | `e11a1fd`; V5 `regressions.json` | 10 native fixture receipts pass original behavior gates | Large retains strict-false; interrupted moves are expected failures; raw receipts absent |
| GT 7/7 primary completion | V5 `native_execution.json` rows + `primary64/summary.json` | Executed full-pose results, contacts and sampled clearance | 4/7 strict zero-report; general static-scene success not proved |
| RGB-D 1/7 completion | Same artifacts | 3 returned plans; 2 vetoed; narrow completes | No meaningful obstacle-induced detour, strict-false |
| More proxies insufficient | V5 `diagnostic128/summary.json` | 2 unsafe plans, 1 base rejection | Separate diagnostic only, no 128 default promotion |
| Genuine occlusion also unsafe | V5 `verified_occlusion/summary.json` | GT strict success; RGB-D veto | Secondary condition, no denominator replacement |
| Multi-view controlled perception | `82d952c`; `docs/assets/multiview_final/checkpoint.json`, `source-freeze.json`, `docs/multiview_perception_final.md` | Frozen image/geometry study, controlled presentation GO | Manipulation NO-GO preserved; raw images absent; provider subsequently reorganized for workers |
| Nonblocking observation integration | `e8f6d64`; `docs/assets/spatial_mapping_v5/evaluation.json`, `runtime_validation.json` | Worker/control timing + manipulation/lifecycle | Historical matched workloads; global search remains synchronous |
| Current offline suite | §8 exact local logs | Fresh offline run only | Does not establish rendering/physics/GUI acceptance |
| Workstation compatibility | Unchanged wrappers/pins/model resources; historical receipts | Static source review and earlier native records | Current HEAD not freshly run at university in this task |
| Earlier Frankfurt GT/V5 experiments | Historical handover / staged AGENTS claims | Not independently recoverable from searched local outputs | **Unverified**; startup log/open stream is not a benchmark result |

The initial filesystem search inspected current `outputs/`, container `/tmp`
and selective result/summary names under `/tmp`, the project and Isaac data/log
roots. It did not locate the earlier Frankfurt native result JSON advertised
in the handover. No claim is made that an external archive cannot exist.

### Reproducibility blockers actually checked

Offline invocation of `scripts.run_global_rgbd_v5_regressions.preflight_detour()`
fails **before SimulationApp** with:

```text
FileNotFoundError: outputs/motion-planning-v4-3/baseline-free/r1/result.json
```

The fixed target also requires `roundtrip_trace.json` and `geometry_reference.npz`.
The frozen fixture references independent-witness/design/control artifacts
under ignored `outputs/`, plus absolute `/home/etfrobotics/Projects/ga253315m/`
paths to `main_design_lock.json` and `control_finalization_protocol.json`.
Those absolute paths are absent here although the corresponding tracked
documents exist at this checkout. `verify_frozen_design` reads referenced paths
literally. V4.2 `load_design_evidence` similarly requires ignored source data.
This is an artifact/path portability defect, **not permission to edit fixture
hashes, fabricate raw evidence, or bypass the checks**.

The strict historical experiment should be restored from hash-matching archives
and, if necessary, a narrowly scoped checkout-relative path resolver that retains
every hash check. If archives are unavailable, retain the historical result and
label a fresh V5 `v43` capture as a new reproduction, not exact archival replay.

## 8. Environment and offline validation

### Read-only environment findings

| Item | Observed |
| --- | --- |
| Host | `brev-ch694ly58`, Ubuntu kernel `6.8.0-1073-gcp`, 47 GiB RAM |
| GPU/driver | NVIDIA L4, 23,034 MiB reported memory, driver 580.178.04 |
| Container | Existing `isaac51`, `nvcr.io/nvidia/isaac-sim:5.1.0`; running throughout investigation |
| Isaac | `/isaac-sim/VERSION`: `5.1.0-rc.19+release.26219.9c81211b.gl` |
| Container Python | 3.11.13; wrapper imports repo `.deps` NumPy 1.26.4 and py_trees 2.6.0 |
| Bundled research libraries | SciPy 1.15.3, PyYAML 6.0.2 |
| Live GPU owner | Container Kit PID 1049, parent `runheadless.sh` PID 1039; only one live Kit observed |
| Stream command | `kit /isaac-sim/apps/isaacsim.exp.full.streaming.kit --no-window ... --merge-config=/isaac-sim/config/open_endpoint.toml --allow-root --/app/livestream/publicEndpointAddress=100.84.229.7 --/app/livestream/port=49100` |
| Stream log | `outputs/isaac-stream-clean.log`; mapped native log `.../logs/Kit/Isaac-Sim Streaming/5.1/kit_20261113_200942.log` |

Commands were `uname -a`, `free -h`, host/container `nvidia-smi`, `docker ps`,
`docker inspect` for existing image/mounts, container `ps`/`pgrep`, selected
process environment, version/import probes and log reads. No routing, driver,
firewall, Docker or VM changes occurred. The supplied history says the WebRTC
stream works through Tailscale and describes NetBird exceptions; the Windows
client's visual/latency state and routing behavior were **not retested**.
The stream log retains an RTC timeout, which alone cannot establish present
client failure or justify network changes.

No `robot_skill_stack.ui` startup entry was found in the active stream's native
log or captured command; the command supplies no project extension directory,
and process environment lacks project `PYTHONPATH`. This supports **project
extension not observed loaded**, rather than an introspected extension-manager
state. The standard embedded Kit executable also conflicts with
`ProcessWorker.__init__`, which requires a Python-named interpreter and tells
users to launch through `run_isaac_sim.sh`. Merely enabling the extension in
this embedded stream is therefore not an established full-runtime solution.

The normal launcher pre-imports NumPy, selects the full experience, supplies
project extension folder/enable flags and ROS startup overrides. Preserve
[run_isaac_python.sh](../../run_isaac_python.sh),
[run_isaac_sim.sh](../../run_isaac_sim.sh) and
[launch_isaac_gui.py](../../scripts/launch_isaac_gui.py).
Cloud streaming integration should be a separate opt-in launch/configuration
path in a later task, with a single Python-launched Kit process. No such command
was tested or claimed operational here.

### Offline runs and actual outcomes

The interrupted earlier turn's container run completed, rather than remaining
active. Its command was:

```bash
docker exec isaac51 bash -lc 'cd /workspace/robot-skill-stack && \
  ISAAC_SIM=/isaac-sim ./run_isaac_python.sh scripts/run_all_regressions.py \
  --unit-only --artifacts-dir outputs/cloud-final-baseline'
```

Its artifact is `outputs/cloud-final-baseline/run-iz0_lv29/summary.json`
(readable inside the container). Architecture failed on bytecode-directory
permissions (`tests/__pycache__`); discovery ran 1,348 tests with four setup
errors because the container has no `git`; all eight standalone modules passed.
Setup errors suppress 25 tests, so 1,348 must not be compared as a smaller
intended suite. No packages or permissions in the container were changed.

A host attempt with `/home/ubuntu/.venv/bin/python` used Python 3.12.16 but had
no NumPy; only architecture passed. Its logs are
`/tmp/robot-final-recon-unit-20261113/run-ltumwjcb/`. Adding `.deps` to that
interpreter is invalid: its NumPy binary is built for CPython 3.11.

An isolated environment `/tmp/robot-recon-py312` was created, leaving repository
and Isaac dependencies intact. After installing only `requirements.txt`, the
runner executed all **1,373** tests, with **14 missing-library errors**
(SciPy and PyYAML), architecture and eight standalone modules PASS. Artifact:
`/tmp/robot-final-recon-pinned-20261113/run-8fdkydi9/summary.json`.
This establishes that README's claim that `requirements.txt` alone supports
the **entire current offline research suite** is incomplete. Runtime dependency
pins need not change; research test prerequisites can be documented separately.

For the complete dependency check, only the temporary environment received
Isaac-matching SciPy 1.15.3 and PyYAML 6.0.2. Exact final command:

```bash
/tmp/robot-recon-py312/bin/python scripts/run_all_regressions.py --unit-only \
  --artifacts-dir /tmp/robot-final-recon-complete-20261113
```

Final result: **PASS: all 1,373 discovered tests (104.623 s), architecture/syntax,
and all eight standalone modules; all ten runner stages pass, overall exit 0.**
Artifact: `/tmp/robot-final-recon-complete-20261113/run-jmj6xu27/summary.json`;
the neighboring `unit_discovery.log` records the count and `OK`.
The environment is Python 3.12.16,
NumPy 1.26.4, py_trees 2.6.0, SciPy 1.15.3, PyYAML 6.0.2. Python 3.12 is an
allowed offline-development environment in README, not a replacement for
native Isaac Python 3.11. Temporary logs are local evidence and are not Git
artifacts; preserve them elsewhere before clearing `/tmp` if needed.

Historical `AGENTS.md` says 1,372/1,373 passed with one multiview hash assertion.
That is a supplied historical observation, **not reproduced** in the complete
fresh run. No assertion or fixture change was needed. No current functional
regression was established by the offline suite; missing dependencies, container
Git, filesystem ownership and archival inputs are separate reproducibility
issues. A fresh native regression remains untested, rather than ruled out.
`test_multiview_observability_research.ObservabilityResearchTests.
test_holdout_cover_revision_preserves_original_captured_fixture_parameters`
contains the relevant frozen fixture digests. Do not change those expectations
without evidence. The older `multiview_final/source-freeze.json` provider hash
also differs from current HEAD, traceable to worker integration `e17a5e7`;
historical freeze manifests describe their own source cohort, not current HEAD
holdout authorization. Native holdout commands requiring that frozen source
must not be advertised as current-tree reruns.

## 9. Final demonstration feasibility and workstation reproduction

### Recommended defense sequence

1. **Manipulation/architecture:** use the ordinary Python GUI launcher and
   `scenes/playground.usd` / `config/scenes/playground.toml`; initialize, observe
   IDs/geometry/health, execute one clear-workspace Pick → held Home → stable
   Place → Home, then a short BT sequence. Explicitly show stable placement and
   source identity. Restart before changing profile/source.
2. **Perception:** separately use `scenes/multiview_perception_final.usda` and
   `config/scenes/multiview_perception_final.toml` to show four controlled
   cube/sphere observations and the editable two-camera geometry. Do not claim
   autonomous occluded-object manipulation from this perception checkpoint.
3. **Planning:** show actual GT global movement around the existing `v43` box,
   supported by the V4.3 free/obstacle joint-route and physical-clearance plots.
   A restored strict V4.3 paired replay is the strongest exact acceptance;
   the existing V5 `v43/static` harness is a smaller fresh-motion candidate
   that does not require the missing V4.3 raw baseline preflight.
4. **Research limits:** show V5 GT/RGB-D comparison figures and explain unsafe
   plans stopped before execution, unknown geometry, proxy omissions and floor
   rejection. Treat that as failure analysis, not a hidden unfinished promise.

The most defensible obstacle scenario is V4.3: its old route is independently
proved obstructed, start/goal are held equal, and the changed route has positive
measured box clearance and zero reports. The primary seed's deviation is modest;
seed 345678 is more visually pronounced but has less repetition evidence.
Prefer primary seed 123456 for repeatability, with existing 345678 plots as
illustration rather than a newly tuned scene.

### Existing commands, status and prerequisites

On the configured university workstation, from the repo root:

```bash
./run_isaac_sim.sh
# Open scenes/playground.usd, choose config/scenes/playground.toml,
# Initialize Runtime. Use direct motion for Pick/Place and both BTs.
```

The GUI accepts a complete TOML with:

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "static"
rrt_seed = 123456
rrt_max_iterations = 10000
max_obstacles = 64
```

However, **no shipped global scene/profile appears in `config/scenes/`**.
Adding this section to an arbitrary playground profile is insufficient: dynamic
external objects can cause static-world rejection, enhanced finger targets
must remain in range, and GUI Move preserves the current orientation while
V4.3 requires a fixed full pose/controlled IK. Profile activation is implemented;
a reproducible GUI detour preset is not already demonstrated. No architecture
change is needed to address this later.

For a prospective **windowed workstation** GT motion capture using an existing
harness (stop/close any other Kit first; not executed in reconnaissance):

```bash
./run_isaac_python.sh scripts/benchmark_global_rgbd_v5.py \
  --scenario v43 --execution-condition static --budget 64 \
  --artifacts-dir outputs/final-demo-v43-01 \
  --result outputs/final-demo-v43-01/result.json
```

Omitting `--headless` creates a window in this script. It runs the shared
authored scene, controls fixture creation/session edits, initializes the
existing runtime, performs paired research measurements, executes selected GT
motion, writes results and closes. It is **not** the GUI panel or an indefinitely
open presentation. Cloud headless execution adds `--headless` and sets
`ISAAC_SIM=/isaac-sim` inside the container, but will **not appear in the existing
stream**. A future cloud presentation must attach streaming to the same
Python-launched process, sequentially replacing/restoring only Kit as authorized
in that task. This report deliberately supplies no invented tested cloud GUI
command.

For strict V4.3 acceptance **after restoring hash-matching archival inputs and
portable path resolution**, use the existing serial driver with a fresh folder:

```bash
./run_isaac_python.sh -m scripts.run_global_rgbd_v5_regressions \
  --artifacts-dir outputs/motion-planning-v5/regressions/final-detour-01 \
  --case detour_free --case detour_obstacle --run
```

This command is currently preflight-blocked here; it is not a working-from-clean-
clone promise. It enforces exact fixed-target/fixture checks and retains both
roundtrip directions. The V5 `v43` candidate above is separately labeled and
does not substitute for this archival comparison.

Success inspection must read selected skill status, actual motion, endpoint
errors, PhysX reports/physical-contact steps, sampled cooked-hull clearance,
route waypoints/deviation, source/world immutability, final hold, solver
restoration and runtime cleanup. `status=PASS` in a research harness establishes
measurement validity only. Use fresh output folders for every trial.

## 10. Remaining work and documentation accuracy

| Rank | Category | Concrete work / source areas | Acceptance / scope |
| --- | --- | --- | --- |
| 1 | MUST FIX | Reproducible **visible GT demo**, using existing harness/profile/factory; `scripts/benchmark_global_rgbd_v5.py`, `scripts/global_rgbd_v5_protocol.py`, GUI launcher/extension inspected but workstation launcher preserved | One labeled GT scenario visibly executes in its actual window/stream; expected full-pose motion, positive sampled fixture clearance, zero reports for v43, no second Kit; exact operators' commands |
| 2 | MUST FIX | Artifact availability/path portability for strict V4.3 and V4.2 replay; `load_inputs`, `verify_frozen_design`, `load_design_evidence` | Restore original files, verify hashes; resolve only checkout relocation if justified. Missing/changed evidence still fails closed; never silently downgrade |
| 3 | MUST FIX | Fresh focused native/manual final acceptance on actual demo launch path | Direct manipulation, reactive preservation, meaningful GT detour + Home/cancel, project UI and worker startup; evidence independent of process exit |
| 4 | MUST FIX before release | Correct README/configuration/motion/validation navigation | Current global selectors documented, old phase records labeled historical, RGB-D limitations and missing prerequisites explicit |
| 5 | SHOULD FIX | Document offline research prerequisites, `requirements.txt` versus SciPy/PyYAML/Git; container ownership expectations | Clean offline setup can run all ten runner stages; avoid changing runtime pins or installing tools into the Isaac SDK |
| 6 | SHOULD FIX | Package a minimal hash-checked evidence archive and demo record | A clean target checkout can obtain required raw inputs without a huge matrix rerun; archival fixtures unchanged |
| 7 | OPTIONAL | Short operator-visible bounded preview/hold for existing benchmark and record demo video | Small harness-only addition if automatic close prevents viewing; no alternate motion loop or accepted-state change |
| 8 | OPTIONAL, low priority | One RGB-D base-enclosure investigation | Only if an equally conservative, source-isolated geometric alternative is concretely justified; test one change, retain observed floor evidence |
| 9 | DEFER | Hybrid/LLM/general-object/payload planners, new cameras, voxel/resource increases, more broad parameter sweeps | No release work; freeze RGB-D if no bounded defensible fix |

Documentation audit:

- **Accurate/current essentials:** launcher ordering/pins in README and GUI
  docs; manipulation release/held-state semantics; source/updater contracts;
  V4.1–V5 individual reports with their explicit evidence qualifications.
- **Stale current-feature claims:** README says global/hybrid unimplemented;
  `docs/configuration.md` only describes direct/reactive; `docs/motion_planning.md`
  opens with no-global wording and retains older unsupported statements;
  `docs/spatial_mapping.md` says no planner/backend consumes mapping before its
  historical sections. Source and newer V5 docs contradict those summaries.
- **Historical, not present acceptance:** `docs/validation.md` Phase 4C branch
  switching/release checklists and 297-test counts; earlier motion/mapping
  checkpoint counts; old “no perception/mapping workers” statements within
  phase-specific narratives. Preserve history but improve current navigation.
- **Reproduction incomplete:** strict replay commands omit raw-archive/path
  requirements; README plain-Python setup omits research SciPy/PyYAML; frozen
  perception reruns require old package cohort, not current provider wrapper.
- **Not evidence of a defect:** old source-freeze manifest differs from HEAD
  because subsequent integration intentionally changed provider structure. Keep
  original manifests, do not rewrite them to match current source.

No existing documentation was edited during this task.

## 11. Bounded finalization roadmap and minimum acceptance

| Milestone | Objective / why | Likely files | Acceptance and tests | Risk / scope / stop condition |
| --- | --- | --- | --- | --- |
| M0 baseline | Preserve exact state and distinguish missing data from failures | This report, existing output receipts | Offline result recorded; HEAD/staged work retained; historical/native confidence explicit | Complete with this reconnaissance; do not infer fresh physics |
| M1 preserve working robot and GT | Establish a source-stable dependable path | Existing direct/reactive/global code reviewed; fix only demonstrated faults | Full offline runner; existing direct and reactive native fixtures; unchanged held/placement/source/lifecycle gates | No proactive refactor; stop if no runtime defect demonstrated |
| M2 visible demo | Make actual robot motion reproducible to an observer | Existing V5 v43 harness/protocol; minimal separate cloud launch aid if needed; strict archival loaders if recovered | One active Kit, Python worker support, UI/render responsiveness, labeled static source; v43 selected GT motion/clearance/contacts; exact recorded command | Do not modify workstation launcher for cloud; no new planner/framework; stop after one reliable scenario |
| M3 RGB-D decision | Bound remaining research | `collision_model.py:preflight_base`, frozen adapter/tests only if justified | Written mechanism and equally conservative source-independent proof before one focused comparison at 64 | No currently established sufficient fix. Default stop: experimental frozen integration, no new native RGB-D trials |
| M4 acceptance | Collect fresh preservation/physical evidence after final changes | Existing runners + new fresh output paths | Matrix below; explicit per-case outcomes, contacts/clearance/errors and lifecycle cleanup | One native process at a time; stop on unsafe route/source/fixture mismatch; no complete V5 matrix by default |
| M5 freeze/handoff | Make thesis claims and launch instructions match evidence | README, configuration, validation, current motion summary, concise final delivery doc | Diff review, applicable offline suite, GUI manual checklist, hashes/resources/pins unchanged, local coherent commit only when separately authorized | No cosmetic expansion; list remaining unverified items rather than claiming release early |

### Minimum sufficient native acceptance for finalization

Run offline first, then reserve the GPU exclusively. Inspect processes before
any launch. In a later authorized task, manage only Kit processes, retain exact
original stream command and restore the stream; never restart container/VM.

1. `direct`: existing `tests/isaac/test_motion_strategy.py` covers Pick, held
   Home, stable Place/reacquisition, overlap/cancel/lifecycle preservation.
2. `reactive`: `tests/isaac/test_reactive_motion.py` covers ordinary move/Home,
   source failure and lifecycle without inventing a new avoidance claim.
3. Strict `detour_free` + `detour_obstacle` at primary seed once each, **if raw
   dependencies restored**, checking route difference and genuine Home with box.
   Otherwise use one new V5 `v43/static` capture plus original evidence and
   explicitly retain the archival replay blocker. Do not claim a fresh 5/5.
4. `global_cancel` to verify genuinely moving cancellation, hold and return Home.
   Add existing Stop/reset/close cases if launch, lifecycle, controller or cleanup
   implementation changes; they are not replaced by documentation review.
5. Existing `tests/regression/test_benchmark.py` for final manipulation accuracy:
   all four trials, maximum physical placement error 25 mm, localization/size
   gates unchanged. Direct smoke alone is not this four-trial acceptance.
6. Manual project GUI in the final launch context: source health/IDs, Move/Home,
   one Pick/Place/BT, no overlap, pause/stale behavior, clean close/reinitialize.
   Multi-view display check in separate session; it does not authorize grasping.

The serial driver supports case-specific selection and 600 s per-case bounds:

```bash
./run_isaac_python.sh -m scripts.run_global_rgbd_v5_regressions \
  --artifacts-dir outputs/motion-planning-v5/regressions/final-preservation-01 \
  --case direct --case reactive --case global_cancel --run
```

Add detour cases only after offline preflight succeeds. Native process-group
watchdogs in `run_obstacle_representation_v3_jobs.py:run_job` terminate only
their own fresh process group (TERM then bounded KILL); they do **not** discover
or protect an independently running stream. `run_all_regressions.py` itself has
no per-case timeout. GPU exclusion is an operator responsibility.

The full 14-case primary V5 matrix, six 128 jobs, camera search, exact-packing
variants and new holdouts are redundant for an unchanged experimental RGB-D
path. If a real RGB-D fix is approved, predeclare one previously base-rejected
case and one previously unsafe partial/large case at 64, with the existing
independent physical veto, plus GT preservation. Stop on unsafe/utility failure;
do not keep searching seeds or changing thresholds.

### Existing thesis evidence worth reusing

- V4.3 `figures/physical_obstruction_scene.png`,
  `global_joint_route_comparison.png`, `actual_joint_targets.png`,
  `physical_clearance_outward_home.png`: obstruction → route change → physical
  execution with measured margins, rather than TCP appearance alone.
- V4.2 `collision_geometry_comparison.png`, `hand_finger_coverage.png`:
  representation coverage and why stock model success is qualified.
- V4.1 `home_tracking_before_after.png`, `home_hold_five_seconds.png`:
  measured engineering failure, correction and remaining hold limitations.
- V5 `outcomes.png`, `cuboid_budget.png`, `primary64/*-worlds-routes.png`,
  `verified_occlusion/worlds-routes.png`, `source_to_plan_timing.png`:
  separate source acceptance, planner success, physical veto and actual goal.
- Multi-view final `camera_layouts.png`, `camera_ranking.png`,
  `holdout_accuracy.png`, `measured_clouds_geometry.png`, `native_latency.png`:
  controlled camera choice/geometry accuracy and throughput without claiming
  complete reconstruction or safe manipulation.
- Mapping V5 `evaluation.json` and architecture diagram in §3: demonstrates
  modular perception/representation/control ownership and worker responsiveness.

All paths above are under `docs/assets/` in their respective phase directories.
No new plots or thesis text were generated. Together these support a coherent
thesis narrative: reusable skills and asynchronous ownership; semantic versus
spatial representation; local versus global planning; controlled independent
physical evaluation; and the information limits of camera-only geometry.

## 12. Immediate next Codex task — proposal only

**Deliver one visible, reproducible GT `v43` demonstration through the existing
runtime, with honest provenance and no workstation-launcher change.**

Inspect `scripts/benchmark_global_rgbd_v5.py` (`main`, `run_native`, `execute`),
`scripts/global_rgbd_v5_protocol.py`, `scripts/global_rgbd_v5_pair.py`,
`scripts/benchmark_global_detour_v4_3.py:load_inputs/verify_frozen_design`,
`scripts/run_global_rgbd_v5_regressions.py`, `scripts/launch_isaac_gui.py`,
`runtime/process_worker.py`, and the current stream's exact command/log.
Reuse the current harness and strategy; do not add another executor.

First prepare an offline-validated, concrete launch procedure for a **single
Python-launched streaming/windowed process**. Keep all cloud options outside
the unchanged university launcher. If needed, add only a bounded harness preview
option and a small opt-in cloud entry point; preserve default benchmark output,
acceptance, cleanup and source hashing. Use existing UI/runtime construction
where practical. A headless subprocess beside the current stream is not a demo.

In parallel with source preparation (not agent delegation), inventory/recover
the exact V4.3 raw inputs from the original workstation/archive if accessible.
Require recorded SHA256 values. If unavailable, report that fact and use the
existing V5 `v43/static` scenario as a **new capture**; do not delete the strict
preflight or claim exact historical reproduction. Portable archival-path
resolution, if needed, must preserve hashes and reject missing/ambiguous files.

Acceptance: full offline runner; focused tests for any changed launch/preview/
path behavior; exclusively owned native GPU; actual visible GT arm motion,
full-pose completion, no reports/physical-contact steps for v43, positive
sampled candidate clearance, original tracking and finger contracts,
runtime/worker closure, solver restoration, and unchanged direct Pick/Place
smoke. Record stream restoration and exact reproduction commands. RGB-D remains
experimental. Stop after this concrete demonstration; select further cleanup
only after reviewing its evidence.

This proposal has **not** been implemented or run. No commit or push was made.
