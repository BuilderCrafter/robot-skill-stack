# Research-only RGB-D geometry-to-RRT feasibility

14 November 2026, UTC. Starting branch `feature/global-rgbd-planning-v5`, clean
HEAD **`3e2556d55c4700d07773e0a1a2d37227a32d33ef`**. Fetch and independent
`git ls-remote` agreed. Read [AGENTS.md](../../AGENTS.md), the
[visibility result](rgbd-visibility-final.md),
[reconstruction design](rgbd-reconstruction-design.md), relevant finalization
reports and the [V5 study](../../docs/motion_planning_v5_rgbd_global.md).

**GO for preparing one research-only native trial of unchanged `partial`, after
user review and separate authorization.** Existing interfaces provide a plausible
path without changing production workers, source checks, collision constraints,
planner or executor. Live reconstruction/freshness/base eligibility remain
mandatory failure gates, not established results. This is **not** a production
integration GO, and no bridge or native experiment is implemented by this task.

## Current operational state

Read-only checks at approximately 13:28 UTC found:

- Local/remote V5 at the SHA above; no staged or unstaged work.
- Existing container `isaac51`, image `nvcr.io/nvidia/isaac-sim:5.1.0`,
  **stopped**, exit 137. Docker reports start 13 November 19:44:10 UTC and
  finish 22:08:07 UTC. The cause was not established.
- Container startup command is a shell running `sleep infinity`; the existing
  repository mount targets `/workspace/robot-skill-stack`.
- No Kit process visible in host process inspection. NVIDIA L4 reports 0 MiB
  used and no compute applications. This is an observation, not a reservation
  of future exclusive access.

No container, Kit, network or infrastructure state changed. A future authorized
native task must recheck ownership and make the existing container runnable
without recreation. There is no currently running stream to interrupt. Historical
stream-restoration receipts do not certify a future session: if one is active
then, preserve its actual argv/environment/configuration and verify reliable
restoration before interruption; otherwise mark native testing blocked.

## 1. Interfaces already support the geometry

[`StaticObstacle` and `StaticWorldSnapshot`](../../robot_skill_stack/integrations/isaac/manipulation/motion/static_world.py)
already hold immutable numeric cuboids, pose/quaternion, dimensions, source
labels and frozen source metadata. `LulaGlobalPlanner._register` maps cuboids
directly to Lula `CUBE` obstacles with `SIDE_LENGTHS` and a transformed pose.
It does not require a USD prim or a semantic object for each descriptor.

A qualified research record can therefore become:

```text
StaticObstacle(
    path='/__research_rgbd_inferred__/<generation>/<revision>/<component>',
    kind='cuboid', position=<measured-model center in meters>,
    orientation=<model yaw as normalized wxyz quaternion>,
    dimensions=<already outward-qualified model dimensions>,
    source_kind='rgbd_inferred_box',
    approximation='single_convex_yaw_box_empirical_bounds')
```

Keep empirical model assumptions, all per-face evidence and uncertainty in
`StaticWorldSnapshot.source_metadata`; do not label an inferred box as exact
observed occupancy. A zero additional numeric proxy-surface error describes
registration of the already defined model, not zero physical inference error
or a certified physical uncertainty bound. No further padding or GT dimensions
are added.

The existing
[`frozen_rgbd.py`](../../robot_skill_stack/integrations/isaac/manipulation/motion/frozen_rgbd.py)
adapter supplies the original exact occupied descriptors and narrowly allowlisted
known support Plane. Preserve its original `FrozenRgbdWorld`/read view unchanged.
Build a **separate research `StaticWorldSnapshot`** for planning; do not put
inferred boxes into `FrozenRgbdBox` or relax exact-cell validators.

`GlobalRrtStrategy` already accepts research capture and planner-factory seams.
The current V5 benchmark uses per-instance `_rgbd_world_capture` and
`_planner_factory` hooks. Those research hooks can supply the combined world;
the strategy's RRT/route/spline/execution code remains unchanged.

## 2. Minimal proposed adapter

Use the existing retained `MappingInput` from
`scripts/benchmark_global_rgbd_v5.py:install_mapping_audit`, which already
keeps up to 16 submitted immutable inputs. Reconstruct directly through
`research.rgbd_box_geometry.reference.reconstruct` and
`visibility.qualify` using that input's frames and robot snapshots.

Do **not** call the archive-specific `inputs.load_capture` or `run.run` for live
data. Their hash allowlist, case paths and one-object replay assertions are
archive-test boundaries, not a live data source. Do not change them to admit a
fresh fixture. The mathematical helpers are already independently callable.
Pass only current workspace/support configuration, not fixture declarations or
tracked semantic/GT objects.

The proposed capture sequence is:

```text
one current accepted mapping publication + retained original MappingInput
  -> exact exposure/calibration/generation association
  -> unchanged bounded reconstruction and visibility qualification, once
  -> existing capture_frozen_rgbd_world, with real clock AFTER reconstruction
  -> assert the final accepted publication is the same one used for geometry
  -> combine <=64 camera proxies, retaining original support and occupied union
  -> existing StaticWorldSnapshot -> existing matched planner and executor
```

Select a retained input only when its generation, minimum original acquisition
time, camera IDs/tokens, dimensions, K/T and per-frame times exactly match the
publication's accepted `source.diagnostics['source_frames']`. The final frozen
world must have the same generation/revision/timestamp and exposure records.
Timestamp proximity alone and a generic latest semantic estimate are insufficient.
While reconstruction runs synchronously on the Kit owner, do not advance physics
or poll/apply another publication. No cross-worker reconstruction publication
or general synchronization mechanism is introduced.

Require nonempty qualified obstacle geometry. Every eligible nonrobot component
must qualify within the existing caps; ambiguous or unqualified components fail
the request. Robot-connected ambiguity must not be silently interpreted as an
empty external scene. Do not use the known fixture count or names to decide
which components matter. An entirely unseen obstacle remains a fundamental
limitation; the independent physical veto must still catch unsafe routes.

### Occupancy preservation and budget

Start with all original selected cuboids, without repacking or selection.
A model may subsume a nonfloor cuboid only if **all eight corners** lie
conservatively inside its convex numeric box. Record original IDs and coverage
associations. Keep every uncovered cuboid and every selected floor cuboid.
Use the original extractor's floor predicate (box top at/below configured support
plus one voxel and its existing numeric allowance); do not introduce a floor
deletion policy. If necessary, retain additional ambiguous floor-adjacent boxes.

Check that every original selected cuboid is either retained verbatim or wholly
enclosed by a registered model. The resulting **camera-derived** count must be
<=64. The existing separately allowlisted support Plane remains additional,
as in V5's original 64-camera-box/65-registered-obstacle convention. Over-budget
combination fails; no slots are reserved by dropping evidence and no 128 fallback
is permitted.

Keep the original occupancy arrays/read view and exact packet untouched.
UNKNOWN/FREE states do not change when a collision model spans inferred volume.
Only bounded descriptor/corner work belongs in this capture adapter. Do not call
the offline scorer's dense map/GT analysis from the capture path.

## 3. Freshness is a real gate after inference

The standard `capture_frozen_rgbd` checks healthy enabled RGB-D mapping,
automatic robot filtering, calibrated complete source records, current generation/
revision, expiry and original acquisition age against
`min(stale_after, max_frame_age)` (currently 1 s). Call it **after** reconstruction,
so inference time is included. Do not first validate, spend arbitrary time
reconstructing, and return that earlier acceptance as a fresh source.

The existing harness performs expensive cooked-reference capture after its
initial warmup. A small research readiness callback can await the **first new,
healthy complete mapping publication after that heavy preparation**, bounded
to five seconds, before sampling the actual start/body context. It uses existing
camera schedules/workers and acquires no extra sensor stream. Then associate
that one batch, reconstruct once and perform the final authoritative capture.
No loop searches for a floor-free or reconstructable exposure. If inference
ages the chosen batch beyond the unchanged limit, stop without another batch.

This ordering is plausible: historical primary partial/large source ages were
0.5270/0.4763 s, while the two existing offline full reconstructions took about
0.16–0.20 s on the host. It is **not** a native latency prediction. New partial
qualification may take longer or fail. A stale result remains a source rejection.

After one valid freeze, V5's original fixed-time static-world semantics govern
RRT, physical evaluation and execution. Preserve its acquisition timestamp and
record actual delays; do not re-date the snapshot or refresh its active world.

## 4. Why static/v43 are unsuitable, and why choose partial

A lightweight SDK-free check instantiated the actual numeric descriptors from
the published models, combined them using the existing eight-corner audit, and
called the **unchanged** enhanced base guard with identity robot-base frame and
the certified support Plane. This is an archived/interface check, not native
source acceptance or a replacement for native FK/base verification.

| Archived case | Camera boxes before/after | Model-only finite-base lower bound | Full original / combined base result |
| --- | --- | ---: | --- |
| static | 64 / 46 | +0.345579 m | Both `base_geometry_rejected` |
| v43 | 64 / 63 | +0.491980 m | Both `base_geometry_rejected` |

Both retain the same offending occupied-cell path
`/__rgbd_occupied__/32:33:69:70:1:2`. The inferred models themselves are not the
problem. The floor evidence remains, and all numeric cuboids still traverse the
original enclosing-sphere guard. The Plane exception is unchanged. Do not spend
a native trial reproducing these established rejections.

Historical primary evidence gives a better eligibility basis:

| Case | Selected floor boxes | Historical GT / RGB-D RRT waypoints | Physical result |
| --- | ---: | --- | --- |
| partial | **0** | 3 / 3 | GT allowed; sparse RGB-D plan vetoed |
| large | **0** | 3 / 5 | GT allowed; sparse RGB-D plan vetoed |

Both reached RRT, so neither is merely an expected base-rejection replay.
**Select unchanged nominal `partial`**: its Post is the same geometry/location
already reconstructed from static sensor data, providing a limited transfer
basis. Existing GT and RGB-D comparison also used IK index 0, avoiding the
additional historical branch-selection complexity of large (GT index 1).
Use the original START-to-GOAL request and authored cameras, seed 123456 and
10,000 iterations. No fixture or goal change is proposed.

The Occluder is a real unverified risk. The nominal partial layout did not
actually occlude the Post in the historical study; do not relabel this as the
separate `verified_occlusion` fixture. Large adds a substantially larger wall
and more unverified visibility/crop/point-cap effects. Neither has a local raw
archive that establishes reconstruction success. If partial lacks enough
current evidence, fail this one trial; do not switch to large or change the rule.
Fresh selected floor evidence can also reappear and must be retained/checked.

## 5. Matched planning, GT boundary and independent veto

Reuse `scripts/global_rgbd_v5_pair.py:MatchedSourcePlanner` as a delegate. It
already plans GT and RGB-D at the same unadvanced actual start, transfers only
the validated joint goal through `FixedJointTargetPlanner`, and applies
`PhysicalMotionEvaluator` before returning a selected plan to the executor.
Missing matched goal, failed alignment, unsafe/unresolved evaluation or missing
source must produce zero **selected-request** commands. Existing ordinary START
setup is separately recorded; it is not claimed absent from the whole process.

Add a small research factory wrapper, not a planning algorithm: after the
delegate returns an independently accepted primary plan and before any selected
command, compute an **explicit counterfactual unobstructed native Lula plan**
using the same actual `start_q`, the exact same validated joint goal, existing
known support only, seed and settings. No physics update may occur. The empty
condition is a separately labeled control, never a replacement RGB-D source.

Keep physical actors unchanged. Evaluate this empty-world route against the
original copied physical obstacles to establish a verified obstruction witness.
Never execute that counterfactual route. Compare the selected RGB-D route with
it using full seven-joint coordinates; require a meaningful deviation (>0.05 rad,
with at least three native waypoints), common endpoints and physical acceptance.
A longer route, a new IK goal or an unresolved obstruction query is insufficient.

This is a **native planner counterfactual**, not a physically empty executed
roundtrip. Record that qualification explicitly. It can establish obstacle-induced
planning change at an identical native state; it cannot establish an executed
empty-scene motion comparison. No such additional motion is included in this GO.

It also does not isolate inference's causal benefit over the **same fresh
occupied-only packet**: that would need another declared counterplan. Historical
unsafe RGB-D routes are not an identical-source control. This bounded experiment
can establish whether the combined sensor-derived world produces a physically
valid obstacle-aware route; do not claim the stronger causal comparison.

GT fixture geometry remains confined to the control and independent evaluator.
The reconstruction/combination callback receives only sensor input, known robot
data, exact selected occupancy and allowlisted support. It receives no fixture
names, physical reference dimensions or GT semantic objects. The validated GT
joint-goal transfer is the existing explicitly labeled experimental control,
not evidence of a general autonomous RGB-D IK solution.

## 6. Proposed files and bounded diff

No file below is implemented by this report.

| Proposed file | Responsibility | Estimated lines |
| --- | --- | ---: |
| New `research/rgbd_box_geometry/rrt_bridge.py` | Exact input association, direct core reconstruction call, final standard source capture, numeric descriptors, union/budget/provenance checks | 100–150 |
| New `research/rgbd_box_geometry/native_trial.py` | Explicit one-run entry point, preparation callback, matched delegate/control wrapper, metadata/source hashes and cleanup | 120–180 |
| `scripts/benchmark_global_rgbd_v5.py` only if needed | Optional research-only callbacks for preparation, capture and planner factory; defaults preserve existing behavior | 20–35 |
| New research-focused tests | Mismatched/stale source, enclosure/budget/floor retention, unchanged read view, matched goal/veto and hook restoration | 80–120 |

Estimated total: **320–485 lines including tests; production diff zero**.
Prefer the current harness seams. If optional callback plumbing is necessary,
keep it to this one existing research harness; no changes are needed in the
matched-pair implementation, `GlobalRrtStrategy`, `LulaGlobalPlanner`, factories,
workers, publication contracts, scene/profile files or launchers. The existing
delegate and fixed-target planner still do all actual planning. If this grows
into multiple subsystem modifications or >roughly 500 lines before testing,
stop and reassess rather than expand the experiment.

The existing harness source hash function omits `research/`; the native entry
must additionally record hashes of the research modules and freeze the protocol
before launch. Do not modify historical evidence or the archived-input allowlist.

## 7. Exact future native sequence and stops

After review and explicit native authorization, predeclare **one unchanged
partial Kit run** with three native planning conditions: matched GT control,
selected reconstructed RGB-D, and support-only unobstructed counterfactual.
Only the selected reconstructed plan may execute. No standalone GT preservation
job is necessary when production is unchanged; the in-process GT control is
retained. This fits within the maximum three-trial budget without a scene sweep.

1. Verify exclusive GPU/Kit ownership and usable existing container. Preserve
   and prove restoration of any then-active stream before interruption. Block
   if ownership/restoration cannot be established. Reuse existing launchers.
2. Start one owned Kit process with a 600 s subprocess timeout and five-second
   termination grace. Use a new output directory and retain every failed result.
   Preserve original profiles, cameras, mapping rays, scene fixtures and limits.
3. Perform the existing START setup, insertion and >TTL warmup. Capture cooked
   physical references for evaluation. Await the first eligible post-preparation
   publication within the bounded readiness window, then sample actual start
   and body context. Reconstruct its retained batch once.
4. Run final unchanged frozen-source validation; require exact source association,
   qualified models, retained occupied union and <=64 camera proxies. Record
   original/final times and age. Any stale, unqualified, mismatched or over-budget
   condition stops before planning/commands; no alternate source or batch.
5. Delegate matched GT/RGB-D planning, unchanged base preflight and independent
   physical veto. Failed base/goal/search/route/spline, missing common joint goal,
   collision, unresolved or unmeasurable evaluation stops selected execution.
6. Before selected execution, compute/evaluate the same-state/same-goal empty
   counterplan. Require verified original-route obstruction and meaningful
   RGB-D route change for a genuine detour claim. If either is absent, report
   no demonstrated detour; do not move fixtures or search endpoints/seeds.
7. Only if the selected plan remains independently accepted, let the unchanged
   timed joint executor attempt motion. Record actual movement, endpoint errors,
   contact/proximity reports, sampled clearance, tracking, finger envelope,
   final hold, solver restoration and lifecycle cleanup. Planner success alone
   is not native completion or safety. Do not execute a vetoed control/primary.
8. Close all planner/control handles, remove callbacks, restore per-run hooks,
   close runtime/app and verify no competing Kit. Restore the exact original
   stream if one was interrupted. Save large artifacts after runtime cleanup,
   as in the existing harness, and publish only verified research findings.

Allow at most one retry for a demonstrated implementation/harness defect, with
the original outputs retained. Stale source, failed qualification, budget/base
rejection, unsafe route or absent detour are **not** retry justifications. No
second scene, reconstruction method, tolerance adjustment, seed search or native
matrix follows failure. The independent veto remains research-only and sampled;
it is not a new production RGB-D safety feature.

## Verification and publication boundary

The SDK-free numeric schema/combination/base checks above succeeded as measured
and reproduced the two expected base rejections. Existing focused tests passed:
**43 tests, 0.580 s, exit 0**, using the host environment:

```bash
/tmp/robot-v5-realignment-py312/bin/python -m unittest \
  tests.unit.test_global_collision_model_v4_2 \
  tests.unit.test_global_rgbd_v5_pair tests.unit.test_frozen_rgbd_world
```

Log: `/tmp/rgbd-rrt-bridge-feasibility-20261114-tests.log`. These verify baseline
contracts, not an unimplemented adapter or installed native SDK behavior. This
task changes only `reports/finalization/rgbd-rrt-bridge-feasibility.md`. Commit
and normal push publish the investigation; stop afterward. No bridge, native
run, production integration or subsequent project phase is started here.
