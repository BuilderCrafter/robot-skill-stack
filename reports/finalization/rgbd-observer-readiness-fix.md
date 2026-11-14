# Read-only research observer and RGB-D mapping readiness

**2026-11-14 — Readiness restored in the single authorized native check.** The
research observer no longer calls the inspected SDK getters that can author USD
DriveAPIs. Observer setup and physical-reference preparation preserved all 383
robot prim schema records and the valid robot-filter model/history. The first
new accepted complete mapping publication arrived within **0.063288926 s**,
with original acquisition age **0.584360683 s**, under the unchanged five-second
readiness and one-second source limits.

This is a mapping-readiness result. No reconstruction, RRT comparison, selected
GOAL request or selected execution was attempted. The native check stopped at
the first new matching publication. A full reconstructed-RRT trial still needs
separate authorization and all original source, geometry and physical checks.

Evidence:
[summary and individual gates](../../research/rgbd_box_geometry/evidence/observer_readiness_fix/summary.json),
[bounded native timeline](../../research/rgbd_box_geometry/evidence/observer_readiness_fix/timeline.json),
[SDK call-path audit](../../research/rgbd_box_geometry/evidence/observer_readiness_fix/sdk-audit.json),
[validation receipt](../../research/rgbd_box_geometry/evidence/observer_readiness_fix/validation.json).

## 1. Starting branch, commit and preservation

Branch: `feature/global-rgbd-planning-v5`. Starting commit:
`00f161212d405220f8dfc7ecf929d180d839b111`. The working tree was clean; an explicit
`git fetch origin feature/global-rgbd-planning-v5` confirmed the same remote tip.
`AGENTS.md` and
[the mapping-stall investigation](rgbd-mapping-readiness-investigation.md) were
read before changes.

A baseline manifest bound 832 tracked files before editing. Six directly related
source/test files changed; the other **826 remain byte-identical**. All raw
artifacts bound by the previous native-proof and diagnostic summaries, **24 and
21 files respectively**, retain their hashes. Previous reports, published
evidence, captures and fixture expectations were preserved. Publication uses a
normal commit/push on the same V5 branch; no main merge or history rewrite.

## 2. SDK getter paths inspected

The stopped container's installed Python sources were copied out for read-only
inspection without starting Kit or modifying SDK files. The audit records exact
SHA256 values for `single_articulation.py`, `articulation.py` and
`controllers/articulation_controller.py` in the official Isaac Sim 5.1.0 image.

| Research path | Installed SDK behavior | Corrected policy |
| --- | --- | --- |
| Robot/controller `get_effort_modes` | View `get_effort_modes`, definition 2580, calls `DriveAPI.Apply` at 2642 when a joint lacks the API | Never invoked; explicit unavailable field |
| Robot `get_max_efforts` | View definition 2790 uses live tensors with a valid handle, but its fallback calls `DriveAPI.Apply` at 2874 | Never invoked; explicit unavailable field |
| Controller `get_gains` | Controller line 129 delegates to view `get_gains`; view definition 3114 has `DriveAPI.Apply` fallback at 3216 | Never invoked; explicit unavailable field |
| `robot.dof_properties` metadata | Single-articulation property calls the same maximum-effort and gain getters at 268 and 270 | Aggregate not accessed; explicit unavailable metadata |
| Applied actions and position/velocity targets, when available | Inspected action path reads existing articulation action buffers; unsupported methods remain unavailable | Existing field names, values and meanings retained |
| Existing dynamics/drive-type diagnostics | Drive types read the physics view; solver-iteration/self-collision fields read existing prim properties | Retained; no unsafe gain/effort fallback substituted |

Controller effort-mode and gain methods delegate to the view, so omitting only
the optional robot methods would leave the demonstrated observer path exposed.
Similarly, retaining `dof_properties` would still reach prohibited conditional
authoring paths even after fixing `public_drive_feedback`.

## 3. Root cause and remaining uncertainty

The previous reproduction recorded a resync of
`/Franka/panda_hand/panda_finger_joint2` during `make_observer`, before physical
capture. The self-filter correctly latched a model error and cleared history;
the next ordinary observation tick correctly installed the source barrier and
blocked mapping acquisition/polling. That guard remains unchanged.

SDK inspection establishes which getter paths can author USD. The exact
individual getter that first triggered the old notice was not stack-recorded,
so this correction does not claim a separately isolated attribution to one
subcall. It removes every identified unsafe path used by these helpers. The
corrected native run produced no model notice or source error. If invalidation
had recurred, the diagnostic would have recorded its triggering stack and
stopped; no second attempt was authorized or used.

## 4. Code changes and rationale

`scripts/diagnose_global_home_v4.py` now:

- Preserves safe applied-action and target feedback.
- Keeps the existing maximum-effort, effort-mode and controller-gain keys, with
  `available=False` and a reason explaining the USD-authoring risk. It never
  calls or probes those unsafe getters, even when a live view might be valid.
- Reports missing methods, failed/absent controller feedback and `None` SDK
  results as unavailable. It fabricates no gains, modes or maximum efforts.
- Supplies explicitly unavailable aggregate DOF-property metadata, with null
  columns/values, rather than calling the unsafe property.

`scripts/diagnose_global_stabilization_v4_1.py` has only the necessary helper
import and aggregate-metadata replacement. The SDK property dependency above
demonstrates why a change beyond the preferred single helper file was needed.
Existing `authored_drive_properties()` still reads the separately labeled USD
metadata; authored values are not substituted for live feedback.

The two existing diagnostic unit modules gained focused regressions.
`research/rgbd_box_geometry/mapping_diagnostic.py` gained preparation-only schema
fingerprints, a triggering stack on model-error transition, and exact accepted
camera/calibration provenance. Its tests gained schema-audit coverage. These
are observational additions: the first-publication predicate, five-second
deadline and guaranteed stop before GOAL are unchanged. No extra cameras,
workers, rays or per-frame geometry processing were introduced.

`IsaacRobotSelfFilter`, observation scheduling, mapping service/worker, source
thresholds, frozen-world representation, reconstruction, Lula, motion execution,
skills, collision resources, scene configuration and launchers are unchanged.

## 5. Offline regression results

**83 tests passed on the host** in 0.209 s, and the **same 83 passed using native
Python 3.11.13 / NumPy 1.26.4** in 0.424 s before constructing Kit.
Architecture boundary, Python syntax and diff whitespace checks passed.

```bash
python -m unittest \
  tests.unit.test_global_home_v4_diagnostic \
  tests.unit.test_global_stabilization_v4_1 \
  tests.unit.test_spatial_mapping_prepare \
  tests.unit.test_observation_runtime \
  research.rgbd_box_geometry.test_mapping_diagnostic \
  research.rgbd_box_geometry.test_rrt_bridge \
  tests.unit.test_global_stabilization_analysis_v4_1 \
  tests.unit.test_global_stabilization_summary_v4_1 \
  tests.unit.test_global_acceptance_v4_1 -v
```

Unsafe robot/controller getter spies raise a `BaseException` immediately if
invoked, so ordinary diagnostic exception handling cannot conceal their use;
the tests also assert they were never called. Tests retain actual safe action
and target values, distinguish missing/failed feedback, and verify JSON
serialization. The real observer initialization/state methods run against SDK
boundaries and a real cached self-filter: unsafe aggregate access is forbidden,
the un-driven finger schema remains absent, cached parts/history are preserved,
and exposure preparation still succeeds. A subsequent genuine joint resync
still invalidates history and fails closed. Existing lifecycle and publication
barrier protections also pass.

## 6. Native observer initialization

Exactly one readiness attempt used nominal `partial`, the original authored two
640×480 cameras, semantic 5 Hz / mapping 2 Hz, shared 1,024-ray budget, 10 mm
voxels and original self-filter/freshness configuration. The 64 camera-proxy
limit and all motion settings remained unchanged. Warmup lasted **8.007664390 s**.

Schema fingerprints immediately before/after observer setup and physical
preparation were all identical: **383 prims**, SHA256
`54f4387f608349a4069cc42c879937c8ae5b04e576ccf2071aac4b78518ced67`.
The fingerprint binds prim paths, types and applied schema lists. No filter
model notice, schema change or source-invalidation error was recorded. Observer
initialization/state sampling and all 11 cooked-body references completed.

## 7. Robot-filter state before and after

| State | Before observer | After physical preparation | At fresh publication |
| --- | --- | --- | --- |
| `source_error`, model/capture error | Absent | Absent | Absent |
| Pose-history count | 256 | 256 | 256 |
| Filter history generation | 0 | 0 | 0 |
| Observation/mapping generation | 0 / 0 | 0 / 0 | 0 / 0 |
| Mapping invalidation error | Absent | Absent | Absent |
| Mapping source failures | 0 | 0 | 0 |
| Worker invalidation epoch | 0 | 0 | 0 |

Simulation remained playing. Ordinary ticks resumed after synchronous
preparation, advancing simulation time from 6.500000339 to 6.516667007. The
bounded trace contains 45 samples, 7 events and 299 observation ticks, with no
instrumentation errors. The filter was not reset, rebound or manually cleared.

## 8. Mapping continuation, identity and freshness

Before observer setup there were **24 submissions / 22 accepted publications**.
At diagnostic stop there were **25 / 23**. Worker started count rose 24 → 25,
completed/delivered 23 → 24, with zero worker/transport errors; one ordinary
mapping job was still running at stop. The single warmup rejection stayed 1;
there were no new rejections or source failures. Semantic attempts also
advanced, and all 61 semantic / 25 mapping camera batches returned advancing
tokens. These mapping counters and exact source association, not semantic
health alone, establish the result.

| Boundary | Monotonic seconds / identity |
| --- | --- |
| Previous accepted warmup publication | generation 0, revision 30, timestamp 24293.700201121 |
| Physical preparation finished | 24294.813133141 |
| Readiness started | 24294.817403896 |
| New post-preparation mapping input acquired / submitted | 24294.842614660 / 24294.858247253 |
| First new accepted publication observed | 24294.878443644 |
| Accepted identity | generation **0**, revision **31**, timestamp **24294.295523393** |
| Exact source association age | **0.584360683 s** |
| Readiness stop | 24294.880689885, elapsed **0.063288926 s** |

Both accepted camera tokens were
`(13.466666666666667, 6.266666993498802)`, with original acquisition timestamp
24294.295523393 and depth shape 480×640. Exact camera IDs, intrinsics, extrinsics,
tokens, timestamp and generation matched one retained camera-only `MappingInput`
through the existing `bridge.associate` check. Full calibration records are in
the published evidence. No timestamp was refreshed.

The accepted exposure was already in flight before preparation. It is a **new
accepted publication and exposure identity relative to the previously accepted
warmup result**, not a reused accepted warmup snapshot. Its original acquisition
time is reported explicitly. A further mapping input was genuinely acquired and
submitted after preparation; that job's completion was not awaited because the
diagnostic stops at the first new matching publication. This check therefore
does not claim a publication whose exposure was acquired after preparation.

The previous accepted source aged past one second during synchronous preparation
(about **1.113 s** afterward) and was not accepted as readiness evidence. The new
publication restored healthy/fresh status with its unchanged original age below
one second. No timeout, source age, admission or acquisition setting was relaxed.

## 9. Execution, runtime cleanup and restoration

```bash
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.mapping_diagnostic \
  --output outputs/motion-planning-v5/rgbd-observer-fix-20261114/attempt-1
```

The subprocess ran **39.580374007 s**, UTC **15:24:32.392870–15:25:11.973485** on
2026-11-14, with a separate 650-second host bound. Only owned Kit PID **312310**
ran, plus the existing semantic/mapping worker pair. No active stream existed.
The existing stopped `isaac51` was started using its saved configuration.

Ordinary direct START setup moved the robot and succeeded in 84 steps, with TCP
position error 0.0238056690 m and orientation error 0.0102736860 rad. No selected
GOAL request or arm command followed; motion samples, operations, paired plans
and physical trajectory evaluations are empty. No RRT or execution-success
claim is made.

The original benchmark records `FAIL` because the diagnostic intentionally
raises `DiagnosticStop: fresh_matching_publication` before its normal GOAL
measurement. The independent readiness gates in the summary record **PASS**;
SDK process exit 0 is not itself the acceptance criterion. `frozen_scene=false`
reflects unreached selected-request signatures, not measured fixture motion.
Runtime closed, source files remained unchanged during the run, and Kit/worker
children exited. Only the saved container sleep process remained before stop.

The container was restored to stopped/exit 137; GPU compute list is empty and
memory is **0 MiB**. Before/after Config/HostConfig/Mounts hash is identical:
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No container recreation, mount, network, VM, SDK or launcher change occurred.
Raw artifacts remain in the fresh directory above, bound by published hashes.
No original failed artifact was overwritten and no second attempt was made.

## 10. Recommendation and stop

**Readiness restored for the original first-new-publication contract.** The
observer's unsafe diagnostic reads have been removed; its schema/model state
remained intact and fresh RGB-D mapping continued afterward under the original
limits. Unsafe diagnostic fields remain explicitly unavailable, rather than
silently authoring robot state to obtain feedback.

The narrow native check does not establish reconstruction, source freshness
after reconstruction cost, base-preflight eligibility, a qualified combined
world, an RRT detour, physical trajectory validity or successful execution.
Those questions remain for the separately authorized original `partial` proof.
This task ends after publishing the correction, tests, evidence and report.
No full RRT rerun or additional development was started.
