# Native reconstructed RGB-D RRT proof — bounded negative result

**2026-11-14 — NO-GO: the requested detour was not demonstrated.** Exactly one
native run of the original nominal `partial` scene was performed, with zero
retries. After START setup, warmup and physical-reference preparation, no new
accepted complete mapping publication arrived within the declared five-second
readiness window. The experiment stopped there. Reconstruction, final frozen
capture, planning, physical trajectory evaluation and selected execution were
not reached. This identifies a source-readiness limitation of this trial; it
does not establish that reconstruction or Lula planning is impossible.

The approved starting design is
[rgbd-rrt-bridge-feasibility.md](rgbd-rrt-bridge-feasibility.md). Published numeric
evidence is in
[summary.json](../../research/rgbd_box_geometry/evidence/native_partial/summary.json)
and [validation.json](../../research/rgbd_box_geometry/evidence/native_partial/validation.json).
The summary includes the original failure traceback, camera calibration,
timestamps, protocol, cleanup receipt and hashes of local raw artifacts.

## 1. Initial and final Git state

Initial branch: `feature/global-rgbd-planning-v5`. Initial HEAD and independently
fetched remote branch both resolved to
`94d91f4dfce498b5e89bdc55f9ccd3d945b47a59`; the working tree was clean.
`AGENTS.md`, the bridge, visibility and reconstruction reports, and the V5 design
documentation were read before implementation.

This report, isolated research implementation, tests and compact evidence are
published in one child commit of that HEAD on the same branch, by normal
fast-forward push. The publishing commit identifies the final SHA; it is also
provided in the task completion message. At publication verification, local
HEAD, upstream and remote branch agree and the working tree is clean.
No main-branch update, history rewrite or force push is involved.

Preservation checks passed: 285 of 286 baseline files remained byte-identical;
the only changed existing file is the benchmark callback seam described below.
All 48 files bound by the previous reconstruction publication manifest and all
16 archived `static`/`v43` input/reference files retain their hashes. The baseline
manifest was saved after initial editing, using original **HEAD blob bytes**
for existing files rather than claiming the edited worktree was the baseline.
Recovery commit `edc1b467c6bf03c192d0d9dccce2b538309a024f` and
`refs/recovery/v5-realignment-20261114-worktree` remain present. Earlier reports,
fixtures, captures and experimental evidence were not overwritten.

## 2. Implementation, size and offline validation

| File | Scope | Lines at native attempt |
| --- | --- | ---: |
| `research/rgbd_box_geometry/rrt_bridge.py` | Exact exposure association, existing reconstruction/qualification, conservative numeric-world combination | 118 new |
| `research/rgbd_box_geometry/native_trial.py` | Fixed partial-only entry point, first-publication readiness, matched counterfactual and detour gate | 128 new |
| `research/rgbd_box_geometry/test_rrt_bridge.py` | Focused offline invariant tests | 159 new |
| `scripts/benchmark_global_rgbd_v5.py` | Optional research callbacks and unconditional restoration of injected strategy hooks | +19 / −3 |

Total before native execution: **427 added/deleted implementation lines**, including
tests. After the receipt-only correction below: **441** (118 + 126 + 175 new
lines, plus the same 22 benchmark diff lines). Evidence and this report are
separate from that implementation count.

The callbacks default to the existing capture and `MatchedSourcePlanner`.
Production strategy, Lula, executor, observation/mapping workers, interfaces,
collision resources, freshness limits, scenes, profiles and launchers are
unchanged. No dependency was added. This is experimental research code, with
native geometry/planning behavior still unvalidated.

Checks completed:

- Host before native: **97 tests passed**, 1.047 s.
- Existing container Python before native, without constructing Kit: **11 bridge
  tests passed**, 0.130 s.
- Host after the receipt correction: **98 tests passed**, 1.031 s.
- Architecture boundary, Python syntax, diff whitespace, original artifact
  hashes and the attempted-source recovery patch: **PASS**.

The host suite command was:

```bash
/tmp/robot-v5-realignment-py312/bin/python -m unittest \
  research.rgbd_box_geometry.test_rrt_bridge \
  tests.unit.test_global_rgbd_v5_pair \
  tests.unit.test_frozen_rgbd_world \
  tests.unit.test_global_collision_model_v4_2 \
  tests.unit.test_global_rgbd_v5_driver \
  tests.unit.test_global_frozen_rgbd_v5 \
  tests.unit.test_lula_global -v
```

These checks exercise exact tokens/times/calibration/publication matching, GT
input rejection, stale final-capture propagation, unqualified and mixed
robot-connected components, eight-corner containment, protected floors, complete
selected-cuboid accounting, proxy overflow, immutable read-view identity, failure
cleanup and existing default planning paths. They do not substitute for native
geometric or physical measurements.

One definite bookkeeping defect was discovered after the run:
`SimulationApp.close()` terminated the process before the entry point's
after-close `research.json` write. The benchmark's own result and diagnostics
were already saved. The published entry point now writes the receipt **before**
SDK close; an offline regression uses a nonreturning close to verify that order.
No native rerun followed this fix. It neither explains nor changes the earlier
readiness failure. The exact four attempted source files were copied into the
local recovery area before correction. Their hashes, the corrected hashes and
a verified corrected-to-attempted patch are published in
[attempt-source-manifest.json](../../research/rgbd_box_geometry/evidence/native_partial/attempt-source-manifest.json)
and [attempt-source.patch](../../research/rgbd_box_geometry/evidence/native_partial/attempt-source.patch).
The recovery artifact is a zero-context unified diff; an isolated checkout uses
`git apply --unidiff-zero` to recover the attempted sources. It is not a request
or authorization to repeat the native run.

## 3. Native environment and process ownership

The existing `isaac51` container was stopped, exit 137, with no Kit or GPU compute
process at inspection. It used its saved official
`nvcr.io/nvidia/isaac-sim:5.1.0` configuration and `sleep infinity` command. It was
started without recreation, volume, networking or VM changes. Native Python was
3.11.13, NumPy 1.26.4, with existing repository dependencies; the host test
environment used Python 3.12.16 and NumPy 1.26.4.

Only one owned Kit process ran: host PID **258304**, under launcher **258297**
and timeout **258291**. The observed GPU compute list contained only that Kit
process (2,164 MiB); two ordinary runtime worker child processes were also
recorded. No working stream was active or interrupted. The only filesystem
permission setup created the new output directory with host owner `ubuntu` and
container group 1234; existing parent/mount permissions were unchanged.

The exact native command was:

```bash
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.native_trial \
  --output outputs/motion-planning-v5/reconstructed-native-proof-20261114/attempt-1
```

The host wrapper also had a 650-second bound. Only `attempt-1` exists. The
subprocess returned **0**, but `result.json` reports **FAIL**. SDK close's exit
behavior must not be interpreted as a successful measurement.

## 4. Original cameras, source provenance and fixed scene

The authored `/World/PerceptionCamera` and `/World/PerceptionCameraSide` remained
640×480. Original rates were semantic 5 Hz and mapping 2 Hz, shared mapping ray
budget 1,024, stride 8, voxel size 0.01 m, evidence TTL 5 s and source freshness
limit 1 s. The enhanced Franka collision model, 64 camera proxies plus separate
support Plane, RRT seed 123456 and 10,000 iterations were unchanged.

The original full poses remained START `(0.45, -0.25, 0.35)` and GOAL
`(0.45, 0.25, 0.35)`, both orientation wxyz `(0, 0, 1, 0)`. This was nominal
`partial`, with Post and Occluder at the original arrangement, **not** the
separate `verified_occlusion` scene. Cooked evaluation-only fixture geometry
agreed numerically with the nominal centers/dimensions; it was not supplied to
the reconstruction callback.

During setup/warmup, mapping acquired 25 advancing two-camera batches and
semantic perception 87. The service health saved after warmup reported 25
submissions, 23 publications, one rejection, zero failures and a ready robot
filter. Seventeen accepted publication records were retained by the harness;
that observation-list count differs from the service's cumulative count.

The last recorded accepted publication was generation **0**, revision **30**,
original acquisition timestamp **20221.524619886**, observed at
**20221.976899053** (monotonic seconds). Both original camera tokens were
`(14.466666666666667, 6.366666998714209)`. The published summary preserves each
camera's exact intrinsics, extrinsics, original timestamp and 480×640 depth
shape. Robot-filter diagnostics then reported 16 and 8 self matches respectively,
with zero pose skew.

**This was warmup provenance, not an accepted source for the selected trial.**
The readiness callback ran after cooked-reference and observer preparation,
awaited Kit updates for at most five seconds and required the first new, unique,
fresh accepted publication and its exact retained camera-only `MappingInput`.
No such publication was accepted. The last mapping submission used original
time **20221.927171356**, tokens
`(14.766666666666667, 6.516667006537318)`. Semantic acquisitions continued through
**20227.127060028** while no later mapping acquisition/publication was retained.
No old warmup exposure or archived world was substituted.

The final mapping/filter admission diagnostics were not persisted on this early
failure. Consequently the reason mapping stopped producing publications is
**unresolved**; source inspection alone cannot assign it to the robot filter,
worker, camera pool or reference preparation. The recorded limiting mechanism
is the failed readiness boundary itself.

## 5. Post and Occluder reconstruction

| Measurement | Post | Occluder |
| --- | --- | --- |
| Sensor-only identification | Not attempted | Not attempted |
| Estimated center/yaw/dimensions | Unmeasured | Unmeasured |
| Six-face qualification, hidden bounds and FREE contradiction | Unmeasured | Unmeasured |
| Independent geometric containment/error | Unmeasured | Unmeasured |

`reference.reconstruct` and `visibility.qualify` were not called natively. The
bridge supplies only the matched RGB-D frames, synchronized robot snapshots and
configured workspace/support limits. It excludes wholly robot-envelope
components and rejects mixed robot-connected components; every other eligible
component must qualify. No fixture names, poses, dimensions or evaluator meshes
are passed. Those are implementation/test properties, not demonstrated live
reconstruction results. The previous archived `static`/`v43` positive results
remain separate evidence and are not relabeled as a successful fresh `partial`
capture.

## 6. Selected occupancy preservation and budget

No selected frozen world or combined inferred world was constructed natively.
Coverage, subsumption IDs and final camera-proxy count are therefore
**unmeasured**. Offline tests establish that the adapter retains every original
selected cuboid unless a qualified inferred box encloses all eight corners;
retains every protected floor cuboid and known Plane; audits the exact selected
ID partition; and rejects a camera-proxy total above 64. It returns a separate
`StaticWorldSnapshot` with `rgbd_inferred_box` labels, evidence and coverage
metadata, without modifying the original occupancy read view or cell states.

The last **warmup** extraction had 589 occupied voxels, 578 candidates, 64
selected hazard cuboids and **zero selected floor cuboids**. Its existing
extractor reported 525 omitted occupied voxels. These are not new losses caused
by the bridge: the preservation contract covers the entire original **selected**
union, not the complete map. Neither a combined-world budget pass nor floor/base
eligibility can be claimed from these warmup numbers. Base preflight was never
reached, and its guard was not weakened.

## 7. Freshness at final authoritative capture

**Not attempted.** At the end of warmup, the saved service health was fresh at
age **0.5465357709981618 s**. That age does not certify a later request. No final
capture age, frozen source timestamp or inferred planning snapshot exists.

The bridge calls the existing authoritative capture only after reconstruction,
using the original clock and unchanged less-than-one-second freshness test,
then requires identical generation/revision/timestamp and exposure/calibration
records. Tests verify stale-capture rejection after inference. No timestamp
refresh, relaxed threshold, second exposure batch or GT fallback was used.

## 8. Matched planning and counterfactual results

| Condition | Native result |
| --- | --- |
| GT control at common actual start/joint goal | Not reached |
| Selected reconstructed RGB-D world | Not reached |
| Support-only unobstructed Lula counterfactual | Not reached |
| Common validated IK joint goal / matched physics instant | Not established |
| Selected waypoint count / >0.05 rad direct-segment deviation | Unmeasured |
| Physically obstructed unobstructed route | Unmeasured |

`paired_plans` is empty; no selected request was issued. The isolated wrapper
delegates the existing matched GT/RGB-D plans and veto, then would compute one
support-only counterplan at the same actual start and validated joint goal,
check that physics did not advance, and require at least three selected
waypoints, more than 0.05 rad deviation and independent obstruction of that
counterplan before returning an executable selected plan. No fourth condition
or substitute control execution was introduced. None of these native detour
criteria was demonstrated.

## 9. Independent physical validity and veto

All 11 cooked robot body references were captured successfully, along with the
physical obstacle reference. `physical_evaluation` and motion payloads are
empty: no selected route/trajectory or counterfactual was evaluated, accepted
or vetoed. Sampled clearance and continuous safety are not established.
The evaluator remains research-only. Evaluation geometry did not enter the
reconstruction/planning source or replace the failed readiness source.

## 10. Actual motion, contacts and endpoints

The ordinary direct **START setup did move the robot**, before the selected
request stage. It succeeded in **84 steps**, with TCP position error
**0.0238056690 m** and orientation error **0.0102736860 rad**, within the original
0.025 m / 0.10 rad setup tolerances.

The selected GOAL request was never issued. Its trace contains zero actions,
zero samples and zero operations. There were **zero selected-request arm
commands**. Selected joint displacement, endpoint error, tracking error,
clearance, PhysX contacts/proximity and final hold are **unmeasured**, not
numerical zeros. No zero-contact or successful GOAL-execution claim is made.
Request-time solver restoration was not exercised. Final strategy diagnostics
report closed, enhanced, frozen RGB-D, seed 123456 and 10,000 iterations.

## 11. Timings and exact artifacts

| Stage | Measured result |
| --- | --- |
| Native subprocess start UTC | 2026-11-14 14:16:29.661113 |
| Native subprocess finish UTC | 2026-11-14 14:17:24.381829 |
| Total native subprocess wall time | **54.72072063 s** |
| Mapping warmup | **8.014158545 s** |
| Post-preparation source readiness | Five-second window expired; exact elapsed receipt unavailable |
| Reconstruction / final capture / planning / evaluation / execution | Not reached; no measured cost |

Warmup's last worker cost was 0.388395 s, including 0.272525 s compute and
0.045538 s robot filtering. These are ordinary mapping costs, not measured
incremental reconstruction costs.

Raw artifacts remain locally under
`outputs/motion-planning-v5/reconstructed-native-proof-20261114/`: the immutable
`attempt-1/`, `native.log`, protocol/declaration, baseline manifest, process and
container receipts, focused-test logs and exact `attempt-1-source/` copy. The
published summary binds their hashes; large/raw diagnostic files remain outside
Git. The original result hash is
`14f19fc34227e953b4c98f1a608dc24a4b5ce4e3ca593f2b7c3cf1348d7810a5`.
`attempt-1/research.json` is absent for the SDK-close bookkeeping reason already
recorded; it was not fabricated after the fact.

## 12. Cleanup and environment restoration

The saved result reports `runtime_closed`, `source_unchanged`,
`no_internal_runtime_error` and `all11_cooked_bodies` true. `frozen_scene` is
false because before/after request-world signatures were never obtained; it
does not establish that fixtures moved. Offline failure tests verify restoration
of both injected strategy hooks and closure of counterplanners even on errors.

The owned Kit and runtime worker children exited. Only the container's original
sleep process remained before shutdown. `isaac51` was then stopped to restore
its original operational state. Final inspection recorded stopped/exit 137,
no GPU processes and **0 MiB** GPU memory. The saved Config/HostConfig/Mounts hash
before start and after stop is identical:
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No stream needed restoration. No VM, Docker configuration, mount or network
change was made.

## 13. Assessment and mandatory stop

**NO-GO for the requested native proof.** A freshly accepted post-preparation
RGB-D source could not be obtained within the fixed readiness window. The
experiment consequently provides no evidence of a fresh reconstructed box
world producing an independently valid obstacle-induced RRT detour or executing
it successfully. The deeper mapping cause remains unresolved; the experiment
does not diagnose a geometry, base-collision, solver or trajectory failure.

One native attempt and zero retries were used. Source failure was not recast as
a harness defect, and the independent receipt correction did not justify a
retry. No other scene, threshold, seed, model, reconstruction approach or
production integration was attempted. The validated research source and honest
negative result are preserved for review. **This task ends with publication;
no further RGB-D development, native run or GUI work is authorized here.**
