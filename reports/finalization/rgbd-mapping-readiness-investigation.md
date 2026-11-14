# RGB-D mapping publication stall — focused diagnosis

**2026-11-14: classification B, confirmed in the single diagnostic reproduction.**
Research observer initialization caused a USD joint-prim resync, which
invalidated the robot self-filter's cached model. On the next observation tick,
the existing source guard blocked both mapping acquisition and publication
polling. Semantic acquisition continued through its separate scheduling branch.
The triggering transition occurred **before** `fresh_physical_capture()`.

This resolves the mechanism in a reproduction of the same nominal `partial`
setup. Attribution to the original failure is high confidence, rather than a
claim that its unrecorded final filter state has been recovered. Reconstruction
and RRT were never reached in either run. No corrective change or full RRT retry
was performed in this investigation.

Published evidence:
[summary](../../research/rgbd_box_geometry/evidence/mapping_readiness/summary.json),
[complete bounded timeline](../../research/rgbd_box_geometry/evidence/mapping_readiness/timeline.json),
[SDK source audit](../../research/rgbd_box_geometry/evidence/mapping_readiness/sdk-source-audit.json),
and [validation](../../research/rgbd_box_geometry/evidence/mapping_readiness/validation.json).

## 1. Original evidence and preservation

Work began on a clean `feature/global-rgbd-planning-v5` at
`e7af29321c2ac632961bb26d8f76da772038b3d1`; local HEAD, upstream and independently
queried GitHub branch agreed. `AGENTS.md`, the
[native proof report](rgbd-rrt-native-proof.md), raw native log, submission,
publication and camera records were inspected first.

The original run retained 25 accepted submissions, 23 cumulative publications,
one rejection and zero failures at warmup completion. Its final recorded mapping
submission was at monotonic **20221.95530112**; its final accepted source timestamp
was **20221.524619886**, generation 0, revision 30. Semantic acquisitions continued
through **20227.127060028**, beyond the five-second readiness interval. No later
mapping batch or accepted submission was logged. Thus acquisition/submission
stopped, rather than a continuing submission stream merely failing publication.
The original records alone did not identify the blocking condition: final
source/filter health and worker counters were missing. Semantic health alone
was not used to diagnose the cause.

All **825 previously tracked files** and all **24 raw artifacts** hashed by the
previous native-proof summary remain byte-identical. The old report and its
then-unresolved conclusion are preserved as historical evidence. This task adds
only two isolated research files, numeric diagnostic evidence and this report.
Existing production and benchmark sources are unchanged.

## 2. Readiness and scheduling timeline

The diagnostic used the unchanged benchmark setup and warmup, then stopped on
the first readiness update that exposed a specific invalid-source barrier.
All times below are original monotonic seconds; no timestamps were rewritten.

| Boundary | Time | Observation |
| --- | ---: | --- |
| Before `make_observer` | 22495.507060623 | Tick 314; filter error absent; history 256; mapping healthy, 25 submitted / 24 published |
| Filter model notice during observer setup | 22495.532941085 | Resync `/Franka/panda_hand/panda_finger_joint2`; model error latched; history cleared |
| After observer setup | 22495.536840888 | Same model error; mapping health still reflects the previous accepted publication |
| Before physical-reference capture | 22495.536991644 | Filter already invalid |
| After physical-reference capture | 22495.612222882 | Same error; no new model-error transition |
| Readiness starts | 22495.616716550 | Tick 314; original source age about 0.635 s |
| First readiness update / stop | 22495.690278112 | Tick 315; acquisition-failure barrier installed; 25 submitted / 24 published; source failures 0 → 1 |

The recorded error is:

```text
Robot subtree changed; reinitialize self-filter: /Franka/panda_hand/panda_finger_joint2
```

Simulation time held at **6.766667019575834** throughout synchronous observer and
physical preparation, then advanced normally to **6.783333687111735** on the next
tick. Physical capture took about **0.075231238 s** between the two sampled
boundaries. It did not stop the timeline or disable mapping in this trace.

At the final sample:

- `_mapping_enabled`, `_semantic_enabled`, `_playing` and `world.is_playing()`
  were true; runtime state was `running`, closed false, runtime error absent.
- `mapping_source_ready` was false, `_filter_needs_resume` false, and
  `_mapping_invalidation_error` matched the latched filter model error.
- Runtime and mapping generation stayed 0; mapping resets stayed 0.
- The semantic deadline was **22495.769442226047**, mapping deadline
  **22495.869442226**, with shared-camera `overlap` true. These are post-tick
  deadlines. Semantic acquisition occurred during that tick; a false sampled
  `semantic_due` reflects its advanced deadline. Mapping was not yet due, but
  its persistent invalid-source guard would prevent subsequent due acquisition
  and already prevented polling independently of the deadline.
- Worker counts were submitted/started 25, completed/delivered 24, running true,
  pending false, errors 0, no transport error. Invalidation advanced worker epoch
  to 1. There is no evidence of a worker-compute failure causing this stop.
- Source status became `error`, reason `acquisition_failure`; published timestamp
  stayed **22494.981527329**, age **0.708792691 s**. The failure was invalid source,
  not exceeding the one-second freshness limit.

## 3. Classification and confidence

**B: mapping submission stopped because the robot filter/source was invalid.**
Confidence is high: the trace records healthy state, the exact synchronous USD
notice and model-error transition, and the next tick's matching service barrier.
The production source code explains the resulting suppression without relying
on semantic camera health.

The trigger belongs to research observer setup. There was no observed mapping
disable, stuck monotonic deadline, Stop/Play, reset, generation change, lost
observation subscription or terminal worker error. The five-second original
readiness predicate was not shown defective. The exact SDK getter subcall
within observer initialization was not stack-instrumented; that finer attribution
remains source-supported rather than directly traced.

## 4. Relevant source-code findings

[ObservationRuntime.tick](../../robot_skill_stack/integrations/isaac/observation_runtime.py#L146)
checks cached `robot_filter.source_error` before acquisition or polling. A new
error calls `MappingService.note_source_failure`, sets
`mapping_source_ready=False`, and blocks both `_acquire_mapping()` and
`spatial_mapping.poll()`. Semantic acquisition/publication remain separate.
`overlap` only aligns due shared-camera mapping with semantic exposures; it
cannot override the invalid-source guard.

[IsaacRobotSelfFilter._model_changed](../../robot_skill_stack/integrations/isaac/spatial_mapping/robot_filter.py#L219)
invalidates the immutable model and clears pose history on a resynced prim under
the robot root, including a joint prim. Its cached
[source_error](../../robot_skill_stack/integrations/isaac/spatial_mapping/robot_filter.py#L334)
then remains non-null. Ordinary physics callbacks cannot clear `_model_error`.
This is the declared response to unexpected model changes, not a scheduler
deadlock. It should not be bypassed to accommodate diagnostic side effects.

[MappingService.note_source_failure](../../robot_skill_stack/world/spatial_mapping/runtime.py#L306)
increments its barrier, invalidates worker work and records acquisition failure.
This also prevents older completed success from clearing a newly invalid source.

The benchmark calls
[make_observer](../../scripts/diagnose_global_stabilization_v4_1.py#L255)
before [fresh_physical_capture](../../scripts/benchmark_global_acceptance_v4_1.py#L184).
Observer setup creates measurement views, gathers metadata and calls
`observer.state()`. Its
[public_drive_feedback](../../scripts/diagnose_global_home_v4.py#L38)
calls SDK `get_effort_modes()` through the robot and controller.

Read-only inspection of the installed official 5.1.0 SDK found that
`isaacsim/core/prims/impl/articulation.py:get_effort_modes` calls
`UsdPhysics.DriveAPI.Apply` at line **2642** when a joint lacks that API. Despite
its getter name, this path can author a schema and emit a prim resync. SDK
`get_max_efforts` and `get_gains` also contain authoring fallback branches;
`get_drive_types` uses the physics view. The source audit binds the exact SDK
file hashes and line locations. No SDK getter was invoked during that audit and
no SDK file was modified. The `get_effort_modes` path is a concrete, plausible
explanation for the observed finger-joint resync; the trace directly establishes
the containing observer phase, not the individual call stack.

## 5. Single diagnostic reproduction

The old artifacts could not establish final admission state, so the optional
single diagnostic reproduction was used. New files:

- `research/rgbd_box_geometry/mapping_diagnostic.py`: **190 lines**, temporary
  observational wrappers around the existing tick, model notice, camera-timing
  setup, observer creation and physical capture; a diagnostic-only stop callback.
- `research/rgbd_box_geometry/test_mapping_diagnostic.py`: **91 lines**, focused
  bounds, owner-thread, hook restoration, exception preservation and guaranteed
  pre-GOAL termination tests.

There are no persistent changes to wrapped classes or existing source files.
Sampling reads state after ordinary ticks, at most once per 0.25 s, plus forced
preparation/failure events. Limits are 128 samples and 64 events. The actual
trace contains **47 samples, 7 events and 315 observation ticks**, occupies
**238,209 bytes**, and recorded zero instrumentation errors. Every recorded
runtime snapshot was on the owner thread. No camera buffers, extra rays or
expensive frame analysis were added.

The original nominal `partial` fixtures, authored two 640×480 cameras, 5 Hz
semantic / 2 Hz mapping rates, 1,024 shared rays, 10 mm voxels, 64 camera-proxy
limit, source thresholds and START setup were preserved. No reconstruction or
planner callback was installed. The readiness callback never returns to the
selected GOAL path: it exits on a fresh, exactly associated first publication,
an identified source barrier, or the unchanged five-second upper bound. Here it
stopped after **0.073559696 s**, on the first captured source barrier; it did not
extend or claim to pass the five-second acceptance rule.

```bash
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.mapping_diagnostic \
  --output outputs/motion-planning-v5/rgbd-mapping-diagnostic-20261114/attempt-1
```

Native wall time was **40.181126577 s**, UTC **14:54:33.156053–14:55:13.337389**
on 2026-11-14. Process exit was 0. Benchmark status is `FAIL` because the
intentional `DiagnosticStop` aborts before its normal measurement; the diagnostic
outcome is explicitly `robot_filter_source_barrier`, not an RRT result.

Warmup lasted **8.081303874 s**. There were 25 mapping camera batches and 62
semantic batches; 24 cumulative accepted publications, zero rejected completions
and zero source failures before observer setup. Differences from the original
23 publications/one rejection reflect the separate run, not replaced evidence.
START setup succeeded in the same 84 steps, with position error
0.0238056690 m and orientation error 0.0102736860 rad. No selected GOAL request,
action, operation or motion sample exists; paired plans and physical trajectory
evaluation are empty. Setup movement is not selected GOAL execution.

The initially stopped `isaac51` was started with its saved configuration. Only
one Kit, host PID **288220**, ran. After exit only the saved sleep process
remained. The container was restored to stopped/exit 137, with no GPU compute
process and **0 MiB** GPU memory. Before/after Config/HostConfig/Mounts SHA256 is
identical: `8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No active stream existed; no container recreation, network, mount, VM or SDK
change occurred. Raw logs/results remain in the fresh directory above and are
hash-bound by the published summary. The original failed attempt is untouched.

## 6. Runtime versus research-harness defect

This trace establishes a **research observer / source-ownership interaction**.
The observation runtime and self-filter correctly fail closed after their
cached robot model receives an unexpected resync. Continued semantics do not
imply mapping is eligible. The existing general runtime need not be redesigned
to explain or address this failure.

A helper documented as observational reaches SDK getters with authoring
branches. That is the narrow research-harness concern. It could affect other
diagnostic users of the helper when they also maintain a robot-model notice
guard; this run does not establish a defect in ordinary GUI/direct/GT operation.

## 7. Smallest recommended correction — not implemented

Review `scripts/diagnose_global_home_v4.py:public_drive_feedback` first. Avoid the
SDK `get_effort_modes()` calls in both robot and controller feedback. Prefer the
already gathered authored-drive metadata using read-only USD access, or mark
that diagnostic field unavailable when its SDK getter can author a missing
DriveAPI. Do not create a drive or invent an absent joint mode for measurement.
This is a small research-helper correction, likely tens of lines including
tests, without changing motion inputs or production interfaces.

Also verify that the helper's gains/effort fallback paths cannot author USD
when no valid tensor view is available; unavailable feedback should remain
explicitly unavailable. If the targeted getter change does not eliminate
observer-created resyncs, stop and obtain further review rather than broadening
the runtime change. The exact subcall was not proven by this trace, so the
proposed correction is not presented as validated.

Do not clear `_model_error`, ignore joint resyncs, rebuild the self-filter
silently, reset the mapping service, use warmup data as a new source or alter
readiness/freshness bounds. No such fix was implemented here.

## 8. Validation and exact requirements for a future correction

Completed host tests: **62 passed** in 0.328 s across the new diagnostic, bridge,
observation runtime, mapping runtime, V5 driver and filter-packet suites; an
additional **13 robot preparation/source tests passed** in 0.074 s. The existing
container launcher ran the **3 new diagnostic tests**, all passed in 0.018 s,
without constructing Kit. Architecture and Python syntax checks passed.
The diagnostic tests ensure bounded buffers, owner-only SDK state sampling,
restored hooks, preserved original exceptions, strict deadline rejection, exact
first-publication association and termination before GOAL on every outcome.

For the proposed helper correction, require:

1. Extend `tests/unit/test_global_home_v4_diagnostic.py` with robot and controller
   spies whose unsafe getter would author a missing API. Feedback must never
   call it; missing data must be labeled unavailable. Preserve other feedback.
2. Extend `tests/unit/test_global_stabilization_v4_1.py` or an isolated observer
   regression to assert that observer initialization/state sampling adds no USD
   schema/API, leaves an un-driven/mimic finger joint unchanged, and does not
   invalidate filter history. Check both valid-view and unavailable-view paths.
3. Retain `tests/unit/test_spatial_mapping_prepare.py` and
   `tests/unit/test_observation_runtime.py` protections: genuine model edits must
   still invalidate history and prevent acquisition/polling of old successes;
   ordinary rigid motion must remain allowed.
4. Before any newly authorized RRT trial, verify the corrected observer has no
   model notice/source error and that the first new accepted complete exposure
   after preparation arrives within five seconds. Preserve its exact source
   provenance and the final capture's original age below one second. A future
   validation must not rely only on counters or an older fresh warmup packet.

No additional native run was performed to test a fix.

## 9. Can the original RRT proof now be repeated?

**Not on the unchanged current path.** The diagnostic reproduced a latched
invalid source before readiness. A narrow research-helper correction offers a
plausible route to repeat eligibility **without relaxing any source or safety
contract**, but that correction and its native eligibility remain unvalidated.
Diagnosis alone does not authorize or establish readiness for a full RRT retry.
Reconstruction, base preflight, proxy preservation, IK, detour, physical veto and
execution would still require all original checks if a later trial is approved.

This task publishes the diagnosis and observational tooling on the existing V5
branch by normal commit/push, then stops. No production fix, full RRT rerun,
GUI work or other research direction was started.
