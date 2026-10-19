# Phase 3A regression repairs

Current status: [workstation result and merge checks](control_diagnostic.md#latest-workstation-result-and-merge-checks).
The latest native benchmark passes all four trials with no operator-observed
contortions. The operator confirmed the final integration/regression groups
and manual GUI testing passed and approved merging Phase 3A.
The fail-fast benchmark policy supersedes the historical cleanup steps below.
The UI keeps stable placement without a release/drop selector, by operator
choice. Intentional release remains available through the skill API.

## Evidence and limits

The workstation archive `run-t1v_uflo.zip` reports four passing scenarios
(grasp planning, multi-object, reactive Pick, recovery) and two failures.
Benchmark ends at the obsolete synchronous `open_gripper()` cleanup call.
Its result has empty metrics and its log contains no original failed-trial
status, code or details. Those facts establish a test migration/reporting
defect, but cannot establish the underlying manipulation failure.

Placement feasibility reaches `TARGET_OCCUPIED` before failing its physical
end-effector drift assertion. This alone does not establish a new command:
the Kit bridge advances simulation, including residual motion from Pick.
Source inspection shows occupied-target rejection precedes backend commands.
The repaired scenario verifies this with spies on all four async command
methods; actual workstation command counts and drift remain to be measured.

The existing async backend command implementations survived Phase 3A; the
standalone execution path changed from blocking stepping to Kit coroutine
execution. Timing is therefore a hypothesis, not a confirmed explanation.
No production changes are justified by the supplied logs.

## Minimal changes

- Benchmark cleanup awaits `open_gripper_async()` through the existing Kit
  bridge. Failed Pick/Place status, codes, messages, details and trial geometry
  are written before cleanup. Cleanup failures retain the original trial and
  stop further trials while release is unconfirmed.
- Before/after snapshots record simulation time, observation age, world update
  count, updater error, object state and held state to diagnose timing.
- Placement rejection asserts zero motion/gripper command invocations, retains
  failure-code and held-state assertions, and records physical drift without
  treating residual motion as proof of a new command.
- Both scenarios retain collected metrics when an assertion raises.

All four benchmark trials must still succeed. Perception limits remain 15 mm
position / 12 mm size; benchmark and safe placement retain 25 mm true target
error limits. Production skills, lifecycle, perception, planning, tracking,
launchers, NumPy handling and dependencies are unchanged.

## Validation

Cloud command (passed):

```bash
.venv/bin/python scripts/run_all_regressions.py --unit-only --artifacts-dir outputs/phase_3a_regression_repairs
```

193 discovered unit tests, eight standalone offline checks and the architecture
check passed. Seven new tests exercise failed Pick/Place reporting, cleanup
failure/exception, unchanged placement accuracy, real Place rejection with
simulated residual drift, each prohibited backend command, wrong failure code
and held-state mutation. These use doubles and do not validate native Isaac.

From the repository root on the workstation:

```bash
mkdir -p outputs/phase_3a_repair
./run_isaac_python.sh tests/regression/test_benchmark.py --result outputs/phase_3a_repair/benchmark.json --artifacts-dir outputs/phase_3a_repair/benchmark
./run_isaac_python.sh tests/regression/test_placement_feasibility.py --result outputs/phase_3a_repair/placement.json --artifacts-dir outputs/phase_3a_repair/placement
python3 scripts/run_all_regressions.py --group regression --artifacts-dir outputs/phase_3a_repair/regressions
```

If benchmark still fails, inspect `manipulation_trials` first: Pick/Place codes
and details distinguish failed motion, grasp, occupancy and post-release
verification. Compare before/after observation ages, visibility, updater errors
and update counts with true final position. Preserve that result and log before
making changes. If failure persists, reproduce the same trial on the Phase 2
baseline in a separate checkout with the same workstation, scene and launcher;
compare result details and physical errors before attributing it to Phase 3A.
Do not add sleeps, retries or relax accuracy gates merely to obtain a pass.

## Workstation follow-up: Phase 2 comparison

The supplied Phase 2 `benchmark.json` reports PASS with all four manipulations
successful. Compared with the Phase 3A `run-hp4ge9ch.zip` results:

| Trial | Phase 2 true target error | Phase 3A outcome |
| --- | --- | --- |
| 1 | 13.62 mm, success | 10.71 mm, success |
| 2 | 4.56 mm, success | 79.18 mm, Place failed |
| 3 | 8.30 mm, success | Release approach timed out |
| 4 | 5.03 mm, success | Object not visible during setup; not executed |

Trial 2's perceived placement error (78.60 mm) agrees with its physical error
(79.18 mm), so stale perception alone cannot explain the miss. Trial 3's release
approach exhausted 1,000 updates: position error was 5.23 mm (limit 15 mm), but
orientation error was 0.12354 rad (limit 0.10 rad). Opening the gripper during
cleanup succeeded. Trial 4's setup exception is a later failure.

Both runs have identical perception benchmark statistics (2.01 mm maximum
position error, 2.95 mm maximum size error). AST comparison against `904c5b7`
confirms unchanged async backend motion/gripper/Home methods and unchanged Place
`execute_async`, `_poses` and `_finish`. Phase 2's benchmark used the removed
synchronous execution path. These observations strengthen the execution-timing
hypothesis but do not establish the exact cause or exclude run variability.
Do not treat this as a proven baseline defect or a perception-only failure.

The new Phase 3A placement-feasibility result passed: 1.184 mm physical drift,
zero calls to every backend command, unchanged held ID, and successful safe
placement with 8.97 mm physical error.

### Narrow diagnostic addition

Benchmark-only wrappers now record each Pick/Place motion and gripper command's
target, result, simulation time, physics-step index, measured TCP pose and true
cube pose before/after the command. Original results, arguments, exceptions and
cancellation propagate; wrappers are restored before cleanup. No per-frame
logging, world stepping, retries, waits or control-target changes are added.
Read-only snapshots add CPU overhead and may affect wall-clock perception
scheduling, so the trace is diagnostic evidence rather than an exact replay.
Failed trial-setup state is also retained without discarding earlier trials.

The trace can distinguish:

- Controller convergence: compare requested pose with measured TCP after motion.
- Execution cadence: compare physics-step deltas with returned motion `steps`.
- Object slip/release drift: compare true cube position relative to TCP across
  approach, descent, opening and retreat.
- Setup failure: compare physical pose with the cached visibility and age.

Offline verification of this addition passed 197 discovered unit tests, all
eight standalone offline checks and the architecture check. Four new tests
cover trace forwarding/restoration, timeout and geometry recording, exception/
cancellation preservation despite diagnostic-read failure, persisted trial
traces, and failed-setup reporting. Native behaviour remains unverified here.

After pulling `refactor/phase-3a-execution`, run from the repository root:

```bash
mkdir -p outputs/phase3a-comparison
./run_isaac_python.sh tests/regression/test_benchmark.py --result outputs/phase3a-comparison/benchmark.json --artifacts-dir outputs/phase3a-comparison/benchmark > outputs/phase3a-comparison/benchmark.log 2>&1
```

Inspect `manipulation_trials[*].backend_commands` for the first failing trial
and keep both JSON and log. No production correction is proposed until these
measurements identify a specific execution defect. The success and accuracy
requirements remain unchanged.

## Command-trace findings and sequential handoff correction

The subsequent instrumented workstation run reproduced the same results and
physical errors, including the 79.18 mm trial-2 miss. The user's observation of
shaking and ground contact is consistent with unstable control/contact; the
trace does not by itself identify a motion-planning mathematics defect.
The user also confirmed shaking/ground contact during the passing Phase 2 run.
This motion-quality problem therefore predates Phase 3A and remains unresolved
independently of the benchmark's execution-path divergence.

Verified observations:

- Every recorded motion has exactly one physics step per controller iteration,
  including trial 3's 1,000-step release approach (16.6667 simulated seconds).
  Gripper opening/closing consume 60/90 physics steps respectively. This rules
  out a count mismatch within these traced loops; update-event phase ordering
  is not established by boundary counts.
- Trial 2 loses the carried object during descent, before the open command.
  At descent start the cube is approximately `[0.4783, -0.0751, 0.1222]` m;
  at descent completion it is on the table at `[0.4328, -0.0317, 0.0250]` m,
  while TCP is `[0.4627, -0.0849, 0.0330]` m. The physical miss is already
  present before release, rather than being solely a perception/reporting error.
- Trial 3 keeps the object near the TCP, but the release approach never satisfies
  orientation tolerance. Its final state is not proof of the path's minimum
  clearance or absence of oscillation.
- Exactly one physics tick separates the end of Pick's lift from the start of
  Place's pre-place command in each recorded trial: 1783 to 1784, 2590 to 2591,
  and 3503 to 3504. TCP and cube move during that tick. Place captures current
  orientation when it starts, so that tick also changes its requested poses.
  Phase 2's direct synchronous skill sequence did not insert this scheduling
  boundary.

The minimal correction keeps sequential Pick and Place in one Kit coroutine
submission, directly awaiting both skills and retaining separate status,
failure details, snapshots and command traces. Failed Pick still skips Place.
Home and scene setup are unchanged. No production backend, skill, lifecycle,
planner, tracking, source, launcher or dependency changes are made. The change
removes a verified orchestration difference; it is not yet established that
this tick causes the full placement error or the observed shaking.

Offline verification: 198 discovered unit tests, eight standalone offline
checks and the architecture check pass. The new handoff test models a physics
tick at task submission, verifies no submission between Pick and Place, and
checks sequencing and argument forwarding. Existing tests retain failed Pick,
Place timeout, failed cleanup and unchanged accuracy coverage. These checks
cannot establish native stability.

Workstation validation uses the benchmark command above after pulling the
Phase 3A branch. The trace should show identical physics-step indices at the
Pick-lift completion / Place-pre-approach start boundary. Check all four trial
statuses and the original 25 mm accuracy gate, and observe the arm throughout.
Even a passing benchmark does not resolve observed shaking or ground strikes.
If those persist, retain this trace and investigate backend feedback timing,
arm targets/velocities retained during gripper phases, and contact behaviour;
do not conceal instability by adjusting tolerances or adding retries.

Two pre-existing limitations also remain relevant: `speed` is recorded but does
not scale RMPflow motion, and this project never registers the support plane or
scene obstacles with RMPflow. Reachability/endpoint checks do not establish a
collision-free trajectory. These are not proven causes of the Phase 3A
divergence and no speculative controller/planner changes are included here.
