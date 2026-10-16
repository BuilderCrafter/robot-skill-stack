# Phase 3A regression repairs

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
