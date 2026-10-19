# Phase 3A: failure handling, placement modes and controller diagnosis

## Current changes

The benchmark stops at the first failed Pick/Place trial. It preserves the
original status, failure code, details, physical measurements, held-state
snapshot and command trace. It issues no cleanup release, Home or next-trial
command and does not clear held state. Completed/expected trial counts are
reported separately. This removes the cleanup command that caused the midair
drop; it does not implement an active arm hold or resolve wobbling. The script
still closes SimulationApp using its existing finalization path.

All four benchmark trials remain required for success, with the same accuracy
limits. All eight original Isaac scenarios remain. No production backend,
skill, perception, planner, tracking, lifecycle, launcher or dependency changes
are included in this step.

The existing Place modes have separate unit-tested contracts:

| Mode | Success verification |
| --- | --- |
| `stable` (default) | Release, then verify observed final position near the object-center target. This does not independently prove contact or settling. |
| `release` | Reach the requested object-center release pose and open the gripper; landing accuracy and visibility are not required. |

Both modes retain occupancy and reachability checks. A failed pre-approach,
descent or open command remains a failure. A motion timeout never authorizes a
drop. Existing `retreat_ok` reporting is unchanged. The GUI still selects
`stable`; exposing release mode follows native controller validation, as agreed.

## Run the focused diagnostic first

From the repository root on the workstation, after switching to and pulling
`refactor/phase-3a-execution`:

```bash
mkdir -p outputs/control-diagnostic
./run_isaac_python.sh scripts/diagnose_isaac_control.py --result outputs/control-diagnostic/control.json --artifacts-dir outputs/control-diagnostic > outputs/control-diagnostic/control.log 2>&1
```

The diagnostic opens the existing playground and builds the normal runtime
using its ground-truth source, without RGB-D processing or grasp planning.
It runs one Kit coroutine: hover at `[0.45, 0, 0.25]`, open/close an empty
gripper, move sideways 80 mm, observe 60 idle updates, move back, and observe
60 more idle updates. The TCP orientation is the benchmark's top-down
orientation. Motion tolerances remain 20 mm / 0.10 rad. Any failed command
stops the diagnostic before subsequent stages. Above-table targets alone do
not guarantee collision-free paths; observed motion still needs review.

Send `control.json` and `control.log` together (a ZIP is fine). `RECORDED` means
the sequence completed and data were collected; it is not a stability or
collision-avoidance verdict. `FAIL` retains the partial trace and original
error. This diagnostic is separate from the eight-scenario regression group.

The trace contains:

- World physics/render timesteps and RMPflow's configured timestep.
- Stage, event order, physics-step index and simulation/wall time.
- Actual joint positions/velocities and TCP pose.
- RMPflow actions, partial gripper actions and retained applied position/velocity
  targets, including their joint indices.
- Samples before/after controller computation, before action application,
  at a temporary physics callback and after the existing awaited update.

The recorder adds CPU overhead but no actions or extra simulation updates.
Its callback and method wrappers are removed on completion, exception or
cancellation; production lifecycle code is unchanged.

## Decide the controller correction from the trace

1. Compare world/controller timesteps. A mismatch is a concrete adapter issue.
2. Compare event order and feedback around computation, callback and awaited
   update. One iteration per physics step alone does not establish correct
   feedback timing.
3. Inspect partial finger actions against retained arm velocity targets and
   actual arm motion during gripper phases and idle. This tests whether old
   arm targets continue to drive the arm after a motion returns.
4. If hover itself fails or oscillates, investigate the controller/feedback path
   independently of held-object contact, perception and grasp planning.

These are hypotheses, not diagnosed causes. Native Isaac is absent in the cloud.
Do not choose a gain, timing, hold or planner change solely from passing SDK
doubles. Wobbling/ground contact also occurred in the passing Phase 2 baseline;
numerical benchmark success is insufficient.

After an evidence-backed correction and smooth diagnostic motion, rerun:

```bash
mkdir -p outputs/phase3a-verified
./run_isaac_python.sh tests/regression/test_benchmark.py --result outputs/phase3a-verified/benchmark.json --artifacts-dir outputs/phase3a-verified/benchmark > outputs/phase3a-verified/benchmark.log 2>&1
./run_isaac_python.sh tests/regression/test_placement_feasibility.py --result outputs/phase3a-verified/placement.json --artifacts-dir outputs/phase3a-verified/placement > outputs/phase3a-verified/placement.log 2>&1
python3 scripts/run_all_regressions.py --group regression --artifacts-dir outputs/phase3a-verified/regressions
```

Then check intentional release at a reachable midair object-center target:
motion must succeed before opening; landing elsewhere does not make release
fail. This is distinct from the stable benchmark, whose accuracy gates remain.
The GUI selector remains a subsequent change after native validation.

## Offline validation

```bash
.venv/bin/python scripts/run_all_regressions.py --unit-only --artifacts-dir outputs/phase3a_failure_control
```

New tests cover fail-fast benchmark handling with original held state and
timeout details, stable versus release semantics, recorder forwarding/event
ordering, retained arm targets versus partial gripper commands, idle periods
without extra actions, timeout stopping and callback/wrapper restoration.
