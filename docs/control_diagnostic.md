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
limits. All eight original Isaac scenarios remain. The initial failure-handling
and diagnostic step changed no production code. The subsequent arm-hold
correction below changes only backend motion finalization; skills, perception,
planning, tracking, lifecycle, launchers and dependencies remain unchanged.

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

## Workstation trace findings and arm-hold correction

The supplied `control.json` reports `RECORDED`, not startup failure. All seven
stages completed. The trace spans 6.50 simulated seconds in 2.06 wall-clock
seconds, with hover/side/back motions completing in 62/29/29 iterations. This
short execution explains why it could appear to stop before the test began.

World physics, rendering and controller timesteps all match at 1/60 s. Event
order is computation/application at step N, physics callback at N+1, then the
awaited-update continuation at N+1 with advanced feedback. That sampled path
does not show skipped physics or computation repeatedly using the same frame.
It does not establish that every contact-heavy benchmark path is stable.

The trace confirms retained nonzero arm velocity targets after motion returns:

| Following phase | Maximum retained arm velocity target | TCP displacement during phase |
| --- | --- | --- |
| Open empty gripper | 0.18149 rad/s | 14.21 mm |
| Idle after sideways motion | 0.16597 rad/s | 14.36 mm |
| Idle after return motion | 0.13188 rad/s | 12.22 mm |

Finger-only commands do not replace the arm targets. Actual velocity eventually
settles even though the old requested velocity remains nonzero; this is not
evidence of unlimited drift. It establishes that a completed command leaves a
nonstationary drive request active. The movement during closure can also change
the object/TCP relationship after Pick's pre-close attachment measurement.
These findings support a bounded correction, not a claim to have explained all
observed wobbling or ground contact.

`move_to_pose_async` now applies one arm-only position-hold action when exiting
after it has issued a motion command. It preserves the last RMPflow joint
position targets and active joint indices and sets their velocity targets to
zero. The `finally` path covers success, timeout and cancellation; an immediately
rejected target or paused motion with no applied action emits no hold. It does
not change gripper targets, mutate the original RMPflow action, await another
frame, relax tolerances or alter active RMPflow calculations. No gains, new
planner, retries or lifecycle changes are introduced.

This changes post-command settling and therefore needs native accuracy and
grasp validation, even though completion tolerances remain unchanged. It is a
position-target hold, not an instantaneous stop or proof of contact safety.
An SDK error while applying the hold still propagates as a command error.

Offline validation passed 212 discovered unit tests, eight standalone checks
and the architecture check. Backend tests cover arm-only target preservation,
zero final velocity targets, original-action immutability, unchanged physics
budgets, timeout/cancellation, rejection without a command, and following
gripper operations. Native stability remains unverified after the change.

After pulling the Phase 3A branch, rerun the diagnostic command above into a
fresh output directory (for example `outputs/control-hold`). Its final arm
velocity targets should now be zero through gripper and idle phases; compare
TCP drift and observed motion rather than requiring instantaneous zero motion.
Then run the benchmark and placement commands above if motion is stable. All
four stable trials and the existing accuracy gates must pass before the full
regression group and intentional-release check. The GUI selector still follows
native validation.

## Native arm-hold comparison

The next supplied `control-hold/control.json` completed all seven stages with
`RECORDED` status. Timesteps still match at 1/60 s. During every gripper/idle
sample, the retained arm velocity targets are zero. Using the same first/last
`after_update` TCP measurements in each phase:

| Phase | Before hold | With hold |
| --- | --- | --- |
| Open empty gripper | 14.21 mm | 1.35 mm |
| Close empty gripper | 0.076 mm | 0.017 mm |
| Idle after sideways motion | 14.36 mm | 1.15 mm |
| Idle after return motion | 12.22 mm | 1.62 mm |

This verifies the intended target change and substantially reduces observed
post-command movement in the empty-gripper case. It does not establish loaded
grasp stability or collision-free manipulation. The configured profile is the
expected playground with ground truth and a 50 mm cube.

Holding the last joint position target also removes the old continued approach
to the Cartesian goal. Motion completion was within the unchanged 20 mm
tolerance, but the last return pose settled approximately 20.32 mm from target
after idle (19.98 mm at reported completion). This small post-command change
must remain visible; native manipulation must still meet its existing grasp,
success and 25 mm true placement-accuracy requirements. No thresholds are
relaxed to compensate. The next required measurement is the stable benchmark,
followed by placement feasibility and the full regression group if successful.
Intentional-release validation and GUI selection remain pending.
