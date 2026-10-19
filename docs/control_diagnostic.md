# Phase 3A: failure handling, placement modes and controller diagnosis

## Latest workstation result and merge checks

The uploaded `phase3a-frame-feedback.zip` reports **PASS** after the frame
velocity correction (`b574cf8`). All four perception-driven Pick/Place trials
and all four retreats succeed. True placement errors are 9.67, 2.55, 7.63 and
3.24 mm (mean 5.77 mm), within the unchanged 25 mm gate. Maximum perception
position/size errors are 2.01/2.95 mm, within the 15/12 mm limits. The log
confirms repository-local NumPy 1.26.4. The operator reports no contortions and
acceptable motion throughout this run.

The 1,716 recorded controller calls use velocity feedback exactly matching
joint-position differences over elapsed simulation time. Raw wrist velocity
still reaches 2.61 rad/s, while controller wrist feedback peaks at 0.98 rad/s.
This supports the contact-sensitive velocity feedback diagnosis. One observed
run does not establish general collision avoidance or stability in every scene.

No further control tuning or architecture changes are proposed before merging.
Keep the opt-in trace for diagnosis; normal regression execution leaves it off.
The latest cloud run passes 221 discovered unit tests, all eight standalone
offline checks and the architecture check. Before merging, run **both** Isaac
groups on the final code, without `--control-trace`, then perform the GUI smoke
test in [validation.md](validation.md#manual-gui-smoke-test):

```bash
git switch refactor/phase-3a-execution
git pull --ff-only
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/phase3a-final/isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/phase3a-final/regression
./run_isaac_sim.sh
```

Both integration scenarios and all six regressions must pass. The regression
group repeats benchmark with tracing disabled and rechecks occupied-target
rejection after the controller change. In the GUI, exercise Home, Move, Pick,
Place and both BT tasks; observe motion and continued rendering/perception.
These final checks are pending. The existing `release` skill mode is preserved;
its GUI selector remains a separate follow-up after this validation. No Phase
3B work or automatic merge is included.

The sections below retain the diagnostic history; earlier failed runs and
initially pending hypotheses do not describe the latest result above.

## Current changes

The benchmark stops at the first failed Pick/Place trial. It preserves the
original status, failure code, details, physical measurements, held-state
snapshot and command trace. It issues no cleanup release, Home or next-trial
command and does not clear held state. Completed/expected trial counts are
reported separately. This removes the cleanup command that caused the midair
drop. The separate backend corrections below address retained arm velocity
targets and active-motion velocity feedback. The script still closes
SimulationApp using its existing finalization path.

All four benchmark trials remain required for success, with the same accuracy
limits. All eight original Isaac scenarios remain. The initial failure-handling
and diagnostic step changed no production code. Subsequent corrections change
backend motion finalization and RMPflow velocity feedback; skills, perception,
planning mathematics, tracking, lifecycle, launchers and dependencies remain
unchanged.

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

## Loaded benchmark still fails: trace the active lift

The subsequent supplied benchmark fails at trial 3 with `grasp_failed`,
`Object was lost during lift.` Trial 1 and 2 complete Pick and Place with
10.72 mm and 15.35 mm physical target errors. Trial 3 closes around the cube
(50.03 mm feedback width), then takes 452 controller iterations to lift and
ends with an effectively empty gripper (0.0021 mm width). The cube is back on
the table. Place is never called for this trial, trial 4 is unrun, and no
cleanup command follows the failure. The benchmark correctly remains FAIL.

The user still observes twisting, shaking and ground contact. The arm hold
corrected a measured post-command defect, but it is not sufficient to make
active loaded trajectories reliable. Command-boundary snapshots cannot locate
the transient joint motion, physical contact or exact object-loss instant.
No new gain, planning or tolerance change is justified by those snapshots.

The same recorder used by the empty-gripper diagnostic now lives in
`tests/support/control_trace.py`. Benchmark has an opt-in `--control-trace`
diagnostic flag that records joint actions/feedback, TCP and true cube poses
through the existing Pick/Place sequence, with trial/skill labels and requested
Cartesian targets. Data are retained in `metrics.control_samples` on failure
or an operator KeyboardInterrupt. Temporary callbacks/wrappers are restored.
Normal benchmark and regression-group execution leave this flag off. All eight
original scenarios and their success/accuracy gates remain unchanged.

On the workstation, from the repository root:

```bash
git switch refactor/phase-3a-execution
git pull --ff-only
mkdir -p outputs/phase3a-loaded-control
./run_isaac_python.sh tests/regression/test_benchmark.py --control-trace --result outputs/phase3a-loaded-control/benchmark.json --artifacts-dir outputs/phase3a-loaded-control/benchmark > outputs/phase3a-loaded-control/benchmark.log 2>&1
```

If the arm starts contorting or contacting the ground, Ctrl+C can stop the
diagnostic and preserve its partial trace; completing the failed sequence is
not necessary. Send a ZIP of `outputs/phase3a-loaded-control` because the frame
trace can be large. No extra control commands or waits are introduced by the
recorder, though its CPU overhead may affect wall-clock scheduling.

Use the trace to distinguish abrupt RMPflow joint targets, poor tracking or
contact-induced motion despite smooth targets, and loss/slip of the object
relative to the hand. Also examine whether orientation requirements or posture
preferences compete with positional motion, especially at controller resets
between segments or near a singular configuration. The user specifically
raised excessive motion constraints as a hypothesis; it is not ruled out.
No position/orientation tolerance, path restriction or gain is changed for
this diagnostic. The project does not register the support plane/scene
obstacles with RMPflow, which remains a concrete collision-avoidance limitation;
the available boundary snapshots do not establish the exact collision source.
Do not equate endpoint reachability or successful endpoint tolerance with safe
intermediate motion. Controller diagnosis and loaded validation remain open;
GUI release selection remains gated on those results.

## Interrupted capture: write the trace while running

The next uploaded `phase3a-loaded-control.zip` contains a log and profile, but
no benchmark JSON. The log ends without Python's final exception/report output.
Consequently there are no per-frame samples to analyze from that run. The
previous claim that Ctrl+C would preserve the trace was too strong: a unit
test exercising Python exception unwinding did not establish interruption
behaviour of the native Isaac process.

With `--control-trace`, benchmark now writes `control_trace.jsonl` inside its
artifacts directory. Each completed sample is written and flushed immediately;
startup and phase markers are also flushed before runtime construction and
before motion begins. This file is the primary interrupted-run artifact. The
final `benchmark.json` remains useful if Python reaches its reporting path,
but is not required to recover the flushed journal prefix. Abrupt termination
can leave the last JSONL line incomplete; retain all preceding complete lines.
This is not a power-loss durability guarantee.

Recorder scope now covers initial settling, perception sampling, Home, scene
setup and Pick/Place, rather than starting after Home. Stage markers and
console phase messages identify whether unexpected motion happens during
initialization, Home or a manipulation skill. SDK runtime construction is
marked but not instrumented or changed. No lifecycle, signal-handler,
controller gain, pose tolerance or planning constraint changes are made.
Recording CPU/disk overhead can affect wall-clock timing, as with any trace.

Offline checks include a child process that writes samples and exits through
`os._exit` without exception handling or file closure; the flushed JSONL prefix
remains readable. Tests also verify startup markers exist before runtime
construction, controller-target metadata is serialized before flushing, and
Home precedes Pick in the recorded phases. Native signal behaviour itself
remains unverified in the cloud.

Run into a fresh directory on the workstation:

```bash
git switch refactor/phase-3a-execution
git pull --ff-only
mkdir -p outputs/phase3a-loaded-journal
./run_isaac_python.sh tests/regression/test_benchmark.py --control-trace --result outputs/phase3a-loaded-journal/benchmark.json --artifacts-dir outputs/phase3a-loaded-journal/benchmark > outputs/phase3a-loaded-journal/benchmark.log 2>&1
```

Stop the simulation when contortion/contact appears; completing the sequence
is unnecessary. Send the whole `outputs/phase3a-loaded-journal` folder as a ZIP,
including **`benchmark/control_trace.jsonl`**, even if `benchmark.json` is absent.
The active-motion cause is still undiagnosed; missing telemetry does not
justify a speculative controller or planner modification.

## Loaded journal: velocity feedback amplification

The next upload, `phase3a-loaded-journal.zip`, contains 6,672 complete journal
rows, including the interrupted second Pick. Trial 1 passes with 10.72 mm true
placement error. Trial 2 is interrupted during lift; it has no final result.
The flushed journal therefore provides the evidence missing from the previous
upload. The user reports slight wobbling on Pick 1, severe shaking/contortion
and ground contact on Pick 2.

During the final 30 control frames of each gripper closure, the held arm has
zero desired arm velocity. Wrist joint 7 feedback differs sharply from its
observed position changes:

| Closure | Joint-position range (rad) | Raw velocity range (rad/s) | Maximum frame displacement / elapsed time (rad/s) |
| --- | --- | --- | --- |
| Trial 1, steps 1197–1226 | 0.00001538 | -0.626 to +0.443 | 0.0003934 |
| Trial 2, steps 2026–2055 | 0.00002372 | +0.177 to +0.792 | 0.0001717 |

At the first second-trial lift command (step 2055), raw wrist velocity is
0.634 rad/s. At step 2056 it is 2.252 rad/s, although the intervening position
change corresponds to about 0.184 rad/s. RMPflow's next wrist target velocity
is 1.678 rad/s. By step 2105 the TCP has swept from approximately
`[0.475, 0.096, 0.034]` to `[0.360, 0.177, 0.025]` m despite a lift target of
`[0.479, 0.099, 0.146]` m. The trace establishes a feedback mismatch and its
amplification into commanded motion. It does not prove that the SDK getter is
buggy: instantaneous contact/solver velocity may differ from displacement
over a whole frame. It also does not establish every cause of ground contact.

The narrow corrective experiment is `FrankaFrameVelocityPolicy`, a small
subclass of Isaac's public `ArticulationMotionPolicy`. It supplies measured
active-joint position differences divided by elapsed **simulation** time as
RMPflow's velocity input. Position feedback remains closed loop. The first
sample, repeated simulation timestamp, or rewound time uses zero velocity
because no valid interval is available. History survives goal resets, and a
gripper-only interval uses its entire duration rather than one frame. This
introduces a frame-averaged velocity estimate and can miss rapid within-frame
motion; the first estimate also assumes zero velocity. It requires native
validation, particularly under loaded contact and after gripper pauses.

Controller construction now explicitly composes the same Franka RMPflow
configuration with Isaac's standard `MotionPolicyController`; reset still
restores the robot base pose. No gains, posture preferences, Cartesian goals,
pose tolerances, speed limits, collision geometry, skills or lifecycle steps
are changed. No clipping or additional trajectory constraints are introduced.
The inherited integration interval comes from the world's physics timestep
(the uploaded run verifies 1/60 s throughout). RMPflow's in-place rollout
cannot mutate the measurement history or recorded feedback copies.

The trace records raw `joint_velocities` alongside
`controller_joint_velocities` after each forward call. Offline tests exercise
the observed near-stationary wrist case, preserve actual high joint velocity,
cover gripper gaps/time resets, and verify SDK rollout mutation cannot corrupt
measurements. These checks establish the adapter's logic, not native motion
stability. Collision avoidance remains limited by the absent registered scene
obstacles. A passing endpoint/benchmark result alone does not establish safe
intermediate motion.

Run this targeted check first, in the existing checkout:

```bash
git switch refactor/phase-3a-execution
git pull --ff-only
mkdir -p outputs/phase3a-frame-feedback
./run_isaac_python.sh tests/regression/test_benchmark.py --control-trace --result outputs/phase3a-frame-feedback/benchmark.json --artifacts-dir outputs/phase3a-frame-feedback/benchmark > outputs/phase3a-frame-feedback/benchmark.log 2>&1
```

Stop immediately if contortion/contact returns and send the folder as a ZIP,
including `benchmark/control_trace.jsonl`. Assess smoothness, loaded lift
paths, retained objects, all four successful trials and the unchanged 25 mm
true placement-error gate. Do not treat a failure or early interruption as a
pass. If motion is stable, also run the placement rejection test and full
regression group before treating the repair as validated:

```bash
./run_isaac_python.sh tests/regression/test_placement_feasibility.py --result outputs/phase3a-frame-feedback/placement.json --artifacts-dir outputs/phase3a-frame-feedback/placement
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/phase3a-frame-feedback/regressions
```

The full group runs all six regression scenarios; the two additional Isaac
integration scenarios are selected with `--group isaac` (or run everything
with `--group all`). Launchers, repository-local NumPy and dependency pins
are unchanged. No merge or GUI release selector is part of this repair.

Cloud validation: `scripts/run_all_regressions.py --unit-only` passes the
architecture check, 221 discovered unit tests and all eight standalone offline
checks. Isaac startup, API integration and loaded motion remain unrun here.
