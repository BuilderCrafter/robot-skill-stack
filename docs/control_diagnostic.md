# Isaac control diagnostics

## Latest workstation result and merge checks

Phase 3A merged into `main` at `6f696d0`. The reviewed
`phase3a-frame-feedback.zip` reports PASS after the frame-velocity correction
(`b574cf8`):

| Check | Result | Unchanged limit |
| --- | --- | --- |
| Pick/Place trials | 4/4 successful; all retreats successful | All four required |
| True placement error | 9.67, 2.55, 7.63, 3.24 mm; mean 5.77 mm | Maximum below 25 mm |
| Maximum perception position error | 2.01 mm | Below 15 mm |
| Maximum perception size error | 2.95 mm | Below 12 mm |
| NumPy | Repository-local 1.26.4 | Verified before SimulationApp |

The operator observed acceptable motion without contortions, then confirmed
both final Isaac groups (two integration scenarios and six regressions) and
manual GUI testing passed. Those final checks are operator-reported; detailed
benchmark measurements come from the reviewed upload. Cloud checks pass 221
unit tests, eight standalone offline checks and the architecture check.

The GUI keeps stable placement without a release/drop toggle. The existing
`release` skill API remains available for intentional drops. Stable mode
checks the observed final position; release mode checks that the intended
release pose was reached and the gripper opened, without landing accuracy or
visibility requirements. Both retain occupancy/reachability checks and command
failure handling. See [manipulation](manipulation.md).

## Why the controller corrections exist

Wobbling and ground contact also occurred on the passing Phase 2 baseline.
A numerical benchmark pass alone did not establish acceptable motion.

Empty-gripper traces showed nonzero desired arm velocities persisting through
finger-only commands. Holding the last arm position targets with zero desired
velocity reduced observed post-command drift:

| Phase | Before hold | With hold |
| --- | --- | --- |
| Open empty gripper | 14.21 mm | 1.35 mm |
| Idle after sideways motion | 14.36 mm | 1.15 mm |
| Idle after return motion | 12.22 mm | 1.62 mm |

The hold changes settling, not instantaneous actual velocity. In that diagnostic
one return pose settled 20.32 mm from its goal after meeting a 20 mm completion
tolerance at 19.98 mm. Manipulation accuracy gates were not relaxed.

Loaded traces then exposed contact-sensitive velocity feedback. During the
last 30 frames of the second gripper closure, wrist-joint position ranged only
0.00002372 rad, while raw velocity ranged +0.177 to +0.792 rad/s. Maximum
position displacement divided by frame time was just 0.0001717 rad/s.
At lift startup, raw wrist feedback reached 2.252 rad/s and RMPflow commanded
1.678 rad/s, followed by a large sideways excursion instead of the lift.

The frame-velocity adapter addresses that amplification using observed joint
positions and simulation time, without clipping or additional pose constraints.
Its history/diagnostic copies are independent of RMPflow's in-place rollout.
In the passing run, all 1,716 controller calls matched the measured frame
velocity estimate. Raw wrist velocity still peaked at 2.61 rad/s; controller
feedback peaked at 0.98 rad/s.

This supports feedback amplification as a cause of the observed instability.
It does not prove an SDK getter bug: instantaneous solver/contact velocity
can differ from displacement over a frame. Scene obstacles/support plane are
not registered with RMPflow, so endpoint checks and one successful run do not
establish general collision avoidance. See the
[execution record](phase_3a_execution.md) for adapter tradeoffs and ownership.

## Loaded benchmark journal

Use the existing launcher from the repository root; it preserves `.deps`
precedence and pre-SimulationApp NumPy verification:

```bash
mkdir -p outputs/control-check
./run_isaac_python.sh tests/regression/test_benchmark.py --control-trace --result outputs/control-check/benchmark.json --artifacts-dir outputs/control-check/benchmark > outputs/control-check/benchmark.log 2>&1
```

`--control-trace` is opt-in. Normal regression execution leaves it disabled.
The recorder observes existing calls and updates; it adds no robot commands
or waits and restores its temporary wrappers/callback. CPU/disk overhead can
affect wall-clock scheduling, so run the ordinary regression suite as well.

Artifacts:

- `benchmark.log`: startup and phase messages.
- `benchmark.json`: summary, trial results, failure details and command-boundary
  snapshots. `metrics.control_trace_file` identifies the journal; full frame
  samples are stored only in JSONL, not duplicated in the result or memory.
- `benchmark/control_trace.jsonl`: flushed startup/phase markers and per-frame
  joint, TCP, cube and action samples, covering Home, setup and manipulation.
  `joint_velocities` holds raw feedback; `controller_joint_velocities` after
  forward holds the velocity actually supplied to RMPflow. Targets, active
  indices and retained applied actions distinguish arm and finger commands.

Stop if contortion or ground contact returns. Send the complete output folder
as a ZIP, including the journal even when final JSON is absent. The native
process may exit before Python's reporting handler runs. Each journal row is
flushed immediately; discard an incomplete final line and retain the complete
prefix. A child-process test covers abrupt exit, not power-loss durability.

## Empty-gripper diagnostic

To isolate the controller from held-object contact and RGB-D processing:

```bash
mkdir -p outputs/control-empty
./run_isaac_python.sh scripts/diagnose_isaac_control.py --result outputs/control-empty/control.json --artifacts-dir outputs/control-empty > outputs/control-empty/control.log 2>&1
```

This uses the normal runtime with ground truth. It hovers at `[0.45, 0, 0.25]`,
opens/closes the empty gripper, moves sideways 80 mm, observes 60 idle updates,
returns, and observes 60 more. Motion tolerances are 20 mm / 0.10 rad. Failed
commands stop the sequence. `RECORDED` means collection completed, not a
stability or collision-avoidance certification. Samples are in `control.json`
if Python reaches reporting; use the loaded journal for interrupted captures.

The observed physics, rendering and controller intervals were all 1/60 s,
with computation/application at step N and updated feedback at N+1. Check
cadence, retained targets, raw versus controller feedback, object/TCP motion
and command failures before changing gains, constraints or accuracy limits.

Run both Isaac groups and the GUI checklist in [validation](validation.md)
after behaviour changes. Older failed runs and abandoned diagnostic procedures
are recoverable from [Git history](history.md#phase-3a-diagnostic-history).
