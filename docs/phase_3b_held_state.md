# Phase 3B: held-state consistency

Branch: `refactor/phase-3b-held-state`, based on merged Phase 3A cleanup
`7e28b27`. Workstation validation is pending. This change uses finger positions
only; it adds no force/contact sensing, trajectory constraints or controller
tuning. NumPy precedence, launchers, dependency pins, lifecycle, perception,
tracking and grasp/placement mathematics are unchanged.

## State and ownership

WorldModel retains `held_object_id` and the measured attachment offset, adding
one `held_object_uncertain` flag. Its `gripper_state` property reports:

| State | Meaning |
| --- | --- |
| Free | No software attachment; position evidence established an empty/open gripper, or the operator reset it |
| Held | Existing finger-position evidence supports the candidate grasp |
| Uncertain | The candidate may still be held; its ID and offset remain protected |

These are estimates, not guarantees of attachment or identity. The existing
held-object label appends `(uncertain)`; no controls or toggles were added.
Clear Held resets the ID, offset, uncertainty and that object's hint without
robot motion. The operator must establish that clearing is appropriate.

`ManipulationBackend.gripper_feedback()` replaces boolean `verify_grasp()`
with `GripperFeedback(state, width, details)`. `inspect_gripper_async()`
provides stationary resolution; the Isaac adapter owns update timing. The
generic `HeldState` helper owns WorldModel changes and command policy.
Home and Move now receive WorldModel, just as Pick and Place already do.
The backend has no WorldModel or perception dependency.

## Position evidence and bounded checks

For Franka, finite two-finger measurements must lie within the finger travel
range (with tolerance derived from the existing minimum grasp width). A total
opening below 5 mm indicates empty closure; fingers at the physical upper
joint limits indicate an open/free gripper. Limits are read from the loaded
articulation using the two finger joint names, independently of the SDK's
nominal open/action targets. Under a close command, intermediate
width up to the existing 75 mm grasp maximum supports Held. Partial opening,
out-of-range values and invalid/missing measurements indicate Uncertain.

Stationary inspection requires three consecutive samples of the same known
state, with successive width changes no greater than 1 mm and advancing
simulation timestamps. It waits at most 12 update intervals (0.2 simulated
seconds at 60 Hz), then returns Uncertain. Stable feedback normally needs two
additional updates. Existing 60/90-update open/close command durations remain;
opening must also reach the measured open position before reporting success.

While carrying, Home/Move/Pick/Place pass a synchronous `grasp_check` to backend
motion commands. Backends must call it after each existing update, before
reporting motion success. Three consecutive suspect samples stop the command
through its existing arm-hold path. Three empty samples establish Free; mixed
or invalid samples establish Uncertain and trigger stationary resolution.
A suspect tail at motion completion is resolved before success is accepted.
Healthy carrying adds queries, not simulation updates or extra robot actions.

Detection is delayed by confirmation and physical finger closure. It cannot
prevent a drop, detect every slip before motion ends, distinguish the intended
object from another obstruction, or support objects below the grasp-width
limit. The hold is not an emergency stop; physical settling can continue.

## Skill transitions and failure policy

- Pick failures before closure leave state Free. Before closing, reserve the
  candidate as Uncertain. Settled grasp confirmation records Held **before**
  lift. Confirmed empty closure fails Pick and clears the attachment; unresolved
  feedback retains Uncertain. Post-lift inspection checks the grasp again.
- Known loss during carrying clears the ID, offset and stale association hint
  and returns `GRASP_LOST`. The cached object becomes invisible until a fresh
  provider observation, preventing a new Pick from using its old pose.
  Unresolved feedback returns `GRASP_UNCERTAIN`.
  The current operation stops even if a subsequent recheck establishes Held.
- Exceptions/cancellation around closing, release or carrying attempt bounded
  resolution. If it also fails, uncertainty remains and the original exception
  or cancellation propagates. A motion timeout keeps its original failure.
- Place rejects uncertain state before commands. Before opening, mark the
  candidate Uncertain. Verified release clears attachment while preserving the
  normal placement-reacquisition hint. Failed/interrupted release clears that
  hint; state follows measured evidence. Retreat failure never reasserts Held.
- Uncertain state blocks automatic Pick, Place, Home and Move. It is rechecked
  while handling the original interruption; unresolved cases require operator
  inspection. State resolution never automatically retries a failed operation.
- Recovery Home/retry is blocked after Pick reached closure, reported grasp
  loss/uncertainty, or raised an internal/busy error. Existing pre-grasp recovery
  remains. Broader recovery and failed-retreat result redesign are deferred.

## Validation

Offline tests cover state transitions, invalid/empty readings, slips during
lift/Place, brief transients, suspect motion completion, cancellation during
closing/lift/opening, timeout/exception preservation, protected candidate IDs,
manual reset, bounded sampling, normal arm hold and recovery rejection.
SDK doubles cannot establish native API compatibility or contact behaviour.

Cloud validation passes 244 discovered unit tests (23 added), all eight
standalone offline checks, architecture/syntax checks and documentation links.
The Isaac benchmark and loss diagnostic below have not been run in the cloud.

The first workstation upload at `0ec88c6` reported FAIL for both tests at the
initial opening. Measured positions were 0.0399999991 m per finger (80 mm total
opening), but the check used the SDK nominal target rather than physical joint
travel. The trace's requested opening targets exceeded the measured travel;
the exact nominal target was not logged. No close or cube-removal injection
occurred. This was a feedback-verification regression, not a successful fast run.

The correction reads the actual articulation limits, retaining the same
position tolerance and 60-update opening duration. Result details now include
physical limits, nominal targets, measured positions and classified state.
Unit tests cover an over-travel nominal target and non-adjacent finger indices.
The loss diagnostic also reports the original Pick failure when injection is
never reached.

The next workstation upload at `aa29f84` failed during runtime construction,
before perception or manipulation trials: `Franka` has no `get_dof_limits()`.
The first constructor double incorrectly provided that method. The corrected
adapter uses the public `dof_properties` structured array's `lower` and `upper`
fields, indexed by finger joint names. This API and the Franka → Robot →
SingleArticulation inheritance were checked against the
[Isaac Sim 5.1.0 source](https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.core.prims/python/impl/single_articulation.py).
The constructor double now matches that array layout and omits the unavailable
getter. No private view access, fallback or hardcoded physical limit is needed.
Native reruns of both scenarios are still required; retain them in a fresh
output directory such as `outputs/phase3b-dof-properties`.

From the existing workstation checkout, first run the ordinary benchmark and
observe all four trials. It must retain the 25 mm gate and smooth motion:

```bash
git fetch origin
git switch refactor/phase-3b-held-state
git pull --ff-only
mkdir -p outputs/phase3b-held-state
./run_isaac_python.sh tests/regression/test_benchmark.py --result outputs/phase3b-held-state/benchmark.json --artifacts-dir outputs/phase3b-held-state/benchmark > outputs/phase3b-held-state/benchmark.log 2>&1
```

Next run the explicit loss diagnostic. It uses ground truth and teleports the
cube away after a confirmed grasp has lifted it at least 40 mm. This is a
test-only object disturbance, not a force sensor or new robot command:

```bash
./run_isaac_python.sh scripts/diagnose_held_state.py --result outputs/phase3b-held-state/loss.json --artifacts-dir outputs/phase3b-held-state/loss > outputs/phase3b-held-state/loss.log 2>&1
```

The diagnostic must report PASS only after object removal triggered, Pick
reported `GRASP_LOST`, state became Free, attachment cleared, and command counts
showed one initial open, one close and no recovery Home or cleanup release.
It streams `loss/control_trace.jsonl`; retain this even after interruption.
This deterministic removal tests loss detection, not natural slip dynamics.
Stop on contortion/contact and send the output directory as a ZIP.

After both checks pass, run all eight original scenarios and GUI smoke tests:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/phase3b-held-state/isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/phase3b-held-state/regression
./run_isaac_sim.sh
```

Check normal Home/Move/Pick/Place and both BT tasks. Failed/cancelled manipulation
must not trigger an automatic release. The existing Clear Held action should
clear uncertainty after manual resolution. Do not interpret these commands as
already executed in the cloud. The loss diagnostic is separate from the eight
original scenarios; no merge is authorized by a test pass alone.
