# Phase 3A execution simplification

Phase 3A merged into `main` at `6f696d0`, after workstation regression and
manual GUI validation. The base was Phase 2 commit `904c5b7`.

## Execution path

All command callers use one cooperative implementation:

```text
GUI / Behavior Tree / standalone Kit bridge
  → await RobotRuntime.execute_async(...)
  → await skill.execute_async(...)
  → await backend command
  → cooperative Kit update (Isaac adapter)
```

Synchronous skill/backend commands, the synchronous BT executor and the
separate `AsyncSkillNode` were removed. Read-only queries and geometric
planners stay synchronous. Live Isaac callers use the existing
`run_kit_coroutine(..., simulation_app)` bridge; `asyncio.run()` is for offline
tests only. Backend command loops no longer use blocking `world.step()`.

## Interfaces and ownership

- `BaseSkill` declares `execute_async`. External callers migrate from
  `runtime.execute(...)` to `await runtime.execute_async(...)`.
- `ManipulationBackend` declares async move, open, close and Home commands,
  synchronous pose/reachability/grasp queries, `is_ready`, and grasp-width
  limits. Isaac-specific objects remain inside the adapter.
- `RobotRuntime` reserves its command before the first await and rejects
  overlap with `FailureCode.BUSY`. Ownership is released on completion,
  failure, exception or cancellation. Commands share one cooperative event
  loop; the runtime does not queue commands or support worker-thread control.
- The GUI uses backend `is_ready` and runtime `busy`. Its own busy state also
  protects an entire task/dialog.
- One `SkillNode` submits async commands and one `BTExecutor.run_async` ticks
  them. Cancellation or tick-budget exhaustion cancels and awaits outstanding
  node tasks before returning. This is task cleanup, not an emergency stop.
- The recovery selector remembers its running child, preserving the intended
  Pick → Home on failure → retry Pick → Place → optional Home sequence.

## Regression repairs and controller corrections

Benchmark Pick and Place are awaited in one Kit task. Separate submissions
had inserted an extra physics tick before Place captured its release
orientation. Removing that difference alone did not resolve unstable motion.

Benchmark stops on the first failed trial, preserving status, failure codes,
details, held state and physical measurements. It issues no cleanup release,
Home or next-trial command. Occupied placement asserts `TARGET_OCCUPIED`,
unchanged held state and zero backend command calls; physical drift during
Kit updates remains a diagnostic rather than a fixed movement threshold.

Two evidence-backed Franka corrections remain:

1. Motion exit retains the last arm position targets and sets desired arm
   velocity to zero. This covers success, timeout and cancellation without
   another simulation update or changes to finger targets.
2. `FrankaFrameVelocityPolicy` supplies measured joint-position differences
   over elapsed simulation time as RMPflow velocity feedback. It keeps
   closed-loop position feedback and the standard Franka RMPflow configuration.
   History survives Cartesian goal resets and uses the whole elapsed interval
   across gripper-only phases. The first sample or non-advancing time uses
   zero velocity. This frame average can miss within-frame motion.

No gains, Cartesian restrictions, pose tolerances, perception, tracking,
planning mathematics, Isaac lifecycle, launchers, NumPy precedence or dependency
pins changed. Production package Python decreased from 4,993 to 4,635 physical
lines (358 lines, 7.2%), including the controller corrections.

## Validation and scope

Cloud validation passes the architecture check, 221 discovered unit tests and
all eight standalone offline checks. SDK doubles establish adapter logic,
not native stability.

The reviewed workstation benchmark passes all four Pick/Place trials and all
retreats, with maximum true placement error 9.67 mm against the unchanged 25 mm
limit. The operator reported no contortions, then confirmed both Isaac groups
(two integration scenarios and six regressions) and manual GUI testing passed.
See [control diagnostics](control_diagnostic.md) for evidence and commands and
[validation](validation.md) for the repeatable suite and GUI checklist.

The GUI keeps stable placement with no release/drop selector planned.
Intentional release remains available through the skill API. Remaining held
state, retreat, recovery and Home limitations are recorded in
[manipulation](manipulation.md#current-execution-limitations); they are not
part of this cleanup or an automatic start of Phase 3B.

Superseded troubleshooting reports and initial size counts are recoverable
from Git as described in [history](history.md#phase-3a-diagnostic-history).
