# Phase 3A execution simplification

Base: merged Phase 2 commit `904c5b7` on `main` (including the direct-script
startup correction). Phase 2 ancestry was verified before creating
`refactor/phase-3a-execution`; the starting checkout was clean.

## Before and after

Before, runtime, skills, Franka commands and BT nodes each had sync and async
paths. GUI commands used async methods not declared by the backend interface;
standalone scenarios used blocking robot commands with `world.step()` loops.

Now all clients follow:

```text
GUI / SkillNode / standalone Kit bridge
  → await RobotRuntime.execute_async(...)
  → await skill.execute_async(...)
  → await backend command
  → cooperative Kit update (Franka adapter only)
```

There is one implementation of each skill and robot command. Queries and
geometric planners remain synchronous. The runtime reserves one command before
its first await, rejects overlap with `FailureCode.BUSY`, and releases ownership
on completion, failure, exception or cancellation. Rejected commands are not
queued. Commands must run on the same cooperative event loop, not worker threads.

## Public interfaces and modified components

| Component/files | Changes |
|---|---|
| `runtime/runtime.py`, `runtime/skill.py` | Remove `execute`; `BaseSkill` declares `execute_async`. Runtime calls it directly, exposes read-only `busy`, and reports `BUSY` for overlap. Existing exception-to-INTERNAL_ERROR mapping remains; cancellation propagates. |
| `manipulation/skills/{home,move_to_pose,pick,place}.py` | Remove synchronous implementations and Pick's synchronous motion helper; retain the original async sequencing/helpers. |
| `manipulation/backend.py` | Declare `move_to_pose_async`, `open_gripper_async`, `close_gripper_async`, `home_async`, read-only pose/reachability/verification queries, `is_ready`, and simple min/max grasp-width capability fields. No capability framework. |
| `integrations/isaac/manipulation/franka_backend.py` | Delete synchronous commands; keep the existing cooperative loops unchanged. Expose readiness through `is_ready` instead of requiring callers to access the articulation. |
| `orchestration/behavior_trees/` | One async `SkillNode`, one `BTExecutor.run_async`. Remove `AsyncSkillNode`, synchronous `run`, and the sync/async node-class selector argument to task factories. |
| GUI `ui/extension.py` | Continue using `execute_async`; use backend `is_ready` and runtime `busy`. Stop choosing an alternative BT node class. GUI-level busy state still protects an entire task/dialog. |
| Standalone Isaac scenarios | Replace command calls with the existing external `run_kit_coroutine(..., simulation_app)` bridge. Recovery runs the same async BT executor with Kit updates. All eight scenario scripts remain. |
| Unit tests and docs | Migrate fake backends and callers; remove duplicate two-mode coverage while retaining shape/attachment/rejection assertions. Add runtime, BT and SDK-boundary tests; update current architecture and validation guidance. |

No source files were moved or removed. The small existing skill registry remains;
this phase does not introduce plugin discovery or a synchronous compatibility API.
External callers must replace `runtime.execute(...)` with an awaited
`execute_async(...)`. Live Isaac must use Kit's bridge, not `asyncio.run()`.

## BT execution details

The recovery selector now remembers its running child. Otherwise every tick
would re-enter the failed first Pick while recovery Home was still executing,
which could overwrite the original failure or compete with recovery commands.
The intended policy is unchanged: first Pick, Home on failure, retry Pick,
Place, optional final Home. Tests hold Home across multiple yields to verify it.

On executor cancellation or tick-budget exhaustion, pending node tasks are
cancelled and awaited before returning. This prevents an orphan command from
leaving the runtime busy or executing after the caller considers the BT finished.
Successful/failed completed node results are retained for diagnostics.

This is coroutine ownership cleanup, not an emergency-stop implementation or a
claim that the physical robot has settled after cancellation.

## Size and duplication

Physical Python lines, including comments and blank lines, counted from tracked
files at `904c5b7` versus this change (including the two new unit-test files):

| Area | Before | After | Change |
|---|---:|---:|---:|
| Execution modules: runtime, skills, backend contract/Franka, BTs | 1,874 | 1,453 | −421 (22.5%) |
| Entire production package | 4,993 | 4,572 | −421 |
| GUI | 808 | 807 | −1 |
| Tests/support | 5,771 | 6,043 | +272 |
| Scripts | 241 | 241 | 0 |
| **All Python** | **11,813** | **11,663** | **−150** |

Savings remove parallel implementations rather than relocate them. Test growth
covers the previously unspecified busy/cancellation/backend contracts.

## Validation evidence

Baseline command:

```bash
.venv/bin/python scripts/run_all_regressions.py --unit-only \
    --artifacts-dir outputs/phase_3a_before
```

Baseline: 173 discovery tests, all eight standalone scripts, architecture/syntax
checks PASS. Final run with `--artifacts-dir outputs/phase_3a_after`: **186 discovery
tests, all eight standalone scripts and static checks PASS**. `pip check`, shell
syntax and `git diff --check` passed. Raw runs are under ignored
`outputs/phase_3a_before/run-51fnvkp2/` and
`outputs/phase_3a_after/run-y2xsdphu/`.

One duplicate sync/async attachment test was removed; its all-shape assertions
remain on the canonical path. Fourteen tests were added: nine runtime/BT cases,
four Franka command-loop cases using SDK doubles, and one external-runtime-busy
GUI case. Existing attachment, manual reset, reactive Pick, placement feasibility,
grasp-planner, perceived-sphere and async command tests were migrated/retained.

Tests cover sequential commands, no motion on busy rejection, exceptions,
timeouts, unreachable targets, cancellation and next-command recovery, BT
success/failure/recovery ordering, and cleanup on executor timeout/cancellation.
Adapter doubles check cooperative waits, unchanged move budgets/tolerances,
60/90 gripper updates, grasp feedback, readiness and Home results without
blocking `world.step`. They are not native Isaac simulations.

AST comparison against the base verified that all retained async skill and
Franka command bodies are unchanged. Perception, primitive fitting, grasp and
placement mathematics, tracking, WorldModel, Isaac lifecycle/camera code,
bootstrap ownership, launch scripts, dependency pins and direct-script dispatch
are unchanged.

## Workstation validation and remaining concerns

After checking out the branch on the existing configured workstation:

```bash
python3 scripts/run_all_regressions.py --group isaac \
    --artifacts-dir outputs/phase_3a_isaac
python3 scripts/run_all_regressions.py --group regression \
    --artifacts-dir outputs/phase_3a_regression
./run_isaac_sim.sh
```

Both integration scenarios and all six regressions must run and report PASS.
Use the [GUI smoke checklist](validation.md#manual-gui-smoke-test), repeat
Home/Move/Pick/Place and both BT tasks, and confirm rendering/perception continue
during movement. Check reactive disturbance callbacks, gripper verification,
placement reacquisition and repeated commands. Controls should reject overlap.

The old async update budgets/tolerances are unchanged; standalone scenarios now
share GUI cooperative timing rather than blocking physics stepping. Actual
wall-clock timing, physics/render cadence and benchmark accuracy must be
revalidated on Isaac. The cloud cannot prove native startup or those behaviors.

Reserved for Phase 3B: uncertain held state after lift failure/cancellation,
failed-retreat reporting, recovery-motion safety, diagnostic-only speed and
initialization-pose Home semantics. Phase 3A does not repair those issues, alter
their skill results, or change camera/lifecycle ownership. No Phase 3B/3C work
or automatic merge is included.
