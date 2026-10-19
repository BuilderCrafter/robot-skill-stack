# Validation and release baseline

## Supported launch and test workflow

Use the existing Isaac Sim **5.1.0-rc.19 / Python 3.11** installation and
repository-local `.deps` with **NumPy 1.26.4** and **py_trees 2.6.0**. From the
repository root:

```bash
./run_isaac_sim.sh
python3 scripts/run_all_regressions.py --unit-only
python3 scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/isaac
python3 scripts/run_all_regressions.py --group regression --artifacts-dir outputs/regression
```

The GUI wrapper sets `ISAAC_SIM` (default `/home/etfrobotics/isaacsim`), uses the
full experience and preserves ROS bridge startup overrides. `run_isaac_python.sh`
prepends `.deps` and the repository to `PYTHONPATH`. GUI/scenario entry scripts
import and verify NumPy before importing `SimulationApp`: Isaac/plugin imports
can otherwise establish an incompatible NumPy first, too late to replace safely.
The version/path log proves pre-import, not successful native SDK startup.
Never add `--no-ros-env`; this Isaac launcher does not support it. Do not change
drivers, workstation dependencies, Isaac installation or SSH configuration to
run this checklist.

Native tests launch conventional files directly through the wrapper. The retired
`python.sh -c` + `runpy` dispatch changed the registered `__main__`, `__file__`,
`__spec__` and argument context and coincided with startup segfaults. The precise
native fault was not established in the cloud; direct-script launching restores
the working execution context without changing the SDK. If startup regresses,
compare a minimal direct-script `SimulationApp` probe and GUI experience/logs
before blaming scenario code. The original probe and evidence are recoverable
from `1f53ed8:docs/isaac_runner_startup.md` (see [history](history.md)).

| Group | Checks |
| --- | --- |
| `unit` / `--unit-only` | Architecture/syntax, unittest discovery and eight standalone offline modules; uses invoking Python 3.11/3.12 with `requirements.txt` |
| `isaac` | Discovery and perception-manipulation integration scenarios |
| `regression` | Benchmark, grasp planning, reactive pick, recovery, multi-object and placement feasibility |

Omit group selection to run all checks. Each selected scenario runs once. Failed
exit codes, missing result JSON or non-PASS native results fail the run; later
checks still run. Fresh artifact directories prevent stale PASS reuse. With
`--artifacts-dir`, inspect logs, scenario JSON and `summary.json` with actual
exit codes. Without it, native result files are temporary and terminal output
remains visible. Do not treat an open window or a NumPy line as a test pass.

Offline doubles cover behavior and boundaries, not real rendering, physics,
camera timing or SDK compatibility. The cloud `.venv` does not substitute for
Isaac's `.deps` launch environment. Unit discovery alone omits eight standalone
modules; use the consolidated runner.

## Final Phase 4C workstation commands

From the normal checkout, without a separate temporary checkout:

```bash
git fetch origin
git switch refactor/phase-4c-final-cleanup
git pull --ff-only
python3 scripts/run_all_regressions.py --unit-only \
  --artifacts-dir outputs/phase4c-unit
python3 scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/phase4c-isaac
python3 scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/phase4c-regression
./run_isaac_python.sh scripts/diagnose_held_state.py \
  --result outputs/phase4c-loss.json \
  --artifacts-dir outputs/phase4c-loss
```

Keep all **two integration scenarios and six regressions**, including the
**four-trial benchmark** and **25 mm maximum physical placement-error gate**.
All trials must pass; no threshold relaxation is part of consolidation. Preserve
the benchmark's existing localization/size limits and same-ID reacquisition.
Occupied-target rejection must invoke zero backend motion/gripper commands;
physical settling during the Kit bridge remains diagnostic, not a command spy.

The separate injected-loss diagnostic must show that removal actually triggered,
Pick returned `GRASP_LOST`, state became Free, attachment cleared, and there was
one initial open, one close and no cleanup release or recovery Home. A quick
pre-injection failure is not a pass. Retain its streamed journal even if the
process is interrupted. It tests deterministic removal, not all natural slips.

## Manual GUI smoke test

Launch `./run_isaac_sim.sh`, open `scenes/playground.usd` and use the checked-in
setup. Then repeat with a complete ground-truth copy as described in
[configuration](configuration.md#required-settings-and-source-selection).
Restart Isaac before switching setups; no code/environment changes are needed.

- [ ] Verify NumPy version/path, initialized backend, source health, selectable
  IDs and live cached poses with **both sources**. Also initialize RGB-D using a
  setup copy without the optional ground-truth section.
- [ ] Observe separated cubes, spheres and cylinders in the configured workspace;
  check geometry display and lost/visible transitions during controlled occlusion.
- [ ] Exercise Home and Move in a clear workspace. Home returns to the captured
  startup TCP reference; it is not a certified collision-free joint pose.
- [ ] Pick a supported object, verify grasp/Held state, then stable Place at a free
  tabletop target. Check actual release, final pose, same-ID reacquisition and
  cleared attachment. The GUI uses stable mode only.
- [ ] Run both BT tasks and sequential commands. Confirm responsive rendering and
  perception throughout, and disabled overlapping actions while busy. Recovery
  must not restart Pick while Home is running or recover after closure/loss.
- [ ] Verify occupied placement and clearly unreachable targets fail before the
  corresponding motion. Do not place physical obstructions in a robot path to
  manufacture a failure.
- [ ] Check disappearance/reacquisition and health on pause/resume. Pause long
  enough for stale health and disabled observation-dependent controls; resume
  and confirm recovery from fresh samples. Held/hinted IDs must stay protected.
- [ ] Check Clear Lost and, only after establishing safe physical state, confirmed
  software-only Clear Held. A busy command or changed ID must reject reset.
  Uncertain state blocks automatic manipulation until explicitly resolved.
- [ ] Exercise Stop/Play, reset/reinitialization and shutdown. Look for duplicate
  callbacks, residual tasks, stale controls or camera/window cleanup errors.
- [ ] Check RGB-D Capture and replay when diagnostics are needed; ground truth
  should not offer frame capture. See [GUI](gui.md) and
  [control diagnostics](control_diagnostic.md).
- [ ] Review any motion/release/retreat failure: FAILED/TIMEOUT must remain a
  failure; confirmed release stays Free. Inspect Place's separate outcome fields
  described in [manipulation](manipulation.md#place-and-intentional-release).
  Deterministic failure cases are exercised offline by the command below. If a
  native failure case does not occur, record it as not exercised natively;
  do not deliberately block the arm or claim that unit doubles prove physics.

```bash
python3 -m unittest tests.unit.test_place_modes tests.unit.test_execution_runtime
```

Stop immediately and investigate if shaking, contortion or unintended contact
returns. Do not mask it by increasing tolerances or accepting inaccurate results.
The [control diagnostic](control_diagnostic.md) gives journal commands; ZIP the
complete output directory, including partial logs/journals after interruption.

## Phase 4C release readiness

Base: clean merged main **`1f53ed8`**, Phase 4B PR #6. The operator reported Phase
4B native tests and manual checks passing before merging. Those results establish
the starting point, not native validation of Phase 4C.

Changes are limited to Place result reporting, honest Home descriptions/tooltip
and unused skill metadata removal. Robot/source factories, source/WorldModel
contracts, controller feedback/hold, automatic held-state handling, perception,
planning mathematics, BT policy, lifecycle and dependency/launch paths remain.
Home's target and trajectory are unchanged. Place failures stop both BTs before
final Home; release mode still intentionally skips landing verification.

| Measure | Baseline | Phase 4C |
| --- | ---: | ---: |
| Production Python, package + GUI, comments/blanks included | 5324 lines / 76 files | 5257 lines / 76 files |
| Active Markdown files under `docs/` | 14 | 8 |
| Discovered unit tests | 286 PASS | 297 PASS |
| Standalone offline modules | 8 PASS | 8 PASS |
| Architecture/syntax | PASS | PASS |
| Native two + six scenarios and loss diagnostic | Prior-phase operator report | **Not run in cloud; pending workstation** |

Final cloud command: `.venv/bin/python scripts/run_all_regressions.py --unit-only
--artifacts-dir outputs/phase4c-unit`, with all ten runner steps PASS. Artifacts
are local/ignored at `outputs/phase4c-unit/run-n2jt3ivp/summary.json` and neighboring
logs. Local documentation links/anchors, historical Git references and
`git diff --check` also pass. Production Python decreases 67 lines (1.3%);
no large implementation shrink is claimed for this finishing phase.

New/extended checks cover both Place modes, release/retreat failures and timeouts,
verification evidence, Free versus retained attachment, no automatic task
recovery, Home named/unknown poses, failure propagation, carrying loss and actual
HomeSkill recovery-tree execution. Existing 286 tests, eight standalone modules,
eight native scenarios and all accuracy gates are retained.

Public migration: custom skills declare `name` rather than `SPEC = SkillSpec(...)`;
unused `SkillSpec` and `SkillRegistry.specs()` are removed. Command names,
`execute_async`, backend/observation contracts and configuration are unchanged.
Place details add explicit stage outcomes; callers must now handle retreat
failure as overall failure, even after successful release.

Implementation files changed: `manipulation/skills/place.py` for reporting,
`runtime/skill.py`, `runtime/registry.py` and all four skill declarations for
metadata removal. `manipulation/backend.py` and Isaac `franka_backend.py` only
change Home documentation/comments; the GUI panel only adds the Home tooltip.
No production module was deleted or moved. Four unit-test modules are extended.

The small registry, placement protocol/types/implementation split, read-only
presentation helpers, operator dialogs and active trace/capture diagnostics were
reviewed and retained. Further merging offers little meaningful reduction here
and risks obscuring already validated boundaries. Duplicate result formatting
remains local where phase-specific failure evidence differs; no generic result
or recovery framework was introduced.

Six historical phase/startup reports and the speculative ROS note are removed
from active docs, with Git recovery references in [history](history.md). Current
schema is in [configuration](configuration.md); behavior and troubleshooting
lessons are consolidated into the durable documents, not another phase report.
Known limits remain: no general collision-free transport, startup-reference
Home, delayed position-only grasp evidence, simple greedy identity association,
no arbitrary-object planning, and no automatic post-release recovery.

Release checklist:

- [x] Final cloud suite and repository/doc consistency checks pass.
- [ ] Both native groups pass with all accuracy gates unchanged.
- [ ] Injected-loss diagnostic passes after actual injection.
- [ ] Both-source manual checklist passes; unexercised native failure paths are recorded honestly.
- [ ] Review the PR and results before merging; do not label/tag a validated release early.

No native Phase 4C validation or release tag is claimed. No additional robot,
ROS/LLM integration, force sensing, planner framework or thesis work was started.
