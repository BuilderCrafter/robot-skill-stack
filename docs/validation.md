# Validation and release baseline

## Supported launch and test workflow

Use the existing Isaac Sim **5.1.0-rc.19 / Python 3.11** installation and
repository-local `.deps` with **NumPy 1.26.4** and **py_trees 2.6.0**. From the
repository root:

```bash
./run_isaac_sim.sh
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/regression
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
| `isaac` | Discovery, perception-manipulation, direct motion/cancellation and reactive motion/lifecycle acceptance |
| `regression` | Benchmark, grasp planning, reactive pick, recovery, multi-object and placement feasibility |
| `multiview` | Source contract, variable camera counts, recovery, occlusion, observability and grasp readiness |
| `spatial_mapping` | GT/RGB-D occupancy, robot filtering, reliability, automatic runtime manipulation/lifecycle and independent camera sources |

Omit group selection to run all checks. Each selected scenario runs once. Failed
exit codes, missing result JSON or non-PASS native results fail the run; later
checks still run. Fresh artifact directories prevent stale PASS reuse. With
`--artifacts-dir`, inspect logs, scenario JSON and `summary.json` with actual
exit codes. Without it, native result files are temporary and terminal output
remains visible. Do not treat an open window or a NumPy line as a test pass.

V5 runtime timing is measured separately in fresh native processes, so offline
tests and other GPU jobs do not compete with a timed run:

```bash
./run_isaac_python.sh scripts/diagnose_isaac_runtime.py \
  --pipeline runtime --mode combined --objects both --seconds 20 \
  --result outputs/v5-combined/result.json --artifacts-dir outputs/v5-combined
```

Repeat with `--mode control`, `semantic` and `mapping`, changing output paths.
All modes keep the same camera/render/filter resources and use ordinary async
MoveToPose/Home. The compact [V5 protocol](assets/spatial_mapping_v5/acceptance_protocol.json)
defines workstation-derived acceptance limits. The historical `--pipeline sync`
path reconstructs synchronous diagnostic scheduling before any measured update;
its mapping callback is explicitly synthetic because V4 had no runtime mapping.
Pick/Place and source/lifecycle tests use actual automatic bootstrap services.
In that V5 diagnostic, no occupancy information changes a command or trajectory.

[Motion V1](motion_planning.md) uses the same unmodified combined diagnostic
before and after direct-strategy extraction. Its additional native case selects
`[motion] strategy = "direct"`, verifies actual Pick/Home/Place with concurrent
perception/mapping, and checks arm ownership, cancellation and final hold. Run
the offline suite and all native groups sequentially, retaining separate artifacts:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only --artifacts-dir outputs/motion-v1-unit
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping --artifacts-dir outputs/motion-v1-mapping
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview --artifacts-dir outputs/motion-v1-multiview
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/motion-v1-isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/motion-v1-regression
```

The Motion V1 checkpoint passes 721 discovered unit tests, eight standalone
modules, architecture/syntax checks and all 22 native scenarios. See the
[verification record](assets/motion_planning_v1/verification.json) and
[matched motion comparison](assets/motion_planning_v1/evaluation.json).

Motion V2 adds a separately selected `reactive` strategy. Its matched native
diagnostic uses real physical fixtures and independent PhysX contacts plus
cooked robot hull clearance. Run one fresh process at a time:

```bash
./run_isaac_python.sh scripts/diagnose_reactive_motion.py \
  --strategy reactive --source ground_truth --scenario static --mapping-hz 2 \
  --result outputs/reactive-static/result.json --artifacts-dir outputs/reactive-static
```

Repeat with `--strategy direct` and a fresh output path. The complete cases,
fixed setup settling, source cadences and acceptance rules are declared in the
[V2 protocol](assets/motion_planning_v2/acceptance_protocol.json). RGB-D matched
experiments explicitly request `--mapping-hz 2.5`; the library default remains
2 Hz, with the original one-second freshness budget. `status = PASS` in this
diagnostic means a valid measurement. Consult `classification` and
`avoidance_success`: contact, timeout, blocked motion and source failure are
separate outcomes and never successful avoidance. The ordinary reactive native
acceptance case also exercises Home, cancellation, source/map/worker failure,
Stop/Play rejection, clean runtime reinitialization and shutdown cleanup.

V2 keeps direct as the default and preserves direct Pick/Place. Reactive
Pick/Place and carried-object transport explicitly return unsupported until an
intended-contact and payload model exists. See the [V2 evaluation](assets/motion_planning_v2/evaluation.json)
and [verification record](assets/motion_planning_v2/verification.json) for actual
outcomes, source failures and regression counts.

The final V2 run passes 787 discovered unit tests, all eight standalone modules,
architecture/syntax and all 23 native scenarios (7 mapping, 6 multi-view, 4 Isaac,
6 manipulation regressions). Its separate obstacle evaluation remains a partial
failure: the RGB-D obstacle cases did not establish reliable avoidance, and the
partial-observation reactive route could not be measured after two setup source
refusals. Passing preservation regressions do not change those outcomes.

Motion V2b retains those records and adds worker-stage evidence, contact severity
and source-availability comparisons. For an ordinary measured run:

```bash
./run_isaac_python.sh scripts/diagnose_reactive_motion.py \
  --strategy reactive --source rgbd --scenario partial --mapping-hz 2 \
  --contact-metrics --mapping-audit-dir outputs/v2b-partial/mapping-audit \
  --result outputs/v2b-partial/result.json --artifacts-dir outputs/v2b-partial
```

`--contact-metrics` observes contact impulses, physics-step timing, COM-correct
body velocity and joint feedback without commanding the robot. Strict V2
zero-contact success remains separate from goal completion and raw severity.
No harmless-contact threshold or force-controlled pushing is enabled.
The audit directory records small worker-stage timing rows. Adding
`--audit-capture-inputs` records bounded original worker inputs and dense arrays
for offline evidence analysis, and adds substantial disk/serialization work;
those full-capture runs are excluded from timing and availability claims.
Keep fresh output paths and one native process at a time.

Offline analysis and the installed-policy composition probe use the same wrapper:

```bash
./run_isaac_python.sh scripts/audit_rgbd_obstacle_evidence.py \
  --capture-dir outputs/v2b-partial --result outputs/v2b-partial/evidence.json
./run_isaac_python.sh scripts/summarize_reactive_proxy_coverage.py \
  --case-dir outputs/v2b-partial --result outputs/v2b-partial/coverage.json
./run_isaac_python.sh scripts/probe_rmpflow_obstacle_composition.py \
  --result outputs/v2b-composition.json
```

The first command requires a full worker-input capture. The second uses copied
physical actor poses and actual numeric registration IDs independently of source
occupancy. The third uses public Lula APIs on CPU without starting Kit or moving
the robot; its natural-policy force/metric terms are not physical contact forces.
See the [V2b protocol](assets/motion_planning_v2b/acceptance_protocol.json),
[physical evaluation](assets/motion_planning_v2b/evaluation.json) and
[source timing](assets/motion_planning_v2b/source_timing.json) for the complete
comparison, failed experiments and remaining limitations.

The final V2b preservation run passes **811 discovered unit tests**, all eight
standalone modules, architecture/syntax, and **23 native scenarios**: seven
spatial-mapping, six multi-view, four Isaac acceptance and six manipulation
regressions. The 40 separate comparison attempts are valid measurements, not
40 avoidance successes. All 22 reactive attempts had zero source failures; the
physical evaluation remains `PARTIAL_FAILURE` because some goals stayed blocked
and the moving case still had impact. Narrow-passage reactive reports were
contact-offset proximity with positive clearance and zero normal impulse,
while its goal still timed out.

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
