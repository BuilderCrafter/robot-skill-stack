# Phase 2 pruning report

This report records pruning checkpoint `0c374da`; its line counts and test results
describe that checkpoint. The subsequent direct-script startup correction and
its validation are documented in [isaac_runner_startup.md](isaac_runner_startup.md).

Branch: `refactor/phase-2-pruning`, based on latest `main` at `a3dcb95`.
Phase 1 was merged and its basic live smoke test was reported successful by the
user. Phase 2 simulator validation has not been performed in this cloud machine.

## Removed components and historical access

The pre-cleanup state is preserved by annotated tag
`archive/perception-v2-pre-phase-2`. See [history.md](history.md) for recovery.
No experiment archive directory was added to the active tree.

Removed:

- `world/perception/v2/` and `integrations/vision/`: worker client, serialization,
  learned masks, async inference, fallback/deduplication, robot surface rendering,
  replay and diagnostic paths.
- `tools/perception_v2/` and `tools/perception_v2_data/`: worker server, benchmarking,
  synthetic-data generation/conversion, training, installers and validation files.
- Isaac `sensors/robot_meshes.py` and `robot_self_filter.py`; experiment-only test
  helpers, six experiment test modules, the experimental scene profile, root
  `best.pt`, model placeholder, and dedicated experiment suite runner.
- Legacy `TopDownGraspPlanner`, its export and unused `SIZE_UNKNOWN` grasp reason;
  `CubeGeometryClassifier` and the provider's optional classifier execution path.
- Five milestone wrapper scripts, replaced by `scripts/run_all_regressions.py`.
  The Phase 1 historical report is retained in the preservation tag rather than
  leaving obsolete commands in current documentation.

Factory, config, bootstrap, GUI, capture/replay and runners no longer import or
invoke the experiment. Configuration has no `PerceptionConfig.type`/`.v2` fields;
unsupported legacy type values fail clearly instead of selecting another source.
Worker endpoint, inference, freshness, mask validation, fallback and mesh-filter
options are gone. World-source selection (`perception` / `ground_truth`) remains.

The removed training requirements/installers contained torch, torchvision,
Ultralytics, OpenCV, Pillow, PyYAML and CUDA-wheel setup. None is needed by the
retained runtime or unit suite. Previously installed optional packages were not
uninstalled from the local test environment. Runtime pins remain NumPy 1.26.4 and
py_trees 2.6.0; `requirements.txt` and working GUI launch scripts are unchanged.

## Active Python size

Physical lines including comments/blank lines, measured over tracked `.py` files
at `a3dcb95` and the resulting tree. Ignored environments/outputs are excluded.

| Area | Before | After |
|---|---:|---:|
| Runtime package, including integrations | 6,501 | 4,993 |
| GUI extension | 829 | 808 |
| Scripts | 779 | 254 |
| Main tests/support | 7,567 | 5,654 |
| Experiment tools and their tests | 2,607 | 0 |
| **Total** | **18,283** | **11,709** |
| Python files | 162 | 112 |

Reduction: **6,574 Python lines (36.0%) and 50 Python files**. This is removal of
active implementation/test burden, not relocation into another package.

## Tests removed, migrated and retained

- The six experiment test modules contributed 183 discovery cases; nine more
  snapshot/experiment diagnostic cases were removed from the mixed bootstrap
  test module. The 71 offline toolkit tests were removed with that toolkit.
- The experiment's V1 import-independence guarantee was migrated into
  `test_primitive_depth_pipeline.py`; an unsupported-source-type check was added.
  Shared `primitive_scene.py`, fake UI adapters, result helpers, bootstrap tests
  and camera lifecycle tests remain.
- Grasp fit/rejection, missing pose/geometry, non-graspable objects, unsupported
  hints, Pick error mapping, planner replacement, reactive replanning and async
  execution tests now use `SimplePrimitiveGraspPlanner`. Missing geometry uses
  the retained planner's `NO_FEASIBLE_GRASP` result, not the removed size-only code.
- The old size-ratio cube-classifier assertions were deleted. The classifier-hook
  test now checks the supported geometry label; the primitive fit matrix remains,
  including 136 shape/view/noise subcases and unsupported-geometry rejection.
- Runner tests were rewritten around the single entry point. They cover offline
  dispatch without Isaac, all retained scenario selection, command and reported
  failures, missing result JSON, import boundaries, fresh artifact directories,
  pinned NumPy pre-import, and existing GUI/replay shell behavior.
- All eight standalone unit scripts, both Isaac integration scripts and all six
  manipulation/perception regression scripts remain. No acceptance scenario was
  removed merely because a milestone runner was deleted.

## Validation evidence

Commands ran from the repository root with the existing Python 3.12 `.venv`:

| Check | Before | After experiment removal | After legacy removal | Final |
|---|---|---|---|---|
| `python -m unittest discover -s tests/unit -t .` | 363 PASS | 171 PASS, plus 2 migrated/new checks PASS | 173 PASS | 170 PASS |
| Eight standalone unit scripts | 8 PASS | Unchanged | All 5 affected scripts PASS | All 8 PASS |
| Offline toolkit discovery | 71 PASS | Removed | Removed | Removed |
| Architecture boundary/syntax check | PASS | PASS | PASS | PASS |

The final test count reflects removal of experiment tests and replacement of 18
milestone-runner tests with 15 consolidated-runner tests; it is not a claim that
all earlier experimental functionality remains supported.

The final complete offline run was:

```bash
.venv/bin/python scripts/run_all_regressions.py --unit-only \
    --artifacts-dir outputs/phase_2_validation
```

It executed architecture checks, 170 discovery tests and all eight standalone
scripts, with all ten runner steps PASS. Logs and `summary.json` were retained
under the ignored `outputs/phase_2_validation/run-4lt6inft/` directory.
`pip check`, shell syntax and `git diff --check` also passed.

## Preserved behavior and remaining checks

Production primitive estimation/fitting, grasp planning/clearance, tracking,
WorldModel lifetime rules, manipulation skills/state, Franka motion, and Isaac
lifecycle modules are byte-for-byte unchanged from `a3dcb95`. The GUI's normal
controls and frame capture remain; only experimental diagnostics were removed.
The provider's production geometry path is unchanged after removing the unused
classifier branch. No sphere or cylinder support was removed.

The new runner's workstation groups invoke the existing `.deps` wrapper and load
verified NumPy 1.26.4 before scenario modules import SimulationApp. A fake-launcher
test verifies dispatch/pre-import only. Real Isaac, camera, GPU and manipulation
validation remain unrun here. Use the [workstation checklist](validation.md#manual-gui-smoke-test),
then run both `--group isaac` and `--group regression` on the workstation.

## Intentional losses and Phase 3 deferrals

Intentional losses are learned perception/training, robot mesh self-filtering,
experimental capture formats, legacy cube-planner/classifier APIs, arbitrary
classifier-hook labels, milestone CLI names and automatic tar archives. Current
capture/replay remains supported; old experimental captures require the tag.

Deferred: sync/async consolidation, completing the backend async contract,
GUI/backend coupling, tracking/lifetime simplification, held-state handling after
failed lift, automatic recovery policy, Place success despite retreat failure,
diagnostic-only speed values and initialization-pose Home semantics. No Phase 3
work, core redesign, automatic merge or physical-safety claim is included.
