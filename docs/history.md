# Historical experiments

Tag `archive/perception-v2-pre-phase-2` preserves commit
`a3dcb95` before Phase 2 pruning, including the YOLO V2 runtime, weights,
robot mesh/self-filtering, offline dataset/training tools, tests, validation
artifacts, and the Phase 1 baseline report. These are no longer supported runtime
components. The current system uses geometry-based V1 for all three primitives.

Recover a file without changing the current checkout:

```bash
git fetch origin tag archive/perception-v2-pre-phase-2
git show archive/perception-v2-pre-phase-2:docs/phase_1_baseline.md
git archive archive/perception-v2-pre-phase-2 > /tmp/robot-skill-stack-pre-phase-2.tar
```

Extract that archive into a separate directory when historical code is needed.
The tag also preserves the legacy cube planner and classifier removed in Phase 2.

## Phase 3A diagnostic history

Merged commit `6f696d0` preserves the full Phase 3A troubleshooting chronology,
including superseded cleanup attempts, failed runs, comparison commands and
initial size counts. Current behaviour and validation are summarized in
[execution](phase_3a_execution.md) and [control diagnostics](control_diagnostic.md).
The duplicate `phase_3a_regression_repairs.md` report was removed during cleanup.

Read the original reports without changing the current checkout:

```bash
git show 6f696d0:docs/phase_3a_regression_repairs.md
git show 6f696d0:docs/control_diagnostic.md
git show 6f696d0:docs/phase_3a_execution.md
```
