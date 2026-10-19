# History and retired implementations

Current behavior is authoritative in README, architecture, configuration,
manipulation, perception and validation. Phase-by-phase implementation records
were consolidated during Phase 4C; their source remains in Git, not an active
archive package.

## Retired experiments

Tag `archive/perception-v2-pre-phase-2` preserves commit `a3dcb95` before pruning:
YOLO V2 workers/weights/training/data tools, robot mesh/self-filtering, the
legacy cube planner/classifier and the Phase 1 baseline. These are unsupported;
current perception is geometry-based V1 for all three supported primitives.

```bash
git fetch origin tag archive/perception-v2-pre-phase-2
git show archive/perception-v2-pre-phase-2:docs/phase_1_baseline.md
git archive archive/perception-v2-pre-phase-2 > /tmp/robot-skill-stack-pre-phase-2.tar
```

Extract separately if historical code is needed; do not restore experiments
into the supported runtime merely to inspect them.

## Simplification records

Merged main `1f53ed8` (Phase 4B, PR #6) preserves the reports removed from the
active documentation. For example:

```bash
git show 1f53ed8:docs/phase_2_pruning.md
git show 1f53ed8:docs/isaac_runner_startup.md
git show 1f53ed8:docs/phase_3a_execution.md
git show 1f53ed8:docs/phase_3b_held_state.md
git show 1f53ed8:docs/phase_4a_world_model.md
git show 1f53ed8:docs/phase_4b_config_adapters_ui.md
```

Phase 3A merged at `6f696d0`; its full earlier troubleshooting chronology,
including failed runs, is available there:

```bash
git show 6f696d0:docs/phase_3a_regression_repairs.md
git show 6f696d0:docs/control_diagnostic.md
```

Important retained lessons: native startup uses direct scripts with NumPy
pre-import; frame-derived joint velocity avoids contact-feedback amplification;
arm motion exits retain position targets with zero desired velocity; gripper
limits come from physical DOF properties; source timestamps and identity hints
have distinct roles. See [control diagnostics](control_diagnostic.md),
[manipulation](manipulation.md) and [perception](perception.md) for current details.

Phase 4A merged at `1d9141f` (PR #5), after its synthetic oversized-cube fixture
was repaired at `9ac2e04`: a never-observed record could expire between Kit task
submission and execution. Install/inspect/remove such a fixture inside the same
coroutine; do not weaken production expiry or planner checks to keep it alive.
Phase 4B subsequently passed operator-reported native/manual validation and was
merged at `1f53ed8`. Earlier test counts and measurements describe their own
checkpoints, not validation of the current branch. Current release status is in
[validation](validation.md#phase-4c-release-readiness).
