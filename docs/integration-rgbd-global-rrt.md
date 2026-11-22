# RGB-D global RRT integration

Integration date: 22 November 2026. Source checkpoint
`cdd2be9bfa6e95295c83b41d69819579b6dcde94`; main base
`34baab2bb2028274c525d8ca9f34164fb65a9c77`. Remote verification confirmed the
source 99 commits ahead, zero behind. The original checkout was clean at the older
`20a1207`; separate detached-source and integration worktrees preserve it.

This is a selective latest-file transfer on `integration/rgbd-global-rrt`, not a
cherry-pick/replay of research history. Production is copied faithfully, including
current source changes after the earlier hardening checkpoint. Beyond exclusion
of two retired reactive modules, only reactive selection in configuration,
backend composition and observation-product enablement is removed.

## Baseline before transfer

Full source offline runner: 1,296 tests, 1 failure and 20 errors; architecture and
all eight standalone checks pass. Recorded failures include stale research imports
of the relocated Franka joint mapper, research AST/history expectations, a removed
capture diagnostic, two mask-helper shadowing mistakes, an incorrect standard-JSON
serialization check for immutable diagnostics, and worktree output-symlink path
assumptions. These are preserved, not silently called a passing source suite.

Source native acceptance passes four bounded, exclusive processes:

- Direct ordinary Pick/held Home/Place/released Home, cancellation, final hold and
  recovery: all 22 checks pass.
- Enhanced static GT global motion: goal reached, 0.990 mm position error,
  zero reported contacts, closed runtime/strategy.
- Nominal reconstructed RGB-D partial: ordinary selected IK, 0.875 mm position
  error, zero contacts, genuine detour, original source age 744.941 ms. Both
  reconstructed and GT controls pass the independent physical evaluator;
  occupied-only/support-only controls are vetoed and never executed.
- Mapping lifecycle: all 17 checks pass, including Stop/Play generation change,
  failure/recovery, no late publication, distinct reinitialization and closure.

Only a process-local external harness compatibility alias redirects the historical
research import to the current Franka adapter; production source files are
unchanged. A headless test override avoids changing the SDK/environment.

## Transfer scope

All current production modules and required enhanced collision YAML/JSON transfer.
Direct manipulation, semantics/held-state policy, asynchronous workers, shared
frames, self-filtering, occupancy/cuboids, frozen sources, reconstruction, Lula,
validation/execution, GUI and lifecycle remain. Existing scenes, launchers and
requirements are unchanged.

ReactiveRmpflowStrategy, its obstacle registry and dedicated tests are excluded;
`reactive` is rejected before SDK construction. Ordinary direct RMPflow, generic
occupancy obstacle extraction and object-motion Pick rechecks remain.

No `research/`, global experiment history, research-only driver/plot/fixture-search
suite or large new experimental report/evidence is copied. Existing main perception
illustrations remain. Useful production tests are retained; small analytic and
physical-reference fixtures live under tests/support and tests/fixtures. Frozen
semantic numeric oracles no longer require Git history or the deleted capture API.
A minimal guarded global acceptance path retains original physical-veto algorithms,
allowances and fixed fixture hashes, without the historical comparison framework.

Retained source test defects are corrected only in tests: mask-helper shadowing,
actual immutable worker serialization, removed-diagnostic references and retired
native inventories. Geometry/safety expectations and original scene/model hashes
are not weakened. Current-facing documentation describes direct default, static
GT, both RGB-D representations, enhanced geometry, limitations and retained checks.

## Validation gate

The integrated offline runner passes 855 discovered tests, architecture/syntax
and all eight standalone scripts. Integrated native checks and final main/remote
sanity are recorded below when completed. Main remains gated until actual native
motion, independent veto and lifecycle acceptance pass.

Raw receipts, preserved failure logs, commands and native artifacts are stored
outside Git under `outputs/main-integration-20261122`. Compact final validation
receipts will accompany this report. Full historical evidence remains on the
preserved development branch.

## Remaining limitations and next task

RGB-D remains an opt-in demo candidate: throughput/freshness margins, floor/base
rejection, missing surfaces/hidden bounds, 65-proxy sparse worlds and persistent
return refusal remain. No fallback, source-age change, geometry threshold tuning,
proxy-budget increase or altered executor is introduced. Sampled physical-veto
and contact evidence does not certify continuous safety.

Future cleanup may address legacy naming/diagnostics, fixture/module consolidation,
unused production helpers and stronger annotations. No such refactor, new motion
feature or GUI redesign is part of this transfer.
