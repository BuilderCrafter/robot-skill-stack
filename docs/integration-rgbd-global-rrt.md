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

## Final validation and Git gate

**Final status: PASS — functional integration validated.**

The initial integrated host runner passes 855 discovered tests / 78.187 s,
architecture/syntax and all eight standalone scripts. Targeted Isaac-Python
checks pass 256 tests / 6.999 s. After completing fixture adaptation, the exact
requested runner under Isaac **Python 3.11** passes **858 tests / 78.812 s**,
architecture/syntax and all eight standalone scripts. The additional cases cover
both native source profiles, contact capture and selected-input pinning.

| Native gate | Source baseline | Integrated result |
|---|---|---|
| Direct Pick/Home/Place/cancel/recovery | 22/22 checks PASS | 22/22 PASS |
| Enhanced GT global motion, mapping disabled | Reached, 0.990 mm, zero contacts | Reached, **0.996 mm**, 14/14 checks PASS, zero contacts |
| Nominal reconstructed RGB-D partial | Reached, 0.875 mm, zero contacts | Reached, **0.875 mm**, 19/19 checks PASS, zero contacts |
| Stop/Play, failure recovery and shutdown | 17/17 checks PASS | 17/17 PASS |

Integrated RGB-D retains ordinary selected Lula IK, **1.057749 rad** obstacle-
induced detour, **110 observed actions**, two qualified boxes and **24 camera
proxies** preserving all original 64 selected cuboids. Source age is **962.149 ms**
with the original strict <1 s requirement: only **37.851 ms** headroom. The
independent eleven-body cooked-geometry evaluator allows the selected route and
spline; the same-endpoint straight control is physically vetoed. Original camera
input is pinned at planner entry and archived unchanged. Offline decoding verifies
camera-only input, generation, original timestamps/tokens, both 640×480 frames and
exact intrinsics/extrinsics. No GT obstacle/joint-goal input enters selected planning.
A closed-source probe after success refuses without commands or fallback; the test
interlock independently prohibits arm writes during that probe.

All native attempts remain recorded: four source cases and seven integration
attempts. Three initial integration receipts fail for confirmed **test-harness**
issues: missing GT semantic root; an extracted contact helper's stale module alias;
and delayed archival lookup after an otherwise successful RGB-D motion. Each is
preserved, diagnosed, corrected narrowly and regression-tested. No geometry,
threshold, camera, seed, freshness or production planner change was made. Both
failed GT attempts stop before selected motion and close cleanly. The earlier
RGB-D receipt retains its successful actual motion but remains overall FAIL for
its missing archive; it is not relabeled. The corrected repeat passes every flag.

The first complete SDK-Python unit run found one remaining Git-dependent direct-
loop oracle (852 tests, one class-setup error). Its original method is now a small
hash-checked fixture, AST-equivalent to the original frozen implementation; all
seven equivalence cases and the final full SDK runner pass. No Git/SDK package
was installed to hide that test dependency.

GUI/bootstrap tests use the real extension logic with a test UI, and native cases
initialize the same supported production bootstrap. This is headless integration
coverage; a final university workstation visual/operator review remains.

[Compact validation receipt](assets/integration-rgbd-global-rrt/validation.json)
binds exact commands, native commits/results, test logs, failures, preservation and
original-input hashes. [External raw manifest](assets/integration-rgbd-global-rrt/raw-manifest.json)
contains 162 hashes at the validation checkpoint. Raw cameras, traces and logs stay
outside Git under `outputs/main-integration-20261122`; worktrees are excluded from
that manifest. The source development checkout/branch remain intact.

Implementation checkpoints: `699d46b` (selective integration), `b9afe4c` (profile
validation/command interlock), `696664b` (contact-helper alias), `bf230c0` (input
pinning), `90ca5b4` (Git-free direct oracle). Only configuration, backend factory
and observation-product selection differ in production from `cdd2be9`, plus removal
of the two reactive modules: **4 added / 498 removed lines** relative to source.
All six tracked scene/launcher/requirements files are byte-identical to both
original main and source. Enhanced YAML/JSON assets are the current source versions.

Main publication uses a normal fast-forward of the final validated integration
commit, after rechecking remote main and running final main architecture/targeted
sanity checks. Exact resulting main SHA and remote push verification are recorded
in the final handover, avoiding a self-referential commit hash in this file.
No force push, history rewrite, source-branch deletion or research-history merge.

## Preservation and environment

Verification preserves 48 original published research files, 16 static/v43 inputs,
24/21/21/29 old native files, 242 integration + 112 performance + 131 admission
artifacts and four prior finalization reports. Original V5 local/remote remains
`71c07f815410a308dc7ea176148c017055c7fd7a`; recovery remains
`edc1b467c6bf03c192d0d9dccce2b538309a024f`. The detached source worktree is clean;
the original source checkout remains at its untouched initial `20a1207`, with
remote source `cdd2be9` validated separately.

Original `isaac51` stopped/exit-137 state is restored with no GPU compute process
or stream. Config/HostConfig/Mounts hash remains
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No launcher, USD scene, SDK, global SSH, Docker/network, dependency or resource-
budget reconfiguration. Native scenarios ran exclusively, one Kit at a time.

## Remaining limitations and next task

RGB-D remains an opt-in demo candidate: throughput/freshness margins, floor/base
rejection, missing surfaces/hidden bounds, 65-proxy sparse worlds and persistent
return refusal remain. No fallback, source-age change, geometry threshold tuning,
proxy-budget increase or altered executor is introduced. Sampled physical-veto
and contact evidence does not certify continuous safety.

Future cleanup may address legacy naming/diagnostics, fixture/module consolidation,
unused production helpers and stronger annotations. No such refactor, new motion
feature or GUI redesign is part of this transfer.
