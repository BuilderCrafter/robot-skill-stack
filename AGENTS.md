# Robot Skill Stack — Agent Instructions

## Objective

University master's thesis project: modular Franka manipulation, perception and
motion planning in Isaac Sim. Prioritize a reproducible demonstration and faithful
functional integration over architectural expansion or speculative optimization.

Current production includes MoveToPose/Home/Pick/Place, async runtime and BTs,
single/multi-camera RGB-D perception, semantic WorldModel, occupancy mapping,
robot self-filtering, bounded observation workers, direct RMPflow, static Lula RRT,
frozen RGB-D global planning and opt-in measured-box reconstruction.
The optional reactive obstacle-avoidance strategy is retired. Ordinary direct
RMPflow, grasp verification and manipulation recovery remain supported.

## Environment and portability

Original demonstration target:
- Isaac Sim 5.1.0-rc.19, Python 3.11.
- NumPy 1.26.4 and py_trees 2.6.0 from the existing repo-local `.deps` workflow.
- Existing `run_isaac_python.sh`, `run_isaac_sim.sh`, USD scenes and full GUI.

Cloud tests use the already-configured official Isaac Sim 5.1.0 container.
Never reinstall/reconfigure the SDK or change bundled files for convenience.
Never add `--no-ros-env`; the original launcher does not support it.
Use `ISAAC_SIM`/environment paths for cloud differences. Do not change global SSH,
Docker/network configuration, or introduce mandatory cloud services/dependencies.
Do not increase CPU/GPU/RAM requirements. Preserve original two 640×480 cameras,
acquisition rates, 1,024 shared mapping rays and 10 mm nominal voxels.
Never launch concurrent Isaac instances or run unbounded experiment batches.

## Architecture

Preserve GUI/BT → RobotRuntime → skills → ManipulationBackend → MotionStrategy.
Semantic WorldModel, perception and Spatial Mapping remain separate subsystems.
Keep Isaac imports out of generic common/runtime/world/manipulation/orchestration/
presentation code. Bootstrap owns SDK resources; workers contain only SDK-free
computation. Preserve public contracts, async ownership, cancellation, error
propagation, immutable snapshots and generation/revision barriers.

Global source selection is explicit. Static GT is independently selectable and
does not require mapping. Frozen RGB-D requires the enhanced collision model,
enabled RGB-D mapping and automatic robot filtering. `occupied` and `reconstructed`
remain available; reconstruction runs in the existing mapping worker. Do not add
a worker, planner, world model or alternate motion executor during routine fixes.
Do not resurrect the removed synchronous perception capture/mapping paths.

## Motion and safety

Direct RMPflow is the default and complete manipulation baseline. Static GT global
RRT is the validated primary-demo option. Reconstructed RGB-D is experimental and
opt-in; nominal partial detours have executed, but general reliability is unproven.

Do not:
- Replace Lula RRT, introduce hybrid planning or silently fall back to GT/direct.
- Disable collision checks, weaken limits/qualification/freshness, discard selected
  occupied evidence or interpret UNKNOWN/OUT_OF_BOUNDS as verified FREE.
- Supply GT obstacle geometry, hidden fixture identities/dimensions or reference
  joint goals to the selected RGB-D planner.
- Guess unsupported hidden geometry bounds or search exposures for favorable fits.
- Drop protected floor evidence, bypass the base guard or raise the 64 camera-proxy
  limit to admit a rejected reconstructed world.
- Execute a trajectory independently identified as physically unsafe.
- Change fixture hashes/expected safety behavior just to make tests pass.
- Claim continuous swept safety from sampled observations or zero contact reports.

Preserve six-face qualification, calibrated edge/visibility evidence, robot
exclusion, FREE contradiction checks and eight-corner subsumption. Retain original
exposure/calibration provenance and strict <1 s source freshness. Cooperative
reconstructed-source admission keeps its existing five-second bound and first-new-
accepted-publication policy; a rejected source is not silently retried.

Keep enhanced Franka geometry/finger contracts, joint/self-collision checks,
trajectory validation, tracking bounds, terminal settling/hold, solver restoration,
cancellation and shutdown. Do not change Pick/Place semantics or assume global
payload carrying/intended-contact motion is supported. Home is not inherently safe.

Known RGB-D limitations include conservative floor/base rejection, incomplete
surface qualification, 65-proxy sparse worlds, limited mapping throughput/narrow
freshness margins and possible refusal of persistent-session return motion.

## Development workflow

Before edits, inspect branches/remotes/worktrees, preserve user changes and read
relevant production code, tests and current documentation. Prefer the smallest
change needed. Do not refactor unrelated production code during integration.

Run offline regressions first:

```sh
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
```

This includes architecture/syntax. Run bounded native cases only as needed, one
Kit at a time. `--group global` retains GT and nominal reconstructed RGB-D with an
independent pre-execution physical veto. Direct manipulation and Stop/Play/worker
shutdown have separate retained scenarios; see `docs/validation.md`.

Inspect result JSON, actual commands/displacement, endpoint accuracy, contacts,
source identity and cleanup. Startup or process exit alone does not prove motion.
Preserve failures and original inputs; distinguish a safe refusal from successful
execution and from a new regression. Never retry/tune until a favorable result.
Validate GUI/bootstrap lifecycle and original workstation portability.

Commit coherent validated changes. Do not reset/clean/force-push/rewrite history,
merge main or publish without authorization. Preserve development/recovery refs
and historical artifacts. Explicit session authorization for integration/push
applies only after the requested validation gates pass.

## Current references

- `README.md`, `docs/architecture.md`, `docs/configuration.md`.
- `docs/motion_planning.md`, `docs/spatial_mapping.md`, `docs/validation.md`.
- `docs/integration-rgbd-global-rrt.md` for this transfer and measured checks.

Full historical research, benchmarks and large evidence remain permanently on
`feature/rgbd-rrt-autonomous`; they are not main/runtime dependencies. Cleanup,
module consolidation, typing changes and GUI redesign require a separate task.
