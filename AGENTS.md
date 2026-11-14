# Robot Skill Stack — Agent Instructions

## Objective

This repository is a university master's thesis project implementing
modular robot manipulation, perception and motion planning in Isaac Sim.

Prioritize a reliable, reproducible final demonstration over architectural
expansion or speculative optimization.

The existing implementation includes:
- Franka Panda manipulation skills: MoveToPose, Home, Pick and Place.
- Behavior-tree orchestration and asynchronous execution.
- RGB-D perception and multi-camera object detection.
- Spatial occupancy mapping with robot self-filtering.
- Direct, reactive and static global motion strategies.
- Lula RRT planning with optional frozen RGB-D obstacle observations.

## Environment and portability

The original university workstation is the final demonstration target:
- Isaac Sim 5.1.0-rc.19, Python 3.11.
- NumPy 1.26.4 and py_trees 2.6.0.
- Repository-local `.deps` for Python dependencies.
- Existing `run_isaac_python.sh` and `run_isaac_sim.sh` launchers.
- The full GUI must remain functional.

Current cloud development uses the official Isaac Sim 5.1.0 Docker image.

Rules:
- Never introduce `--no-ros-env`; the original launcher does not support it.
- Do not modify established launch scripts for cloud-only convenience.
- Use ISAAC_SIM and environment variables for cloud-specific paths.
- Do not modify global SSH configuration.
- Do not introduce mandatory cloud services or dependencies.
- Do not increase minimum GPU, CPU or RAM requirements.
- Preserve the existing cameras, resolution and resource budgets.
- Do not modify Isaac Sim SDK installations or bundled files.

All committed changes must remain portable to the original workstation.

## Architecture

Preserve the established separation:

GUI / Behavior Trees
  -> RobotRuntime
  -> Skills
  -> ManipulationBackend
  -> MotionStrategy
  -> Isaac Sim robot integration

Semantic world modeling, perception and spatial occupancy mapping remain
separate subsystems.

Keep Isaac-specific dependencies out of generic domain logic.

Preserve existing public APIs, factories, lifecycle handling, asynchronous
execution, cancellation and failure propagation.

Prefer compact, idiomatic Python. Avoid unnecessary abstractions, duplicated
implementations, broad refactors and excessive comments.

## Motion planning and collision safety

The static ground-truth global planner is the validated fallback.
Frozen RGB-D global planning is experimental and opt-in.

Do not:
- Replace Lula RRT or introduce a new planning framework.
- Introduce hybrid planning without explicit authorization.
- Disable collision checking or weaken safety thresholds.
- Treat UNKNOWN occupancy as verified FREE.
- Silently discard observed occupied evidence.
- Import simulator ground-truth obstacle geometry into RGB-D planning.
- Use hidden fixture coordinates, dimensions or names as perception input.
- Execute trajectories independently identified as physically unsafe.
- Silently fall back from RGB-D to GT or direct planning.
- Modify fixture hashes or expected results just to pass tests.
- Claim continuous collision safety from sampled measurements.

Preserve joint limits, self-collision checks, trajectory validation,
tracking limits, terminal settling and cancellation behavior.

Do not change Pick/Place semantics or assume collision-safe payload carrying.

## Resource and experiment limits

Keep the established two-camera 640x480 configuration.
Preserve the existing acquisition rates, ray budget and voxel resolution.
The normal frozen RGB-D collision-proxy budget is 64; 128 is diagnostic only.

Do not increase scene complexity, mapping resolution or GPU load merely
because the cloud development machine has more resources.

Use bounded experiments. Do not launch concurrent Isaac Sim instances.
Never run unbounded parameter searches or long benchmark batches without
an explicit task requirement.

## Development workflow

Before editing:
1. Inspect the current Git branch and working tree.
2. Read the relevant code, existing tests and research documentation.
3. Identify the smallest change that addresses the demonstrated problem.

During implementation:
- Preserve existing user changes.
- Make incremental, independently testable modifications.
- Add focused regression tests.
- Separate experimental measurements from confirmed functionality.
- Preserve existing GT-based planning and direct manipulation.
- Avoid modifying unrelated modules.

After implementation:
- Run relevant offline tests first.
- Run native Isaac benchmarks only when necessary.
- Check result JSON, actual motion, contact reports and goal accuracy.
- Record executed commands, results, limitations and outstanding failures.
- Commit coherent changes locally when validated.

Never reset, clean, force-push, rewrite history or merge to main without
explicit authorization. Do not push to GitHub without authorization.

Do not overwrite historical benchmark artifacts.

## Known project state

V4.3 demonstrated obstacle-induced static GT global RRT detours.

V5 integrated frozen RGB-D occupancy but has not established reliable
obstacle-induced RGB-D avoidance.

Known V5 problems:
- Camera-observed floor cuboids can conflict with fixed-base collision checks.
- Partial obstacle surfaces do not represent unseen physical volume.
- Bounded obstacle proxies can omit observed occupied evidence.

Previous cloud validation:
- Isaac Sim GPU startup succeeded.
- V4 empty-scene global RRT executed with target reached and zero contacts.
- V5 GT static benchmark completed successfully; detailed outcome was
  not yet independently reviewed.
- 1372/1373 unit tests passed. The remaining historical multiview
  fixture-hash assertion must not be changed merely to obtain a pass.

Consult:
- docs/motion_planning_v4_3_detour_verification.md
- docs/motion_planning_v5_rgbd_global.md

## Final priority

Finish a working thesis demonstration with existing functionality.

Prefer a demonstrated, reliable static GT global planner over an unverified
RGB-D obstacle avoidance claim.

Preserve GUI compatibility and portability to the university workstation.
