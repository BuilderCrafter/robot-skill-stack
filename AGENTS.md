# Robot Skill Stack — Agent Instructions

## Project identity and priorities

Robot Skill Stack is a master's thesis prototype and experimental research
application. Treat it as a scientific Python project for a controlled university
demonstration in Isaac Sim. No real-robot deployment is in scope.

Prioritize, in order:

1. Readability and comprehensibility for the thesis author.
2. Simplicity and minimal code.
3. Correct implementation of the intended algorithms.
4. Reproducible experimental results.
5. Essential testing and failure detection.
6. Additional robustness only where a concrete need justifies it.

We accept narrower input support, simpler recovery and explicit exceptions in
unsupported conditions. Do not impose production-grade robustness or generality.

The implemented stack includes MoveToPose/Home/Pick/Place, async runtime and BTs,
RGB-D perception, occupancy mapping, robot filtering, direct RMPflow and Lula RRT.
Direct motion is the default; static GT global planning is the primary global
demo option. Reconstructed RGB-D planning remains experimental and opt-in.
The optional reactive avoidance strategy is retired.

## Simplicity and scope

Follow [the refactoring guidelines](docs/refactoring-guidelines.md). Each module
requires a separate user-authorized task. Branch setup and these documents do not
authorize a module refactor. Finish the authorized module, commit for review and
stop before starting another.

When simplifying, actually remove code: dead functionality, compatibility layers,
low-value wrappers, duplicate representations, excess state and obsolete tracing.
Keep mathematical algorithms visible. Do not relocate complexity or replace each
deleted helper with a new abstraction. A substantial line-count increase is a
warning sign; explain a compelling benefit or reconsider the approach.

Question each defensive check. Remove redundant validation, required-interface
`getattr`/`hasattr` fallbacks, impossible-state guards and elaborate exception
translation when they serve no intended scenario. Natural exceptions and useful
tracebacks are acceptable. Do not build recovery machinery for hypothetical cases.

Use ordinary parameter/return annotations, existing classes, simple collections
and practical NumPy types. Some dynamic typing is fine. Avoid collections of
`TypedDict`s, one-use protocols, complex generics and typing-only abstraction
layers. Strict Pyright compliance is not a cleanup goal. Optional development
tooling can be considered in a later task; do not import the unsuccessful
cleanup's checker configuration, dependencies or typing infrastructure.

Remove diagnostics and their tests when they no longer help explain an active
experiment, failure or result. An old test asserting a field is not a reason to
retain it. Keep representative mathematical, manipulation/planning, camera/mapping,
lifecycle and known-regression tests. Test totals are not quality targets.

## Essential experimental correctness

Keep protections that prevent a concrete invalid result or unintended motion in
our supported setup. In particular:

- Use the correct coordinate conventions, units and robot-joint indices.
- Reject observations from the wrong exposure time or simulation generation;
  retain the provenance needed to interpret the experiment.
- Reject clearly invalid or colliding trajectories, and honor cancellation,
  controller ownership and invalidation before further motion commands.
- Keep RGB-D planning inputs distinct from GT evaluation. Do not silently swap
  sources, fabricate FREE evidence, guess unsupported geometry or bypass a real
  collision/source rejection to obtain a successful run.

These principles do not require preserving every existing related check. Evaluate
checks individually against the intended experiment. Unsupported inputs may fail
with ordinary exceptions. Report failed experiments honestly; sampled checks and
zero reported contacts do not establish continuous or real-robot safety.

Use the existing GUI/BT → RobotRuntime → skills → ManipulationBackend →
MotionStrategy organization as context. Keep Isaac calls in the integration and
bootstrap code, with SDK-free numerical worker computation. Simplify interfaces
with their actual callers; do not add workers, planners or services speculatively.

## Supported environment

Support the demonstrated university setup, not arbitrary robots or deployments:

- Isaac Sim 5.1.0-rc.19, Python 3.11, NumPy 1.26.4 and py_trees 2.6.0.
- Existing `.deps` workflow, launch scripts, USD scenes, GUI and workstation use.
- Existing two 640×480 cameras, acquisition rates, 1,024 shared mapping rays,
  10 mm nominal voxels and declared experimental resource budgets.

Do not reinstall or modify the SDK, alter global SSH/Docker/network configuration,
introduce cloud-specific dependencies or increase hardware requirements. Use
existing environment paths for cloud differences. Never add the unsupported
`--no-ros-env` flag. Run at most one Isaac instance, with bounded experiments.

## Proportionate validation and Git

Documentation-only work needs review and `git diff --check`, with no runtime tests.
For code, choose checks by impact: small static/focused checks for naming or typing,
algorithm tests for local mathematics, focused integration for shared runtime, and
an appropriate native Isaac regression for actual motion changes. A full suite is
a deliberate milestone, not a default after every edit. Do not launch broad native
matrices automatically. Rerun the relevant failed test instead of the whole suite.
Historical validation documents describe available tools and past results, not
mandatory gates or test-count requirements for every cleanup.

Inspect branch ancestry, worktrees and actual callers before edits; preserve user
changes. Commit small, understandable changes within the authorized scope. Do not
reset/clean away work, delete branches, rewrite history, merge main or publish
without authorization. Preserve `feature/rgbd-rrt-autonomous` as the historical
research archive. Do not copy, cherry-pick or merge anything from
`refactor/post-integration-cleanup` into this fresh cleanup branch; the user handles
that old branch separately.

Reference `docs/architecture.md`, `docs/configuration.md`, `docs/motion_planning.md`,
`docs/spatial_mapping.md` and `docs/validation.md` for implementation context.
