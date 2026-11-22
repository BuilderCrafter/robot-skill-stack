# Validation

Use the unchanged launchers and existing Isaac Sim installation. University target:
Isaac Sim 5.1.0-rc.19, Python 3.11, NumPy 1.26.4 and py_trees 2.6.0. Do not add
`--no-ros-env`, change SDK files or increase the camera/mapping budgets.

## Offline checks

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
```

This runs architecture/syntax checking, discovered production unit tests and eight
standalone ordinary manipulation/perception/runtime checks. Plain Python 3.11/3.12
with `requirements.txt` is also supported for these SDK-free checks. Independent
physical-reference unit tests additionally use SciPy when available; the native
acceptance path uses the existing Isaac scientific environment. Production adds
no dependency. Tests requiring native SDK construction belong to native groups.

Coverage includes direct/backend delegation, skills and held state, BT/runtime
ownership, global Lula/joint routes/static world/enhanced model/solver scope,
frozen RGB-D assembly and admission, source timestamps and FREE evidence,
reconstruction, self-filtering, occupancy/obstacle extraction, process-worker
immutability/isolation, observation lifecycle and GUI/bootstrap behavior.

## Bounded native checks

Run one Isaac process at a time and use fresh output directories:

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --group global --artifacts-dir outputs/global
./run_isaac_python.sh scripts/run_all_regressions.py --group spatial_mapping --artifacts-dir outputs/mapping
./run_isaac_python.sh scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/isaac
./run_isaac_python.sh scripts/run_all_regressions.py --group regression --artifacts-dir outputs/manipulation
```

| Group | Retained coverage |
|---|---|
| `global` | Enhanced GT motion without mapping, then nominal partial reconstructed RGB-D with actual mapping and a pre-execution independent physical veto |
| `spatial_mapping` | Normal asynchronous runtime, source matrix, Stop/Play, failure/recovery, reinitialization and shutdown |
| `isaac` | Perception discovery/manipulation and explicit direct Pick/Home/Place/cancellation |
| `regression` | Ordinary manipulation benchmark, grasp planning, reactive object-motion Pick rechecks, recovery, multi-object and placement feasibility |
| `multiview` | Camera source contract with the retained multi-view fixture |

The `test_reactive_pick.py` name refers to object-motion rechecks in ordinary
manipulation. It does not select the retired reactive obstacle-avoidance strategy.
Default `all` runs the retained groups sequentially; integration verification
uses the bounded cases above, not historical experiment matrices.

Individual headless acceptance commands:

```bash
./run_isaac_python.sh tests/multiview/test_final_native.py --headless \
  --result outputs/NEW-native-final/result.json --artifacts-dir outputs/NEW-native-final
./run_isaac_python.sh tests/isaac/test_motion_strategy.py --headless \
  --result outputs/NEW-direct/result.json --artifacts-dir outputs/NEW-direct
./run_isaac_python.sh tests/isaac/test_global_planning.py --source static \
  --result outputs/NEW-gt/result.json --artifacts-dir outputs/NEW-gt
./run_isaac_python.sh tests/isaac/test_global_planning.py --source frozen_rgbd \
  --result outputs/NEW-rgbd/result.json --artifacts-dir outputs/NEW-rgbd
./run_isaac_python.sh tests/spatial_mapping/test_runtime_integration.py --headless --lifecycle-only \
  --result outputs/NEW-lifecycle/result.json --artifacts-dir outputs/NEW-lifecycle
```

The final-scene multiview check uses native cameras, the production asynchronous
runtime, a planner-selected pre-grasp approach and bounded lifecycle controls.
Read its [measured results](multiview-native-validation.md): geometry and lifecycle
pass, while stale visibility intervals remain a demo blocker.

Global acceptance copies only frozen fixture data and the necessary independent
numeric/cooked-geometry reference helpers. It does not import research scripts.
Each selected plan uses normal production source capture and IK. Actual PhysX
cooked geometry, independent URDF FK, original route/spline sampling, query budgets
and numerical allowances can veto execution before any arm command. Physical
references never enter the RGB-D planning world. The RGB-D check also requires a
meaningful detour and physically obstructed same-endpoint straight control, checks
original source timestamps/selected occupied union/budget, then verifies a closed
source is refused without commands or fallback. The latter is a controlled failure
probe after the successful motion, not an automatic source retry.

## Acceptance evidence

Inspect result JSON, not only process exit status or SDK startup. Require:
actual movement, target position/orientation results, sampled physics coverage,
contacts/proximity and saturation status, planner ownership/closure, source
identity/age/qualification, immutable publication and final shutdown. Capture
failures and safe refusals separately from successful motion. Stop/Play and
cancellation must invalidate late work and leave no stale controller ownership.
GUI unit tests use the real extension callbacks with a test UI; native runtime
construction exercises the same bootstrap. The final workstation GUI display
still needs its normal operator review.

GT remains the primary demo. Reconstructed RGB-D is an opt-in candidate with
limited throughput, narrow freshness margins, qualified-surface/proxy constraints
and possible persistent-return refusal. Do not weaken safety/source checks or
retry exposures until a favorable result appears. Sampled clearance/zero contacts
do not certify continuous safety. See [integration results](integration-rgbd-global-rrt.md).

Historical benchmark drivers, research-only tests and large experimental reports
are excluded from this integration. The preserved development branch retains that
history and its original evidence.
