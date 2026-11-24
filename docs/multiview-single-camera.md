# Single-camera multiview feasibility

Current configuration and Isaac construction support one or more cameras. The
gate descriptions and enablement recommendation below describe the historical
experiment baseline; its measured viewpoint limitations remain unchanged.

**CONDITIONAL GO:** the existing numerical pipeline and provider work with one
RGB-D camera. Enabling the configuration requires a small, separate change;
matching two-camera recognition through occlusion is not established. Production
configuration and algorithms remain unchanged in this experiment.

Source baseline: `2e8be5a08b23753b8f2be7320a72761cc8a4448f`.
Native evidence: the original six 640×480 static/pre-grasp/home captures under
`outputs/multiview-native-20261122/session-2/`, with unchanged depth, calibration,
tokens and acquisition times. Manifest SHA-256:
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.

## A. World refactor validation

All **234 focused SDK-free tests passed**, with no regressions or pre-existing
failures encountered. The following modules were passed to `python -m unittest -q`
in four processes; every module has the prefix `tests.unit.`:

| Group | Modules | Passed / failed |
|---|---|---|
| Public APIs, configuration and Isaac mocks | `test_perception_factory`, `test_setup_config`, `test_spatial_mapping_config`, `test_multiview_config_factory`, `test_multiview_camera_lifecycle`, `test_runtime_bootstrap`, `test_observation_runtime` | 60 / 0 |
| V1, publication, shared acquisition and real worker transport | `test_primitive_depth_pipeline`, `test_perception_lifecycle`, `test_semantic_processing`, `test_shared_frames`, `test_process_worker`, `test_process_worker_buffers` | 71 / 0 |
| Multiview mathematics and provider | `test_multiview_acquisition`, `test_multiview_projection_window`, `test_multiview_segmentation`, `test_multiview_segmentation_readiness`, `test_multiview_fitting`, `test_multiview_fitting_optimization`, `test_multiview_fitting_uncertainty`, `test_multiview_observability`, `test_multiview_provider`, `test_multiview_variable_cameras`, `test_multiview_final_refinements` | 66 / 0 |
| Mapping and reconstructed planning inputs | `test_spatial_mapping_runtime`, `test_spatial_mapping_rgbd`, `test_spatial_mapping_ground_truth`, `test_occupancy_obstacles`, `test_mapping_obstacle_product`, `test_frozen_rgbd_world`, `test_rgbd_reconstruction` | 37 / 0 |

The standalone WorldModel update-rate check and architecture/syntax checker also
passed. Frozen semantic comparisons retain V1/multiview geometry, identities,
confidence, graspability and clouds. Production files, scene TOMLs, frozen
fixtures and original capture hashes were checked unchanged. Isaac was not started.

## B. One-camera feasibility

Two production construction gates currently reject a singleton:

- [MultiViewConfig](../robot_skill_stack/world/perception/config.py) rejects a
  nonempty camera list shorter than two; inactive configurations may stay empty.
- [Isaac create_cameras](../robot_skill_stack/integrations/isaac/world/multiview.py)
  rejects fewer than two paths. `load_scene_config()` separately rejects an empty
  selected multiview source and uses an “at least two” error message.

The numerical path has no two-camera minimum. `FrameSynchronizer` admits one
exposure, applies the same clock/duplicate checks, and has zero within-batch skew.
`SharedFramePool` accepts a nonempty singleton subset with independent consumer
admission. Provider acquisition, one-cloud concatenation/voxelization, coarse and
radius connectivity, fitting, observability and tracking all execute unchanged.
Observability may obtain different edges from the same camera; it requires the
existing measured boundary evidence rather than two distinct camera identities.

The test-only [replay harness](../tests/multiview/single_camera_replay.py) selects
original `DepthFrame` subsets and supplies already-created replay cameras to the
public semantic builder. It does not alter the two-camera TOML or Isaac factory.
Original stride 2, 2 mm voxels, 12,000-point cap, 24-fit budget, workspace, support
height and all numerical thresholds remain fixed. GT enters only evaluation.

## C. Native detection comparison

Main is `/World/PerceptionCamera`; Side is `/World/PerceptionCameraSide`.
Each detection's contributing cameras are exactly its input selection: Main,
Side, or both. There were no extra unmatched detections.

These are **raw geometric detections**, before three-hit publication. Errors are
center / maximum dimension / radius, in millimetres; radius applies to spheres.
Box XY dimensions use the existing sorted-axis comparison. Yaw error is modulo
90° and is descriptive, not an additional acceptance threshold. The unchanged
native evaluator requires the correct class, center and dimension errors ≤10 mm,
and graspability. `O/G` means parameter-observable / graspable.

| Capture | Input | Object | Class | Center / dimension / radius error, mm | Yaw error, ° | Confidence | O/G |
|---|---|---|---|---|---|---|---|
| Static | Main | Square | cube | .736 / 1.074 / — | .000 | .668553 | yes / yes |
| Static | Side | Square | cube | .408 / .773 / — | .000 | 1.000000 | yes / yes |
| Static | Both | Square | cube | .135 / .273 / — | .000 | 1.000000 | yes / yes |
| Static | Main | Rectangle | cube | .449 / .770 / — | .000 | 1.000000 | yes / yes |
| Static | Side | Rectangle | UNKNOWN | 2.280 / 28.708 / — | — | .000000 | no / no |
| Static | Both | Rectangle | cube | .116 / .234 / — | .000 | 1.000000 | yes / yes |
| Static | Main | SmallSphere | sphere | .160 / .292 / .146 | — | .953939 | yes / yes |
| Static | Side | SmallSphere | sphere | .181 / .314 / .157 | — | .972317 | yes / yes |
| Static | Both | SmallSphere | sphere | .176 / .314 / .157 | — | .970753 | yes / yes |
| Static | Main | LargeSphere | sphere | .230 / .395 / .198 | — | .967757 | yes / yes |
| Static | Side | LargeSphere | sphere | .410 / .494 / .247 | — | .946228 | yes / yes |
| Static | Both | LargeSphere | sphere | .286 / .542 / .271 | — | .948450 | yes / yes |
| Gripper pre-grasp | Main | Square | not detected | — | — | — | — |
| Gripper pre-grasp | Side | Square | cube | .408 / .773 / — | .000 | 1.000000 | yes / yes |
| Gripper pre-grasp | Both | Square | cube | .408 / .773 / — | .000 | 1.000000 | yes / yes |

Other pre-grasp objects match their static geometry exactly; home restores the
static geometric results for every input selection. Static raw counts are
**4 detected / 4 graspable** Main, **4 / 3** Side, **4 / 4** Both. Pre-grasp counts
are **3 / 3**, **4 / 3**, **4 / 4**, respectively. UNKNOWN Rectangle dimensions are
fallback bounds, not a qualified physical Rectangle estimate. No native case
produced a qualified object outside the evaluator's accuracy limits.

## D. Geometric limitations and provider contract

**Main's occluded Square is absent, not an inaccurate or merely unobservable cube.**
Matching the Main cloud's voxel keys to the two-view Square component gives only
24 Main samples, versus 465 Side samples (487 fused cells after overlap).
The 25-point coarse-component minimum and 30-point support minimum remain intact.
No Square candidate survives the Main-only pipeline. Unrelated gripper geometry
is rejected as oversized; it is not substituted for the Square.

**Side's Rectangle is detected but UNKNOWN/ungraspable.** Its 627-point coarse
cloud has retained 4 mm components of 454 and 25 points: 76.40% coverage. The
actual parent cube fit has confidence .999999919 and 95th-percentile surface
residual .000819 mm, but its largest measured XY component has only 571/627
points (91.07%), below the existing 95% recovery condition. Refinement therefore
returns `unresolved_components`; final parameter observability is false.
This is conservative algorithmic refusal tied to this sampling/viewpoint, not
proof that the object's physical dimensions are fundamentally unobservable.
Improving that recognition would be separate algorithm work, not a configuration fix.

Qualification remains under the existing supported-box/curved-primitive model:
one image does not directly observe every hidden surface. The passed deterministic
controls are separate from native evidence: clear/rotated shapes, covered boxes,
invalid/missing depth, planar/insufficient samples, partial curved surfaces,
short cylinder arcs, missing endcaps and missing sphere support. In particular,
known cylinders can remain dimension-unobservable, and a high-confidence partial
box can fail boundary qualification. No check was weakened or threshold tuned.

Both ordinary `observe()` and shared-frame `acquire → compute → accepts → associate
→ publish` were exercised with each selection and the three **distinct original**
static/pre-grasp/home exposures. Duplicate exposures returned no batch; shared
consumers reused immutable original-time arrays. Publication used original timestamps
and existing context-revision admission; `forget()` removed provider geometry and repeated `close()`
released each owned camera once. All contract assertions passed.

Main publishes `[0, 0, 3]`: Rectangle and both spheres keep IDs and reach three hits;
Square's tentative `multiview_2` disappears under occlusion, and home creates
`multiview_5` with only one hit. Side publishes `[0, 0, 4]`, with three graspable
objects and an UNKNOWN Rectangle (`class_name=None`). Both publishes `[0, 0, 4]`
with all four graspable. Side and Both retain every ID. These are the existing
confirmation rules, not singleton API failures. The sparse archive does not test
continuous live visibility, TTL behavior between captures, or post-confirmation
occlusion recovery; those boundaries retain their separate regression coverage.

## E. Numerical processing cost

One process, Python 3.11.18 / NumPy 1.26.4, OpenBLAS/OMP/MKL threads fixed to one,
CPU affinity 0–11, no cgroup CPU quota. Other validation processes had finished.
Each input/capture received one untimed warm-up and three measured calls using
the existing `profiled_compute`; geometry and fit inputs matched exactly across
repeats. Times below are medians in milliseconds; total p95 is descriptive with
only three samples. Acquisition, scoring, association and publication are excluded.

| Capture | Input | Projection/voxel | Coarse | Refinement | Fit + parameters | Observability | Total p50 / p95 | Fits |
|---|---|---|---|---|---|---|---|---|
| Static | Main | 2.863 | 1.534 | 4.118 | 201.531 | 7.970 | 219.313 / 220.287 | 4 |
| Static | Side | 9.974 | 3.254 | 194.430 | 263.002 | 4.240 | 340.195 / 343.655 | 4 |
| Static | Both | 15.945 | 4.236 | 25.183 | 277.973 | 15.316 | 342.376 / 343.987 | 4 |
| Gripper pre-grasp | Main | 2.873 | 1.876 | 4.805 | 143.965 | 3.969 | 158.923 / 159.920 | 3 |
| Gripper pre-grasp | Side | 11.392 | 4.555 | 207.689 | 264.723 | 4.214 | 359.032 / 359.572 | 4 |
| Gripper pre-grasp | Both | 14.786 | 5.736 | 118.545 | 272.537 | 14.001 | 362.507 / 362.877 | 4 |

Refinement includes metric connectivity and any parent/child fits, which also
appear in the fitting column: these timings overlap and must not be summed.
Main saves about 36% of static processor time; its larger pre-grasp saving also
reflects the missing Square and one fewer fit. Side is nearly as costly as Both:
fragment refinement offsets the projection saving. Camera count alone does not
determine cost. These measurements establish neither sustained Isaac publication
rates nor a fix for the previously observed live freshness issue.

## F. Recommendation and reproduction

**CONDITIONAL GO for a separately reviewed configuration/integration change.**
Allow one nonempty camera in `MultiViewConfig` and Isaac `create_cameras`, update
their messages and the selected-source empty-list message, then add the singleton
to the existing construction tests. Keep empty inactive configuration, duplicate
path rejection, all numerical checks, worker behavior and freshness unchanged.
Document the view-dependent failures above; retain two cameras when the full
four-object demonstration must survive this gripper pose. No numerical redesign
is required merely to run one camera. Recognition improvements are not authorized
by this result, and no production singleton support was enabled here.

Reproduce from the repository root with the existing Python 3.11 environment:

```bash
PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ubuntu/.local/share/uv/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11 \
  -m tests.multiview.single_camera_replay \
  --data outputs/multiview-performance-ab-20261122/input.json \
  --output outputs/UNUSED-single-camera/result.json --repeats 3
```

One local raw artifact contains per-object metrics, all 18 measured samples,
provider histories, source/input hashes and environment snapshots:
`outputs/multiview-single-camera-20261124/result.json` (337,485 bytes), SHA-256
`31070957ecf1119430007436bd6198e6273d92f58790b3094e967adf3ca44782`.
No captures or frozen fixtures were regenerated. This investigation stops here.
