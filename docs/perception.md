# Perception V1 — geometry-based RGB-D primitives

Perception V1 is the default geometry-based world-state source. It discovers
separated tabletop objects and fits cubes/boxes, spheres, and upright or lying
cylinders without Isaac semantic labels.

## Runtime architecture

```text
RGB-D camera
   ↓
PerceptionFrame
   ↓
DepthObjectDiscoverer
   - workspace filter
   - support-plane removal
   - connected components
   - position + size estimate
   ↓
PrimitiveEstimator
   - box, sphere, upright/lying cylinder fits
   - unsupported or ambiguous geometry remains unknown
   ↓
ObjectTracker
   - persistent object_N identity
   - position/size association
   - held-object retention
   - action-derived association hints
   ↓
ObjectObservation[]
   ↓
WorldModelUpdater
   ↓
WorldModel
```

Ground truth implements the same `WorldObservationProvider` contract and remains
an oracle/reference provider. `[world].provider` swaps between `ground_truth`
and `perception` without changing skills, behavior trees, or the runtime API.
The setup's `[scene]` supplies workspace bounds and support-plane height.
Discovery, tracking and fitting thresholds use the existing Python defaults;
the former algorithm-tuning TOML sections are removed. RGB-D startup validates
only the robot and camera prims, without requiring ground-truth object IDs,
sizes or wrappers. See [Phase 4B](phase_4b_config_adapters_ui.md).

V1 uses the current `PrimitiveEstimator` geometry as the class label. Unused
semantic voting/history has been removed. One greedy distance/size association
pass gives placement hints priority over ordinary proximity. Three consecutive
fresh detections confirm an ID; only complete detections refresh it. Partial
occlusion can therefore hide objects sooner, while confirmed IDs remain
available for reacquisition until the updater expires them.

## Observation ownership

Sources implement `observe(context) -> ObservationBatch`, `forget(id)` and
`close()`. A fresh batch has a strictly increasing local monotonic timestamp;
a fresh frame with no accepted objects has an empty observation list. No new
camera token or missing depth returns a batch without a timestamp, not cached
observations. Errors raise and the updater records the source failure.

Only `WorldModelUpdater` applies observations, marks missing/stale objects
invisible and coordinates expiration with source forgetting. It uses a 1 s
freshness budget and the existing `[world].stale_object_ttl` (15 s by default).
The source adapter owns its camera. WorldModel keeps cached state, attachments
and placement hints; GUI refresh does not perform lifetime maintenance.
See [Phase 4A](phase_4a_world_model.md) for health and API details.

## Manipulation boundary

Perception answers where an object is and what geometry is currently observed.

Isaac bootstrap selects `SimplePrimitiveGraspPlanner`, which handles supported
primitives with conservative top-down grasps. Unknown geometry is not guessed.

Physical grasp success is verified by the manipulation backend from Franka
gripper feedback rather than requiring the object to remain visually observable
inside the gripper.

While a held object is occluded, its persistent track is retained. During Place,
the WorldModel publishes a short-lived expected-position association hint. The
tracker can use that prior to reacquire the same object ID after robot-mediated
transport instead of incorrectly spawning a new ID.

The expected pose is never written into the measured object pose; it is only an
association prior.

## Geometry storage

The WorldModel does not store masks or point clouds.

The tracker owns the latest short-lived mask. The perception provider keeps the
latest RGB-D frame and exposes lazy geometry access:

```python
provider.get_mask(object_id)
provider.get_point_cloud(object_id)
```

The point cloud is generated only when requested and is not persisted.

## V1 assumptions

- fixed external RGB-D camera
- flat configured support plane
- small number of separated supported primitives within configured size bounds
- no stacked/touching objects
- box yaw and cylinder axis are fitted; arbitrary 6D geometry is not supported
- simple top-down parallel-gripper grasping

Deferred work includes arbitrary-object 6D geometry, learned/general grasp
planning, difficult occlusion, touching/stacked segmentation, and robust
large-scale multi-object re-identification.

## Validation

```bash
python3 scripts/run_all_regressions.py --unit-only
python3 scripts/run_all_regressions.py --group isaac --artifacts-dir outputs/regressions
```

Offline tests cover primitive fits, uncertainty rejection, tracking, lazy geometry,
frame capture/replay and manipulation logic. The Isaac group retains the original
cube discovery and manipulation acceptance scenarios, including multiple objects,
gripper verification and same-ID reacquisition after placement. Physical sphere
and cylinder manipulation still requires the workstation smoke checks.
See `validation.md` for group requirements and failure/result handling.
