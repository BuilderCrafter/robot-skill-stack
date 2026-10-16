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

The tracker maintains `SemanticBelief`, but the V1 provider emits its geometry
shape as the class label rather than the resolved semantic history. Production
V1 uses `PrimitiveEstimator`; there is no separate classifier selection path.

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

## Acceptance suite

Run:

```bash
python3 scripts/run_perception_v1_suite.py
```

This historical, cube-focused acceptance suite validates software behavior,
Isaac discovery/tracking, multi-cube discovery, lazy point-cloud access,
perception-driven Pick + Place, gripper
verification, and same-ID reacquisition after placement.

It requires Isaac Python and a simulator-capable workstation. Broader primitive
unit modules are included in `scripts/run_primitive_perception_suite.py`,
which also dispatches through the Isaac Python wrapper; see `validation.md` for
plain-Python checks. Passing offline fits is not physical manipulation evidence.

Expected output archive:

```text
outputs/perception_v1_suite_<timestamp>.tar.gz
```
