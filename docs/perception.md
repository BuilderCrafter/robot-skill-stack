# Perception V1 — frozen cube-focused RGB-D milestone

Perception V1 is the current stable lower-level perception/world-state milestone.
It uses geometry-first RGB-D discovery and does not depend on Isaac semantic
labels for normal perception.

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
ObjectTracker
   - persistent object_N identity
   - position/size association
   - held-object retention
   - action-derived association hints
   ↓
SemanticBelief
   - repeated cube/unknown evidence
   - class may remain unknown
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

## Manipulation boundary

Perception answers where an object is and what geometry is currently observed.

The existing top-down grasp planner remains the V1 cube grasp planner.

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
- small number of separated cube-like objects
- cubes roughly 2–10 cm
- no stacked/touching objects
- object orientation may remain unknown
- simple top-down parallel-gripper grasping

Deferred work includes arbitrary-object 6D geometry, learned/general grasp
planning, difficult occlusion, touching/stacked segmentation, and robust
large-scale multi-object re-identification.

## Acceptance suite

Run:

```bash
python3 scripts/run_perception_v1_suite.py
```

The suite validates software behavior, Isaac discovery/tracking, multi-cube
discovery, lazy point-cloud access, perception-driven Pick + Place, gripper
verification, and same-ID reacquisition after placement.

Expected output archive:

```text
outputs/perception_v1_suite_<timestamp>.tar.gz
```
