# Perception V1 — cube-focused RGB-D discovery

This overlay replaces the simulator-semantic-label perception path with a geometry-first RGB-D pipeline. No additional Python packages are required beyond the repo's existing `.deps` setup.

## Architecture

```text
RGB + depth
   ↓
PerceptionFrame (latest frame only)
   ↓
DepthObjectDiscoverer
   - workspace filter
   - support-plane removal
   - connected components
   - position + size estimate
   ↓
ObjectTracker
   - stable object_N IDs
   - nearest-neighbour association
   - short occlusion tolerance
   ↓
Semantic belief
   - repeated cube/unknown geometric evidence
   - sliding window, never trusts one pass forever
   ↓
ObjectObservation[]
   ↓
WorldModelUpdater
   ↓
WorldModel
```

The WorldModel never stores masks or point clouds. Masks remain in short-lived tracker state. `PerceptionStateProvider.get_point_cloud(object_id)` reconstructs an object-only cloud lazily from the latest mask + depth frame and does not persist it.

`IsaacGroundTruthProvider` is unchanged and remains the oracle/reference provider. `[world].provider` still swaps between `ground_truth` and `perception`.

## V1 assumptions

- fixed external RGB-D camera
- flat support plane at configured Z
- small number of separated cube-like objects
- cube dimensions roughly 2–10 cm
- no stacked/touching objects
- orientation is intentionally left unknown
- existing `TopDownGraspPlanner` remains unchanged

The semantic pass is deliberately weak in V1: it classifies cube-like geometry as `cube` and otherwise leaves the class unknown. Identity (`object_1`) is independent from semantic class and persists through repeated observations.

## Apply

Overlay the contents of this archive onto the repository root. No USD changes and no new package installation are required.

## Run the complete suite

From the repo root:

```bash
python3 scripts/run_perception_v1_suite.py
```

It runs:

1. pure software/unit tests
2. Isaac discovery/geometry/tracking/multi-cube tests
3. perception-driven Pick + stable Place integration

The suite writes logs, JSON metrics and visual artifacts, then creates one archive:

```text
outputs/perception_v1_suite_<timestamp>.tar.gz
```

Upload that archive for analysis if anything fails.

## Main acceptance targets

- single cube discovered without semantic labels
- position error < 10 mm
- size error < 12 mm per dimension
- stable ID while static
- stable ID after cube motion
- second unconfigured cube receives a new ID
- removed cube is retained as invisible
- point cloud generated only on request
- perception-discovered ID can complete Pick + stable Place

Semantic classification may remain `unknown` without failing discovery. A confidently wrong class should be treated as a semantic-layer issue rather than an object-discovery failure.
