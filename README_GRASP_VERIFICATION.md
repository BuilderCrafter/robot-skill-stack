# Gripper-feedback grasp verification patch

This patch changes only grasp verification and the Perception V1 manipulation test.

## Why

The diagnostic run showed:

- the Franka physically lifted the cube by about 100 mm,
- RGB-D discovery lost the cube while the fingers/hand occluded it,
- the cached WorldModel pose therefore stayed at the pre-grasp pose,
- `PickSkill` incorrectly returned `GRASP_FAILED`.

The empty gripper closed to ~0 m total opening, while the held 5 cm cube left the
two finger joints at ~0.025 m each (~0.05 m total opening).

## New responsibility split

- perception: locate/track objects when visible,
- grasp planner: choose the grasp,
- manipulation backend: verify physical grasp using gripper feedback,
- perception: reacquire and verify the object after release/retreat.

`ManipulationBackend` now exposes:

    verify_grasp() -> BackendResult

`IsaacFrankaBackend` estimates total parallel-gripper opening from the two finger
joint positions. V1 considers 5-75 mm a plausible held-object width. The limits
are backend configuration, not `PickSkill` knowledge.

`PickSkill` verifies twice:

1. immediately after closing,
2. again after the lift.

Visual lift is still recorded when the object remains visible, but it no longer
decides grasp success.

## Run

After copying this patch over the repo, run the complete suite:

    python3 scripts/run_perception_v1_suite.py

Upload the resulting:

    outputs/perception_v1_suite_<timestamp>.tar.gz

The updated manipulation result retains partial metrics even when Pick or Place
fails, including physical ground-truth lift (test-only), WorldModel visibility,
grasp-verification details, and place details.

## Expected next question

Pick should now succeed even if the cube is temporarily invisible in the gripper.
The next likely stress point is persistent-ID reacquisition after moving/releasing
an occluded object. The full manipulation test is intentionally left unchanged
at that architectural boundary so the result archive can tell us whether it
needs refinement.
