# Perception V1 Pick Diagnostic

This patch adds diagnostics only. It does not change production perception, tracking, skills, or manipulation behavior.

## Purpose

Determine whether the current Pick failure is:

- a real physical grasp/motion failure, or
- a perception false-negative caused by the cube being occluded/merged with the Franka during grasp/lift.

The diagnostic first calibrates the Franka gripper while empty (open, empty-closed, reopened), then records before and after Pick:

- true Isaac cube pose (evaluation only),
- cached WorldModel pose,
- perception tracker position/size/visibility/hits/misses/mask size,
- gripper joint positions,
- end-effector pose,
- RGB/depth/mask artifacts when available.

It also samples the state for ~1 second after Pick and automatically derives a diagnostic verdict. The empty-close calibration will help decide whether gripper joint feedback is suitable for production grasp verification and what separation/threshold is realistic.

## Run

From the repository root on the Isaac Sim workstation:

```bash
python3 scripts/run_perception_pick_diagnostic.py
```

Do not change perception thresholds or PickSkill before running it.

The runner creates:

```text
outputs/perception_pick_diagnostic/<timestamp>/
outputs/perception_pick_diagnostic_<timestamp>.tar.gz
```

Upload the generated `.tar.gz` archive back to ChatGPT.

## Verdicts

- `PERCEPTION_FALSE_NEGATIVE`: true cube lifted >= 30 mm, but cached/perceived lift did not.
- `PHYSICAL_GRASP_OR_MOTION_FAILURE`: true cube itself did not lift >= 30 mm.
- `PICK_VERIFICATION_MISMATCH`: true lift succeeded and cached lift also suggests it moved, but Pick still failed.
- `PICK_REPORTED_SUCCESS`: Pick succeeded during this run.
