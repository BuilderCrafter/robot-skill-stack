# Phase A — Robust lower-level execution

This patch extends the frozen Perception V1 milestone without changing its
discovery, semantic-belief, tracking, grasp-verification, or action-aware
reacquisition algorithms.

## Scope

Phase A adds four things:

1. deterministic workspace + Pick/Place benchmarking,
2. reactive Pick with bounded local replanning,
3. real Behavior Tree recovery validation,
4. two-object manipulation with persistent identities.

Phase B (not included here) will refactor grasp planning into a standalone
planner interface with structured feasibility results.

## Reactive Pick

`PickSkill` now remembers the object position used to build each grasp plan.
After reaching pre-grasp and again after the grasp approach, it compares the
latest visible WorldModel pose with that planned position.

Default policy:

- movement threshold: 20 mm
- maximum local replans: 2

If the object moved farther than the threshold, Pick regenerates the grasp from
the latest WorldModel state and retries locally. If it keeps moving beyond the
bounded retry budget, Pick returns:

    FailureCode.OBJECT_MOVED

This is intentionally below the Behavior Tree / future LLM layer.

## Benchmark

The benchmark covers 12 deterministic workspace poses and 4 full
perception-driven Pick+Place trials.

Recorded metrics include:

- localization error,
- size error,
- persistent object ID,
- Pick/Place success,
- local replans,
- physical target error.

The test uses Isaac ground truth only as an evaluator, never as the perception
input.

## Recovery scenario

The recovery test:

1. moves the robot away from Home,
2. moves the cube outside the perception workspace,
3. confirms the WorldModel marks it invisible,
4. starts the existing recovery BT,
5. restores the cube during Recovery Home,
6. verifies the first Pick fails with `OBJECT_POSE_UNKNOWN`,
7. verifies Home succeeds,
8. verifies Retry Pick succeeds,
9. verifies the overall BT completes Pick+Place successfully.

No production-only recovery hook is added.

## Multi-object scenario

A second 5 cm physical rigid cube is spawned under `/World/Objects`.
Perception must discover two persistent IDs.

The test then:

- Pick+Place first ID,
- preserve the second ID,
- Pick+Place second ID,
- verify both original IDs remain visible,
- verify no duplicate tracks appear,
- verify both physical target errors.

## Run

Apply the patch over the repository root, then run:

```bash
python3 scripts/run_phase_a_suite.py
```

The suite always continues through all stages and packages the results as:

```text
outputs/phase_a_suite_<timestamp>.tar.gz
```

Upload that archive if any stage fails.

## Files

Production changes:
- `core/skill.py`
- `skills/pick.py`

Test/support additions:
- `scripts/phase_a_common.py`
- `scripts/test_phase_a_unit.py`
- `scripts/test_phase_a_benchmark.py`
- `scripts/test_phase_a_reactive_pick.py`
- `scripts/test_phase_a_recovery.py`
- `scripts/test_phase_a_multi_object.py`
- `scripts/run_phase_a_suite.py`
