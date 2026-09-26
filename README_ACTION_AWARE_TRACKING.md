# Action-aware tracking / placement reacquisition patch

This patch fixes the identity failure seen after successful physical Pick + Place.

## Problem confirmed by the previous suite

The robot transported the original cube about 25 cm. The tracker only had the
old pre-pick visual position and a normal 18 cm association gate, so the
released cube was spawned as a new object ID. Place then verified the stale
original track and incorrectly failed.

## Architecture

Discovery stays action-agnostic.

The WorldModel can now publish a short-lived association prior:

    object_1 expected near [x, y, z]

The updater sends an `ObservationContext` to state providers. Ground truth
ignores it. Perception passes it to the tracker.

Tracker association is now:

1. resolve active association hints first,
2. run normal position/size association for remaining tracks/candidates,
3. spawn only still-unmatched candidates.

Hint matches still require a valid spawnable candidate, spatial proximity to the
expected position, and compatible size. A hint is a prior, not a forced pose.

Held tracks and tracks protected by an active hint are not expired while
temporarily invisible.

## Place timing detail

`PlaceSkill` creates the expected-position hint immediately before
`open_gripper()`, after the release pose has been reached.

This is intentional: `IsaacFrankaBackend.open_gripper()` advances simulation
for multiple frames, so perception can see the object during the gripper-open
operation. Creating the hint only after `open_gripper()` returns would leave a
race where the tracker could already spawn a new ID.

If opening fails, the hint is cleared.

The WorldModel does NOT overwrite the object's measured pose with the commanded
target. The target is used only for association.

## Files

New:
- world_model/context.py

Replaced:
- world_model/world_model.py
- world_model/updater.py
- backends/isaac/ground_truth_provider.py
- perception/tracker.py
- perception/state_provider.py
- skills/place.py
- scripts/test_world_model_update_rate.py
- scripts/test_perception_v1_unit.py
- scripts/test_perception_v1_manipulation.py

## Tests added

The unit suite now checks:

- held tracks survive beyond `max_misses`,
- a > normal-gate movement is reacquired through a place hint,
- hint priority works with two cube tracks,
- the WorldModel consumes a successful hint.

The Isaac manipulation test now checks:

- physical Pick succeeds,
- Place succeeds,
- the same persistent object ID is reacquired,
- no duplicate same-class visible track is created,
- the association hint is consumed.

## Run

Run the existing complete suite:

    python3 scripts/run_perception_v1_suite.py

Upload:

    outputs/perception_v1_suite_<timestamp>.tar.gz

No new packages and no USD changes are required.
