# Perception V1 cleanup patch

This is a minor cleanup only. It intentionally does not change discovery,
tracking thresholds, semantic belief, grasp behavior, or action-aware
association behavior.

Changes:
- formal `WorldObservationProvider` protocol
- type the runtime/updater provider boundary
- use Kit's `omni.kit.async_engine.run_coroutine` helper for standalone async
  runtime bootstrap instead of direct `asyncio.ensure_future`
- update legacy standalone runtime/BT/diagnostic scripts to use the helper
- remove the stale GUI call to deleted `WorldModel.refresh_all()`
- make the GUI object field prefer current WorldModel object IDs (`object_1`)
- consolidate `README_PERCEPTION_V1.md` into the final frozen V1 architecture

After applying, run:

    python3 scripts/run_perception_v1_suite.py

The main thing to inspect is whether the prior `[Error] [asyncio] Cannot enter
into task ... build_runtime()` startup spam has disappeared. Functional results
should remain unchanged.
