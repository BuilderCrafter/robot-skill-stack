# Phase 4A: WorldModel and observations

Branch: `refactor/phase-4a-world-model`, based on `main` at `c994ffa`.
This phase is proposed for review, not merged. Phase 4B/4C are not started.

## Architecture and ownership

Previously the tracker, WorldModel and GUI-triggered cleanup all participated
in object aging. The updater discovered optional `age`, `forget` and `close`
methods; bootstrap separately discovered/closed the camera. Semantic voting ran
although production labels came directly from the primitive estimator.

Now:

```text
RGB-D -> discovery / primitive fitting -> tracker -> ObservationBatch
Isaac ground truth -------------------------------> ObservationBatch
                                                         |
                                                  WorldModelUpdater
                                                         |
                                                     WorldModel
                                                         |
                                               skills / read-only GUI
```

- Sources own normalized observations and source resources. RGB-D owns its
  camera and lazy geometry; ground truth owns no subscriptions or camera.
- Updater alone schedules samples, applies observations, marks missing/stale
  objects invisible, expires IDs/hints, coordinates forgetting, exposes health,
  and closes the source through the existing callback teardown path.
- WorldModel stores object state, attachments and placement association hints.
  Its removal operation refuses held/uncertain IDs and active hint IDs.
- Tracker stores only association/confirmation data and current-frame masks.
  Confirmed tracks are forgotten by the updater; missing tentative tracks are
  discarded because confirmation requires consecutive fresh detections.
- UI refresh reads state; Clear Lost explicitly invokes the updater. The existing
  source label shows health and existing action buttons use it. No visual layout
  or controls were added.

## Public interface changes

`WorldObservationProvider` explicitly requires three methods:

```python
observe(context=None) -> ObservationBatch
forget(object_id) -> None
close() -> None
```

`ObservationBatch(observations, timestamp)` contains existing `ObjectObservation`
records. Timestamp is in the local monotonic clock domain and must increase.
A missing timestamp means no new sample; source failures raise exceptions.
A fresh empty list is a successful frame with zero accepted objects. Supplied per-object measurement timestamps are preserved, so a new batch cannot
refresh an old cached pose. Missing object timestamps inherit the batch time.
Sources must not relabel cached observations as fresh. RGB-D uses the existing SDK frame
token and consumes each token once, including failed/invalid frames. External
sources must translate their timestamps to the same clock domain.

The updater publishes one `ObservationHealth` record through WorldModel:

| Status | Meaning |
| --- | --- |
| waiting | No fresh sample accepted yet |
| fresh | A sample, including an empty frame, arrived within the freshness budget |
| stale | No advancing sample within the budget |
| failed | Source/processing error, or source closed |

The default freshness budget remains 1 s; the updater now owns it. Object TTL
still comes from `[world].stale_object_ttl` (15 s by default). Repeated/old samples
cannot refresh health, poses, or visibility. A source failure immediately clears
visibility; only a new valid sample clears the error. Cleanup also runs between
scheduled observations. While physics is paused, GUI refresh does not mutate
cached visibility, but the time-based health read still becomes stale.

Pick/Place reject new operations with `WORLD_STATE_UNAVAILABLE` when source state
is unhealthy. Pick rechecks between approach stages, before closure. Recovery
cannot automatically Home/retry that failure. Home/Move retain their existing
control/held-state behavior. Manually populated standalone models with no bound
updater remain usable; every production runtime binds an updater.

`WorldModel(stale_object_ttl=...)`, `expire_stale` and `clear_lost` have been
replaced by updater construction/operations. `association_hints()` is now a pure
read; updater cleanup prunes expired hints. `ObjectTracker.update()` no longer
takes semantic predictions, and its constructor has only distance, size and
confirmation settings. Direct source consumers now unpack batches; the Isaac
discovery scenario exercises the updater and retains its original assertions.

## Removed complexity and tradeoffs

- Deleted `world/perception/semantics.py`, `SemanticBelief` and
  `SemanticPrediction`. Production geometry labels are unchanged.
- Combined two association/mutation passes into one ordered greedy assignment:
  eligible placement-hint pairs first, then ordinary distance/size pairs.
- Removed miss-count expiration, tracker aging/TTL, cumulative semantic history,
  duplicate consecutive-hit counters and registration-time bookkeeping.
- Removed partial-detection matching/size-update heuristics. Partial occlusion
  can hide objects sooner. Confirmed IDs/geometry remain cached until TTL;
  ordinary full detections and placement hints reacquire the same IDs.
- Removed obsolete `max_misses`, `occlusion_distance` and semantic-voting config
  fields/keys. Remaining TOML sections and source-selection layout are unchanged;
  no robot/source/scene configuration redesign was attempted.
- Removed the ground-truth lifetime exception: without source updates its stale
  records expire too. Scene-defined IDs return on the next valid observation.
  Never-observed manually registered records may be removed immediately; the
  extra registration clock/store is unnecessary in the observation-fed runtime.
- Removed periodic GUI cleanup and optional source-method introspection.
  Bootstrap delegates source cleanup, including failed construction and repeated
  initialization, without changing camera/SDK lifecycle implementation.

Greedy association can still be ambiguous for crossing/touching similar objects.
No stronger re-identification claim is made. Geometry fitting, primitive grasp
math, scene geometry, Franka/RMPflow feedback, cooperative motion and gripper
verification are unchanged. Held/uncertain IDs and attachment transitions remain
owned by manipulation. Grasp-loss invalidation cannot be undone by a repeated
source sample. Placement hints remain association priors, never measured poses.
Place retreat-result semantics remain unchanged and deferred.

## Size and validation

Physical Python lines, including comments/blanks, compared with `c994ffa`:

| Scope | Before | After |
| --- | ---: | ---: |
| Tracker | 305 | 110 |
| Perception tracker/provider/factory/semantics | 505 | 252 |
| Reviewed model/source/bootstrap/UI-controller files | 1,646 | 1,436 |
| All production Python (`robot_skill_stack/` + `exts/`) | 5,665 | 5,456 |

The perception group shrank about 50%; the broader reviewed files about 13%.
The updater/contract grew to make formerly implicit responsibilities explicit.
Behavioral tests and this review record add lines; total repository diff size is
not presented as an implementation reduction.

Baseline: 244 unit tests, eight standalone offline checks and architecture/syntax
passed. Final cloud run: 266 unit tests, all eight offline checks and
architecture/syntax passed. Tests cover both production adapters (SDK access
doubled for ground truth), multiple primitives/IDs, missing/reacquired objects,
confirmation, TTL/forgetting, held/uncertain/hint protection, long-distance
placement reacquisition, empty/frozen/failed sources, source cleanup, repeated
initialization, read-only UI refresh, stale-action guards and manipulation state.
Tests for deleted semantic internals were removed; behavioral assertions remain.
Offline SDK doubles do not prove native camera or physics behavior.

Reproduce with the prepared environment's Python:

```bash
python3 scripts/run_all_regressions.py --unit-only
# Cloud equivalent: .venv/bin/python scripts/run_all_regressions.py --unit-only
```

## First workstation results and fixture repair

The first uploaded `phase4a-regression.zip` reports five regression
scenarios passing, including the four-trial benchmark, and one failing:
`test_grasp_planning` expected `NO_VALID_GRASP` but received `OBJECT_NOT_FOUND`.
The operator separately reported both Isaac scenarios passing; those artifacts
were not included in this ZIP.

The failure was reproduced offline using the actual scenario, updater, runtime
and planner. The test registered a synthetic 90 mm cube without a measurement
timestamp before submitting its Kit task. Cleanup during that submission gap
removed the never-observed record before Pick inspected it. The planner's size
check was never reached.

The test now installs the synthetic record, directly awaits Pick and removes
the record in `finally`, all within one Kit task. Oversized rejection occurs
before any yielding backend command. Tests exercise cleanup before task startup,
verify `NO_VALID_GRASP` / `object_too_large` with zero backend commands, and
check fixture removal on error/cancellation while preserving the original error.
The original physical-motion and 25 mm placement gates remain unchanged.
No production lifetime, freshness, control or planner changes were needed.
After the repair at `9ac2e04`, the operator reported that the tests now pass.
The final rerun artifacts were not uploaded; this is operator-reported native
validation, not cloud execution. Before Phase 4B began, the operator also
confirmed that both-source GUI checks, pause/resume, repeated initialization and
shutdown, and the injected-loss diagnostic all passed. Phase 4A is merged into
main at `1d9141f` (PR #5).

## Workstation validation

No Isaac-dependent scenario ran in the cloud. From this branch on the workstation:

```bash
git fetch origin
git switch refactor/phase-4a-world-model
git pull --ff-only
python3 scripts/run_all_regressions.py --group isaac \
  --artifacts-dir outputs/phase4a-isaac
python3 scripts/run_all_regressions.py --group regression \
  --artifacts-dir outputs/phase4a-regression
```

The runner invokes each native scenario through the unchanged Isaac wrapper.
Retain all eight scenarios and the existing benchmark success/accuracy gates.
NumPy 1.26.4 is still verified before SimulationApp; `.deps`, py_trees 2.6.0,
launch scripts and dependency pins are unchanged.

Run the existing injected-loss diagnostic:

```bash
mkdir -p outputs/phase4a-held-state
./run_isaac_python.sh scripts/diagnose_held_state.py \
  --result outputs/phase4a-held-state/loss.json \
  --artifacts-dir outputs/phase4a-held-state/loss \
  > outputs/phase4a-held-state/loss.log 2>&1
```

Then perform GUI smoke checks with both observation sources: Home/Move/Pick/Place, both BT
tasks, cube/sphere/cylinder display, same-ID placement reacquisition, Clear Lost,
source switching and repeated initialization. Pause long enough to see stale
health and disabled perception-dependent controls; resume and confirm recovery
from new frames. Retain the existing stop-on-contortion/contact procedure.
Do not merge based on offline checks alone.

## Deferred to Phase 4B or later

Robot-backend/source/scene configuration portability and selection remain in the
existing Isaac composition root. GUI capture is intentionally source-specific.
There is no plugin registry, event bus, new monitoring subsystem, force sensing,
controller redesign or new automatic recovery policy. No Phase 4B/4C work is
included in this branch.
