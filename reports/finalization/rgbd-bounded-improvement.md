# Bounded RGB-D fixed-base clearance experiment

> Publication decision, 14 November 2026: **candidate not retained**. The V5
> production and test trees were restored to `e11a1fd`; candidate code and its
> nine tests remain recoverable in local snapshot
> `edc1b467c6bf03c192d0d9dccce2b538309a024f`. This report preserves the negative
> findings, not an accepted implementation. See [V5 realignment](v5-realignment.md).

## Declaration and starting state

13 November 2026, Europe/Budapest. Branch `finish/cloud-final`, starting HEAD
`e11a1fd2635d6b06a0de46feab07fb7b20cd4550`. Pre-existing staged `AGENTS.md`
and untracked `project-familiarization.md` are preserved. No commit, push or
merge is planned for this review candidate.

## Phase A: feasibility finding (before implementation/native execution)

`EnhancedCollisionModel.preflight_base` uses a cooked-solid enclosing sphere
against finite obstacles. It can reject a floor cuboid outside the physical
base footprint because the sphere extends beyond that footprint. The known
support Plane already uses a separate halfspace proof; it must stay unchanged.

`scripts/generate_franka_collision_model_v4_2.py:generate` takes the min/max of
every vertex of the single `panda_link0` cooked convex hull. Convexity puts the
whole solid inside those bounds. `config/motion/franka_enhanced_v4_2.json`
retains the eight corners, exact coordinate bounds, source geometry/result
hashes, generator hash and conditional frame-alignment certificate. The existing
loader validates eight distinct axis-aligned corners and the installed URDF is
hash-checked by the planner. The original raw cooked archive is absent here;
the existing source-bound certificate is the available evidence, not a newly
recomputed physical model. The new native harness will independently capture
cooked geometry for physical evaluation.

Local bounds in meters are X [-0.1540856041, 0.0715442785],
Y [-0.0945959547, 0.0945388010], Z [0, 0.1399999969]. A synthetic 1 cm floor
cell centered at (0.08, 0, 0.005) overlaps the sphere but is separated from
the AABB along X. This is a geometric feasibility example, not a replay of a
missing historical selected voxel.

Proposed proof: transform the eight base corners and the eight obstacle corners;
project both enclosing convex boxes onto unit face/cross axes. For any axis,
`max(min(B)-max(A), min(A)-max(B))` is a separation lower bound when positive.
The maximum of independently valid axis bounds remains a lower bound on
Euclidean enclosure separation. Subtract the existing 10 µm measured-frame
allowance plus a scale-dependent float64 roundoff guard; accept only a strictly
positive remaining bound. Skip degenerate cross axes; a missing axis can only
cause conservative rejection. Overlap, tangency and numerical ambiguity never
prove physical collision and never grant this refinement's acceptance.

Only a rejected sphere check for an exact `rgbd_occupied_cells` cuboid in a
source-labeled RGB-D snapshot is eligible. GT primitives, known support,
sphere/capsule checks, finger contracts and all model-resource hashes retain
their existing path. No occupancy evidence is removed or transformed into free
space. This proof is conditional on the existing cooked-base/frame certificate;
it is not a continuous robot trajectory certificate.

Native declaration, after focused/full offline tests: one `static/frozen_rgbd`
budget-64 request (primary target), one `partial/frozen_rgbd` budget-64 request
(safety regression), and one fresh `v43/static` budget-64 GT capture because
strict archival inputs are missing. Execute serially with the existing paired
planner, fixed common IK goal, independent physical evaluator and veto. Use new
`outputs/motion-planning-v5/rgbd-base-clearance-20261113/` native subdirectories;
stream/supervisor receipts use `outputs/rgbd-base-clearance-20261113/`.
Order: GT preservation first, primary static RGB-D second, partial safety check
last only if no primary stop condition occurs. No seed search or full
matrix. If the newly admitted static plan is independently unsafe, stop candidate
development and do not pursue another RGB-D solution; record any remaining
declared condition as unrun rather than silently dropping it.

## Phase B: exact implementation

Only production file changed:
`robot_skill_stack/integrations/isaac/manipulation/motion/collision_model.py`.
The candidate adds `_base_box_separation` (eight vertices per box, at most 15
axes), a 10 µm allowance constant, and a supplementary branch in
`EnhancedCollisionModel.preflight_base`. The branch requires all of:

- Existing sphere clearance is nonpositive.
- Snapshot metadata identifies `source_type='rgbd'`.
- Obstacle kind is cuboid, source is `rgbd_occupied_cells`, approximation is
  `exact_selected_occupied_cells`, and surface approximation error is zero.

The helper validates enclosing corners, finite dimensions/positions/quaternion,
and proper orthonormal base rotation. It projects in a base-origin-relative
frame to reduce cancellation. Axes shorter than 1e-12 are omitted. Each selected
axis is normalized; the roundoff guard is `128 * float64_epsilon * scale`, with
scale at least one meter and at least the largest absolute world origin or
transformed vertex coordinate. Invalid/nonfinite calculations reject. Only a
strictly positive gap after **both** guard deductions supplements sphere
rejection. This is a conditional floating-point geometric proof, not formal
interval arithmetic or an exact physical closest-distance calculation.

Diagnostics preserve the original sphere bound, projected gap, lower bound
after allowances, selected axis, frame allowance and roundoff guard. Inconclusive
cases keep `base_geometry_rejected` and `physical_collision_proven=false`.
Positive original sphere checks retain their old method; all GT and known-plane
checks retain their old path and fields. The uninflated plane-halfspace logic
was not changed. No collision descriptor/certificate resource, camera, mapping,
skill, runtime, planner, executor, finger constraint or launcher was changed.
No new production dependency was introduced.

New tests: `tests/unit/test_rgbd_base_clearance.py`, nine discovered methods.
They cover a sphere false rejection provably resolved by the box, overlap,
tangency/near-contact and the full frame allowance, source/approximation gating,
rotated/translated geometry, malformed/nonfinite inputs, large-world numerical
uncertainty, near-parallel axes and unchanged known-plane behavior. Forty
deterministic arbitrary-orientation cases additionally compare accepted bounds
against an independent convex-hull intersection LP; both clear and inconclusive
cases occur. Existing tests continue covering GT cuboid/sphere/capsule behavior,
gripper envelopes, model hashes and plane-domain rejection.

## Phase C: offline results

Reused `/tmp/robot-recon-py312`: Python 3.12.16, NumPy 1.26.4, py_trees 2.6.0,
SciPy 1.15.3 and PyYAML 6.0.2. This is offline only; native uses Isaac Python
3.11.13 and repository `.deps`. Commands from the host checkout:

```bash
/tmp/robot-recon-py312/bin/python -m unittest \
  tests.unit.test_rgbd_base_clearance \
  tests.unit.test_global_collision_model_v4_2 \
  tests.unit.test_franka_collision_model_v4_2 \
  tests.unit.test_global_frozen_rgbd_v5 tests.unit.test_lula_global
# 61 tests PASS, 0.978 s

/tmp/robot-recon-py312/bin/python scripts/run_all_regressions.py --unit-only \
  --artifacts-dir /tmp/rgbd-base-clearance-unit-20261113
# 1,382 discovered tests PASS, 104.099 s; architecture/syntax PASS;
# all eight standalone modules PASS; overall exit 0.
```

Complete logs and summary:
`/tmp/rgbd-base-clearance-unit-20261113/run-_iq7pro0/`.
The increase over the 1,373 baseline is exactly the nine new tests. Historical
fixture assertions and expected hashes are unchanged. An additional container
Python 3.11 run passed 22 focused cases:

```bash
docker exec isaac51 bash -lc 'cd /workspace/robot-skill-stack && \
  ISAAC_SIM=/isaac-sim ./run_isaac_python.sh -m unittest \
  tests.unit.test_rgbd_base_clearance tests.unit.test_global_collision_model_v4_2'
```

No full suite was repeated inside the container: its missing Git executable and
host-owned bytecode directories remain documented environment constraints.

## Phase D: bounded native results and explicit stop

The original stream was the sole Kit PID 1049, parent 1039, cwd `/isaac-sim`.
Before stopping it, saved its exact argv/environment and stream configuration
in `outputs/rgbd-base-clearance-20261113/stream-before.json` (private mode 0600)
and `stream-config-before.toml`. Configuration SHA256:
`fafc7c2b2486b0066a617d3df8f316b9c749baff2746ad6f710173ef2d511d68`.
The mounted repository remained the same host checkout at
`/workspace/robot-skill-stack`; container image/lifecycle was unchanged.

An ignored task-local supervisor, retained as
`outputs/rgbd-base-clearance-20261113/run_experiment.py`, checked exclusive Kit
ownership, sent **TERM only to 1049**, waited for it to exit, ran native jobs
serially with a 600 s timeout/5 s kill grace, and restored the original stream
in `finally`. It did not modify the harness or its acceptance checks. The
higher-level batch driver's Git-dependent provenance function is unavailable
in this container, so the supervisor called the existing native harness through
the unchanged project wrapper directly. The harness's own source hashes,
result/trajectory/contact/cooked-geometry capture and physical veto remain
active. A host-side source manifest additionally verifies no source changed
during native execution (`source-before.json`).

Exact native commands inside the container, from `/workspace/robot-skill-stack`
with `ISAAC_SIM=/isaac-sim`:

```bash
timeout --kill-after=5s 600 ./run_isaac_python.sh scripts/benchmark_global_rgbd_v5.py \
  --scenario v43 --execution-condition static --budget 64 --headless \
  --artifacts-dir outputs/motion-planning-v5/rgbd-base-clearance-20261113/v43/static \
  --result outputs/motion-planning-v5/rgbd-base-clearance-20261113/v43/static/result.json

timeout --kill-after=5s 600 ./run_isaac_python.sh scripts/benchmark_global_rgbd_v5.py \
  --scenario static --execution-condition frozen_rgbd --budget 64 --headless \
  --artifacts-dir outputs/motion-planning-v5/rgbd-base-clearance-20261113/static/frozen_rgbd \
  --result outputs/motion-planning-v5/rgbd-base-clearance-20261113/static/frozen_rgbd/result.json
```

Both exit 0 and report valid measurement `PASS`; those labels do not establish
RGB-D planning or motion success. Exact session receipts/exit codes are in
`outputs/rgbd-base-clearance-20261113/native-session.json`.

| Measurement | Fresh V5 `v43/static` GT preservation | Fresh `static/frozen_rgbd` primary |
| --- | --- | --- |
| Process wall time including startup/save/close | 47.614 s | 38.395 s |
| Selected planning | Success, 3 waypoints | Rejected at source capture, before planner/base check |
| Planner path length | 1.2705546141 rad | Unmeasured; no plan |
| Independent physical evaluation | `ALLOW_SAMPLED_NUMERIC_CLEARANCE`, 286/286 samples | Not invoked; no route |
| Actual selected movement | Completed, maximum joint displacement 1.02894 rad | No commands, no movement |
| Endpoint position / orientation error | 1.06014 mm / 0.00169978 rad | Not a completed target; no accuracy success |
| PhysX reports / physical-contact steps | 0 / 0 | 0 / 0 while stationary, not avoidance |
| Actual sampled box clearance | 15.92227 mm | No executed route |
| Actual sampled whole-body minimum | 23.8419 nm to mounting plane; not useful path margin | No executed route |
| Worst GT joint tracking | 0.0365212 rad | Not applicable |
| Final hold / solver opinion restoration | Both true; requested TGS 32→255→32 | No moving-plan solver lease |
| Runtime/strategy closure | Closed, ownership released | Closed, no internal runtime error |

These measurements are read from each `result.json`, `operations`,
`physical_evaluation`, and `metrics`, not inferred from exit status. GT's
3-waypoint route, path length, clearance and endpoint error reproduce the
historical primary-seed V5 v43 outward outcome. This is a **new V5 capture**,
not an exact archival V4.3 paired roundtrip/Home rerun; missing archival
preflights were not bypassed. GT's optional RGB-D counterplan also reported
stale source and is not counted as another selected RGB-D trial.

The selected RGB-D source timestamp was `8797.578719046`, evaluated at
`8798.638724886` (same monotonic clock): **1.060005840 s** age versus the unchanged
**1.000 s** freshness limit. `planning_source_stale` was preserved in strategy
details and surfaced as skill `path_blocked`. It captured once, refreshed zero
times, issued **zero actions**, and never reached `_base_box_separation` or the
paired planner/physical evaluator. Do not describe this as the new base test
failing a native geometric condition.

Following the declared stop rule, **partial/frozen_rgbd was not run**. Its
unsafe-path veto regression remains unverified on this candidate. No retry,
freshness relaxation, timestamp repair, acquisition adjustment, second solution,
empty-scene GPU run or full matrix was attempted. A matched empty-world route
comparison cannot be made for this primary request because it returned no route;
no obstacle-induced RGB-D detour is demonstrated.

## Offline audit of the newly archived, rejected-source packet

The harness correctly archived even the stale publication with
`source_usable=false`. To finish attribution without another native trial,
`outputs/rgbd-base-clearance-20261113/audit_packet.py` projects the **unchanged
64 archived cuboids** against the certified base at its recorded actual pose.
It does not call frozen capture, refresh any time, plan, or permit execution.
Its input hashes and all per-cell results are retained in `packet-audit.json`.

The same capture independently retains 40 cooked fixed-base vertices; **all
are inside the stored AABB, maximum outside distance 0 m**. This corroborates
the certificate for this fresh native geometry rather than assuming it from
the test's exit code.

| Geometry-only count | Result |
| --- | ---: |
| Original selected cuboids | 64 |
| Sphere rejects | 1 |
| Supplemented test still rejects | 1 |
| Sphere rejections resolved | **0** |

The unresolved exact cell is `32:33:69:70:1:2`, center
`(0.025, 0.095, -0.005)` m, dimensions `(0.01,0.01,0.01)` m.
Its sphere lower bound is **−3.729275 mm**. Its best projected gap is only the
approximately **23.8419 nm** base/plane gap; after the 10 µm frame allowance and
roundoff guard, the candidate lower bound is **−9.976158 µm**. Acceptance would
require weakening the declared frame/numerical margin. The candidate correctly
continues rejecting it. This is a concrete **no-useful-distinction** result for
the current selected packet, although it is not a replay of the unavailable
historical publication.

## Stream restoration

Restoration command, without container/VM/network changes:

```bash
docker exec -d -w /isaac-sim isaac51 sh -c \
  'exec /isaac-sim/runheadless.sh --/app/livestream/publicEndpointAddress=100.84.229.7 --/app/livestream/port=49100 > /workspace/robot-skill-stack/outputs/rgbd-base-clearance-20261113/stream-restored.log 2>&1'
```

Restored Kit PID **3015** uses the original argv/cwd/configuration. Server log
records `app ready` and `Isaac Sim Full Streaming App is loaded.`. Detailed
server checks are in `stream-restored.json`. The stream's original standard
embedded experience was restored, not replaced with the project GUI. Client
video was **not visually verified**.

## Historical comparison and final decision

Historical V5 evidence remains byte-for-byte untouched: primary GT 7/7
completed; RGB-D 3/7 plans, four base rejections, two unsafe plans vetoed,
one completed non-detouring narrow route. The fresh primary here fails source
freshness before geometry, so it must not be pooled with or replace that
historical baseline. No newly accepted trajectory was presented to the physical
evaluator, and no RGB-D movement completed.

**Engineering success criterion is not met:** the proof is conservative and
distinguishes synthetic sphere false rejections, but resolves zero actual
rejections in the archived selected packet. **Research success criterion is
not met:** there is no accepted RGB-D plan, independent trajectory acceptance,
executed motion or demonstrated detour. Safety of a newly admitted RGB-D route
has not been established, nor has partial-scene veto preservation been rerun.

**Recommendation: do not retain this candidate for the final release; revert
the production/test candidate after review.** It adds code without measured
benefit to the target packet. The currently uncommitted implementation and
tests remain available for the user's review as requested. Keep this negative
experiment report and fresh GT preservation evidence. Do not weaken the frame
allowance or fix source timing as a second research campaign in this task.

Direct/reactive/GT planner/executor sources, skills, camera/mapping algorithms,
launch wrappers and enhanced resources are unchanged. All existing offline
regressions pass; the fresh GT move passes. Direct/reactive native manipulation
was not separately rerun, and full GT Home/cancel/lifecycle acceptance is not
newly claimed. The research-only independent physical veto remains outside the
ordinary production RGB-D safety path.

## Git state at experiment completion (13 November; historical snapshot)

Final HEAD remains `e11a1fd2635d6b06a0de46feab07fb7b20cd4550`, branch
`finish/cloud-final`; no commits, pushes, merges, resets or infrastructure
configuration changes. Existing staged `AGENTS.md` and familiarization report
remain intact. Candidate changes left unstaged:

- Modified `robot_skill_stack/integrations/isaac/manipulation/motion/collision_model.py`.
- New `tests/unit/test_rgbd_base_clearance.py` (nine tests).
- New `reports/finalization/rgbd-bounded-improvement.md` (this report).

Native evidence is under
`outputs/motion-planning-v5/rgbd-base-clearance-20261113/{v43/static,static/frozen_rgbd}/`:
`native.log`, `result.json`, original worker occupancy/camera inputs, cooked
geometry and trajectory/contact traces. Stream restoration, supervisor source,
source manifest and offline packet audit are under
`outputs/rgbd-base-clearance-20261113/`. These are ignored local artifacts;
historical results were neither overwritten nor reclassified.
