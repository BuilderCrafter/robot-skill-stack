# Native reconstructed RGB-D RRT after observer fix

**2026-11-14 — SUCCESS in one unchanged nominal `partial` research trial.**
Fresh sensor observations produced two qualified boxes; the final original-source
age was **0.945622 s**. The conservative combined world preserved every original
selected occupied cuboid using **24 camera proxies plus the known Plane**.
Lula returned a three-waypoint route with **1.057749 rad** deviation from the
matched direct segment. Independent physical evaluation accepted its route and
trajectory and proved that the unexecuted unobstructed counterfactual intersects
the Post. The selected RGB-D trajectory executed successfully: **0.875 mm** final
TCP position error, **zero contact/proximity reports**, and **41.143 mm** minimum
sampled moving-arm clearance. Final hold, solver restoration and cleanup passed.

This answers the task's question positively for this one frozen scene. It is
not production integration, a continuous collision certificate or evidence of
general RGB-D planning reliability. Exactly one native attempt, zero retries
and no alternative scene or method were used.

Published numerical evidence:
[summary and success gates](../../research/rgbd_box_geometry/evidence/native_partial_postfix/summary.json),
[original research receipt](../../research/rgbd_box_geometry/evidence/native_partial_postfix/research.json),
[three native plans](../../research/rgbd_box_geometry/evidence/native_partial_postfix/plans.json),
[physical evaluations](../../research/rgbd_box_geometry/evidence/native_partial_postfix/physical-evaluation.json),
[motion measurements](../../research/rgbd_box_geometry/evidence/native_partial_postfix/motion.json),
[source manifest](../../research/rgbd_box_geometry/evidence/native_partial_postfix/source-manifest.json),
[sensor-result seal](../../research/rgbd_box_geometry/evidence/native_partial_postfix/sensor-result-seal.json)
and [validation](../../research/rgbd_box_geometry/evidence/native_partial_postfix/validation.json).

## 1. Starting Git state, source hashes and focused checks

Branch `feature/global-rgbd-planning-v5`, clean starting HEAD
`43a35d0f34def4aba503af3d39a95006967f2d5c`. An explicit fetch confirmed the same
remote tip. `AGENTS.md` and the requested readiness-fix, stall-investigation,
native-proof, bridge-feasibility, visibility and V5 study reports were reviewed.
The old native proof failed entirely before reconstruction; its artifacts were
preserved rather than counted as a reconstruction/planning result.

The source manifest binds the actual native entry point, bridge, reference and
visibility algorithms, corrected observer helpers, benchmark, protocol, matched
planner, independent evaluator and authored scene/profile by SHA256. All remain
at their starting bytes except the following small research receipt correction:

- `research/rgbd_box_geometry/native_trial.py`: +25/−15 lines, largely indentation
  around the unchanged readiness loop; records before/after source identity,
  health, camera provenance and final runtime/filter diagnostics in `finally`.
- `research/rgbd_box_geometry/test_rrt_bridge.py`: +9/−2 lines verifying that this
  evidence survives both successful readiness and rejected/late publications.

The previously documented absence of final readiness diagnostics justified that
correction. No admission predicate, source selection, deadline, clock, algorithm
or planner behavior changed. The existing before-SDK-close receipt write remains
in place. No readiness-only native experiment was repeated.

**151 focused tests passed on host Python 3.12.16** in 1.179 s; the **same 151
passed on native Python 3.11.13 / NumPy 1.26.4** in 1.463 s before Kit startup.
They cover bridge invariants, observer read-only behavior, genuine model-change
invalidation, mapping, frozen sources, matched planning, base checks and Lula
validation. Architecture boundary, syntax and whitespace checks passed. The
validation receipt binds test logs. No production file or fixture expectation
was changed.

## 2. Native environment and fixed configuration

The existing official `nvcr.io/nvidia/isaac-sim:5.1.0` container `isaac51` was
stopped, exit 137, with no Kit/GPU compute process. It was started using its saved
configuration. Only owned Kit PID **327770** ran, with the existing observation
worker mechanisms. No stream was active or interrupted.

The nominal `partial` arrangement, Post/Occluder geometry, both authored 640×480
cameras, semantic 5 Hz / mapping 2 Hz rates, stride 8 mapping, shared 1,024-ray
budget, 10 mm voxels, 64 camera proxies, support Plane, enhanced Franka model,
seed 123456 and 10,000-iteration bound were unchanged. START was
`(0.45, −0.25, 0.35)`, GOAL `(0.45, 0.25, 0.35)`, orientation wxyz `(0,0,1,0)`.
This was not `verified_occlusion` or a newly arranged fixture.

```bash
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.native_trial \
  --output outputs/motion-planning-v5/reconstructed-native-proof-postfix-20261114/attempt-1
```

A separate 650-second host bound applied. No other attempt directory was used.

## 3. Readiness and original-source freshness

Warmup completed in **8.018340107 s** after the established START setup and
fixture preparation. Corrected observer initialization and physical-reference
preparation left the filter valid; its source error was absent. Mapping
generation remained 0, worker invalidation epoch 0, with zero source/worker
failures and no reset. Readiness accepted the first new publication, not a
different batch chosen for easier fitting.

| Measurement | Recorded value |
| --- | --- |
| Previous accepted identity | generation 0, revision 31, timestamp 25443.681117198 |
| Selected new identity | generation 0, revision 32, timestamp **25444.063689434** |
| Post-preparation readiness wait | **0.104989321 s**, below 5 s |
| Original source age at readiness | **0.488691875 s** |
| Age after reconstruction/qualification | **0.938170743 s** |
| Final authoritative capture time | **25445.009311096** |
| Final original acquisition age | **0.945621662 s**, below 1 s |
| Final freshness headroom | **54.378338 ms** |

Both original camera tokens were `(14.2, 6.566667009145021)`, with acquisition
timestamp 25444.063689434 and 480×640 depth arrays. Camera identities, exact
tokens/times, intrinsics/extrinsics and generation matched exactly one retained
camera-only `MappingInput`. The final authoritative capture retained the same
generation/revision/timestamp and calibration records.

This newly published exposure was already in flight before preparation ended,
as permitted by the task. Its original acquisition time was preserved. The
previous accepted warmup publication was not reused, and no timestamp was
refreshed. There was one planning source capture and zero active-world obstacle
refreshes. Freshness was checked after inference; later synchronous planning
and evaluation did not restamp the accepted fixed-time static world.

## 4. Sensor-derived Post and Occluder geometry

The unchanged `reference.reconstruct` and `visibility.qualify` ran once on the
selected frames, synchronized robot snapshots and whitelisted workspace/support
configuration. No fixture names, dimensions, poses, semantic identities or
physical-reference colliders were reconstruction inputs. The result used the
existing 12,000-point cap and produced seven eligible clusters/two numerical
hypotheses. Object labels below were assigned only during post-hoc comparison.
Saved sensor results were hashed before that physical-reference inspection.

| Final model | Post, component 1 | Occluder, component 0 |
| --- | --- | --- |
| Measured component points | 3,348 | 6,487 |
| Robot-envelope points | 0 | 0 |
| Center XYZ, m | (0.450000045, −0.000413367, 0.349335331) | (0.787551741, −0.300554363, 0.529048788) |
| Dimensions XYZ, m | (0.094001105, 0.124827453, 0.265329971) | (0.088897306, 0.405108916, 0.405903510) |
| Yaw, rad | −1.937704721e−18 | −1.937704721e−18 |
| Final qualification | Qualified | Qualified |
| Residual score, not probability | 0.999999798 | 0.999999816 |

All six face bounds qualified under the existing single-convex-yaw-box empirical
model. Direct evidence and selected calibrated edge witnesses were:

| Face | Post | Occluder |
| --- | --- | --- |
| −X | 363 direct points | 181 edge transitions, primary camera / +Z carrier |
| +X | 284 direct points | 2,790 direct points |
| −Y rear | 68 transitions, primary / +X carrier | 186 transitions, primary / +X carrier |
| +Y | 1,129 direct points | 24 direct points |
| −Z lower | 62 transitions, side / +Y carrier | 153 transitions, primary / +X carrier |
| +Z | 56 direct points | 329 direct points |

Each selected edge witness spanned three supported transition bins. Post's
visibility qualification had valid +X/+Z primary patches and a six-pixel 3×2 +Y
side patch; excluded out-of-FOV probes supplied no evidence. Occluder had nine
supported primary probes on each of +X/+Y/+Z. Hidden bounds remain conditional
on the declared box model and measured edge transitions, not guessed support
height or simulator dimensions.

Post-hoc eight-vertex reference checks found both inferred boxes conservatively
contain their copied physical solids: minimum margins **2.000321 mm** (Post) and
**2.000092 mm** (Occluder). Maximum absolute face-extent errors were outward
excesses **3.329660 mm** and **6.896913 mm** respectively. These are geometry
scores, not trajectory clearance or universal uncertainty guarantees.

## 5. Qualification failures and robot contamination

There was **no final qualification failure among eligible nonrobot components**.
Five entirely robot-envelope clusters, totaling **2,040 points**, were rejected
as possible robot observations; each had all its points in the envelope. No
mixed robot-connected component was silently omitted.

The saved reference prototype classified component 1 as rejected for
`unsupported_or_merged_box_surface`. The already published visibility-aware
qualifier then qualified it using its measured patches and six bounds, exactly
as designed; no new rule or threshold was introduced. Both final model records
are retained alongside that intermediate result.

Current free-ray checks reported **0/76 contradictions for Post** and **0/51 for
Occluder**; accepted plane patches also had zero contradictory FREE probes.
These are bounded measured-ray checks, not exhaustive certification of all
camera rays or exact agreement with every FREE voxel boundary classification.

## 6. Combined collision-world accounting

| Quantity | Count |
| --- | ---: |
| Original selected occupied cuboids | 64 |
| Subsumed by Occluder model | 23 |
| Subsumed by Post model | 19 |
| Original cuboids retained verbatim | 22 |
| Added inferred boxes | 2 |
| Total camera proxies | **24** |
| Separate known support Plane | 1 |
| Total registered obstacles | **25** |
| Original selected floor cuboids | 0 |

Every subsumed cuboid passed conservative eight-corner containment, and a
post-run numerical recheck independently verified all 42. The 22 retained
descriptors preserve exact centers/dimensions. Original IDs form a complete,
disjoint retained/subsumed partition. All IDs and inferred associations are in
the research receipt; numeric registered obstacles are in `plans.json`.

The known support was preserved, with no finite floor observations dropped.
The original immutable occupancy read view and FREE/UNKNOWN labels were not
completed or rewritten. Models are explicitly labeled `rgbd_inferred_box`.
The original map had 589 occupied/supported cells and selected 64, omitting 525
under its existing bounded extractor; this experiment preserves the entire
**selected union**, not every occupied cell in the map. No GT fixture collider
entered the selected planning world.

## 7. Lula RRT outcome

The unchanged enhanced-model base guard passed. Selected RRT succeeded with
**3 native waypoints**, path length **2.663699439 rad**, and trajectory duration
**1.274673680 s**. Seed and iteration bound remained 123456 / 10,000. The normal
route and spline collision/limit checks passed, including **183 route samples**
and **334 trajectory samples**. Finger-envelope checks remained enabled over
the existing 0.001–0.005 m range. No route, direct-motion or GT fallback was used.

## 8. Matched GT and unobstructed comparisons

All three plans used identical actual seven-joint starts and the same validated
IK joint goal, at one unadvanced physics instant. Endpoint arrays match exactly.
Only the validated numeric joint goal crossed the GT-control seam; GT obstacle
geometry did not enter reconstruction or the RGB-D world.

| Condition | Waypoints | Trajectory duration, s | Planner wall time, s | Independent result | Executed |
| --- | ---: | ---: | ---: | --- | --- |
| GT control | 3 | 2.209641482 | 0.489131722 | Accepted | No |
| Reconstructed RGB-D | 3 | 1.274673680 | 0.292259539 | Accepted | **Yes** |
| Support-only unobstructed | 2 | 1.037396800 | 0.233646879 | Physical collision proven | No |

The support-only snapshot retained the selected world's Plane exactly. GT and
RGB-D support descriptors retain their original captured numeric XY origins,
which differ by at most **2.384186e−9 m**; their z=0 halfspace, orientation and
50 m clipped-domain dimensions agree. Those original values were preserved,
not normalized or changed to obtain a comparison.

There was no fourth same-source occupied-only counterplan. The experiment
establishes combined sensor-world avoidance relative to the support-only
counterfactual; it does not isolate inference's causal benefit over an otherwise
identical occupied-only packet. Historical unsafe V5 plans are separate evidence.

## 9. Independent physical evaluation

The existing evaluator used copied actual cooked robot solids, physical obstacle
references and independent URDF FK. Alignment passed: maximum translation error
**3.03236e−7 m**, rotation-matrix error **3.14873e−7**, within unchanged allowances.
It did not alter any planning world.

| Condition | Evaluated samples | Queries | Status |
| --- | ---: | ---: | --- |
| GT | 923/923: 345 route + 578 spline | 30,459 | `ALLOW_SAMPLED_NUMERIC_CLEARANCE` |
| RGB-D | 517/517: 183 route + 334 spline | 17,061 | `ALLOW_SAMPLED_NUMERIC_CLEARANCE` |
| Unobstructed | 19/384; stopped on route collision | 627 | `VETO_PREDICTED_PHYSICAL_COLLISION` |

Selected evaluation had no collision witnesses or unresolved queries. Its
minimum nonmount clearance **lower bound** was **0.645213 mm**. Overall minimum,
including the mounted base/Plane pair, was **2.384186e−8 m**; that mounted-support
number must not be presented as moving-arm clearance.

The unobstructed route's panda-hand/Post intersection was independently proven
at route sample 18, with a verified positive intersection ball of radius
**1.491023 mm**. Early termination explains its incomplete sample count; it is
a positive collision witness, not an unresolved acceptance. It was never
executed. The independent evaluator covers sampled external geometry and fixed
measured fingers; production enhanced-model self checks remain separate.
Neither method supplies a general continuous swept-volume certificate.

## 10. Genuine obstacle-induced detour

**Yes, under the declared comparison.** The original direct joint segment is the
two-waypoint unobstructed native route, whose physical Post obstruction was
proven. The selected route has three waypoints and maximum seven-joint distance
from that segment **1.057749167 rad**, exceeding 0.05 rad. Starts/goals match,
physics did not advance during comparison, and independent selected route and
spline evaluation passed. The result is supported by obstruction and physical
checks, rather than route length or visual appearance alone.

## 11. Actual execution and contacts

| Measurement | Selected RGB-D execution |
| --- | --- |
| Result / strict measured outcome | Success / true |
| Maximum observed joint displacement | **1.107707798 rad** |
| Final TCP position error | **0.000874674 m** |
| Final TCP orientation error | **0.002424932 rad** |
| Worst joint tracking error | **0.036278454 rad** |
| Minimum sampled moving-arm solid clearance | **0.041143002 m**, against Post |
| PhysX contact/proximity points / penetrating pairs | **0 / 0** |
| Contact buffer saturated | No |
| Unique sampled physics steps | **110**, steps 404–513, no gaps |
| Timed commands / recorded actions including hold | 109 / 110 |
| Final hold / applied arm velocity targets zero | True / true |
| Maximum measured terminal arm speed | 0.004443327 rad/s, below existing 0.01 limit |

The operation retained 220 pose samples, with duplicate physics samples handled
by the existing reducers. The full trace has 221 samples. Actual physical
clearance above comes from sampled cooked-solid distance evaluation; it is
different from the conservative pre-execution AABB separation lower bound.
Zero reports include positive-separation proximity reports, not only impacts.

The existing executor's temporary TGS requested position-iteration setting ran
32 → 255 and restored its original session opinion. Final hold and restoration
are recorded true. This was existing execution behavior; no drive/gain/limit or
saved physics setting was edited by the task. Ordinary START setup separately
moved the robot and succeeded in 84 steps; it is not counted as selected GOAL
execution.

## 12. Timings and performance

| Stage | Time |
| --- | ---: |
| Native subprocess, including startup and cleanup | **55.570406590 s** |
| Warmup | 8.018340107 s |
| Readiness | 0.104989321 s |
| Reference reconstruction total | **410.171494 ms** |
| Robot envelopes / projection-fusion-clustering / reference qualification | 10.066842 / 251.989228 / 148.115424 ms |
| Complete research capture, including visibility/final capture/combination | **445.018895 ms** |
| Selected native RRT alone | 5.705214 ms |
| Selected route / spline validation | 69.585474 / 187.864736 ms |
| GT / RGB-D / unobstructed physical owner waits | 1.496997 / 0.933386 / 0.206756 s |
| Request to first selected command | **4.194568715 s** |
| Selected execution wall / simulation time | **8.410095137 / 1.816666761 s** |
| Execution control interval p50 / p95 / worst | 74.080 / 127.754 / 157.974 ms |

The native subprocess ran UTC **15:43:42.338787–15:44:37.909418**. Paired planning
and independent evaluator waits block the Kit owner to preserve the physics
instant; the longest recorded Kit interval was about **4.244 s**. No real-time
responsiveness claim follows. The 54.378 ms source-freshness headroom is narrow;
one successful timing does not demonstrate repeated eligibility or reliability.

## 13. Cleanup, restoration and retained artifacts

All native measurement contracts passed: runtime closed, source unchanged,
all 11 cooked bodies captured, no internal runtime error, and stable scene
signature/fixture state. Strategy diagnostics report closed. Existing harness
hooks, observer hooks and callbacks were restored/removed. Owned Kit and worker
children exited; only the saved container sleep process remained before stop.

`isaac51` was restored to stopped/exit 137, with no GPU compute process and
**0 MiB** memory. Before/after Config/HostConfig/Mounts SHA256 is identical:
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No stream, container configuration, network, volume, VM, SDK or launcher changed.

Raw camera frames, retained input, frozen occupancy, references, motion trace,
native log, receipts and test logs remain under the new output root in section
2, approximately 30.7 MB for `attempt-1`. Published summaries bind their hashes;
large raw files remain outside Git. Prior experiments and recovery reference
`refs/recovery/v5-realignment-20261114-worktree` at
`edc1b467c6bf03c192d0d9dccce2b538309a024f` remain preserved.

## 14. Final assessment and next action

**SUCCESS:** the unchanged nominal `partial` scene demonstrated a fresh,
qualified sensor-derived collision world, a genuinely obstacle-induced native
RRT detour, independent sampled physical acceptance and successful contact-free
native execution. No blocking failure stage occurred.

Review this frozen one-scene evidence before deciding any further thesis claim
or separately authorized development. It does not validate production RGB-D
integration, other scenes, general hidden-object reconstruction, payload
carrying, dynamic obstacles or continuous physical safety. This task stops after
publication; no second experiment, algorithm change, integration or GUI work
was started.
