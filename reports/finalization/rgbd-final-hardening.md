# Final RGB-D global RRT optimization and demo hardening

Final assessment, 2026-11-14 UTC. Started 19:55 UTC; bounded sprint completed
before the 21:25 UTC limit. Clean fetched remote/local starting HEAD:
`84585b777afc98b86d0e4e9bdb565420dae643c1`, branch
`feature/rgbd-rrt-autonomous`. Original V5/recovery refs and all earlier results
remain preserved. `isaac51` initially stopped/exit 137, no GPU process or stream.

## A — one measured optimization

Reviewed AGENTS, all four requested reports, numerical/current worker ownership,
immutable snapshot/source contracts, admission, benchmark and physical-veto tests.
The successful normal-runtime source/IK/route/executor path is the baseline.
No geometry, freshness, admission, collision/proxy or executor change was made.

Existing profiles: combined map/inference ~465–468 ms, reconstruction ~214–232 ms,
transport residual ~135–179 ms; five distinct occupancy arrays total 29,890,800
bytes. Their semantics differ and they must all remain. The previous checked
mixed-radix voxel grouping stays unchanged. Remaining numerical work includes
ray traversal, clustering/union, projection and six-face qualification; no second
geometry optimization was selected. Inspected the retained post-voxel-optimization
CPU profile: ray traversal 92 ms, clustering 85 ms, qualification 69 ms,
projection/fusion 55 ms, robot envelopes 10 ms, snapshot construction 27 ms in
one 446 ms computation. These are nested/cumulative measurements, not additive
independent costs. Owner-side acquisition is already shared; existing publication
and queue telemetry remain the source of native timing measurements.

**Selected single candidate: avoid redundant copies of already immutable bytes
inside `_TransportPickler`.** It currently calls `array.tobytes()` for every
read-only array, even when the entire C-contiguous array is backed by immutable
Python bytes. Reuse that full bytes object when its length exactly matches the
array's byte length. Keep copying mutable-backed, partial, strided/transposed or
otherwise unsupported views. Decode still constructs bytes-backed read-only
NumPy arrays in the receiving process. No shared-memory IPC or mutable sharing.

Before production edits, profiled a hash-verified archived admitted-partial input
and a fresh real MappingProcessor completion. Eight alternating old/candidate
encodes for each kind; all wire bytes were identical (one SHA per kind across
all encoders/trials). Median encode cost:

| Payload | Bytes | Original ms | Candidate ms | Original / candidate traced peak MB |
|---|---:|---:|---:|---:|
| Camera/robot input | 20,411,990 | 35.793 | 19.351 | 42.146 / 21.735 |
| Mapping completion | 29,924,959 | 65.246 | 45.516 | 62.039 / 32.145 |

Expected benefit is about 36 ms less encode work across the two directions and
lower temporary memory, not a promise that pipe/queue time disappears or mapping
reaches 1.8 Hz. These in-memory serialization measurements exclude actual pipe
waits; native residuals will be measured independently. No numerical evaluation
order, result field, precision, timestamp or qualification predicate changes.

Final production diff: nine added / one removed lines in
`runtime/process_worker.py`; tests cover
exact output, irreversible immutability, mutable backing, view/pathological
layouts, actual cross-process isolation, worker barriers and archived full
completion/geometry/world equivalence. No worker/service/protocol/configuration
change. Abandon this optimization if those equivalence checks fail.

## Reliability protocol declared before native execution

D1: one fresh ordinary ungated partial request, original configuration/seed,
production admission and ordinary Lula IK, independent physical veto.

D2: one persistent runtime with three ordinary reconstructed RGB-D requests:
**GOAL → START → GOAL**, eight seconds of cooperative settling/observation between
completed legs. Every leg, including the return, must obtain its own sensor
world and pass native planning plus the independent physical veto before any
selected arm command. The original direct START setup is used only before test
obstacles are introduced, as in the established harness. It is never used as an
unvalidated return. No teleport, fixture removal, reset or GT fallback.

If a leg fails, record and check cleanup; do not perform dependent motion from an
unestablished start or claim a repeat detour. Continue safe independent testing.
A small isolated session harness will reuse the existing runtime, source capture,
planner observer/evaluator and artifact writers; no production session framework.
No original research publication wait will be used.

Chronological decisions:

- 19:55–19:58: verify branch/environment and inspect current source/profiles.
  Measured redundant immutable-byte copies before changing production. Chose only
  this bounded serializer optimization; no field compression/removal or geometry
  change. Predeclared persistent sensor-planned return rather than assuming direct
  START motion is collision-safe.
- 19:59–20:10: implemented nine added/one removed production lines in the
  existing serializer; no other production change. A new test initially assumed
  that decoding/re-encoding must reproduce the saved original pickle memo. The
  original encoder itself changes that packet from 29,924,959 to 29,925,060 bytes.
  Comparing old/new encoders on the same decoded object gives identical SHA-256;
  all decoded arrays and fields are bitwise equal. Corrected only that new test
  oracle; no historical expectation or numerical tolerance changed. Initial
  failing test source and diagnostic hashes remain in the new raw evidence area.
  Seven transport tests pass, including eight hash-verified original archived
  scene inputs, fresh MappingProcessor results and a real child-process roundtrip
  of the complete 29.9 MB completion. First full suite: 1431 PASS / 120.826 s.
- Added only a two-line optional research follow-up hook in the existing
  benchmark and an isolated `persistent_trial.py`; no production session code.
  It retains distinct source/plan/veto records per request and uses three normal
  MoveToPose calls. Three focused sequence tests verify normal targets, distinct
  receipts and stopping dependent requests after failure or absent physical veto.
- 20:11: complete offline gate PASS: 1434 full host tests / 122.803 s;
  68 focused host tests / 18.456 s; same 68 native Python 3.11 tests / 25.581 s.
  Architecture, Python syntax and whitespace PASS. Native test gate starts only
  after these checks, with the saved container running only `sleep infinity` and
  no GPU compute process. No safety/numerical equivalence failure was observed.
- 20:11–20:13: D1 independent ungated partial PASS, followed by persistent run.
- 20:13–20:15: persistent first motion PASS; return rejected by existing measured
  surface qualification, zero commands. Stop dependent outward repeat. Same
  runtime/workers remained healthy; no attempt to force a return.
- 20:16–20:17: after safe cleanup, fresh independent confirmation PASS. Sealed
  original persistent receipts and return sensor bytes before diagnostic replay.
- 20:17–20:19: unchanged Isaac Python 3.11 transport replay confirms lower copy
  cost and byte-identical packets. Offline rejected-return replay reproduces
  published status/reason exactly. One optional original sparse scene safely
  refuses the 65-proxy world; no other scene or algorithm attempted.
- 20:19–20:21: a one-line research archival correction retains the rejected
  follow-up's original inferred record in future runs, matching the existing
  single-request harness. Added a focused assertion; no motion/admission change
  or rerun seeking a better exposure. Original failed request remains untouched;
  its diagnostic replay is explicitly separate. Final unit/check receipts and
  historical preservation verified; stopped container/stream state restored.

## B/C — implementation and exact equivalence

The full bytes-backed buffer is already irreversibly immutable. Reusing its
reference inside the sender's pickler avoids an extra temporary `tobytes()`;
the length-prefixed pipe still sends the complete packet and the receiver owns
new immutable bytes. No parent/child mutable memory is shared. Mutable-backed,
owning, sliced, non-C-contiguous and partial views keep the copying fallback;
object-dtype rejection is unchanged. Mapping's five occupancy arrays and their
precision, TTL, FREE/OCCUPIED/UNKNOWN semantics and original stamps all remain.
The result is still about 29.925 MB; this does not redesign IPC or remove its cost.

Replayed the same trusted archived camera/robot inputs for partial, static, v43,
empty, large, sparse, narrow and verified_occlusion. Compared original/transferred
inputs and fresh MappingProcessor outputs recursively, including cached robot
mesh/BVH attributes, all occupancy arrays, original source/calibration, extracted
proxies, model bounds, component/robot decisions and visibility witnesses.
All deterministic fields and numerical arrays match exactly, bit for bit. Only
explicit measured-duration fields are excluded from *fresh computation* equality;
transport roundtrip equality includes those durations too. No floating tolerance
or evaluation order changed. Separately, the complete before-change completion's
camera collision descriptors and eight-corner union audit match exactly after
optimized transport. Existing floor, 64-budget, stale/reset/late-completion,
cancelled-request, source-error and collision tests remain unchanged.

New tests cover nonfinite/extreme numeric values, big-endian arrays, empty arrays,
full/partial/transposed/reversed/strided and dtype views, nested immutable aliases,
mutable backing, actual child PID separation and irreversible receiver immutability.
Existing voxel tests cover duplicate points and mixed-radix overflow; no voxel
implementation was changed. Full suites include GT/direct/BT and GUI architecture
boundaries. The original published research/geometry code and fixtures are intact.

The reproducible profile freezes the original copying reducer explicitly, so
running it after the change still measures the original algorithm. Native
Python 3.11 / NumPy 1.26.4, eight old/current pairs on the same hash-verified input:

| Payload | Original encode ms | Optimized encode ms | Traced peak MB, old / new |
|---|---:|---:|---:|
| Input | 33.706 | 16.945 | 42.146 / 21.735 |
| Completion | 63.273 | 42.830 | 62.039 / 32.145 |

All packets in each comparison are byte-identical. About **37.2 ms** of combined
encoding and **20.4/29.9 MB** of per-direction temporary allocation are avoided.
Peak columns are separate encoding calls, not an asserted process RSS reduction.
Keep this compact optimization for measured copy/memory savings. Native scheduling
variance is larger than some latency improvements; do not claim a mapping-rate gain.

## D — actual native execution and persistent refusal

Four exclusive, bounded Kit processes, all at commit
`1054ac54edd3ff6438fe9c56751dadc6b77decc8`, exited normally with all five measurement
contracts true: runtime closed, code unchanged, no internal error, all 11 cooked
robot bodies retained, unchanged physical scene. The later one-line research
archive correction was validated offline/native Python; it cannot alter these
motion receipts. Each invocation retained its exact command, UTC times, commit,
and exclusive Docker/GPU preflight. No Kit instances overlapped.

Every selected request was ordinary MoveToPose with **zero ungated research delay**,
production first-new-publication admission, one authoritative source capture and
ordinary Lula IK. No original research readiness wait, selected GT joint target,
GT obstacle injection, source retry, reset or hidden fallback. Test setup and
normal warmup are unchanged; these do not test cold startup.

| Trial / request | Outcome | Source age ms | Headroom ms | Admission ms | Commands |
|---|---|---:|---:|---:|---:|
| D1 independent | Executed | 882.201 | 117.799 | 426.171 | 110 |
| D2 outward | Executed | 972.496 | 27.504 | 430.063 | 110 |
| D2 return START | Geometry refused | See below | — | 223.285 | 0 |
| Independent confirmation | Executed | 811.021 | 188.979 | 1189.496 | 110 |
| Optional sparse | 64-proxy limit refused | — | — | 518.054 | 0 |

All three successful outward requests have **0.874674 mm** TCP error,
**0.00242493 rad** orientation error, **1.107708 rad** maximum actual joint
displacement and zero reported contact/proximity/penetration points. Their
three-waypoint routes deviate **1.057749 rad** from the same-endpoint straight
joint segment. Sampled moving-arm clearance is **41.143 mm**. Actual starts,
routes, spline q samples and spline time samples are exactly equal to the previous
admitted-0 baseline, as well as each other. Both inferred obstacles plus 22 retained
cell proxies preserve all original 64 selected cuboids in **24 camera proxies**;
the allowlisted known support Plane remains separate.

The selected reconstructed plans pass the independent cooked-geometry physical
veto before execution. Matched-start/joint-goal GT controls pass; occupied-only and
support-only controls are physically vetoed and never executed. GT control joint
goals are obtained *from* the selected ordinary plan afterward. This confirms
preserved native GT planning, without claiming a new GT execution regression run.
GT/direct execution paths are also covered by the full unchanged unit suite.

Selected Lula RRT latency is approximately 6 ms. D1 planning/validation including
research matched controls and independent veto takes 4.380 s, request-to-first
command 4.934 s, actual execution 8.584 s. Do not present instrumentation-blocked
latency as pure Lula or production planning cost. Native sample traces, tracking,
terminal hold and solver-restoration evidence remain in the raw receipts.

In D2 one runtime ran outward and then, after the declared eight-second cooperative
settle, attempted the ordinary return. It accepted the first new mapping identity
`(0,59,41662.752491684)` and refused at reconstruction assembly, before creating a
planner or issuing any return command:
`component_1: insufficient_observed_three_axis_patch_or_residual`.
The original rejected-publication archive has that same identity and original
camera/robot input, with a **770.653 ms post-failure read-view age**. This is an
archive observation, not a fabricated authoritative live-capture age; successful
source capture preceded the geometry refusal.

After sealing the failed receipt/input, replaying only those original depth,
calibration, workspace and synchronized robot observations reproduces the exact
published rejection. One 5,095-point nonrobot component qualifies; the 2,624-point
nonrobot component lacks the required box-surface support. Six other components
have every point in synchronized robot envelopes and are rejected as robot-derived.
No GT is used by the replay. Do not infer a new geometry rule or tune a threshold
to repair this posture-dependent observation limitation.

Mapping worker PID **943**, semantic PID **942**, generation **0**, and reset
count **0** persist across both D2 requests. Worker/source/processing errors remain
zero, robot filter is `ready` with zero pose skew, runtime stays running without
an internal error. Return joint displacement is exactly **0 rad**, no contacts,
and sampled physics coverage is complete. Per-request strategy remains open
between requests as intended; final shutdown closes it and removes all worker
processes. `final_hold=false` on the refused request denotes that no new arm
command was issued; the previous GOAL targets/positions remain unchanged.
The dependent third outward request is **not run**. D2 proves a clean second
request refusal, not repeated successful planning/execution in one session.

Optional original sparse: one qualified inferred box plus all 64 original
unsubsumed selected cuboids requires **65** camera proxies. Offline eight-corner
count confirms the original refusal; there is no repacking, floor removal or
budget increase. Zero selected commands, zero displacement, zero contacts,
clean shutdown. No other original scene was launched.

## Throughput, remaining limitations and demonstration decision

Pre-request accepted-publication windows, using the same reduction rule as the
prior report. Durations below are medians of matched accepted completion records;
they exclude research physical-veto pauses during execution.

| Partial trial | Mapping Hz | Reconstruction ms | Full compute ms | Transport residual ms | Queue ms |
|---|---:|---:|---:|---:|---:|
| D1 | 1.015 | 218.351 | 485.675 | 129.202 | 91.291 |
| D2 first | 1.118 | 223.109 | 490.519 | 148.446 | 140.460 |
| Confirmation | 0.837 | 238.104 | 518.198 | 140.838 | 139.797 |

Prior successful reports measured roughly .9–1.2 Hz, 465–468 ms compute,
135–179 ms transport residual, 118–154 ms queue. Current conditions are comparable,
not paired native scheduling samples. Compute and geometry are numerically
unchanged. Copy microbenchmarks establish the optimization's benefit; native
transport remains substantial and **1.8 Hz is not achieved**. Minimum source
headroom is now only **27.5 ms**, so there is no demonstrated freshness-reliability
gain over the prior 32.1 ms minimum. Expiry must remain an explicit possible refusal.

Final classification: **`RGBD_DEMO_CANDIDATE`**.
**NO-GO for promoting RGB-D to demo-ready or promising repeated-session success.**
Retain the small verified copy optimization and current opt-in feature. Proceed
to GUI finalization with validated static GT global RRT as the primary demo;
nominal partial RGB-D can be a clearly labeled secondary candidate with honest
refusal behavior. No further reconstruction/IPC optimization is recommended here.

Recommended experimental configuration remains:

```toml
[motion]
strategy = "global"
global_collision_model = "enhanced"
global_obstacle_source = "frozen_rgbd"
global_rgbd_representation = "reconstructed"
max_obstacles = 64
```

Keep original two 640×480 cameras, semantic 5 Hz / mapping 2 Hz, 1,024 rays,
10 mm voxels, seed 123456, 10,000 iterations and strict <1 s freshness. GT remains
independently selectable. No GUI, launcher, SDK, collision resource, camera,
configuration, geometry threshold, factory, worker lifecycle or executor change.
Python 3.11/NumPy 1.26.4 compatibility is tested; no new dependencies or increased
resource requirements. Final university workstation GUI/native behavior still
requires its own review; these tests ran headless on the original cloud L4.

Historical limitations remain: static/v43/empty floor/base guard; sparse 65 proxies;
narrow hidden bound; large extra unqualified components/freshness; verified_occlusion
missing surface patch. New persistent return observation limitation remains.
No unsafe selected motion or uncontrolled runtime state was observed. Sampled
clearance/contact evidence is not continuous safety certification, arbitrary
obstacle support, dynamic-world handling or payload-carrying validation.

## Receipts, preservation and reproduction

Final checks after the research archival correction: **1434 full host tests PASS /
122.514 s**, plus **10 native-Python transport/session tests PASS / 10.352 s**.
The earlier native-Python gate was **68 PASS / 25.581 s**. Final architecture,
Python syntax and whitespace checks PASS. No production edit followed the
pre-native full-suite gate or any native run.

Supporting [numeric native results](../../research/rgbd_box_geometry/evidence/final_hardening/summary.json),
[validation receipt](../../research/rgbd_box_geometry/evidence/final_hardening/validation.json),
[transport profile](../../research/rgbd_box_geometry/evidence/final_hardening/transport-profile-native.json),
[rejected-return replay](../../research/rgbd_box_geometry/evidence/final_hardening/return-diagnostic-replay.json),
[preservation](../../research/rgbd_box_geometry/evidence/final_hardening/preservation.json)
and [raw manifest](../../research/rgbd_box_geometry/evidence/final_hardening/raw-manifest.json).
Full external inputs/traces and exact commands remain under
`outputs/motion-planning-v5/final-hardening-20261114`; large cameras/pickles stay
outside Git. The publication is numerical evidence, not a raw archive backup.

Preservation verifies all 48 original published files, 16 static/v43 inputs/references,
24/21/21/29 older native files, 242 integration + 112 performance + 131 admission
raw files, and all four requested prior reports unchanged. Original V5 local/remote
still `71c07f815410a308dc7ea176148c017055c7fd7a`; recovery reference still
`edc1b467c6bf03c192d0d9dccce2b538309a024f`.

Original stopped/exit-137 `isaac51` and absent streaming/GPU process restored.
Config/HostConfig/Mounts hash remains
`8ba65b71c877882bdbae47aef4293b87aef89627f49856616ffce347defa4918`.
No Docker/network/VM/SDK configuration change or stream interruption.

Reproduce with fresh paths, one Kit at a time:

```sh
uv run --offline --no-project --python 3.12 --with numpy==1.26.4 \
  --with py_trees==2.6.0 --with scipy==1.15.3 \
  python -m unittest discover -s tests/unit -t . -q
python3 scripts/check_architecture.py

docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.autonomous_trial --scenario partial --request-delay 0 \
  --output outputs/motion-planning-v5/NEW-independent/attempt-1
docker exec -w /workspace/robot-skill-stack -e ISAAC_SIM=/isaac-sim isaac51 \
  timeout --signal=TERM --kill-after=5s 600s ./run_isaac_python.sh \
  -m research.rgbd_box_geometry.persistent_trial \
  --output outputs/motion-planning-v5/NEW-persistent/attempt-1
```

The container must be intentionally running for these reproduction commands;
the final delivered environment is stopped. Stop after publishing this report;
no GUI work, new algorithm, architecture expansion or main integration was started.
