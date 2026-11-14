# Autonomous RGB-D performance and freshness recovery

Running engineering record, 2026-11-14 UTC. Start 18:09 UTC; bounded budget
45–90 minutes, stop development by 19:39 UTC at the latest. Continue on
`feature/rgbd-rrt-autonomous` from clean, fetched remote/local HEAD
`99be1540b98986d8836c9238247f060b4216a74b`. Original V5 remains unchanged.

## A — profile and optimization decision

Reviewed AGENTS, the integration report and twelve retained native measurements,
the successful research postfix proof, V5 documentation and actual worker,
publication, capture, numerical and harness code. Container `isaac51` initially
stopped/exit 137, original official 5.1.0 image, no GPU compute process or stream.
No native matrix replay before profiling.

Baseline partial: 0/4 fresh authoritative captures, 1005.905–1062.561 ms original
source ages; accepted mapping .442–.576 Hz. Selected inference 284–319 ms,
combined compute approximately 560 ms, transport roughly 145–190 ms and queued
work consume most freshness margin. Shared mapping camera-batch access typically
.18 ms (already acquired semantic buffers); semantic batch acquisition about
20 ms. These are consumer timings, not camera hardware exposure latency.
Publication-to-readiness/request adds existing owner/update and harness delay.
The prior harness already deferred large report serialization and GT hashing;
its corrected partial still failed, initially reaching readiness at age 997 ms.

New bounded offline profile: use only hash-verified partial attempt-4 camera and
robot payload (20,411,990 bytes), original scene workspace/support settings and
original algorithm. Three fresh worker computations with cProfile: 507–552 ms.
One decomposition: inference 301 ms, occupancy/ray work 146 ms, extraction 29 ms,
immutable snapshot 26 ms; inference includes voxel grouping 106 ms, clustering
62 ms, robot envelopes ~10 ms and numerical qualification ~65 ms. Result payload
29,924,959 bytes. Input encode 9–26 ms/decode 5–6 ms; result encode 39–78 ms/decode
6–19 ms. These local serialization microtimings exclude pipe wait/scheduling;
previous native worker-minus-compute is the combined transport residual, not a
separately measured pure wire latency. Full timings and CPU profile are retained.

**Choose Option A, one numerical implementation optimization:** voxel grouping
currently performs expensive structured three-column integer sorting via
`np.unique(axis=0)`. Encode the same three integer coordinates into a checked,
collision-free mixed-radix int64 key, preserving lexicographic group order and
input-order bincount accumulation. Fall back to the original path if the key
range would overflow. No voxel size, points, cap, bins, thresholds, model or
qualification predicate changes. This is representation/ordering optimization,
not removal of geometric work or new shape inference.

An isolated scratch prototype produced exactly equal voxelized arrays for five
cloud sizes through 100,000 points and exactly equal centers/dimensions/rotations/
yaw for both partial models. Four baseline inference runs had median 281.256 ms;
four candidate runs median 220.263 ms (21.7% lower). Projection/fusion/clustering
fell from ~199–210 to ~117–131 ms. This is promising, not native evidence.

Estimated production diff: approximately 20 lines in the existing SDK-free cloud
voxel utility, plus focused equivalence/overflow/archived regressions. The shared
semantic path calls the same utility, so exact output equivalence and existing
semantic regression tests are required. Production research/visibility files
stay unchanged; original frozen research files/evidence remain immutable.

Rejected request-scoped Option B for this checkpoint: normal GlobalRrtStrategy
capture is synchronous. Correct per-request use of the existing bounded worker
would require asynchronous admission/waiting, a request token through replaceable
pending work, cancellation/reset/rejection barriers, and exactly-once inference
in the child to prevent two queued jobs reconstructing one request. Binding the
accepted product to its corresponding immutable occupancy remains possible, but
is materially larger than this measured local sorting correction. Do not combine
that architecture with Option A. Scheduling/worker/process contracts remain
unchanged and continuous inference still runs once per accepted mapping compute.

Chronological decisions:

- 18:09–18:13: verified clean branch/remote and original stopped environment;
  profiled archived payload before any Kit run. Identified structured voxel
  sorting as avoidable cost rather than blaming all time on the box model.
- 18:13: select reversible Option A after exact numeric scratch comparisons.
  Native throughput/freshness improvement remains to be established. No threshold,
  resource, worker or source-selection policy change is authorized by this choice.

## Validation plan

Before native: exact voxel grouping/cap/order equivalence, negative/extreme
coordinates and overflow fallback; archived box/face/robot/FREE evidence
comparison; existing mapping/frozen-source/lifecycle/cancellation/configuration,
ordinary GT/direct/skills/BT and semantic regressions; architecture checks.
Then original partial through normal autonomous runtime plus independent physical
veto. If a valid planning world is reached, run up to two identical repeats,
recording every result. One GT native control next; secondary geometry only if
primary path works and budget remains. No seed/fixture/threshold search.
