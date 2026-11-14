# Final RGB-D global RRT optimization and demo hardening

Running record, 2026-11-14 UTC. Started 19:55 UTC; stop development by 21:25 UTC
at the latest. Clean fetched remote/local starting HEAD:
`84585b777afc98b86d0e4e9bdb565420dae643c1`, branch
`feature/rgbd-rrt-autonomous`. Original V5/recovery refs and all earlier results
remain preserved. `isaac51` initially stopped/exit 137, no GPU process or stream.

## A — one measured optimization

Reviewed AGENTS, all four requested reports, numerical/current worker ownership,
immutable snapshot/source contracts, admission, benchmark and physical-veto tests.
The successful normal-runtime source/IK/route/executor path is the baseline.
No geometry, freshness, admission, collision/proxy or executor change is proposed.

Existing profiles: combined map/inference ~465–468 ms, reconstruction ~214–232 ms,
transport residual ~135–179 ms; five distinct occupancy arrays total 29,890,800
bytes. Their semantics differ and they must all remain. The previous checked
mixed-radix voxel grouping stays unchanged. Remaining numerical work includes
ray traversal, clustering/union, projection and six-face qualification; no second
geometry optimization was selected.

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

Estimated production diff: ~10 lines in `runtime/process_worker.py`; tests cover
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
