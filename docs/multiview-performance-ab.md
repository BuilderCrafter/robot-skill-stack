# Controlled multiview performance A/B investigation

**No significant refactor-related computational regression was found for the
ordinary four-object scene.** There is a repeatable **3.54% slowdown**:
409.69 → 424.19 ms, primarily the unified metric graph. The numerical fitter is
unchanged, fits identical clouds, and receives four calls in both versions.

**A separate correctness regression is confirmed:** on the saved gripper-occluded
native batch, original main qualifies the Square as an accurate, graspable cube;
the refactor returns UNKNOWN. Its lower processing time on that batch is not an
optimization. Production code was not changed during this investigation.

## Commits and production source comparison

- A, original main: `5b61d4442d74e3afd76420ffc11e8e3ab0b21937`.
- B, refactored: `7940ec41a1a429dc84172bd20fa24e28443daaed`.

Both SHAs matched the remote at the start. Separate detached worktrees were
created at `outputs/multiview-performance-ab-20261122/original` and `refactored`.
The main working checkout stayed on `refactor/thesis-cleanup`; no branch was
switched, reset or merged. Source comparison preceded measurement.

| Production file / group | Comparison |
|---|---|
| `perception_multiview/fitting.py` | Byte-identical, including all of `SurfaceFitter.fit()` and finalization |
| `perception_multiview/observability.py` | Byte-identical |
| `runtime/worker.py`, `process_worker.py`, `worker_process.py` | All byte-identical |
| `world/model/updater.py`, `provider.py`, `world_model.py` | All byte-identical |
| Isaac config loader and final-scene TOML | Byte-identical |
| Multiview `clouds.py` | Typing/readability changes, redundant projection checks and unused PLY writer removed; projection/voxel mathematics retained |
| Multiview `provider.py` | Diagnostics/timers removed, prepared/result records narrowed; shared fitting cap retained; candidate loop simplified |
| Multiview `segmentation.py` | Unified bounded graph; removed 4.5 mm retry, reassignment, SAT/gap resolution; conservative qualification and narrow sphere/background recovery |
| Multiview `acquisition.py`, `shared_frames.py` | Token normalization/copy preparation simplified; diagnostics/counters removed |
| Multiview `tracking.py` | State, diagnostics and validation simplified; not part of processor compute |
| Isaac `observation_runtime.py` | Removed diagnostic gates/state, cached configuration, simplified tags/lifecycle coordination; worker owns invalidation epoch |

Both providers perform per-view project/crop/voxelization, fusion, coarse
clustering, refinement, fitting, parameter finalization and observability. Both
retain 16 candidate / 8 detection / 24 numerical-fit limits. They use the same
2 mm voxels, stride 2, 12,000-point cap, support plane and workspace configuration.
Actual coarse inputs are byte-identical in A/B for both saved scenarios; static
fitter inputs and final detections are also byte-identical.

Both observation coordinators acquire due inputs before polling results and use
the same wall-clock deadline rule. Worker running/pending/replacement policy,
transport implementation and updater freshness checks are unchanged. The old
coordinator's generation tuple/diagnostic reporting differs from the new revision
tag plus worker epoch. This source review is not a claim that every historical
runtime API or lifecycle path is identical. No second native runtime comparison
was needed to explain the small compute difference.

## Machine and execution environment

All measurements ran SDK-free on the existing L4 host, sequentially and without
Isaac or a heavy background workload. The existing container stayed stopped.

| Item | Recorded value |
|---|---|
| CPU | Intel Xeon CPU @ 2.20 GHz, family 6/model 85, KVM guest |
| Available CPUs | 12 logical CPUs, affinity 0–11; reported 6 cores with 2 threads/core |
| Frequency | `/proc/cpuinfo` about 2200.210 MHz; cpufreq/governor interface not exposed |
| CPU quota | Visible `user.slice/cpu.max` is `max 100000`; no finite inherited quota found |
| Throttling | Exposed `nr_periods`, `nr_throttled`, `throttled_usec` remain zero; some child scopes do not expose these controller fields |
| Memory limits | Session/user memory limits report `max`; Docker memory/swap limits configured as 0 |
| Container CPU limits | NanoCPUs/quota/period 0, cpuset unset; container not used for benchmarking |
| GPU | NVIDIA L4, driver 580.178.04, 0 MiB used during checks |
| Python | Same executable for A/B: CPython 3.11.18 from the existing uv installation |
| NumPy | Same existing `.deps/numpy`, version 1.26.4 |
| Actual loaded BLAS | OpenBLAS 0.3.23.dev, ILP64, SkylakeX, dynamic architecture, MAX_THREADS=64 |
| Active BLAS threads | 12, queried from the loaded library in every benchmark process |
| Thread environment | OPENBLAS/OMP/MKL/BLIS/NUMEXPR thread overrides all unset in both versions |

Affinity, interpreter, NumPy library, active BLAS settings and environment match
across all 16 processes. No CPU, BLAS or system settings were changed. Hardware,
load, cgroup counters, library paths/configuration and before/after environment
snapshots are retained with the results. Initial load average was .08/.16/.34.
The guest's physical host scheduling/frequency is not fully observable.

The previous native session used SDK Python 3.11.13 with Kit/rendering active;
this A/B uses host Python 3.11.18 with Kit absent. The versions within this A/B
are controlled; the native-session versus replay comparison is not an exact
execution-environment control.

## Native input and measurement method

Original inputs remain in `outputs/multiview-native-20261122/session-2/`:

| Batch | Original files | Acquisition time, both cameras | Native token, both cameras |
|---|---|---:|---|
| Final four objects | `static-0.npz`, `static-1.npz` | 31479.050366052 | (12.766666666666667, 5.966666977852583) |
| Gripper-occluded | `pregrasp-0.npz`, `pregrasp-1.npz` | 31507.69450422 | (48.0, 23.583334563300014) |

Depth, intrinsics, transforms, tokens and timestamps are loaded unchanged into
immutable arrays. Camera identities are recovered by exact calibration matching
against the original native result JSON: `/World/PerceptionCamera` and
`/World/PerceptionCameraSide`. No analytic renderer, image masking, resampling or
new camera capture is used. Historical acquisition times are only metadata;
`time.perf_counter()` measures computation. No WorldModel freshness check or
live publication is attempted, and returned timestamps equal the original input.

Only complete native batches were saved. Consequently, there are **no isolated
sphere/box full-frame scenarios**. The two per-fit controls use unchanged native
coarse clouds exported once: large sphere 337 points (`cloud_3`) and rotated
Square 570 points (`cloud_1`). These exact clouds also appear in both versions'
static fitting calls. This isolates numerical fitting without inventing camera
inputs. These controls exclude finalization/observability and do not establish
live grasp eligibility by themselves.

The new `tests/multiview/performance_ab.py` is shared by both detached worktrees,
with separate Python interpreters and runtime checks of every loaded project
module path and expected HEAD. The only interface adaptation is supplying the
original input record's removed acquisition/synchronization diagnostic fields.
No production algorithm is patched; test wrappers time/count original methods.

Each process performs two warm-ups and five measured repetitions. Static and
per-fit order is A–B–B–A; occluded order is B–A–A–B. Each implementation therefore
has 10 measured samples across two clean processes per scenario/control. No two
benchmarks run concurrently. Primary compute timing surrounds the full normal
`MultiViewProcessor.compute()` call; wrapper setup, input loading, hashing and
JSON serialization are outside its timer. Minor wrapper overhead is common to
both versions. All outputs are deterministic within each scenario/version.

## Controlled timing comparison

Milliseconds; p95 uses linear interpolation over the ten measured samples.
Small samples characterize this fixed workload, not a population confidence
interval. Differences below are B relative to A.

| Static final scene metric | A: original main | B: refactored | Difference |
|---|---:|---:|---:|
| Compute p50 | 409.69 | 424.19 | +14.51 ms / **+3.54%** |
| Compute p95 | 411.66 | 427.10 | +15.44 ms / +3.75% |
| Compute range | 407.85–412.20 | 420.74–427.20 | Small, separated ranges |
| Projection/voxelization p50 | 19.13 | 19.18 | +.05 ms |
| Initial clustering p50 | 9.86 | 9.88 | +.02 ms |
| Refinement inclusive p50 | 20.78 | 41.20 | +20.42 ms |
| Metric graph p50 | 20.74 | 40.30 | +19.55 ms |
| Primitive fitting p50 | 337.02 | 329.63 | −7.39 ms |
| Parameter finalization p50 | 2.38 | 2.35 | −.03 ms |
| Geometric observability p50 | 16.69 | 16.77 | +.08 ms |
| Numerical fitting calls/frame | 4 | 4 | 0 |
| Detected / UNKNOWN objects | 4 / 0 | 4 / 0 | Identical results |

Separate static-process medians are A: 409.75, 409.63; B: 421.48, 425.54.
The repeatable added cost is the unified graph's neighbor/block iteration, not
more fitting calls. Its stage time is about twice the old graph, but fitting
still dominates the total. Fitting's small contextual timing variation is not a
new numerical optimization: fitter source and inputs are identical. Refinement
includes graph and sometimes fitting; stage medians are not additive.

| Gripper-occluded metric | A | B | Interpretation |
|---|---:|---:|---|
| Compute p50 / p95 | 709.27 / 713.16 | 431.12 / 432.76 | −39.22% median, **different capability** |
| Compute range | 707.78–714.56 | 428.76–432.84 | Repeatable |
| Fitting p50 | 608.32 | 308.31 | Original does more recovery work |
| Metric graph p50 | 40.60 | 69.97 | Unified graph is slower despite fewer retry passes |
| Numerical fitting calls | 10 | 4 | Old sparse-seed/retry fits removed |
| Detected objects | 4 | 4 | Count alone conceals qualification loss |
| Known / UNKNOWN / graspable | 4 / 0 / 4 | 3 / 1 / 3 | Square regresses |

Original occluded fitting-cloud sizes are
[740,291,37,38,309,55,38,487,337,270], versus [740,487,337,270] after refactoring.
The additional six fits belong to sparse-component recovery attempts, not an
increased burden introduced by cleanup. Original ultimately retains a qualified
whole-box explanation; refactored sparse recovery only retains whole spheres.

| Direct `SurfaceFitter.fit()` control | A p50 / p95 | B p50 / p95 | Median change | A / B ranges |
|---|---:|---:|---:|---|
| Native large sphere, 337 points | 70.69 / 71.99 | 70.30 / 71.72 | −.55% | 70.35–72.50 / 69.31–71.72 |
| Native rotated Square, 570 points | 77.78 / 78.70 | 77.10 / 79.84 | −.87% | 76.58–78.77 / 76.53–80.63 |

Each direct control executes one actual fitter call, including its normal
competing primitive hypotheses. Input hashes and raw fit outputs match exactly.
These results support equal per-fit numerical cost on this machine.

## Correctness alongside timing

Both versions produce **exactly equal** static outputs, including confidence,
geometry, per-view membership and parameter eligibility. Centers are meters;
dimensions are millimeters in the fitted primitive's local-axis order:

| Object | Class | Center, A = B | Dimensions, A = B | Confidence | Graspable |
|---|---|---|---|---:|---|
| Square | cube | (.359878,−.160057,.027501) | (55.273,55.003,55.001) | 1.000000 | yes |
| Rectangle | cube | (.539937,.159903,.024001) | (42.234,68.003,48.002) | 1.000000 | yes |
| SmallSphere | sphere | (.360037,.159931,.022157) | 44.314 diameter | .970753 | yes |
| LargeSphere | sphere | (.539973,−.160087,.030271) | 60.542 diameter | .948450 | yes |

The processor-level graspable flag reported here uses known shape plus finalized
parameter observability, matching provider association. It does not claim that
historical observations are fresh or that old/new live WorldModels publish them.

On the occluded batch, the other three objects remain exactly equal. Square:

| Result | A: original | B: refactored |
|---|---|---|
| Class / confidence | cube / 1.000000 | UNKNOWN / 0 |
| Center (m) | (.360041,−.160406,.027501) | (.359932,−.158105,.029502) |
| Dimensions/bounds (mm) | (55.273,55.773,55.001) | (72.697,69.298,50.997) measured world bounds |
| Center / dimension error (mm) | .408 / .773 | 2.758 / 17.697 |
| Graspable | yes | no |
| Reason | qualified box | `unresolved_components` |

This upgrades the previous native report's occlusion limitation to a **proven
processor qualification regression against original main on real native data**.
It does not prove original main would maintain live freshness during the approach:
that runtime was not run here, and its recovery computation is slower. The faster
refactored result cannot be counted as equivalent-work acceleration.

## Causal assessment and recommendation

1. **Significant computational regression: not detected.** The fixed static workload
   is 3.54% slower, with no additional fits or output changes. The graph explains
   the small measured software cost. This is not a claim of zero slowdown.
2. **Per-fit slowdown: not detected.** Full fitter source is byte-identical and
   native sphere/box controls differ by less than 1% at median.
3. **CPU/environment explanation: plausible, not proven.** Both versions take
   about 410–424 ms on this Xeon guest. The GPU is not executing these NumPy/Python
   fits. Ryzen 7 7800X3D / RTX 4070 SUPER workstation results remain valid historical
   measurements for that setup, but are not a controlled cross-machine or
   cross-version comparison. The same captured inputs must be rerun there to
   assign a hardware contribution. Prior native 491 ms also included a different
   Python patch version and concurrent Kit/rendering; its exact difference from
   this quiet replay cannot be assigned to hardware or software alone.
4. **Runtime overhead dominance: unsupported.** Worker code is unchanged and the
   prior native report measured about 10 ms transport/IO against about 491 ms
   computation. No worker redesign or native historical replay was justified.
5. **Freshness remains unresolved.** Offline historical replay intentionally does
   not test or repair live admission, publication rates or visibility. Nothing here
   overturns the measured current 2.01 Hz/stale-object failures.

Recommendation **2 — Minor measured slowdown: accept and document the computational
tradeoff**, and defer optimization pending a controlled workstation replay. The
percentage bands are guidelines; this small, repeatable 3.54% cost is explicitly
reported rather than called identical performance. This recommendation applies
only to performance. Review the proven occluded-Square qualification regression
separately before claiming demo readiness. No automatic repair, freshness change
or further production refactor is authorized by these results.

## Reproduction and retained evidence

`outputs/multiview-performance-ab-20261122/` retains the source blob comparison,
machine/environment snapshots, frozen input manifest and common native clouds,
all individual timings and detections, pooled summary, and artifact hashes.
Input-manifest SHA-256:
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.
Original native NPZs were rehashed after measurement and remain unchanged.

Prepare once from this checkout, using unused output paths:

```bash
probe_python=/home/ubuntu/.local/share/uv/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps
mkdir -p outputs/NEW-ab
"$probe_python" tests/multiview/performance_ab.py prepare \
  --captures outputs/multiview-native-20261122/session-2 --output outputs/NEW-ab/input.json
```

Create detached worktrees for the two SHAs. In each worktree, invoke the **absolute
path to the same new harness**, with `PYTHONPATH=.:/absolute/existing/.deps` and the
same interpreter:

```bash
python /absolute/current/tests/multiview/performance_ab.py compute \
  --data /absolute/outputs/NEW-ab/input.json --scenario static \
  --expected-head FULL_CHECKOUT_SHA --warmup 2 --repeats 5 \
  --output /absolute/outputs/NEW-ab/UNIQUE-result.json
```

Repeat for `--scenario pregrasp`; use `fit --scenario cloud_3` and
`fit --scenario cloud_1` for the two direct controls. Follow the recorded mirror
order and retain separate result names. Copy original native inputs/result JSON
if moving machines; do not regenerate analytic observations. The prepare output
parent must already exist. The harness checks source paths/HEAD and refuses
result overwrite.

Validation: syntax check, 16 sequential benchmark processes, exact static output
and common-input comparisons, preserved timestamp/input hashes, matching
interpreter/BLAS settings and zero exposed quota throttling. No full suite,
production modifications, native recapture or container start. Stop for review.
