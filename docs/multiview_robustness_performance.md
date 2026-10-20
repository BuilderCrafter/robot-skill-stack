# Multi-view robustness and performance, iteration two

This research continues `research/multiview-rgbd-fusion` at
`42e3258ac5a5c79bba766262357e77f80d158c14`. The new branch is
`research/multiview-robustness-performance`; its PR targets the experimental
parent. The [first report](multiview_rgbd.md), original benchmark scenarios and
previous artifacts remain the comparison baseline. V1 remains the default.

## Checkpoint A: preserve and diagnose

Before algorithm changes, the established wrapper passed 320 discovered unit
tests, eight standalone suites and architecture checks
(`outputs/multiview-iteration2-baseline-unit/run-kw3e885w/`). The unchanged native
source-contract test passed
(`outputs/multiview-iteration2-baseline-native/run-nfp_32pv/`): four objects,
persistent identities, frozen-camera failure, stale repeated frames, protected
uncertain identities, placement-hint reacquisition, partial-initialization
cleanup and two reinitializations. No robot commands were issued.

`scripts/preserve_multiview_baseline.py` exports the seven original package files
from Git into an independent `baseline_multiview` import namespace under ignored
`outputs/multiview-iteration2/baseline_snapshot/`. Its manifest records original
commit/tree and original/exported SHA256 hashes. The only package transformation
is the internal import prefix, so baseline and improved implementations can
consume identical saved frames in one replay without module-cache collisions.
The original experimental TOML and report are copied alongside it.

The generated `benchmark_original.py` retains the original 24 native scenes,
88 expected observations, V1 comparisons and live timing procedure. It retains
the latest fresh frames in memory and exports full camera/fused clouds only
after the timed samples. Thus rejected components, including occluders, are
available for diagnosis; the old accepted-object-only exports omitted them.
The generated script's hash and instrumentation are recorded in the manifest.

The original 8 mm occupied-cell graph connects all 26 adjacent cells without a
sample-distance check. Points in adjoining cells may be separated by nearly
27.7 mm in three dimensions. It can bridge a real 4.5 mm gap and reject the
combined object/occluder as oversized. Merely decreasing a global threshold has
a competing failure: sparse surfaces from one object become disconnected.
Initial saved-cloud probes retained every case, including those false splits.

The frozen fitter was profiled on 114 saved native object clouds (72 isolated
geometry clouds plus 42 accepted automatic clouds). Instrumented CPU replay
took 6.855 s: cylinder hypotheses accounted for 5.037 s cumulative (73.5%) and
box search 1.673 s (24.4%). There were 37,290 percentile calls (2.274 s) and
25,467 least-squares calls (0.674 s). Repeated orientation/residual work dominates
more than least squares alone. The profile is retained at
`outputs/iteration2-fitting/baseline-profile.txt` and `baseline.prof`; its
60.2 ms/cloud is profiled offline replay, not live Isaac throughput.

The previous automatic benchmark remains 84/88 correct, four missing, zero
extra/duplicate/mixed accepted observations, 0.221 mm mean center error,
1.136 mm mean dimension error, 241 ms processing and 3.11 Hz measured loop rate.
Those historical numbers are kept separate from this iteration's fresh runs.

Reproduce the preserved baseline with the existing environment:

```bash
./run_isaac_python.sh scripts/preserve_multiview_baseline.py
./run_isaac_python.sh outputs/multiview-iteration2/baseline_snapshot/benchmark_original.py \
  --result outputs/multiview-iteration2-baseline-detection/result.json \
  --artifacts-dir outputs/multiview-iteration2-baseline-detection --repeats 2
```

Measurements use the university workstation: AMD Ryzen 7 7800X3D, RTX 4070 SUPER
(12,282 MiB), NVIDIA driver 580.167.08, Isaac Sim 5.1.0-rc.19, repository-local
NumPy 1.26.4, py_trees 2.6.0 and the established Isaac Python wrapper. Dependencies,
launcher, baseline scene, V1 and the robotics architecture are unchanged.
