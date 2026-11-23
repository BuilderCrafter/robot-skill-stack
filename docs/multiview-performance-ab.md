# Multiview fixed-input geometry and performance experiments

These studies distinguish computational cost from lost perception capability.
A low-latency UNKNOWN result is not an optimization over an accurate, graspable
object. The current [pipeline](multiview_rgbd.md) retains the recovered sparse
sphere and occluded Square; nearby-box separation and live availability remain
limited. [Native runtime measurements](multiview-native-validation.md) are a
separate experiment from these SDK-free replays.

## Inputs, scoring and experimental controls

Native inputs are unchanged `outputs/multiview-native-20261122/session-2/` buffers:

| Batch | Files | Original acquisition time | Original token, both cameras |
|---|---|---:|---|
| Static | `static-{0,1}.npz` | 31479.050366052 | (12.766666666666667, 5.966666977852583) |
| Pre-grasp | `pregrasp-{0,1}.npz` | 31507.69450422 | (48.0, 23.583334563300014) |

Full 640×480 depth, intrinsics, transforms, tokens and timestamps are retained.
Identities `/World/PerceptionCamera` and `/World/PerceptionCameraSide` are recovered
by exact calibration matching against the native result JSON. No image masking,
resampling or analytic replacement enters native replay. Manifest
`outputs/multiview-performance-ab-20261122/input.json` SHA-256:
`e84b53762529261b8bf0efe6257de51a80efc7a5ed919a88bdfaf55f3fe66e1e`.

Analytic segmentation controls instead use fixed seed 71/72 depth rendering,
seed 720/721 dropout, stride 2, 2 mm voxels, 12,000 points, 24 shared fits,
workspace [.2,−.4,0]–[.7,.4,.22] and support z=0. Each static scene has four
synthetic exposure records at 100.0/.2/.4/.6 with increasing tokens; these are
not native frames. Acquisition, processing, association and updater publication
are exercised; nonempty cases confirm on record 3 and retain IDs on record 4.
GT is confined to rendering and one-to-one scoring within 60 mm. A usable
geometry needs the correct class and center/maximum dimension errors ≤10 mm;
graspability is scored separately. Box horizontal dimensions are sorted for
scoring, sphere dimensions are diameters. UNKNOWN errors describe measured bounds.

Frozen analytic manifests under `outputs/segmentation-regression-20261122/`:

| Directory | SHA-256 |
|---|---|
| `captures` (320×240) | `4af1aff6174fd6a80950e730ab5688381aa25d9a198e829fe58f0e31f5eb1583` |
| `captures-640` | `c6af7e92e3b9769495ca923c3c0f8c09783b556f204ef21785d515b933e6905d` |
| `captures-close` (640×480) | `711ef2b5bcd1bcc5b8144b82e4f7abb0cbbb552f9af9b2c6e816c67753b8a5e1` |

## Sparse surfaces, foreground and nearby boxes

Comparison sources: baseline `f7255f08d384f94bd991cde896186ec9b5b7dcf4`, simplified
`825f1a5173ef19b5f9313702ba90f8a709e82623`, and repaired segmentation with SHA-256
`e12f6153c951897b3f679253296c27ebc3c3308abf30c7d356a7b334f520da1d`.
This study preceded native occluded-box recovery. All versions were evaluated in
separate processes against identical frozen bytes; the baseline is not a correctness
oracle when it confidently merges objects.

The final-scene approximation uses the [authored objects](multiview_perception_final.md)
and USDA camera transforms, with assumed 24 mm lenses / 20.955 mm horizontal
aperture. The inherited main lens was unavailable without Isaac. Native calibration
later measured different focal lengths, so even the 640×480 analytic scene is an
approximation. Other views look at (.45,0,.025) from (1.05,.55,.90) and
(.25,−.85,.65), focal length .95×width; closer controls use (.85,.35,.60) and
(.2,−.5,.45), without changing processing thresholds.

B controls use a 25 mm-radius sphere or 68×42×50 mm box at (.45,0,.025), box yaw
−33°, with 20%/65% missing pixels. Sphere occlusion is rendered by a 25×40×40 mm
cuboid at (.51,0,.075). C uses two such boxes offset ±.024 m perpendicular to
their major axis: facing surfaces have a 6 mm gap while world AABBs overlap.
The axis control uses (.45,±.024,.025). D adds an elevated oversized proxy,
45×240×50 mm at (.45,y,.040): y=.185/.150/.145 gives 40 mm gap / 5 mm gap /
tangency. It is a foreground control, not a Franka model.

At 640×480 all versions retain the four analytic final objects: Square errors
.14/.27 mm, Rectangle .12/.23 mm, both spheres <.01/<.01 mm; all graspable,
confidence 1. `S` below is one such accurate, graspable sphere; `U(c,d,q)` is
one ungraspable UNKNOWN with center/dimension errors in mm and confidence q.
`0` means no observation. Every two-box case loses the second object; B/D produce
no extra confirmed foreground object.

| Control | Baseline | Simplified | Repaired |
|---|---|---|---|
| Sphere, 20% missing | S | U(2.97,5.99,0) | S |
| Box, 20% missing | cube (.15,.30), graspable, q=1 | same | same |
| Sphere, rendered occlusion | S | U(2.96,6.02,0) | S |
| Box, 65% missing | cube (.08,.16), ungraspable, q=1 | same | same |
| Rotated box pair | merged cube (24.00,26.58), graspable, q=1 | U(23.95,69.78,0) | same UNKNOWN |
| Axis-aligned pair | merged cube (24.00,26.00), graspable, q=1 | U(24.08,48.00,0) | same UNKNOWN |
| Rotated pair, closer | U(24.00,70.50,1) | U(23.97,70.42,0) | same UNKNOWN |
| Sphere alone / near proxy | S | U(2.96,6.02,0) | S |
| Sphere coarse-connected / tangent, farther views | 0 | 0 | 0 |
| Sphere coarse-connected / tangent, closer views | S | 0 | S |

Only 36.4% of the dropout sphere and 60.4% of the intact sphere survive the
25-point metric-component cutoff, although a whole-sphere fit explains the samples
with p95 residual <.001 mm. Rejecting sparse components before trying that fit
caused the loss. Closer foreground cases retain 98.2%/98.0% coverage: target
530/498 points, oversized background 3,959/3,936 points. Recovering separate
qualified targets fixes those cases. Farther views retain only about 63% at
4 mm or 67% at 4.5 mm; neither version recovers them. No samples are invented
or reassigned. SAT separates the closer fitted boxes, but a competing parent
box remains unresolved; this does not establish end-to-end box separation.

The initial 320×240 comparison would have missed both regressions: all 11 cases
match in every version. Final Square is UNKNOWN (1.48/17.77 mm, q=.542), Rectangle
cube (.64/1.29 mm, q=.779), spheres S. The moderate-dropout box is ungraspable
(.86/1.71 mm, q=.685), severe-dropout box absent. Nearby pairs become graspable
merged cubes (24.00/26.34 or 26.61 mm), while coarse-connected/tangent foreground
loses the sphere. Four coarse clusters alone are insufficient validation.

Historical compute medians over four synthetic records, ms:

| 640×480 control | Baseline | Simplified | Repaired |
|---|---:|---:|---:|
| Final scene | 430.74 | 441.87 | 448.48 |
| Sphere / box 20% missing | 156.68 / 176.11 | 39.38 / 178.72 | 158.04 / 181.76 |
| Occluded sphere / severe box | 231.51 / 86.66 | 160.35 / 86.82 | 233.32 / 88.38 |
| Rotated / axis pair | 570.90 / 432.29 | 71.05 / 69.52 | 148.42 / 148.57 |
| Sphere alone / near proxy | 267.42 / 284.88 | 44.66 / 55.33 | 163.53 / 191.64 |
| Coarse-connected / tangent, farther | 90.01 / 90.29 | 43.97 / 44.12 | 95.80 / 93.42 |
| Rotated pair, closer | 560.87 | 323.47 | 325.27 |
| Coarse-connected / tangent, closer | 223.04 / 221.53 | 68.70 / 70.21 | 247.83 / 243.96 |

The unified graph preserves exact canonical component hashes and sample ownership.
Five-call graph medians / separately traced peaks:

| 640 input | Points | Baseline / simplified / repaired ms | Baseline / repaired peak bytes |
|---|---:|---|---|
| Final | 2,216 | 14.01 / 28.52 / 28.70 | 466,895 / 393,608 |
| Rotated pair | 1,011 | 13.65 / 13.55 / 12.36 | 468,802 / 401,466 |
| Coarse-connected | 1,921 | 11.24 / 19.61 / 19.27 | 555,516 / 452,297 |

At 320, baseline/repaired graph medians are 3.27/5.35 ms (572 points), 1.04/1.79
(258), 1.49/2.73 (502). The 128-pair ×32×32×3 float64 delta tensor is at most
3 MiB, not a bound on total RSS. End-to-end repaired final compute was 4.1%
slower in this sample, already about 2.2–2.3 cycles/s before native rendering.

## Controlled native compute comparison

A is original main `5b61d4442d74e3afd76420ffc11e8e3ab0b21937`; B is
`7940ec41a1a429dc84172bd20fa24e28443daaed`. Fitting, observability, worker transport
and updater source are byte-identical in this comparison. Both use identical
static coarse/fitter inputs and 16-candidate / eight-detection / 24-fit limits.
The graph and sparse recovery differ. This is not a comparison of current fitter
source, which was subsequently simplified without changing native outputs.

SDK-free runs were sequential, with Isaac stopped: Xeon 2.20 GHz family 6/model
85 KVM guest, 12 logical CPUs/affinity 0–11, CPython 3.11.18, NumPy 1.26.4,
OpenBLAS 0.3.23.dev ILP64/SkylakeX with 12 active threads and no thread overrides.
Visible cgroups have no finite CPU/memory quota and exposed throttling counters
stay zero; physical-host scheduling/frequency is not fully observable. The L4
GPU does not execute these NumPy/Python fits. No CPU/BLAS settings were changed.

Each clean process uses two warm-ups and five samples. Static/direct-fit order
is A–B–B–A; pre-grasp B–A–A–B: ten samples/version/case across 16 processes.
The shared harness checks module paths/HEAD and times the normal processor call;
loading, hashing and JSON writes are outside the timer. Outputs are deterministic.
p95 is interpolated over ten samples; stage medians overlap and are not additive.

| Native static metric | A | B |
|---|---:|---:|
| Compute p50 / p95, ms | 409.69 / 411.66 | 424.19 / 427.10 |
| Compute range, ms | 407.85–412.20 | 420.74–427.20 |
| Projection/voxelization p50 | 19.13 | 19.18 |
| Initial clustering p50 | 9.86 | 9.88 |
| Refinement inclusive / metric graph p50 | 20.78 / 20.74 | 41.20 / 40.30 |
| Fitting p50 | 337.02 | 329.63 |
| Finalization / observability p50 | 2.38 / 16.69 | 2.35 / 16.77 |
| Fits / known / graspable objects | 4 / 4 / 4 | 4 / 4 / 4 |

Static compute is repeatably **3.54% slower**, primarily the graph; separate
process medians A=409.75/409.63, B=421.48/425.54 ms. Identical fitter code/input
rules out a changed per-fit algorithm. Direct controls confirm this:

| Native cloud, one full competing-shape fit | A p50/p95 ms (range) | B p50/p95 ms (range) |
|---|---|---|
| Large sphere, 337 points | 70.69/71.99 (70.35–72.50) | 70.30/71.72 (69.31–71.72) |
| Square, 570 points | 77.78/78.70 (76.58–78.77) | 77.10/79.84 (76.53–80.63) |

Native static geometry is exact in both versions: Square center
(.359878,−.160057,.027501), dimensions (55.273,55.003,55.001) mm; Rectangle
(.539937,.159903,.024001), (42.234,68.003,48.002) mm; SmallSphere
(.360037,.159931,.022157), diameter 44.314 mm; LargeSphere
(.539973,−.160087,.030271), diameter 60.542 mm. Confidence is respectively
1.000000/1.000000/.970753/.948450; both cameras contribute and all are graspable.

| Native pre-grasp metric | A | B, before box recovery |
|---|---:|---:|
| Compute p50 / p95, ms | 709.27 / 713.16 | 431.12 / 432.76 |
| Compute range, ms | 707.78–714.56 | 428.76–432.84 |
| Fitting / metric graph p50, ms | 608.32 / 40.60 | 308.31 / 69.97 |
| Fits / known / UNKNOWN / graspable | 10 / 4 / 0 / 4 | 4 / 3 / 1 / 3 |

The 39.22% lower median is lost capability. A fits cloud sizes
[740,291,37,38,309,55,38,487,337,270]; B fits [740,487,337,270]. The six extra
original fits are recovery attempts. The other three objects match exactly,
but B's Square is UNKNOWN (q=0), center error 2.758 mm, measured-bound error
17.697 mm, reason `unresolved_components`. A's graspable cube has .408/.773 mm
errors. This proves a qualification regression on real native input, rather
than a target-count or hardware explanation.

Both versions require about 410–424 ms for the static workload on this guest.
The modest graph slowdown does not explain the full gap to historical Ryzen 7
7800X3D / RTX 4070 SUPER workstation measurements. Native SDK Python 3.11.13 also
runs with Kit/rendering; this host replay uses 3.11.18 without Kit. Cross-machine
or native-versus-replay differences cannot be assigned causally without matched
inputs/environments. Worker overhead dominance is unsupported: native transport
was about 10 ms against 491 ms compute. Live freshness remains a separate failure.

## Native occluded-box qualification

The saved 487-point pre-grasp Square has 4 mm retained components [291,37,38],
only 75.15% coverage. A whole-box fit alone is insufficient: nearby boxes also
fit an inaccurate parent confidently. Their measured XY connectivity distinguishes
these fixed cases without GT dimensions:

| Input | Retained 3D coverage | Largest XY component / all points | Result |
|---|---:|---:|---|
| Native Square | 75.15% | 477/487 = 97.95% | Cached cube recovered |
| Nearby rotated boxes | 71.22% | 531/1011 = 52.52% | UNKNOWN |
| Nearby axis-aligned boxes | 67.44% | 584/1041 = 56.10% | UNKNOWN |
| Missing-band box | 100% | 1156/2312 = 50% | UNKNOWN |

The 95% footprint condition follows the existing consistency fraction, using
unchanged 4 mm connectivity. No source sample is removed, reassigned or replaced.
The recovered cached fit still passes measured-envelope and independent camera
boundary qualification. Its center is (.360040528,−.160406280,.027500704) m,
local dimensions (55.272921,55.773421,55.001408) mm, yaw 25°, q=.999999864,
errors **.408/.773 mm**, graspable: exactly original main's detection.

The four horizontal bounds use calibrated occupied/FREE transitions; top uses
measured plane support and bottom the configured support plane plus low-side
samples. This empirical model is not a universal proof of hidden solid volume.
The unit fixture `tests/fixtures/multiview_occluded_box.npz` preserves the complete
487-point native cloud and 1,011-point nearby-box negative; its JSON sidecar binds
source/array hashes. Reference center was used only to select the complete cloud
for the fixture, not as a production point mask or classifier input.

A two-warm-up/five-sample recovery comparison measured pre-grasp p50/p95
431.72/448.63 →452.88/454.43 ms (+4.90% median), still four fits. Static
423.35→427.16 ms (+.90%) preserves exact outputs. Six selected frozen controls
retain sparse-sphere/foreground recovery while nearby boxes remain UNKNOWN.
The historical three-exposure static/pre-grasp/home WorldModel replay retains
four IDs, publication counts [0,0,4], and four qualified objects. Its clock is
original acquisition plus measured compute; it does not establish live freshness.

## Fitting and pixel-processing controls

Twelve frozen clouds cover four native static objects, the occluded Square,
off-grid/wrap boxes, noisy sphere/upright cylinder, two lying axes and a circular
cap. Comparisons use fitting source `41034d2`→`2006046`; observability comparisons
use `2006046`→`a8689ec`. No angular steps or qualification tolerances were tuned.

| Alternative | Measured loss | Retained decision |
|---|---|---|
| Replace convex hull by rectangular area | Circular cap plus 16 wall samples changes UNKNOWN (q=.0259) to cube (q=1), accepted by fitter finalization | Keep hull coverage; low residual is insufficient |
| Drop .05° box refinement | Off-grid/wrap yaw loses .25°; center/size shifts up to .136/.244 mm; q drops up to .00778 | Keep precision beyond authored native angles |
| Drop .25° lying-axis refinement | Axes shift .25°/.50°, world size .282/.737 mm; q drops .00556/.00966 | Keep cylinder refinement |
| Use one/two radial passes | Clean native changes are small; partial-arc outliers expose a real accuracy loss below | Keep four robust passes |

Radial control: 64 points on a 120° arc of a 30 mm circle, Gaussian noise .1 mm
(seed 113), four points shifted 3 mm in X. Center/radius errors are
**1.111/1.014 mm** with one pass, **.553/.517 mm** with two,
**.243/.236 mm** with four. Committed fitting tests preserve this accuracy and
circular-cap rejection. Upright/lying cylinders retain curvature and cap evidence;
a short arc or missing endpoint remains unqualified.

Native fitted geometry, confidence, camera evidence and graspability match after
fitting simplification. Across all 12 clouds, classes/qualification match; noisy
upright-cylinder world size differs only 6.94e−17 m and class scores 3.75e−16.
Box observability comparisons also retain 16 control outcomes and exact supported
bounds: foreground, NaN/zero/negative depth, absent frames/top/support, sparse or
out-of-frame evidence and physically covered/oblique partial boxes remain
unresolved; a small hidden extension remains inside the conservative geometry.
Three/five-camera geometry and graspability match exactly, including reordered
and valid-empty-camera controls.

Thirty-two native edge ribbons, two warm-ups/five measured passes, preserve exact
pixel coordinates. Median totals: original scalar deduplication **7.16 ms**,
`np.unique(axis=0)` **48.84 ms**, simplified bounded scalar encoding **4.47 ms**.
The material axis-based cost justifies retaining scalar encoding. Row-major
order permits direct neighbor-ID search; arbitrary integer-overflow behavior
is outside the bounded image-coordinate contract.

Later bounded full-processor comparisons (five samples/version, same environment):

| Change | Static p50/p95 before → after, ms | Pre-grasp p50/p95 before → after, ms |
|---|---|---|
| Fitting simplification | 422.67/422.81 →406.90/408.76 | 450.49/461.04 →426.37/429.76 |
| Observability simplification | 402.70/406.55 →402.33/403.88 | 426.87/430.22 →427.38/428.70 |

All native geometry remains exact and fits remain four. Fitting-stage medians
329.74→312.52 ms static, 306.73→285.26 pre-grasp; observability medians
16.91→15.21 and 15.46→14.02 ms. These descriptive samples do not establish
sustained live throughput or overturn the native stale-visibility measurements.

## Reproduction and evidence

Use the existing Python 3.11/NumPy 1.26.4 environment. On the VM:
`/home/ubuntu/.local/share/uv/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11`,
with `PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps` in each checkout.
No benchmark processes run concurrently. Output paths must be unused.

Prepare native inputs once:

```bash
mkdir -p outputs/NEW-ab
python tests/multiview/performance_ab.py prepare \
  --captures outputs/multiview-native-20261122/session-2 --output outputs/NEW-ab/input.json
```

Create detached checkouts for comparison sources. From each, invoke the absolute
path to the same current harness with identical interpreter/inputs:

```bash
python /absolute/current/tests/multiview/performance_ab.py compute \
  --data /absolute/outputs/NEW-ab/input.json --scenario static \
  --expected-head FULL_CHECKOUT_SHA --warmup 2 --repeats 5 \
  --output /absolute/outputs/NEW-ab/UNIQUE-result.json
```

Repeat for `pregrasp`; `fit --scenario cloud_3` / `cloud_1` are the unchanged
337-point sphere / 570-point Square controls, excluding finalization. Follow
the recorded mirror order for the original controlled comparison. Copy original
NPZ/result JSON if moving machines; do not regenerate analytic substitutes.
The harness checks source paths/HEAD, input/configuration hashes and overwrite.

Analytic controls use the same absolute-path/separate-checkout method:

```bash
python tests/multiview/segmentation_ab.py capture --data outputs/NEW-synthetic \
  --width 640 --cases A_final B_sphere_20pct B_box_20pct B_sphere_occluded \
  B_box_severe C_rotated C_axis_control D_alone D_near D_coarse_connected D_touching
python tests/multiview/segmentation_ab.py capture --data outputs/NEW-close \
  --width 640 --cases C_rotated_close D_coarse_connected_close D_touching_close
python /absolute/current/tests/multiview/segmentation_ab.py evaluate \
  --data /absolute/outputs/NEW-synthetic --output /absolute/outputs/UNIQUE.json
```

Use `--width 320` in a distinct directory for the low-resolution control.
Frozen original arrays and raw results remain in the local roots below; source,
input, component and artifact hashes are recorded rather than inferred from timing.

| Local evidence root under `outputs/` | Contents |
|---|---|
| `multiview-performance-ab-20261122/` | Source comparison, machine/cgroup/BLAS snapshots, native manifest/clouds, individual and pooled timing/results |
| `segmentation-regression-20261122/` | Three analytic datasets, isolated A/B outputs and graph hashes/peaks |
| `occluded-box-recovery-20261123/` | Native/negative qualification, six frozen controls, recovery timing and historical-clock replay |
| `fitting-audit-20261123/` | Frozen 12 clouds, variant/radial measurements, source/input hashes and native comparisons |
| `observability-audit-20261123/` | Frozen pixel arrays, 16 controls, direct native bounds, N-camera comparisons and timing |

The 12-cloud archive SHA-256 is
`bd18e2c7723b63253861ee40543613e35aff767eccb0218e5debf4496113c803`.
Committed native-derived fixtures and ordinary mathematical regression tests
preserve key controls without depending on these local research directories.
Use the [validation guide](validation.md) for supported execution commands.
