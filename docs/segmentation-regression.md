# Segmentation regression investigation and targeted recovery

The fixed-input comparison found two demo-relevant regressions in `825f1a5`:
sparse observations of an ordinary sphere became UNKNOWN, and a valid sphere
beside a separate oversized foreground component disappeared. Both are repaired.
The four final demonstration objects were retained before and after the repair
at 640×480. Native Isaac verification is **NOT RUN**. Nearby boxes, foreground
contamination at sparse sampling densities, and processing throughput remain
limitations requiring review.

## Comparison and reproducibility

Investigated on `refactor/thesis-cleanup`, starting clean at
`825f1a5173ef19b5f9313702ba90f8a709e82623`, matching origin. Baseline:
`f7255f08d384f94bd991cde896186ec9b5b7dcf4`, in a detached worktree. Main and
origin/main remained `5b61d4442d74e3afd76420ffc11e8e3ab0b21937`.
The baseline is a comparison, not a specification: inaccurate known fits are
failures too. No code from the retired cleanup branch was used.

`tests/multiview/segmentation_ab.py` generates fixed analytic depth buffers, then
evaluates each checkout in a separate Python process. No old/new perception
modules are combined. The 11 original cases were frozen first at 320×240, then
at 640×480. Three closer-view controls were frozen and compared before production
edits. These controls test whether sufficient sampling exposes losses hidden by
failures common to both implementations. They do not change algorithm thresholds.

All variants receive identical bytes: two cameras, stride 2, 2 mm voxelization,
12,000-point cap, workspace `[.2,-.4,0]`–`[.7,.4,.22]`, plane z=0, 24 numerical
fits, and the same renderer seed 71/72. No depth noise was added. Dropout uses
seed 720/721. Exact frozen input, source, scene and component hashes are recorded.
Local captures and raw results are preserved under
`outputs/segmentation-regression-20261122/` (ignored, not packaged into runtime).

| Capture directory | Manifest SHA-256 |
|---|---|
| `captures` (320) | `4af1aff6174fd6a80950e730ab5688381aa25d9a198e829fe58f0e31f5eb1583` |
| `captures-640` | `c6af7e92e3b9769495ca923c3c0f8c09783b556f204ef21785d515b933e6905d` |
| `captures-close` (640) | `711ef2b5bcd1bcc5b8144b82e4f7abb0cbbb552f9af9b2c6e816c67753b8a5e1` |

Each capture contains four distinct synthetic exposure records, timestamps
100.0/100.2/100.4/100.6 and increasing tokens, reusing the unchanged static depth
arrays. These are offline synthetic records, not live or archived Isaac frames.
Each run exercises acquire → compute → associate → WorldModelUpdater publication.
Nonempty cases confirm on observation 3 and retain IDs on observation 4; no
confirmed object changes ID. UNKNOWN observations remain ungraspable. Predictions
are produced before scoring; truth is confined to rendering and evaluation.

Scoring matches predictions one-to-one within 60 mm. A geometry success requires
the right class, center error ≤10 mm and maximum dimension error ≤10 mm.
Graspability is reported separately; a known but ungraspable box is not a pick
success. Box dimensions compare sorted horizontal sides and vertical height;
sphere dimensions are diameters. UNKNOWN errors refer to measured bounds rather
than valid shape parameters. All errors below are millimeters.

## Scenes

The final scene uses the exact authored geometry and centers:

| Object | Center (m) | Geometry |
|---|---|---|
| Square | (.36, −.16, .0275) | 55 mm cube, yaw 25° |
| Rectangle | (.54, .16, .024) | 68×42×48 mm, yaw −33° |
| SmallSphere | (.36, .16, .022) | radius 22 mm |
| LargeSphere | (.54, −.16, .03) | radius 30 mm |

Both final camera transforms are read from the USDA, converting USD camera axes
to ROS axes. Camera positions are approximately (1.65896,−.05281,1.10358) and
(.18,.8,.65). Both use an assumed 24 mm lens / 20.955 mm horizontal aperture;
the side lens is authored, while the inherited main lens could not be recovered
from the binary base scene without Isaac. Analytic surfaces and this main-lens
assumption make this an approximation, even at 640×480.

Other cases use cameras at (1.05,.55,.90), (.25,−.85,.65), looking at (.45,0,.025),
with focal length .95×image width. Closer controls use (.85,.35,.60),
(.2,−.5,.45), otherwise identical processing and scoring.

- B: sphere center (.45,0,.025), radius .025; rotated box center (.45,0,.025),
  size (.068,.042,.050), yaw −33°. Moderate dropout removes 20% of pixels;
  severe box dropout removes 65%. Rendered sphere occlusion adds a cuboid centered
  at (.51,0,.075), size (.025,.040,.040), through normal nearest-surface rendering.
- C: two copies of the B box, centers (.45,0,.025) ± .024×(−sin(yaw),cos(yaw),0).
  Their facing surfaces have a 6 mm gap and their world AABBs overlap. An
  axis-aligned control uses centers (.45,±.024,.025). Both begin as one coarse
  cluster. The closer rotated case also has two dense 4 mm object components.
- D: the same sphere, with a robot-link proxy of size (.045,.240,.050), centered
  at (.45,y,.040). y=.185 gives a 40 mm gap; .150 gives a 5 mm gap; .145 is
  tangent. The latter two form one coarse cluster. The proxy is deliberately
  oversized and elevated; this is foreground contamination, not a Franka model.

## Object results at 640×480

Final-scene outputs are identical in baseline, simplified and repaired versions:

| Object | Class | Center error | Dimension error | Graspable | Confidence |
|---|---|---:|---:|---|---:|
| Square | cube | .14 | .27 | yes | 1.000 |
| Rectangle | cube | .12 | .23 | yes | 1.000 |
| SmallSphere | sphere | <.01 | <.01 | yes | 1.000 |
| LargeSphere | sphere | <.01 | <.01 | yes | 1.000 |

For the following table, `S` means one correct sphere, both errors <.01 mm,
graspable, confidence 1.000. `U(c,d,q)` means one UNKNOWN with center/dimension
errors c/d, confidence q, ungraspable. `0` means no confirmed observation.
Every two-box case has only one observation and loses the second object.
There are no additional confirmed foreground objects in the B/D cases.

| Case | Baseline | Simplified before repair | Repaired |
|---|---|---|---|
| B sphere, 20% missing | S | U(2.97,5.99,0) | S |
| B box, 20% missing | cube (.15,.30), graspable, q=1 | same | same |
| B sphere, rendered occlusion | S | U(2.96,6.02,0) | S |
| B box, 65% missing | cube (.08,.16), **ungraspable**, q=1 | same | same |
| C rotated | merged cube (24.00,26.58), **graspable**, q=1 | U(23.95,69.78,0) | same UNKNOWN |
| C axis control | merged cube (24.00,26.00), **graspable**, q=1 | U(24.08,48.00,0) | same UNKNOWN |
| C rotated, closer | U(24.00,70.50,1) | U(23.97,70.42,0) | same UNKNOWN |
| D alone | S | U(2.96,6.02,0) | S |
| D near | S | U(2.96,6.02,0) | S |
| D coarse-connected | 0 | 0 | 0 |
| D tangent | 0 | 0 | 0 |
| D coarse-connected, closer | S | 0 | S |
| D tangent, closer | S | 0 | S |

The sphere regression is caused by treating sparse metric components as evidence
for multiple objects: only 36.4% of the dropout sphere's points survive the
25-point component threshold; 60.4% survive for the intact sphere. Nevertheless,
one fitted sphere explains the original samples with 95th-percentile residual
below .001 mm. The old algorithm falls back to that whole-object fit. The new
algorithm previously returned UNKNOWN before any numerical fit.

The closer foreground cases have 98.2%/98.0% retained coverage: respectively
530/498 target samples and 3,959/3,936 oversized background samples. The fitted
target establishes hidden support contact; its observed component alone has
no table-contact samples. The previous early oversized-parent rejection lost
this valid partition. In the farther views, coverage is only about 63% at 4 mm
and 67% at 4.5 mm: the old recovery also fails. The repair intentionally does
not invent or reassign missing samples to rescue those cases.

The rotated-box AABB check does reject the closer case's separated fitted boxes.
However, baseline SAT passes those fits and still returns UNKNOWN because the
competing parent box is not disambiguated by camera evidence. The farther case
instead becomes an inaccurate high-confidence merged box in the baseline.
Neither is an end-to-end separation success; these inputs do not justify
restoring SAT or camera-gap resolution. The 24 mm localization errors and
26–70 mm dimension errors are not counted as usable detections.

## Initial 320×240 approximation

All 11 cases have identical classifications, errors, graspability, confidence
and stable IDs in baseline, simplified and final repaired runs:

| Cases | Confirmed output in all versions |
|---|---|
| A final | 4 objects; Square UNKNOWN (1.48,17.77), ungraspable, q=.542; Rectangle cube (.64,1.29), graspable, q=.779; both spheres S |
| B sphere dropout / rendered occlusion | S in each case |
| B box dropout | cube (.86,1.71), ungraspable, q=.685 |
| B severe box | 0 |
| C rotated / axis | one merged cube (24.00,26.34 / 26.61), graspable, q=1; second box missing |
| D alone / near | S in each case |
| D coarse-connected / tangent | 0 in each case |

This lower-resolution comparison alone would have missed both proven regressions.
Its Square failure also shows why four clusters are not sufficient validation.

## Runtime and graph correctness

Median SDK-free compute time over four observations, milliseconds, measured in
sequential isolated processes on this host. These small samples are descriptive,
not a statistically established benchmark; they include fitting/observability,
and exclude rendering, acquisition copying, association and graph profiling.

| 640×480 case | Baseline | Simplified | Repaired |
|---|---:|---:|---:|
| A final | 430.74 | 441.87 | 448.48 |
| B sphere dropout | 156.68 | 39.38 | 158.04 |
| B box dropout | 176.11 | 178.72 | 181.76 |
| B occluded sphere | 231.51 | 160.35 | 233.32 |
| B severe box | 86.66 | 86.82 | 88.38 |
| C rotated | 570.90 | 71.05 | 148.42 |
| C axis | 432.29 | 69.52 | 148.57 |
| D alone | 267.42 | 44.66 | 163.53 |
| D near | 284.88 | 55.33 | 191.64 |
| D coarse-connected | 90.01 | 43.97 | 95.80 |
| D tangent | 90.29 | 44.12 | 93.42 |
| C rotated, closer | 560.87 | 323.47 | 325.27 |
| D coarse-connected, closer | 223.04 | 68.70 | 247.83 |
| D tangent, closer | 221.53 | 70.21 | 243.96 |

The graph probe uses the actual fused coarse clouds from A/C/D, five timed calls,
and one separate tracemalloc call. Canonical component hashes agree exactly
across all versions at both resolutions. With minimum size 1, all input samples
are assigned; focused brute-force reference tests also verify ownership and
connectivity, including dense duplicate cells and negative coordinates.

| 640 input | Points | Baseline graph median ms | Simplified ms | Final ms | Baseline / final traced peak bytes |
|---|---:|---:|---:|---:|---:|
| A final | 2,216 | 14.01 | 28.52 | 28.70 | 466,895 / 393,608 |
| C rotated | 1,011 | 13.65 | 13.55 | 12.36 | 468,802 / 401,466 |
| D coarse-connected | 1,921 | 11.24 | 19.61 | 19.27 | 555,516 / 452,297 |

At 320, baseline/final graph medians are 3.27/5.35 ms (572 points), 1.04/1.79
ms (258), and 1.49/2.73 ms (502). No graph algorithm was changed in this repair.
The unified implementation retains packets of at most 128 pairs of 32-point
blocks; the delta tensor is at most 3 MiB in float64. Other temporaries and
input-sized structures also consume memory; these peaks are not universal RSS
bounds. Dense-cell chunking is still covered by the focused tests.

The four-object graph costs about 14.7 ms more than baseline. Repaired end-to-end
compute is about 4.1% slower in this sample, while avoiding the old fallback
retries makes the intact single sphere faster. Both versions already exceed
the 200 ms interval for the four-object scene (about 2.2–2.3 compute cycles/s).
This is a real 5 Hz throughput risk, not proof of native responsiveness. Graph
optimization alone cannot close that gap; no broad fitting redesign was attempted.

## Targeted changes and validation

Only `segmentation.py` changes production behavior:

1. For 2–8 sparse retained components below 90% coverage, try one cached whole
   fit. Retain it only for a sphere whose 95th-percentile surface residual is
   ≤3 mm. Subsequent support, parameter observability and grasp checks remain.
   Other fragmented explanations remain UNKNOWN.
2. Permit an oversized parent to reach metric refinement. Recover qualified
   components only when a separate oversized component exists and total retained
   coverage is ≥90%. Each target passes the existing fit, support/inferred
   contact, residual and mutual separation checks. The provider discards the
   oversized component. A truly metric-connected oversized cloud remains rejected.

The existing shared fit cap is unchanged: final A/B maximum 4 actual fits per
compute (2 in the closer controls), versus the permitted 24. There is no 4.5 mm
retry, point reassignment, FREE-gap inference, semantic robot masking or new
interface. Provider, fitting, observability, tracking, mapping and motion are
unchanged. Segmentation is **191 → 213 lines (+22)**, compared with baseline 348;
provider remains 197, fitting 441 and observability 241.

Focused final validation:

```bash
python -m unittest tests.unit.test_multiview_segmentation tests.unit.test_multiview_segmentation_readiness
# 16 tests, 4.813 s, OK
python -m unittest tests.unit.test_multiview_provider
# 9 tests, 4.166 s, OK
```

The old whole-background-rejection test now checks both a recoverable separate
component and a truly connected rejected control. A new rendered 640×480 test
checks missing-depth sphere recovery and tangent foreground recovery through
the actual processor, including final parameter observability. Existing missing
box-band, excess-fragment, unknown-patch and insufficient-budget controls pass.
All three frozen A/B datasets were replayed after the final production edit;
manifest, graph, final-source hashes and stable IDs were independently checked.
No full project suite ran.

Native check: **NOT RUN**. NVIDIA L4 is present, but the expected launcher SDK
path `/home/etfrobotics/isaacsim` is absent on the host and the existing `isaac51`
container is stopped. No container, Isaac process, GUI, dependency or launch
configuration was changed. Synthetic results do not validate live camera timing,
real depth noise, robot motion, or native cleanup.

## Rerunning the comparison

Use Python 3.11 and the existing NumPy `.deps` environment. On this host:

```bash
probe_python=/home/ubuntu/.local/share/uv/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps
probe_cases=(A_final B_sphere_20pct B_box_20pct B_sphere_occluded B_box_severe
             C_rotated C_axis_control D_alone D_near D_coarse_connected D_touching)
# From this checkout; use unused output directories (capture refuses overwrite).
for width in 320 640; do
  "$probe_python" tests/multiview/segmentation_ab.py capture \
    --data "outputs/segmentation-replay-$width" --width "$width" --cases "${probe_cases[@]}"
done
"$probe_python" tests/multiview/segmentation_ab.py capture \
  --data outputs/segmentation-replay-close --width 640 \
  --cases C_rotated_close D_coarse_connected_close D_touching_close
git worktree add --detach /tmp/segmentation-baseline f7255f08d384f94bd991cde896186ec9b5b7dcf4
```

For each capture directory, invoke the **absolute path to the new harness** with
`evaluate --data ABSOLUTE_CAPTURE_DIRECTORY --output ABSOLUTE_RESULT_FILE`.
Run sequentially from the baseline worktree, an `825f1a5` worktree, and the final
checkout, with `PYTHONPATH=.:/home/ubuntu/projects/robot-skill-stack/.deps` in each
process. The harness asserts that segmentation imports from the current worktree.
Keep each result file separate. Existing local results are `before-baseline*`,
`before-current*`, and `final-current*`; `after-current-640.json` is an intermediate
repair replay, not the final reported run. Final segmentation source SHA-256:
`e12f6153c951897b3f679253296c27ebc3c3308abf30c7d356a7b334f520da1d`.

## Review boundary

The observed regressions are repaired with limited segmentation changes. Do not
interpret this as general manipulation readiness: close boxes are not reliably
separated, sparse foreground contamination can still erase targets, semantic
perception still lacks robot masking, and even a wrong merged box can be marked
graspable at low resolution. Native final-scene validation and throughput remain
open. No fitting/observability redesign or further refactoring is authorized by
these results. Stop for review.
