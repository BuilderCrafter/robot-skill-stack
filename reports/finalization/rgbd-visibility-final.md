# Final visibility-aware RGB-D geometry experiment

14 November 2026, UTC. Branch `feature/global-rgbd-planning-v5`, clean starting
HEAD **`b16ec78589d1747d11d59cf75d48cc128c381880`**. `git fetch origin` and an
independent `git ls-remote` agreed with local HEAD. Read [AGENTS.md](../../AGENTS.md),
the [design](rgbd-reconstruction-design.md), [box feasibility gate](rgbd-box-feasibility.md)
and [fixed-grid investigation](rgbd-coverage-final-attempt.md).

**GO for this bounded offline geometry checkpoint:** both archived obstacles
qualify, both reconstructed boxes conservatively contain their copied physical
reference solids, and all ten negative controls withhold geometry. This is a
meaningful improvement over the previous 1/2 reference qualification result.

**RGB-D global planning remains NO-GO for the final demonstration.** Floor/base
rejection, original-source freshness and broader scene coverage remain
unresolved. No runtime integration, native planning, Isaac/container operation,
second geometry rule or GUI modification was performed. The experiment ends
with publication; any further development requires explicit review.

## 1. Visibility criterion and geometric justification

The previous rule required every plane's fixed 3×3 probe grid to span all three
rows and columns. Static's +Y face supplied six valid depth samples, but its
three upper probes lay outside the side image. Treating those absent locations
as a required observed band rejected a physically accurate hypothesis.

One new rule was frozen: **visibility-aware observed-plane patch coverage**.
The fitting algorithm, numerical budgets, robot exclusion, directly measured
plane checks, calibrated edge witnesses and all six extent checks remain
unchanged. The original prototype is preserved as the reference.

For each directly measured plane, project the same fixed grid at local tangent
fractions `(-0.25, 0, 0.25)` using the original intrinsics and camera transform.
Each probe is classified separately:

| State | Meaning | Positive surface evidence? |
| --- | --- | --- |
| Supported | Valid depth within the existing 2 mm plane-depth band | Yes |
| Contradictory FREE | Valid depth farther than the predicted plane plus the band | No; rejects the model |
| Occluded | A nearer surface precedes the candidate plane | No |
| Outside FOV / behind camera | The probe cannot be observed by this camera | No |
| Invalid depth / back-facing | No usable current plane measurement | No |

A passing observed patch needs **six unique supported image pixels** spanning
at least **two grid bands in each tangent direction**, with support in every
currently observable row/column. No outside-FOV, occluded or invalid sample is
counted toward that minimum. A line, a tiny fragment, a wholly unobserved face,
or partial visibility alone is insufficient. Any visible plane-probe
contradiction rejects the model, even if another view has sufficient support.

All three orthogonal normal-axis families X/Y/Z need such a patch on an
already supported direct plane. The reference's existing direct-plane checks
still require six interior points, rank-three plane estimation, at least 25%
span in both tangents and bounded residual/slope. The six-pixel minimum reuses
that evidence minimum; the two-direction requirement establishes a 2D observed
patch without requiring an out-of-frame grid band. The residual score must
still reach 0.65 with the unchanged 1.5 mm residual tolerance.

**All six independent face bounds remain mandatory.** Direct planes constrain
observed face locations; calibrated occupied/free transitions constrain the
remaining extents. A camera image edge never supplies a physical box boundary.
When a lower/rear witness is missing, its bound remains null and the model is
unqualified. This separates local plane evidence from evidence for full object
extent rather than assuming a partially visible patch establishes thickness.

Under the explicit **single static convex yaw-box assumption**, these planes
and six bounds can define a closed inferred box. Its uncertainty envelope uses
the existing depth bands and oblique-ray projection allowance. This is empirical
model qualification, not a proof of arbitrary hidden shape or calibrated safety
probability. Box interiors remain labeled inferred; occupancy is not completed.

## 2. Blind experiment and scoring separation

Only the same two external archives were used:

```text
rgbd-base-clearance-20261113/static/frozen_rgbd/
rgbd-base-clearance-20261113/v43/static/
```

The blind reader verifies sensor/map file hashes, requires two 640×480 frames
and `ground_truth=None`, and uses only depth, calibration, configured workspace/
support and synchronized robot snapshots. It reads no fixture identities,
dimensions or reference geometry. Point limits remain 12,000 dense points and
1,200 fit points, at most four hypotheses and 16 components. There is one fit
per scene and **zero additional fits, acquisitions, workers or mapping rays**.

The new method consumes each freshly recomputed reference result. Both reference
results must exactly reproduce the earlier sealed predictions before the new
coverage rule is evaluated. The original timestamps, exposure tokens, K/T and
generation-related archived metadata are retained; no timestamp is refreshed.

The declaration and all nine then-existing Python source hashes were frozen
before execution. Predictions and all control outcomes were sealed at
**12:39:18.223704 UTC**. Independent GT scoring began at
**12:39:32.633795 UTC**, after seal verification. GT was used only for subsequent
association, containment and volume scoring. The analyst had seen earlier
reports; this is algorithm/input separation, not a double-blind study. No rule
or numeric threshold changed after scoring. A later preservation-test module
and documentation do not change the frozen algorithm.

The runner stops on the first positive/control failure; none occurred. The
independent scoring also passed the declared containment, strict FREE-voxel
and 64-proxy checks. No second alternative was implemented.

## 3. Positive-scene results

| Measurement | Static | v43 |
| --- | --- | --- |
| Sensor-only nonrobot component | 4,929 points; correct obstacle confirmed afterward | 941 points; correct obstacle confirmed afterward |
| Reference hull-based score / decision | 0.030929 / rejected | 0.902215 / qualified |
| Previous fixed-grid result | Rejected | Unrun in that historical attempt |
| New visibility-gated residual score / decision | 0.999999793 / **qualified** | 0.999999893 / **qualified** |
| Accepted center XYZ, m | (0.450000045, -0.000413345, 0.349335386) | (0.550000393, -0.225313668, 0.299527505) |
| Accepted dimensions XYZ, m | (0.094001105, 0.124827496, 0.265330078) | (0.064000592, 0.064629598, 0.124946930) |
| Yaw / quaternion | Approximately 0 / identity | Approximately 0 / identity |

Both numerical yaw values are `-1.9377047211159066e-18` rad; quaternions are
approximately `(1,0,0,0)` in wxyz. The score is residual quality gated by measured
coverage, not a probability and not directly equivalent to the old hull score.
Both v43's geometry and all reference fits/bounds remain numerically unchanged.
Static now receives a model from its previously computed, independently checked
bounds; no fitter or observed dimensions were changed to achieve acceptance.

| Current positive patch | Static | v43 |
| --- | --- | --- |
| +X, primary camera | 9 supported, 3×3 bands | 9 supported, 3×3 bands |
| +Y | Side: **6 supported, 3×2 bands, 3 outside FOV** | Primary: 9 supported, 3×3 bands; side also has 6 supported |
| +Z, primary camera | 9 supported, 3×3 bands | 9 supported, 3×3 bands |

Static's excluded +Y pixels are still (322,-14), (301,-16), (282,-18). They
remain explicitly outside-FOV, supplying **zero** positive evidence. The six
in-frame occupied pixels, the existing 1,559-point +Y plane support and the
independent extent witnesses supply the accepted evidence. No probes were moved.

## 4. Six-face evidence and independent containment

| Face | Static | v43 |
| --- | --- | --- |
| -X | 500 direct interior plane points | 64 direct interior plane points |
| +X | 511 direct interior plane points | 115 direct interior plane points |
| -Y rear | Three-bin +X/primary-camera boundary; selected trial has 68 transitions | Same type; 34 transitions |
| +Y | 1,559 direct interior plane points | 258 direct interior plane points |
| -Z lower | Three-bin +Y/side-camera boundary; selected trial has 62 transitions | Same type; 32 transitions |
| +Z | 105 direct interior plane points | 37 direct interior plane points |

Rear/bottom planes have no directly observed interior samples. Their bounds
remain conditional on the single-box model and independently measured edge
transitions. All per-face plane estimates, status flags, calibrated witness
pixels and original source metadata are preserved in the published blind JSON.

Independent scoring compares every copied physical reference vertex with the
model's six local faces. Both references contain all eight corners of their
axis-aligned physical box. A convex model containing those vertices contains
the copied convex solid for that frozen reference; it does not establish
containment for unseen objects outside the declared model family.

| Independent score | Static | v43 |
| --- | ---: | ---: |
| Minimum reference-to-model boundary margin | **+2.000404 mm** | **+1.999904 mm** |
| Contains copied physical solid within existing 10 µm allowance | **Yes; also without allowance** | **Yes; also without allowance** |
| Maximum absolute face-extent error | 3.329658 mm, outward excess | 2.945962 mm, outward excess |
| Model / reference volume, m³ | 0.003113363 / 0.002808000 | 0.000516822 / 0.000432000 |
| Excess volume | **10.8747%** | **19.6347%** |

These margins describe separation of the reference solid from its enclosing
model boundary, **not robot trajectory clearance**. No route, spline, motion
contact result or continuous swept-volume safety was evaluated. The unexpanded
surface envelopes still undershoot by 0.030942/0.067389 mm and fail the unchanged
10 µm containment allowance; accurate visible extrema alone remain insufficient.

## 5. FREE evidence, robot contamination and original occupancy

The unchanged current-pixel free-ray checks find **0/76** contradictions for
static and **0/78** for v43. The new positive plane grids also have zero visible
free-depth contradictions. These are bounded checks, not exhaustive image-ray
or general free-space certification.

The archived map has **18 / 1 FREE cell centers** inside the conservative
static/v43 models at the observed face/grid boundary scale. There are **zero
complete FREE voxels** wholly inside the strict model interiors after the
existing 2 mm band and 10 mm grid quantization are accounted for. Thus no strict
volume-interior contradiction was demonstrated, but boundary-level disagreement
remains. It is not legitimate to claim exact agreement with every FREE map
classification. Those classifications are retained unchanged.

Accepted components have **zero points overlapping the synchronized conservative
robot envelopes**. Five static components/5,048 points and seven v43 components/
4,422 points are withheld as possible robot observations. The envelopes can
reject legitimate nearby observations too; they are not universal identity
classification. The prior 1,024-ray filter audit is preserved separately as
historical evidence, without another mapping update in this experiment.

The research audit permits a selected proxy to be subsumed only if all eight
of its corners lie inside the model with the existing conservative 1e-12 m
numerical check. Every floor proxy and every uncovered proxy must remain.

| Hypothetical combined representation | Static | v43 |
| --- | ---: | ---: |
| Original selected cuboids | 64 | 64 |
| Selected floor cuboids retained | **28** | **57** |
| Whole nonfloor cuboids contained by model | 19 | 2 |
| Total preserving the original selected union | **64 - 19 + 1 = 46** | **64 - 2 + 1 = 63** |
| Within normal 64-proxy limit | Yes | Yes |

This is a geometric feasibility audit, not a modified production packet. All
original observed volume can remain represented; no omitted occupied evidence
is claimed recovered. Occupancy arrays, IDs, FREE/UNKNOWN states and exact-cell
packet contracts were not edited. Model interiors remain explicitly inferred.

## 6. Negative controls and focused verification

The four previous controls are reproduced exactly from the same sensor data.
An additional robot-connected control joins the observed cloud to a captured
robot-mesh vertex with a <=4 mm point chain, remaining under the cloud cap.

| Fixed negative control | Static | v43 |
| --- | --- | --- |
| Remove measured lower strip and its calibrated image ribbons | Unqualified lower bound; no model | Unqualified lower bound; no model |
| Robot-only observations | Robot-envelope rejection | Robot-envelope rejection |
| Robot-connected observations | Entire component rejected | Entire component rejected |
| Merge a measured translated copy | Free-ray contradiction rejection | Free-ray contradiction rejection |
| Move valid depth observations 50 mm behind proposed surfaces | Free-ray contradiction rejection | Free-ray contradiction rejection |

**10/10 controls withhold geometry.** Lower-bound flags become unqualified/null
even though other visible patches remain. No GT-shaped negative fixture was
generated and no threshold was tuned against reference dimensions. The merged
control includes inconsistency with current depth; passing it does not rule out
every possible merged or non-box object.

From the published repository location, **14 focused tests pass, 1.897 s, no
skips**, with the external captures supplied. They check state classification,
unique-pixel counting, rejection of inadequate/unobservable support, retained
robot/bound guards, all historical seals, source lineage, camera provenance,
exact deterministic positive/control replay, the previous XYZ-median regression,
the old fixed-grid failure and absence of production imports of research.
Without archives, the same suite passes its 11 portable checks and explicitly
skips three replay tests. Architecture and Python syntax checks pass. No full
production suite or native run was needed for this isolated publication.

## 7. CPU cost and remaining planning blockers

| Single host call | Static | v43 |
| --- | ---: | ---: |
| Existing reference reconstruction, wall ms | 200.749 | 161.725 |
| Added visibility rule, wall ms | **1.668** | **2.764** |
| Added rule, process CPU ms | **1.674** | **17.996** |
| Entire reference, process CPU ms | 1,913.175 | 1,787.078 |

Process CPU includes all process threads; these single-call measurements include
host scheduling/thread activity and are not latency distributions or SDK-owner
pauses. The added rule does no fitting, dense fusion or mapping integration.
Native Python 3.11 and university-workstation freshness/rate gates were not
measured. No CPU/GPU/thread configuration or dependency was changed.

Floor/base rejection remains: selected floor geometry is retained, so these
models cannot unblock the same rejected base-preflight world. The known support
Plane, collision resources and all existing checks remain unchanged. Source
freshness also remains independent: the earlier static capture age of
**1.060005840 s** still exceeds **1.000 s**, and the archived source usability
flags remain false. This offline assessment does not revalidate or refresh them.

Two frozen ideal simulated boxes do not establish reliability for `large`,
`partial`, dynamic objects, arbitrary shapes or unobserved extensions. No new
native result, obstacle-induced detour or collision-safe RGB-D motion is claimed.

## 8. Portable preservation and exact reproduction

Validated research is published under
[`research/rgbd_box_geometry/`](../../research/rgbd_box_geometry/README.md).
It preserves the old prototype byte-for-byte, the previous fixed-grid method
with only its package import adapted, the new visibility rule, portable input/
prediction/scoring tools, focused tests and small immutable numeric evidence
for all three methods. Historical predictions and reports remain unchanged.
Nothing under production imports this package.

No raw image/robot-mesh capture, private environment or machine-specific source
path is committed. [`external-inputs.json`](../../research/rgbd_box_geometry/external-inputs.json)
records the external archive paths and hashes separately. Git now preserves the
research source, tests and numerical evidence even if the VM is deleted; a full
sensor replay still requires those external raw bytes. This publication does
not claim to back up the raw captures.

Exact CLI form from the repository root, using the existing Python environment:

```bash
python -m research.rgbd_box_geometry.run \
  --captures /path/to/rgbd-base-clearance-20261113 \
  --output /path/to/new-blind-run --method visibility
python -m research.rgbd_box_geometry.score \
  --captures /path/to/rgbd-base-clearance-20261113 \
  --output /path/to/new-blind-run
RGBD_RESEARCH_CAPTURE_ROOT=/path/to/rgbd-base-clearance-20261113 \
  python -m unittest research.rgbd_box_geometry.test_research \
  research.rgbd_box_geometry.test_preservation -v
```

The actual host executable was `/tmp/robot-v5-realignment-py312/bin/python`,
Python 3.12.16 with NumPy 1.26.4 and existing dependencies. The local frozen
run is retained under the existing research workspace's
`visibility-final-20261114/run-01/`. No output directory is overwritten.

Before publication, **55 previous local research files and all 16 original
archive files** were hash-verified unchanged. All nine sources declared before
scoring still match. Key SHA256 bindings are:

| Artifact | SHA256 |
| --- | --- |
| Original/published `reference.py` | `9969eda0b98e7258c7475f9476c644e6f74b14502753d3c24b39690bd3b0dba0` |
| `visibility.py` | `266949860bc86fd39dd019218b8c9a6bdfaab58695867fab8d8b41edf6d8422a` |
| Visibility `blind-results.json` | `a57ea0cc22ec4e5e58f051b86f8cafc2177a3f41d94fcfe4e7b61b150684cd5a` |
| Visibility `negative-controls.json` | `85f39bd630b05eb397e0aa67de5c0eda86827a301854353c6e3a846fdc09cf05` |
| Visibility `scoring.json` | `2564f62de55e6cc942c7e0ffdf403bc179ef81ebb74ba99382f63fbd264df736` |

`publication-manifest.json` binds the complete published research package;
`lineage.json` explains the original source/evidence bindings. Only validated
experimental artifacts and this new report are committed. Production source,
runtime workers, planners, collision resources, camera acquisition, source
freshness, launchers, existing tests and historical documents remain untouched.
The authorized experiment is complete; stop after publication.
