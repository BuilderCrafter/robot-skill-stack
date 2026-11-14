# Final RGB-D coverage investigation

14 November 2026, UTC. Branch `feature/global-rgbd-planning-v5`, clean starting
HEAD `3024101bedb7662812b19ea13c35640d1aaf408d`. Fetch and independent remote
tip lookup agreed. Read [AGENTS.md](../../AGENTS.md), the
[reconstruction design](rgbd-reconstruction-design.md),
[previous feasibility gate](rgbd-box-feasibility.md), and
[V5 planning study](../../docs/motion_planning_v5_rgbd_global.md).

**Final recommendation: NO-GO. Close RGB-D global-planning research for this
course project and proceed to reviewed GUI/presentation finalization.**

The static rejection has an identifiable sampling/coverage cause, despite a
good surface fit. One obstacle-specific alternative was declared and tested.
It also rejected static: the required Y-face pixel grid is partly outside the
side camera's image. The first alternative therefore failed its first positive
case. Testing stopped immediately; no v43 alternative run, negative-control
batch, second rule, threshold change or runtime work followed. Subsequent work
only scored the sealed failed result, verified preservation and wrote this
report. The approximately two-hour investigation limit was not approached.

## 1. Why the original score rejects static

The relevant helpers are in
[`fitting.py`](../../robot_skill_stack/world/perception_multiview/fitting.py).
`_box_search` selects yaw by distances of measured side points to the candidate
XY rectangle. The offline prototype then uses existing full-cluster extrema
for its surface hypothesis. `_box_distance` measures distance of observed
points to that box surface; it does not penalize absent observations on other
parts of the surface. Thus small residuals and accurate observed dimensions
can coexist with poor measured top coverage.

The reference prototype uses:

```text
residual_score = exp(-0.5 * (residual_p80 / 0.0015 m)^2)
fill = hull_area(measured_top_points_xy) / (box_width * box_depth)
coverage_score = clip((fill - 0.78) / 0.16, 0, 1)
box_score = residual_score * coverage_score
acceptance requires box_score >= 0.65
```

With near-unit residual score, this requires at least **88.4% top-hull fill**.
Static's residual score is 0.999999793. Its low combined score therefore comes
almost entirely from measured top-hull fill, not geometric residual.

Task A reran only sensor-derived projection/clustering and score decomposition,
without changing thresholds, fitting or acceptance. The sole nonrobot component
was identified using the unchanged synchronized robot envelopes, without fixture
identity or reference geometry.

| Task A diagnostic | Static fit sample | Static existing full cloud | v43 existing cloud |
| --- | ---: | ---: | ---: |
| Component/sample points | 1,200 | 4,929 | 941 |
| Top-plane samples | 56 | 222 | 93 |
| Top fraction of all samples | 4.667% | 4.504% | 9.883% |
| Top XY hull area, m² | 0.008475473 | 0.009433556 | 0.003324672 |
| Estimated footprint, m² | 0.010797488 | Same | 0.003596751 |
| Hull fill | **0.784949** | **0.873681** | **0.924354** |
| Residual p80, mm | 0.000965 | 0.000949 | 0.000693 |
| Combined score at unchanged formula | **0.030929** | **0.585504** | **0.902215** |
| Meets 0.65? | No | No | Yes |

The full-cloud column is a read-only diagnosis on points already available in
the capped dense pipeline; it was not a second candidate or an increased fit
budget. Even removing the 1,200-point sampling effect would not make static
pass the original rule.

The production `SurfaceFitter.fit` additionally voxelizes at 1.5 mm before its
1,200-point cap. Applying that preprocessing diagnostically gives 50 static
top samples, fill **0.766343**, and score **0** against the same footprint.
The previously reported 0.030929 is the isolated research prototype's score;
it is not a claim that the production semantic provider emits this obstacle.

The cap reduces static's top hull area by about 10.15% relative to the existing
full-cloud top hull. It does not preferentially remove the top *fraction* in
this capture: the capped fraction is slightly larger. The problem is sparse
spatial coverage of hull boundaries/corners after subsampling. These two
captures do not establish a universal systematic sampling bias.

Full-cloud static top X samples start at approximately 0.410646 m, while the
component's measured X extent starts at 0.404999 m. Top samples therefore do
not cover the entire footprint established by other surfaces. The complete
existing top sample hull is still 12.63% short of the footprint area.

## 2. Is this insufficient data or an unsuitable criterion?

The ratio tests whether the measured top resembles a sufficiently complete
rectangle. In the original fitter it helps prevent circular caps and isolated
planar patches from becoming boxes. That is a legitimate conservative shape
check. Floating versus table-supported placement does not by itself invalidate
this mathematical check; partial visibility and sampling are the relevant
limitations here.

It is also not equivalent to obstacle observability. A single convex yaw-box
model can have measured planes and calibrated extent witnesses distributed
across several faces while its top hull is incomplete. Static's reference
result has four directly supported planes and boundary witnesses for the two
remaining extents. This is a concrete reason to test an obstacle-specific
cross-face criterion, rather than lower the old top-area threshold.

However, useful extents and a small residual alone do not justify removing the
shape-coverage check. Missing observations do not prove complete solid volume.
This investigation found a plausible mismatch in the *kind* of evidence used,
but did not validate a replacement. It cannot declare the original rejection
unnecessary or safely override it.

## 3. The single declared alternative

All work is isolated below the existing research workspace:

```text
/home/ubuntu/research/rgbd-box-feasibility-20261114/
  coverage-final-20261114/
    diagnose.py, diagnosis.json
    alternative.py, run_blind.py, score_frozen.py
    preflight-preservation.json
    run-01/  # declaration, sealed predictions, failure audit, score, sources
```

The original `prototype.py` and previous `run-02/` remain unchanged. The new
alternative consumes the existing reference result, sharing its one fit,
components, surface hypothesis and six face records. It performs no refitting,
resampling, extra projection/fusion pipeline or worker integration.

The frozen rule, `calibrated_three_axis_face_coverage`, requires:

- Unchanged robot-component rejection, finite box hypothesis, current free-ray
  contradiction rejection and **all six** existing extent checks.
- A directly measured plane from each of the three orthogonal normal-axis
  families X, Y and Z, already passing the reference plane-support check.
- On each selected plane, one camera supplies at least six unique occupied
  pixels from a fixed 3×3 interior grid at local fractions `(-0.25, 0, 0.25)`.
  Witnesses must cover all three row bins and all three column bins. Back-facing
  views, invalid depth and out-of-frame pixels supply no support.
- Occupied pixels match the calibrated plane-ray intersection within the
  unchanged 2 mm depth band. The existing residual score must still be >=0.65,
  using the unchanged 1.5 mm residual tolerance.

Its reported score is residual score gated by that Boolean three-axis coverage
decision. It is not a probability or the same quantity as the old hull-based
score. The change substitutes explicit distributed current-pixel plane evidence
for top-area coverage; it does not reduce a numeric top-area threshold. Six
unique witnesses reuse the existing plane-support minimum, and three separated
bins reuse the established boundary-evidence pattern.

Assumptions remain one static convex yaw box with empirical depth/plane bands.
No GT dimensions, fixture identity, support-height guess or learned completion
enter the rule. Fit/cloud caps remain 1,200/12,000 points, four hypotheses,
16 components and the existing two 640×480 cameras. The additional plane-grid
test reads at most 108 existing pixel positions per hypothesis; it integrates
no mapping rays and adds no worker, dependency or GPU allocation.

## 4. Fixed experiment and first failure

The runner wrote the declaration and source hashes before executing the rule.
It also required the recomputed static reference result to equal the previous
sealed result exactly. Candidate predictions were saved and hashed at
**11:42:35.073963 UTC**, before this attempt's independent reference scoring
at **11:45:39.079009 UTC**. Earlier reports already contained GT comparisons;
this separation concerns algorithm inputs and fixed predictions, not an analyst
who has never seen the references.

| Static test | Reference | Alternative |
| --- | --- | --- |
| Robot contamination in retained component | 0 envelope-overlapping points | Unchanged |
| Residual score | 0.999999793 | Same |
| Coverage score | 0.030929 top-hull score | 0 Boolean three-axis score |
| Qualification | Rejected | **Rejected** |
| Reason | Inadequate top-hull coverage | Inadequate Y-face grid distribution |
| Emitted collision geometry | None | None |

| Alternative face-grid evidence | Unique supported pixels | Row/column bins | Result |
| --- | ---: | --- | --- |
| +X, primary camera | 9 | 3 / 3 | Pass |
| +Y, side camera | 6 | **3 / 2** | **Fail** |
| +Z, primary camera | 9 | 3 / 3 | Pass |
| -X, side camera, additional trial | 6 | 3 / 2 | Fail; +X already supplies X evidence |

A read-only audit of the saved failing grid confirmed that the three missing
+Y samples project to **(322,-14), (301,-16), (282,-18)** in the side image.
Their image rows are negative. This is actual field-of-view loss under the
declared calibrated grid, not an occupied/free classification error. The other
six +Y samples have valid occupied depth. The primary view is back-facing for
this plane. No offset grid, relocated probes, weaker bin requirement or camera
change was attempted.

The six reference extent checks remain qualified: -X/+X/+Y/+Z direct-plane
interior counts are 500/511/1,559/105; -Y rear uses three-bin calibrated edge
witnesses, and -Z lower uses three-bin side-face edge witnesses. The failure is
the alternative's additional spatial face-coverage requirement, not a newly
unqualified extent. Such extent witnesses still provide conditional box-model
bounds rather than general hidden-geometry certification.

## 5. Scoring, controls and unrun conditions

After the failure, the only independent scoring was of already sealed static
data. No candidate was rerun or revised. The rejected reference surface envelope
has maximum face-extent error **0.030942 mm**, containment slack
**-0.030942 mm**, and volume **0.0028070173 m³** against reference
**0.0028079999 m³**: a **0.034996% deficit**. It does not contain the reference
within the unchanged 10 µm allowance. Tiny aggregate volume error is therefore
not a conservative containment proof.

There is **no accepted alternative box**. Its conservative containment and
excess-volume metrics are unavailable; no unqualified envelope was promoted
to fill those fields.

The unchanged static reference free-ray check reports **0/76 contradictions**.
The archived map still has 18 FREE cell centers inside the rejected raw
envelope, with no complete FREE voxel wholly inside its strict interior after
the existing depth/grid band. This remains boundary-level disagreement, not a
clean consistency certificate. No occupancy state or observed evidence changed.
All 64 original selected cuboids, including 28 floor cuboids, remain present.
No accepted combined model exists to establish new proxy-budget feasibility.
The rejected raw-envelope diagnostic's 47-proxy arithmetic is not a usable
reconstruction result.

The explicit first-failure stop took precedence over continuing Task C:

| Declared condition | This alternative's status |
| --- | --- |
| Static positive capture | Tested; failed |
| v43 positive capture | **Unrun after stop** |
| Missing-boundary controls | **Unrun after stop** |
| Robot-contamination controls | **Unrun after stop** |
| Merged-object controls | **Unrun after stop** |
| Contradictory-depth controls | **Unrun after stop** |

Task A diagnosed both archives. Historically the reference method qualifies
v43 at score 0.902215 and withheld geometry for all eight controls, as recorded
in the previous report. Those are reference results, **not alternative passes**.
The new alternative's control collection is empty. No result claims that this
candidate accepts both objects or discriminates the negative controls.

## 6. Cost, reproducibility and preservation

The single static host call measured:

| Operation | Wall time, ms | Process CPU, ms |
| --- | ---: | ---: |
| Unchanged reference reconstruction, including clouds and checks | 194.507 | 1,874.672 |
| Additional alternative coverage check | **2.053** | **2.047** |

Reference wall components were 134.870 ms projection/fusion/clustering,
9.956 ms robot envelopes and 49.681 ms qualification. Process CPU includes all
process threads; it is not an SDK-owner pause or elapsed-time measurement.
These are one-call host timings, not a distribution, workstation/native
performance or proof of runtime freshness. No BLAS/runtime resource setting
was changed. Native publication rates and existing floor/base/freshness gates
were not tested.

Commands used the existing Python 3.12.16 host environment with NumPy 1.26.4;
no dependency installation occurred. From the repository root:

```bash
export PYTHONPATH=/home/ubuntu/projects/robot-skill-stack:/home/ubuntu/research/rgbd-box-feasibility-20261114
/tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/coverage-final-20261114/diagnose.py
/tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/coverage-final-20261114/run_blind.py
# Post-stop scoring only; verifies the blind seal before reading references:
/tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/coverage-final-20261114/score_frozen.py
```

The runner refuses an existing output directory, preserves the reference and
halts on its first baseline/control failure. Do not rerun over the retained
evidence. The separate failing-pixel audit uses only the fixed saved rule and
original calibration; it performs no candidate re-evaluation.

Post-stop verification confirms **33 previous research files and 16 original
archived files unchanged**, exact reproduction of the reference static result,
unchanged declared source hashes, matching blind seals and unchanged camera
provenance. No new candidate test batch or production suite followed the stop.
No historical fixture, expected test result or report was edited.

Key SHA256 bindings, relative to `coverage-final-20261114/`:

| Artifact | SHA256 |
| --- | --- |
| `alternative.py` | `e0be5843eb55859af5dfd018520e560883bc1271a2efc7c79a8c1ddabc72d849` |
| `diagnosis.json` | `b0afab21ee0e44c9df230d29b5e98da564271f2d364052a1ee09ad3bd37e532c` |
| `run-01/blind-results.json` | `146edae41332e8473b7e12f97c8b703416b42a80d01635c59cb7a886ddce338a` |
| `run-01/static-blind.json` | `ded58b2bee94af5461395e5af6fbd084a68c05a3203eee6a998b4502c43c23fc` |
| `run-01/failure-observation-audit.json` | `4877084e1919c1f334ef12ddcaa9b796cdcbb6806d9dfa64aa36b6608070f399` |
| `run-01/scoring.json` | `c6b54120efc41e5f661d71786fca7876609abad370edbb976a51e0636d1d8a84` |
| `run-01/final-manifest.json` | `33785ec2c9e3045e83bc9b58247d0cc04d15da9e58f0c2db28383cc0636b5a96` |

The final manifest also binds source snapshots, the declaration, input manifest,
timings and explicit stopped/unrun status. Only this report is committed and
pushed to V5. All experimental code remains in the existing local research
workspace. There is no production/runtime/planner modification, Isaac launch,
container operation or GUI work. The final offline attempt is finished.
