# Final RGB-D box reconstruction feasibility gate

14 November 2026, UTC. Starting branch `feature/global-rgbd-planning-v5`, clean
HEAD `3b7464d250a76528dcb036b5f25a72f4e3846ed3`; fetched origin and independent
`git ls-remote` agreed. Read [AGENTS.md](../../AGENTS.md), the
[design checkpoint](rgbd-reconstruction-design.md), and the relevant existing
projection, fitting, observability, robot-filter, occupancy, frozen-world and
V5 planning/evaluator code.

**Recommendation: NO-GO. Conclude this bounded RGB-D reconstruction attempt.**
The sensor-only procedure identifies the correct nonrobot component in both
captures, but accepts only v43. Static fails the unchanged box surface-coverage
criterion. No threshold adjustment, resampling search, alternate reconstruction
algorithm or runtime work followed this result. Pixel evidence does support
conditional rear/lower extents under the declared single-box model; absence of
face-interior samples alone is not the failure explanation. This experiment
does not prove that reconstruction is fundamentally impossible. It fails the
required two-scene feasibility gate within the chosen bounded reuse approach.

Only this report is published. The stopped prototype, inputs manifests, sealed
predictions, independent scores and tests are retained outside the repository:
`/home/ubuntu/research/rgbd-box-feasibility-20261114/`. No production interface,
configuration, fixture, expected result, existing documentation, launcher,
runtime, planner, worker contract, SDK/container or stream was changed. No Isaac
process or native motion was launched. GUI finalization remains a separate
reviewed task; it was not started here.

## Experiment and implementation boundary

The only captures used were:

```text
outputs/motion-planning-v5/rgbd-base-clearance-20261113/static/frozen_rgbd/
outputs/motion-planning-v5/rgbd-base-clearance-20261113/v43/static/
```

These are the same recent archives examined in Task 1, not new measurements or
the original seven-case primary cohort. Each accepted input contains two
640×480 depth frames and synchronized robot snapshots, with `ground_truth=None`.
Neither archived RGB-D source is certified usable for live planning by this
experiment. In particular, the historical static freshness failure remains.

`prototype.py` is an offline-callable NumPy research module. `blind.py` drives
the two fixed cases; `score.py` is the separate reference reader;
`robot_audit.py` verifies the conservative robot rejection against the existing
mapper filter; `test_prototype.py` performs focused checks. These scripts use
private numerical helpers and the local archive layout and remain stopped
research material outside Git. They are not imported by the application.

The fixed algorithm was:

1. Reuse `project_depth`, `crop_workspace`, `voxelize` and `cluster_points` from
   `world/perception_multiview/`: stride 2, 2 mm cloud voxels, 12,000-point cap,
   8 mm clustering and minimum 25 points. Keep the largest 16 components.
2. Validate each robot snapshot's camera token and simulation time. Transform
   the existing 11 robot-part meshes into world coordinates and form per-part
   AABBs expanded by the existing 5 mm robot-filter depth tolerance. Reject an
   entire component if any point overlaps an envelope. This is conservative
   exclusion of possible robot components, not exact semantic classification.
   No new robot capture or dense mesh ray tracing is required.
3. Consider at most four remaining components. Use at most 1,200 fit points
   and the existing `_box_search` coarse/refined yaw search. Observed extrema
   propose all dimensions, including Z; minimum Z supplies no hidden-face
   evidence. Reuse `_box_distance` and `_hull_area` for the existing 1.5 mm
   residual / top-coverage score, with the unchanged 0.65 acceptance threshold.
   There is no support-height or fixture-size prior.
4. For each of six faces, require measured plane support or current calibrated
   edge witnesses. Plane support requires at least six interior samples,
   sufficient span and a bounded plane residual/slope. For unseen extents,
   generalize the existing observability ribbon method to a directly measured
   carrier face: 48 edge positions, radius-three image ribbons, adjacent
   occupied/free pixels in three separated bins, and the existing 2 mm depth
   band. Nearer foreground and invalid depth do not establish free space.
   Unqualified bounds remain explicitly null.
5. Check a fixed small surface grid against current calibrated pixels for
   observed free-ray contradictions. Return numeric inferred geometry only
   when score, all face bounds and consistency checks pass. The conservative
   model envelope includes the existing depth band and its oblique projection;
   it is separate from occupancy, whose states are never modified.

Every result preserves original camera IDs/tokens, acquisition times, K and
world-from-camera matrices. Timing is recorded separately from deterministic
numeric predictions. There is no worker, SDK owner, mapping service, planner
or execution connection. The prototype uses one numerical box hypothesis per
scene, below the four-hypothesis cap and the existing 24-fit envelope.

## Blind-result separation and reproducibility

The definitive run is `run-02/`. Its policy/source hashes were declared before
execution. Predictions and negative controls were saved and sealed at
**10:47:46.017364 UTC**. Only after verifying that seal did the scoring process
read `geometry_reference.json`, beginning at **10:49:17.891790 UTC**. Scoring
associated the sole precomputed nonrobot component with the sole reference
solid; it did not select, crop or refit sensor points. The earlier design task
had already examined these references, so this is algorithm/input separation,
not a claim of an analyst who had never seen reference dimensions. No reference
geometry or fixture identity is read by the prototype or blind driver.

`run-01/` is retained as superseded: an implementation error took the top-plane
median over XYZ instead of Z. It was corrected before this task's reference
scoring, without changing the algorithm or thresholds. A horizontal-translation
invariance regression now catches that error. Its rejected static output is
not counted as experimental evidence. No prototype change followed run-02's
geometry result or GT scoring.

Exact commands, with the existing host environment:

```bash
cd /home/ubuntu/projects/robot-skill-stack
PYTHONPATH=. /tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/blind.py \
  --repo /home/ubuntu/projects/robot-skill-stack \
  --out /home/ubuntu/research/rgbd-box-feasibility-20261114/run-02

# Executed only after blind-seal.json existed and its hashes were verified:
PYTHONPATH=. /tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/score.py \
  --repo /home/ubuntu/projects/robot-skill-stack \
  --run /home/ubuntu/research/rgbd-box-feasibility-20261114/run-02

PYTHONPATH=. /tmp/robot-v5-realignment-py312/bin/python \
  /home/ubuntu/research/rgbd-box-feasibility-20261114/robot_audit.py

cd /home/ubuntu/research/rgbd-box-feasibility-20261114
PYTHONPATH=/home/ubuntu/projects/robot-skill-stack \
  /tmp/robot-v5-realignment-py312/bin/python -m unittest -v test_prototype
```

The blind driver refuses to overwrite an existing run. For a later independent
reproduction, choose a new output directory; the recorded run-02 stays fixed.
Environment: Python 3.12.16, NumPy 1.26.4, py_trees 2.6.0, SciPy 1.15.3,
PyYAML 6.0.2. No dependency was installed for this checkpoint. This is host
offline evidence, not native Python 3.11 or university-workstation performance.

## Identification and numeric geometry

| Measurement | Static | v43 |
| --- | --- | --- |
| Dense points / components | 9,977 / 6 | 5,363 / 8 |
| Nonrobot components, selected without reference input | One: component 0, 4,929 points | One: component 3, 941 points |
| Independent association confirms correct obstacle | Yes | Yes |
| Rejected robot-overlapping components / points | 5 / 5,048 | 7 / 4,422 |
| Surface hypothesis center, m | (0.450000045, 0.000015096, 0.350015687) | (0.550000378, -0.224971385, 0.300034337) |
| Surface hypothesis dimensions XYZ, m | (0.090001105, 0.119970613, 0.259969480) | (0.060000622, 0.059945225, 0.119933894) |
| Fitted yaw | Approximately 0, modulo equivalent box axes | Approximately 0, modulo equivalent box axes |
| Surface residual p80 | 0.000965 mm | 0.000693 mm |
| Top hull fill / score | 0.784949 / **0.030929** | 0.924354 / **0.902215** |
| Prototype outcome | **Rejected**, score below 0.65 | Empirically qualified under single-box model |
| Accepted model center, m | None | (0.550000393, -0.225313668, 0.299527505) |
| Accepted model dimensions XYZ, m | None | (0.064000592, 0.064629598, 0.124946930) |

Both fitted rotations are numerically identity: yaw
`-1.9377047211159066e-18` rad, quaternion approximately `(1,0,0,0)` in wxyz.
This describes a fitted face grid, not an independently calibrated angular
confidence interval. No object name, reference position or dimension is used
to select these components.

Static's tiny residual and accurate envelope do not override its failing
coverage score. The existing score combines residual with top-hull coverage
through `clip((fill-.78)/.16, 0, 1)`. The capped fit samples yield inadequate
top coverage for that rule. No lower threshold, different sampling strategy or
alternative score was tried after failure. A rejected hypothesis is not an
accepted collision solid.

## All six faces and hidden bounds

| Face | Static evidence | v43 evidence |
| --- | --- | --- |
| -X | Direct plane: 500 interior points | Direct plane: 64 interior points |
| +X | Direct plane: 511 interior points | Direct plane: 115 interior points |
| -Y rear | No interior plane samples; +X carrier / primary-camera boundary, 3 bins, 68 transitions | No interior plane samples; +X carrier / primary-camera boundary, 3 bins, 34 transitions |
| +Y | Direct plane: 1,559 interior points | Direct plane: 258 interior points |
| -Z lower | No interior plane samples; +Y carrier / side-camera boundary, 3 bins, 62 transitions | No interior plane samples; +Y carrier / side-camera boundary, 3 bins, 32 transitions |
| +Z | Direct plane: 105 interior points | Direct plane: 37 interior points |

The table reports the selected strongest bound for each hidden face; additional
carrier/view trials and exact pixel witnesses are retained in the blind JSON.
For example, v43's lower-bound free witnesses are side-camera pixels
(306,48), (298,47), (290,46); its rear witnesses are primary-camera pixels
(233,246), (231,230), (230,223). These were computed before reference scoring.

Rear/lower extents are **conditionally bounded by measured edge transitions
under the single convex yaw-box assumption**. They are not directly observed
rear/bottom planes, table-support guesses, learned completion or a general
physical containment certificate. The missing-lower-bound control removes
these witnesses and leaves the lower bound unqualified in both scenes.
Static's per-face checks pass, but its full model remains rejected by the
independent coverage rule. No static collision geometry is emitted.

## Independent extent and containment scores

| Geometry scored | Maximum absolute face-extent error | Minimum containment slack | Contains reference within unchanged 10 µm allowance? |
| --- | ---: | ---: | --- |
| Static surface hypothesis, **not accepted** | 0.030942 mm | -0.030942 mm | No |
| v43 surface hypothesis, before outward evidence bands | 0.067389 mm | -0.067389 mm | No |
| v43 accepted empirical model | 2.945962 mm, outward excess | +1.999904 mm | Yes, also without allowance |

Errors compare all reference vertices with predicted local box face extents.
Positive containment slack means all reference vertices lie inside the box.
These results explain why a very accurate measured AABB still cannot serve as
a conservative solid. The v43 envelope contains this one reference capture;
that fact does not establish calibrated uncertainty, robustness across scenes,
or continuous robot collision safety.

## FREE evidence, robot rejection and proxy preservation

The current-pixel free-ray check found **0/76** contradictions for static's
surface hypothesis and **0/78** for v43's. It is a bounded check, not exhaustive
pixel/ray consistency. The separate archived-map audit found 18 FREE cell
centers inside static's unaccepted envelope, and **one FREE cell center** inside
v43's accepted envelope. The v43 center is (0.555, -0.195, 0.305) m, at the
observed face/grid boundary scale. No complete FREE voxel lies wholly inside
either strict model interior after the unchanged 2 mm band and 10 mm grid
quantization are accounted for.

Thus no volume-interior FREE contradiction was demonstrated, but a clean
absence-of-conflict claim is unwarranted: boundary-level occupancy disagreement
remains. Map cell classifications were neither erased nor relabeled. No extra
algorithm was introduced to reconcile these cells after failure.

All points in each of the 12 rejected components overlap the conservative robot
envelopes. The two retained obstacle components have zero such points. A
post-freeze audit reused the existing `RgbdMapper` allocation and
`RobotSelfFilter` on **exactly 1,024 rays per archive**: all 35 static and all
34 v43 automatically masked endpoints were inside the envelopes. Four/three
retained endpoints also overlapped an envelope, demonstrating its conservative
excess rejection. This is not proof of robot classification in arbitrary scenes.
The audit used an archive-local clock only to reproduce those original rays;
it did not refresh camera timestamps or validate live freshness.

| Selected-evidence audit | Static | v43 |
| --- | --- | --- |
| Original selected cuboids / protected floor cuboids | 64 / 28 | 64 / 57 |
| Whole cuboids contained by accepted inferred model | No accepted model | 2 nonfloor cuboids |
| Total needed while preserving all selected geometry | Original 64 remain; no valid overlay | **63**: original 64 minus 2 plus 1 model |
| Within normal 64-proxy budget? | No accepted reconstruction to assess | Yes, geometrically |

Containment checks all eight corners of each original cuboid with a conservative
1e-12 m numerical allowance; center-only containment is not used. Every selected
floor cuboid is protected. For transparency, static's unaccepted raw envelope
would contain 18 whole nonfloor boxes and give 47 total proxies; that arithmetic
does not authorize its use. The v43 result removes no selected occupied volume.
Neither case claims recovery of previously omitted observations or changes the
map's UNKNOWN states. Existing floor/base rejection and source freshness remain
independent unresolved planning gates.

## Fixed negative controls and verification

Controls are in-memory perturbations of the same archived observations or
captured robot meshes; original files stay unchanged. No extra captures or
GT-generated object shapes are used.

| Fixed control, per scene | Static outcome | v43 outcome |
| --- | --- | --- |
| Remove the measured lower 8 mm cloud strip and calibrated lower-edge pixel ribbons | Rejected; lower bound unqualified, also fails coverage | Unqualified lower bound; no geometry |
| Supply up to 1,200 sampled vertices from captured robot parts | Robot-envelope rejection | Robot-envelope rejection |
| Merge component with a copy translated by 0.6 of its measured horizontal extent | Rejected; 50 current free-ray conflicts | Rejected; 43 current free-ray conflicts |
| Shift valid depth observations 50 mm behind the proposed model while retaining the component | Rejected; 76 current free-ray conflicts | Rejected; 78 current free-ray conflicts |

**8/8 control outcomes withheld geometry.** Static's baseline already fails
coverage, so its rejection controls alone do not demonstrate discrimination of
accepted static models. v43 provides the positive-to-negative control change.
The merged-object control includes inconsistency with measured depth; it is
not a general proof against every visually plausible merged object.

**6 focused tests pass, 0.974 s, exit 0.** They verify exact deterministic replay
of both sealed numeric results, horizontal-translation invariance of vertical
estimation, all control outcomes, explicit null lower bounds when witnesses
are removed, rejection of mismatched robot tokens, and preservation of original
calibration/timestamps/input hashes. No full production suite or native run was
needed: the repository implementation is unchanged.

## Incremental cost and final disposition

| Single host call, milliseconds | Static | v43 |
| --- | ---: | ---: |
| Existing projection/fusion/clustering reused | 135.583 | 97.261 |
| Robot envelope construction | 9.583 | 9.789 |
| Box qualification, including rejected components | 50.404 | 59.640 |
| Increment beyond already available clouds/clusters | **59.987** | **69.429** |
| Total offline call | 195.570 | 166.690 |

These are individual wall-time samples, not p95/p99, native costs, measured
runtime overhead or steady-state throughput. Projection is counted once; the
increment excludes that existing work. The prototype adds no camera, mapping
ray, worker, model dependency or GPU resource. Workstation freshness/rate gates
were not tested and cannot be claimed from these numbers.

The final feasibility result is **1/2 accepted measured-box reconstructions**,
with unresolved boundary-level FREE agreement and no runtime evidence. Static
fails the declared acceptance rule despite useful measured extent information.
The bounded checkpoint ends here. Do not promote the v43-only result into a
runtime feature, change safety/coverage thresholds to rescue static, rerun the
base experiment, add a reconstruction family, or start native planning. Per
the task's failure disposition, conclude RGB-D global-planning research and
await explicit review for GUI finalization.

## Evidence bindings and publication scope

All 12 inputs bound by `run-02/input-manifest.json` are unchanged. A final
cross-check against the Task 1 manifest additionally verifies unchanged
references/results: **16 distinct archived files** match their pre-existing
hashes. Source snapshots for the definitive run are in `run-02/source/`;
`run-01/source/` preserves the superseded implementation. Key SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| `prototype.py` | `9969eda0b98e7258c7475f9476c644e6f74b14502753d3c24b39690bd3b0dba0` |
| `blind.py` | `604076680efd81dfe64586a746c1e3dbd4d9df436b6c3c5f7979a8db0f5b3ee6` |
| `score.py` | `c6342a57993d85710a1c228e620f050118734058e88652fd4e2f56f0f8e47ffb` |
| `run-02/static-blind.json` | `e70f784e3eeebd9948487a027970b434c3a1eb7d89ffc24c1a162cf80363deed` |
| `run-02/v43-blind.json` | `ca81089aba92da1b1634ab19014e7d976d6c87e9c813a720d0153877e44c6613` |
| `run-02/blind-results.json` | `e5c168e1b0e88af6bef52333fd5e125856fb806763d59016caafa16c747abbcc` |
| `run-02/negative-controls.json` | `7df99837b426701134f0001450b39265e2e6c1a41cd6bc852a14e74f3d981b00` |
| `run-02/scoring.json` | `22d9ad05085e525a874bcd77c4b43ea7dfe2bd4d4bb42695194c2c332d2c96a9` |
| `run-02/final-manifest.json` | `8b2922940e986a24df83a6b353a7df8800483045cf54c088421e7b4eebf1ed6d` |

The final manifest also binds the audit script, tests, logs, declaration and
original blind seal. Only `reports/finalization/rgbd-box-feasibility.md` is
committed and normally pushed to the existing V5 branch. The local recovery
work area is preserved; no runtime integration or subsequent task is started.
