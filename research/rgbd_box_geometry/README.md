# Experimental offline RGB-D box geometry

This directory preserves stopped thesis research. Nothing in the robot runtime
imports it. It does not acquire cameras, run workers, launch Isaac, register
Lula obstacles or execute motion. Keep it outside production configuration.

The visibility-aware experiment passes the two archived positive scenes and
ten fixed negative controls. Independent reference scoring confirms containment
and hypothetical evidence-preserving representations within 64 proxies. This
is an **offline geometry result**, not reliable RGB-D global planning or a
physical safety certificate. The floor/base and source-freshness blockers remain.

## Preserved methods

- `reference.py`: exact byte-for-byte copy of the previous isolated prototype,
  SHA256 `9969eda0b98e7258c7475f9476c644e6f74b14502753d3c24b39690bd3b0dba0`.
- `fixed_grid.py`: the previous failed alternative; only its module import was
  changed to a relative package import. `lineage.json` binds both source hashes.
- `visibility.py`: the one newly authorized rule. A measured plane needs six
  unique supported pixels in a 2D patch spanning at least two grid bands per
  tangent. All three normal-axis families and all six original extent bounds
  remain necessary. Outside-FOV, occluded, invalid and back-facing locations
  supply no positive evidence. Visible contradictions reject the model.
- `controls.py`: the previous sensor-derived missing-boundary, robot-only,
  merged-object and contradictory-depth controls, plus an explicit observed
  component joined to a known robot vertex by a <=4 mm point chain.
- `run.py` and `score.py`: separate blind prediction and independent scoring.
  Both refuse to overwrite results; scoring validates the blind seal first.
- Focused tests validate visibility states, retained source/evidence hashes,
  guards, optional archived replay, control outcomes and deterministic geometry.

The existing 1,200-point fit, 12,000-point cloud, four-hypothesis, calibration,
robot exclusion and six-bound rules are reused. The model is one static convex
yaw box with empirical measurement bands. Scores are diagnostics, not safety
probabilities. Geometry is tagged inferred; occupancy is never completed.

## External inputs

No raw depth/RGB images, robot meshes, private environment, credentials or
machine-specific paths are distributed. The required input bytes remain
external. `external-inputs.json` lists their relative paths and SHA256 values.
An archive root must contain:

```text
static/frozen_rgbd/
v43/static/
```

Each contains `accepted_mapping_input.pickle`, `camera_frames.npz`,
`camera_metadata.json`, `runtime_profile.toml`, `occupancy.json`, `occupancy.npz`,
and the independently scored `geometry_reference.json`. The manifest also binds
the historical result JSON. Use only trusted hash-matching historical pickles.
The blind reader verifies the six sensor/map files and reads only depth,
calibration, workspace/support configuration and synchronized robot snapshots.
Only the separate scorer reads reference geometry after sealing predictions.

If the original VM is deleted, the source, tests and numerical evidence in this
directory remain recoverable from Git. A full sensor replay still requires the
external archives; the Git publication is not a backup of those raw bytes.

## Reproduce from the repository root

Use the existing project Python environment with NumPy 1.26.4 and its existing
dependencies. The tools use Python 3.11-compatible standard-library interfaces;
the recorded host run used Python 3.12.16. No new dependency is required.

```bash
python -m unittest research.rgbd_box_geometry.test_research \
  research.rgbd_box_geometry.test_preservation -v

python -m research.rgbd_box_geometry.run \
  --captures /path/to/rgbd-base-clearance-20261113 \
  --output /path/to/new-blind-run --method visibility

# Run only after the preceding command has written blind-seal.json:
python -m research.rgbd_box_geometry.score \
  --captures /path/to/rgbd-base-clearance-20261113 \
  --output /path/to/new-blind-run

# Optional archived replay checks, including the old fixed-grid failure:
RGBD_RESEARCH_CAPTURE_ROOT=/path/to/rgbd-base-clearance-20261113 \
  python -m unittest research.rgbd_box_geometry.test_research \
  research.rgbd_box_geometry.test_preservation -v
```

Without archives, three replay tests explicitly skip; the portable checks still
run. `--method reference` and `--method fixed_grid` retain the previous rules
with the same first-failure stop. They are reference reproductions, not new
alternatives. Historical numeric predictions are in `evidence/reference/` and
`evidence/fixed_grid/`; current results are in `evidence/visibility/`.

The original numeric records and seals are copied unchanged. Historical manifests
retain their original relative `outputs/...` paths for hash fidelity; the
portable external-input manifest resolves paths from the explicit archive root.
`publication-manifest.json` binds all published source, documentation and
evidence. The report is
[`rgbd-visibility-final.md`](../../reports/finalization/rgbd-visibility-final.md).

Do not infer runtime readiness from this result. No native run, integration,
second reconstruction approach or GUI modification is authorized by this package.
