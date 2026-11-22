# Multi-view RGB-D perception

`world.provider = "multi_view"` selects the existing geometry-based multi-camera
source. It combines synchronized depth, calibration, workspace cropping, voxel
fusion, clustering, primitive fitting and persistent association. The ordinary
playground remains single-camera geometry V1 by default.

Use `scenes/multiview_perception_final.usda` with
`config/scenes/multiview_perception_final.toml` for the original two editable
640×480 cameras. Camera configuration is checked at initialization; processing
uses the declared point and fitting budgets. Adding views does not make all
hidden geometry observable or all objects graspable.

Refinement splits well-covered 4 mm components only when supported primitive fits
have separate conservative horizontal bounds and no competing single-solid fit.
Unresolved multi-component explanations remain UNKNOWN; candidates without a
credible split retain ordinary single-object fitting. Oversized connected
background is rejected as a whole. Foreground recovery, surface reassignment and
camera-gap resolution are outside this demo policy. Close rotated boxes or
cylinders whose world-axis bounds overlap can remain unresolved.

Semantic processing runs in bounded SDK-free workers. The owner retains frame
identity, association and WorldModel updates. Spatial Mapping can reuse the same
exposures; it publishes independent occupancy rather than replacing semantic IDs.
See [architecture](architecture.md), [configuration](configuration.md),
[perception](perception.md) and [mapping](spatial_mapping.md).

```bash
./run_isaac_python.sh scripts/run_all_regressions.py --unit-only
./run_isaac_python.sh scripts/run_all_regressions.py --group multiview --artifacts-dir outputs/multiview
```

Historical camera studies, plots, large benchmark matrices and reproduction tools
remain in the preserved [source checkpoint](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_rgbd.md).
They are excluded from the current production regression inventory.
