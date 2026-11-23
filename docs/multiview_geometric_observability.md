# Geometric observability

Additional views reduce some occlusions but do not certify hidden geometry. The production pipeline retains uncertainty, calibrated source identity and explicit failure when evidence is insufficient. Reconstructed global planning additionally requires all six qualified bounds and the existing visibility/FREE checks.

For multiview object perception, `assess()` returns a qualified box center and
local dimensions, or `None` when the dimensions cannot be supported. The provider
applies this check only to boxes and combines it with the fitter's parameter
qualification to determine graspability. Pixel adjacency, separated edge
coverage, measured face planes and top-height uncertainty still constrain the
bounds; detailed per-camera reporting is no longer published in geometry
metadata. See the [algorithm audit and regression results](multiview-observability-audit.md).

Current configuration and retained checks are documented in
[perception](perception.md), [multi-view](multiview_rgbd.md),
[motion planning](motion_planning.md) and [validation](validation.md).

The detailed historical study and its experiment commands remain on the preserved
[source checkpoint](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_geometric_observability.md).
Those research drivers are excluded from main's supported test commands.
