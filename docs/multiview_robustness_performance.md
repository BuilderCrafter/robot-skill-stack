# Multi-view robustness and performance

The current pipeline preserves bounded frame/point processing, immutable workers and explicit stale/ambiguous-source behavior. Nominal RGB-D global execution is a candidate with limited throughput and narrow freshness margins; it does not establish general scene or persistent-session reliability.

Current configuration and retained checks are documented in
[perception](perception.md), [multi-view](multiview_rgbd.md),
[motion planning](motion_planning.md) and [validation](validation.md).

The detailed historical study and its experiment commands remain on the preserved
[source checkpoint](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_robustness_performance.md).
Those research drivers are excluded from main's supported test commands.
