# Multi-view perception demonstration

Open `scenes/multiview_perception_final.usda` and select
`config/scenes/multiview_perception_final.toml` in the Robot Skill Stack panel.
Initialize through the normal GUI. The two existing 640×480 cameras remain editable
in Isaac's Stage; restart before switching setups.

The geometry pipeline supports measured primitive observations and persistent IDs.
Occlusion, merged surfaces, fit uncertainty and stale exposures can prevent a
reliable model. An observed or visually plausible object is not automatically
eligible for grasping. Default motion remains direct RMPflow; autonomous multi-view
manipulation is not a general validated capability.

Frozen RGB-D global planning is separately opt-in, with its own mapping, enhanced
collision model and qualification checks. See [motion planning](motion_planning.md)
for the nominal partial success and the remaining limits. The static GT global
option remains independently available as the primary demonstration.

Use the [retained validation commands](validation.md). The original detailed
perception checkpoint, figures and research measurements are preserved on the
[source branch](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_perception_final.md).
