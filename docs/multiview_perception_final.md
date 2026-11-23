# Multi-view perception demonstration

Open `scenes/multiview_perception_final.usda`, select
`config/scenes/multiview_perception_final.toml` in the Robot Skill Stack panel,
and initialize through the normal GUI. The original two 640×480 cameras remain
editable in Isaac's Stage; restart before switching setups. See
[configuration](configuration.md) for source selection and camera paths.

| Reference object | Center XYZ, m | Physical geometry |
|---|---|---|
| Square | (.36, −.16, .0275) | 55 mm cube, yaw 25° |
| Rectangle | (.54, .16, .024) | 68×42×48 mm box, yaw −33° |
| SmallSphere | (.36, .16, .022) | Radius 22 mm |
| LargeSphere | (.54, −.16, .03) | Radius 30 mm |

These are scene/evaluation references, never classifier inputs. Perception uses
only camera depth/calibration, workspace and support height. Native calibration
and measured errors are retained in the [runtime study](multiview-native-validation.md).
Both cameras must produce distinct synchronized exposures; adding views increases
rendering/computation and does not guarantee hidden-geometry observability.

The [multiview guide](multiview_rgbd.md) defines current segmentation, primitive
fitting, box qualification and persistent-ID behavior. A visible or plausible
object is not automatically graspable. Saved native static and pre-grasp captures
now qualify all four objects, including the fragmented Square, but continuous
availability remains a separate issue: the native Xeon/L4 study measured stale
visibility gaps at about 2 Hz publication. Offline replay does not establish a
reliable live Pick/Place demo. Manual GUI operation and full multiview manipulation
were not validated in that headless study.

Default motion remains direct RMPflow. Frozen RGB-D global planning is separately
opt-in with independent mapping, enhanced collision geometry and qualification;
[static GT global planning](motion_planning.md) remains the primary global demo.
Do not bypass source or intended-contact/carrying restrictions to run this scene.
Use the [retained validation commands](validation.md).

The original detailed setup, figures and workstation measurements remain at the
[preserved research checkpoint](https://github.com/BuilderCrafter/robot-skill-stack/blob/cdd2be9bfa6e95295c83b41d69819579b6dcde94/docs/multiview_perception_final.md).
