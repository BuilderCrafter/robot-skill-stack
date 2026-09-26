# Repository restructure

The validated course-level stack was reorganized from many root-level packages
into one architectural package. The migration intentionally changes module
paths, not robot behavior.

```text
core/types.py                    → robot_skill_stack/common/types.py
core/skill.py                    → robot_skill_stack/runtime/skill.py
core/manipulation.py             → robot_skill_stack/manipulation/backend.py
runtime/{robot_runtime,registry} → robot_skill_stack/runtime/
world_model/*                    → robot_skill_stack/world/model/
perception/*                     → robot_skill_stack/world/perception/
skills/*                         → robot_skill_stack/manipulation/skills/
manipulation/grasping/*          → robot_skill_stack/manipulation/grasping/
behavior_trees/*                 → robot_skill_stack/orchestration/behavior_trees/
backends/isaac/*                 → robot_skill_stack/integrations/isaac/
runtime/runtime_builder.py       → robot_skill_stack/integrations/isaac/bootstrap.py
runtime/scene_config.py          → robot_skill_stack/integrations/isaac/config.py
```

Historical diagnostic-only scripts and temporary patch READMEs were removed.
The meaningful acceptance/regression tests now live under:

```text
tests/unit/
tests/isaac/
tests/regression/
```

The optional Isaac UI extension uses its own Python namespace
`robot_skill_stack_ui` to avoid colliding with the main `robot_skill_stack`
package.
