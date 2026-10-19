from robot_skill_stack.runtime.skill import BaseSkill


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name):
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    def names(self):
        return list(self._skills)
