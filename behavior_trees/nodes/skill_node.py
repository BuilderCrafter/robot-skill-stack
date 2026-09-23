import py_trees

from runtime.robot_runtime import RobotRuntime


class SkillNode(py_trees.behaviour.Behaviour):
    def __init__(self, name, runtime: RobotRuntime, skill_name: str, **kwargs):
        super().__init__(name)
        self.runtime = runtime
        self.skill_name = skill_name
        self.kwargs = kwargs
        self.result = None

    def initialise(self):
        self.result = None

    def update(self):
        self.result = self.runtime.execute(self.skill_name, **self.kwargs)
        self.feedback_message = self.result.message

        return (
            py_trees.common.Status.SUCCESS
            if self.result.ok
            else py_trees.common.Status.FAILURE
        )