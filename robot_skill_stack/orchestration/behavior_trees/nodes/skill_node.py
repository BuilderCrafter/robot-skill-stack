import asyncio

import py_trees

from robot_skill_stack.runtime.runtime import RobotRuntime


class SkillNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name,
        runtime: RobotRuntime,
        skill_name: str,
        **kwargs,
    ):
        super().__init__(name)
        self.runtime = runtime
        self.skill_name = skill_name
        self.kwargs = kwargs
        self.result = None

    def initialise(self):
        self.result = None

    def update(self):
        self.result = self.runtime.execute(
            self.skill_name,
            **self.kwargs,
        )
        self.feedback_message = self.result.message
        return (
            py_trees.common.Status.SUCCESS
            if self.result.ok
            else py_trees.common.Status.FAILURE
        )


class AsyncSkillNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name,
        runtime: RobotRuntime,
        skill_name: str,
        **kwargs,
    ):
        super().__init__(name)
        self.runtime = runtime
        self.skill_name = skill_name
        self.kwargs = kwargs
        self.result = None
        self._task = None

    def initialise(self):
        self.result = None
        self._task = asyncio.ensure_future(
            self.runtime.execute_async(
                self.skill_name,
                **self.kwargs,
            )
        )

    def update(self):
        if self._task is None or not self._task.done():
            return py_trees.common.Status.RUNNING

        self.result = self._task.result()
        self.feedback_message = self.result.message
        return (
            py_trees.common.Status.SUCCESS
            if self.result.ok
            else py_trees.common.Status.FAILURE
        )

    def terminate(self, new_status):
        if (
            self._task is not None
            and not self._task.done()
            and new_status != py_trees.common.Status.RUNNING
        ):
            self._task.cancel()
