from __future__ import annotations

import asyncio
from typing import Any

import py_trees

from robot_skill_stack.runtime.runtime import RobotRuntime
from robot_skill_stack.runtime.skill import SkillResult


class SkillNode(py_trees.behaviour.Behaviour):
    """Bridge one runtime task to BT status, retaining the completed skill result."""

    def __init__(
        self,
        name: str,
        runtime: RobotRuntime,
        skill_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(name)
        self.runtime = runtime
        self.skill_name = skill_name
        self.kwargs: dict[str, Any] = kwargs
        self.result: SkillResult | None = None
        self._task: asyncio.Task[SkillResult] | None = None

    def initialise(self) -> None:
        self.result = None
        self._task = asyncio.ensure_future(
            self.runtime.execute_async(self.skill_name, **self.kwargs)
        )

    def update(self) -> py_trees.common.Status:
        if self._task is None or not self._task.done():
            return py_trees.common.Status.RUNNING

        self.result = self._task.result()
        self.feedback_message = self.result.message
        return (
            py_trees.common.Status.SUCCESS
            if self.result.ok
            else py_trees.common.Status.FAILURE
        )

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if (
            self._task is not None
            and not self._task.done()
            and new_status != py_trees.common.Status.RUNNING
        ):
            self._task.cancel()
