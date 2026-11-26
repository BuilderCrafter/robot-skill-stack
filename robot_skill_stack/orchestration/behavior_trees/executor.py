from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import py_trees

from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import SkillNode


class BTExecutor:
    """Tick a tree cooperatively and cancel/drain unfinished skills on exit."""

    def __init__(self, root: py_trees.behaviour.Behaviour) -> None:
        self.root = root
        self.tree = py_trees.trees.BehaviourTree(root)
        self.root.setup_with_descendants()

    def tick(self) -> py_trees.common.Status:
        self.tree.tick()
        return self.root.status

    async def run_async(
        self,
        wait_next: Callable[[], Awaitable[object]],
        max_ticks: int = 10000,
        verbose: bool = False,
    ) -> py_trees.common.Status:
        """Await wait_next between RUNNING ticks; its return value is ignored."""
        try:
            for tick_number in range(1, max_ticks + 1):
                status = self.tick()
                if verbose:
                    print(f"\n--- BT Tick {tick_number} ---")
                    print(py_trees.display.unicode_tree(self.root, show_status=True))
                if status != py_trees.common.Status.RUNNING:
                    return status
                await wait_next()
            return py_trees.common.Status.FAILURE
        finally:
            pending_tasks = [node._task for node in self.root.iterate()
                             if isinstance(node, SkillNode) and node._task is not None and not node._task.done()]
            if pending_tasks:
                self.root.stop(py_trees.common.Status.INVALID)
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
