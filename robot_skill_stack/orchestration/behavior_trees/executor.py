import asyncio

import py_trees

from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import SkillNode


class BTExecutor:
    def __init__(self, root):
        self.root = root
        self.tree = py_trees.trees.BehaviourTree(root)
        self.root.setup_with_descendants()

    def tick(self):
        self.tree.tick()
        return self.root.status

    async def run_async(
        self,
        wait_next,
        max_ticks=10000,
        verbose=False,
    ):
        try:
            for tick in range(1, max_ticks + 1):
                status = self.tick()
                if verbose:
                    print(f"\n--- BT Tick {tick} ---")
                    print(
                        py_trees.display.unicode_tree(
                            self.root,
                            show_status=True,
                        )
                    )
                if status != py_trees.common.Status.RUNNING:
                    return status
                await wait_next()
            return py_trees.common.Status.FAILURE
        finally:
            pending = [node._task for node in self.root.iterate()
                       if isinstance(node, SkillNode) and node._task is not None and not node._task.done()]
            if pending:
                self.root.stop(py_trees.common.Status.INVALID)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
