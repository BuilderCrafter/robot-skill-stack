import py_trees


class BTExecutor:
    def __init__(self, root):
        self.root = root
        self.tree = py_trees.trees.BehaviourTree(root)
        self.root.setup_with_descendants()

    def tick(self):
        self.tree.tick()
        return self.root.status

    def run(self, max_ticks=100, verbose=True):
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

        return py_trees.common.Status.FAILURE