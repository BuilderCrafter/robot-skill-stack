"""
Compatibility shim.

New code should import from `manipulation.grasping`. Existing tests and callers
can keep importing `GraspPlan` / `TopDownGraspPlanner` from this module.
"""

from manipulation.grasping.top_down import TopDownGraspPlanner
from manipulation.grasping.types import GraspPlan

__all__ = ["GraspPlan", "TopDownGraspPlanner"]
