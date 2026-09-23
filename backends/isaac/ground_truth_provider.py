from __future__ import annotations

import numpy as np

from core.types import Pose

class IsaacGroundTruthProvider:
    """
    Temporary pre-perception world-state provider.

    Maps semantic object IDs to Isaac scene objects and reads their real simulated world pose.

    This entire class is intended to be replace by perception later.
    """

    def __init__(self, objects: dict) -> None:
        self.objects = objects

    def get_object_pose(self, object_id: str) -> Pose | None:
        obj = self.objects.get(object_id)

        if obj is None:
            return None

        position, orientation = obj.get_world_pose()

        return Pose(
            position=np.asarray(position, dtype=float),
            orientation=np.asarray(orientation, dtype=float),
            frame="world"
            )