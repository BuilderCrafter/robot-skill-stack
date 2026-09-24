from __future__ import annotations

import numpy as np

from core.types import Pose
from world_model.observations import ObjectObservation


class IsaacGroundTruthProvider:
    def __init__(self, objects: dict, object_configs: dict | None = None):
        self.objects = objects
        self.object_configs = object_configs or {}

    def observe(self) -> list[ObjectObservation]:
        observations = []

        for object_id, obj in self.objects.items():
            position, orientation = obj.get_world_pose()
            cfg = self.object_configs.get(object_id)

            observations.append(
                ObjectObservation(
                    object_id=object_id,
                    class_name=object_id,
                    pose=Pose(
                        position=np.asarray(position, dtype=float),
                        orientation=np.asarray(orientation, dtype=float),
                        frame="world",
                    ),
                    size=None if cfg is None else cfg.size,
                    graspable=None if cfg is None else cfg.graspable,
                    visible=True,
                    confidence=1.0,
                    source="ground_truth",
                )
            )

        return observations

    def get_object_pose(self, object_id: str) -> Pose | None:
        obj = self.objects.get(object_id)
        if obj is None:
            return None

        position, orientation = obj.get_world_pose()
        return Pose(
            position=np.asarray(position, dtype=float),
            orientation=np.asarray(orientation, dtype=float),
            frame="world",
        )