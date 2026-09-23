from __future__ import annotations

from typing import Protocol

from core.types import Pose
from world_model.entities import WorldObject

class WorldStateProvider(Protocol):
    """
    Source capable of providing object poses.

    Current implementation:
        Isaac ground truth

    Future implementation:
        perception system / world-state estimator
    """

    def get_object_pose(self, object_id: str) -> Pose | None:
        ...

class WorldModel:
    """
    Semantic state of the enviroment.

    Skilss ask this object about the world rather than talking directly to Isaac prims.
    """

    def __init__(self, state_provider: WorldStateProvider | None = None):
        self._objects: dict[str, WorldObject] = {}
        self.state_provider = state_provider

        # None means the robot currently believes that its gripper is empty.
        self.held_object_id: str | None = None

    # Object registration
    def register(self, obj: WorldObject) -> None:
        self._objects[obj.object_id] = obj

    def exists(self, object_id: str) -> bool:
        return (object_id in self._objects)

    def get(self, object_id: str) -> WorldObject | None:
        return self._objects.get(object_id)

    def require(self, object_id: str) -> WorldObject:
        obj = self.get(object_id)
        if obj is None:
            raise KeyError(f"Unknown object {object_id}")
        return obj

    def refresh_object(self, object_id: str) -> bool:
        """
        Refresh the object's pose from the currently configured world-state provider.
        
        Returns:
            True if a new pose was obtained
        """

        if self.state_provider is None:
            return False

        obj = self.get(object_id)
        pose = (self.state_provider.get_object_pose(object_id))

        if pose is None:
            return False

        obj.pose = pose

        return True

    def refresh_all(self) -> None:
        for object_id in self._objects:
            self.refresh_object(object_id)

    def set_held(self, object_id: str | None) -> None:
        if object_id is not None and not self.exists(object_id):
            raise KeyError(f"Unknown object: {object_id}")

        self.held_object_id = object_id