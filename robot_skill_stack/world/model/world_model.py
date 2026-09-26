from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from robot_skill_stack.world.model.context import AssociationHint, ObservationContext
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.observations import ObjectObservation


class WorldModel:
    def __init__(self):
        self._objects: dict[str, WorldObject] = {}
        self._association_hints: dict[str, AssociationHint] = {}
        self.held_object_id: str | None = None

    def register(self, obj: WorldObject) -> None:
        self._objects[obj.object_id] = obj

    def exists(self, object_id: str) -> bool:
        return object_id in self._objects

    def get(self, object_id: str) -> WorldObject | None:
        return self._objects.get(object_id)

    def require(self, object_id: str) -> WorldObject:
        obj = self.get(object_id)
        if obj is None:
            raise KeyError(f"Unknown object '{object_id}'")
        return obj

    def objects(self) -> tuple[WorldObject, ...]:
        return tuple(self._objects.values())

    def find(
        self,
        *,
        class_name: str | None = None,
        graspable: bool | None = None,
        visible: bool | None = None,
    ) -> list[WorldObject]:
        result = list(self._objects.values())
        if class_name is not None:
            result = [o for o in result if o.class_name == class_name]
        if graspable is not None:
            result = [o for o in result if o.graspable == graspable]
        if visible is not None:
            result = [o for o in result if o.visible == visible]
        return result

    def visible_objects(self) -> list[WorldObject]:
        return self.find(visible=True)

    def expect_object_at(
        self,
        object_id: str,
        position,
        *,
        radius: float = 0.08,
        ttl: float = 3.0,
        reason: str = "action",
    ) -> AssociationHint:
        self.require(object_id)
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        hint = AssociationHint(
            object_id=object_id,
            expected_position=np.asarray(position, dtype=float),
            max_distance=radius,
            expires_at=time.monotonic() + float(ttl),
            reason=reason,
        )
        self._association_hints[object_id] = hint
        return hint

    def clear_association_hint(self, object_id: str) -> None:
        self._association_hints.pop(object_id, None)

    def association_hints(self) -> tuple[AssociationHint, ...]:
        now = time.monotonic()
        expired = [
            object_id
            for object_id, hint in self._association_hints.items()
            if hint.expires_at <= now
        ]
        for object_id in expired:
            del self._association_hints[object_id]
        return tuple(self._association_hints.values())

    def observation_context(self) -> ObservationContext:
        return ObservationContext(
            held_object_id=self.held_object_id,
            association_hints=self.association_hints(),
        )

    def apply_observations(
        self,
        observations: Iterable[ObjectObservation],
        *,
        mark_missing_invisible: bool = False,
    ) -> None:
        observations = list(observations)
        seen = set()
        now = time.monotonic()

        for obs in observations:
            seen.add(obs.object_id)
            obj = self.get(obs.object_id)

            if obj is None:
                obj = WorldObject(
                    object_id=obs.object_id,
                    class_name=obs.class_name,
                    pose=obs.pose,
                    size=obs.size,
                    graspable=True if obs.graspable is None else obs.graspable,
                )
                self.register(obj)

            if (
                obs.class_name is not None
                or obs.metadata.get("class_belief_authoritative", False)
            ):
                obj.class_name = obs.class_name
            if obs.pose is not None:
                obj.pose = obs.pose
            if obs.size is not None:
                obj.size = obs.size.copy()
            if obs.graspable is not None:
                obj.graspable = obs.graspable

            obj.visible = obs.visible
            obj.confidence = obs.confidence
            obj.source = obs.source
            obj.metadata.update(obs.metadata)

            if obs.visible:
                obj.last_seen = obs.timestamp if obs.timestamp is not None else now

            if obs.visible and obs.metadata.get("association_hint_match", False):
                self.clear_association_hint(obs.object_id)

        if mark_missing_invisible:
            for object_id, obj in self._objects.items():
                if object_id not in seen:
                    obj.visible = False

    def set_held(self, object_id: str | None) -> None:
        if object_id is not None and not self.exists(object_id):
            raise KeyError(f"Unknown object '{object_id}'")
        self.held_object_id = object_id
