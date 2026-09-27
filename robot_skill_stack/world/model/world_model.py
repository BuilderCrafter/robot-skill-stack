from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from robot_skill_stack.world.model.context import AssociationHint, ObservationContext
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.observations import ObjectObservation


class WorldModel:
    def __init__(self, *, stale_object_ttl=15.0):
        self._objects: dict[str, WorldObject] = {}
        self._association_hints: dict[str, AssociationHint] = {}
        self.held_object_id: str | None = None
        self.held_object_offset_in_ee: np.ndarray | None = None
        self.stale_object_ttl = float(stale_object_ttl)
        if not np.isfinite(self.stale_object_ttl) or self.stale_object_ttl <= 0:
            raise ValueError("stale_object_ttl must be finite and > 0")
        self._registered_at: dict[str, float] = {}

    def register(self, obj: WorldObject) -> None:
        self._objects[obj.object_id] = obj
        self._registered_at.setdefault(obj.object_id, time.monotonic())

    def _protected(self, now=None):
        return {self.held_object_id, *(hint.object_id for hint in self.association_hints(now))}

    def _discard(self, object_id, forget=None):
        if forget is not None:
            forget(object_id)
        self._objects.pop(object_id, None)
        self._registered_at.pop(object_id, None)
        self._association_hints.pop(object_id, None)

    def remove(self, object_id: str) -> bool:
        if object_id in self._protected() or object_id not in self._objects:
            return False
        self._discard(object_id)
        return True

    def clear_lost(self, forget=None) -> tuple[str, ...]:
        protected = self._protected()
        removed = []
        for obj in list(self._objects.values()):
            if obj.visible or obj.object_id in protected or obj.source == "ground_truth":
                continue
            self._discard(obj.object_id, forget)
            removed.append(obj.object_id)
        return tuple(removed)

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

    def association_hints(self, now=None) -> tuple[AssociationHint, ...]:
        now = time.monotonic() if now is None else float(now)
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
                    geometry=obs.geometry,
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
            if obs.geometry is not None:
                obj.geometry = obs.geometry
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

    def expire_stale(self, now=None, forget=None) -> tuple[str, ...]:
        now = time.monotonic() if now is None else float(now)
        protected = self._protected(now)
        removed = []
        for obj in list(self._objects.values()):
            if obj.visible or obj.object_id in protected or obj.source == "ground_truth":
                continue
            last_seen = obj.last_seen
            if last_seen is None:
                last_seen = self._registered_at[obj.object_id]
            if now-last_seen >= self.stale_object_ttl:
                self._discard(obj.object_id, forget)
                removed.append(obj.object_id)
        return tuple(removed)

    def set_held(self, object_id: str | None, *, object_offset_in_ee=None) -> None:
        if object_id is not None and not self.exists(object_id):
            raise KeyError(f"Unknown object '{object_id}'")
        offset = None
        if object_id is not None and object_offset_in_ee is not None:
            offset = np.asarray(object_offset_in_ee, dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("object_offset_in_ee must be a finite XYZ vector")
            offset = offset.copy()
        self.held_object_id = object_id
        self.held_object_offset_in_ee = offset
