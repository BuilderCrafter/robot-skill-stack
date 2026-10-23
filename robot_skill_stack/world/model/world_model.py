from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from robot_skill_stack.world.model.context import AssociationHint, ObservationContext
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.common.types import GripperState
from robot_skill_stack.world.model.provider import ObservationHealth


class WorldModel:
    def __init__(self):
        self._objects: dict[str, WorldObject] = {}
        self._association_hints: dict[str, AssociationHint] = {}
        self._context_revision = 0
        self._held_object_id: str | None = None
        self._held_object_offset_in_ee: np.ndarray | None = None
        self._held_object_uncertain = False
        self.source_health: ObservationHealth | None = None

    @property
    def context_revision(self):
        """Owner-context association/control epoch, including ABA transitions."""
        return self._context_revision

    @property
    def held_object_id(self):
        return self._held_object_id

    @held_object_id.setter
    def held_object_id(self, value):
        if value != self._held_object_id:
            self._held_object_id = value
            self._context_revision += 1

    @property
    def held_object_offset_in_ee(self):
        return self._held_object_offset_in_ee

    @held_object_offset_in_ee.setter
    def held_object_offset_in_ee(self, value):
        old = self._held_object_offset_in_ee
        if (old is None and value is None) or (old is not None and value is not None and
                                               np.array_equal(old, value)):
            return
        self._held_object_offset_in_ee = (None if value is None else
            np.frombuffer(np.asarray(value, dtype=float).tobytes(), dtype=float))
        self._context_revision += 1

    @property
    def held_object_uncertain(self):
        return self._held_object_uncertain

    @held_object_uncertain.setter
    def held_object_uncertain(self, value):
        value = bool(value)
        if value != self._held_object_uncertain:
            self._held_object_uncertain = value
            self._context_revision += 1

    def register(self, obj: WorldObject) -> None:
        self._objects[obj.object_id] = obj
        self._context_revision += 1

    def _protected(self, now=None):
        return {self.held_object_id, *(hint.object_id for hint in self.association_hints(now))}

    def remove(self, object_id: str) -> bool:
        if object_id in self._protected() or object_id not in self._objects:
            return False
        del self._objects[object_id]
        self._context_revision += 1
        self.clear_association_hint(object_id)
        return True

    @property
    def observations_healthy(self):
        # Standalone manually populated models have no connected source.
        return self.source_health is None or self.source_health.status == 'fresh'

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
        self._context_revision += 1
        return hint

    def clear_association_hint(self, object_id: str) -> None:
        if self._association_hints.pop(object_id, None) is not None:
            self._context_revision += 1

    def association_hints(self, now=None) -> tuple[AssociationHint, ...]:
        now = time.monotonic() if now is None else float(now)
        return tuple(h for h in self._association_hints.values() if h.expires_at > now)

    def expire_hints(self, now):
        active = {
            key: hint for key, hint in self._association_hints.items() if hint.expires_at > now
        }
        if len(active) != len(self._association_hints):
            self._association_hints = active
            self._context_revision += 1

    def observation_context(self) -> ObservationContext:
        return ObservationContext(
            held_object_id=self.held_object_id,
            association_hints=self.association_hints(),
            revision=self.context_revision,
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
                or obs.geometry is not None
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

    @property
    def gripper_state(self) -> GripperState:
        if self.held_object_id is None:
            return GripperState.FREE
        return GripperState.UNCERTAIN if self.held_object_uncertain else GripperState.HELD

    def set_held(self, object_id: str | None, *, object_offset_in_ee=None, uncertain=False) -> None:
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
        self.held_object_uncertain = object_id is not None and bool(uncertain)

    def clear_held_state(self) -> str | None:
        """Clear software attachment only; the caller must ensure the robot is idle."""
        previous = self.held_object_id
        self.set_held(None)
        if previous is not None:
            self.clear_association_hint(previous)
        return previous
