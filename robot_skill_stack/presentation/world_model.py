from __future__ import annotations

from dataclasses import dataclass

from robot_skill_stack.world.model.world_model import WorldModel


@dataclass(frozen=True)
class WorldObjectView:
    object_id: str
    class_name: str | None
    visible: bool
    position: tuple[float, float, float] | None
    size: tuple[float, float, float] | None
    confidence: float | None
    source: str | None
    held: bool

    def signature(self):
        def rounded(values):
            return None if values is None else tuple(round(v, 4) for v in values)

        return (
            self.object_id,
            self.class_name,
            self.visible,
            rounded(self.position),
            rounded(self.size),
            None if self.confidence is None else round(self.confidence, 3),
            self.held,
        )


class WorldModelViewModel:
    """Read-only, simulator-independent presentation model for WorldModel state."""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.selected_id: str | None = None

    @staticmethod
    def _tuple3(value):
        if value is None:
            return None
        return tuple(float(x) for x in value)

    def rows(self) -> tuple[WorldObjectView, ...]:
        rows = []
        for obj in sorted(self.world_model.objects(), key=lambda o: o.object_id):
            rows.append(
                WorldObjectView(
                    object_id=obj.object_id,
                    class_name=obj.class_name,
                    visible=bool(obj.visible),
                    position=(
                        None
                        if obj.pose is None
                        else self._tuple3(obj.pose.position)
                    ),
                    size=self._tuple3(obj.size),
                    confidence=obj.confidence,
                    source=obj.source,
                    held=self.world_model.held_object_id == obj.object_id,
                )
            )
        return tuple(rows)

    def ensure_selection(self) -> str | None:
        if self.selected_id and self.world_model.exists(self.selected_id):
            return self.selected_id

        rows = self.rows()
        preferred = next((row for row in rows if row.visible), None)
        chosen = preferred or (rows[0] if rows else None)
        self.selected_id = None if chosen is None else chosen.object_id
        return self.selected_id

    def select(self, object_id: str) -> None:
        if not self.world_model.exists(object_id):
            raise KeyError(f"Unknown object '{object_id}'")
        self.selected_id = object_id

    def selected_object(self):
        self.ensure_selection()
        return (
            None
            if self.selected_id is None
            else self.world_model.get(self.selected_id)
        )

    def signature(self):
        self.ensure_selection()
        return (
            self.selected_id,
            tuple(row.signature() for row in self.rows()),
        )
