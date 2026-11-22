"""Select one prepared RGB-D publication for a fixed-time planning request."""
from collections.abc import Mapping
from dataclasses import dataclass

from .runtime import MappingReadView


class FrozenRgbdSourceError(RuntimeError):
    """Refuse a planning source without substituting another world."""
    def __init__(self, code: str, message: str, details: Mapping | None = None) -> None:
        self.code, self.details = code, details or {}
        super().__init__(message)


@dataclass(frozen=True, eq=False)
class FrozenRgbdWorld:
    """Retain the immutable worker view and its original camera records."""
    view: MappingReadView
    source_frames: tuple[Mapping[str, object], ...]

    @property
    def generation(self) -> int:
        return self.view.generation

    @property
    def revision(self) -> int:
        return self.view.revision

    @property
    def timestamp(self) -> float:
        return self.view.timestamp

    @property
    def spec(self):
        return self.view.spec

    @property
    def boxes(self):
        return self.view.obstacles.boxes

    @property
    def age_seconds(self) -> float:
        return self.view.evaluated_at-self.timestamp

    def query_many(self, positions):
        """Query occupancy at the selected view's fixed evaluation time."""
        return self.view.query_many(positions)

    def query(self, position):
        return self.view.query(position)


def capture_frozen_rgbd(source, *, max_obstacles: int = 64) -> FrozenRgbdWorld:
    """Select fresh worker cuboids without acquisition, scanning or extraction."""
    if source is None:
        raise FrozenRgbdSourceError('unavailable', 'Spatial Mapping is unavailable')
    settings = source.settings
    if not settings.enabled or settings.source != 'rgbd' or not settings.automatic_robot_filter:
        raise FrozenRgbdSourceError('unavailable', 'RGB-D mapping with robot filtering is required')
    now = source.clock()
    view = source.snapshot(now=now)
    packet = view.obstacles
    if view.timestamp is None or packet is None:
        raise FrozenRgbdSourceError('unavailable', 'No accepted mapping publication')
    if not 0 <= now-view.timestamp < min(settings.stale_after, settings.max_frame_age):
        raise FrozenRgbdSourceError('stale', 'Original RGB-D observation is stale or future-dated')
    if not view.health['fresh']:
        raise FrozenRgbdSourceError('unavailable', 'RGB-D mapping is not healthy and fresh')
    if (view.generation != source.generation or
            (packet.generation, packet.revision, packet.timestamp) !=
            (view.generation, view.revision, view.timestamp)):
        raise FrozenRgbdSourceError('unusable', 'Obstacles and occupancy identify different publications')
    if (not packet.boxes or len(packet.boxes) > max_obstacles or
            any(box.expires_at <= now for box in packet.boxes)):
        raise FrozenRgbdSourceError('unusable', 'Selected cuboids are empty, expired or over budget')
    return FrozenRgbdWorld(view, source.diagnostics['source_frames'])
