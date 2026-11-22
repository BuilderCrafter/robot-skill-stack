"""Select one prepared RGB-D publication for a fixed-time planning request."""
from .runtime import MappingReadView


class FrozenRgbdSourceError(RuntimeError):
    """Refuse a planning source without substituting another world."""
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def capture_frozen_rgbd(source, *, max_obstacles: int = 64) -> MappingReadView:
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
    if view.generation != source.generation:
        raise FrozenRgbdSourceError('unusable', 'Mapping publication belongs to a different generation')
    if (not packet.boxes or len(packet.boxes) > max_obstacles or
            any(box.expires_at <= now for box in packet.boxes)):
        raise FrozenRgbdSourceError('unusable', 'Selected cuboids are empty, expired or over budget')
    return view
