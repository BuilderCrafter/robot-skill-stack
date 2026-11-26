"""Resource construction only; geometry and source policy live outside Isaac."""
from __future__ import annotations

from collections.abc import Sequence

from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.world.common.camera import FrameSynchronizer


class IsaacMultiViewCamera(IsaacRgbdCamera):
    """Expose numeric exposure clocks for synchronization instead of V1's opaque strings."""

    def get_frame_token(self) -> tuple[float, float] | None:
        frame = self.camera.get_current_frame()
        reference, stamp = frame.get('rendering_frame'), frame.get('rendering_time')
        if reference is None or stamp is None:
            return None
        # Isaac 5.1 publishes a ReferenceTime rational, not an integer frame ID.
        if isinstance(reference, dict):
            denominator = reference.get('referenceTimeDenominator', 0)
            if not denominator:
                return None
            reference = reference['referenceTimeNumerator']/denominator
        return reference, stamp


def create_cameras(paths: Sequence[str], resolution: tuple[int, int] = (640, 480),
                   *, max_skew: float = .025) -> FrameSynchronizer:
    """Initialize an ordered camera rig that owns and closes its cameras."""
    if len(paths) < 1 or len(set(paths)) != len(paths):
        raise ValueError('Multi-view requires at least one distinct camera path')
    cameras: dict[str, IsaacMultiViewCamera] = {}
    try:
        for index, path in enumerate(paths):
            camera = IsaacMultiViewCamera(path, name=f'multiview_camera_{index}', resolution=resolution)
            cameras[path] = camera
            camera.initialize(semantic_segmentation=False)
        return FrameSynchronizer(cameras, max_skew=max_skew)
    except BaseException as exc:
        for camera in reversed(tuple(cameras.values())):
            try:
                camera.close()
            except Exception as cleanup:
                exc.add_note(f'Camera cleanup also failed: {cleanup}')
        raise
