from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import omni.usd
from isaacsim.sensors.camera import Camera


class IsaacRgbdCamera:
    """Own an Isaac RGB-D sensor and its annotators; reads expose SDK buffers."""

    def __init__(self, prim_path: str = "/World/PerceptionCamera", name: str = "perception_camera",
                 resolution: tuple[int, int] = (640, 480)) -> None:
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Camera prim not found: {prim_path}")
        if prim.GetTypeName() != "Camera":
            raise ValueError(f"Prim is not a Camera: {prim_path}")
        self._closed = False
        self.camera = Camera(prim_path=prim_path, name=name, resolution=resolution)

    def initialize(self, semantic_segmentation: bool = False) -> None:
        """Register RGB and distance-to-image-plane depth using the existing SDK defaults."""
        self.camera.initialize(attach_rgb_annotator=False)
        self.camera.add_rgb_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()
        if semantic_segmentation:
            self.camera.add_semantic_segmentation_to_frame(init_params={"colorize": False})

    def get_rgb(self) -> np.ndarray | None:
        rgb = self.camera.get_rgb()
        return None if rgb is None else np.asarray(rgb)

    def get_depth(self) -> np.ndarray | None:
        depth = self.camera.get_depth()
        return None if depth is None else np.asarray(depth)

    def get_intrinsics(self) -> np.ndarray:
        return np.asarray(self.camera.get_intrinsics_matrix())

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self.camera.get_world_pose(camera_axes="world")
        return np.asarray(position), np.asarray(orientation)

    def image_points_to_world(self, pixels: ArrayLike, depth: ArrayLike) -> np.ndarray:
        return np.asarray(self.camera.get_world_points_from_image_coords(
            np.asarray(pixels, dtype=np.float32), np.asarray(depth, dtype=np.float32)))

    def get_world_from_camera_transform(self) -> np.ndarray:
        """Invert the ROS view matrix to map camera coordinates into world coordinates."""
        return np.linalg.inv(np.asarray(self.camera.get_view_matrix_ros()))

    def get_frame_token(self) -> tuple[str, str] | None:
        """Keep V1's opaque string token, including the SDK's unparsed time value."""
        frame = self.camera.get_current_frame()
        number = frame.get("rendering_frame")
        if number is None:
            return None
        return (str(number), str(frame.get("rendering_time")))

    def reset_clock(self) -> None:
        """Reset Isaac 5.1 sampling time after Stop/Play, retaining resources.

        Camera.post_reset resets elapsed/previous time; its BaseSensor hook is
        a no-op. It does not change camera pose, annotators or render products.
        The runtime separately blocks retained tokens until a new exposure.
        """
        if self._closed:
            raise RuntimeError('RGB-D camera is closed')
        self.camera.post_reset()

    def close(self) -> None:
        """Dispose owned SDK resources in order, continuing after cleanup failures."""
        if self._closed:
            return
        self._closed = True
        # Camera.destroy() in 5.1 leaves separate event/ReferenceTime handles.
        errors: list[Exception] = []

        def attempt(operation: Callable[[], object]) -> None:
            try:
                operation()
            except Exception as exc:
                errors.append(exc)

        attempt(self.camera.pause)
        self.camera._stage_open_callback = None
        self.camera._timer_reset_callback = None
        reference = getattr(self.camera, '_fabric_time_annotator', None)
        if reference is not None:
            attempt(lambda: reference.detach([self.camera.get_render_product_path()]))
            self.camera._fabric_time_annotator = None
        attempt(self.camera.destroy)
        if errors:
            raise RuntimeError('Camera cleanup failed: '+ '; '.join(str(e) for e in errors)) from errors[0]
