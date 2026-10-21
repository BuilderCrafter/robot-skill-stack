"""Borrow existing camera snapshots without initializing another render pipeline."""
import time

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame


class SharedFrames:
    """Tap an existing rig's captures; close restores it and never closes the rig.

    Semantic perception continues to own acquisition. Mapping reads the same
    immutable snapshots and original timestamps; its token checks reject reuse.
    Install only once per rig, and close the tap before replacing its owner.
    """
    def __init__(self, rig):
        if getattr(rig, '_spatial_mapping_frame_tap', None) is not None:
            raise RuntimeError('This rig already has a spatial-mapping frame tap')
        self.rig, self.original = rig, rig.capture
        self.latest, self.error, self.closed = None, None, False

        def capture():
            try:
                frames = self.original()
            except Exception as exc:
                self.latest, self.error = None, exc
                raise
            if frames is not None:
                self.latest, self.error = frames, None
            return frames
        self.hook = capture
        rig.capture = capture
        rig._spatial_mapping_frame_tap = self

    def capture(self):
        if self.closed:
            raise RuntimeError('Shared frame tap is closed')
        if self.error is not None:
            raise RuntimeError(f'Shared acquisition failed: {self.error}') from self.error
        return self.latest

    def close(self):
        if not self.closed:
            self.closed = True
            if self.rig.capture is self.hook:
                self.rig.capture = self.original
            if getattr(self.rig, '_spatial_mapping_frame_tap', None) is self:
                del self.rig._spatial_mapping_frame_tap
            self.latest = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CameraSnapshots:
    """Read one or more already-initialized Isaac camera resources, borrowing them.

    Useful for V1's single camera. For a multi-view rig prefer SharedFrames,
    which also avoids a second buffer read. This class creates no SDK cameras.
    """
    def __init__(self, cameras, *, clock=time.monotonic):
        if not cameras or len({id(camera) for camera in cameras.values()}) != len(cameras):
            raise ValueError('Supply distinct initialized cameras')
        self.cameras, self.clock, self.closed = dict(cameras), clock, False

    def capture(self):
        from robot_skill_stack.integrations.isaac.world.multiview import IsaacMultiViewCamera
        if self.closed:
            raise RuntimeError('Camera snapshot reader is closed')
        # Reuse the existing adapter's Isaac 5.1 ReferenceTime conversion even
        # for a borrowed V1 camera; it only reads the existing SDK camera object.
        token = lambda camera: IsaacMultiViewCamera.get_frame_token(camera)
        tokens = {key: token(camera) for key, camera in self.cameras.items()}
        acquired = self.clock()
        frames = []
        for key, camera in self.cameras.items():
            depth = camera.get_depth()
            if tokens[key] is None or depth is None:
                raise RuntimeError(f'Missing mapping camera frame: {key}')
            frames.append(DepthFrame(key, tokens[key], np.array(depth, dtype=float, copy=True),
                np.array(camera.get_intrinsics(), dtype=float, copy=True),
                np.array(camera.get_world_from_camera_transform(), dtype=float, copy=True), acquired))
        if any(token(camera) != tokens[key] for key, camera in self.cameras.items()):
            raise RuntimeError('Camera changed during mapping snapshot acquisition')
        return tuple(frames)

    def close(self):
        self.closed = True  # Cameras remain owned by their existing source.
