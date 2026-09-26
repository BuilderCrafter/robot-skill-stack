from __future__ import annotations

import numpy as np


class ObjectGeometryService:
    def __init__(self, localizer, tracker):
        self.localizer = localizer
        self.tracker = tracker
        self.latest_frame = None

    def update_frame(self, frame):
        self.latest_frame = frame

    def get_mask(self, object_id: str):
        frame = self.latest_frame
        track = self.tracker.get(object_id)
        if (
            frame is None
            or track is None
            or track.mask is None
            or track.frame_id != frame.frame_id
        ):
            return None
        return track.mask.copy()

    def get_point_cloud(self, object_id: str):
        frame = self.latest_frame
        mask = self.get_mask(object_id)
        if frame is None or mask is None:
            return None

        valid = mask & np.isfinite(frame.depth) & (frame.depth > 0)
        ys, xs = np.nonzero(valid)
        if not xs.size:
            return None

        return self.localizer.pixels_to_world(
            np.column_stack((xs, ys)),
            frame.depth[ys, xs],
            frame.world_from_camera,
        )
