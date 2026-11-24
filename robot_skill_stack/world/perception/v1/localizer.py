import numpy as np


class RgbdLocalizer:
    """Convert depth-image pixels into 3D points using camera calibration."""
    def __init__(self, intrinsics):
        self.K = np.asarray(intrinsics, dtype=float)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def pixels_to_camera(self, pixels, depth):
        """Back-project pixels and depth values into 3D camera coordinates."""
        pixels = np.asarray(pixels, dtype=float)
        depth = np.asarray(depth, dtype=float)
        u, v = pixels[:, 0], pixels[:, 1]
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.column_stack((x, y, z))

    def camera_to_world(self, points, world_from_camera):
        """Transform 3D camera points into the world coordinate frame."""
        points = np.asarray(points, dtype=float)
        T = np.asarray(world_from_camera, dtype=float)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (homogeneous @ T.T)[:, :3]

    def pixels_to_world(self, pixels, depth, world_from_camera):
        """Back-project pixels and transform the resulting points into world space."""
        return self.camera_to_world(
            self.pixels_to_camera(pixels, depth),
            world_from_camera,
        )
