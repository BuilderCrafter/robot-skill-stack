import numpy as np


class RgbdLocalizer:
    def __init__(self, intrinsics):
        self.K = np.asarray(intrinsics, dtype=float)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def pixels_to_camera(self, pixels, depth):
        pixels = np.asarray(pixels, dtype=float)
        depth = np.asarray(depth, dtype=float)

        u = pixels[:, 0]
        v = pixels[:, 1]
        z = depth

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.column_stack((x, y, z))

    def camera_to_world(self, points, world_from_camera):
        points = np.asarray(points, dtype=float)
        T = np.asarray(world_from_camera, dtype=float)

        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (homogeneous @ T.T)[:, :3]

    def pixels_to_world(self, pixels, depth, world_from_camera):
        camera_points = self.pixels_to_camera(pixels, depth)
        return self.camera_to_world(camera_points, world_from_camera)