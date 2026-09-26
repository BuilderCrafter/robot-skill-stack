import numpy as np
import omni.usd
from isaacsim.sensors.camera import Camera

class IsaacRgbdCamera:
    def __init__(self,prim_path="/World/PerceptionCamera",name="perception_camera",resolution=(640,480)):
        prim=omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid(): raise ValueError(f"Camera prim not found: {prim_path}")
        if prim.GetTypeName()!="Camera": raise ValueError(f"Prim is not a Camera: {prim_path}")
        self.camera=Camera(prim_path=prim_path,name=name,resolution=resolution)
    def initialize(self,semantic_segmentation=False):
        self.camera.initialize(attach_rgb_annotator=False); self.camera.add_rgb_to_frame(); self.camera.add_distance_to_image_plane_to_frame()
        if semantic_segmentation: self.camera.add_semantic_segmentation_to_frame(init_params={"colorize":False})
    def get_rgb(self):
        d=self.camera.get_rgb(); return None if d is None else np.asarray(d)
    def get_depth(self):
        d=self.camera.get_depth(); return None if d is None else np.asarray(d)
    def get_intrinsics(self): return np.asarray(self.camera.get_intrinsics_matrix())
    def get_world_pose(self):
        p,q=self.camera.get_world_pose(camera_axes="world"); return np.asarray(p),np.asarray(q)
    def image_points_to_world(self,pixels,depth): return np.asarray(self.camera.get_world_points_from_image_coords(np.asarray(pixels,dtype=np.float32),np.asarray(depth,dtype=np.float32)))
    def get_world_from_camera_transform(self): return np.linalg.inv(np.asarray(self.camera.get_view_matrix_ros()))
