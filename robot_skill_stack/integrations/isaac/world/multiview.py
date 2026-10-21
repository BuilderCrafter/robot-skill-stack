"""Resource construction only; geometry and source policy live outside Isaac."""
from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.world.perception_multiview.acquisition import FrameSynchronizer


class IsaacMultiViewCamera(IsaacRgbdCamera):
    def get_frame_token(self):
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


def create_cameras(paths, resolution=(640, 480), *, max_skew=.025):
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError('Multi-view requires distinct camera paths')
    cameras = {}
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
