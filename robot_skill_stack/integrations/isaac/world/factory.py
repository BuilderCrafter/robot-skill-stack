"""Source-specific construction and prim requirements; no tracking/lifetime policy."""
import omni.kit.app

from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.integrations.isaac.world.ground_truth_provider import IsaacGroundTruthProvider
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.model.provider import WorldObservationProvider


def required_source_prims(config):
    if config.world.provider == 'perception':
        if not config.perception.camera_prim_path:
            raise ValueError('Perception source requires camera_prim_path')
        return [config.perception.camera_prim_path]
    if config.world.provider == 'ground_truth':
        if not config.ground_truth.objects_root:
            raise ValueError('Ground truth source requires objects_root')
        return [config.ground_truth.objects_root,
                *(obj.prim_path for obj in config.ground_truth.objects.values())]
    if config.world.provider == 'multi_view':
        if not config.multi_view.camera_prim_paths:
            raise ValueError('Multi-view requires camera_prim_paths')
        return list(config.multi_view.camera_prim_paths)
    raise ValueError(f'Unsupported observation source: {config.world.provider}')


async def create_source(config) -> WorldObservationProvider:
    required_source_prims(config)
    if config.world.provider == 'ground_truth':
        return IsaacGroundTruthProvider(config.ground_truth.objects_root, config.ground_truth.objects)
    if config.world.provider == 'multi_view':
        from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
        from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
        settings = config.multi_view
        rig = create_cameras(settings.camera_prim_paths, settings.resolution, max_skew=settings.max_skew)
        try:
            app = omni.kit.app.get_app()
            for _ in range(60):
                await app.next_update_async()
            return MultiViewProvider(rig, config.scene, voxel_size=settings.voxel_size, stride=settings.stride)
        except BaseException:
            rig.close()
            raise
    camera = IsaacRgbdCamera(config.perception.camera_prim_path,
                             resolution=config.perception.resolution)
    try:
        camera.initialize(semantic_segmentation=False)
        app = omni.kit.app.get_app()
        for _ in range(60):
            await app.next_update_async()
        return build_perception_provider(camera, config)
    except BaseException:
        camera.close()
        raise
