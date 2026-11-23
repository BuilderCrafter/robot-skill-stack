"""Source-specific construction and prim requirements; no tracking/lifetime policy."""
import omni.kit.app

from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.integrations.isaac.world.ground_truth_provider import IsaacGroundTruthProvider
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.world.model.provider import WorldObservationProvider


def required_source_prims(config):
    """List the source prims from a validated scene configuration."""
    if config.world.provider == 'perception':
        return [config.perception.camera_prim_path]
    if config.world.provider == 'ground_truth':
        return [config.ground_truth.objects_root,
                *(obj.prim_path for obj in config.ground_truth.objects.values())]
    if config.world.provider == 'multi_view':
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
        except BaseException as exc:
            try:
                rig.close()
            except Exception as cleanup:
                exc.add_note(f'Camera cleanup also failed: {cleanup}')
            raise
    camera = IsaacRgbdCamera(config.perception.camera_prim_path,
                             resolution=config.perception.resolution)
    try:
        camera.initialize(semantic_segmentation=False)
        app = omni.kit.app.get_app()
        for _ in range(60):
            await app.next_update_async()
        return build_perception_provider(camera, config)
    except BaseException as exc:
        try:
            camera.close()
        except Exception as cleanup:
            exc.add_note(f'Camera cleanup also failed: {cleanup}')
        raise
