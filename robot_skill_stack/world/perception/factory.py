from __future__ import annotations

from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.provider import PerceptionStateProvider
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.tracker import ObjectTracker


def build_perception_provider(camera, config):
    localizer = RgbdLocalizer(camera.get_intrinsics())
    scene = config.scene
    discoverer = DepthObjectDiscoverer(
        localizer, support_plane_z=scene.support_plane_z,
        workspace_min=scene.workspace_min, workspace_max=scene.workspace_max,
    )
    return PerceptionStateProvider(camera, localizer, discoverer, ObjectTracker(),
                                   primitive_estimator=PrimitiveEstimator())
