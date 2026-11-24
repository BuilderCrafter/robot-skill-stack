from __future__ import annotations

from robot_skill_stack.world.perception.v1.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.v1.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.v1.provider import PerceptionStateProvider
from robot_skill_stack.world.perception.v1.primitive_estimator import PrimitiveEstimator
from robot_skill_stack.world.perception.v1.tracker import ObjectTracker


def build_v1_provider(camera, scene) -> PerceptionStateProvider:
    """Assemble the single-camera discovery, geometry, and tracking pipeline."""
    localizer = RgbdLocalizer(camera.get_intrinsics())
    discoverer = DepthObjectDiscoverer(
        localizer, support_plane_z=scene.support_plane_z,
        workspace_min=scene.workspace_min, workspace_max=scene.workspace_max,
    )
    return PerceptionStateProvider(camera, localizer, discoverer, ObjectTracker(),
                                   primitive_estimator=PrimitiveEstimator())
