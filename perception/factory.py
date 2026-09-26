from __future__ import annotations

from perception.discovery import DepthObjectDiscoverer
from perception.rgbd_localizer import RgbdLocalizer
from perception.semantics import CubeGeometryClassifier
from perception.state_provider import PerceptionStateProvider
from perception.tracker import ObjectTracker


def build_perception_provider(camera, config):
    localizer = RgbdLocalizer(camera.get_intrinsics())
    d = config.perception.discovery
    t = config.perception.tracking
    s = config.perception.semantics

    discoverer = DepthObjectDiscoverer(
        localizer,
        support_plane_z=d.support_plane_z,
        workspace_min=d.workspace_min,
        workspace_max=d.workspace_max,
        min_object_height=d.min_object_height,
        max_object_height=d.max_object_height,
        min_component_pixels=d.min_component_pixels,
        max_component_pixels=d.max_component_pixels,
        candidate_max_extent=d.candidate_max_extent,
        spawn_min_extent=d.spawn_min_extent,
        spawn_max_extent=d.spawn_max_extent,
        spawn_compactness_ratio=d.spawn_compactness_ratio,
        support_contact_tolerance=d.support_contact_tolerance,
    )
    tracker = ObjectTracker(
        max_distance=t.max_distance,
        max_size_ratio=t.max_size_ratio,
        max_misses=t.max_misses,
        occlusion_distance=t.occlusion_distance,
        semantic_window=s.window_size,
        semantic_min_samples=s.min_samples,
        semantic_threshold=s.assignment_threshold,
    )
    classifier = CubeGeometryClassifier(
        ratio_max=s.cube_ratio_max,
        min_size=s.cube_min_size,
        max_size=s.cube_max_size,
    )
    return PerceptionStateProvider(
        camera,
        localizer,
        discoverer,
        tracker,
        classifier,
    )
