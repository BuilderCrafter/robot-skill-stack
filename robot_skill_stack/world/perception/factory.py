from __future__ import annotations

from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.provider import PerceptionStateProvider
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator
from robot_skill_stack.world.perception.tracker import ObjectTracker


def build_perception_provider(camera, config, *, robot_source=None):
    if config.perception.type == "yolo_v2":
        from robot_skill_stack.world.perception.v2.client import VisionClient
        from robot_skill_stack.world.perception.v2.pipeline import FrameProcessor
        from robot_skill_stack.world.perception.v2.provider import YoloPerceptionProvider
        p = config.perception
        return YoloPerceptionProvider(camera, FrameProcessor(p.discovery, p.primitives, p.v2),
                                      build_tracker(config), VisionClient(p.v2), robot_source=robot_source)
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
        component_neighbor_distance=d.component_neighbor_distance,
    )
    tracker = build_tracker(config)
    pr = config.perception.primitives
    return PerceptionStateProvider(
        camera, localizer, discoverer, tracker,
        primitive_estimator=PrimitiveEstimator(pr.classification_threshold, pr.ambiguity_margin,
                                              pr.top_band, fit_tolerance=pr.fit_tolerance),
    )


def build_tracker(config):
    t, s = config.perception.tracking, config.perception.semantics
    return ObjectTracker(
        max_distance=t.max_distance,
        max_size_ratio=t.max_size_ratio,
        max_misses=t.max_misses,
        occlusion_distance=t.occlusion_distance,
        semantic_window=s.window_size,
        semantic_min_samples=s.min_samples,
        semantic_threshold=s.assignment_threshold,
        min_confirm_hits=t.min_confirm_hits,
        stale_track_ttl=config.world.stale_object_ttl,
    )
