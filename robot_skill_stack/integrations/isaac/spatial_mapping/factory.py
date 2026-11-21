"""Explicit mapping construction, with no runtime, planner or camera initialization."""
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyMap


def _close_resources(resources: Sequence[object | None]) -> list[Exception]:
    """Close distinct owned resources once, collecting rather than hiding errors."""
    errors, closed = [], set()
    for resource in resources:
        if resource is None or id(resource) in closed:
            continue
        closed.add(id(resource))
        close = getattr(resource, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as exc:
                errors.append(exc)
    return errors


class MappingSession:
    """Synchronous adapter for RGB-D or ground-truth occupancy updates.

    ``source`` is the selected mapper, not a camera or semantic provider.
    The existing camera source is borrowed unless ``own_frame_source`` is set.
    """
    def __init__(self, occupancy_map: OccupancyMap, source, *, frame_source=None,
                 own_frame_source: bool = False, update_hz: float | None = None,
                 clock: Callable[[], float] = time.monotonic, source_kind: str = 'rgbd',
                 stage=None, robot_filter=None) -> None:
        self.map, self.source, self.frame_source = occupancy_map, source, frame_source
        self.own_frame_source, self.update_hz, self.clock = own_frame_source, update_hz, clock
        self.last_attempt, self.closed = None, False
        self.source_kind, self.stage = source_kind, stage
        self.robot_filter = robot_filter

    def _combined_masks(self, frames: Sequence,
                        pixels_by_camera: Mapping[str, tuple[np.ndarray, np.ndarray]],
                        exclusion_masks: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        """Validate and union explicit exclusions with per-camera robot masks."""
        frame_shapes = {frame.camera_id: frame.depth.shape for frame in frames}

        def validate_masks(masks: Mapping[str, np.ndarray], label: str,
                           *, complete: bool = False) -> dict[str, np.ndarray]:
            if not hasattr(masks, 'items'):
                raise ValueError(f'{label} must be a camera-to-mask mapping')
            masks = dict(masks)
            if set(masks)-set(frame_shapes) or (complete and set(masks) != set(frame_shapes)):
                raise ValueError(f'{label} camera IDs do not match the depth frames')
            for key, value in masks.items():
                if np.asarray(value).dtype != np.bool_ or np.shape(value) != frame_shapes[key]:
                    raise ValueError(f'{label} must match boolean depth image: {key}')
            return masks

        manual_masks = validate_masks({} if exclusion_masks is None else exclusion_masks, 'Exclusion masks')
        robot_masks = validate_masks(self.robot_filter.masks(frames, pixels_by_camera=pixels_by_camera),
                             'Automatic robot masks', complete=True)
        return {key: np.logical_or(robot_masks[key], manual_masks.get(key, False)) for key in frame_shapes}

    def update(self, frames: Sequence | None = None, *,
               exclusion_masks: Mapping[str, np.ndarray] | None = None,
               now: float | None = None) -> bool:
        """Run at most one due update using supplied or borrowed camera evidence."""
        if self.closed:
            raise RuntimeError('Mapping session is closed')
        explicit_now = now is not None
        now = self.clock() if now is None else now
        self.map.expire(now=now)
        if self.update_hz is not None and self.last_attempt is not None:
            if now-self.last_attempt < 1/self.update_hz:
                return False
        self.last_attempt = now
        try:
            if self.source_kind == 'rgbd' and self.frame_source is not None and frames is None:
                frames = self.frame_source.capture()
                if not explicit_now:
                    now = self.clock()
                if frames is None:
                    self.map.note_no_frame(now=now)
                    return False
            if self.source_kind == 'rgbd':
                if frames is None:
                    raise RuntimeError('RGB-D mapping requires supplied DepthFrames or a shared frame source')
                if self.robot_filter is not None:
                    return self.source.update(frames, exclusion_masks=exclusion_masks,
                        now=now if explicit_now else None,
                        mask_provider=lambda current, pixels: self._combined_masks(
                            current, pixels, exclusion_masks))
                return self.source.update(frames, exclusion_masks=exclusion_masks,
                                          now=now if explicit_now else None)
            return self.source.update(stage=self.stage, now=now)
        except Exception as exc:
            self.map.note_failure(exc, now=now)
            raise

    def close(self) -> None:
        """Release owned mapper/filter resources without closing borrowed cameras."""
        if not self.closed:
            self.closed = True
            errors = []
            try:
                self.map.note_no_frame(reason='closed', now=self.clock())
            except Exception as exc:
                errors.append(exc)
            resources = [self.source, self.robot_filter]
            if self.own_frame_source:
                resources.append(self.frame_source)
            errors.extend(_close_resources(resources))
            if errors:
                for error in errors[1:]:
                    errors[0].add_note(f'Additional mapping cleanup failure: {error}')
                raise errors[0]


def create_spatial_mapping(settings, *, stage=None, frame_source=None,
                           own_frame_source: bool = False,
                           clock: Callable[[], float] = time.monotonic,
                           robot_prim_path: str | None = None, robot_filter=None
                           ) -> MappingSession | None:
    """Disabled returns None. RGB-D borrows snapshots; it never initializes cameras.

    Pass an existing rig/tap, or feed update(frames) explicitly. Camera paths are
    expected frame IDs, independent of [world].provider and semantic bounds.
    Automatic RGB-D filtering requires ``robot_prim_path=config.robot.prim_path``.
    A supplied filter replaces geometry acquisition and transfers ownership only
    when automatic RGB-D filtering is enabled; cameras remain borrowed.
    """
    if not settings.enabled:
        return None
    mapping_config = settings
    selected_mapper, robot_self_filter = None, None
    try:
        filtering = mapping_config.source == 'rgbd' and mapping_config.automatic_robot_filter
        if filtering and (not isinstance(robot_prim_path, str) or not robot_prim_path.startswith('/') or
                any(not part.isidentifier() for part in robot_prim_path[1:].split('/'))):
            raise ValueError('Automatic robot filtering requires the configured absolute robot_prim_path')
        occupancy_map = OccupancyMap(mapping_config.grid_spec(), clock=clock,
            stale_after=mapping_config.stale_after, evidence_ttl=mapping_config.evidence_ttl)
        if mapping_config.source == 'ground_truth':
            from .ground_truth import IsaacGroundTruthMapper
            selected_mapper = IsaacGroundTruthMapper(occupancy_map, excluded_paths=mapping_config.excluded_prim_paths,
                                            max_candidate_voxels=mapping_config.max_candidate_voxels)
        else:
            from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
            selected_mapper = RgbdMapper(occupancy_map, stride=mapping_config.stride, max_rays=mapping_config.max_rays,
                max_voxel_visits=mapping_config.max_voxel_visits,
                max_depth=mapping_config.max_depth, max_skew=mapping_config.max_skew,
                max_frame_age=mapping_config.max_frame_age,
                expected_camera_ids=mapping_config.camera_prim_paths or None, clock=clock)
            if filtering:
                robot_self_filter = robot_filter
                if robot_self_filter is None:
                    from .robot_filter import IsaacRobotSelfFilter
                    robot_self_filter = IsaacRobotSelfFilter(stage, robot_prim_path,
                        stride=mapping_config.stride, depth_tolerance=mapping_config.robot_depth_tolerance,
                        pose_tolerance=mapping_config.robot_pose_tolerance)
        session = MappingSession(occupancy_map, selected_mapper, frame_source=frame_source,
            own_frame_source=own_frame_source, update_hz=mapping_config.update_hz, clock=clock,
            source_kind=mapping_config.source, stage=stage, robot_filter=robot_self_filter)
        return session
    except BaseException as exc:
        resources = [selected_mapper, robot_self_filter]
        if own_frame_source:
            resources.append(frame_source)
        for cleanup in _close_resources(resources):
            exc.add_note(f'Mapping resource cleanup also failed: {cleanup}')
        raise
