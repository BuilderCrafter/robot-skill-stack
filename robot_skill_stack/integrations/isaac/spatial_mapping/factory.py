"""Explicit mapping construction, with no runtime, planner or camera initialization."""
import time

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyMap


def _close_resources(resources):
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
    def __init__(self, occupancy_map, source, *, frame_source=None, own_frame_source=False,
                 update_hz=None, clock=time.monotonic, source_kind='rgbd', stage=None,
                 robot_filter=None):
        self.map, self.source, self.frame_source = occupancy_map, source, frame_source
        self.own_frame_source, self.update_hz, self.clock = own_frame_source, update_hz, clock
        self.last_attempt, self.closed = None, False
        self.source_kind, self.stage = source_kind, stage
        self.robot_filter = robot_filter

    def _combined_masks(self, frames, pixels_by_camera, exclusion_masks):
        shapes = {frame.camera_id: frame.depth.shape for frame in frames}

        def validate(masks, label, *, complete=False):
            if not hasattr(masks, 'items'):
                raise ValueError(f'{label} must be a camera-to-mask mapping')
            masks = dict(masks)
            if set(masks)-set(shapes) or (complete and set(masks) != set(shapes)):
                raise ValueError(f'{label} camera IDs do not match the depth frames')
            for key, value in masks.items():
                if np.asarray(value).dtype != np.bool_ or np.shape(value) != shapes[key]:
                    raise ValueError(f'{label} must match boolean depth image: {key}')
            return masks

        manual = validate({} if exclusion_masks is None else exclusion_masks, 'Exclusion masks')
        automatic = validate(self.robot_filter.masks(frames, pixels_by_camera=pixels_by_camera),
                             'Automatic robot masks', complete=True)
        return {key: np.logical_or(automatic[key], manual.get(key, False)) for key in shapes}

    def update(self, frames=None, *, exclusion_masks=None, now=None):
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

    def close(self):
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


def create_spatial_mapping(settings, *, stage=None, frame_source=None, own_frame_source=False,
                           clock=time.monotonic, robot_prim_path=None, robot_filter=None):
    """Disabled returns None. RGB-D borrows snapshots; it never initializes cameras.

    Pass an existing rig/tap, or feed update(frames) explicitly. Camera paths are
    expected frame IDs, independent of [world].provider and semantic bounds.
    Automatic RGB-D filtering requires ``robot_prim_path=config.robot.prim_path``.
    A supplied filter replaces geometry acquisition and transfers ownership only
    when automatic RGB-D filtering is enabled; cameras remain borrowed.
    """
    if not settings.enabled:
        return None
    source, active_filter = None, None
    try:
        filtering = settings.source == 'rgbd' and settings.automatic_robot_filter
        if filtering and (not isinstance(robot_prim_path, str) or not robot_prim_path.startswith('/') or
                any(not part.isidentifier() for part in robot_prim_path[1:].split('/'))):
            raise ValueError('Automatic robot filtering requires the configured absolute robot_prim_path')
        occupancy_map = OccupancyMap(settings.grid_spec(), clock=clock,
            stale_after=settings.stale_after, evidence_ttl=settings.evidence_ttl)
        if settings.source == 'ground_truth':
            from .ground_truth import IsaacGroundTruthMapper
            source = IsaacGroundTruthMapper(occupancy_map, excluded_paths=settings.excluded_prim_paths,
                                            max_candidate_voxels=settings.max_candidate_voxels)
        else:
            from robot_skill_stack.world.spatial_mapping.rgbd import RgbdMapper
            source = RgbdMapper(occupancy_map, stride=settings.stride, max_rays=settings.max_rays,
                max_voxel_visits=settings.max_voxel_visits,
                max_depth=settings.max_depth, max_skew=settings.max_skew,
                max_frame_age=settings.max_frame_age,
                expected_camera_ids=settings.camera_prim_paths or None, clock=clock)
            if filtering:
                active_filter = robot_filter
                if active_filter is None:
                    from .robot_filter import IsaacRobotSelfFilter
                    active_filter = IsaacRobotSelfFilter(stage, robot_prim_path,
                        stride=settings.stride, depth_tolerance=settings.robot_depth_tolerance,
                        pose_tolerance=settings.robot_pose_tolerance)
        session = MappingSession(occupancy_map, source, frame_source=frame_source,
            own_frame_source=own_frame_source, update_hz=settings.update_hz, clock=clock,
            source_kind=settings.source, stage=stage, robot_filter=active_filter)
        return session
    except BaseException as exc:
        resources = [source, active_filter]
        if own_frame_source:
            resources.append(frame_source)
        for cleanup in _close_resources(resources):
            exc.add_note(f'Mapping resource cleanup also failed: {cleanup}')
        raise
