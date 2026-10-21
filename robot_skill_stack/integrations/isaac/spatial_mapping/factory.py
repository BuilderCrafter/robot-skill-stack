"""Explicit mapping construction, with no runtime, planner or camera initialization."""
import time

from robot_skill_stack.world.spatial_mapping.occupancy import OccupancyMap


class MappingSession:
    def __init__(self, occupancy_map, source, *, frame_source=None, own_frame_source=False,
                 update_hz=None, clock=time.monotonic, source_kind='rgbd', stage=None):
        self.map, self.source, self.frame_source = occupancy_map, source, frame_source
        self.own_frame_source, self.update_hz, self.clock = own_frame_source, update_hz, clock
        self.last_attempt, self.closed = None, False
        self.source_kind, self.stage = source_kind, stage

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
            resources = [self.source]
            if self.own_frame_source and self.frame_source is not None and self.frame_source is not self.source:
                resources.append(self.frame_source)
            for resource in resources:
                close = getattr(resource, 'close', None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        errors.append(exc)
            if errors:
                for error in errors[1:]:
                    errors[0].add_note(f'Additional mapping cleanup failure: {error}')
                raise errors[0]


def create_spatial_mapping(settings, *, stage=None, frame_source=None, own_frame_source=False,
                           clock=time.monotonic):
    """Disabled returns None. RGB-D borrows snapshots; it never initializes cameras.

    Pass an existing rig/tap, or feed update(frames) explicitly. Camera paths are
    expected frame IDs, independent of [world].provider and semantic bounds.
    """
    if not settings.enabled:
        return None
    try:
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
        session = MappingSession(occupancy_map, source, frame_source=frame_source,
            own_frame_source=own_frame_source, update_hz=settings.update_hz, clock=clock,
            source_kind=settings.source, stage=stage)
        return session
    except BaseException as exc:
        if own_frame_source and frame_source is not None:
            try:
                frame_source.close()
            except Exception as cleanup:
                exc.add_note(f'Mapping frame source cleanup also failed: {cleanup}')
        raise
