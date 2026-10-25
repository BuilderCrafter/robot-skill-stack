"""One source-bound RGB-D planning publication, without SDK or dense extraction.

Capture validates and copies at most 128 already prepared occupied-cell cuboids.
It never scans occupancy, fills UNKNOWN cells, repacks geometry, updates a map,
or acquires another camera frame. Sparse queries retain the original fixed
capture time; later publications and expiry do not mutate the returned world.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np

from .occupancy import GridSpec, _readonly
from .obstacles import ObstaclePacket
from .runtime import MappingReadView


def _freeze(value):
    """Own small source metadata even when an adapter supplied mutable values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key:_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Frozen source metadata must contain only numeric/string records')


class FrozenRgbdSourceError(RuntimeError):
    """Explicit source rejection; callers must not substitute a fallback world."""
    def __init__(self, code, message, details=None):
        self.code = str(code)
        self.details = _freeze({} if details is None else details)
        super().__init__(message)


def _fail(code, message, **details):
    raise FrozenRgbdSourceError(code, message, details)


@dataclass(frozen=True)
class FrozenRgbdBox:
    id: str
    center: tuple
    dimensions: tuple
    expires_at: float
    voxel_count: int
    orientation_wxyz: tuple = (1., 0., 0., 0.)

    def __post_init__(self):
        center, dimensions = tuple(map(float, self.center)), tuple(map(float, self.dimensions))
        orientation = tuple(map(float, self.orientation_wxyz))
        expiry = float(self.expires_at)
        if (not isinstance(self.id, str) or not self.id or len(center) != 3 or len(dimensions) != 3
                or not np.isfinite(center+dimensions).all() or any(value <= 0 for value in dimensions)
                or orientation != (1.,0.,0.,0.) or not math.isfinite(expiry) or expiry < 0
                or type(self.voxel_count) is not int or self.voxel_count < 1):
            raise ValueError('Frozen RGB-D boxes require exact finite axis-aligned cell geometry')
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'dimensions', dimensions)
        object.__setattr__(self, 'orientation_wxyz', orientation)
        object.__setattr__(self, 'expires_at', expiry)


@dataclass(frozen=True, eq=False)
class FrozenRgbdWorld:
    generation: int
    revision: int
    timestamp: float
    evaluated_at: float
    spec: GridSpec
    boxes: tuple
    metadata: object
    view: MappingReadView = field(repr=False)

    def __post_init__(self):
        boxes = tuple(self.boxes)
        if (type(self.generation) is not int or self.generation < 0
                or type(self.revision) is not int or self.revision < 1
                or not isinstance(self.spec, GridSpec) or not isinstance(self.view, MappingReadView)
                or not 1 <= len(boxes) <= 128 or any(not isinstance(box, FrozenRgbdBox) for box in boxes)
                or len({box.id for box in boxes}) != len(boxes)
                or not math.isfinite(self.timestamp) or self.timestamp < 0
                or not math.isfinite(self.evaluated_at) or self.evaluated_at < 0
                or self.timestamp > self.evaluated_at+1e-9
                or self.view.generation != self.generation or self.view.revision != self.revision
                or self.view.timestamp != self.timestamp or self.view.evaluated_at != self.evaluated_at
                or self.view.spec != self.spec):
            raise ValueError('Frozen world must retain one consistent immutable mapping read view')
        object.__setattr__(self, 'boxes', boxes)
        object.__setattr__(self, 'metadata', _freeze(self.metadata))

    @property
    def age_seconds(self):
        return max(0., self.evaluated_at-self.timestamp)

    def query_many(self, positions):
        """Sparse fixed-time occupancy queries; UNKNOWN stays distinct from FREE."""
        return self.view.query_many(positions)

    def query(self, position):
        return self.view.query(position)


def _count(diagnostics, key):
    value = diagnostics.get(key)
    if type(value) is not int or value < 0:
        _fail('invalid_product', 'Prepared obstacle counts are invalid', field=key)
    return value


def _boxes(packet, spec, *, now, maximum):
    """Check bounded exact grid descriptors without touching the dense grid."""
    if len(packet.boxes) > maximum:
        _fail('budget_mismatch', 'Prepared obstacle product exceeds the selected bound',
              selected=len(packet.boxes), maximum=maximum)
    output, bounds, ids = [], [], set()
    lower, upper = np.asarray(spec.bounds_min), np.asarray(spec.bounds_max)
    for box in packet.boxes:
        try:
            pieces = tuple(int(part) for part in box.id.split(':'))
            if len(pieces) != 6 or ':'.join(map(str, pieces)) != box.id:
                raise ValueError('Expected canonical occupied-cell bounds')
            start, stop = np.asarray(pieces[::2]), np.asarray(pieces[1::2])
            if np.any(start < 0) or np.any(stop > spec.shape) or np.any(stop <= start):
                raise ValueError('Cell bounds leave the represented grid')
            minimum = lower+start*spec.voxel_size
            maximum_point = np.minimum(lower+stop*spec.voxel_size, upper)
            center, dimensions = (minimum+maximum_point)/2, maximum_point-minimum
            if (not np.array_equal(center, box.center) or not np.array_equal(dimensions, box.dimensions)
                    or type(box.voxel_count) is not int or box.voxel_count != math.prod(stop-start)):
                raise ValueError('Descriptor differs from exact occupied-cell geometry')
            expiry = float(box.expires_at)
            if not math.isfinite(expiry) or expiry < 0:
                raise ValueError('Invalid cell evidence deadline')
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            _fail('invalid_product', 'Prepared RGB-D cuboid is invalid: '+str(error))
        if box.id in ids:
            _fail('invalid_product', 'Prepared RGB-D cuboids contain a duplicate ID', obstacle_id=box.id)
        if expiry <= now:
            _fail('expired_evidence', 'Selected occupied evidence expired before capture',
                  obstacle_id=box.id, expires_at=expiry, evaluated_at=now)
        ids.add(box.id)
        bounds.append((start, stop, box.id))
        output.append(FrozenRgbdBox(box.id, tuple(center), tuple(dimensions), expiry, box.voxel_count))
    if len(bounds) > 1:
        # This is bounded descriptor validation (at most 128x128 pairs), never
        # an occupancy scan or a second extraction on the owner thread.
        starts = np.asarray([item[0] for item in bounds])
        stops = np.asarray([item[1] for item in bounds])
        overlap = np.all(starts[:,None,:] < stops[None,:,:], axis=2)
        overlap = overlap & overlap.T
        indices = np.argwhere(np.triu(overlap, k=1))
        if len(indices):
            first, second = map(int, indices[0])
            _fail('invalid_product', 'Prepared RGB-D cuboids overlap occupied cells',
                  obstacle_ids=(bounds[first][2], bounds[second][2]))
    return tuple(output)


def _camera_metadata(diagnostics, settings, timestamp, now):
    frames = diagnostics.get('source_frames')
    count = diagnostics.get('accepted_frame_count')
    if not isinstance(frames, (tuple, list)) or type(count) is not int or count < 1 or len(frames) != count:
        _fail('invalid_product', 'Accepted RGB-D publication has no camera-source identity')
    ids, stamps = [], []
    for frame in frames:
        try:
            camera = frame['camera_id']
            stamp = float(frame['acquired_at'])
            token = np.asarray(frame['token'], float)
            shape = tuple(frame['depth_shape'])
            k, t = np.asarray(frame['intrinsics']), np.asarray(frame['world_from_camera'])
            if (not isinstance(camera, str) or not camera or camera in ids
                    or not math.isfinite(stamp) or stamp < 0 or stamp > now+1e-9
                    or token.shape != (2,) or not np.isfinite(token).all() or token[0] <= 0 or token[1] < 0
                    or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape)
                    or k.shape != (3,3) or t.shape != (4,4) or not np.isfinite(k).all() or not np.isfinite(t).all()):
                raise ValueError('Invalid calibrated exposure record')
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            _fail('invalid_product', 'Accepted RGB-D camera metadata is invalid: '+str(error))
        ids.append(camera)
        stamps.append(stamp)
    expected = tuple(settings.camera_prim_paths)
    if expected and set(ids) != set(expected):
        _fail('invalid_product', 'Accepted cameras differ from the configured mapping cameras')
    if min(stamps) != timestamp or set(diagnostics.get('cameras', ())) != set(ids):
        _fail('product_mismatch', 'Camera acquisitions do not identify this mapping publication')
    robot_filter = diagnostics.get('robot_filter') or {}
    if robot_filter.get('state') != 'ready' or set(robot_filter.get('cameras', ())) != set(ids):
        _fail('filter_required', 'Accepted RGB-D publication lacks complete automatic robot filtering')
    return tuple(ids), tuple(frames)


def capture_frozen_rgbd(source, *, max_obstacles=64):
    """Capture one current accepted RGB-D publication for one static request.

    The service's existing clock fixes read time. Any source problem rejects
    the whole capture, including an empty product or expired selected cuboid.
    Counts are worker/extraction-time diagnostics, explicitly not a new dense
    capture-time scan. Later sparse queries continue using this fixed read time.
    """
    if type(max_obstacles) is not int or not 1 <= max_obstacles <= 128:
        _fail('invalid_budget', 'Frozen RGB-D obstacle bound must be between 1 and 128')
    if source is None:
        _fail('unavailable', 'Spatial Mapping service is unavailable')
    try:
        settings = source.settings
        if settings.enabled is not True:
            _fail('disabled', 'Spatial Mapping is disabled')
        if settings.source != 'rgbd':
            _fail('wrong_source', 'Frozen RGB-D global planning requires the RGB-D mapping source')
        if settings.automatic_robot_filter is not True:
            _fail('filter_required', 'Frozen RGB-D planning requires automatic robot self-filtering')
        generation = source.generation
        if type(generation) is not int or generation < 0:
            _fail('generation_changed', 'Spatial Mapping generation is invalid')
        now = float(source.clock())
        if not math.isfinite(now) or now < 0:
            _fail('invalid_clock', 'Mapping capture clock is invalid')
        view = source.snapshot(now=now)
        if not isinstance(view, MappingReadView):
            _fail('invalid_product', 'Expected the existing immutable mapping read view')
        health, packet = view.health, view.obstacles
        stamp = view.timestamp
        if stamp is None:
            _fail('no_publication', 'Spatial Mapping has no accepted publication')
        stamp = float(stamp)
        limit = min(float(settings.stale_after), float(settings.max_frame_age))
        if (not math.isfinite(stamp) or stamp < 0 or stamp > now+1e-9
                or not math.isfinite(limit) or limit <= 0 or now-stamp >= limit):
            _fail('stale', 'RGB-D acquisition is stale, future-dated or invalid',
                  timestamp=stamp, evaluated_at=now, freshness_limit_seconds=limit)
        if health.get('status') != 'ok' or health.get('fresh') is not True:
            _fail('unhealthy', 'Spatial Mapping is not currently healthy and fresh',
                  health=dict(health))
        if (view.generation != generation or source.generation != generation
                or type(view.revision) is not int or view.revision < 1):
            _fail('generation_changed', 'Mapping generation changed before snapshot acceptance')
        if packet is None:
            _fail('no_publication', 'Accepted mapping has no prepared obstacle product')
        if not isinstance(packet, ObstaclePacket):
            _fail('invalid_product', 'Expected the existing immutable worker obstacle product')
        if (packet.generation != generation or packet.revision != view.revision or packet.timestamp != stamp
                or view.evaluated_at != now or view.stale_after != settings.stale_after
                or view.evidence_ttl != settings.evidence_ttl):
            _fail('product_mismatch', 'Prepared obstacles and occupancy publication disagree')
        if view.spec != settings.grid_spec():
            _fail('product_mismatch', 'Prepared world does not match configured mapping bounds')
        extracted_at = float(packet.extracted_at)
        if not math.isfinite(extracted_at) or not stamp <= extracted_at <= now+1e-9:
            _fail('product_mismatch', 'Prepared obstacle extraction time is invalid')
        diagnostics = source.diagnostics
        if diagnostics.get('accepted') is not True or diagnostics.get('source_type') != 'rgbd':
            _fail('no_publication', 'Mapping diagnostics do not identify an accepted RGB-D publication')
        ids, frames = _camera_metadata(diagnostics, settings, stamp, now)
        extraction = packet.diagnostics
        if extraction.get('max_obstacles') != max_obstacles:
            _fail('budget_mismatch', 'Prepared obstacle budget differs from the selected global bound')
        if extraction.get('padding_m') != 0. or extraction.get('temporal_probabilities') is not True:
            _fail('invalid_product', 'Prepared obstacle policy is not the established exact RGB-D policy')
        boxes = _boxes(packet, view.spec, now=now, maximum=max_obstacles)
        occupied, supported, represented, omitted, candidates, proxies = (
            _count(extraction, key) for key in ('occupied_voxels','supported_voxels','represented_voxels',
                'omitted_voxels','candidate_boxes','proxy_count'))
        if (not occupied >= supported >= represented or represented != sum(box.voxel_count for box in boxes)
                or omitted != supported-represented or candidates < len(boxes) or proxies != len(boxes)):
            _fail('invalid_product', 'Prepared occupied-cell accounting is inconsistent')
        if not boxes:
            _fail('no_usable_evidence', 'RGB-D publication contains no usable selected occupied evidence',
                  extraction=dict(extraction))
        if source.generation != generation or source.settings is not settings:
            _fail('generation_changed', 'Mapping source changed before frozen-world acceptance')
        metadata = dict(source_type='rgbd', snapshot_policy='frozen_worker_prepared_exact_occupied_cuboids',
            timestamp=stamp, original_acquisition_timestamp=stamp, evaluated_at=now,
            generation=generation, revision=view.revision, map_age_seconds=max(0.,now-stamp),
            freshness_limit_seconds=limit, evidence_ttl_seconds=settings.evidence_ttl,
            mapping_health=dict(health), bounds_min=view.spec.bounds_min, bounds_max=view.spec.bounds_max,
            voxel_size=view.spec.voxel_size, dimensions=view.spec.shape,
            selected_obstacle_count=len(boxes), max_obstacles=max_obstacles,
            occupied_cells_at_extraction=occupied, supported_occupied_cells_at_extraction=supported,
            represented_occupied_cells=represented, omitted_supported_cells_at_extraction=omitted,
            counts_evaluated_at=extracted_at, extraction=dict(extraction),
            camera_ids=ids, accepted_frame_count=len(frames), source_frames=frames,
            mapping_diagnostics=diagnostics, robot_filter=dict(diagnostics['robot_filter']),
            uncertainty_policy='UNKNOWN is neither verified FREE nor registered as OCCUPIED; OUT_OF_BOUNDS remains distinct',
            no_occupancy_completion=True, padding_m=0.)
        return FrozenRgbdWorld(generation, view.revision, stamp, now, view.spec, boxes, metadata, view)
    except FrozenRgbdSourceError:
        raise
    except Exception as error:
        raise FrozenRgbdSourceError('unavailable', 'RGB-D mapping capture failed: '+str(error)) from error
