"""One optional, exposure-bound measured-box product in the mapping worker."""
from dataclasses import dataclass
import math
import time

import numpy as np

from .geometry import freeze
from .box_geometry import reference, visibility


@dataclass(frozen=True)
class InferredGeometry:
    """Optional measured-box hypotheses tied to one mapping publication."""
    generation: int
    revision: int
    timestamp: float
    status: str
    reason: str
    models: tuple[object, ...]
    source_frames: tuple[object, ...]
    diagnostics: object

    def __post_init__(self) -> None:
        if (type(self.generation) is not int or self.generation < 0
                or type(self.revision) is not int or self.revision < 1
                or not math.isfinite(self.timestamp) or self.timestamp < 0
                or self.status not in ('qualified', 'empty', 'rejected', 'error')):
            raise ValueError('Invalid inferred-geometry publication identity/status')
        for key in ('models', 'source_frames', 'diagnostics'):
            object.__setattr__(self, key, freeze(getattr(self, key)))


def validate_geometry(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate the dimensions, pose, and yaw of an inferred box."""
    center, dimensions, rotation = (np.asarray(g[k], float) for k in ('center', 'dimensions', 'rotation'))
    yaw = float(g['yaw_rad'])
    if (center.shape != (3,) or dimensions.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(center).all() or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or not math.isfinite(yaw)):
        raise ValueError('Inferred box must have finite positive geometry')
    c, s = math.cos(yaw), math.sin(yaw)
    if not np.allclose(rotation, [[c,-s,0],[s,c,0],[0,0,1]], atol=1e-12, rtol=0.):
        raise ValueError('Inferred rotation and yaw disagree')
    return center, dimensions, rotation, yaw


def infer(payload, scene, snapshot) -> InferredGeometry:
    """Never query SDK, semantic identities or fixtures; never mutate occupancy."""
    mapping_input, occupancy_snapshot = payload, snapshot
    started = time.perf_counter()
    models, diagnostics, status, reason = [], {}, 'error', 'invalid_sensor_input'
    frames = tuple(dict(camera_id=f.camera_id, token=f.token, acquired_at=f.acquired_at,
        depth_shape=f.depth.shape, intrinsics=f.intrinsics, world_from_camera=f.world_from_camera)
        for f in mapping_input.frames)
    try:
        if (mapping_input.ground_truth is not None or len(mapping_input.frames) != 2
                or any(f.depth.shape != (480,640) for f in mapping_input.frames)
                or any(not np.any(np.isfinite(f.depth)&(f.depth>0)) for f in mapping_input.frames)
                or mapping_input.robot_snapshots is None):
            raise ValueError('Reconstruction requires two original calibrated RGB-D frames and robot snapshots')
        result, _, _ = reference.reconstruct(mapping_input.frames, scene, mapping_input.robot_snapshots)
        all_models = [dict(component=c['component'], **visibility.qualify(c, mapping_input.frames))
                      for c in result['components']]
        diagnostics.update(reconstruction=result, qualifications=all_models)
        status, reason = 'qualified', 'all_eligible_components_qualified'
        for c, model in zip(result['components'], all_models):
            if c.get('reason') == 'possible_robot_component':
                if c['robot_envelope_points'] != c['points']:
                    status, reason = 'rejected', 'robot_connected_component'
                    break
                continue
            if model['status'] != 'qualified':
                status, reason = 'rejected', f"component_{c['component']}: {model['reason']}"
                break
            validate_geometry(model['geometry'])
            models.append(model)
        if status == 'qualified' and not models:
            status, reason = 'empty', 'no_eligible_nonrobot_component; observed cuboids retained'
        if status != 'qualified':
            models = []
    except Exception as exc:
        status, reason, models = 'error', f'{type(exc).__name__}: {exc}', []
    diagnostics['elapsed_seconds'] = time.perf_counter()-started
    return InferredGeometry(mapping_input.generation, occupancy_snapshot.revision, occupancy_snapshot.timestamp,
                            status, reason, tuple(models), frames, diagnostics)


def combine(frozen) -> tuple[list[dict], dict, InferredGeometry]:
    """Return numeric camera cuboids; preserve the exact original selected union."""
    product = frozen.view.inferred
    if not isinstance(product, InferredGeometry):
        raise ValueError('No inferred geometry accompanies this occupancy publication')
    if (product.generation, product.revision, product.timestamp) != (frozen.generation, frozen.revision, frozen.timestamp):
        raise ValueError('Inferred geometry and occupancy identity differ')
    records = frozen.metadata['source_frames']
    if len(records) != len(product.source_frames):
        raise ValueError('Inferred source cameras differ')
    for a, b in zip(records, product.source_frames):
        if any(not np.array_equal(a[k], b[k]) for k in
               ('camera_id','token','acquired_at','depth_shape','intrinsics','world_from_camera')):
            raise ValueError('Inferred exposure/calibration differs from frozen occupancy')
    if product.status not in ('qualified', 'empty'):
        raise ValueError('Reconstruction '+product.status+': '+product.reason)
    if (product.status == 'qualified') != bool(product.models):
        raise ValueError('Inferred status/model count disagree')
    models = [validate_geometry(m['geometry']) for m in product.models if m['status'] == 'qualified']
    if len(models) != len(product.models):
        raise ValueError('Unqualified geometry cannot enter planning')
    descriptors, subsumed, retained, floors = [], {}, [], []
    for i, (center, dimensions, rotation, yaw) in enumerate(models):
        path = f'/__rgbd_inferred__/{frozen.generation}/{frozen.revision}/{i}'
        descriptors.append(dict(path=path, position=tuple(center), dimensions=tuple(dimensions),
            orientation=(math.cos(yaw/2),0.,0.,math.sin(yaw/2)), source_kind='rgbd_inferred_box'))
        subsumed[path] = []
    for box in frozen.boxes:
        top = box.center[2]+box.dimensions[2]/2
        support_z = float(frozen.metadata['extraction']['support_plane_z'])
        floor = top <= support_z+frozen.spec.voxel_size+1e-12
        if floor:
            floors.append(box.id)
        owner = None
        if not floor:
            corners = np.asarray(box.center)+reference.CORNERS*np.asarray(box.dimensions)
            for descriptor, (center, dimensions, rotation, yaw) in zip(descriptors, models):
                if np.all(abs((corners-center)@rotation)+1e-12 <= dimensions/2):
                    owner = descriptor['path']
                    break
        if owner is None:
            retained.append(box.id)
            descriptors.append(dict(path='/__rgbd_occupied__/'+box.id, position=box.center,
                dimensions=box.dimensions, orientation=box.orientation_wxyz, source_kind='rgbd_occupied_cells'))
        else:
            subsumed[owner].append(box.id)
    if len(descriptors) > 64:
        raise ValueError('Combined inferred/observed world exceeds 64 camera proxies')
    original = {b.id for b in frozen.boxes}
    replaced = {i for ids in subsumed.values() for i in ids}
    if set(retained)|replaced != original or set(retained)&replaced or not set(floors)<=set(retained):
        raise ValueError('Original occupied union or protected floor evidence was lost')
    audit = dict(original_selected=len(original), retained_selected_ids=retained,
        subsumed=subsumed, protected_floor_ids=floors, combined_camera_proxies=len(descriptors),
        occupied_union_preserved=True, inference_status=product.status)
    return descriptors, audit, product
