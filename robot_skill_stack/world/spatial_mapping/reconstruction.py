"""Optional measured boxes and conservative assembly with observed cuboids."""
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .box_geometry import reference, visibility


@dataclass(frozen=True)
class InferredGeometry:
    """Qualified geometry, or a refusal reason; identity belongs to the map view."""
    models: tuple[Mapping[str, object], ...] = ()
    reason: str | None = None


def infer(payload, scene) -> InferredGeometry:
    """Qualify measured candidates using the exposure-matched robot observations."""
    result, _, _ = reference.reconstruct(payload.frames, scene, payload.robot_snapshots)
    models = []
    for component in result['components']:
        if component['reason'] == 'possible_robot_component':
            if component['robot_envelope_points'] != component['points']:
                return InferredGeometry(reason='robot_connected_component')
            continue
        qualified = visibility.qualify(component, payload.frames)
        if qualified['status'] != 'qualified':
            return InferredGeometry(reason=f"component_{component['component']}: {qualified['reason']}")
        models.append(qualified['geometry'])
    return InferredGeometry(tuple(models))


def combine(frozen) -> list[dict[str, object]]:
    """Replace only fully contained, nonfloor cuboids with qualified boxes."""
    product = frozen.view.inferred
    if product is None:
        raise ValueError('No inferred geometry accompanies this occupancy publication')
    if product.reason is not None:
        raise ValueError('Reconstruction: '+product.reason)
    models = product.models
    descriptors = []
    for i, model in enumerate(models):
        yaw = model['yaw_rad']
        descriptors.append(dict(path=f'/__rgbd_inferred__/{frozen.generation}/{frozen.revision}/{i}',
            position=tuple(model['center']), dimensions=tuple(model['dimensions']),
            orientation=(math.cos(yaw/2),0.,0.,math.sin(yaw/2)), source_kind='rgbd_inferred_box'))
    floor_top = frozen.view.obstacles.diagnostics['support_plane_z']+frozen.spec.voxel_size+1e-12
    for box in frozen.boxes:
        if box.center[2]+box.dimensions[2]/2 > floor_top:
            corners = np.asarray(box.center)+reference.CORNERS*np.asarray(box.dimensions)
            if any(np.all(abs((corners-model['center'])@model['rotation'])+1e-12 <= model['dimensions']/2)
                   for model in models):
                continue
        descriptors.append(dict(path='/__rgbd_occupied__/'+box.id, position=box.center,
            dimensions=box.dimensions, orientation=(1.,0.,0.,0.), source_kind='rgbd_occupied_cells'))
    if len(descriptors) > 64:
        raise ValueError('Combined inferred/observed world exceeds 64 camera proxies')
    return descriptors
