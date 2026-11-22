"""Optional measured boxes and conservative assembly with observed cuboids."""
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from . import box_geometry
from .runtime import MappingInput, MappingReadView


@dataclass(frozen=True)
class InferredGeometry:
    """Qualified geometry, or a refusal reason; identity belongs to the map view."""
    models: tuple[Mapping[str, object], ...] = ()
    reason: str | None = None


def infer(payload: MappingInput, scene) -> InferredGeometry:
    """Qualify measured candidates using the exposure-matched robot observations."""
    return InferredGeometry(*box_geometry.reconstruct(payload.frames, scene, payload.robot_snapshots))


def combine(view: MappingReadView) -> list[dict[str, object]]:
    """Replace only fully contained, nonfloor cuboids with qualified boxes."""
    product = view.inferred
    if product is None:
        raise ValueError('No inferred geometry accompanies this occupancy publication')
    if product.reason is not None:
        raise ValueError('Reconstruction: ' + product.reason)
    models = product.models
    descriptors = []
    for i, model in enumerate(models):
        yaw = model['yaw_rad']
        descriptors.append(dict(path=f'/__rgbd_inferred__/{view.generation}/{view.revision}/{i}',
            position=tuple(model['center']), dimensions=tuple(model['dimensions']),
            orientation=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)), source_kind='rgbd_inferred_box'))
    floor_top = view.obstacles.support_plane_z + view.spec.voxel_size + 1e-12
    for box in view.obstacles.boxes:
        if box.center[2] + box.dimensions[2] / 2 > floor_top:
            corners = np.asarray(box.center) + box_geometry.CORNERS * np.asarray(box.dimensions)
            if any(np.all(abs((corners - model['center']) @ model['rotation']) + 1e-12 <= model['dimensions'] / 2)
                   for model in models):
                continue
        descriptors.append(dict(path='/__rgbd_occupied__/' + box.id, position=box.center,
            dimensions=box.dimensions, orientation=(1., 0., 0., 0.), source_kind='rgbd_occupied_cells'))
    if len(descriptors) > 64:
        raise ValueError('Combined inferred/observed world exceeds 64 camera proxies')
    return descriptors
