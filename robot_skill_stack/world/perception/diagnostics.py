"""Explicit on-demand frame capture; no point clouds are stored in WorldModel."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from robot_skill_stack.world.perception.discovery import DepthObjectDiscoverer
from robot_skill_stack.world.perception.frame import PerceptionFrame
from robot_skill_stack.world.perception.localizer import RgbdLocalizer
from robot_skill_stack.world.perception.primitives import PrimitiveEstimator


DISCOVERY_FIELDS = (
    'support_plane_z', 'workspace_min', 'workspace_max', 'min_object_height',
    'max_object_height', 'min_component_pixels', 'max_component_pixels',
    'candidate_max_extent', 'spawn_min_extent', 'spawn_max_extent',
    'spawn_compactness_ratio', 'support_contact_tolerance', 'component_neighbor_distance',
)


def describe(candidate, geometry):
    return {
        'position': candidate.position.tolist(), 'size': candidate.size.tolist(),
        'pixels': int(candidate.mask.sum()), 'spawnable': candidate.spawnable,
        'shape': None if geometry is None else geometry.shape.value,
        'scores': {} if geometry is None else geometry.scores,
        'confidence': None if geometry is None else geometry.confidence,
        'yaw': None if geometry is None else geometry.yaw,
        'axis': None if geometry is None or geometry.axis is None else geometry.axis.tolist(),
        'radius': None if geometry is None else geometry.radius,
        'length': None if geometry is None else geometry.length,
        'reason': None if geometry is None else geometry.metadata.get('reason'),
    }


def save_capture(provider, path):
    if hasattr(provider, "save_capture"):
        return provider.save_capture(path)
    frame = provider.geometry.latest_frame
    if frame is None:
        raise ValueError('No camera frame has been received yet.')
    estimator = provider.primitive_estimator
    if estimator is None:
        raise ValueError('Primitive estimator is not enabled.')
    discovery = {}
    for name in DISCOVERY_FIELDS:
        value = getattr(provider.discoverer, name)
        discovery[name] = value.tolist() if isinstance(value, np.ndarray) else value
    settings = {
        'version': 1, 'source': provider.source, 'discovery': discovery,
        'primitives': {'classification_threshold': estimator.threshold, 'ambiguity_margin': estimator.margin,
                       'top_band': estimator.top_band, 'fit_tolerance': estimator.tolerance,
                       'max_fit_points': estimator.max_points},
        'candidates': provider.diagnostics, 'components': provider.discoverer.diagnostics,
    }
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Capture filename must end in .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=frame.rgb, depth=frame.depth, intrinsics=frame.intrinsics,
                        world_from_camera=frame.world_from_camera, timestamp=frame.timestamp,
                        frame_id=frame.frame_id, settings=json.dumps(settings))
    return path


def replay_capture(path):
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(str(data['settings']))
        if settings.get('version') == 2:
            from robot_skill_stack.world.perception.v2.replay import replay_saved
            return replay_saved(data, settings, path)
        if settings.get('version') != 1:
            raise ValueError('Unsupported capture version')
        frame = PerceptionFrame(int(data['frame_id']), data['rgb'], data['depth'], data['intrinsics'],
                                data['world_from_camera'], float(data['timestamp']))
    discoverer = DepthObjectDiscoverer(RgbdLocalizer(frame.intrinsics), **settings['discovery'])
    estimator = PrimitiveEstimator(**settings['primitives'])
    result = []
    for candidate in discoverer.discover(frame):
        geometry = estimator.estimate(candidate)
        if 'center' in geometry.metadata:
            candidate.position = np.asarray(geometry.metadata['center'], float)
            candidate.size = np.asarray(geometry.metadata['size_world'], float)
            candidate.spawnable = discoverer.is_spawnable(candidate.size)
        result.append(describe(candidate, geometry))
    return {'capture': str(path), 'candidates': result, 'components': discoverer.diagnostics}
