"""Evaluation-only oracle extraction and metrics. Never imported at runtime."""
from dataclasses import dataclass

import numpy as np

from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth
from tests.support.primitive_scene import Solid, render


@dataclass
class ResearchObject:
    name: str
    shape: str
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.
    radius: float = .025
    length: float = .08
    axis: tuple = (0., 0., 1.)


def oracle_mask(points, obj, tolerance=.0025):
    """Select visible depth samples close to a known analytic object's surface.

    This is deliberately oracle-assisted research segmentation, not a detector.
    Points near more than one object's surface must be excluded by the caller.
    """
    delta = points-obj.center
    if obj.shape == 'sphere':
        error = abs(np.linalg.norm(delta, axis=1)-obj.radius)
    elif obj.shape == 'cube':
        c, s = np.cos(obj.yaw), np.sin(obj.yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        q = abs(delta @ r)-obj.size/2
        error = abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))
    else:
        axis = np.asarray(obj.axis)
        along = delta @ axis
        q = np.column_stack((np.linalg.norm(delta-along[:, None]*axis, axis=1)-obj.radius,
                             abs(along)-obj.length/2))
        error = abs(np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0))
    return error <= tolerance


def metric_row(obj, fit, **extra):
    geometry = fit.geometry
    if obj.shape == 'cube':
        truth_size = np.sort(obj.size)
        predicted = np.sort(geometry.box_size) if geometry.box_size is not None else None
    elif obj.shape == 'sphere':
        truth_size = np.full(3, 2*obj.radius)
        predicted = np.full(3, 2*geometry.radius) if geometry.radius is not None else None
    else:
        truth_size = np.array([2*obj.radius, 2*obj.radius, obj.length])
        predicted = np.array([2*geometry.radius, 2*geometry.radius, geometry.length]) if (
            geometry.radius is not None and geometry.length is not None) else None
    axis_error = None
    if obj.shape == 'cylinder' and geometry.axis is not None:
        axis_error = float(np.rad2deg(np.arccos(np.clip(abs(np.asarray(obj.axis) @ geometry.axis), 0, 1))))
    if obj.shape == 'cube' and geometry.yaw is not None:
        axis_error = float(np.rad2deg(abs((geometry.yaw-obj.yaw+np.pi/4) % (np.pi/2)-np.pi/4)))
    return dict(object=obj.name, truth=obj.shape, predicted=geometry.shape.value,
                correct=geometry.shape.value == obj.shape, unknown=geometry.shape.value == 'unknown',
                center_error_mm=(float(np.linalg.norm(fit.center-obj.center)*1000)
                                 if extra.get('detected', True) else None),
                dimension_error_mm=None if predicted is None else float(np.max(abs(predicted-truth_size))*1000),
                axis_error_deg=axis_error, confidence=geometry.confidence,
                scores=geometry.scores, estimated_center=fit.center.tolist(),
                estimated_size=fit.size.tolist(), truth_center=obj.center.tolist(),
                estimated_dimensions=None if predicted is None else predicted.tolist(),
                estimated_axis=None if geometry.axis is None else geometry.axis.tolist(),
                estimated_yaw=geometry.yaw, residuals_m=geometry.metadata.get('residuals_m', {}),
                truth_dimensions=truth_size.tolist(), **extra)


def analytic_frames(solids, *, eyes=((1.05, .55, .90), (.25, -.85, .65)), seed=1, noise=0.):
    frames = []
    for index, eye in enumerate(eyes):
        frame = render(solids, eye=eye, resolution=(320, 240), seed=seed+index, noise=noise)
        # The analytic renderer uses integer pixel centres. Convert its calibration
        # explicitly so the new production projector uses Isaac's half-pixel centres.
        k = frame.intrinsics.copy()
        k[0, 2] += .5
        k[1, 2] += .5
        frames.append(DepthFrame(str(index), (1, .1), frame.depth, k, frame.world_from_camera, 1., frame.rgb))
    return frames
