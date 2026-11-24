"""Coordinate calibrated multiview geometry and persistent object observations."""
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.context import ObservationContext
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.common.camera import DepthFrame, FrameSynchronizer
from robot_skill_stack.world.common.components import cluster_points
from robot_skill_stack.world.common.point_clouds import crop_workspace, project_depth, voxelize
from robot_skill_stack.world.perception.multiview.fitting import (
    PrimitiveFit, SurfaceFitter, unknown_fit,
)
from robot_skill_stack.world.perception.multiview.observability import assess
from robot_skill_stack.world.perception.multiview.segmentation import refine_cluster, supported_object_cloud
from robot_skill_stack.world.perception.multiview.tracking import Detection, MultiViewTracker


class _FitBudget:
    """Share one numerical-fit budget between refinement and uncached fits."""
    def __init__(self, fitter: SurfaceFitter, max_calls: int = 24):
        self.fitter, self.remaining = fitter, max_calls

    def fit(self, points: np.ndarray, **kwargs) -> PrimitiveFit:
        if self.remaining <= 0:
            return unknown_fit(points, reason='candidate_budget_exhausted')
        self.remaining -= 1
        return self.fitter.fit(points, **kwargs)


def _voxel_ids(points: np.ndarray, voxel: float) -> np.ndarray:
    """Encode voxel coordinates for matching cluster points to each camera cloud."""
    keys = np.ascontiguousarray(np.floor(points / voxel + 1e-9), dtype=np.int64)
    return keys.view(np.dtype((np.void, 3 * keys.dtype.itemsize))).ravel()


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Freeze directly captured arrays in independent worker-owned storage."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class MultiViewInput:
    """Immutable synchronized frames and their oldest acquisition timestamp."""
    frames: tuple[DepthFrame, ...]
    timestamp: float


@dataclass(frozen=True)
class MultiViewResult:
    """Fused object detections retaining their original observation time."""
    timestamp: float
    detections: tuple[Detection, ...]


class MultiViewProcessor:
    """SDK-free geometry only; persistent identities remain with the provider."""
    def __init__(self, scene, *, voxel_size=.002, stride=2, max_points=12000):
        self.scene = deepcopy(scene)
        self.voxel_size, self.stride, self.max_points = voxel_size, stride, max_points
        self.fitter = SurfaceFitter()

    def compute(self, prepared: MultiViewInput) -> MultiViewResult:
        """Fuse depth clouds, discover clusters, fit shapes, and assess geometry."""
        frames = prepared.frames
        clouds = [voxelize(crop_workspace(project_depth(frame, stride=self.stride,
                           bounds=(self.scene.workspace_min, self.scene.workspace_max))[0], self.scene),
                           self.voxel_size, self.max_points) for frame in frames]
        view_voxels = [_voxel_ids(cloud, self.voxel_size) for cloud in clouds]
        # Each camera has already been density-balanced above; repeating its
        # voxelization in fuse_clouds would reproduce exactly the same centroids.
        fused = voxelize(np.concatenate(clouds), self.voxel_size, self.max_points)
        detections = []
        fit_budget = _FitBudget(self.fitter)
        for candidate in sorted(cluster_points(fused), key=len, reverse=True)[:16]:
            if len(detections) >= 8:
                break
            partitions = refine_cluster(candidate, fit_budget, support_plane_z=self.scene.support_plane_z)
            for cluster, cached_fit in partitions:
                if len(detections) >= 8:
                    break
                supported, reason = supported_object_cloud(cluster, self.scene.support_plane_z)
                if not supported and reason != 'no_observed_support_contact':
                    continue
                fit = cached_fit or fit_budget.fit(cluster, support_plane_z=self.scene.support_plane_z)
                cluster_voxels = _voxel_ids(cluster, self.voxel_size)
                view_clouds = tuple(cloud[np.isin(keys, cluster_voxels)]
                                    for cloud, keys in zip(clouds, view_voxels))
                fit = self.fitter.finalize(fit, cluster, support_plane_z=self.scene.support_plane_z,
                                           view_clouds=view_clouds)
                if not supported:
                    # Curved surfaces may establish a hidden support contact.
                    bottom = fit.center[2] - fit.size[2] / 2
                    inferred = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                                abs(bottom - self.scene.support_plane_z) <= .004)
                    if not inferred:
                        continue
                geometry = fit.geometry
                if geometry.shape.value == 'cube':
                    bounds = assess(fit, cluster, frames, support_plane_z=self.scene.support_plane_z)
                    geometry.metadata.pop('fitted_top_z')
                    geometry.metadata['parameter_observable'] = bool(
                        geometry.metadata['parameter_observable'] and bounds is not None)
                    if bounds is not None:
                        fit.center, geometry.box_size = bounds
                        c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
                        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                        fit.size = abs(rotation) @ geometry.box_size
                views = tuple(frame.camera_id for frame, cloud in zip(frames, view_clouds) if len(cloud) >= 5)
                detections.append(Detection(fit, cluster, views))
        return MultiViewResult(prepared.timestamp, tuple(detections))


class MultiViewProvider:
    """Connect synchronized frame capture, 3D processing, and object tracking.

    Publish confirmed detections as observations for the WorldModel.
    """
    def __init__(self, rig: FrameSynchronizer, scene, *, voxel_size=.002, stride=2, max_points=12000):
        self.rig = rig
        self.processor = MultiViewProcessor(scene, voxel_size=voxel_size, stride=stride, max_points=max_points)
        self.tracker = MultiViewTracker()
        self._closed = False

    def acquire(self, frames: Sequence[DepthFrame] | None = None) -> MultiViewInput | None:
        """Freeze direct captures or reuse a fresh immutable SharedFramePool batch.

        The rig and pool each admit complete, distinct exposures before this call.
        Supplied frames already own their buffers; no second frame copy is needed.
        """
        if self._closed:
            return None
        if frames is None:
            frames = self.rig.capture()
            if frames is None:
                return None
            frames = tuple(DepthFrame(frame.camera_id, frame.token, _immutable_array(frame.depth),
                _immutable_array(frame.intrinsics), _immutable_array(frame.world_from_camera), frame.acquired_at,
                None if frame.rgb is None else _immutable_array(frame.rgb)) for frame in frames)
        return MultiViewInput(tuple(frames), min(frame.acquired_at for frame in frames))

    def associate(self, result: MultiViewResult | None,
                  context: ObservationContext | None = None) -> ObservationBatch:
        """Associate on the owner and publish confirmed geometry and identities."""
        if self._closed or result is None:
            return ObservationBatch()
        timestamp, detections = result.timestamp, result.detections
        tracks = self.tracker.update(detections, timestamp, context)
        observations = []
        for track in tracks:
            fit = track.detection.fit
            geometry = fit.geometry
            known = geometry.shape.value != 'unknown'
            orientation = None if geometry.yaw is None else np.array([
                np.cos(geometry.yaw / 2), 0., 0., np.sin(geometry.yaw / 2)])
            observations.append(ObjectObservation(
                object_id=track.object_id, class_name=geometry.shape.value if known else None,
                pose=Pose(fit.center.copy(), orientation), size=fit.size.copy(), geometry=geometry,
                graspable=bool(known and geometry.metadata.get('parameter_observable', False)),
                confidence=geometry.confidence,
                source='multi_view_geometry', timestamp=timestamp,
                metadata={'contributing_views': list(track.detection.views),
                          'association_hint_match': track.matched_by_hint}))
        return ObservationBatch(observations, timestamp)

    def observe(self, context: ObservationContext | None = None) -> ObservationBatch:
        prepared = self.acquire()
        if prepared is None:
            return ObservationBatch()
        return self.associate(self.processor.compute(prepared), context)

    def forget(self, object_id: str) -> None:
        self.tracker.forget(object_id)

    def get_point_cloud(self, object_id: str) -> np.ndarray | None:
        """Return the current fused points for a visible tracked object."""
        track = self.tracker.tracks.get(object_id)
        return None if track is None or not track.visible else track.detection.points.copy()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.tracker.tracks.clear()
            self.rig.close()
