"""Fuse per-object XYZ surfaces before fitting; no V1 or simulator dependencies."""
import time

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter, unknown_fit
from robot_skill_stack.world.perception_multiview.observability import assess
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points, refine_cluster, supported_object_cloud
from robot_skill_stack.world.perception_multiview.tracking import Detection, MultiViewTracker


class _FitTimer:
    """Account for cached child fits separately from geometric partition work."""
    def __init__(self, fitter, max_calls=24):
        self.fitter, self.seconds = fitter, 0.
        self.max_calls, self.calls, self.exhaustions = max_calls, 0, 0

    @property
    def remaining(self):
        return max(0, self.max_calls-self.calls)

    def fit(self, points, **kwargs):
        started = time.perf_counter()
        try:
            if not self.remaining:
                self.exhaustions += 1
                return unknown_fit(points, reason='candidate_budget_exhausted')
            self.calls += 1
            return self.fitter.fit(points, **kwargs)
        finally:
            self.seconds += time.perf_counter()-started


def _voxel_ids(points, voxel):
    keys = np.ascontiguousarray(np.floor(points/voxel+1e-9), dtype=np.int64)
    return keys.view(np.dtype((np.void, 3*keys.dtype.itemsize))).ravel()


class MultiViewProvider:
    def __init__(self, rig, scene, *, voxel_size=.002, stride=2, max_points=12000):
        self.rig, self.scene = rig, scene
        self.voxel_size, self.stride, self.max_points = voxel_size, stride, max_points
        self.fitter, self.tracker = SurfaceFitter(), MultiViewTracker()
        self.diagnostics = {}
        self._closed = False

    def observe(self, context=None):
        if self._closed:
            return ObservationBatch()
        started = time.perf_counter()
        try:
            frames = self.rig.capture()
        except Exception:
            self.diagnostics['synchronization'] = dict(self.rig.status)
            raise
        if frames is None:
            self.diagnostics['synchronization'] = dict(self.rig.status)
            return ObservationBatch()
        acquired = time.perf_counter()
        clouds = [voxelize(crop_workspace(project_depth(frame, stride=self.stride,
                           bounds=(self.scene.workspace_min, self.scene.workspace_max))[0], self.scene),
                           self.voxel_size, self.max_points) for frame in frames]
        view_voxels = [_voxel_ids(cloud, self.voxel_size) for cloud in clouds]
        projected = time.perf_counter()
        # Each camera has already been density-balanced above; repeating its
        # voxelization in fuse_clouds would reproduce exactly the same centroids.
        fused = voxelize(np.concatenate(clouds), self.voxel_size, self.max_points)
        merged = time.perf_counter()
        clusters = cluster_points(fused)
        segmented = time.perf_counter()
        detections, rejected = [], []
        fit_timer = _FitTimer(self.fitter)
        refinement_seconds, observability_seconds, split_components = 0., 0., 0
        for index, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
            if len(detections) >= 8 or index >= 16:
                rejected.append({'points': len(cluster), 'reason': 'object_count_limit'})
                continue
            before_refine, before_fits = time.perf_counter(), fit_timer.seconds
            partitions = refine_cluster(cluster, fit_timer, support_plane_z=self.scene.support_plane_z, frames=frames)
            refinement_seconds += time.perf_counter()-before_refine-(fit_timer.seconds-before_fits)
            split_components += len(partitions) > 1
            for cluster, cached_fit in partitions:
                supported, reason = supported_object_cloud(cluster, self.scene.support_plane_z)
                if ((not supported and reason != 'no_observed_support_contact') or len(detections) >= 8):
                    rejected.append({'points': len(cluster), 'reason': reason if not supported else 'object_count_limit'})
                    continue
                fit = cached_fit or fit_timer.fit(cluster, support_plane_z=self.scene.support_plane_z)
                cluster_voxels = _voxel_ids(cluster, self.voxel_size)
                view_clouds = tuple(cloud[np.isin(keys, cluster_voxels)]
                                    for cloud, keys in zip(clouds, view_voxels))
                fit = self.fitter.finalize(fit, cluster, support_plane_z=self.scene.support_plane_z,
                                           view_clouds=view_clouds)
                if not supported:
                    # Curved surfaces may establish a hidden support contact.
                    bottom = fit.center[2]-fit.size[2]/2
                    inferred = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                                abs(bottom-self.scene.support_plane_z) <= .004)
                    if not inferred:
                        rejected.append({'points': len(cluster), 'reason': reason})
                        continue
                    fit.geometry.metadata['support_contact_inferred'] = True
                before_observability = time.perf_counter()
                evidence = assess(fit, cluster, frames, view_clouds=view_clouds,
                                  support_plane_z=self.scene.support_plane_z)
                geometry = fit.geometry
                geometry.metadata['geometric_observability'] = evidence
                if evidence['applicable']:
                    geometry.metadata['surface_parameters_observable'] = bool(
                        geometry.metadata.get('parameter_observable', False))
                    geometry.metadata['parameter_observable'] = bool(
                        geometry.metadata['surface_parameters_observable'] and evidence['observable'])
                    bounds = evidence['conservative_geometry']
                    if bounds is not None:
                        geometry.metadata['measured_box_geometry'] = dict(center=fit.center.tolist(),
                            box_size=geometry.box_size.tolist(), size_world=fit.size.tolist())
                        fit.center = np.asarray(bounds['center'], dtype=float)
                        geometry.box_size = np.asarray(bounds['box_size'], dtype=float)
                        c, s = np.cos(geometry.yaw), np.sin(geometry.yaw)
                        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                        fit.size = abs(rotation) @ geometry.box_size
                        geometry.metadata.update(center=fit.center.tolist(), size_world=fit.size.tolist())
                observability_seconds += time.perf_counter()-before_observability
                views = tuple(frame.camera_id for frame, cloud in zip(frames, view_clouds) if len(cloud) >= 5)
                detections.append(Detection(fit, cluster, views))
        fitted = time.perf_counter()
        timestamp = min(frame.acquired_at for frame in frames)
        tracks = self.tracker.update(detections, timestamp, context)
        observations = []
        for track in tracks:
            fit = track.detection.fit
            geometry = fit.geometry
            known = geometry.shape.value != 'unknown'
            orientation = None if geometry.yaw is None else np.array([
                np.cos(geometry.yaw/2), 0., 0., np.sin(geometry.yaw/2)])
            observations.append(ObjectObservation(
                object_id=track.object_id, class_name=geometry.shape.value if known else None,
                pose=Pose(fit.center.copy(), orientation), size=fit.size.copy(), geometry=geometry,
                graspable=bool(known and geometry.metadata.get('parameter_observable', False)),
                confidence=geometry.confidence,
                source='multi_view_geometry', timestamp=timestamp,
                metadata={'class_confidence': geometry.confidence,
                          'contributing_views': list(track.detection.views),
                          'camera_count': len(frames), 'fused_points': len(track.detection.points),
                          'association_hint_match': track.hint_reason is not None,
                          'association_hint_reason': track.hint_reason}))
        finished = time.perf_counter()
        self.diagnostics = dict(synchronization=dict(self.rig.status), workspace_points=len(fused),
            clusters=len(clusters), split_components=split_components, detections=len(detections), rejected=rejected,
            numerical_fits=fit_timer.calls, fit_budget=fit_timer.max_calls, fit_budget_exhaustions=fit_timer.exhaustions,
            acquisition_ms=(acquired-started)*1000, projection_ms=(projected-acquired)*1000,
            fusion_ms=(merged-projected)*1000,
            segmentation_ms=(segmented-merged+refinement_seconds)*1000,
            fitting_ms=(fitted-segmented-refinement_seconds-observability_seconds)*1000,
            observability_ms=observability_seconds*1000, tracking_ms=(finished-fitted)*1000,
            total_ms=(finished-started)*1000)
        return ObservationBatch(observations, timestamp)

    def forget(self, object_id):
        self.tracker.forget(object_id)

    def get_point_cloud(self, object_id):
        track = self.tracker.tracks.get(object_id)
        return None if track is None or not track.visible else track.detection.points.copy()

    def close(self):
        if not self._closed:
            self._closed = True
            self.tracker.tracks.clear()
            self.rig.close()
