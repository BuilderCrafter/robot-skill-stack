"""Fuse per-object XYZ surfaces before fitting; no V1 or simulator dependencies."""
import time

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.world.model.observations import ObjectObservation
from robot_skill_stack.world.model.provider import ObservationBatch
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, fuse_clouds, project_depth, voxelize
from robot_skill_stack.world.perception_multiview.fitting import SurfaceFitter
from robot_skill_stack.world.perception_multiview.segmentation import cluster_points, supported_object_cloud
from robot_skill_stack.world.perception_multiview.tracking import Detection, MultiViewTracker


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
        clouds = [voxelize(crop_workspace(project_depth(frame, stride=self.stride)[0], self.scene),
                           self.voxel_size, self.max_points) for frame in frames]
        projected = time.perf_counter()
        fused = fuse_clouds(clouds, voxel=self.voxel_size, max_points=self.max_points)
        merged = time.perf_counter()
        clusters = cluster_points(fused)
        segmented = time.perf_counter()
        detections, rejected = [], []
        for index, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
            supported, reason = supported_object_cloud(cluster, self.scene.support_plane_z)
            if ((not supported and reason != 'no_observed_support_contact') or
                    len(detections) >= 8 or index >= 16):
                rejected.append({'points': len(cluster), 'reason': reason if not supported else 'object_count_limit'})
                continue
            fit = self.fitter.fit(cluster, support_plane_z=self.scene.support_plane_z)
            if not supported:
                # Curved surfaces can establish the support contact even when its
                # pixels are hidden. Elevated fitted links still fail this test.
                bottom = fit.center[2]-fit.size[2]/2
                inferred = (fit.geometry.shape.value in ('sphere', 'cylinder') and
                            abs(bottom-self.scene.support_plane_z) <= .004)
                if not inferred:
                    rejected.append({'points': len(cluster), 'reason': reason})
                    continue
                fit.geometry.metadata['support_contact_inferred'] = True
            lo, hi = cluster.min(0)-.004, cluster.max(0)+.004
            views = tuple(frame.camera_id for frame, cloud in zip(frames, clouds)
                          if np.count_nonzero(((cloud >= lo) & (cloud <= hi)).all(1)) >= 5)
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
                graspable=None if known else False, confidence=geometry.confidence,
                source='multi_view_geometry', timestamp=timestamp,
                metadata={'class_confidence': geometry.confidence,
                          'contributing_views': list(track.detection.views),
                          'camera_count': len(frames), 'fused_points': len(track.detection.points),
                          'association_hint_match': track.hint_reason is not None,
                          'association_hint_reason': track.hint_reason}))
        finished = time.perf_counter()
        self.diagnostics = dict(synchronization=dict(self.rig.status), workspace_points=len(fused),
            clusters=len(clusters), detections=len(detections), rejected=rejected,
            acquisition_ms=(acquired-started)*1000, projection_ms=(projected-acquired)*1000,
            fusion_ms=(merged-projected)*1000, segmentation_ms=(segmented-merged)*1000,
            fitting_ms=(fitted-segmented)*1000, tracking_ms=(finished-fitted)*1000,
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
