"""Visible USD robot meshes and exposure-matched read-only PhysX pose history."""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import (
    MeshInstance, RobotSelfFilter, RobotSnapshot, TriangleMesh,
)


def _usd_modules():
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def _simulation_interfaces():
    from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
    return SimulationManager, IsaacEvents


def _readonly(values):
    values = np.ascontiguousarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _triangles(vertices, counts, indices, holes):
    counts, indices, holes = np.asarray(counts), np.asarray(indices), np.asarray(holes)
    if (counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 3)
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or int(counts.sum()) != len(indices) or np.any(indices < 0)
            or np.any(indices >= len(vertices)) or holes.ndim != 1
            or (len(holes) and (holes.dtype.kind not in 'iu' or np.any(holes < 0) or np.any(holes >= len(counts))))):
        raise ValueError('Robot mesh has invalid polygon topology')
    selected = np.ones(len(counts), bool)
    selected[holes.astype(int)] = False
    if np.all(counts == 3):
        return indices.reshape(-1, 3)[selected]
    triangles, cursor = [], 0
    for keep, count in zip(selected, counts):
        face = indices[cursor:cursor+count]
        cursor += count
        if not keep:
            continue
        if count > 3:
            points = vertices[face]
            normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
            magnitude = np.linalg.norm(normal)
            if magnitude <= 1e-15:
                raise ValueError('Robot mesh polygon has no stable plane')
            normal /= magnitude
            scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
            turns = np.cross(np.roll(points, -1, axis=0)-points,
                             np.roll(points, -2, axis=0)-np.roll(points, -1, axis=0)) @ normal
            if (np.max(abs((points-points[0]) @ normal)) > scale*1e-6 or
                    np.any(turns < -scale**2*1e-10)):
                raise ValueError('Robot mesh needs planar convex polygons or authored triangles')
        triangles.extend((face[0], face[i], face[i+1]) for i in range(1, int(count)-1))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _world_meters(matrix, meters_per_unit):
    """Local vertices are already meters; only the USD translation changes units."""
    matrix = np.asarray(matrix, dtype=float).T.copy()
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError('Robot geometry needs a finite affine USD transform')
    matrix[:3, 3] *= meters_per_unit
    return matrix


def _rigid_part(matrix):
    """Keep rigid placement; retain any static stretch in the mesh-to-body map."""
    left, _, right = np.linalg.svd(matrix[:3, :3])
    rotation = left @ right
    if np.linalg.det(rotation) <= 0:
        raise ValueError('Mirrored rigid robot geometry is unsupported')
    result = matrix.copy()
    result[:3, :3] = rotation
    return result


def _pose_matrices(poses, meters_per_unit):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError('PhysX robot poses must be finite XYZ/XYZW rows')
    quaternions = poses[:, 3:]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('PhysX robot pose has an invalid quaternion')
    x, y, z, w = (quaternions/norms[:, None]).T
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]).transpose(2, 0, 1)
    result[:, :3, 3] = poses[:, :3]*meters_per_unit
    return result


@dataclass(frozen=True)
class _Part:
    path: str
    rigid_path: str
    mesh: TriangleMesh
    local_to_rigid: np.ndarray


class IsaacRobotSelfFilter:
    """An optional owner of one pose subscription; never owns robot/camera control.

    Static visible meshes are cached once. Reinitialize after model/topology or
    scene changes. Missing history fails explicitly, including during startup;
    current USD transforms are never substituted for an older depth exposure.
    """
    def __init__(self, stage, robot_prim_path, *, stride=8, depth_tolerance=.005, pose_tolerance=.01):
        initialized_at = time.perf_counter()
        if (not isinstance(robot_prim_path, str) or not robot_prim_path.startswith('/') or
                any(not part.isidentifier() for part in robot_prim_path[1:].split('/'))):
            raise ValueError('Robot filter requires an explicit absolute robot prim path')
        self.robot_prim_path = robot_prim_path
        self._filter = RobotSelfFilter(stride=stride, depth_tolerance=depth_tolerance, pose_tolerance=pose_tolerance)
        self._history, self._generation, self._capture_error = deque(maxlen=256), 0, None
        self._pose_capture_calls, self._pose_capture_seconds, self._pose_capture_max_seconds = 0, 0., 0.
        self._closed, self._callback_id, self._view, self._manager = False, None, None, None
        self._parts = ()
        self.diagnostics = {'state': 'initializing', 'robot_prim_path': robot_prim_path}
        try:
            if stage is None:
                import omni.usd
                stage = omni.usd.get_context().get_stage()
            self._stage = stage
            self._parts = self._geometry(stage)
            manager, events = _simulation_interfaces()
            self._manager = manager
            simulation_view = manager.get_physics_sim_view()
            if simulation_view is None:
                raise RuntimeError('Robot filter requires initialized physics; reinitialize after physics is ready')
            paths = sorted({part.rigid_path for part in self._parts})
            self._view = simulation_view.create_rigid_body_view(paths)
            self._body_paths = tuple(self._view.prim_paths)
            if set(self._body_paths) != set(paths) or self._view.count != len(paths):
                raise RuntimeError('PhysX robot rigid-body view does not match the visible model')
            self._body_indices = {path: i for i, path in enumerate(self._body_paths)}
            self._record()
            self._callback_id = manager.register_callback(
                self._post_step, events.POST_PHYSICS_STEP, order=100)
            self._initialization_seconds = time.perf_counter()-initialized_at
            self.diagnostics.update(state='waiting_for_frames', meshes=len(self._parts),
                triangles=sum(len(part.mesh.triangles) for part in self._parts),
                initialization_seconds=self._initialization_seconds, pose_capture=self._pose_capture_metrics())
        except Exception:
            self.close()
            raise

    def _geometry(self, stage):
        Usd, UsdGeom, UsdPhysics = _usd_modules()
        if stage is None:
            raise RuntimeError('No active stage for robot self-filter')
        root = stage.GetPrimAtPath(self.robot_prim_path)
        if not root.IsValid() or not root.IsActive() or not root.IsLoaded():
            raise RuntimeError(f'Robot prim is unavailable: {self.robot_prim_path}')
        units = self._meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isfinite(units) or units <= 0:
            raise ValueError('Robot stage metersPerUnit must be finite and positive')
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parts = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
                continue
            if not prim.IsLoaded():
                raise RuntimeError(f'Robot model has an unloaded payload: {prim.GetPath()}')
            if not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or
                    imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(f'Unsupported visible robot geometry: {prim.GetPath()}')
            mesh = UsdGeom.Mesh(prim)
            if str(mesh.GetSubdivisionSchemeAttr().Get()) != 'none':
                raise RuntimeError(f'Robot mesh needs explicit unsubdivided geometry: {prim.GetPath()}')
            attrs = [mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(),
                     mesh.GetFaceVertexIndicesAttr(), mesh.GetHoleIndicesAttr()]
            if any(attr.ValueMightBeTimeVarying() for attr in attrs):
                raise RuntimeError(f'Deforming robot meshes require reinitialization: {prim.GetPath()}')
            points, counts, indices, holes = [np.asarray(attr.Get()) for attr in attrs]
            points = np.asarray(points, dtype=float)*units
            triangles = _triangles(points, counts, indices, holes)
            ancestor = prim
            while ancestor and ancestor != root and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            if not ancestor or not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f'Visible robot mesh has no rigid-body ancestor: {prim.GetPath()}')
            rigid_world = _rigid_part(_world_meters(cache.GetLocalToWorldTransform(ancestor), units))
            mesh_world = _world_meters(cache.GetLocalToWorldTransform(prim), units)
            parts.append(_Part(str(prim.GetPath()), str(ancestor.GetPath()), TriangleMesh(points, triangles),
                               _readonly(np.linalg.inv(rigid_world) @ mesh_world)))
        if not parts:
            raise RuntimeError(f'Robot model has no supported visible meshes: {self.robot_prim_path}')
        return tuple(parts)

    def _record(self):
        stamp = float(self._manager.get_simulation_time())
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError('Robot pose history requires finite nonnegative simulation time')
        raw = self._view.get_transforms()
        if hasattr(raw, 'cpu'):
            raw = raw.cpu()
        if hasattr(raw, 'numpy'):
            raw = raw.numpy()
        raw = _readonly(raw)
        if raw.shape != (len(self._body_paths), 7):
            raise ValueError('PhysX robot rigid-body count changed; reinitialize the filter')
        _pose_matrices(raw, self._meters_per_unit)  # Validate before retaining any history.
        if abs(float(self._manager.get_simulation_time())-stamp) > 1e-9:
            raise RuntimeError('Physics changed during robot pose capture')
        if self._history and stamp < self._history[-1][0]:
            self._history.clear()
            self._generation += 1
        if self._history and stamp == self._history[-1][0]:
            self._history.pop()
        self._history.append((stamp, raw))
        self._capture_error = None

    def _post_step(self, step_dt):
        if self._closed:
            return
        started = time.perf_counter()
        try:
            self._record()
        except Exception as exc:
            self._history.clear()
            self._capture_error = str(exc)  # Report on lookup, never disrupt the physics callback loop.
        finally:
            elapsed = time.perf_counter()-started
            self._pose_capture_calls += 1
            self._pose_capture_seconds += elapsed
            self._pose_capture_max_seconds = max(self._pose_capture_max_seconds, elapsed)

    def _pose_capture_metrics(self):
        return dict(calls=self._pose_capture_calls, total_seconds=self._pose_capture_seconds,
                    mean_seconds=self._pose_capture_seconds/max(self._pose_capture_calls, 1),
                    max_seconds=self._pose_capture_max_seconds)

    def masks(self, frames, *, pixels_by_camera=None):
        started = time.perf_counter()
        frames = tuple(frames)
        try:
            if self._closed:
                raise RuntimeError('Robot self-filter is closed')
            if self._capture_error is not None:
                raise RuntimeError(f'Robot pose capture failed: {self._capture_error}')
            if not self._history:
                raise RuntimeError('Robot pose history is empty; wait for fresh physics/render snapshots')
            if float(self._manager.get_simulation_time()) < self._history[-1][0]:
                self._history.clear()
                self._generation += 1
                raise RuntimeError('Robot simulation clock reset; wait for fresh pose history')
            snapshots, assembled = {}, {}
            for frame in frames:
                stamp, poses = min(self._history, key=lambda item: abs(item[0]-float(frame.token[1])))
                if abs(stamp-float(frame.token[1])) > self._filter.pose_tolerance:
                    raise RuntimeError(f'No robot pose at camera render time: {frame.camera_id}, {frame.token[1]}')
                if stamp not in assembled:
                    matrices = _pose_matrices(poses, self._meters_per_unit)
                    assembled[stamp] = tuple(MeshInstance(part.mesh,
                        matrices[self._body_indices[part.rigid_path]] @ part.local_to_rigid)
                        for part in self._parts)
                snapshots[frame.camera_id] = RobotSnapshot(assembled[stamp], stamp, frame.token)
            masks = self._filter.masks(frames, snapshots, pixels_by_camera=pixels_by_camera)
            self.diagnostics = dict(self._filter.diagnostics, robot_prim_path=self.robot_prim_path,
                meshes=len(self._parts), triangles=sum(len(part.mesh.triangles) for part in self._parts),
                history_count=len(self._history), history_generation=self._generation,
                history_range=[self._history[0][0], self._history[-1][0]],
                pose_times={key: value.simulation_time for key, value in snapshots.items()},
                initialization_seconds=self._initialization_seconds, pose_capture=self._pose_capture_metrics(),
                elapsed_seconds=time.perf_counter()-started)
            return masks
        except Exception as exc:
            self.diagnostics = dict(state='error', reason=str(exc), robot_prim_path=self.robot_prim_path,
                history_count=len(self._history), history_generation=self._generation,
                initialization_seconds=self._initialization_seconds, pose_capture=self._pose_capture_metrics(),
                elapsed_seconds=time.perf_counter()-started)
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                if self._callback_id is not None:
                    self._manager.deregister_callback(self._callback_id)
            finally:
                self._callback_id, self._view = None, None
                self._history.clear()
