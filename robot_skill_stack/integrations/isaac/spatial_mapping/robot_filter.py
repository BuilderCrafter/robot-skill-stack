"""Cached USD robot meshes and exposure-matched PhysX poses for the mapping worker."""
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import threading

import numpy as np

from robot_skill_stack.world.spatial_mapping.robot_filter import (
    MeshInstance, RobotSnapshot, TriangleMesh,
)


def _usd_modules() -> tuple[object, object, object]:
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def _register_model_notice(stage, callback):
    from pxr import Tf, Usd
    return Tf.Notice.Register(Usd.Notice.ObjectsChanged, callback, stage)


def _simulation_interfaces() -> tuple[object, object]:
    from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
    return SimulationManager, IsaacEvents


def _world_meters(matrix, meters_per_unit: float) -> np.ndarray:
    """Local vertices are already meters; convert the USD translation only."""
    result = np.asarray(matrix, dtype=float).T.copy()
    result[:3, 3] *= meters_per_unit
    return result


def _rigid_part(matrix: np.ndarray) -> np.ndarray:
    """Leave static stretch in the mesh-to-body map, outside the PhysX pose."""
    left, _, right = np.linalg.svd(matrix[:3, :3])
    rotation = left @ right
    if np.linalg.det(rotation) <= 0:
        raise ValueError('Mirrored rigid robot geometry is unsupported')
    result = matrix.copy()
    result[:3, :3] = rotation
    return result


def _pose_matrices(poses: np.ndarray, meters_per_unit: float) -> np.ndarray:
    """Own world transforms from the configured NumPy PhysX backend's XYZ/XYZW rows."""
    poses = np.asarray(poses, dtype=float)
    quaternions = poses[:, 3:]
    x, y, z, w = (quaternions / np.linalg.norm(quaternions, axis=1)[:, None]).T
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]).transpose(2, 0, 1)
    result[:, :3, 3] = poses[:, :3] * meters_per_unit
    return result


@dataclass(frozen=True)
class _Part:
    """Visible triangle mesh and its static placement within a rigid body."""
    path: str
    rigid_path: str
    mesh: TriangleMesh
    local_to_rigid: np.ndarray


class IsaacRobotSelfFilter:
    """Cache the fixed robot model; match camera exposures to post-physics poses.

    Model edits require reinitialization. Stop/Play retains meshes but requires
    fresh poses. SDK access stays on the owner thread; prepared parts own their
    numerical data and can outlive this adapter in the mapping worker.
    """
    def __init__(self, stage, robot_prim_path: str, *, pose_tolerance: float = .01) -> None:
        self._owner_thread = threading.get_ident()
        self.robot_prim_path, self.pose_tolerance = robot_prim_path, pose_tolerance
        self._history = deque(maxlen=256)
        self._capture_error = self._model_error = None
        self._closed = False
        self._callback_id = self._model_notice = None
        try:
            self._parts = self._geometry(stage)
            self._manager, events = _simulation_interfaces()
            self._bind(self._manager.get_physics_sim_view())
            self._record()
            self._callback_id = self._manager.register_callback(
                self._post_step, events.POST_PHYSICS_STEP, order=100)
            self._model_notice = _register_model_notice(stage, self._model_changed)
        except Exception:
            self.close()
            raise

    def _geometry(self, stage) -> tuple[_Part, ...]:
        """Cache static, authored triangle meshes from the visible robot subtree."""
        Usd, UsdGeom, UsdPhysics = _usd_modules()
        root = stage.GetPrimAtPath(self.robot_prim_path)
        if not root or not root.IsLoaded():
            raise RuntimeError(f'Robot prim is unavailable: {self.robot_prim_path}')
        units = self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parts = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
                continue
            if not prim.IsLoaded():
                raise RuntimeError(f'Unloaded robot geometry: {prim.GetPath()}')
            if not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or
                    imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(f'Unsupported visible robot geometry: {prim.GetPath()}')
            mesh = UsdGeom.Mesh(prim)
            if (mesh.GetSubdivisionSchemeAttr().Get() != 'none' or
                    np.any(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) != 3)):
                raise ValueError(f'Robot mesh requires authored triangles: {prim.GetPath()}')
            points = np.asarray(mesh.GetPointsAttr().Get(), float)*units
            triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            triangles = np.delete(triangles, np.asarray(mesh.GetHoleIndicesAttr().Get(), int), axis=0)
            body = prim
            while body != root and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f'Visible robot mesh has no rigid body: {prim.GetPath()}')
            rigid_world = _rigid_part(_world_meters(cache.GetLocalToWorldTransform(body), units))
            mesh_world = _world_meters(cache.GetLocalToWorldTransform(prim), units)
            parts.append(_Part(str(prim.GetPath()), str(body.GetPath()), TriangleMesh(points, triangles),
                               np.linalg.inv(rigid_world) @ mesh_world))
        if not parts:
            raise RuntimeError(f'Robot has no visible meshes: {self.robot_prim_path}')
        return tuple(parts)

    def _bind(self, simulation_view) -> None:
        """Bind cached mesh bodies in the order returned by PhysX."""
        paths = sorted({part.rigid_path for part in self._parts})
        view = simulation_view.create_rigid_body_view(paths)
        if set(view.prim_paths) != set(paths):
            self._model_error = 'Robot rigid-body paths changed; reinitialize self-filter'
            raise RuntimeError(self._model_error)
        self._view, self._simulation_view = view, simulation_view
        self._body_indices = {path: i for i, path in enumerate(view.prim_paths)}

    def _model_changed(self, notice, sender) -> None:
        """Invalidate cached surfaces, while allowing PhysX to move rigid bodies."""
        root = self.robot_prim_path
        geometry = {'points', 'faceVertexCounts', 'faceVertexIndices', 'holeIndices',
                    'subdivisionScheme', 'visibility', 'purpose'}
        for resync, paths in ((True, notice.GetResyncedPaths()), (False, notice.GetChangedInfoOnlyPaths())):
            for path in paths:
                prim, _, attribute = str(path).partition('.')
                if not (prim == root or prim.startswith(root+'/') or root.startswith(prim.rstrip('/')+'/')):
                    continue
                static_transform = attribute.startswith('xformOp') and any(
                    prim != part.rigid_path and prim.startswith(part.rigid_path+'/') and
                    (part.path == prim or part.path.startswith(prim+'/')) for part in self._parts)
                if ((resync and not attribute) or attribute in geometry or static_transform or
                        attribute.startswith('xformOp:scale')):
                    self._model_error = f'Robot model changed; reinitialize self-filter: {path}'
                    self._history.clear()
                    return

    def rebind_after_play(self) -> bool:
        """Refresh a replaced PhysX view without rebuilding meshes or sampling a pose."""
        self._assert_owner()
        if self._closed or self._model_error:
            raise RuntimeError(self.source_error)
        current = self._manager.get_physics_sim_view()
        if current is self._simulation_view:
            return False  # Pause retains its view; the runtime already reset history.
        self.reset_history()
        if current is None:
            self._capture_error = 'Waiting for an initialized physics view after Play'
            raise RuntimeError(self._capture_error)
        self._bind(current)
        return True

    def _record(self) -> None:
        """Convert each post-physics sample once; never retain a live tensor buffer."""
        timestamp = self._manager.get_simulation_time()
        matrices = _pose_matrices(self._view.get_transforms(), self._meters_per_unit)
        if self._history and timestamp < self._history[-1][0]:
            self._history.clear()
        if self._history and timestamp == self._history[-1][0]:
            self._history.pop()
        self._history.append((timestamp, matrices))
        self._capture_error = None

    def _post_step(self, step_dt: float) -> None:
        if self._closed or self._model_error:
            return
        try:
            self._record()
        except Exception as exc:
            self._history.clear()
            self._capture_error = str(exc)  # Never raise into the simulation loop.

    @property
    def source_error(self) -> str | None:
        """Cached failure for the publication barrier; no SDK reads."""
        return 'Robot self-filter is closed' if self._closed else self._model_error or self._capture_error

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('Robot pose preparation requires its SDK owner thread')

    def reset_history(self) -> None:
        """Discard old exposures at Stop/reset; leave permanent model errors intact."""
        self._assert_owner()
        self._history.clear()
        self._capture_error = None

    def prepare(self, frames: Sequence) -> dict[str, RobotSnapshot]:
        """Prepare numerical mesh instances at each camera's exposure time."""
        self._assert_owner()
        if self.source_error:
            raise RuntimeError(self.source_error)
        if self._history and self._manager.get_simulation_time() < self._history[-1][0]:
            self.reset_history()
        if not self._history:
            raise RuntimeError('Robot pose history is empty; wait for fresh physics')
        snapshots, instances = {}, {}
        for frame in frames:
            timestamp, matrices = min(self._history, key=lambda item: abs(item[0]-frame.token[1]))
            if abs(timestamp-frame.token[1]) > self.pose_tolerance:
                raise RuntimeError(f'No robot pose at camera exposure: {frame.camera_id}, {frame.token[1]}')
            if timestamp not in instances:
                instances[timestamp] = tuple(MeshInstance(part.mesh,
                    matrices[self._body_indices[part.rigid_path]] @ part.local_to_rigid) for part in self._parts)
            snapshots[frame.camera_id] = RobotSnapshot(instances[timestamp], timestamp, frame.token)
        return snapshots

    def close(self) -> None:
        """Release only this adapter's subscriptions, never shared PhysX resources."""
        if self._closed:
            return
        self._assert_owner()
        self._closed = True
        if self._model_notice is not None:
            self._model_notice.Revoke()
        if self._callback_id is not None:
            self._manager.deregister_callback(self._callback_id)
        self._history.clear()
