"""Capture robot-only Fabric poses after camera acquisition, never after HTTP inference."""
from collections import deque
import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.prims import XFormPrim
from robot_skill_stack.integrations.isaac.sensors.robot_meshes import load_robot_meshes
from robot_skill_stack.world.perception.v2.self_filter import RobotSnapshot


def pose_matrices(positions, orientations):
    p, q = np.asarray(positions, float), np.asarray(orientations, float)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4) or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('Invalid measured robot poses')
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < .9):
        raise ValueError('Invalid measured robot quaternion')
    w, x, y, z = (q/norms[:, None]).T  # Isaac core prim API: wxyz
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]).transpose(2, 0, 1)
    t = np.tile(np.eye(4), (len(p), 1, 1))
    t[:, :3, :3], t[:, :3, 3] = R, p
    return t


class IsaacRobotSnapshotSource:
    def __init__(self, world, camera, robot_root, excluded_roots, time_tolerance):
        self.world, self.camera, self.tolerance = world, camera, float(time_tolerance)
        self.stage = omni.usd.get_context().get_stage()
        self.model, self.inventory = load_robot_meshes(self.stage, robot_root, excluded_roots)
        self.view = XFormPrim(prim_paths_expr=list(self.model.names), name='perception_robot_self_filter',
                             reset_xform_properties=False)
        paths = list(self.view.prim_paths)
        if set(paths) != set(self.model.names):
            raise ValueError('Robot pose view differs from fixed mesh allow-list')
        self.order = [paths.index(name) for name in self.model.names]
        if not np.allclose(np.asarray(self.view.get_world_scales()), 1., atol=1e-5):
            raise ValueError('Scaled robot link roots are unsupported by the self-filter')
        self.latest, self.last_error, self.closed = None, 'waiting for matched camera/robot state', False
        self.history = deque(maxlen=256)
        self._subscription = omni.usd.get_context().get_rendering_event_stream().create_subscription_to_pop_by_type(
            int(omni.usd.StageRenderingEventType.NEW_FRAME), self._on_frame,
            name='robot_skill_stack.self_filter_snapshot', order=1100)
        self._update_subscription = omni.kit.app.get_app().get_post_update_event_stream().create_subscription_to_pop(
            self._on_frame, name='robot_skill_stack.self_filter_pose_history')
        print(f'[V2 self-filter] {len(self.model.names)} links, {self.model.triangle_count} triangles; '
              f'model {self.model.sha256[:12]}; surface matching (not silhouettes)', flush=True)

    def _sample_pose(self):
        before = float(self.world.current_time)
        if not np.isfinite(before):
            raise RuntimeError('Non-finite physics time')
        if self.history and before < self.history[-1][0]-1e-6:
            self.history.clear()
            self.latest = None
        p, q = self.view.get_world_poses(usd=False)
        poses = pose_matrices(p, q)[self.order]
        if abs(float(self.world.current_time)-before) > 1e-6:
            raise RuntimeError('Physics advanced while copying robot poses')
        # Freeze measured poses. An old GPU frame must never use newer joint state.
        poses.setflags(write=False)
        if self.history and abs(before-self.history[-1][0]) <= 1e-6:
            self.history[-1] = (before, poses)
        else:
            self.history.append((before, poses))

    def _on_frame(self, event):
        if self.closed:
            return
        try:
            if omni.usd.get_context().get_stage() != self.stage:
                raise RuntimeError('Robot self-filter stage changed; reinitialize runtime')
            self._sample_pose()
            token = self.camera.get_frame_token()
            if token is None or (self.latest is not None and token == self.latest.token):
                return
            capture_time = float(self.camera.camera.get_current_frame()['rendering_time'])
            if not np.isfinite(capture_time):
                raise RuntimeError('Non-finite camera rendering time')
            when, poses = min(self.history, key=lambda item: abs(item[0]-capture_time))
            if abs(when-capture_time) > self.tolerance:
                raise RuntimeError(f'No matching robot pose: render={capture_time:.6f}s nearest={when:.6f}s')
            if token != self.camera.get_frame_token():
                raise RuntimeError('Camera changed during robot snapshot')
            self.latest = RobotSnapshot(self.model, poses, capture_time, when, token)
            self.last_error = None
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'

    def snapshot(self, token):
        if self.closed or omni.usd.get_context().get_stage() != self.stage:
            raise RuntimeError('Robot self-filter stage changed; reinitialize runtime')
        if self.latest is None or self.latest.token != token:
            raise RuntimeError('No same-frame robot pose: '+str(self.last_error or 'waiting for render callback'))
        return self.latest

    def close(self):
        self.closed = True
        self._subscription, self._update_subscription, self.latest = None, None, None
        self.history.clear()
