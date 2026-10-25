"""Small installed-Isaac-5.1 adapter for frozen-world, seven-joint planning.

All SDK calls belong to the caller's Kit thread. This module imports without
Isaac; the SDK is loaded only for an explicitly selected global request.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from robot_skill_stack.manipulation.motion.joint_route import (
    PlanningFailure, check_joint_limits, interpolate_route, route_array,
)


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('A finite nonzero wxyz orientation is required')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def orientation_error(first, second):
    relative = np.asarray(first).T @ np.asarray(second)
    return float(np.arccos(np.clip((np.trace(relative)-1.)/2., -1., 1.)))


def adjacent_self_collision_pairs(urdf_path):
    """Exclude rigidly joined and immediately adjacent links, from the URDF.

    Stock Franka collision spheres intentionally overlap at adjacent joints.
    Never learn an allowlist from the current robot's possibly colliding pose.
    Fixed joints are collapsed before one-actuated-joint adjacency is computed.
    """
    document = ET.parse(urdf_path).getroot()
    links = [element.attrib['name'] for element in document.findall('link')]
    parent = {name: name for name in links}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    joints = []
    for element in document.findall('joint'):
        first, last = element.find('parent').attrib['link'], element.find('child').attrib['link']
        joints.append((first, last, element.attrib.get('type')))
        if element.attrib.get('type') == 'fixed':
            parent[root(first)] = root(last)
    neighbors = {frozenset((root(a), root(b))) for a, b, kind in joints if kind != 'fixed'}
    return frozenset(frozenset((a, b)) for index, a in enumerate(links)
                     for b in links[index+1:]
                     if root(a) == root(b) or frozenset((root(a), root(b))) in neighbors)


@dataclass(frozen=True)
class LulaPlanResult:
    joint_names: tuple
    route: np.ndarray
    trajectory: object
    duration: float
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    jerk_limits: np.ndarray
    diagnostics: dict


def _load_sdk():
    import lula
    from isaacsim.robot_motion.motion_generation import (
        LulaKinematicsSolver, LulaCSpaceTrajectoryGenerator, interface_config_loader,
    )
    from isaacsim.robot_motion.motion_generation.lula import RRT
    return SimpleNamespace(lula=lula, RRT=RRT, Kinematics=LulaKinematicsSolver,
        Generator=LulaCSpaceTrajectoryGenerator,
        load_config=interface_config_loader.load_supported_path_planner_config)


class LulaGlobalPlanner:
    """One bounded IK/RRT/trajectory transaction against a frozen scene.

    The stock external collision model has 55 arm/hand spheres and assumes
    fixed fingers at .025 m. It is not a complete physical mesh certificate.
    Dense SDK-model checks precede execution; independent native measurements
    assess the actual articulated meshes, including fingers.
    """
    def __init__(self, backend, *, seed=123456, max_iterations=10000, _sdk=None):
        if type(seed) is not int or seed <= 0:
            raise ValueError('RRT seed must be a positive integer')
        if type(max_iterations) is not int or not 1 <= max_iterations <= 50000:
            raise ValueError('RRT max_iterations must be from 1 through 50000')
        self.backend, self.seed, self.max_iterations = backend, seed, max_iterations
        self._sdk, self._owner, self._closed = _sdk, threading.get_ident(), False
        self._planner = self._inspector = self._generator = None
        self._handles = []
        self._last_diagnostics = {}
        self._config = None

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Global Lula calls must stay on the owning Kit thread')
        if self._closed:
            raise PlanningFailure('planner_closed', 'Global planner is closed')

    @property
    def diagnostics(self):
        return dict(self._last_diagnostics)

    @property
    def joint_names(self):
        """Actual RRT ordering, available before reading articulation positions."""
        self._assert_owner()
        self._sdk = self._sdk or _load_sdk()
        if self._config is None:
            self._config = dict(self._sdk.load_config('Franka', 'RRT'))
        if self._planner is None:
            self._planner = self._sdk.RRT(**self._config)
        names = tuple(self._planner.get_active_joints())
        if set(names) != {f'panda_joint{i}' for i in range(1, 8)} or len(names) != 7:
            raise PlanningFailure('joint_mapping_failed', 'Installed Franka RRT must expose seven distinct arm joints')
        return names

    def clear(self):
        self._assert_owner()
        errors = []
        if self._planner is not None:
            for handle in self._handles:
                try:
                    self._planner._world.remove_obstacle(handle)
                except Exception as error:
                    errors.append(error)
        self._handles.clear()
        self._inspector = self._generator = self._planner = None
        if errors:
            raise RuntimeError('Failed removing global collision registrations: '+
                               '; '.join(map(str, errors))) from errors[0]

    def close(self):
        if self._closed:
            return
        try:
            self.clear()
        finally:
            self._closed = True

    def _register(self, snapshot, base_position, base_rotation):
        lula, world = self._sdk.lula, self._planner._world
        started = time.perf_counter()
        for item in snapshot.obstacles:
            if item.kind == 'cuboid':
                obstacle = lula.create_obstacle(lula.Obstacle.Type.CUBE)
                obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(item.dimensions, float))
            elif item.kind in ('sphere', 'capsule'):
                kind = lula.Obstacle.Type.SPHERE if item.kind == 'sphere' else lula.Obstacle.Type.CYLINDER
                obstacle = lula.create_obstacle(kind)
                obstacle.set_attribute(lula.Obstacle.Attribute.RADIUS, float(item.radius))
                if item.kind == 'capsule':
                    obstacle.set_attribute(lula.Obstacle.Attribute.HEIGHT, float(item.height))
            else:
                raise PlanningFailure('unsupported_geometry', 'Unsupported static collision primitive',
                                      {'path': item.path, 'kind': item.kind})
            translation = base_rotation.T @ (np.asarray(item.position)-base_position)
            rotation = base_rotation.T @ quaternion_matrix(item.orientation)
            self._handles.append(world.add_obstacle(obstacle, lula.Pose3(lula.Rotation3(rotation), translation)))
        self._planner._rrt.update_world_view()
        self._inspector = lula.create_robot_world_inspector(
            self._planner._robot_description, world.add_world_view())
        self._inspector.update_world_view()
        return time.perf_counter()-started

    def _configuration(self, q, *, label, snapshot, base_position, base_rotation, excluded_pairs):
        if self._inspector.in_collision_with_obstacle(q):
            raise PlanningFailure(label, 'Configuration intersects the frozen planning world')
        pairs = self._inspector.frames_in_self_collision(q)
        unmasked = [tuple(pair) for pair in pairs if frozenset(pair) not in excluded_pairs]
        if unmasked:
            raise PlanningFailure(label, 'Configuration has non-adjacent self collision', {'pairs': unmasked})
        positions = np.asarray(self._inspector.collision_sphere_positions(q))
        world_positions = positions @ base_rotation.T + base_position
        try:
            snapshot.assert_within_domains(world_positions, self._inspector.collision_sphere_radii())
        except ValueError as error:
            raise PlanningFailure('unsupported_geometry', str(error)) from error

    def _validate_route(self, route, snapshot, base_position, base_rotation, excluded_pairs, limits):
        check_joint_limits(route, *limits)
        samples = interpolate_route(route, max_joint_step=.01)
        for q in samples:
            self._configuration(q, label='route_invalid', snapshot=snapshot,
                base_position=base_position, base_rotation=base_rotation, excluded_pairs=excluded_pairs)
        return len(samples)

    def _validate_trajectory(self, trajectory, snapshot, base_position, base_rotation, excluded_pairs,
                             lower, upper, velocity, acceleration, jerk):
        first, last = float(trajectory.start_time), float(trajectory.end_time)
        if not math.isfinite(first+last) or last <= first or last-first > 120.:
            raise PlanningFailure('trajectory_failed', 'Invalid or excessive generated trajectory duration')
        # <= 1/240 simulation second and <= .01rad L-infinity at robot velocity limits.
        step = min(1/240., .01 / float(np.max(velocity)))
        count = int(math.ceil((last-first)/step))+1
        if count > 100000:
            raise PlanningFailure('trajectory_failed', 'Trajectory validation sample bound exceeded')
        maxima = np.zeros((3, 7))
        previous = None
        for stamp in np.linspace(first, last, count):
            q, qd = trajectory.get_joint_targets(float(stamp))
            q, qd = np.asarray(q, float), np.asarray(qd, float)
            qdd = np.asarray(trajectory.trajectory.eval(float(stamp), 2), float)
            qddd = np.asarray(trajectory.trajectory.eval(float(stamp), 3), float)
            if any(value.shape != (7,) or not np.isfinite(value).all() for value in (q, qd, qdd, qddd)):
                raise PlanningFailure('trajectory_failed', 'Generated trajectory contains invalid joint targets')
            check_joint_limits(q[None, :], lower, upper)
            for index, (value, limit) in enumerate(((qd, velocity), (qdd, acceleration), (qddd, jerk))):
                if np.any(np.abs(value) > limit*(1+1e-6)+1e-8):
                    raise PlanningFailure('trajectory_failed', 'Generated trajectory exceeds robot derivative limits',
                                          {'derivative': index+1, 'time': float(stamp)})
                maxima[index] = np.maximum(maxima[index], np.abs(value))
            if previous is not None and np.max(np.abs(q-previous)) > .01001:
                raise PlanningFailure('trajectory_failed', 'Generated trajectory validation spacing exceeded')
            previous = q
            self._configuration(q, label='trajectory_failed', snapshot=snapshot,
                base_position=base_position, base_rotation=base_rotation, excluded_pairs=excluded_pairs)
        return dict(samples=count, maximum_time_step_seconds=step, maximum_joint_step_rad=.01,
            maximum_velocity_rad_s=maxima[0].tolist(), maximum_acceleration_rad_s2=maxima[1].tolist(),
            maximum_jerk_rad_s3=maxima[2].tolist(), collision_validation='PASS_SAMPLED_SDK_MODEL')

    def plan(self, target, start_q, snapshot, position_tolerance, orientation_tolerance):
        self._assert_owner()
        self.clear()
        self._last_diagnostics = dict(seed=self.seed, max_iterations=self.max_iterations,
                                      rrt_step_size_rad=.05, state='initializing')
        if target.orientation is None:
            raise PlanningFailure('ik_failed', 'Global planning requires a full end-effector orientation')
        names = self.joint_names
        cfg = self._config
        solver = self._sdk.Kinematics(robot_description_path=cfg['robot_description_path'], urdf_path=cfg['urdf_path'])
        if tuple(solver.get_joint_names()) != names:
            raise PlanningFailure('joint_mapping_failed', 'IK and RRT joint orders differ')
        position, orientation = self.backend.robot.get_world_pose()
        position, orientation = np.asarray(position, float), np.asarray(orientation, float)
        base_position, base_rotation = position*float(snapshot.meters_per_unit), quaternion_matrix(orientation)
        self._planner.set_robot_base_pose(position, orientation)
        solver.set_robot_base_pose(position, orientation)
        self._planner.set_max_iterations(self.max_iterations)
        # Isaac5.1's Python set_random_seed does not forward seed for C-space planning.
        self._planner.set_random_seed(self.seed)
        if not self._planner._rrt.set_param('seed', self.seed):
            raise PlanningFailure('planner_unavailable', 'Installed Lula rejected the recorded RRT seed')
        update_seconds = self._register(snapshot, base_position, base_rotation)
        self._last_diagnostics.update(static_obstacles=len(snapshot.obstacles),
            world_update_seconds=update_seconds, active_joint_names=list(names))
        excluded_pairs = adjacent_self_collision_pairs(cfg['urdf_path'])
        lower, upper = (np.asarray(value, float) for value in solver.get_cspace_position_limits())
        velocity = np.asarray(solver.get_cspace_velocity_limits(), float)
        acceleration = np.asarray(solver.get_cspace_acceleration_limits(), float)
        jerk = np.asarray(solver.get_cspace_jerk_limits(), float)
        if any(value.shape != (7,) or not np.isfinite(value).all() for value in
               (lower, upper, velocity, acceleration, jerk)) or np.any(np.minimum.reduce((velocity, acceleration, jerk)) <= 0):
            raise PlanningFailure('planner_unavailable', 'Installed Franka limits are incomplete')
        start_q = np.asarray(start_q, float)
        try:
            check_joint_limits(start_q[None, :], lower, upper)
            self._configuration(start_q, label='invalid_start', snapshot=snapshot,
                base_position=base_position, base_rotation=base_rotation, excluded_pairs=excluded_pairs)
        except PlanningFailure as error:
            raise PlanningFailure('invalid_start', str(error), error.details) from error
        solver.max_num_descents, solver.sampling_seed = 10, self.seed
        solver.ccd_max_iterations, solver.bfgs_max_iterations = 100, 100
        solver.set_default_cspace_seeds([])
        default = np.asarray(self._planner._robot_description.default_c_space_configuration(), float)
        seeds = [start_q, default, np.clip(start_q+np.array([.6, -.3, .4, 0, 0, 0, 0]), lower, upper),
                 np.clip(start_q+np.array([-.6, .3, -.4, 0, 0, 0, 0]), lower, upper)]
        target_rotation = quaternion_matrix(target.orientation)
        attempts, solutions = [], []
        self._last_diagnostics.update(ik_attempts=attempts, self_collision_mask='URDF rigid/adjacent links only',
            sdk_configuration_sha256={key: hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
                for key in ('robot_description_path', 'urdf_path', 'rrt_config_path')},
            collision_model='Installed 55 arm/hand spheres; fixed fingers .025m; sampled validation, not a mesh/continuous guarantee')
        current_position, current_rotation = solver.compute_forward_kinematics(
            self.backend.end_effector_frame, start_q)
        if (np.linalg.norm(np.asarray(current_position)-target.position) <= float(position_tolerance)
                and orientation_error(current_rotation, target_rotation) <= float(orientation_tolerance)):
            self._last_diagnostics.update(state='planned', already_at_goal=True,
                ik_candidates_attempted=0, rrt_waypoints=1, geometric_joint_path_length=0.,
                rrt_seconds=0., trajectory_generation_seconds=0., trajectory_duration=0.,
                trajectory_validation=dict(samples=1, collision_validation='PASS_SAMPLED_SDK_MODEL',
                    maximum_velocity_rad_s=[0.]*7, maximum_acceleration_rad_s2=[0.]*7,
                    maximum_jerk_rad_s3=[0.]*7))
            return LulaPlanResult(names, start_q[None, :].copy(), None, 0., lower, upper,
                velocity, acceleration, jerk, dict(self._last_diagnostics))
        final_error = PlanningFailure('ik_failed', 'No validated full-pose IK candidate was found')
        for index, seed in enumerate(seeds):
            entry = dict(index=index, warm_start=np.asarray(seed).tolist())
            attempts.append(entry)
            started = time.perf_counter()
            q, success = solver.compute_inverse_kinematics(self.backend.end_effector_frame,
                np.asarray(target.position, float), np.asarray(target.orientation, float),
                warm_start=np.asarray(seed), position_tolerance=min(float(position_tolerance), .001),
                orientation_tolerance=min(float(orientation_tolerance), .005))
            entry['ik_seconds'], entry['ik_success'] = time.perf_counter()-started, bool(success)
            if not success:
                continue
            try:
                q = np.asarray(q, float)
                check_joint_limits(q[None, :], lower, upper)
                fk_position, fk_rotation = solver.compute_forward_kinematics(self.backend.end_effector_frame, q)
                p_error = float(np.linalg.norm(np.asarray(fk_position)-target.position))
                r_error = orientation_error(fk_rotation, target_rotation)
                entry.update(fk_position_error=p_error, fk_orientation_error=r_error)
                if p_error > float(position_tolerance) or r_error > float(orientation_tolerance):
                    raise PlanningFailure('invalid_goal', 'IK forward-kinematics full-pose validation failed')
                if any(np.max(np.abs(q-existing)) < 1e-6 for existing in solutions):
                    entry['status'] = 'duplicate_ik'
                    continue
                solutions.append(q.copy())
                self._configuration(q, label='invalid_goal', snapshot=snapshot,
                    base_position=base_position, base_rotation=base_rotation, excluded_pairs=excluded_pairs)
                self._planner.set_cspace_target(q)
                started = time.perf_counter()
                route = self._planner.compute_path(start_q, np.empty(0, dtype=float))
                entry['rrt_seconds'] = time.perf_counter()-started
                if route is None:
                    raise PlanningFailure('search_exhausted', 'Bounded RRT search returned no route')
                route = route_array(route)
                if np.max(np.abs(route[0]-start_q)) > 1e-6 or np.max(np.abs(route[-1]-q)) > 1e-6:
                    raise PlanningFailure('route_invalid', 'RRT route endpoints differ from the validated joint states')
                entry['route_validation_samples'] = self._validate_route(route, snapshot,
                    base_position, base_rotation, excluded_pairs, (lower, upper))
                self._generator = self._sdk.Generator(robot_description_path=cfg['robot_description_path'], urdf_path=cfg['urdf_path'])
                if tuple(self._generator.get_active_joints()) != names:
                    raise PlanningFailure('joint_mapping_failed', 'Trajectory and RRT joint orders differ')
                started = time.perf_counter()
                trajectory = self._generator.compute_c_space_trajectory(route)
                entry['trajectory_generation_seconds'] = time.perf_counter()-started
                if trajectory is None:
                    raise PlanningFailure('trajectory_failed', 'Lula could not generate a joint trajectory')
                first_command, _ = trajectory.get_joint_targets(float(trajectory.start_time))
                last_command, _ = trajectory.get_joint_targets(float(trajectory.end_time))
                if (np.asarray(first_command).shape != (7,) or np.asarray(last_command).shape != (7,)
                        or not np.isfinite(first_command).all() or not np.isfinite(last_command).all()
                        or np.max(np.abs(np.asarray(first_command)-route[0])) > 1e-6
                        or np.max(np.abs(np.asarray(last_command)-route[-1])) > 1e-6):
                    raise PlanningFailure('trajectory_failed', 'Generated trajectory changed the selected joint-space endpoints')
                started = time.perf_counter()
                entry['trajectory_validation'] = self._validate_trajectory(trajectory, snapshot,
                    base_position, base_rotation, excluded_pairs, lower, upper, velocity, acceleration, jerk)
                entry['trajectory_validation_seconds'] = time.perf_counter()-started
                entry['status'] = 'planned'
                self._last_diagnostics.update(state='planned', ik_candidates_attempted=len(attempts), rrt_waypoints=len(route),
                    geometric_joint_path_length=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                    selected_ik_index=index, rrt_seconds=sum(row.get('rrt_seconds', 0) for row in attempts),
                    trajectory_generation_seconds=entry['trajectory_generation_seconds'],
                    trajectory_validation=entry['trajectory_validation'],
                    trajectory_duration=float(trajectory.end_time-trajectory.start_time))
                return LulaPlanResult(names, route.copy(), trajectory, float(trajectory.end_time-trajectory.start_time),
                    lower, upper, velocity, acceleration, jerk, dict(self._last_diagnostics))
            except PlanningFailure as error:
                entry.update(status=error.code, error=str(error), details=error.details)
                final_error = error
        self._last_diagnostics['ik_candidates_attempted'] = len(attempts)
        raise PlanningFailure(final_error.code, str(final_error), dict(self._last_diagnostics))
