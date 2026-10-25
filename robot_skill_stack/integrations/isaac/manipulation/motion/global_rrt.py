"""Opt-in frozen-world RRT planning followed by timed joint execution on Kit."""
from __future__ import annotations

import asyncio
import math
import threading
import time

import numpy as np

from robot_skill_stack.common.types import GripperState
from robot_skill_stack.manipulation.backend import BackendResult
from robot_skill_stack.manipulation.motion.joint_route import (
    PlanningFailure, map_arm_joints, simulation_progress,
)


def _action(**kwargs):
    from isaacsim.core.utils.types import ArticulationAction
    return ArticulationAction(**kwargs)


def _snapshot(backend):
    import omni.usd
    from .static_world import capture_static_world
    return capture_static_world(omni.usd.get_context().get_stage(),
                                robot_prim_path=backend.robot.prim_path)


def _planner(backend, **kwargs):
    from .lula_global import LulaGlobalPlanner
    return LulaGlobalPlanner(backend, **kwargs)


def _distribution(values):
    return dict(count=len(values), **({} if not values else {
        key: float(np.percentile(values, percentile))
        for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("worst", 100))}))


class GlobalRrtStrategy:
    """Plan once, validate before moving, and preserve the seven-joint route.

    Static geometry is not refreshed during execution. No mapping source,
    Cartesian RMPflow fallback, worker thread or joint teleport is used.
    """
    supports_contact_motion = False
    tracking_limit_rad = .15
    tracking_grace_seconds = .25
    tracking_abort_rad = .40
    settle_seconds = 3.
    physics_stall_seconds = 2.

    def __init__(self, backend, *, seed=123456, max_iterations=10000,
                 planner_factory=_planner, world_capture=_snapshot,
                 action_factory=_action, clock=time.perf_counter):
        self._backend = backend
        self.seed, self.max_iterations = seed, max_iterations
        self._planner_factory, self._world_capture = planner_factory, world_capture
        self._action, self._clock = action_factory, clock
        self._owner, self._closed, self._epoch = threading.get_ident(), False, 0
        self._active_planner = self._last_action = None
        self._state = "ready"
        self._details, self.last_plan_trace = {}, {}
        # Home's reference was measured by the backend at this same startup
        # state. Retain its IK branch as a goal hint, never as an unchecked path.
        home = getattr(backend, "named_poses", {}).get("home")
        self._home_reference = (None if home is None else
            (np.asarray(home.position).copy(), np.asarray(home.orientation).copy()))
        self._startup_joint_names = tuple(backend.robot.dof_names)
        self._startup_joints = np.asarray(backend.robot.get_joint_positions(), dtype=float).copy()

    @property
    def diagnostics(self):
        return dict(self._details, state=self._state, closed=self._closed,
                    motion_strategy="global", seed=self.seed, max_iterations=self.max_iterations)

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Global strategy must remain on its owning Kit thread")

    def _failure(self, error):
        self._state = "failed"
        self._details.update(error.details)
        self._details.update(failure_code=error.code, failure_message=str(error))
        return BackendResult(False, str(error),
            timed_out=error.code == "execution_timeout", details=self.diagnostics)

    def _live(self, epoch):
        if self._closed or epoch != self._epoch:
            raise PlanningFailure("lifecycle_invalidated", "Global strategy was closed or invalidated.")
        if not self._backend.world.is_playing() or not self._backend.is_ready:
            raise PlanningFailure("lifecycle_invalidated", "Simulation stopped or articulation was invalidated.")

    def _hold(self, action):
        self._backend.robot.apply_action(self._action(
            joint_positions=np.asarray(action.joint_positions).copy(),
            joint_velocities=np.zeros_like(action.joint_positions, dtype=float),
            joint_indices=np.asarray(action.joint_indices).copy()))

    def _goal_hint(self, target, names):
        if self._home_reference is None or target.orientation is None:
            return None
        position, orientation = self._home_reference
        if (np.linalg.norm(target.position-position) > 1e-9
                or min(np.linalg.norm(target.orientation-orientation),
                       np.linalg.norm(target.orientation+orientation)) > 1e-8):
            return None
        if (self._startup_joints.shape != (len(self._startup_joint_names),)
                or not np.isfinite(self._startup_joints).all()):
            return None
        indices = map_arm_joints(names, self._startup_joint_names)
        return self._startup_joints[list(indices)].copy()

    async def move_to_pose_async(self, target, position_tolerance=None,
                                 orientation_tolerance=None, grasp_check=None):
        self._assert_owner()
        backend = self._backend
        self._details, self.last_plan_trace = {}, {}
        epoch, planner, last_action, primary = self._epoch, None, None, None
        result, wall_start = None, None
        intervals, tracking, elapsed = [], [], 0.
        initial_time = initial_index = None
        pos_tol = backend.position_tolerance if position_tolerance is None else position_tolerance
        ori_tol = backend.orientation_tolerance if orientation_tolerance is None else orientation_tolerance
        self._details.update(position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol)
        try:
            self._live(epoch)
            if (target.frame != "world" or target.orientation is None
                    or not np.isfinite(target.position).all()
                    or not np.isfinite(target.orientation).all()
                    or not math.isfinite(pos_tol) or pos_tol <= 0
                    or not math.isfinite(ori_tol) or ori_tol <= 0):
                raise PlanningFailure("invalid_target", "Global planning requires a finite full world-frame pose and positive tolerances.")
            feedback = backend.gripper_feedback()
            if feedback.state != GripperState.FREE:
                raise PlanningFailure("unsupported_payload", "Global carrying is unsupported: payload collision geometry is not modeled.",
                                      {"grasp_state": feedback.state.value})
            self._state = "capturing_world"
            started = self._clock()
            snapshot = self._world_capture(backend)
            self._details["snapshot_seconds"] = self._clock()-started
            self.last_plan_trace["snapshot"] = snapshot
            planner = self._planner_factory(backend, seed=self.seed, max_iterations=self.max_iterations)
            self._active_planner = planner
            names = tuple(planner.joint_names)
            indices = np.asarray(map_arm_joints(names, backend.robot.dof_names), dtype=int)
            start_q = np.asarray(backend.robot.get_joint_positions(), dtype=float)[indices]
            if not np.isfinite(start_q).all():
                raise PlanningFailure("invalid_start", "Current arm configuration is unavailable.")
            initial_time, initial_index = float(backend.world.current_time), int(backend.world.current_time_step_index)
            base_position, base_orientation = backend.robot.get_world_pose()
            self._state = "planning"
            started = self._clock()
            self._details["planning_blocks_owner_loop"] = True
            preferred = self._goal_hint(target, names)
            self._details["goal_configuration_hint"] = (
                "recorded_startup_configuration_for_named_home" if preferred is not None else "none")
            try:
                plan = planner.plan(target, start_q, snapshot,
                    position_tolerance=pos_tol, orientation_tolerance=ori_tol,
                    preferred_goal_configuration=preferred)
            finally:
                self._details["planning_and_validation_wall_seconds"] = self._clock()-started
            self._details.update(plan.diagnostics)
            self._live(epoch)
            if tuple(plan.joint_names) != names:
                raise PlanningFailure("joint_mapping_failure", "Planner changed joint order during the request.")
            if (abs(float(backend.world.current_time)-initial_time) > 1e-9
                    or int(backend.world.current_time_step_index) != initial_index
                    or np.max(np.abs(np.asarray(backend.robot.get_joint_positions())[indices]-start_q)) > .005):
                raise PlanningFailure("start_changed", "Robot or simulation state changed while building the static plan.")
            duration = float(plan.duration)
            if duration == 0. and plan.diagnostics.get("already_at_goal", False):
                current = backend.get_end_effector_pose()
                position_error = float(np.linalg.norm(current.position-target.position))
                orientation_error = backend._quat_error(current.orientation, target.orientation)
                if position_error > pos_tol or orientation_error > ori_tol:
                    raise PlanningFailure("final_pose_failure", "No-motion plan failed the actual full-pose check.")
                self._details.update(position_error_m=position_error,
                    orientation_error_rad=orientation_error, trajectory_duration_seconds=0.,
                    commanded_updates=0)
                self.last_plan_trace.update(joint_names=names, joint_indices=indices.tolist(),
                    route=np.asarray(plan.route).tolist(), trajectory=[], commanded=[], actual=[])
                self._state = "succeeded"
                result = BackendResult(True, "Robot already satisfies the validated full pose.", details=self.diagnostics)
                return result
            if not math.isfinite(duration) or duration <= 0 or duration > 120.:
                raise PlanningFailure("trajectory_generation_failure", "Trajectory duration is invalid or exceeds the bounded execution window.")
            self._details.update(trajectory_duration_seconds=duration, joint_names=list(names),
                joint_indices=indices.tolist(), start_q=start_q.tolist(),
                tracking_limit_rad=self.tracking_limit_rad,
                tracking_grace_seconds=self.tracking_grace_seconds,
                tracking_abort_rad=self.tracking_abort_rad,
                execution_uses_physics_time=True, static_snapshot_refreshes=0,
                maximum_control_updates=backend.max_motion_steps)
            self.last_plan_trace.update(joint_names=names, joint_indices=indices.tolist(),
                route=np.asarray(plan.route).tolist(), trajectory=[], commanded=[], actual=[])
            for stamp in np.linspace(0., duration, min(6001, int(math.ceil(duration/.01))+1)):
                q, qd = plan.trajectory.get_joint_targets(float(stamp))
                self.last_plan_trace["trajectory"].append(dict(time=float(stamp),
                    q=np.asarray(q).tolist(), qd=np.asarray(qd).tolist()))
            self._state = "executing"
            wall_start = last_wall = self._clock()
            last_progress_wall, previous_time, previous_index = wall_start, initial_time, initial_index
            bad_tracking_since = None
            complete = False
            terminal_commanded = False
            elapsed = 0.
            exhausted_updates = True
            for step in range(backend.max_motion_steps):
                self._live(epoch)
                sim_time, sim_index = float(backend.world.current_time), int(backend.world.current_time_step_index)
                advanced = simulation_progress(previous_time, sim_time, previous_index, sim_index,
                    maximum_pause_seconds=self.physics_stall_seconds,
                    stalled_wall_seconds=self._clock()-last_progress_wall)
                if sim_time < initial_time-1e-9 or sim_index < initial_index:
                    raise PlanningFailure("lifecycle_invalidated", "Simulation reset invalidated the trajectory.")
                if advanced:
                    last_progress_wall = self._clock()
                previous_time, previous_index = sim_time, sim_index
                elapsed = sim_time-initial_time
                if self._clock()-wall_start > max(15., 5.*duration+10.):
                    raise PlanningFailure("execution_timeout", "Trajectory execution exceeded its wall-clock watchdog.")
                if elapsed > duration+self.settle_seconds:
                    exhausted_updates = False
                    break
                if step and not advanced:
                    await backend._next_update()
                    continue
                position, orientation = backend.robot.get_world_pose()
                if (np.linalg.norm(np.asarray(position)-base_position) > 1e-5
                        or backend._quat_error(orientation, base_orientation) > 1e-5):
                    raise PlanningFailure("lifecycle_invalidated", "Robot base moved after the planning snapshot.")
                q, qd = plan.trajectory.get_joint_targets(min(elapsed, duration))
                q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
                if elapsed >= duration:
                    qd = np.zeros(7)
                if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(qd).all():
                    raise PlanningFailure("execution_failure", "Trajectory returned an invalid joint command.")
                actual = np.asarray(backend.robot.get_joint_positions(), dtype=float)[indices]
                error = float(np.max(np.abs(actual-q)))
                if not np.isfinite(actual).all() or error > self.tracking_abort_rad:
                    raise PlanningFailure("tracking_failure", "Joint tracking exceeded the immediate abort bound.",
                                          {"tracking_error_rad": error})
                tracking.append(error)
                if error > self.tracking_limit_rad:
                    bad_tracking_since = sim_time if bad_tracking_since is None else bad_tracking_since
                    if sim_time-bad_tracking_since >= self.tracking_grace_seconds:
                        raise PlanningFailure("tracking_failure", "Joint tracking remained outside the declared bound.",
                                              {"tracking_error_rad": error})
                else:
                    bad_tracking_since = None
                current = backend.get_end_effector_pose()
                pos_error = float(np.linalg.norm(current.position-target.position))
                ori_error = float(backend._quat_error(current.orientation, target.orientation))
                self._details.update(position_error_m=pos_error, orientation_error_rad=ori_error)
                self.last_plan_trace["actual"].append(dict(sim_time=sim_time,
                    q=actual.tolist(), tcp=np.asarray(current.position).tolist(),
                    tracking_error_rad=error))
                if (terminal_commanded and elapsed >= duration and pos_error <= pos_tol
                        and ori_error <= ori_tol and error <= .05):
                    complete = True
                    exhausted_updates = False
                    break
                last_action = self._action(joint_positions=q, joint_velocities=qd, joint_indices=indices)
                backend.robot.apply_action(last_action)
                terminal_commanded = terminal_commanded or elapsed >= duration
                self._last_action = last_action
                stamp = self._clock()
                if self.last_plan_trace["commanded"]:
                    intervals.append(1000.*(stamp-last_wall))
                last_wall = stamp
                self.last_plan_trace["commanded"].append(dict(sim_time=sim_time, q=q.tolist(), qd=qd.tolist()))
                await backend._next_update()
                if grasp_check is not None:
                    failure = grasp_check()
                    if failure is not None:
                        raise PlanningFailure("unsupported_payload", "Grasp state changed during unsupported carrying motion.")
            self._details.update(execution_wall_seconds=self._clock()-wall_start,
                execution_sim_seconds=elapsed, control_interval_ms=_distribution(intervals),
                joint_tracking_error_rad=_distribution(tracking),
                commanded_updates=len(self.last_plan_trace["commanded"]))
            if not complete:
                if exhausted_updates:
                    raise PlanningFailure("execution_timeout", "Trajectory exceeded the backend's control-update budget.")
                raise PlanningFailure("final_pose_failure", "Trajectory ended without reaching the requested full pose.")
            self._state = "succeeded"
            result = BackendResult(True, "Global joint trajectory reached the requested pose.", details=self.diagnostics)
            return result
        except PlanningFailure as error:
            result = self._failure(error)
            return result
        except asyncio.CancelledError as error:
            primary = error
            self._state = "cancelled"
            self._details["failure_message"] = str(error)
            raise
        except Exception as error:
            code = ("unsupported_geometry" if self._state == "capturing_world" else
                    "execution_failure" if self._state == "executing" else "planning_failure")
            result = self._failure(PlanningFailure(code,
                f"Global {code}: {type(error).__name__}: {error}"))
            return result
        finally:
            errors = []
            try:
                can_hold = (last_action is not None and epoch == self._epoch and not self._closed
                    and backend.is_ready and backend.world.is_playing()
                    and float(backend.world.current_time) >= initial_time
                    and int(backend.world.current_time_step_index) >= initial_index)
            except Exception:
                can_hold = False
            if can_hold:
                try:
                    self._hold(last_action)
                    self._details["final_hold"] = True
                except Exception as error:
                    errors.append(error)
            else:
                self._details["final_hold"] = False
            self._last_action = None
            if planner is not None:
                self._details.update(planner.diagnostics)
                try:
                    planner.close()
                except Exception as error:
                    errors.append(error)
            self._active_planner = None
            if wall_start is not None:
                self._details.update(execution_wall_seconds=self._clock()-wall_start,
                    execution_sim_seconds=elapsed, control_interval_ms=_distribution(intervals),
                    joint_tracking_error_rad=_distribution(tracking),
                    commanded_updates=len(self.last_plan_trace.get("commanded", ())))
            if primary is not None:
                for error in errors:
                    primary.add_note(f"Global cleanup also failed: {error}")
            elif errors:
                if result is not None:
                    result.ok, result.message = False, f"Global cleanup failed: {errors[0]}"
                    self._details["failure_code"] = "cleanup_failure"
                    self._state = "failed"
                else:
                    raise errors[0]
            if result is not None:
                result.details.update(self.diagnostics)

    def close(self):
        self._assert_owner()
        if self._closed:
            return
        # Invalidate first: even a failed final hold must not leave a live plan.
        self._closed, self._epoch, self._state = True, self._epoch+1, "closed"
        errors = []
        try:
            if (self._last_action is not None and self._backend.is_ready
                    and self._backend.world.is_playing()):
                indices = np.asarray(self._last_action.joint_indices)
                actual = np.asarray(self._backend.robot.get_joint_positions())[indices]
                self._backend.robot.apply_action(self._action(joint_positions=actual,
                    joint_velocities=np.zeros(7), joint_indices=indices))
        except Exception as error:
            errors.append(error)
        if self._active_planner is not None:
            try:
                self._active_planner.close()
            except Exception as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Global shutdown also failed: {error}")
            raise errors[0]
