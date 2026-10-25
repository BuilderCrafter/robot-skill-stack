"""Research-only matched source comparison and independent execution veto.

The production global strategy still captures its selected world and executes
its selected ordinary Lula plan. This factory also plans the control at the
same unadvanced physics instant. Only a validated joint goal crosses from the
GT control to RGB-D; fixture geometry never crosses that boundary.
"""
from __future__ import annotations

import threading
import time
import numpy as np

from scripts.benchmark_global_motion_v4 import serial
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from robot_skill_stack.runtime.worker import BoundedWorker


def sample_trajectory(plan):
    duration = float(plan.duration)
    velocity = np.asarray(plan.velocity_limits,float)
    if (not np.isfinite(duration) or duration<0. or velocity.shape!=(7,)
            or not np.isfinite(velocity).all() or np.any(velocity<=0.)):
        raise PlanningFailure('research_veto_budget','Invalid trajectory sampling limits')
    step = min(1/240., .01/float(np.max(velocity)))
    count = max(2,int(np.ceil(duration/step))+1)
    if count > 12000:
        raise PlanningFailure('research_veto_budget', 'Independent trajectory sample budget exceeded')
    times = np.linspace(0., duration, count)
    positions = np.asarray([plan.trajectory.get_joint_targets(float(t))[0] for t in times])
    return times, positions


def bounded_evaluate(evaluator, payload, *, timeout=120.):
    """Existing bounded worker, with explicitly measured pre-command Kit wait.

    No simulation update is allowed here: the unchanged executor requires its
    actual starting state and physics time to survive synchronous planning.
    This experimental wait is not production source/planner latency.
    """
    started = time.perf_counter()
    worker = BoundedWorker(evaluator, name='v5-independent-physical-veto')
    try:
        if not worker.submit(payload, tag='physical-veto'):
            raise RuntimeError('Physical worker refused its only job')
        event = threading.Event()
        while time.perf_counter()-started < timeout:
            result = worker.poll()
            if result is not None:
                if result.error:
                    return dict(execution_allowed=False, status='VETO_EVALUATION_ERROR', error=result.error,
                        experimental_owner_wait_seconds=time.perf_counter()-started)
                return dict(result.value, experimental_owner_wait_seconds=time.perf_counter()-started,
                    worker_seconds=result.duration, queue_seconds=result.started_at-result.queued_at)
            event.wait(.005)
        return dict(execution_allowed=False, status='VETO_EVALUATION_TIMEOUT',
            experimental_owner_wait_seconds=time.perf_counter()-started)
    finally:
        worker.close()


class MatchedSourcePlanner:
    """Harness factory seam; never used by production configuration."""
    def __init__(self, backend, *, state, selected_source, capture_rgbd, gt_snapshot,
                 physical_context, seed=123456, max_iterations=10000, collision_model='enhanced'):
        from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import LulaGlobalPlanner
        self.backend, self.state, self.selected_source = backend, state, selected_source
        self.capture_rgbd, self.gt_snapshot = capture_rgbd, gt_snapshot
        self.context = physical_context
        self.options = dict(seed=seed, max_iterations=max_iterations, collision_model=collision_model)
        self.gt = LulaGlobalPlanner(backend, **self.options)
        self.rgbd = None
        self._last_diagnostics = {}

    @property
    def joint_names(self):
        return self.gt.joint_names

    @property
    def diagnostics(self):
        """The unchanged executor reads this protocol during final cleanup."""
        selected = self.state['paired_plans'].get(self.selected_source, {})
        return dict(selected.get('diagnostics', {}), **self._last_diagnostics)

    def _run(self, label, planner, target, start_q, snapshot, tolerances):
        record = dict(source=label, start_q=np.asarray(start_q).tolist(),
            target=serial(target), snapshot=serial(snapshot), planner_success=False)
        self.state['paired_plans'][label] = record
        began = time.perf_counter()
        try:
            plan = planner.plan(target, start_q, snapshot, **tolerances)
        except PlanningFailure as error:
            record.update(failure_code=error.code, failure_message=str(error),
                diagnostics=serial(error.details), planner_wall_seconds=time.perf_counter()-began)
            return None, error
        record.update(planner_success=True, joint_names=list(plan.joint_names), route=plan.route.tolist(),
            duration=plan.duration, diagnostics=serial(plan.diagnostics),
            limits={name:getattr(plan, name).tolist() for name in
                ('lower_limits','upper_limits','velocity_limits','acceleration_limits','jerk_limits')},
            planner_wall_seconds=time.perf_counter()-began)
        try:
            times, positions = sample_trajectory(plan)
            payload = dict(self.context, route_joint_names=list(plan.joint_names), route=plan.route.tolist(),
                trajectory_times=times.tolist(), trajectory_q=positions.tolist(), stop_on_collision=True)
            self.state['motion_payloads'][label] = payload
            return plan, None
        except PlanningFailure as error:
            self.state['physical_evaluation'][label] = dict(execution_allowed=False,
                status='VETO_RESEARCH_SAMPLING_BUDGET',failure_code=error.code,error=str(error))
            return plan, None

    def plan(self, target, start_q, selected_world, **tolerances):
        from scripts.global_detour_fixed_target_v4_3 import FixedJointTarget, FixedJointTargetPlanner
        from robot_skill_stack.integrations.isaac.manipulation.motion.lula_global import LulaGlobalPlanner
        from scripts.evaluate_global_rgbd_v5 import PhysicalMotionEvaluator
        before = (self.backend.world.current_time, self.backend.world.current_time_step_index)
        rgbd_world = selected_world if self.selected_source == 'frozen_rgbd' else None
        if rgbd_world is None:
            try:
                rgbd_world = self.capture_rgbd()
            except PlanningFailure as error:
                self.state['paired_plans']['frozen_rgbd'] = dict(planner_success=False,
                    failure_code=error.code, failure_message=str(error), stage='SOURCE', diagnostics=serial(error.details))
        gt_plan, gt_error = self._run('static', self.gt, target, start_q, self.gt_snapshot, tolerances)
        self.state['urdf_path'] = self.gt._config['urdf_path']
        fixed = None
        if gt_plan is not None and gt_plan.diagnostics.get('selected_ik_index') is not None:
            index = gt_plan.diagnostics['selected_ik_index']
            fixed = FixedJointTarget(tuple(gt_plan.joint_names),
                tuple(gt_plan.diagnostics['ik_attempts'][index]['ik_q']),
                tuple(target.position), tuple(target.orientation), end_effector_frame=self.backend.end_effector_frame)
            self.state['fixed_joint_target'] = fixed.as_dict()
        self.state['matched_joint_goal'] = fixed is not None
        rgbd_plan = rgbd_error = None
        if rgbd_world is not None:
            self.rgbd = (LulaGlobalPlanner(self.backend, **self.options) if fixed is None else
                FixedJointTargetPlanner(self.backend, fixed_target=fixed, **self.options))
            rgbd_plan, rgbd_error = self._run('frozen_rgbd', self.rgbd, target, start_q, rgbd_world, tolerances)
        if before != (self.backend.world.current_time, self.backend.world.current_time_step_index):
            raise PlanningFailure('research_pair_start_changed', 'Physics advanced during matched planning')
        self.state['matched_start_and_physics_instant'] = True
        # Independent evaluator uses only copied physical geometry and URDF.
        evaluator = PhysicalMotionEvaluator(self.state['robot_parts'], self.state['physical_obstacles'],
            self.gt._config['urdf_path'], max_queries=500000, max_samples=12000)
        for label in ('static','frozen_rgbd'):
            if label in self.state['motion_payloads']:
                veto = bounded_evaluate(evaluator, self.state['motion_payloads'][label])
                self.state['physical_evaluation'][label] = veto
        selected = gt_plan if self.selected_source == 'static' else rgbd_plan
        selected_error = gt_error if self.selected_source == 'static' else rgbd_error
        if selected is None:
            if selected_error is not None:
                raise selected_error
            raise PlanningFailure('planning_source_unavailable', 'RGB-D control source unavailable')
        self._last_diagnostics = dict(selected.diagnostics)
        if not self.state['physical_evaluation'][self.selected_source]['execution_allowed']:
            raise PlanningFailure('research_physical_veto',
                'Independent physical evaluation vetoed this planner-success route; no execution',
                dict(planner_success=True, physical_evaluation=self.state['physical_evaluation'][self.selected_source]))
        return selected

    def close(self):
        self.gt.close()
        if self.rgbd is not None:
            self.rgbd.close()
