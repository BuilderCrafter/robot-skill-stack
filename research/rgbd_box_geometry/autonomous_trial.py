"""Normal-runtime RGB-D trial; selected IK/world remain production-owned.

The only planning instrumentation observes native plans and independently vetoes
unsafe execution. Matched control goals come FROM the selected ordinary plan.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import benchmark_global_rgbd_v5 as bench
from scripts.global_rgbd_v5_pair import MatchedSourcePlanner, bounded_evaluate
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure
from .native_trial import prepare
from . import reference as frozen_reference


async def fixed_request_delay(delay, next_update, clock=time.monotonic):
    """Fixed timing only: no observation/qualification predicate is consulted."""
    started=clock()
    while clock()-started < delay:
        await next_update()
    return dict(preparation_completed_at=started,requested_delay_s=delay,
                actual_delay_s=clock()-started,publication_wait=False)


class AutonomousPlanner:
    _run = MatchedSourcePlanner._run

    def __init__(self, backend, *, state, selected_source, capture_rgbd, gt_snapshot,
                 physical_context, production_factory, **options):
        self.backend, self.state, self.context = backend, state, physical_context
        self.gt_snapshot, self.options = gt_snapshot, options
        self.primary = production_factory(backend, **options)
        self.controls, self._last_diagnostics = [], {}

    @property
    def joint_names(self):
        return self.primary.joint_names

    @property
    def diagnostics(self):
        return self._last_diagnostics or self.primary.diagnostics

    def plan(self, target, start_q, world, **tolerances):
        from scripts.global_detour_fixed_target_v4_3 import FixedJointTarget, FixedJointTargetPlanner
        from scripts.evaluate_global_rgbd_v5 import PhysicalMotionEvaluator
        from scripts.analyze_global_detour_v4_3 import distances_to_polyline
        from robot_skill_stack.integrations.isaac.manipulation.motion.frozen_rgbd import planning_world_from_frozen
        before = (self.backend.world.current_time, self.backend.world.current_time_step_index)
        strategy = self.backend.motion_strategy
        frozen = strategy.last_plan_trace['frozen_rgbd']
        self.state['research_record']['planning_entered_at']=time.monotonic()
        self.state['research_record']['_selected_inferred']=frozen.view.inferred
        if self.state.get('ordinary_request_timing'):
            self.context=bench.physical_context_from_observer(self.backend,self.state['observer'],self.state['capture'])
        self.state['frozen_world'] = frozen
        self.state['selected_mapping_input'] = next((p for p in self.state['mapping_inputs']
            if p.generation == frozen.generation and min(f.acquired_at for f in p.frames) == frozen.timestamp), None)
        self.state['autonomous_primary'] = dict(planner=type(self.primary).__name__,
            goal_source='ordinary selected Lula IK; no GT goal input', capture_injection=False)
        self.state['research_record']['primary'] = self.state['autonomous_primary']
        selected, error = self._run('reconstructed', self.primary, target, start_q, world, tolerances)
        if selected is None:
            raise error
        self._last_diagnostics = dict(selected.diagnostics)
        self.state['research_record']['selected_uses_fixed_joint_target'] = 'research_fixed_joint_target' in selected.diagnostics
        fixed = FixedJointTarget(tuple(selected.joint_names), tuple(selected.route[-1]),
            tuple(target.position), tuple(target.orientation), end_effector_frame=self.backend.end_effector_frame)
        self.state['fixed_joint_target'] = fixed.as_dict()
        self.state['matched_joint_goal'] = True
        support = replace(world, obstacles=tuple(o for o in world.obstacles if o.source_kind=='Plane'),
                          source_metadata={'source_type':'research_support_only_control'})
        occupied = planning_world_from_frozen(frozen, support)
        for label, snapshot in [('static',self.gt_snapshot),('occupied',occupied),('unobstructed',support)]:
            planner = FixedJointTargetPlanner(self.backend, fixed_target=fixed, **self.options)
            self.controls.append(planner)
            self._run(label, planner, target, start_q, snapshot, tolerances)
        self.state['urdf_path'] = self.primary._config['urdf_path']
        evaluator = PhysicalMotionEvaluator(self.state['robot_parts'],self.state['physical_obstacles'],
            self.primary._config['urdf_path'], max_queries=500000,max_samples=12000)
        for label,payload in self.state['motion_payloads'].items():
            self.state['physical_evaluation'][label] = bounded_evaluate(evaluator,payload)
        if before != (self.backend.world.current_time,self.backend.world.current_time_step_index):
            raise PlanningFailure('research_start_changed','Physics advanced during controlled evaluation')
        self.state['matched_start_and_physics_instant'] = True
        other = self.state['paired_plans'].get('unobstructed',{})
        same = bool(other.get('planner_success') and np.array_equal(other['route'][0],selected.route[0])
                    and np.array_equal(other['route'][-1],selected.route[-1]))
        deviation = float(distances_to_polyline(selected.route,np.array([start_q,fixed.q])).max())
        obstructed = self.state['physical_evaluation'].get('unobstructed',{}).get('status')=='VETO_PREDICTED_PHYSICAL_COLLISION'
        selected_evaluation = self.state['physical_evaluation'].get('reconstructed',{})
        self.state['research_record']['detour'] = dict(same_endpoints=same,waypoints=len(selected.route),
            deviation_rad=deviation,unobstructed_physically_obstructed=obstructed,
            genuine=bool(same and len(selected.route)>=3 and deviation>.05 and obstructed
                         and selected_evaluation.get('execution_allowed')))
        if selected_evaluation.get('execution_allowed') is not True:
            raise PlanningFailure('research_physical_veto','Selected ordinary RGB-D plan failed independent physical evaluation',
                                  dict(physical_evaluation=selected_evaluation))
        return selected

    def close(self):
        error = None
        for planner in [self.primary,*self.controls]:
            try:
                planner.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error


def classify(result):
    selected = result['metrics'].get('selected_result') or {}
    code = selected.get('details',{}).get('failure_code') or selected.get('failure_code') or ''
    if selected.get('status') == 'success':
        return 'PASS_PLANNING_AND_EXECUTION'
    if code == 'base_geometry_rejected':
        return 'EXPECTED_SAFE_REJECTION'
    if code == 'research_physical_veto':
        return 'PHYSICAL_VETO'
    if code == 'planning_source_reconstruction':
        message=selected.get('message','')
        if any(word in message for word in ('identity','exposure','calibration','No inferred geometry')):
            return 'SOURCE_FAILURE'
        return 'COLLISION_WORLD_FAILURE' if '64 camera' in message else 'RECONSTRUCTION_FAILURE'
    if code.startswith('planning_source_') or not selected:
        return 'SOURCE_FAILURE'
    if code.startswith(('tracking','execution','final_pose','terminal')):
        return 'EXECUTION_FAILURE'
    return 'PLANNING_FAILURE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=bench.protocol.SCENES+bench.protocol.DIAGNOSTIC_SCENES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--request-delay',type=float,choices=(0.,.3,.6),default=None,
                        help='Fixed ungated delay after physical-reference preparation; omit for historical research readiness')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True});record=dict(scenario=args.scenario,production_goal=True)
    inferred_product=None
    original_profile=bench.protocol.write_profile
    original_signature=bench.shared._world_signature
    def signature(stage):
        started=time.monotonic()
        value=original_signature(stage)
        record.setdefault('test_world_signature_times',[]).append(dict(
            started_at=started,finished_at=time.monotonic()))
        return value
    def profile(path,source,budget=64):
        original_profile(path,source,budget)
        path.write_text(bench.protocol._set_table_values(path.read_text(),'motion',{'global_rgbd_representation':'reconstructed'}))
        return path
    async def ready(bundle,state):
        nonlocal inferred_product
        if args.request_delay is None:
            await prepare(bundle,state)
        else:
            import omni.kit.app
            record['request_timing']=await fixed_request_delay(args.request_delay,
                omni.kit.app.get_app().next_update_async)
            bench.retain_mapping_publication(state,bundle.spatial_mapping)
        view=bundle.spatial_mapping.snapshot();state['capture_view']=view
        state['selected_mapping_input']=next((p for p in state['mapping_inputs']
            if p.generation==view.generation and min(f.acquired_at for f in p.frames)==view.timestamp),None)
        inferred_product=view.inferred
        record['normal_configuration']=bench.baseline.serial(bundle.config.motion)
        record['readiness_return_at']=time.monotonic()
    native=SimpleNamespace(scenario=args.scenario,execution_condition='frozen_rgbd',budget=64,
        headless=True,artifacts_dir=str(args.output),result=str(args.output/'result.json'))
    try:
        with patch.object(bench.protocol,'write_profile',profile), \
             patch.object(bench.shared,'_world_signature',signature):
            bench.execute(native,app,research=dict(autonomous_runtime=True,research_prepare=ready,
                ordinary_request_timing=args.request_delay is not None,
                research_planner_factory=AutonomousPlanner,research_record=record))
        result=json.loads((args.output/'result.json').read_text())
        record['category']=classify(result)
    finally:
        record['inferred']=bench.baseline.serial(record.pop('_selected_inferred',inferred_product))
        frozen_reference.save(args.output/'autonomous.json',record)
        app.close()


if __name__=='__main__':
    main()
