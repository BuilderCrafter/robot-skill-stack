"""One authorized nominal-partial native proof; never a production entry point."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import tomllib

import numpy as np

from . import reference as ref,rrt_bridge as bridge
from scripts import benchmark_global_rgbd_v5 as bench
from scripts.global_rgbd_v5_pair import MatchedSourcePlanner,bounded_evaluate
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


async def prepare(bundle,state):
    import omni.kit.app
    source=bundle.spatial_mapping;first=source.health['published']
    state['research_record']['readiness_before']=dict(publication=bridge.identity(source.snapshot()),health=dict(source.health))
    started=time.monotonic()
    try:
        while time.monotonic()-started<5.:
            await omni.kit.app.get_app().next_update_async()
            if time.monotonic()-started>=5.:break
            health=bench.retain_mapping_publication(state,source)
            if health['published']>first:
                if health['published']!=first+1 or health.get('fresh') is not True:
                    bridge.fail('readiness','First new publication is not fresh or unique')
                view,payload=bridge.associate(source,state['mapping_inputs'])
                state['research_ready']=bridge.identity(view)
                state['selected_mapping_input']=payload
                state['research_record']['readiness']=dict(publication=bridge.identity(view),
                    wait_s=time.monotonic()-started,age_s=source.clock()-view.timestamp)
                return
        bridge.fail('readiness','No accepted complete publication within five seconds')
    finally:
        observation=getattr(bundle,'observation_runtime',None)
        filt=getattr(observation,'robot_filter',None)
        state['research_record']['readiness_after']=dict(publication=bridge.identity(source.snapshot()),
            health=dict(source.health),source_frames=bench.baseline.serial(source.diagnostics.get('source_frames',())),
            observation=None if observation is None else bench.baseline.serial(observation.diagnostics),
            robot_filter_error=None if filt is None else filt.source_error)


def capture(backend,source,state,base_capture):
    return bridge.capture(backend,source,state['mapping_inputs'],state['research_ready'],
                          state['research_scene'],state['research_record'],base_capture)


class Comparison:
    """Record one extra native counterplan; delegate every actual planner/check."""
    def __init__(self,backend,**options):
        self.delegate=MatchedSourcePlanner(backend,**options)
        self.backend,self.state=backend,options['state']
        self.empty=None

    @property
    def joint_names(self):return self.delegate.joint_names

    @property
    def diagnostics(self):return self.delegate.diagnostics

    def plan(self,target,start_q,world,**tolerances):
        from scripts.global_detour_fixed_target_v4_3 import FixedJointTarget,FixedJointTargetPlanner
        from scripts.evaluate_global_rgbd_v5 import PhysicalMotionEvaluator
        from scripts.analyze_global_detour_v4_3 import distances_to_polyline
        before=(self.backend.world.current_time,self.backend.world.current_time_step_index)
        selected=self.delegate.plan(target,start_q,world,**tolerances)
        if not self.state.get('matched_joint_goal'):bridge.fail('unmatched_goal','No validated common joint target')
        fixed=FixedJointTarget.from_dict(self.state['fixed_joint_target'])
        self.empty=FixedJointTargetPlanner(self.backend,fixed_target=fixed,**self.delegate.options)
        support=replace(world,obstacles=tuple(o for o in world.obstacles if o.source_kind=='Plane'),
                        source_metadata={'source_type':'research_unobstructed_counterfactual'})
        plan,error=self.delegate._run('unobstructed',self.empty,target,start_q,support,tolerances)
        if plan is None:raise error
        if 'unobstructed' not in self.state['motion_payloads']:bridge.fail('control_unmeasurable','Counterplan sampling failed')
        evaluator=PhysicalMotionEvaluator(self.state['robot_parts'],self.state['physical_obstacles'],
            self.delegate.gt._config['urdf_path'],max_queries=500000,max_samples=12000)
        evaluation=bounded_evaluate(evaluator,self.state['motion_payloads']['unobstructed'])
        self.state['physical_evaluation']['unobstructed']=evaluation
        same=bool(np.allclose(plan.route[0],selected.route[0],atol=1e-12,rtol=0.)
                  and np.allclose(plan.route[-1],selected.route[-1],atol=1e-12,rtol=0.))
        deviation=float(distances_to_polyline(selected.route,np.array([start_q,fixed.q])).max())
        detour=dict(same_start_and_goal=same,selected_waypoints=len(selected.route),
                    direct_segment_deviation_rad=deviation,
                    original_obstruction=evaluation.get('status')=='VETO_PREDICTED_PHYSICAL_COLLISION')
        self.state['research_record']['detour']=detour
        if before!=(self.backend.world.current_time,self.backend.world.current_time_step_index):
            bridge.fail('start_changed','Physics advanced during research comparison')
        if not (same and len(selected.route)>=3 and deviation>.05 and detour['original_obstruction']):
            bridge.fail('no_detour','Meaningful independently obstructed-reference detour not established')
        return selected

    def close(self):
        try:
            self.delegate.close()
        finally:
            if self.empty is not None:self.empty.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    native=SimpleNamespace(scenario='partial',execution_condition='frozen_rgbd',budget=64,
        headless=True,artifacts_dir=str(args.output),result=str(args.output/'result.json'))
    record=dict(scenario='partial',source='frozen_rgbd_reconstructed',seed=123456,iterations=10000,
                timeout_s=600,no_gt_reconstruction_input=True)
    hashes={p.name:ref.digest(p) for p in Path(__file__).parent.glob('*.py')}
    ref.save(args.output/'research-protocol.json',dict(record,source_sha256=hashes,
        native_protocol=bench.protocol.PROTOCOL,controls=['static','frozen_rgbd','unobstructed']))
    app=None;passed=False
    try:
        if np.__version__!='1.26.4':raise RuntimeError('Native NumPy must remain 1.26.4')
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True})
        async def ready(bundle,state):
            cfg=tomllib.loads((args.output/'runtime_profile.toml').read_text())['scene']
            state['research_scene']=SimpleNamespace(**{k:cfg[k] for k in ('workspace_min','workspace_max','support_plane_z')})
            await prepare(bundle,state)
        passed=bench.execute(native,app,research=dict(research_prepare=ready,research_capture=capture,
                              research_planner_factory=Comparison,research_record=record))
        record['measurement_pass']=passed
    except BaseException as e:
        record['entry_failure']=repr(e)
        raise
    finally:
        record['research_sources_unchanged']=all(ref.digest(Path(__file__).parent/n)==h for n,h in hashes.items())
        ref.save(args.output/'research.json',record)
        if app is not None:app.close()
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
