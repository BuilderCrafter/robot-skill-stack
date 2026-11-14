"""Experimental repeated ordinary RGB-D requests in one unchanged native runtime.

Every leg reuses AutonomousPlanner's independent pre-execution physical veto.
No direct return, setup replay, fixture edit, joint teleport or source retry.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

from scripts import benchmark_global_rgbd_v5 as bench
from .autonomous_trial import AutonomousPlanner, fixed_request_delay, classify
from . import reference


async def followups(bundle,state,observer,*,settle=None,motion=None):
    """Stop dependent legs on failure; retain each exact source before reuse."""
    from robot_skill_stack.common.types import Pose
    if settle is None:
        import omni.kit.app
        async def settle():
            await fixed_request_delay(8.,omni.kit.app.get_app().next_update_async)
    motion=motion or bench.acceptance.motion
    receipts=state['session_requests']
    def retain(label):
        started=state['request_started']
        commands=[row for row in observer.actions if row.get('at',-1.)>=started]
        record=state['research_record']
        record.update(label=label,request_started=started,request_finished=time.monotonic(),
            request_to_first_command_seconds=commands[0]['at']-started if commands else None,
            strategy_after=bench.baseline.serial(bundle.backend.motion_strategy.diagnostics),
            mapping_after=bench.mapping_request_state(bundle.spatial_mapping,time.monotonic()),
            runtime_after=bench.baseline.serial(bundle.observation_runtime.diagnostics))
        retained={k:state.get(k) for k in ('capture','robot_parts','physical_obstacles',
            'urdf_path','frozen_world','capture_view','selected_mapping_input','mapping_inputs',
            'mapping_publications','mapping_submissions','motion_payloads','physical_evaluation',
            'camera_acquisition','paired_plans','selected_result','research_record')}
        retained['operation_index']=len(observer.operations)-1
        receipts.append(retained)
    retain('outward')
    for label,target in [('return_start',bench.protocol.START),('outward_repeat',bench.protocol.GOAL)]:
        if (state['selected_result'].get('status')!='success' or
                state['physical_evaluation'].get('reconstructed',{}).get('execution_allowed') is not True):
            state['session_stop_reason']='Previous leg did not complete a physically validated motion'
            break
        await settle()
        for key in ('frozen_world','capture_view','selected_mapping_input'):
            state.pop(key,None)
        for key in ('paired_plans','motion_payloads','physical_evaluation'):
            state[key]={}
        state['research_record']=dict(production_goal=True,production_capture_unmodified=True)
        state['request_started']=time.monotonic()
        state['research_record']['ordinary_request']=bench.mapping_request_state(
            bundle.spatial_mapping,state['request_started'])
        result=await motion(observer,label,'move_to_pose',target=Pose(target,bench.protocol.ORIENTATION))
        state['selected_result']=bench.baseline.serial(result)
        if state.get('frozen_world') is None:
            state['capture_view']=bundle.spatial_mapping.snapshot()
        retain(label)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    requests=[];holder={}
    original=bench.protocol.write_profile
    def profile(path,source,budget=64):
        original(path,source,budget)
        path.write_text(bench.protocol._set_table_values(path.read_text(),'motion',
            {'global_rgbd_representation':'reconstructed'}))
        return path
    async def ready(bundle,state):
        holder['state']=state
        # No readiness/qualification predicate, extra publication wait or delay.
        bench.retain_mapping_publication(state,bundle.spatial_mapping)
    native=SimpleNamespace(scenario='partial',execution_condition='frozen_rgbd',budget=64,
        headless=True,artifacts_dir=str(args.output),result=str(args.output/'result.json'))
    try:
        with patch.object(bench.protocol,'write_profile',profile):
            bench.execute(native,app,research=dict(autonomous_runtime=True,
                ordinary_request_timing=True,research_prepare=ready,research_followups=followups,
                research_planner_factory=AutonomousPlanner,research_record=dict(production_goal=True),
                session_requests=requests))
        result=json.loads((args.output/'result.json').read_text());summaries=[]
        for index,state in enumerate(requests):
            output=args.output/f'request-{index+1}';output.mkdir(exist_ok=False)
            # Shared trace/reference includes every operation; source archives
            # remain distinct and retain the original per-request timestamps.
            bench.save_capture(SimpleNamespace(artifacts_dir=str(output)),state)
            record=state['research_record']
            record['inferred']=bench.baseline.serial(record.pop('_selected_inferred',None))
            summary=dict(index=index+1,operation_index=state['operation_index'],
                selected_result=state['selected_result'],record=record,
                paired_plans=state['paired_plans'],
                physical_evaluation={k:{a:b for a,b in v.items() if a!='per_sample'}
                    for k,v in state['physical_evaluation'].items()})
            summary['category']=classify({'metrics':{'selected_result':state['selected_result']}})
            reference.save(output/'request.json',summary);summaries.append(summary)
        reference.save(args.output/'session.json',dict(requests=summaries,
            stop_reason=holder.get('state',{}).get('session_stop_reason'),
            session_measurement_status=result['status'],
            requested_sequence=['outward','return_start','outward_repeat'],settle_seconds=8.,
            one_runtime=True,no_reset_or_setup_replay=True))
    finally:
        app.close()


if __name__=='__main__':
    main()
