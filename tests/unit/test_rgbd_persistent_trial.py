"""Research sequence never replays direct setup or advances after a failed leg."""
import unittest
from types import SimpleNamespace as NS

from research.rgbd_box_geometry.persistent_trial import followups
from scripts import benchmark_global_rgbd_v5 as bench


class PersistentTrialTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,*,initial='success',veto=True,return_status='success'):
        observer=NS(actions=[],operations=[{}])
        view=NS(generation=7,revision=22,timestamp=1.)
        mapping=NS(snapshot=lambda **kw:view,health={'generation':7,'published':22})
        bundle=NS(spatial_mapping=mapping,backend=NS(motion_strategy=NS(diagnostics={'closed':True})),
            observation_runtime=NS(diagnostics={'robot_filter_ready':True}))
        state=dict(session_requests=[],request_started=1.,research_record={},
            selected_result={'status':initial},physical_evaluation={'reconstructed':{'execution_allowed':veto}},
            paired_plans={},motion_payloads={},mapping_inputs=[],mapping_publications=[],mapping_submissions=[])
        calls=[];settles=[]
        async def settle():settles.append(True)
        async def motion(obs,label,name,*,target):
            calls.append((label,name,target))
            self.assertEqual(state['physical_evaluation'],{})
            self.assertNotIn('frozen_world',state)
            state['physical_evaluation']={'reconstructed':{'execution_allowed':True}}
            state['research_record']['identity']=[7,len(calls),10.+len(calls)]
            obs.operations.append({})
            return {'status':return_status if len(calls)==1 else 'success'}
        await followups(bundle,state,observer,settle=settle,motion=motion)
        return state,calls,settles

    async def test_three_ordinary_requests_keep_distinct_receipts(self):
        state,calls,settles=await self.exercise()
        self.assertEqual([row[:2] for row in calls],[('return_start','move_to_pose'),('outward_repeat','move_to_pose')])
        self.assertEqual(tuple(calls[0][2].position),bench.protocol.START)
        self.assertEqual(tuple(calls[1][2].position),bench.protocol.GOAL)
        self.assertEqual(len(settles),2)
        self.assertEqual(len(state['session_requests']),3)
        self.assertIsNot(state['session_requests'][1]['research_record'],state['session_requests'][2]['research_record'])
        self.assertEqual([r['operation_index'] for r in state['session_requests']],[0,1,2])

    async def test_failed_or_unvalidated_first_leg_prevents_return(self):
        for options in ({'initial':'failure'},{'veto':False}):
            state,calls,settles=await self.exercise(**options)
            self.assertEqual(calls,[]);self.assertEqual(settles,[])
            self.assertEqual(len(state['session_requests']),1)
            self.assertIn('Previous leg',state['session_stop_reason'])

    async def test_failed_return_prevents_dependent_outward_repeat(self):
        state,calls,settles=await self.exercise(return_status='failure')
        self.assertEqual(len(calls),1);self.assertEqual(len(settles),1)
        self.assertEqual(len(state['session_requests']),2)


if __name__=='__main__':unittest.main()
