"""Verify observational hooks and guaranteed pre-GOAL termination without SDK."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import AsyncMock, Mock, patch

from . import mapping_diagnostic as d


def fixture(error=None):
    filt = NS(source_error=error, _model_error=error, _capture_error=None, _closed=False,
        _generation=0, _history=[], _pose_capture_calls=3, diagnostics={'state': 'prepared'})
    source = NS(health={'published': 23, 'fresh': True, 'reason': 'acquisition_failure' if error else None},
                clock=lambda:10., diagnostics={'source_frames': ()})
    runtime = NS(robot_filter=filt, spatial_mapping=source, _owner=threading.get_ident(),
        config=NS(world=NS(provider='multi_view'), multi_view=NS(camera_prim_paths=('a','b'))),
        _mapping_settings=NS(source='rgbd', camera_prim_paths=('a','b')),
        world=NS(current_time=6., current_time_step_index=360, is_playing=lambda:True),
        clock=lambda:10., _playing=True, _closed=False, _state='running', _error=None,
        _generation=0, _semantic_enabled=True, _mapping_enabled=True,
        _semantic_due=9.9, _mapping_due=9.8, _filter_needs_resume=False,
        _mapping_invalidation_error=error, _semantic={'attempts':25})
    return NS(observation_runtime=runtime, spatial_mapping=source)


class DiagnosticTests(unittest.TestCase):
    def test_schema_audit_is_deterministic_and_detects_api_addition(self):
        bundle=fixture();timeline=d.Timeline();timeline.runtime=bundle.observation_runtime
        schemas=[]
        prim=NS(GetPath=lambda:'/Franka/joint',GetTypeName=lambda:'PhysicsPrismaticJoint',GetAppliedSchemas=lambda:schemas)
        timeline.runtime.stage=NS(GetPrimAtPath=lambda path:prim)
        timeline.runtime.robot_filter.robot_prim_path='/Franka'
        pxr=ModuleType('pxr')
        pxr.Usd=NS(PrimRange=lambda *a:[prim],TraverseInstanceProxies=lambda p:p,PrimAllPrimsPredicate=object())
        with patch.dict(sys.modules,{'pxr':pxr}):
            before=timeline.robot_schemas()
            self.assertTrue(before['available'])
            self.assertEqual(before,timeline.robot_schemas())
            self.assertEqual(schemas,[])
            schemas.append('PhysicsDriveAPI:linear')
            self.assertNotEqual(before['sha256'],timeline.robot_schemas()['sha256'])

    def test_sampling_is_bounded_and_only_reads_owner_state(self):
        bundle=fixture();timeline=d.Timeline(clock=lambda:10.);timeline.runtime=bundle.observation_runtime
        original=dict(vars(bundle.observation_runtime))
        for _ in range(300):timeline.record('forced',force=True)
        self.assertEqual(len(timeline.events),64);self.assertEqual(timeline.event_count,300)
        row=timeline.events[-1]
        self.assertTrue(row['mapping_source_ready']);self.assertTrue(row['mapping_due']);self.assertTrue(row['overlap'])
        self.assertEqual(vars(bundle.observation_runtime),original)
        bundle.observation_runtime._owner=-1
        bundle.observation_runtime.world.is_playing=Mock(side_effect=AssertionError('foreign SDK read'))
        timeline.record('foreign',force=True)
        self.assertFalse(timeline.events[-1]['owner_thread'])
        bundle.observation_runtime.world.is_playing.assert_not_called()

    def test_hooks_preserve_calls_exceptions_and_restore(self):
        bundle=fixture();timeline=d.Timeline();calls=[]
        class Runtime:
            def tick(self):calls.append('tick');raise ValueError('original failure')
        class Filter:
            source_error=None
            def _model_changed(self,notice,sender):self.source_error='model changed'
        original_tick=Runtime.tick;original_change=Filter._model_changed
        observer=Mock(return_value='observer');physical=Mock(side_effect=ValueError('capture'))
        with patch.object(d.bench.controlled,'make_observer',observer),patch.object(d.bench.acceptance,'fresh_physical_capture',physical):
            with timeline.installed(Runtime,Filter):
                with self.assertRaisesRegex(ValueError,'original failure'):Runtime().tick()
                with self.assertRaisesRegex(ValueError,'capture'):d.bench.acceptance.fresh_physical_capture(bundle)
                notice=NS(GetResyncedPaths=lambda:['/Franka'],GetChangedInfoOnlyPaths=lambda:[])
                Filter()._model_changed(notice,None)
            self.assertIs(d.bench.controlled.make_observer,observer)
            self.assertIs(d.bench.acceptance.fresh_physical_capture,physical)
        self.assertIs(Runtime.tick,original_tick);self.assertIs(Filter._model_changed,original_change)
        self.assertEqual(calls,['tick'])
        self.assertTrue(any(r['event']=='after_physical_capture' for r in timeline.events))
        with TemporaryDirectory() as tmp:
            timeline.save(Path(tmp)/'diagnostic.json')
            self.assertTrue(json.loads((Path(tmp)/'diagnostic.json').read_text())['no_selected_goal'])


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_outcome_stops_before_selected_goal(self):
        omni,kit,app=(ModuleType(n) for n in ('omni','omni.kit','omni.kit.app'))
        omni.kit=kit;kit.app=app;update=AsyncMock();app.get_app=lambda:NS(next_update_async=update)
        for health,clock,error,outcome in (
            ({'published':23,'reason':'acquisition_failure'},[0,0,0,1,1,1],'bad model','robot_filter_source_barrier'),
            ({'published':24,'fresh':True},[0,0,0,1,1,1,1],None,'fresh_matching_publication'),
            ({'published':25,'fresh':True},[0,0,0,1,1,1],None,'first_publication_invalid'),
            ({'published':24,'fresh':True},[0,0,0,5.1,5.1,5.1],None,'five_second_readiness_expired')):
            with self.subTest(outcome=outcome):
                bundle=fixture(error);timeline=d.Timeline(clock=Mock(side_effect=clock))
                with patch.dict(sys.modules,{'omni':omni,'omni.kit':kit,'omni.kit.app':app}), \
                     patch.object(d.bench,'retain_mapping_publication',return_value=health) as retain, \
                     patch.object(d.bridge,'associate',return_value=(NS(generation=0,revision=30,timestamp=9.),NS(generation=0))) as associate:
                    with self.assertRaises(d.DiagnosticStop):await timeline.ready(bundle,{'mapping_inputs':[]})
                    self.assertEqual(timeline.outcome,outcome)
                    self.assertFalse(any(r['event']=='instrumentation_error' for r in timeline.events))
                    if outcome=='fresh_matching_publication':associate.assert_called_once()
                    else:associate.assert_not_called()
                    if outcome=='five_second_readiness_expired':retain.assert_not_called()


if __name__=='__main__':unittest.main()
