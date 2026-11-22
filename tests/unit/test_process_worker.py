"""Real clean-interpreter transport, ownership, bounds and failure contracts."""
import os
from pathlib import Path
import tempfile
import threading
import time
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.runtime.process_worker import ProcessWorker
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.robot_filter import MeshInstance, RobotSnapshot, TriangleMesh
from robot_skill_stack.world.spatial_mapping.runtime import MappingInput, MappingProcessor


def _restore_probe(owner):
    return owner


class SerializationProbe:
    def __reduce__(self):
        return _restore_probe, (threading.get_ident(),)


class PureProcessor:
    def __init__(self):
        self.count = 0

    def compute(self, payload):
        import sys
        self.count += 1
        if isinstance(payload, dict) and payload.get('operation') == 'hold':
            Path(payload['started']).touch()
            while not Path(payload['release']).exists():
                time.sleep(.005)
        if isinstance(payload, dict) and payload.get('operation') == 'error':
            raise ValueError('deliberate pure computation failure')
        if isinstance(payload, dict) and payload.get('operation') == 'exit':
            os._exit(17)
        if isinstance(payload, dict) and payload.get('operation') == 'sdk':
            __import__('omni.usd')
        print('computation stdout is not protocol')
        return dict(pid=os.getpid(), count=self.count, payload=payload,
                    sdk_modules=[key for key in sys.modules if key.split('.')[0] in
                                 {'omni','pxr','carb','isaacsim','isaac'}])


class ProcessWorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = ProcessWorker(PureProcessor().compute)
        self.addCleanup(self.worker.close)

    def wait(self, predicate, timeout=10.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.005)
        self.fail('subprocess condition did not complete')

    def result(self):
        return self.wait(self.worker.poll)

    def test_embedded_kit_interpreter_is_rejected_before_thread_or_process_start(self):
        with patch('sys.executable','/isaac-sim/kit/kit'), \
                patch('subprocess.Popen') as launch, patch('threading.Thread') as thread:
            with self.assertRaisesRegex(RuntimeError,r'\./run_isaac_sim\.sh'):
                ProcessWorker(PureProcessor().compute)
            launch.assert_not_called()
            thread.assert_not_called()

    def test_clean_child_retains_processor_state_and_reports_pid_without_sdk_imports(self):
        self.assertIsNone(self.worker.health['pid'])
        self.worker.submit('first',tag=1)
        first=self.result()
        self.assertIsNone(first.error)
        self.assertNotEqual(first.value['pid'],os.getpid())
        self.assertFalse(first.value['sdk_modules'])
        self.worker.submit('second',tag=2)
        second=self.result()
        self.assertEqual(second.value['count'],2)
        self.assertEqual(second.value['pid'],first.value['pid'])
        self.assertEqual(second.tag,2)
        self.assertEqual(self.worker.health['execution'],'process')
        self.assertIsNone(self.worker.health['transport_error'])

    def test_serialization_and_startup_execute_only_on_background_thread(self):
        owner=threading.get_ident()
        self.worker.submit(SerializationProbe())
        completed=self.result()
        self.assertIsNone(completed.error)
        self.assertNotEqual(completed.value['payload'],owner)
        self.assertEqual(completed.value['payload'],self.worker._thread.ident)

    def test_large_payload_mapping_proxies_and_immutable_arrays_round_trip_exactly(self):
        values=np.arange(600000,dtype=np.float64).reshape(1000,600)
        immutable=np.frombuffer(values.tobytes(),dtype=values.dtype).reshape(values.shape)
        payload=MappingProxyType({'nested':MappingProxyType({'values':immutable})})
        self.worker.submit(payload)
        completed=self.result()
        self.assertIsNone(completed.error)
        actual=completed.value['payload']['nested']['values']
        np.testing.assert_array_equal(actual,values)
        with self.assertRaises(ValueError):actual.setflags(write=True)
        with self.assertRaises(TypeError):completed.value['payload']['other']=1

    def test_actual_mapping_input_mesh_snapshot_and_completed_grid_cross_transport(self):
        now=time.monotonic()
        frames=tuple(DepthFrame(key,(1.,.1),np.full((2,2),.3),np.eye(3),np.eye(4),now)
                     for key in ('A','B'))
        mesh=TriangleMesh(np.array([[0.,0.,.2],[.1,0.,.2],[0.,.1,.2]]),np.array([[0,1,2]]))
        snapshots={key:RobotSnapshot((MeshInstance(mesh,np.eye(4)),),.1,(1.,.1)) for key in ('A','B')}
        payload=MappingInput(frames=frames,robot_snapshots=snapshots,
                             exclusion_masks={key:np.zeros((2,2),bool) for key in ('A','B')})
        # Echo exercises all immutable input/BVH fields without depending on
        # whether this fixture's depth samples intersect its tiny mesh.
        self.worker.submit(payload)
        echoed=self.result().value['payload']
        for array in (echoed.frames[0].depth,echoed.exclusion_masks['A'],
                      echoed.robot_snapshots['A'].parts[0].mesh._lower,
                      echoed.robot_snapshots['A'].parts[0].world_from_local):
            with self.assertRaises(ValueError):array.setflags(write=True)
        settings=SpatialMappingConfig(source='rgbd',bounds_min=(-.1,-.1,0.),
            bounds_max=(1.,1.,.5),voxel_size=.1,stride=1,max_rays=8,
            max_frame_age=5.,stale_after=5.)
        worker=ProcessWorker(MappingProcessor(settings).compute)
        self.addCleanup(worker.close)
        worker.submit(payload)
        completed=self.wait(worker.poll)
        self.assertIsNone(completed.error)
        actual=completed.value
        self.assertTrue(actual.accepted)
        self.assertEqual(actual.input_timestamp,now)
        self.assertGreater(actual.snapshot.counts()['occupied'],0)
        for array in (actual.snapshot.states,actual.snapshot.probabilities,
                      actual.snapshot.observed,actual.snapshot.last_observed,actual.snapshot.expired):
            with self.assertRaises(ValueError):array.setflags(write=True)
        with self.assertRaises(TypeError):actual.diagnostics['changed']=True
        with self.assertRaises(TypeError):actual.snapshot.health['changed']=True

    def test_job_exception_recovers_in_same_child_without_resetting_state(self):
        self.worker.submit('first');first=self.result()
        self.worker.submit({'operation':'error'});failed=self.result()
        self.assertIn('deliberate pure computation failure',failed.error)
        self.assertIsNone(self.worker.health['transport_error'])
        self.worker.submit('third');third=self.result()
        self.assertEqual(third.value['pid'],first.value['pid'])
        self.assertEqual(third.value['count'],3)

    def test_sdk_import_is_rejected_as_a_recoverable_pure_job_failure(self):
        self.worker.submit({'operation':'sdk'})
        failed=self.result()
        self.assertIn('Simulator SDK import forbidden',failed.error)
        self.worker.submit('recovered')
        self.assertFalse(self.result().value['sdk_modules'])

    def hold(self, temporary):
        started,release=Path(temporary)/'started',Path(temporary)/'release'
        self.worker.submit(dict(operation='hold',started=str(started),release=str(release)),tag='held')
        self.wait(started.exists)
        return release

    def test_backpressure_keeps_only_latest_pending_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            release=self.hold(temporary)
            self.worker.submit('obsolete',tag='obsolete')
            self.worker.submit('latest',tag='latest')
            self.assertEqual(self.worker.health['replaced'],1)
            release.touch()
            completed=None
            deadline=time.monotonic()+10.
            while time.monotonic()<deadline:
                result=self.worker.poll()
                if result is not None and result.tag=='latest':
                    completed=result;break
                time.sleep(.005)
            self.assertIsNotNone(completed)
            self.assertEqual(completed.value['payload'],'latest')
            self.assertEqual(completed.value['count'],2)

    def test_invalidated_running_result_never_crosses_epoch_barrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            release=self.hold(temporary)
            self.worker.invalidate()
            self.worker.submit('fresh',tag='fresh')
            release.touch()
            completed=self.result()
            self.assertEqual(completed.tag,'fresh')
            self.assertGreaterEqual(self.worker.health['discarded'],1)

    def test_close_does_not_wait_for_child_computation_or_buffered_pipe_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.hold(temporary)
            child=self.worker._child
            started=time.monotonic();self.worker.close()
            self.assertLess(time.monotonic()-started,.25)
            self.assertFalse(self.worker.submit('late'))
            self.wait(lambda: child.poll() is not None)
            self.wait(lambda:not self.worker._thread.is_alive())
            self.assertIsNone(self.worker.poll())
            self.worker.close()

    def test_transport_death_is_terminal_and_does_not_silently_reset_processor(self):
        self.worker.submit({'operation':'exit'})
        failed=self.result()
        self.assertIn('transport failed',failed.error)
        pid=self.worker.health['pid']
        self.worker.submit('cannot reset map')
        repeated=self.result()
        self.assertIn('requires reinitialization',repeated.error)
        self.assertEqual(self.worker.health['pid'],pid)
        self.assertIsNotNone(self.worker.health['transport_error'])


if __name__=='__main__':
    unittest.main()
