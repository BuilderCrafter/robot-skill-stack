"""Full immutable buffer reuse with exact values and real process isolation."""
from dataclasses import is_dataclass
import hashlib,io,json,os,pickle,time
from pathlib import Path
import unittest
from types import MappingProxyType

import numpy as np

from robot_skill_stack.runtime.process_worker import ProcessWorker,_TransportPickler,_restore_readonly_array
from tests.unit import test_process_worker as process_tests


class CopyingPickler(_TransportPickler):
    """Frozen original reducer: copying is the only transport difference."""
    def reducer_override(self,value):
        if isinstance(value,np.ndarray) and not value.flags.writeable:
            if value.dtype.hasobject:raise TypeError('numeric arrays required')
            return _restore_readonly_array,(value.tobytes(),value.dtype,value.shape)
        return super().reducer_override(value)


def encoded(value,cls=_TransportPickler):
    stream=io.BytesIO();cls(stream,protocol=5).dump(value);return stream.getvalue()


def exact(test,a,b):
    if isinstance(a,np.ndarray):
        test.assertEqual((a.dtype,a.shape),(b.dtype,b.shape));test.assertEqual(a.tobytes(),b.tobytes())
        test.assertFalse(b.flags.writeable)
        with test.assertRaises(ValueError):b.setflags(write=True)
    elif is_dataclass(a):
        test.assertIs(type(a),type(b))
        test.assertEqual(vars(a).keys(),vars(b).keys())
        for key in vars(a):exact(test,getattr(a,key),getattr(b,key))
    elif hasattr(a,'keys'):
        test.assertEqual(a.keys(),b.keys())
        for key in a:exact(test,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):exact(test,x,y)
    elif isinstance(a,float) and np.isnan(a):test.assertTrue(np.isnan(b))
    else:test.assertEqual(a,b)


class BufferTests(unittest.TestCase):
    def test_complete_immutable_buffer_is_reused_without_a_new_copy(self):
        data=np.arange(600,dtype=np.float64).tobytes()
        array=np.frombuffer(data,dtype=np.float64).reshape(20,30)
        _,args=_TransportPickler(io.BytesIO(),protocol=5).reducer_override(array)
        self.assertIs(args[0],data)
        exact(self,array,pickle.loads(encoded(array)))

    def test_views_dtypes_empty_nonfinite_and_extreme_values_round_trip_exactly(self):
        raw=np.arange(48,dtype='>f8').tobytes();base=np.frombuffer(raw,dtype='>f8').reshape(6,8)
        values=[base,base.T,base[::-1],base[:,::2],base[1:4],base.view(np.uint8),
                np.frombuffer(b'',dtype='f8').reshape(0,3),
                np.frombuffer(np.array([np.nan,np.inf,-np.inf,-0.,1e300,-1e300]).tobytes(),dtype='f8'),
                np.frombuffer(np.array([np.iinfo('int64').min,np.iinfo('int64').max]).tobytes(),dtype='int64')]
        for array in values:
            with self.subTest(shape=array.shape,strides=array.strides):exact(self,array,pickle.loads(encoded(array)))
        for array in (base.T,base[::-1],base[:,::2],base[1:4]):
            _,args=_TransportPickler(io.BytesIO(),protocol=5).reducer_override(array)
            self.assertIsNot(args[0],raw)

    def test_mutable_backing_cannot_become_shared_readonly_publication(self):
        storage=bytearray(np.arange(12,dtype=np.float64).tobytes())
        array=np.frombuffer(storage,dtype=np.float64);array.setflags(write=False)
        saved=encoded(array);actual=pickle.loads(saved)
        expected=actual.copy();np.frombuffer(storage,dtype=np.float64)[:]=-999.
        np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(ValueError):actual.setflags(write=True)
        owner=np.arange(8.);owner.setflags(write=False);saved=encoded(owner)
        owner.setflags(write=True);owner[:]=-42
        np.testing.assert_array_equal(pickle.loads(saved),np.arange(8.))

    def test_object_dtype_stays_rejected_and_nested_aliases_stay_immutable(self):
        objects=np.array([object()],dtype=object);objects.setflags(write=False)
        with self.assertRaises(TypeError):encoded(objects)
        data=np.arange(24,dtype='i8').tobytes();a=np.frombuffer(data,dtype='i8')
        payload=MappingProxyType({'flat':a,'matrix':a.reshape(4,6),'again':a})
        result=pickle.loads(encoded(payload));exact(self,payload,result)
        with self.assertRaises(TypeError):result['new']=1

    def worker_result(self,worker,value):
        self.assertTrue(worker.submit(value))
        deadline=time.monotonic()+15.
        while time.monotonic()<deadline:
            result=worker.poll()
            if result is not None:
                self.assertIsNone(result.error);return result.value
            time.sleep(.005)
        self.fail('Worker result timed out')

    def test_real_process_isolation_and_irreversible_receiver_immutability(self):
        worker=ProcessWorker(process_tests.PureProcessor().compute);self.addCleanup(worker.close)
        source=np.arange(800000,dtype='f8');source.setflags(write=False)
        expected=source.copy();result=self.worker_result(worker,source)
        self.assertNotEqual(result['pid'],os.getpid());self.assertFalse(result['sdk_modules'])
        source.setflags(write=True);source[:]=-1
        np.testing.assert_array_equal(result['payload'],expected)
        with self.assertRaises(ValueError):result['payload'].setflags(write=True)

    def test_trusted_full_completion_wire_decode_and_cross_process_equivalence(self):
        root=Path('outputs/motion-planning-v5/final-hardening-20261114')
        old=root/'original-completion.pickle';candidate=root/'candidate-completion.pickle'
        if not old.exists():self.skipTest('Local pre-optimization completion fixture unavailable')
        a=pickle.loads(old.read_bytes());b=pickle.loads(candidate.read_bytes());exact(self,a,b)
        # Saved original wire bytes are a before-change oracle, including every
        # snapshot field, box, visibility witness and calibration timestamp.
        # Re-encoding an unpickled object may normalize pickle memo aliases;
        # compare both encoders on that same object, not to its old wire memo.
        original_wire=encoded(a,CopyingPickler);new_wire=encoded(a)
        exact(self,pickle.loads(original_wire),pickle.loads(new_wire))
        self.assertEqual(hashlib.sha256(new_wire).digest(),hashlib.sha256(original_wire).digest())
        worker=ProcessWorker(process_tests.PureProcessor().compute);self.addCleanup(worker.close)
        actual=self.worker_result(worker,a)['payload'];exact(self,a,actual)
        self.assertEqual(a.generation,actual.generation);self.assertEqual(a.snapshot.revision,actual.snapshot.revision)

    def test_all_trusted_archived_inputs_preserve_all_fields_and_processing(self):
        from functools import partial
        from robot_skill_stack.integrations.isaac.config import load_scene_config
        from robot_skill_stack.world.spatial_mapping.runtime import MappingProcessor
        from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleExtractionSettings
        root=Path('outputs/motion-planning-v5/autonomous-integration-20261114')
        if not root.exists():self.skipTest('Trusted sensor archives unavailable')
        manifest=json.loads(Path('research/rgbd_box_geometry/evidence/autonomous_integration/raw-manifest.json').read_text())['files']
        timing={'compute_seconds','mapping_compute_seconds','elapsed_seconds','extract_seconds','timing_ms'}
        def semantic(value):
            if is_dataclass(value):return {k:semantic(v) for k,v in vars(value).items()}
            if hasattr(value,'keys'):return {k:semantic(v) for k,v in value.items() if k not in timing}
            if isinstance(value,(tuple,list)):return tuple(semantic(v) for v in value)
            return value
        for scene in ('partial','static','v43','empty','large','sparse','narrow','verified_occlusion'):
            folder=root/scene/'attempt-1';path=folder/'accepted_mapping_input.pickle';raw=path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest[str(path.relative_to(root))])
            original=pickle.loads(raw);transferred=pickle.loads(encoded(original));exact(self,original,transferred)
            cfg=load_scene_config(folder/'runtime_profile.toml');products=[]
            for payload in (original,transferred):
                processor=MappingProcessor(cfg.spatial_mapping,clock=partial(float,payload.frames[0].acquired_at+.1),
                    obstacle_settings=ObstacleExtractionSettings(64,0.),reconstruction_scene=cfg.scene)
                products.append(processor.compute(payload))
            with self.subTest(scene=scene):exact(self,semantic(products[0]),semantic(products[1]))
