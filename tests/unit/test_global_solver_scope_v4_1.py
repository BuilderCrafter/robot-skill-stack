"""Session ownership and failure recovery without an Isaac or USD dependency."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robot_skill_stack.integrations.isaac.manipulation.motion import global_solver as solver
from robot_skill_stack.manipulation.motion.joint_route import PlanningFailure


DEFAULT_TIME = object()


class AttributeSpec:
    def __init__(self, info=None):
        self.info = dict(info or {})

    def HasInfo(self, key): return key in self.info
    def GetInfo(self, key): return self.info[key]
    def ListInfoKeys(self): return list(self.info)


class PrimSpec:
    specifier = 'over'

    def __init__(self, layer, path):
        self.layer, self.path, self.properties, self.info = layer, path, {}, {}

    @property
    def nameChildren(self):
        return [path for path in self.layer.prims
            if path.rsplit('/', 1)[0] == self.path]

    def ListInfoKeys(self): return ['specifier', *self.info]


class Layer:
    def __init__(self): self.prims = {}
    def GetPrimAtPath(self, path): return self.prims.get(str(path))

    def GetAttributeAtPath(self, path):
        prim, name = str(path).rsplit('.', 1)
        spec = self.prims.get(prim)
        return None if spec is None else spec.properties.get(name)

    def ensure(self, path):
        parts = path.strip('/').split('/')
        for index in range(1, len(parts)+1):
            ancestor = '/'+ '/'.join(parts[:index])
            self.prims.setdefault(ancestor, PrimSpec(self, ancestor))
        return self.prims[path]


class Attribute:
    def __init__(self, stage, path): self.stage, self.path = stage, path

    def Get(self):
        spec = self.stage.session.GetAttributeAtPath(self.path+'.'+solver.POSITION_ATTRIBUTE)
        return spec.GetInfo('default') if spec and spec.HasInfo('default') else self.stage.weaker_count

    def Set(self, value):
        assert self.stage.edit_target is self.stage.session
        prim = self.stage.session.ensure(self.path)
        prim.properties.setdefault(solver.POSITION_ATTRIBUTE, AttributeSpec()).info['default'] = value
        self.stage.set_calls.append(value)
        if self.stage.fail_after_write:
            failure, self.stage.fail_after_write = self.stage.fail_after_write, None
            if isinstance(failure, Exception): raise failure
            return False
        return True

    def ClearAtTime(self, time):
        assert self.stage.edit_target is self.stage.session
        assert time is DEFAULT_TIME
        self.stage.cleared_times.append(time)
        self.stage.session.GetAttributeAtPath(self.path+'.'+solver.POSITION_ATTRIBUTE).info.pop('default', None)


class Prim:
    def __init__(self, stage, path): self.stage, self.path = stage, path
    def GetAttribute(self, name):
        assert name == solver.POSITION_ATTRIBUTE
        return Attribute(self.stage, self.path)

    def RemoveProperty(self, name):
        assert self.stage.edit_target is self.stage.session
        self.stage.session.prims[self.path].properties.pop(name)
        self.stage.removed_properties.append((self.path, name))


class Stage:
    def __init__(self, weaker_count=32):
        self.session, self.edit_target = Layer(), object()
        self.weaker_count, self.fail_after_write = weaker_count, None
        self.set_calls, self.cleared_times, self.removed_properties, self.removed_prims = [], [], [], []

    def GetSessionLayer(self): return self.session
    def GetPrimAtPath(self, path): return Prim(self, str(path))

    def RemovePrim(self, path):
        assert self.edit_target is self.session
        self.session.prims.pop(path)
        self.removed_prims.append(path)
        return True


class EditContext:
    def __init__(self, stage, layer): self.stage, self.layer = stage, layer
    def __enter__(self):
        self.previous, self.stage.edit_target = self.stage.edit_target, self.layer
    def __exit__(self, *args): self.stage.edit_target = self.previous


def modules(stage, *, maximum=255):
    usd = SimpleNamespace(EditContext=EditContext, TimeCode=SimpleNamespace(Default=lambda:DEFAULT_TIME))
    scene = SimpleNamespace(GetMaxPositionIterationCountAttr=lambda:SimpleNamespace(Get=lambda:maximum))
    pxr = SimpleNamespace(Usd=usd, Sdf=SimpleNamespace(SpecifierOver='over'),
        PhysxSchema=SimpleNamespace(PhysxSceneAPI=lambda prim:scene))
    omni_usd = SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage))
    return {'pxr':pxr, 'omni':SimpleNamespace(usd=omni_usd), 'omni.usd':omni_usd}


def backend(kind='TGS'):
    return SimpleNamespace(robot=SimpleNamespace(prim_path='/World/Franka'),
        world=SimpleNamespace(get_physics_context=lambda:SimpleNamespace(
            get_solver_type=lambda:kind, prim_path='/PhysicsScene')))


class SolverScopeTests(unittest.TestCase):
    def test_absent_opinion_removes_only_owned_ancestors_and_close_is_idempotent(self):
        stage = Stage()
        original_edit_target = stage.edit_target
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            self.assertEqual(Attribute(stage, '/World/Franka').Get(), 255)
            self.assertIs(stage.edit_target, original_edit_target)
            lease.close()
            self.assertEqual(stage.session.prims, {})
            self.assertEqual(stage.removed_prims, ['/World/Franka', '/World'])
            self.assertEqual(Attribute(stage, '/World/Franka').Get(), 32)
            lease.close()
        self.assertEqual(stage.set_calls, [255])
        self.assertEqual(len(stage.removed_properties), 1)
        self.assertTrue(lease.diagnostics['original_session_opinion_restored'])
        self.assertIs(stage.edit_target, original_edit_target)

    def test_preexisting_default_and_metadata_are_restored(self):
        stage = Stage()
        original = dict(default=64, documentation='foreign attribute metadata')
        stage.session.ensure('/World/Franka').properties[solver.POSITION_ATTRIBUTE] = AttributeSpec(original)
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            lease.close()
        spec = stage.session.GetAttributeAtPath('/World/Franka.'+solver.POSITION_ATTRIBUTE)
        self.assertEqual(spec.info, original)
        self.assertEqual(stage.set_calls, [255, 64])
        self.assertFalse(stage.removed_prims)

    def test_no_default_preserves_time_samples_and_metadata_with_default_only_clear(self):
        stage = Stage()
        original = dict(timeSamples={0.:32, 1.:64}, documentation='retain samples')
        stage.session.ensure('/World/Franka').properties[solver.POSITION_ATTRIBUTE] = AttributeSpec(deepcopy(original))
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            lease.close()
        self.assertEqual(stage.session.GetAttributeAtPath('/World/Franka.'+solver.POSITION_ATTRIBUTE).info, original)
        self.assertEqual(stage.cleared_times, [DEFAULT_TIME])
        self.assertEqual(stage.set_calls, [255])

    def test_close_uses_captured_stage_after_global_stage_replacement(self):
        old, replacement = Stage(), Stage(weaker_count=64)
        fake = modules(old)
        with patch.dict('sys.modules', fake):
            lease = solver.PositionIterationLease(old, '/World/Franka', 255)
            fake['omni.usd'].get_context = lambda:SimpleNamespace(get_stage=lambda:replacement)
            lease.close()
        self.assertEqual(old.session.prims, {})
        self.assertEqual(replacement.session.prims, {})
        self.assertFalse(replacement.set_calls or replacement.removed_prims)

    def test_partial_activation_failure_restores_authored_presence(self):
        for failure in (False, RuntimeError('native setter failed after writing')):
            with self.subTest(failure=failure):
                stage = Stage()
                stage.fail_after_write = failure if failure is not False else 'return-false'
                with patch.dict('sys.modules', modules(stage)), self.assertRaises(RuntimeError):
                    solver.PositionIterationLease(stage, '/World/Franka', 255)
                self.assertEqual(stage.session.prims, {})
                self.assertEqual(Attribute(stage, '/World/Franka').Get(), 32)

    def test_cleanup_preserves_foreign_prim_metadata_new_attributes_and_children(self):
        stage = Stage()
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            stage.session.ensure('/World/Franka').info['customData'] = {'foreign':True}
            stage.session.ensure('/World/Franka').properties['foreign:attribute'] = AttributeSpec(dict(default='keep'))
            stage.session.ensure('/World/Franka/NewChild').info['documentation'] = 'keep child'
            lease.close()
        self.assertIsNone(stage.session.GetAttributeAtPath('/World/Franka.'+solver.POSITION_ATTRIBUTE))
        self.assertEqual(stage.session.GetPrimAtPath('/World/Franka').info['customData'], {'foreign':True})
        self.assertIsNotNone(stage.session.GetAttributeAtPath('/World/Franka.foreign:attribute'))
        self.assertIsNotNone(stage.session.GetPrimAtPath('/World/Franka/NewChild'))
        self.assertFalse(stage.removed_prims)

    def test_foreign_default_is_preserved_and_cleanup_can_retry_after_owner_releases_it(self):
        stage = Stage()
        stage.session.ensure('/World/Franka').properties[solver.POSITION_ATTRIBUTE] = AttributeSpec(dict(default=64))
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            spec = stage.session.GetAttributeAtPath('/World/Franka.'+solver.POSITION_ATTRIBUTE)
            spec.info['default'] = 128
            with self.assertRaisesRegex(RuntimeError, 'default changed by another owner'):
                lease.close()
            self.assertEqual(Attribute(stage, '/World/Franka').Get(), 128)
            self.assertEqual(stage.set_calls, [255])
            self.assertFalse(lease.closed or lease.restored)
            spec.info['default'] = 255
            lease.close()
        self.assertEqual(Attribute(stage, '/World/Franka').Get(), 64)
        self.assertEqual(stage.set_calls, [255, 64])
        self.assertTrue(lease.closed and lease.restored)

    def test_new_count_attribute_foreign_metadata_and_samples_are_preserved_until_released(self):
        for key, value in [('documentation', 'foreign metadata'), ('timeSamples', {0.:128, 1.:255})]:
            with self.subTest(key=key):
                stage = Stage()
                with patch.dict('sys.modules', modules(stage)):
                    lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
                    spec = stage.session.GetAttributeAtPath('/World/Franka.'+solver.POSITION_ATTRIBUTE)
                    spec.info[key] = deepcopy(value)
                    before = deepcopy(spec.info)
                    with self.assertRaisesRegex(RuntimeError, 'metadata changed by another owner'):
                        lease.close()
                    self.assertEqual(spec.info, before)
                    self.assertFalse(stage.removed_properties or stage.removed_prims)
                    self.assertFalse(lease.closed or lease.restored)
                    spec.info.pop(key)
                    lease.close()
                self.assertEqual(stage.session.prims, {})
                self.assertEqual(Attribute(stage, '/World/Franka').Get(), 32)
                self.assertTrue(lease.closed and lease.restored)

    def test_already_at_requested_accuracy_makes_no_session_edits(self):
        stage = Stage(weaker_count=255)
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.PositionIterationLease(stage, '/World/Franka', 255)
            lease.close()
        self.assertFalse(lease.changed)
        self.assertEqual(stage.session.prims, {})
        self.assertFalse(stage.set_calls)

    def test_factory_rejects_scene_ceiling_before_authoring_and_skips_pgs(self):
        stage = Stage()
        with patch.dict('sys.modules', modules(stage, maximum=254)):
            with self.assertRaises(PlanningFailure) as failure:
                solver.execution_solver_scope(backend())
            self.assertEqual(failure.exception.code, 'solver_configuration_failure')
            unchanged = solver.execution_solver_scope(backend('PGS'))
            unchanged.close()
        self.assertFalse(unchanged.diagnostics['changed'])
        self.assertEqual(stage.session.prims, {})
        self.assertFalse(stage.set_calls)

    def test_factory_uses_validated_255_and_rejects_unknown_solver(self):
        stage = Stage()
        with patch.dict('sys.modules', modules(stage)):
            lease = solver.execution_solver_scope(backend())
            self.assertEqual(lease.requested, 255)
            lease.close()
            with self.assertRaises(PlanningFailure) as failure:
                solver.execution_solver_scope(backend('unknown'))
        self.assertEqual(failure.exception.code, 'solver_configuration_failure')
        self.assertEqual(stage.session.prims, {})


if __name__ == '__main__':
    unittest.main()
