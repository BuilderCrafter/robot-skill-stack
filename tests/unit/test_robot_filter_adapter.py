"""USD/Kit binding-contract doubles, not native Isaac execution."""
from contextlib import contextmanager
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import sys
import unittest
import numpy as np
from robot_skill_stack.world.perception.v2.mesh_shapes import box_triangles
from robot_skill_stack.world.perception.v2.self_filter import RobotSurfaceModel

ROOT = Path(__file__).resolve().parents[2]
SENSORS = ROOT/'robot_skill_stack/integrations/isaac/sensors'


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Attr:
    def __init__(self, value, varying=False): self.value, self.varying = value, varying
    def Get(self): return self.value
    def ValueMightBeTimeVarying(self): return self.varying


class Prim:
    def __init__(self, name, parent=None, kind='Xform', body=False, visible=True, purpose='default'):
        self.name, self.parent, self.kind, self.body = name, parent, kind, body
        self.visible, self.purpose = visible, purpose
        self.children, self.resets, self.animated = [], False, False
        self.relative = np.eye(4)
        self.attrs = dict(Size=Attr(.1), Radius=Attr(.04), Height=Attr(.1), Axis=Attr('Z'))
        if parent is not None: parent.children.append(self)
    def GetName(self): return self.name
    def GetPath(self): return (str(self.parent.GetPath()) if self.parent else '')+'/'+self.name
    def GetParent(self): return self.parent
    def GetTypeName(self): return self.kind
    def IsValid(self): return True
    def IsA(self, schema):
        return (schema == Schema and self.kind != 'Material') or (schema == 'Gprim' and self.kind not in {'Xform', 'Material'})
    def HasAPI(self, api): return api == 'RigidBodyAPI' and self.body


def walk(prim):
    yield prim
    for child in prim.children: yield from walk(child)


class Schema:
    def __init__(self, p): self.p = p
    def ComputeVisibility(self):
        p = self.p
        while p is not None:
            if not p.visible: return 'invisible'
            p = p.parent
        return 'inherited'
    def ComputePurpose(self): return self.p.purpose
    def TransformMightBeTimeVarying(self): return self.p.animated
    def __getattr__(self, name):
        if name.startswith('Get') and name.endswith('Attr'):
            return lambda: self.p.attrs[name[3:-4]]
        raise AttributeError(name)


@contextmanager
def fake_usd(root):
    traversals = []
    sentinel = object()
    def prim_range(prim, flags):
        if flags is not sentinel: raise AssertionError('Instance proxy traversal required')
        traversals.append(str(prim.GetPath()))
        return list(walk(prim))
    stage = NS(up='Z', meters=1., GetPrimAtPath=lambda path: next((p for p in walk(root) if p.GetPath() == path),
                                                               NS(IsValid=lambda: False)))
    cache = NS(ComputeRelativeTransform=lambda prim, link: (prim.relative, prim.resets))
    usd = NS(PrimRange=prim_range, TraverseInstanceProxies=lambda: sentinel, TimeCode=NS(Default=lambda: 'default'))
    geom = NS(GetStageUpAxis=lambda s: s.up, GetStageMetersPerUnit=lambda s: s.meters,
              Tokens=NS(z='Z', invisible='invisible', default_='default', render='render'), Gprim='Gprim',
              XformCache=lambda time: cache, Imageable=Schema, Xformable=Schema)
    for name in ('Mesh', 'Cube', 'Sphere', 'Cylinder', 'Capsule', 'Cone'): setattr(geom, name, Schema)
    pxr = ModuleType('pxr'); pxr.Usd, pxr.UsdGeom, pxr.UsdPhysics = usd, geom, NS(RigidBodyAPI='RigidBodyAPI')
    with patch.dict(sys.modules, {'pxr': pxr}):
        yield load('mesh_adapter_test', SENSORS/'robot_meshes.py'), stage, traversals


class MeshAdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = Prim('Robot')
        self.hand = Prim('hand', self.root, body=True)
        self.visual = Prim('visual', self.hand, kind='Cube')

    def read(self, excluded=()):
        with fake_usd(self.root) as (adapter, stage, calls):
            return adapter.load_robot_meshes(stage, '/Robot', excluded, ('hand',))

    def test_read_only_triangle_model_with_instance_proxy_traversal(self):
        with fake_usd(self.root) as (adapter, stage, calls):
            model, inventory = adapter.load_robot_meshes(stage, '/Robot', (), ('hand',))
        self.assertEqual(model.names, ('/Robot/hand',))
        self.assertEqual(model.triangle_count, 12)
        self.assertEqual(inventory[0]['geometry_paths'], ['/Robot/hand/visual'])
        self.assertIn('/Robot/hand', calls)

    def test_excludes_nested_carried_rigid_body(self):
        held = Prim('carried', self.hand, body=True)
        Prim('surface', held, kind='Sphere')
        self.assertEqual(self.read()[0].triangle_count, 12)

    def test_excludes_configured_target_even_without_rigid_body(self):
        target = Prim('carried', self.hand)
        Prim('surface', target, kind='Cube')
        self.assertEqual(self.read(('/Robot/hand/carried',))[0].triangle_count, 12)

    def test_invisible_and_guide_meshes_not_used(self):
        Prim('collision', self.hand, kind='Cube', visible=False)
        Prim('guide', self.hand, kind='Cube', purpose='guide')
        self.assertEqual(self.read()[0].triangle_count, 12)

    def test_gf_row_matrix_transposed_to_link_coordinates(self):
        self.visual.relative[3, :3] = [.1, .2, .3]
        triangles = self.read()[0].bvhs[0].triangles
        np.testing.assert_allclose(triangles.min((0, 1)), [.05, .15, .25])
        np.testing.assert_allclose(triangles.max((0, 1)), [.15, .25, .35])

    def test_missing_link_and_duplicate_link_fail(self):
        with fake_usd(self.root) as (adapter, stage, calls):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                adapter.load_robot_meshes(stage, '/Robot', (), ('missing',))
        Prim('hand', self.root)
        with self.assertRaisesRegex(ValueError, 'exactly one'): self.read()

    def test_material_with_link_name_is_not_a_second_body(self):
        looks = Prim('Looks', self.root)
        Prim('hand', looks, kind='Material')
        self.assertEqual(self.read()[0].triangle_count, 12)

    def test_animation_on_visual_parent_is_rejected(self):
        self.hand.children.clear()
        transform = Prim('moving_visual', self.hand)
        transform.animated = True
        Prim('mesh', transform, kind='Cube')
        with self.assertRaisesRegex(ValueError, 'Animated'): self.read()

    def test_child_link_geometry_not_duplicated_in_parent(self):
        finger = Prim('finger', self.hand, body=True)
        Prim('mesh', finger, kind='Cube')
        with fake_usd(self.root) as (adapter, stage, calls):
            model, _ = adapter.load_robot_meshes(stage, '/Robot', (), ('hand', 'finger'))
        self.assertEqual([len(b.triangles) for b in model.bvhs], [12, 12])

    def test_no_visible_mesh_fails_not_silently_unfiltered(self):
        self.visual.visible = False
        with self.assertRaisesRegex(ValueError, 'No visible'): self.read()

    def test_reset_local_transform_fails(self):
        self.visual.resets = True
        with self.assertRaisesRegex(ValueError, 'Reset transform'): self.read()

    def test_animated_local_mesh_fails(self):
        self.visual.animated = True
        with self.assertRaisesRegex(ValueError, 'Animated'): self.read()

    def test_stage_units_and_up_axis_validated(self):
        with fake_usd(self.root) as (adapter, stage, calls):
            for unit, up in [(100., 'Z'), (1., 'Y')]:
                stage.meters, stage.up = unit, up
                with self.assertRaisesRegex(ValueError, 'meters/Z-up'):
                    adapter.load_robot_meshes(stage, '/Robot', (), ('hand',))

    def test_mesh_schema_triangle_extraction(self):
        self.visual.kind = 'Mesh'
        self.visual.attrs.update(Points=Attr([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                                 FaceVertexCounts=Attr([3]), FaceVertexIndices=Attr([0, 1, 2]), HoleIndices=Attr([]))
        self.assertEqual(self.read()[0].triangle_count, 1)
        self.visual.attrs['Points'].varying = True
        with self.assertRaisesRegex(ValueError, 'Deforming'): self.read()

    def test_unknown_gprim_rejected(self):
        self.visual.kind = 'FancyUnknownSurface'
        with self.assertRaisesRegex(ValueError, 'Unsupported'): self.read()


class Stream:
    def __init__(self): self.calls = []
    def create_subscription_to_pop_by_type(self, *args, **kwargs):
        self.calls.append((args, kwargs)); return object()
    def create_subscription_to_pop(self, *args, **kwargs):
        self.calls.append((args, kwargs)); return object()


@contextmanager
def fake_runtime():
    model = RobotSurfaceModel([('/Robot/hand', box_triangles(.1)), ('/Robot/finger', box_triangles(.01))])
    env = NS(stage=object(), world=NS(current_time=1.), render=Stream(), update=Stream(), token='A', image_time=1.,
             p=np.array([[1., 0, 0], [2., 0, 0]]), q=np.array([[1., 0, 0, 0], [1., 0, 0, 0]]), reads=[], scales=np.ones((2, 3)))
    env.camera = NS(get_frame_token=lambda: env.token, camera=NS(get_current_frame=lambda: {'rendering_time': env.image_time}))
    context = NS(get_stage=lambda: env.stage, get_rendering_event_stream=lambda: env.render)
    def pose_read(usd=True):
        env.reads.append(usd); return env.p.copy(), env.q.copy()
    def view(**kwargs):
        env.view_args = kwargs
        return NS(prim_paths=list(reversed(model.names)), get_world_scales=lambda: env.scales,
                  get_world_poses=pose_read)
    omni, kit, app, usd = (ModuleType(name) for name in ('omni', 'omni.kit', 'omni.kit.app', 'omni.usd'))
    omni.kit, omni.usd, kit.app = kit, usd, app
    app.get_app = lambda: NS(get_post_update_event_stream=lambda: env.update)
    usd.get_context, usd.StageRenderingEventType = lambda: context, NS(NEW_FRAME=99)
    prims = ModuleType('isaacsim.core.prims'); prims.XFormPrim = view
    meshes = ModuleType('robot_skill_stack.integrations.isaac.sensors.robot_meshes')
    meshes.load_robot_meshes = lambda *a: (model, [{'known_only': True}])
    names = {'omni': omni, 'omni.kit': kit, 'omni.kit.app': app, 'omni.usd': usd,
             'isaacsim': ModuleType('isaacsim'), 'isaacsim.core': ModuleType('isaacsim.core'),
             'isaacsim.core.prims': prims, meshes.__name__: meshes}
    with patch.dict(sys.modules, names):
        adapter = load('pose_adapter_test', SENSORS/'robot_self_filter.py')
        yield adapter, env


class PoseAdapterTests(unittest.TestCase):
    def source(self, adapter, env):
        return adapter.IsaacRobotSnapshotSource(env.world, env.camera, '/Robot', ('/World/Objects',), .002)

    def test_fabric_measured_pose_wxyz_and_fixed_link_order(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); s._on_frame(None)
            self.assertEqual(e.reads, [False])
            self.assertFalse(e.view_args['reset_xform_properties'])
            self.assertEqual(s.snapshot('A').world_from_links[:, 0, 3].tolist(), [2., 1.])
            self.assertEqual(e.render.calls[0][1]['order'], 1100)
            self.assertEqual(len(e.update.calls), 1)

    def test_delayed_render_uses_history_not_newer_pose(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); s._on_frame(None)
            e.world.current_time, e.token = 1.1, 'B'
            e.p[:, 0] = 10.
            s._on_frame(None)
            snap = s.snapshot('B')
            self.assertEqual(snap.pose_time, 1.)
            self.assertEqual(snap.world_from_links[:, 0, 3].tolist(), [2., 1.])

    def test_timestamp_mismatch_not_paired_with_latest(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); e.image_time = .5; s._on_frame(None)
            with self.assertRaisesRegex(RuntimeError, 'No matching'): s.snapshot('A')

    def test_same_time_samples_do_not_grow_queue(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e)
            for _ in range(10): s._on_frame(None)
            self.assertEqual(len(s.history), 1)

    def test_time_reset_discards_old_history(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); s._on_frame(None)
            e.world.current_time, e.image_time, e.token = .1, .1, 'reset'
            s._on_frame(None)
            self.assertEqual(s.snapshot('reset').pose_time, .1)
            self.assertEqual(len(s.history), 1)

    def test_history_is_bounded(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e)
            for i in range(300):
                e.world.current_time = e.image_time = float(i)
                e.token = str(i); s._on_frame(None)
            self.assertEqual(len(s.history), 256)

    def test_stage_change_refuses_stale_geometry(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); s._on_frame(None); e.stage = object()
            with self.assertRaisesRegex(RuntimeError, 'stage changed'): s.snapshot('A')

    def test_close_releases_both_subscriptions_and_history(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e); s._on_frame(None); s.close(); s.close()
            self.assertIsNone(s._subscription); self.assertIsNone(s._update_subscription)
            self.assertFalse(s.history)
            with self.assertRaises(RuntimeError): s.snapshot('A')

    def test_scaled_link_root_rejected(self):
        with fake_runtime() as (a, e):
            e.scales *= 2
            with self.assertRaisesRegex(ValueError, 'Scaled'): self.source(a, e)

    def test_camera_token_race_rejected(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e)
            tokens = iter(['A', 'B'])
            e.camera.get_frame_token = lambda: next(tokens)
            s._on_frame(None)
            self.assertIsNone(s.latest)
            self.assertIn('Camera changed', s.last_error)

    def test_time_advances_during_pose_copy_rejected(self):
        with fake_runtime() as (a, e):
            s = self.source(a, e)
            def read(**kw): e.world.current_time += .1; return e.p, e.q
            s.view.get_world_poses = read; s._on_frame(None)
            self.assertIsNone(s.latest)
            self.assertIn('Physics advanced', s.last_error)

    def test_quaternion_mapping(self):
        with fake_runtime() as (a, e):
            t = a.pose_matrices([[1, 2, 3]], [[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
            np.testing.assert_allclose(t[0, :3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
            with self.assertRaises(ValueError): a.pose_matrices([[0, 0, 0]], [[0, 0, 0, 0]])


if __name__ == '__main__': unittest.main(verbosity=2)
