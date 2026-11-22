"""Robot geometry/PhysX boundaries and exposure history without Isaac imports."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import IsaacRobotSelfFilter
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame
from robot_skill_stack.world.spatial_mapping.robot_filter import RobotSelfFilter
from tests.unit.test_spatial_mapping_robot_filter import pixels

ADAPTER = 'robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter'


def matrix(z=0.):
    result = np.eye(4)
    result[2, 3] = z
    return result


class Prim:
    def __init__(self, path, *, parent=None, kind='xform', rigid=False, world=None):
        self.path, self.parent, self.kind, self.rigid = path, parent, kind, rigid
        self.world = np.eye(4) if world is None else world
        self.purpose, self.visibility = 'default', 'inherited'
        self.scheme = 'none'
        self.points = np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [.2, .2, 0.], [-.2, .2, 0.]])
        self.counts, self.indices, self.holes = np.array([3, 3]), np.array([0, 1, 2, 0, 2, 3]), np.array([], int)
    def GetPath(self): return self.path
    def GetParent(self): return self.parent
    def IsActive(self): return True
    def IsLoaded(self): return True
    def IsDefined(self): return True
    def IsAbstract(self): return False
    def HasAPI(self, api): return self.rigid
    def IsA(self, schema): return self.kind in ('mesh', 'cube') if schema == 'gprim' else self.kind == schema


class Scene:
    def __init__(self):
        self.root = Prim('/Robots/Panda')
        self.body = Prim('/Robots/Panda/Arm', parent=self.root, rigid=True, world=matrix(1.))
        self.mesh = Prim('/Robots/Panda/Arm/Visual', parent=self.body, kind='mesh', world=matrix(1.))
        self.prims, self.units = [self.root, self.body, self.mesh], 1.
    def GetPrimAtPath(self, path):
        return next((prim for prim in self.prims if prim.path == path), None)


def usd_modules(scene):
    class Mesh:
        def __init__(self, prim): self.prim = prim
        def attr(self, name):
            return NS(Get=lambda: getattr(self.prim, name))
        def GetSubdivisionSchemeAttr(self): return self.attr('scheme')
        def GetPointsAttr(self): return self.attr('points')
        def GetFaceVertexCountsAttr(self): return self.attr('counts')
        def GetFaceVertexIndicesAttr(self): return self.attr('indices')
        def GetHoleIndicesAttr(self): return self.attr('holes')
    usd = NS(TimeCode=NS(Default=lambda: None), PrimAllPrimsPredicate=object(),
             TraverseInstanceProxies=Mock(return_value=object()), PrimRange=Mock(return_value=scene.prims))
    geom = NS(Mesh=Mesh, Gprim='gprim', Tokens=NS(default_='default', render='render', invisible='invisible'),
              Imageable=lambda prim: NS(ComputePurpose=lambda: prim.purpose,
                                        ComputeVisibility=lambda: prim.visibility),
              GetStageMetersPerUnit=lambda stage: stage.units,
              XformCache=lambda time: NS(GetLocalToWorldTransform=lambda prim: prim.world.T))
    # IsA receives the actual schema constructor, just as USD does.
    original = Prim.IsA
    return usd, geom, NS(RigidBodyAPI=object()), original


def manager_fixture(scene):
    state = NS(stamp=1., raw=np.array([[0., 0., 1., 0., 0., 0., 1.]]), error=None, callback=None)
    def transforms():
        if state.error is not None:
            raise RuntimeError(state.error)
        return state.raw
    view = NS(prim_paths=[scene.body.path], count=1, get_transforms=Mock(side_effect=transforms))
    def register(callback, event, order):
        state.callback = callback
        return 42
    manager = NS(get_simulation_time=lambda: state.stamp,
                 get_physics_sim_view=Mock(return_value=NS(create_rigid_body_view=Mock(return_value=view))),
                 register_callback=Mock(side_effect=register), deregister_callback=Mock())
    return state, view, manager


@contextmanager
def boundaries(scene, state=None, view=None, manager=None):
    if state is None:
        state, view, manager = manager_fixture(scene)
    usd, geom, physics, is_a = usd_modules(scene)
    with patch(ADAPTER+'._register_model_notice', return_value=NS(Revoke=Mock())), \
         patch(ADAPTER+'._usd_modules', return_value=(usd, geom, physics)), \
         patch(ADAPTER+'._simulation_interfaces', return_value=(manager, NS(POST_PHYSICS_STEP='post'))), \
         patch.object(Prim, 'IsA', lambda prim, schema: prim.kind == 'mesh' if schema is geom.Mesh else is_a(prim, schema)):
        yield state, view, manager


def frame(stamp=1., depth=1., camera_id='A', number=10):
    return DepthFrame(camera_id, (number, stamp), np.full((3, 3), depth),
                      np.array([[100., 0, 1.5], [0, 100., 1.5], [0, 0, 1.]]), np.eye(4), 100.)


class IsaacRobotFilterTests(unittest.TestCase):
    def test_visible_meshes_are_cached_and_only_owned_subscriptions_are_closed(self):
        scene = Scene()
        hidden = Prim(scene.body.path+'/Collider', parent=scene.body, kind='cube')
        hidden.visibility = 'invisible'
        guide = Prim(scene.body.path+'/Guide', parent=scene.body, kind='cube')
        guide.purpose = 'guide'
        scene.prims.extend([hidden, guide])
        with boundaries(scene) as (_, view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            first = source.prepare((frame(),))['A']
            second = source.prepare((frame(),))['A']
            self.assertEqual(len(first.parts), 1)
            np.testing.assert_array_equal(first.parts[0].mesh.triangles, [[0, 1, 2], [0, 2, 3]])
            self.assertIs(first.parts[0].mesh, second.parts[0].mesh)
            listener = source._model_notice
            view.close = Mock()
            source.close()
            source.close()
            listener.Revoke.assert_called_once()
            manager.deregister_callback.assert_called_once_with(42)
            view.close.assert_not_called()
            with self.assertRaises(RuntimeError):
                source.prepare((frame(),))

    def test_moving_robot_uses_owned_historical_poses_and_keeps_camera_tokens(self):
        scene = Scene()
        with boundaries(scene) as (state, view, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            state.stamp, state.raw[0, 2] = 1.1, 1.2
            state.callback(.1)
            state.raw[0, 2] = 4.  # The capture must not alias the PhysX buffer.
            scene.mesh.world = matrix(8.)  # Nor may preparation read current USD placement.
            frames = (frame(1., 1., 'old'), frame(1.1, 1.2, 'new'), frame(1.1, 1., 'front'))
            view.get_transforms.reset_mock()
            prepared = source.prepare(frames)
            view.get_transforms.assert_not_called()
            masks = RobotSelfFilter().masks(frames, prepared, pixels_by_camera=pixels(frames))
            self.assertTrue(masks['old'].all())
            self.assertTrue(masks['new'].all())
            self.assertFalse(masks['front'].any())
            for camera in frames:
                self.assertEqual(prepared[camera.camera_id].frame_token, camera.token)
            self.assertIs(prepared['new'].parts, prepared['front'].parts)
            source.close()

    def test_missing_exposure_and_clock_rewind_never_use_newer_poses(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            with self.assertRaises(RuntimeError):
                source.prepare((frame(.5),))
            state.stamp = .2
            state.callback(.1)
            with self.assertRaises(RuntimeError):
                source.prepare((frame(1.),))
            self.assertEqual(source.prepare((frame(.2),))['A'].simulation_time, .2)
            source.close()

    def test_callback_failure_is_cached_without_sdk_reads_and_fresh_physics_recovers(self):
        scene = Scene()
        with boundaries(scene) as (state, _, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            state.error = 'invalid tensor handle'
            state.callback(.1)  # No exception may escape into Isaac's physics loop.
            with patch.object(manager, 'get_simulation_time', side_effect=AssertionError('SDK read')):
                self.assertIn('invalid tensor handle', source.source_error)
                with self.assertRaises(RuntimeError):
                    source.prepare((frame(),))
            state.error, state.stamp = None, 1.2
            state.callback(.1)
            self.assertIsNone(source.source_error)
            self.assertEqual(source.prepare((frame(1.2),))['A'].simulation_time, 1.2)
            source.close()

    def test_static_scale_mesh_offset_stage_units_and_physx_rotation(self):
        scene = Scene()
        scene.units = .01
        scene.body.world, scene.mesh.world = matrix(100.), matrix(110.)
        scene.body.world[:3, :3] = scene.mesh.world[:3, :3] = np.diag([2., 1., 1.3])
        scene.mesh.points *= 100
        state, view, manager = manager_fixture(scene)
        state.raw[0] = [0., 0., 100., 0., 0., np.sqrt(.5), np.sqrt(.5)]
        with boundaries(scene, state, view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            prepared = source.prepare((frame(depth=1.1),))['A']
            np.testing.assert_allclose(prepared.parts[0].world_from_local,
                [[0., -1., 0., 0.], [2., 0., 0., 0.], [0., 0., 1.3, 1.1], [0., 0., 0., 1.]], atol=1e-12)
            np.testing.assert_allclose(prepared.parts[0].mesh.vertices, scene.mesh.points*.01)
            source.close()

    def test_unsupported_visible_geometry_is_refused_instead_of_skipped_or_triangulated(self):
        for kind, counts in (('cube', [3, 3]), ('mesh', [4])):
            scene = Scene()
            scene.mesh.kind, scene.mesh.counts = kind, np.array(counts)
            with self.subTest(kind=kind), boundaries(scene) as (_, _, manager):
                with self.assertRaises((RuntimeError, ValueError)):
                    IsaacRobotSelfFilter(scene, scene.root.path)
                manager.register_callback.assert_not_called()

    def test_model_edits_invalidate_cached_geometry_but_body_motion_is_allowed(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path)
            notice = lambda path: NS(GetResyncedPaths=lambda: [], GetChangedInfoOnlyPaths=lambda: [path])
            source._model_changed(notice(scene.body.path+'.xformOp:translate'), scene)
            self.assertIsNone(source.source_error)
            source._model_changed(notice(scene.mesh.path+'.points'), scene)
            state.stamp = 1.1
            state.callback(.1)
            source.reset_history()  # Fresh poses/reset cannot clear a permanent model error.
            self.assertIsNotNone(source.source_error)
            with self.assertRaises(RuntimeError):
                source.prepare((frame(1.1),))
            with self.assertRaises(RuntimeError):
                source.rebind_after_play()
            source.close()


if __name__ == '__main__':
    unittest.main()
