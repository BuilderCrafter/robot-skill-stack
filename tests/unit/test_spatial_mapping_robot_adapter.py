"""Robot geometry/PhysX boundaries and exposure history without Isaac imports."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import IsaacRobotSelfFilter, _triangles
from robot_skill_stack.world.perception_multiview.acquisition import DepthFrame

ADAPTER = 'robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter'


def matrix(z=0.):
    result = np.eye(4)
    result[2, 3] = z
    return result


class Prim:
    def __init__(self, path, *, parent=None, kind='xform', rigid=False, world=None):
        self.path, self.parent, self.kind, self.rigid = path, parent, kind, rigid
        self.world = np.eye(4) if world is None else world
        self.valid = self.active = self.loaded = self.defined = True
        self.abstract, self.purpose, self.visibility = False, 'default', 'inherited'
        self.scheme, self.varying = 'none', False
        self.points = np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [.2, .2, 0.], [-.2, .2, 0.]])
        self.counts, self.indices, self.holes = np.array([4]), np.arange(4), np.array([], int)
    def __bool__(self): return self.valid
    def GetPath(self): return self.path
    def GetParent(self): return self.parent
    def IsValid(self): return self.valid
    def IsActive(self): return self.active
    def IsLoaded(self): return self.loaded
    def IsDefined(self): return self.defined
    def IsAbstract(self): return self.abstract
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
            return NS(Get=lambda: getattr(self.prim, name), ValueMightBeTimeVarying=lambda: self.prim.varying)
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
    with patch(ADAPTER+'._usd_modules', return_value=(usd, geom, physics)), \
         patch(ADAPTER+'._simulation_interfaces', return_value=(manager, NS(POST_PHYSICS_STEP='post'))), \
         patch.object(Prim, 'IsA', lambda prim, schema: prim.kind == 'mesh' if schema is geom.Mesh else is_a(prim, schema)):
        yield state, view, manager


def frame(stamp=1., depth=1., camera_id='A', number=10):
    return DepthFrame(camera_id, (number, stamp), np.full((3, 3), depth),
                      np.array([[100., 0, 1.5], [0, 100., 1.5], [0, 0, 1.]]), np.eye(4), 100.)


class IsaacRobotFilterTests(unittest.TestCase):
    def test_visible_geometry_cached_once_configured_path_and_owned_callback(self):
        scene = Scene()
        hidden = Prim(scene.body.path+'/HiddenCollider', parent=scene.body, kind='cube')
        hidden.visibility = 'invisible'
        guide = Prim(scene.body.path+'/Guide', parent=scene.body, kind='cube')
        guide.purpose = 'guide'
        scene.prims.extend([hidden, guide])
        with boundaries(scene) as (state, view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            mesh = source._parts[0].mesh
            self.assertTrue(source.masks((frame(),))['A'].all())
            self.assertIs(source._parts[0].mesh, mesh)
            with self.assertRaises(ValueError):
                mesh.vertices.setflags(write=True)
            manager.register_callback.assert_called_once_with(source._post_step, 'post', order=100)
            self.assertEqual(view.prim_paths, [scene.body.path])
            source.close()
            source.close()
            manager.deregister_callback.assert_called_once_with(42)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                source.masks((frame(),))

    def test_old_frames_use_exposure_history_not_latest_usd_or_physx_pose(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            state.stamp, state.raw[0, 2] = 1.1, 1.2
            state.callback(1/60)
            scene.mesh.world = matrix(8.)  # Stale/current USD cannot replace cached local geometry/history.
            self.assertTrue(source.masks((frame(1., 1.),))['A'].all())
            self.assertTrue(source.masks((frame(1.1, 1.2),))['A'].all())
            self.assertFalse(source.masks((frame(1.1, 1.),))['A'].any())  # Foreground obstacle survives.
            self.assertEqual(source.diagnostics['pose_times'], {'A': 1.1})
            source.close()

    def test_every_camera_gets_a_depth_aware_mask_and_front_obstacles_remain(self):
        scene = Scene()
        with boundaries(scene):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            frames = [frame(depth=depth, camera_id=str(i)) for i, depth in enumerate((1., .8, 1.1, 1., .7))]
            masks = source.masks(tuple(reversed(frames)))
            self.assertEqual(set(masks), {'0', '1', '2', '3', '4'})
            for key in ('0', '2', '3'):
                self.assertTrue(masks[key].all())
            for key in ('1', '4'):
                self.assertFalse(masks[key].any())
            source.close()

    def test_selected_ray_pixels_are_forwarded_without_dense_masking(self):
        scene = Scene()
        with boundaries(scene):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            selected = {'A': (np.array([1]), np.array([1])),
                        'B': (np.array([], int), np.array([], int))}
            masks = source.masks((frame(camera_id='A'), frame(camera_id='B')), pixels_by_camera=selected)
            self.assertEqual(int(masks['A'].sum()), 1)
            self.assertTrue(masks['A'][1, 1])
            self.assertFalse(masks['B'].any())
            self.assertEqual(source.diagnostics['cameras']['A']['sampled_pixels'], 1)
            source.close()

    def test_missing_and_evicted_history_fails_without_newer_pose_fallback(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            with self.assertRaisesRegex(RuntimeError, 'No robot pose'):
                source.masks((frame(.5),))
            for index in range(300):
                state.stamp = 1.02+index*.02
                state.callback(.02)
            self.assertEqual(len(source._history), 256)
            with self.assertRaisesRegex(RuntimeError, 'No robot pose'):
                source.masks((frame(1.),))
            source.close()

    def test_callback_error_is_deferred_and_fresh_physics_recovers(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            state.error = 'invalid tensor handle'
            state.callback(.02)  # Must not throw into physics/control.
            with self.assertRaisesRegex(RuntimeError, 'capture failed: invalid tensor handle'):
                source.masks((frame(),))
            state.error, state.stamp = None, 1.2
            state.callback(.02)
            self.assertTrue(source.masks((frame(1.2),))['A'].all())
            source.close()

    def test_pose_capture_timing_counts_success_and_deferred_failure(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            self.assertGreaterEqual(source.diagnostics['initialization_seconds'], 0.)
            self.assertEqual(source.diagnostics['pose_capture']['calls'], 0)
            with patch(ADAPTER+'.time.perf_counter', side_effect=(10., 10.003, 20., 20.007)):
                state.stamp = 1.1
                state.callback(.02)
                state.error = 'invalid tensor handle'
                state.callback(.02)  # Failed capture is measured and still deferred.
            with self.assertRaisesRegex(RuntimeError, 'capture failed: invalid tensor handle'):
                source.masks((frame(1.1),))
            timing = source.diagnostics['pose_capture']
            self.assertEqual(timing['calls'], 2)
            self.assertAlmostEqual(timing['total_seconds'], .010)
            self.assertAlmostEqual(timing['mean_seconds'], .005)
            self.assertAlmostEqual(timing['max_seconds'], .007)
            source.close()

    def test_reset_discards_old_generation_and_does_not_interpolate(self):
        scene = Scene()
        with boundaries(scene) as (state, _, _):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            state.stamp = .2
            state.callback(.02)
            self.assertEqual(source._generation, 1)
            self.assertEqual(len(source._history), 1)
            with self.assertRaisesRegex(RuntimeError, 'No robot pose'):
                source.masks((frame(1.),))
            self.assertTrue(source.masks((frame(.2),))['A'].all())
            source.close()

    def test_static_scale_and_centimeter_stage_units_are_preserved(self):
        scene = Scene()
        scene.units = .01
        scene.body.world, scene.mesh.world = matrix(100.), matrix(110.)
        scene.body.world[:3, :3] = scene.mesh.world[:3, :3] = np.diag([2., 1., 1.3])
        scene.mesh.points *= 100
        state, view, manager = manager_fixture(scene)
        state.raw[0, 2] = 100.
        with boundaries(scene, state, view, manager):
            source = IsaacRobotSelfFilter(scene, scene.root.path, stride=1)
            self.assertTrue(source.masks((frame(depth=1.1),))['A'].all())
            np.testing.assert_allclose(source._parts[0].mesh.vertices, scene.mesh.points*.01)
            source.close()

    def test_invalid_or_unsupported_models_fail_before_callback_registration(self):
        for change in ('missing', 'unloaded', 'subdivision', 'deforming', 'no_rigid', 'visible_cube'):
            scene = Scene()
            if change == 'missing': scene.root.valid = False
            if change == 'unloaded': scene.mesh.loaded = False
            if change == 'subdivision': scene.mesh.scheme = 'catmullClark'
            if change == 'deforming': scene.mesh.varying = True
            if change == 'no_rigid': scene.body.rigid = False
            if change == 'visible_cube': scene.mesh.kind = 'cube'
            with self.subTest(change=change), boundaries(scene) as (_, _, manager):
                with self.assertRaises(RuntimeError):
                    IsaacRobotSelfFilter(scene, scene.root.path)
                manager.register_callback.assert_not_called()

    def test_partial_initialization_failure_never_closes_shared_physics_view(self):
        scene = Scene()
        with boundaries(scene) as (_, view, manager):
            manager.register_callback.side_effect = RuntimeError('subscription failed')
            view.close = Mock()
            with self.assertRaisesRegex(RuntimeError, 'subscription failed'):
                IsaacRobotSelfFilter(scene, scene.root.path)
            view.close.assert_not_called()
            manager.deregister_callback.assert_not_called()

    def test_holes_and_concave_polygons_do_not_create_invented_surface(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.assertEqual(_triangles(points, [4], [0, 1, 2, 3], []).tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(len(_triangles(points, [4], [0, 1, 2, 3], [0])), 0)
        concave = np.array([[0., 0., 0.], [1., 0., 0.], [.2, .2, 0.], [0., 1., 0.]])
        with self.assertRaisesRegex(ValueError, 'planar convex'):
            _triangles(concave, [4], [0, 1, 2, 3], [])


if __name__ == '__main__':
    unittest.main()
