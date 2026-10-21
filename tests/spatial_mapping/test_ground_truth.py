"""Native USD environment occupancy; isolated in-memory stage, no robot commands."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import json
from pathlib import Path
import time

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robot_skill_stack.integrations.isaac.spatial_mapping.ground_truth import IsaacGroundTruthMapper
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState
from tests.support.results import fail_result, write_result


def pose(shape, center, scale=None, rotation=None):
    xform = UsdGeom.Xformable(shape.GetPrim())
    position = xform.AddTranslateOp()
    position.Set(Gf.Vec3d(*center))
    if rotation is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    return position


def cube(stage, path, center, size=.06, **kwargs):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(size)
    return shape, pose(shape, center, **kwargs)


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'ground_truth'
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    parent = UsdGeom.Xform.Define(stage, '/Environment/RotatedParent')
    pose(parent, (-.25, -.2, .15), (1.5, .7, 1.2), (0., 0., 45.))
    cube(stage, '/Environment/RotatedParent/UnclassifiedBox', (0., 0., 0.), size=1., scale=(.12, .06, .08))
    sphere = UsdGeom.Sphere.Define(stage, '/Environment/Ellipsoid')
    sphere.CreateRadiusAttr(.05)
    pose(sphere, (.18, -.1, .18), (1.8, .8, 1.1), (0., 0., 25.))
    cylinder_centers = ((-.45, 0., .2), (-.05, .35, .2), (.25, .35, .4))
    for axis, center in zip(('X', 'Y', 'Z'), cylinder_centers):
        cylinder = UsdGeom.Cylinder.Define(stage, '/Environment/Cylinder'+axis)
        cylinder.CreateRadiusAttr(.025)
        cylinder.CreateHeightAttr(.12)
        cylinder.CreateAxisAttr(axis)
        pose(cylinder, center, (1.1, .8, 1.2), (25., 15., 35.))
    moving, translation = cube(stage, '/Environment/Moving', (.42, -.1, .08))
    invisible, _ = cube(stage, '/Environment/InvisibleCollider', (.3, .2, .2))
    invisible.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(invisible.GetPrim())
    guide, _ = cube(stage, '/Helpers/Guide', (.4, -.35, .15))
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cube(stage, '/Franka/ExcludedLink', (.4, .35, .1))
    camera = UsdGeom.Camera.Define(stage, '/Helpers/Camera')
    pose(camera, (0., 0., .5))
    mesh = UsdGeom.Mesh.Define(stage, '/Environment/InvisibleUnsupportedMesh')
    points = [Gf.Vec3f(x, y, z) for x in (-.04, .04) for y in (-.04, .04) for z in (-.04, .04)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*6)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1,
                                    2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-.04), Gf.Vec3f(.04)])
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    pose(mesh, (.25, -.35, .15))
    # A referenced instance exercises actual instance-proxy traversal.
    asset = Usd.Stage.CreateInMemory()
    prototype = UsdGeom.Xform.Define(asset, '/Prototype')
    cube(asset, '/Prototype/Obstacle', (0., 0., 0.))
    instance = UsdGeom.Xform.Define(stage, '/Environment/Instance')
    instance.GetPrim().GetReferences().AddReference(asset.GetRootLayer().identifier, '/Prototype')
    instance.GetPrim().SetInstanceable(True)
    pose(instance, (-.4, .35, .1))
    assert stage.GetPrimAtPath('/Environment/Instance/Obstacle').IsInstanceProxy()
    excluded_payload = stage.DefinePrim('/Franka/UnloadedPayload', 'Xform')
    excluded_payload.GetPayloads().AddPayload(asset.GetRootLayer().identifier, '/Prototype')
    stage.Unload(excluded_payload.GetPath())
    stage.GetRootLayer().Export(str(output/'fixture.usda'))

    occupancy = OccupancyMap(GridSpec((-.6, -.6, -.1), (.6, .6, .6), .01))
    source = IsaacGroundTruthMapper(occupancy)
    updates, queries = [], []
    def rebuild():
        stamp = time.monotonic()
        assert source.update(stage, timestamp=stamp, now=stamp)
        snapshot = occupancy.snapshot(now=stamp)
        updates.append(dict(timestamp=stamp, counts=snapshot.counts(), health=dict(snapshot.health),
                            diagnostics=source.diagnostics))
        return snapshot
    def check(label, position, expected):
        query = occupancy.query(position)
        queries.append(dict(label=label, position=list(position), state=query.state.name))
        assert query.state == expected, queries[-1]
    try:
        snapshot = rebuild()
        for label, point in [('rotated_parent_box', (-.25, -.2, .15)),
                             ('ellipsoid', (.18, -.1, .18)), ('invisible_collider', (.3, .2, .2)),
                             ('invisible_mesh_aabb', (.25, -.35, .15)), ('instance_proxy', (-.4, .35, .1)),
                             ('moving_initial', (.42, -.1, .08))]+[
                             ('cylinder_'+axis, center) for axis, center in zip('XYZ', cylinder_centers)]:
            check(label, point, OccupancyState.OCCUPIED)
        for label, point in [('empty_camera_helper', (0., 0., .5)), ('excluded_robot', (.4, .35, .1)),
                             ('guide', (.4, -.35, .15)), ('rotated_aabb_corner', (-.176, -.126, .15)),
                             ('ellipsoid_aabb_corner', (.255, -.14, .18))]:
            check(label, point, OccupancyState.FREE)
        assert source.diagnostics['unsupported_aabb_prims'] == 1, source.diagnostics
        np.savez_compressed(output/'occupancy.npz', states=snapshot.states,
                            bounds_min=occupancy.spec.bounds_min, bounds_max=occupancy.spec.bounds_max,
                            voxel_size=occupancy.spec.voxel_size)
        translation.Set(Gf.Vec3d(.42, .12, .08))
        rebuild()
        check('vacated_after_move', (.42, -.1, .08), OccupancyState.FREE)
        check('occupied_after_move', (.42, .12, .08), OccupancyState.OCCUPIED)
        stage.RemovePrim(str(moving.GetPrim().GetPath()))
        rebuild()
        check('vacated_after_remove', (.42, .12, .08), OccupancyState.FREE)
        before = occupancy.snapshot()
        limited = IsaacGroundTruthMapper(occupancy, max_candidate_voxels=1)
        stamp = time.monotonic()
        try:
            limited.update(stage, timestamp=stamp, now=stamp)
        except RuntimeError as exc:
            assert 'budget' in str(exc), str(exc)
        else:
            raise AssertionError('Candidate budget failure was not reported')
        failed = occupancy.snapshot(now=stamp)
        assert (failed.timestamp, failed.revision) == (before.timestamp, before.revision)
        np.testing.assert_array_equal(before.states, failed.states)
        assert failed.health['status'] == 'error' and not failed.health['fresh']
        snapshot = rebuild()
        unloaded = stage.DefinePrim('/UnloadedEnvironment', 'Xform')
        unloaded.GetPayloads().AddPayload(asset.GetRootLayer().identifier, '/Prototype')
        stage.Unload(unloaded.GetPath())
        try:
            source.update(stage)
        except RuntimeError as exc:
            assert 'Unloaded USD geometry payload' in str(exc), str(exc)
        else:
            raise AssertionError('Unloaded environment payload was classified as free')
        assert occupancy.snapshot().timestamp == snapshot.timestamp
        stage.RemovePrim(unloaded.GetPath())
        snapshot = rebuild()
        centimeters = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(centimeters, .01)
        cube(centimeters, '/Box', (20., 0., 10.), size=10.)
        cm_mesh = UsdGeom.Mesh.Define(centimeters, '/Mesh')
        cm_mesh.CreatePointsAttr([point*100 for point in points])
        cm_mesh.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
        cm_mesh.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
        cm_mesh.CreateExtentAttr([Gf.Vec3f(-4.), Gf.Vec3f(4.)])
        pose(cm_mesh, (-20., 0., 10.))
        cm_map = OccupancyMap(GridSpec((-.4, -.2, 0.), (.4, .2, .2), .01))
        cm_source = IsaacGroundTruthMapper(cm_map)
        assert cm_source.update(centimeters)
        for point in ((.2, 0., .1), (-.2, 0., .1)):
            assert cm_map.query(point).state == OccupancyState.OCCUPIED
        assert cm_map.query((0., 0., .1)).state == OccupancyState.FREE
        cm_source.close()
        (output/'queries.json').write_text(json.dumps(queries, indent=2))
        (output/'updates.json').write_text(json.dumps(updates, indent=2))
        write_result(args.result, status='PASS', metrics=dict(dimensions=occupancy.spec.shape,
            voxel_size=occupancy.spec.voxel_size, counts=snapshot.counts(), health=dict(snapshot.health),
            updates=len(updates), update_ms=[u['diagnostics']['update_seconds']*1000 for u in updates],
            queries=len(queries), unsupported_aabb_prims=1, instance_proxies=True,
            moved_removed_free=True, invisible_colliders=True, failed_budget_preserved_timestamp=True,
            nonmeter_stage_units=True, unloaded_environment_failed=True, excluded_unloaded_robot=True,
            existing_scene_opened=False, existing_scenes_modified=False, robot_commands=0))
    finally:
        source.close()
        source.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir')
    args = parser.parse_args()
    code = 0
    try:
        main(args)
    except Exception as exc:
        fail_result(args.result, exc)
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
