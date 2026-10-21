"""Verify saved USD camera edits through the ordinary opt-in acquisition adapter.

No benchmark look-at override, robot backend, controller or skill is used.
The edited copy is a new experimental layer under the requested artifact path.
"""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected repository NumPy 1.26.4, got {np.__version__}')

import argparse
import json
from pathlib import Path


def run(args, app):
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from pxr import Gf, Sdf, Usd, UsdGeom
    from robot_skill_stack.integrations.isaac.config import load_scene_config
    from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
    from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider

    output = Path(args.artifacts_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_scene_config(Path(args.config).resolve())
    # These are existing camera prim paths, not algorithmic coordinates.
    paths = ('/World/PerceptionCamera', '/World/PerceptionCameraSide')

    def capture(scene):
        if not open_stage(str(scene)):
            raise RuntimeError(f'Could not open {scene}')
        for _ in range(600):
            if not is_stage_loading():
                break
            app.update()
        else:
            raise RuntimeError('Stage loading timed out')
        world = World(stage_units_in_meters=1.)
        world.reset()
        world.play()
        rig = create_cameras(paths)
        source = MultiViewProvider(rig, config.scene)
        try:
            for _ in range(60):
                world.step(render=True)
            transforms, intrinsics, observations = {}, {}, []
            for index in range(3):
                for _ in range(8):
                    world.step(render=True)
                batch = source.observe()
                if batch.timestamp is None:
                    raise AssertionError('No fresh synchronized observation')
                if index == 2:
                    observations = [dict(id=o.object_id, shape=o.geometry.shape.value,
                        center=o.pose.position.tolist(), orientation=None if o.pose.orientation is None
                        else o.pose.orientation.tolist()) for o in batch.observations]
                for path, camera in rig.cameras.items():
                    transforms[path] = camera.get_world_from_camera_transform().tolist()
                    intrinsics[path] = camera.get_intrinsics().tolist()
            frozen = source.observe()
            if frozen.timestamp is not None or frozen.observations:
                raise AssertionError('Repeated camera buffers published a new observation')
            return dict(transforms=transforms, intrinsics=intrinsics,
                        observations=observations, frozen_silent=True)
        finally:
            source.close()
            world.stop()
            World.clear_instance()

    scene = Path(args.scene).resolve()
    original = capture(scene)
    expected = json.loads(Path(args.expected_layout).read_text())
    matrices = expected.get('world_from_cameras', expected.get('layout', {}).get('world_from_cameras'))
    if matrices is None:
        raise ValueError('Expected-layout JSON must contain world_from_cameras')
    for path, matrix in zip(paths, matrices):
        np.testing.assert_allclose(original['transforms'][path], matrix, atol=2e-6, rtol=0)
    if not original['observations']:
        raise AssertionError('Saved demonstration scene yielded no confirmed objects')
    expected_objects = expected.get('objects', [])
    if not expected_objects:
        raise ValueError('Expected-layout JSON must contain demonstration truth objects')

    def verify_objects(snapshot):
        observations = snapshot['observations']
        if len(observations) != len(expected_objects) or len({o['id'] for o in observations}) != len(observations):
            raise AssertionError('Demo object count/unique identities differ from the saved scene')
        available, matched = set(range(len(observations))), []
        for truth in expected_objects:
            nearest = min(available, key=lambda index: np.linalg.norm(
                np.asarray(observations[index]['center'])-truth['center']))
            observation = observations[nearest]
            error = float(np.linalg.norm(np.asarray(observation['center'])-truth['center']))
            if observation['shape'] != truth['shape'] or error > .010:
                raise AssertionError(f'Demo target {truth["name"]} missing/wrong class/center')
            available.remove(nearest)
            matched.append(dict(truth=truth['name'], object_id=observation['id'], center_error_mm=error*1000))
        snapshot['demo_matches'] = matched

    verify_objects(original)
    if any(o['orientation'] is not None for o in original['observations'] if o['shape'] == 'sphere'):
        raise AssertionError('A featureless sphere was assigned an orientation')

    edited_path = output/'manual-camera-edit.usda'
    if edited_path.exists():
        raise FileExistsError(edited_path)
    layer = Sdf.Layer.CreateNew(str(edited_path))
    layer.subLayerPaths = [str(scene)]
    stage = Usd.Stage.Open(layer)
    offsets = ((.013, -.009, .006), (-.011, .007, .004))
    yaw_offsets_degrees = (2., -1.5)
    edited_expected = {}
    for path, offset, yaw_degrees in zip(paths, offsets, yaw_offsets_degrees):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Camera):
            raise AssertionError(f'{path} is not an ordinary USD Camera')
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        transform = np.asarray(original['transforms'][path]).copy()
        yaw = np.deg2rad(yaw_degrees)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                             [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        transform[:3, :3] = rotation @ transform[:3, :3]
        transform[:3, 3] += offset
        edited_expected[path] = transform
        world = Gf.Matrix4d(transform.T.tolist())
        # ROS optical +Z/down-Y -> USD optical -Z/up-Y.
        world = Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1)*world
        local = world*parent.GetInverse()
        attribute = prim.CreateAttribute('xformOp:transform', Sdf.ValueTypeNames.Matrix4d)
        attribute.Set(local)
        UsdGeom.Xformable(prim).CreateXformOpOrderAttr().Set(['xformOp:transform'])
    layer.Save()
    stage = None
    edited = capture(edited_path)
    verify_objects(edited)
    for path, offset in zip(paths, offsets):
        before, after = np.asarray(original['transforms'][path]), np.asarray(edited['transforms'][path])
        np.testing.assert_allclose(after[:3, 3], before[:3, 3]+offset, atol=2e-6, rtol=0)
        np.testing.assert_allclose(after[:3, :3], edited_expected[path][:3, :3], atol=2e-6, rtol=0)
        np.testing.assert_allclose(original['intrinsics'][path], edited['intrinsics'][path])
    return dict(status='PASS', original=original, saved_edit=edited,
        saved_layer=str(edited_path), translation_offsets=offsets,
        world_yaw_rotation_offsets_degrees=yaw_offsets_degrees,
        ordinary_camera_prims=list(paths), robot_commands=0,
        note='Transforms read at ordinary startup; neither capture repositions cameras.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-layout', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': False})
    code = 0
    try:
        result = run(args, app)
    except Exception as exc:
        result = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        code = 1
        import traceback
        traceback.print_exc()
    finally:
        path = Path(args.artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        app.close()
    raise SystemExit(code)
