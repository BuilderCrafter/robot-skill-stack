"""Three ordinary USD cameras through configuration/factory; no robot commands."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import hashlib
import json
from pathlib import Path

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Gf, Usd, UsdGeom

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.factory import create_source
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.isaac_async import run_kit_coroutine
from tests.support.results import fail_result, write_result

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
THIRD = '/World/PerceptionCameraThird'


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'three_cameras'
    output.mkdir(parents=True, exist_ok=True)
    protected = (SCENE, PROFILE, ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open the validated demo scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        simulation_app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    world = World(stage_units_in_meters=1.)
    world.reset()
    world.play()
    stage = omni.usd.get_context().get_stage()
    paths = (*load_scene_config(PROFILE).multi_view.camera_prim_paths, THIRD)
    original_poses = {path: np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).
        ComputeLocalToWorldTransform(Usd.TimeCode.Default())) for path in paths[:-1]}
    # All changes belong to this process's anonymous session layer, never a saved USD.
    stage.SetEditTarget(stage.GetSessionLayer())
    third = UsdGeom.Camera.Define(stage, THIRD)
    reference = stage.GetPrimAtPath(paths[1])
    for name in ('projection', 'focalLength', 'horizontalAperture', 'verticalAperture', 'clippingRange'):
        third.GetPrim().GetAttribute(name).Set(reference.GetAttribute(name).Get())
    pose = original_poses[paths[1]].copy()
    pose[3, :3] += [.08, 0., .03]
    UsdGeom.Xformable(third.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*pose.ravel().tolist()))
    text = PROFILE.read_text()
    old = next(line for line in text.splitlines() if line.startswith('camera_prim_paths ='))
    profile = output/'three_cameras.toml'
    profile.write_text(text.replace(old, 'camera_prim_paths = '+json.dumps(paths)))
    config = load_scene_config(profile)
    source = run_kit_coroutine(create_source(config), simulation_app)
    model = WorldModel()
    updater = WorldModelUpdater(model, source)
    captures = []
    capture = source.rig.capture

    def record_capture():
        frames = capture()
        if frames is not None:
            assert tuple(frame.camera_id for frame in frames) == paths
            assert all(frame.depth.shape == (480, 640) for frame in frames)
            captures.append([dict(camera_id=frame.camera_id, token=frame.token,
                acquired_at=frame.acquired_at, intrinsics=frame.intrinsics.tolist(),
                world_from_camera=frame.world_from_camera.tolist()) for frame in frames])
        return frames

    source.rig.capture = record_capture

    def observe():
        for _ in range(8):
            world.step(render=True)
        updater.update()

    try:
        assert tuple(source.rig.cameras) == paths
        for _ in range(4):
            observe()
            assert updater.health.status == 'fresh', updater.health.error
        visible = model.visible_objects()
        assert len(visible) == 4, [(obj.object_id, obj.class_name) for obj in visible]
        assert sorted(obj.class_name for obj in visible) == ['cube', 'cube', 'sphere', 'sphere']
        ids = {obj.object_id for obj in visible}
        assert all(obj.metadata['camera_count'] == 3 for obj in visible)
        contributing = {view for obj in visible for view in obj.metadata['contributing_views']}
        assert contributing == set(paths), contributing
        assert source.diagnostics['workspace_points'] <= 12000
        assert source.diagnostics['numerical_fits'] <= 24
        assert source.observe().timestamp is None, 'Repeated buffers published a new batch'
        last = updater.health.last_frame_at
        source.rig.cameras[THIRD].camera.pause()
        observe()
        assert updater.health.status == 'failed', updater.health.error
        assert THIRD in updater.health.error
        assert updater.health.last_frame_at == last and not model.visible_objects()
        source.rig.cameras[THIRD].camera.resume()
        observe()
        assert updater.health.status == 'fresh', updater.health.error
        assert {obj.object_id for obj in model.visible_objects()} == ids
        for path, before in original_poses.items():
            after = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).
                ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)
        assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        (output/'captures.json').write_text(json.dumps(captures, indent=2)+'\n')
    finally:
        updater.stop()
        source.close()
        for camera in source.rig.cameras.values():
            assert camera._closed and camera.camera._render_product is None
            assert camera.camera._fabric_time_annotator is None
        stage.RemovePrim(THIRD)
    write_result(args.result, status='PASS', metrics=dict(camera_paths=paths, camera_count=3,
        fresh_batches=len(captures), known_objects=4, persistent_ids=sorted(ids),
        contributing_views=sorted(contributing), repeated_buffers_silent=True,
        frozen_third_camera_failed=True, resumed_original_ids=True, all_cameras_closed=True,
        existing_camera_poses_unchanged=True, saved_scenes_and_config_unchanged=True, robot_commands=0))


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
