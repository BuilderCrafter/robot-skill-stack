from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from pxr import Gf, Usd, UsdGeom

from robot_skill_stack.integrations.isaac.sensors.rgbd_camera import IsaacRgbdCamera
from robot_skill_stack.world.perception.factory import build_perception_provider
from robot_skill_stack.integrations.isaac.config import load_scene_config
from tests.support.results import save_pgm, save_ppm, write_result
from robot_skill_stack.world.model.world_model import WorldModel

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "scenes" / "playground.usd"
PROFILE = ROOT / "config" / "scenes" / "playground.toml"
MOVE_TARGET = np.array([0.42, 0.10, 0.025])
SECOND_POS = np.array([0.38, -0.14, 0.025])
SECOND_PATH = "/World/Objects/TestCubeB"


def visible(observations):
    return [o for o in observations if o.visible]


def nearest(observations, position):
    return min(
        observations,
        key=lambda o: np.linalg.norm(o.pose.position - position),
    )


def observe(provider, world, count=1, steps=6):
    observations = []
    for _ in range(count):
        for _ in range(steps):
            world.step(render=True)
        observations = provider.observe()
    return observations


def ground_truth(prim_path):
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(transform.ExtractTranslation(), dtype=float)


def main(result_path=None, artifacts_dir=None):
    metrics = {}
    artifacts = Path(artifacts_dir) if artifacts_dir else None
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)

    print("[1] Opening scene...")
    if not open_stage(str(SCENE)):
        raise RuntimeError(f"Failed to open {SCENE}")
    while is_stage_loading():
        simulation_app.update()

    config = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.0)
    cube = world.scene.add(
        SingleXFormPrim(
            prim_path=config.objects["cube"].prim_path,
            name="perception_v1_cube",
            reset_xform_properties=False,
        )
    )
    world.reset()
    world.play()

    print("[2] Creating geometry-only perception provider...")
    camera = IsaacRgbdCamera(
        config.perception.camera_prim_path,
        resolution=config.perception.resolution,
    )
    camera.initialize(semantic_segmentation=False)
    for _ in range(60):
        world.step(render=True)
    provider = build_perception_provider(camera, config)

    print("[3] Single-cube discovery...")
    obs = observe(provider, world, count=5)
    vis = visible(obs)
    metrics["single_visible_count"] = len(vis)
    if len(vis) != 1:
        raise AssertionError(f"Expected exactly one visible object, got {len(vis)}")

    track = vis[0]
    first_id = track.object_id
    truth = ground_truth(config.objects["cube"].prim_path)
    pos_error = float(np.linalg.norm(track.pose.position - truth))
    size_error = np.abs(track.size - np.array([0.05, 0.05, 0.05]))
    metrics.update(
        object_id=first_id,
        semantic_class=track.class_name,
        position_error_mm=pos_error * 1000,
        size_error_mm=(size_error * 1000).tolist(),
        class_confidence=track.metadata.get("class_confidence", 0.0),
    )
    print("ID:", first_id)
    print("Class:", track.class_name)
    print("Position:", np.round(track.pose.position, 6))
    print("Truth:", np.round(truth, 6))
    print("Position error:", f"{pos_error * 1000:.2f} mm")
    print("Size:", np.round(track.size, 6))
    print("Size error [mm]:", np.round(size_error * 1000, 2))
    assert pos_error < 0.010
    assert float(np.max(size_error)) < 0.012

    if artifacts:
        rgb = camera.get_rgb()
        if rgb is not None:
            save_ppm(artifacts / "single_cube_rgb.ppm", rgb)
        mask = provider.get_mask(first_id)
        if mask is not None:
            save_pgm(artifacts / "single_cube_mask.pgm", mask.astype(np.uint8) * 255)

    print("[4] Static ID stability...")
    ids = []
    for _ in range(6):
        current = visible(observe(provider, world, count=1))
        if len(current) != 1:
            raise AssertionError("Single-cube scene lost stable single-object discovery")
        ids.append(current[0].object_id)
    metrics["static_id_stable"] = all(object_id == first_id for object_id in ids)
    assert metrics["static_id_stable"]

    print("[5] Moving cube; ID must remain stable...")
    _, orientation = cube.get_world_pose()
    cube.set_world_pose(position=MOVE_TARGET, orientation=orientation)
    obs = observe(provider, world, count=6, steps=8)
    vis = visible(obs)
    moved = nearest(vis, MOVE_TARGET)
    moved_error = float(np.linalg.norm(moved.pose.position - cube.get_world_pose()[0]))
    metrics["moved_id_stable"] = moved.object_id == first_id
    metrics["moved_position_error_mm"] = moved_error * 1000
    print("Moved ID:", moved.object_id)
    print("Moved error:", f"{moved_error * 1000:.2f} mm")
    assert moved.object_id == first_id
    assert moved_error < 0.015

    print("[6] Lazy point cloud request...")
    cloud = provider.get_point_cloud(first_id)
    metrics["lazy_point_cloud_points"] = 0 if cloud is None else int(len(cloud))
    print("Point cloud points:", metrics["lazy_point_cloud_points"])
    assert cloud is not None and len(cloud) > 50

    print("[7] Spawning an unconfigured second cube...")
    stage = omni.usd.get_context().get_stage()
    second = UsdGeom.Cube.Define(stage, SECOND_PATH)
    second.CreateSizeAttr(0.05)
    xform = UsdGeom.Xformable(second.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*SECOND_POS.tolist()))
    obs = observe(provider, world, count=6, steps=8)
    vis = visible(obs)
    metrics["two_cube_visible_count"] = len(vis)
    print("Visible IDs:", [o.object_id for o in vis])
    if len(vis) != 2:
        raise AssertionError(f"Expected two visible cubes, got {len(vis)}")

    first_after = nearest(vis, cube.get_world_pose()[0])
    second_obs = nearest(vis, SECOND_POS)
    metrics["first_id_preserved_with_second_cube"] = first_after.object_id == first_id
    metrics["second_object_id"] = second_obs.object_id
    second_error = float(np.linalg.norm(second_obs.pose.position - SECOND_POS))
    metrics["second_position_error_mm"] = second_error * 1000
    assert first_after.object_id == first_id
    assert second_obs.object_id != first_id
    assert second_error < 0.015

    print("[8] Removing second cube; track should persist but become invisible...")
    second_id = second_obs.object_id
    stage.RemovePrim(SECOND_PATH)
    obs = observe(provider, world, count=2, steps=8)
    second_after = next((o for o in obs if o.object_id == second_id), None)
    metrics["removed_track_retained"] = second_after is not None
    metrics["removed_track_invisible"] = bool(second_after is not None and not second_after.visible)
    assert second_after is not None and not second_after.visible

    print("[9] WorldModel integration...")
    model = WorldModel()
    model.apply_observations(obs, mark_missing_invisible=True)
    assert model.exists(first_id)
    assert model.exists(second_id)
    assert model.require(first_id).visible
    assert not model.require(second_id).visible
    metrics["world_model_integration"] = True

    print("\n=== PERCEPTION V1 DISCOVERY ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS")
    print("================================")
    write_result(result_path, status="PASS", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    try:
        main(args.result, args.artifacts_dir)
    except Exception as exc:
        traceback.print_exc()
        write_result(
            args.result,
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
        )
        sys.exit(1)
    finally:
        simulation_app.close()
