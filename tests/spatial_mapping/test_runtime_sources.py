"""Native automatic mapping source independence and shared camera ownership."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import omni.kit.app
import omni.usd
from isaacsim.core.utils.stage import open_stage, is_stage_loading
from pxr import Gf, UsdGeom
from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from scripts.diagnose_isaac_runtime import ROOT, SCENE, PROFILE, EYES, hashes, serial
from tests.support.isaac_async import run_kit_coroutine
from tests.support.multiview_native import set_viewpoint

CAMERAS = ('/World/PerceptionCamera', '/World/PerceptionCameraSide', '/World/V5SourceCamera')


async def until(predicate, timeout=12.):
    started = time.monotonic()
    while not predicate():
        if time.monotonic()-started >= timeout:
            raise AssertionError('Automatic source/ownership condition did not complete within12s')
        await omni.kit.app.get_app().next_update_async()


def profile(output, name, semantic, mapping, cameras):
    text = (ROOT/'config/scenes/playground.toml').read_text() if semantic == 'ground_truth' else PROFILE.read_text()
    if semantic == 'ground_truth':
        text = text.replace('provider = "perception"', 'provider = "ground_truth"', 1)
    text += '\n[spatial_mapping]\n'
    text += f'enabled = true\nsource = "{mapping}"\n'
    text += 'bounds_min = [-0.3,-0.6,-0.02]\nbounds_max = [0.85,0.6,1.12]\nvoxel_size = 0.01\n'
    text += f'camera_prim_paths = {json.dumps(cameras)}\n'
    text += 'stride = 8\nmax_rays = 1024\nupdate_hz = 2.0\n'
    text += f'automatic_robot_filter = {str(mapping == "rgbd").lower()}\n'
    text += 'max_frame_age = 1.0\nstale_after = 1.0\nevidence_ttl = 5.0\n'
    path = output/(name+'.toml')
    path.write_text(text)
    return path


def third_camera(stage):
    stage.SetEditTarget(stage.GetSessionLayer())
    camera = UsdGeom.Camera.Define(stage, CAMERAS[2])
    camera.CreateFocalLengthAttr(24.)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.71625)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 10.))
    eye, target = np.array([.4, 1.1, .95]), np.array([.15, 0., .45])
    forward = (target-eye)/np.linalg.norm(target-eye)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[3, :3] = np.column_stack((right, up, -forward)).T, eye
    UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(*matrix.ravel().tolist()))


async def evaluate(output, rows):
    cases = [('gt_gt', 'ground_truth', 'ground_truth', []),
             ('gt_rgbd_single', 'ground_truth', 'rgbd', [CAMERAS[0]]),
             ('partial_overlap', 'multi_view', 'rgbd', [CAMERAS[0], CAMERAS[2]])]
    for name, semantic, mapping, paths in cases:
        bundle = None
        row = dict(case=name, checks={})
        rows.append(row)
        try:
            if name == 'partial_overlap':
                third_camera(omni.usd.get_context().get_stage())
            bundle = await build_runtime(profile(output, name, semantic, mapping, paths))
            coordinator, service = bundle.observation_runtime, bundle.spatial_mapping
            pool = coordinator.pool
            cameras = {} if pool is None else pool.cameras
            for key, camera in cameras.items():
                eye = EYES[0] if key == CAMERAS[0] else EYES[1] if key == CAMERAS[1] else (.4, 1.1, .95)
                set_viewpoint(camera, eye, target=(.15, 0., .45))
            seen, shared = {}, []
            if pool is not None:
                original_batch = pool.batch

                def batch(consumer, *args, **kwargs):
                    frames = original_batch(consumer, *args, **kwargs)
                    if frames and name == 'partial_overlap':
                        for frame in frames:
                            if frame.camera_id != CAMERAS[0]:
                                continue
                            key = tuple(frame.token)
                            previous = seen.get(key)
                            if previous is not None and previous[0] != consumer:
                                shared.append(dict(token=serial(frame.token),
                                    same_object=previous[1] is frame,
                                    same_acquired_at=previous[1].acquired_at == frame.acquired_at,
                                    immutable=not frame.depth.flags.writeable))
                            seen[key] = (consumer, frame)
                            while len(seen) > 64:
                                seen.pop(next(iter(seen)))
                    return frames
                pool.batch = batch
            # Pose changes must be followed by genuinely rendered exposures;
            # bootstrap may still have an in-flight pre-override image.
            for _ in range(60):
                await omni.kit.app.get_app().next_update_async()
            await until(lambda: bundle.world_updater.health.status == 'fresh' and service.health['fresh'])
            checks = row['checks']
            checks['semantic_provider_unchanged'] = bundle.config.world.provider == semantic
            checks['mapping_source_unchanged'] = service.settings.source == mapping
            expected = 0 if mapping == 'ground_truth' else 1 if semantic == 'ground_truth' else 3
            checks['exact_camera_objects'] = len(cameras) == len({id(camera) for camera in cameras.values()}) == expected
            products = [camera.camera.get_render_product_path() for camera in cameras.values()]
            checks['unique_render_products'] = (len(set(products)) == expected and all(products))
            expected_owned = () if mapping == 'ground_truth' else (CAMERAS[0],) if semantic == 'ground_truth' else (CAMERAS[2],)
            owned = tuple(camera for camera in coordinator._owned_cameras)
            checks['only_missing_mapping_cameras_owned'] = (len(owned) == len(expected_owned)
                and all(cameras[key] in owned for key in expected_owned))
            if name == 'partial_overlap':
                checks['semantic_cameras_borrowed'] = all(
                    bundle.world_updater.provider.rig.cameras[key] is cameras[key] for key in CAMERAS[:2])
                await until(lambda: bool(shared))
                checks['overlap_exact_same_frame'] = all(
                    item['same_object'] and item['same_acquired_at'] and item['immutable'] for item in shared)
                before_semantic = coordinator.diagnostics['semantic']['published']
                before_failures = service.health['source_failures']
                with patch.object(cameras[CAMERAS[2]], 'get_depth', side_effect=RuntimeError('native camera C failure')):
                    await until(lambda: service.health['source_failures'] > before_failures
                        and coordinator.diagnostics['semantic']['published'] > before_semantic+1)
                    row['partial_failure'] = serial(coordinator.diagnostics)
                    checks['camera_c_failure_keeps_semantic_ab_fresh'] = bundle.world_updater.health.status == 'fresh'
                    checks['camera_c_failure_map_not_fresh'] = not service.health['fresh']
                await until(lambda: service.health['fresh'])
                checks['camera_c_recovered'] = True
                row['shared_a_frames'] = shared[:4]
            row.update(health=serial(coordinator.diagnostics), cameras={key: dict(
                object_id=id(camera), render_product=products[index],
                owned_by_mapping=camera in owned) for index, (key, camera) in enumerate(cameras.items())})
        finally:
            if bundle is not None:
                cameras = {} if bundle.observation_runtime.pool is None else dict(bundle.observation_runtime.pool.cameras)
                bundle.close()
                row['checks']['all_camera_resources_closed'] = all(
                    camera._closed and camera.camera._render_product is None for camera in cameras.values())
                row['checks']['service_closed'] = bundle.spatial_mapping.health['closed']


def main(args):
    output = Path(args.artifacts_dir); output.mkdir(parents=True, exist_ok=True)
    before, rows, error = hashes(), [], None
    try:
        if not open_stage(str(SCENE)):
            raise RuntimeError('Could not open existing scene')
        for _ in range(600):
            if not is_stage_loading():
                break
            simulation_app.update()
        else:
            raise RuntimeError('Stage loading timed out')
        stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
        stage.GetPrimAtPath('/World/Objects/Cube').SetActive(True)
        run_kit_coroutine(evaluate(output, rows), simulation_app)
    except Exception:
        error = traceback.format_exc()
    checks = dict(all_three_cases_completed=len(rows) == 3,
        unchanged_source_and_saved_scenes=before == hashes(),
        case_contracts=all(row['checks'] and all(row['checks'].values()) for row in rows))
    result = dict(status='PASS' if error is None and all(checks.values()) else 'FAIL',
        checks=checks, cases=rows, error=error, hashes=before,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(serial(result), indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], checks=checks, error=error)), flush=True)
    return result['status'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()
    try:
        passed = main(args)
    finally:
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
