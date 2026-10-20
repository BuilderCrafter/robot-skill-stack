"""Native multi-camera detection, freshness and lifecycle; no robot commands."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import omni.usd
from pxr import Gf, UsdGeom

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.world.factory import create_source
from robot_skill_stack.integrations.isaac.world import multiview as adapter
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.isaac_async import run_kit_coroutine
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, research_objects
from tests.support.results import fail_result, write_result


def main(args):
    output = Path(args.artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    world = open_experiment(simulation_app)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    base = research_objects()
    positions = ((.34, -.16), (.48, .14), (.54, -.15), (.36, .15))
    objects = [replace(obj, center=np.r_[xy, obj.center[2]]) for obj, xy in
               zip((base[0], base[2], base[4], base[7]), positions)]
    populate(objects)
    source = run_kit_coroutine(create_source(cfg), simulation_app)
    model = WorldModel()
    updater = WorldModelUpdater(model, source, stale_object_ttl=2.5)
    samples = []

    def observe(count=1):
        for _ in range(count):
            for _ in range(8):
                world.step(render=True)
            batch = updater.update()
            samples.append(dict(health=updater.health.status, observations=len(batch),
                                diagnostics=source.diagnostics))
            (output/'samples.json').write_text(json.dumps(samples, indent=2))
        print('Visible:', [(o.object_id, o.class_name) for o in model.visible_objects()], flush=True)

    def check_closed(provider):
        for camera in provider.rig.cameras.values():
            assert camera._closed
            assert camera.camera._acquisition_callback is None
            assert camera.camera._fabric_time_annotator is None
            assert camera.camera._render_product is None

    try:
        observe(4)
        visible = model.visible_objects()
        assert len(visible) == 4, ([(o.object_id, o.class_name) for o in visible], source.diagnostics)
        assignment = {}
        for obj in objects:
            nearest = min(visible, key=lambda o: np.linalg.norm(o.pose.position-obj.center))
            assert nearest.class_name == obj.shape, (obj.name, nearest.class_name, nearest.geometry.scores)
            assert np.linalg.norm(nearest.pose.position-obj.center) < .010
            assignment[obj.name] = nearest.object_id
        assert len(set(assignment.values())) == 4, 'Duplicate or mixed identities'
        observe(2)
        assert {o.object_id for o in model.visible_objects()} == set(assignment.values())

        camera = source.rig.cameras[CAMERAS[1]]
        last = updater.health.last_frame_at
        camera.camera.pause()
        observe()
        assert updater.health.status == 'failed', updater.health.error
        assert updater.health.last_frame_at == last
        assert not model.visible_objects()
        camera.camera.resume()
        observe()
        assert updater.health.status == 'fresh', updater.health.error

        for camera in source.rig.cameras.values():
            camera.camera.pause()
        last = updater.health.last_frame_at
        until = time.monotonic()+1.1
        while time.monotonic() < until:
            world.step(render=True)
        updater.update()
        assert updater.health.last_frame_at == last
        assert updater.health.status == 'stale'
        for camera in source.rig.cameras.values():
            camera.camera.resume()
        observe()
        assert updater.health.status == 'fresh'

        sphere = objects[1]
        key = assignment[sphere.name]
        model.set_held(key, uncertain=True)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath('/World/MultiViewObjects/'+sphere.name)
        prim.SetActive(False)
        until = time.monotonic()+2.7
        while time.monotonic() < until:
            observe()
        assert model.exists(key) and model.held_object_id == key
        target = np.array([.45, .29, sphere.radius])
        model.expect_object_at(key, target, ttl=10., reason='place')
        model.set_held(None)
        prim.SetActive(True)
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(*target.tolist()))
        observe(2)
        assert model.require(key).visible, (source.diagnostics, updater.health.error)
        assert np.linalg.norm(model.require(key).pose.position-target) < .010
        assert not model.association_hints()

        updater.stop()
        source.close()
        check_closed(source)
        failed_cameras = []
        camera_type = adapter.IsaacMultiViewCamera
        def recording_camera(*a, **kw):
            instance = camera_type(*a, **kw)
            failed_cameras.append(instance)
            return instance
        bad = replace(cfg, multi_view=replace(cfg.multi_view, camera_prim_paths=(CAMERAS[0], '/World/MissingCamera')))
        with patch.object(adapter, 'IsaacMultiViewCamera', side_effect=recording_camera):
            try:
                run_kit_coroutine(create_source(bad), simulation_app)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing camera initialization succeeded')
        assert len(failed_cameras) == 1 and failed_cameras[0]._closed
        assert failed_cameras[0].camera._render_product is None
        for _ in range(2):
            replacement = run_kit_coroutine(create_source(cfg), simulation_app)
            replacement.close()
            replacement.close()
            check_closed(replacement)
        write_result(args.result, status='PASS', metrics={
            'objects': 4, 'classes': [obj.shape for obj in objects],
            'persistent_ids': assignment, 'frozen_camera_failed': True,
            'both_frozen_stale': True, 'uncertain_identity_protected': True,
            'placement_hint_reacquired': True, 'partial_initialization_cleanup': True,
            'reinitializations': 2, 'robot_commands': 0,
            'timings_ms': [s['diagnostics'].get('total_ms') for s in samples]})
    finally:
        updater.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir', required=True)
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
