"""Research-only real-Franka observation fixture; commands are opt-in and fixed.

No Pick/Place, gripper action, joint target, controller change or object-oracle
input enters perception. Motion uses only the documented empty-gripper diagnostic
TCP sequence. All SDK access remains in this native test helper.
"""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.manipulation.factory import create_backend, create_robot
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from robot_skill_stack.world.perception_multiview.provider import MultiViewProvider
from robot_skill_stack.world.perception_multiview.clouds import crop_workspace, project_depth
from scripts.benchmark_multiview_robustness import save_frames
from tests.support.isaac_async import run_kit_coroutine
from tests.support.multiview_iteration2 import box, cylinder, object_payload, score_observations, sphere, surface_memberships
from tests.support.multiview_native import CAMERAS, ROOT, open_experiment, populate, set_viewpoint
from tests.support.results import write_result


HOVER = ((.45, 0., .25), (0., 0., 1., 0.))
SIDE = ((.45, .08, .25), (0., 0., 1., 0.))
MOTION_PROVENANCE = 'scripts/diagnose_isaac_control.py: hover, side (+80 mm Y), back; docs/control_diagnostic.md'


class RecordingRig:
    def __init__(self, rig): self.rig, self.frames = rig, None
    def capture(self):
        self.frames = None
        frames = self.rig.capture()
        if frames is not None:
            self.frames = frames
        return frames
    @property
    def status(self): return self.rig.status
    def close(self): self.rig.close()


class RecordingSource:
    def __init__(self, source): self.source, self.batch = source, None
    def observe(self, context=None):
        self.batch = None
        self.batch = self.source.observe(context)
        return self.batch
    def forget(self, key): self.source.forget(key)
    def close(self): self.source.close()


def run_robot_evaluation(app, result, artifacts, *, supervised_motion=False,
                         occlusion_fixtures=False, workspace_max_z=None,
                         temporal_viewpoints=False):
    output = Path(artifacts)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_scene_config(ROOT/'config/scenes/multiview_playground.toml')
    scene = cfg.scene
    if workspace_max_z is not None:
        maximum = scene.workspace_max.copy()
        maximum[2] = float(workspace_max_z)
        scene = replace(scene, workspace_max=maximum)
    world = open_experiment(app)
    robot = create_robot(world, cfg.robot)
    world.reset()
    world.play()
    # Preserve the configured support/workspace. An explicit expanded-height
    # ablation is recorded separately, rather than silently changing defaults.
    for _ in range(60):
        world.step(render=True)
    backend, _ = create_backend(world, robot, cfg.robot)
    rig = RecordingRig(create_cameras(CAMERAS))
    if temporal_viewpoints:
        # Deliberately overlapping viewing shadows exercise full occlusion and
        # recovery with the same validated robot motion. This is a separate
        # research stress arrangement, not a production camera recommendation.
        for camera, eye in zip(rig.rig.cameras.values(), ((1.05, -.10, 1.25), (1.05, .10, 1.25))):
            set_viewpoint(camera, eye)
    source = MultiViewProvider(rig, scene)
    recorder = RecordingSource(source)
    model = WorldModel()
    updater = WorldModelUpdater(model, recorder, stale_object_ttl=15.)
    samples, rows, counts, commands = [], [], [], []
    manifest = dict(kind='real_franka_native_snapshots', split='robot', scenes=[],
                    motion_supervised=supervised_motion, motion_provenance=MOTION_PROVENANCE,
                    occlusion_fixtures=occlusion_fixtures,
                    temporal_viewpoints=temporal_viewpoints,
                    workspace_min=scene.workspace_min.tolist(), workspace_max=scene.workspace_max.tolist())
    current, objects, identities = None, (), {}
    last_delivery = time.perf_counter()
    object_set = (box('robot_probe_box', (.35, -.18), (.05, .04, .05), .27),
                  sphere('robot_probe_sphere', (.51, .18), .025),
                  cylinder('robot_probe_cylinder', (.52, -.17), .019, .075))
    if occlusion_fixtures:
        # Test construction only: the initial native run's measured arm depths
        # locate these viewing shadows. No robot pose/trajectory is changed.
        # Tall targets permit arm occlusion inside the original tabletop ROI.
        object_set = (box('shadow_A_box', (.245, 0.), (.035, .035, .15), 0.),
                      cylinder('shadow_B_cylinder', (.55, .25), .018, .16),
                      box('near_gripper_box', (.45, 0.), (.045, .045, .18), .1))
    if temporal_viewpoints:
        object_set = (sphere('shadow_recovery_sphere', (.30, 0.), .024),)

    def provenance():
        paths = sorted((ROOT/'robot_skill_stack/world/perception_multiview').glob('*.py'))
        paths += [Path(__file__), ROOT/'scripts/benchmark_multiview_robot.py']
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    (output/'provenance_start.json').write_text(json.dumps(provenance(), indent=2)+'\n')

    def flush():
        for name, data in (('samples', samples), ('rows', rows), ('detections', counts),
                           ('capture_manifest', manifest), ('commands', commands)):
            (output/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')

    def phase(name, targets):
        nonlocal current, objects
        objects = tuple(targets)
        populate(objects)
        current = dict(key=name, name=name, split='robot', seed=0, gap_mm=None, temporal=True,
                       note='Actual simulated Franka; no object truth enters the provider', clutter=[], frames=[],
                       needs_confirmation=any(obj.name not in identities for obj in objects))
        manifest['scenes'].append(current)
        flush()

    def sample():
        nonlocal identities, last_delivery
        started = time.perf_counter()
        updater.update()
        finished = time.perf_counter()
        delivered_at = time.monotonic()
        batch = recorder.batch
        frames = rig.frames
        if batch is None or batch.timestamp is None or frames is None:
            samples.append(dict(phase=current['key'], frame_index=None, source_health=updater.health.status,
                source_error=updater.health.error, fresh_capture=False, batch_timestamp=None,
                delivered_at=delivered_at, delivery_ms=(finished-started)*1000,
                instrumented_loop_ms=(finished-last_delivery)*1000, diagnostics=dict(source.diagnostics)))
            flush()
            last_delivery = finished
            return
        index = len(current['frames'])
        files = save_frames(output, current['key'], index, frames)
        visibility = {}
        for camera_index, frame in enumerate(frames):
            cloud = crop_workspace(project_depth(frame)[0], scene)
            labels = surface_memberships(cloud, objects)
            visibility[str(camera_index)] = {obj.name: int(np.count_nonzero(labels == oi))
                                              for oi, obj in enumerate(objects)}
        current['frames'].append(dict(index=index, files=files, objects=[object_payload(obj) for obj in objects],
                                      visible_surface_points=visibility))
        observations = batch.observations
        selected, count, present = score_observations(objects, observations,
            [source.get_point_cloud(obs.object_id) for obs in observations],
            method='improved_live', trial=current['key'], frame_index=index, previous_ids=identities)
        identities.update(present)
        rows.extend(selected)
        count['confirmation_frame'] = index < 2 and current['needs_confirmation']
        counts.append(count)
        samples.append(dict(phase=current['key'], frame_index=index, source_health=updater.health.status,
            source_error=updater.health.error, batch_timestamp=batch.timestamp,
            captured_at=min(frame.acquired_at for frame in frames), delivered_at=delivered_at, fresh_capture=True,
            delivery_ms=(finished-started)*1000,
            acquisition_to_delivery_ms=(delivered_at-min(frame.acquired_at for frame in frames))*1000,
            instrumented_loop_ms=(finished-last_delivery)*1000, physics_step=int(world.current_time_step_index),
            simulation_time_s=float(world.current_time), joint_positions=robot.get_joint_positions().tolist(),
            tcp_position=backend.get_end_effector_pose().position.tolist(), diagnostics=dict(source.diagnostics)))
        flush()
        last_delivery = finished

    def stationary(count=6):
        for _ in range(count):
            for _ in range(8):
                world.step(render=True)
            sample()

    async def move(name, target):
        phase(name, object_set)
        task = asyncio.ensure_future(backend.move_to_pose_async(Pose(*target), position_tolerance=.02,
                                                               orientation_tolerance=.10))
        updates = 0
        try:
            while not task.done():
                await backend._next_update()
                updates += 1
                if updates % 8 == 0:
                    sample()
            outcome = await task
        except BaseException:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise
        commands.append(dict(name=name, target_position=list(target[0]), target_orientation=list(target[1]),
                             ok=outcome.ok, message=outcome.message, details=outcome.details))
        flush()
        if not outcome.ok:
            raise RuntimeError(f'{name}: {outcome.message}')

    try:
        for _ in range(60):
            world.step(render=True)
        phase('rest_empty', ())
        stationary()
        phase('rest_objects', object_set)
        stationary()
        phase('moving_object_stationary_robot', object_set)
        stationary(3)
        for offset in (.008, .016, .024):
            objects = tuple(replace(obj, center=obj.center+[offset, 0., 0.]) if index == 1 else obj
                            for index, obj in enumerate(object_set))
            populate(objects)
            stationary(2)
        if supervised_motion:
            run_kit_coroutine(move('documented_hover', HOVER), app)
            phase('hover_empty', ())
            stationary()
            phase('hover_stationary', object_set)
            stationary()
            run_kit_coroutine(move('documented_side', SIDE), app)
            phase('side_empty', ())
            stationary()
            phase('side_stationary', object_set)
            stationary()
            run_kit_coroutine(move('documented_return', HOVER), app)
            phase('retreated_objects', object_set)
            stationary()
        evaluated = [row for row in counts if not row['confirmation_frame']]
        evaluated_keys = {(row['trial'], row['frame_index']) for row in evaluated}
        evaluated_rows = [row for row in rows if (row['trial'], row['frame_index']) in evaluated_keys]
        metrics = dict(kind='real_franka_perception_evaluation', phases=len(manifest['scenes']), samples=len(samples),
            supervised_documented_motion_executed=supervised_motion, robot_commands=len(commands),
            pick_place_commands=0, gripper_commands=0,
            final_health=updater.health.status, source_failures=sum(row['source_health'] == 'failed' for row in samples),
            false_positives=sum(row['false_positives'] for row in evaluated),
            missing=sum(row['missing'] for row in evaluated), identity_switches=sum(row['identity_switches'] for row in evaluated),
            false_splits=sum(row['false_splits'] for row in evaluated), duplicate_ids=sum(row['duplicate_ids'] for row in evaluated),
            expected=sum(row['expected'] for row in evaluated),
            correct_classifications=sum(row['correct'] for row in evaluated_rows),
            unknown=sum(row['unknown'] for row in evaluated_rows),
            readiness_note='Measured coverage only; inspect per-phase occlusion/visibility before claiming moving-robot readiness',
            motion_gap=None if supervised_motion else 'Documented motion awaits workstation supervision; stationary coverage only')
        write_result(result, status='PASS', metrics=metrics)
        return metrics
    except BaseException as exc:
        flush()
        write_result(result, status='FAIL', metrics=dict(samples=len(samples), commands=commands),
                     error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'provenance_end.json').write_text(json.dumps(provenance(), indent=2)+'\n')
        updater.stop()
