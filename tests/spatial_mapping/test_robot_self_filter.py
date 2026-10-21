"""Native model-based robot filtering versus independent instance-ID references."""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy 1.26.4, got {np.__version__} from {np.__file__}')
print(f'Test NumPy: {np.__version__} from {np.__file__}', flush=True)
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False})

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, Usd, UsdGeom

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping import SharedFrames, SpatialMappingConfig, create_spatial_mapping
from robot_skill_stack.integrations.isaac.world.multiview import create_cameras
from robot_skill_stack.world.spatial_mapping import OccupancyState
from tests.support.multiview_native import set_viewpoint
from tests.support.results import fail_result, save_ppm, write_result

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT/'scenes/multiview_perception_final.usda'
PROFILE = ROOT/'config/scenes/multiview_perception_final.toml'
PROBES = ('forearm', 'wrist', 'gripper', 'foreground', 'occluded')


def sampled_reference(frame, camera, stride, robot_path):
    # Validation only. Production has no segmentation/instance annotation input.
    packet = camera.camera.get_current_frame()['instance_id_segmentation']
    image = np.asarray(packet['data']).squeeze()[::stride, ::stride]
    labels = packet['info']['idToLabels']
    robot_ids = [int(key) for key, path in labels.items() if path.startswith(robot_path+'/')]
    selections = {'robot': np.isin(image, robot_ids)}
    for name in PROBES:
        selections[name] = np.isin(image, [int(key) for key, path in labels.items()
                                         if path == '/World/SelfFilterProbes/'+name])
    y, x = np.indices(image.shape)
    y, x = y.ravel()*stride, x.ravel()*stride
    depth = frame.depth[y, x]
    k, transform = frame.intrinsics, frame.world_from_camera
    optical = np.c_[(x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones(len(x))]
    valid = np.isfinite(depth) & (depth > 0)
    points = (optical*np.where(valid, depth, 0.)[:, None]) @ transform[:3, :3].T+transform[:3, 3]
    return selections, points, valid


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'robot_self_filter'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SCENE, PROFILE)}
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open existing scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        simulation_app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    config = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.)
    robot = world.scene.add(Franka(prim_path=config.robot.prim_path, name='self_filter_fixture'))
    world.reset()
    world.play()
    home = robot.get_joint_positions().copy()
    rig = tap = filtered = baseline = truth = None
    rows, frame_records = [], []
    moving_robot_indices = set()
    targets = ((home, 'home_closed'),
        (np.array([0., -.45, 0., -1.65, 0., 1.4, .65, .04, .04]), 'extended_open'),
        (np.array([.75, -.6, -.4, -2., .35, 1.55, .25, 0., 0.]), 'rotated_closed'),
        (np.array([-.65, .25, .3, -1.6, -.3, 1.9, .6, .04, .04]), 'extended_side_open'))
    try:
        rig = create_cameras(config.multi_view.camera_prim_paths)
        for camera, eye in zip(rig.cameras.values(), ((1.2, -.85, .95), (-.65, .85, .85))):
            set_viewpoint(camera, eye, target=(.15, 0., .45))
            camera.camera.add_instance_id_segmentation_to_frame()  # Independent test oracle only.
        tap = SharedFrames(rig)
        settings = SpatialMappingConfig(enabled=True, source='rgbd', automatic_robot_filter=True,
            bounds_min=(-.3, -.6, -.02), bounds_max=(.85, .6, 1.12), voxel_size=.01,
            camera_prim_paths=config.multi_view.camera_prim_paths, stride=8, max_rays=1024,
            max_frame_age=5., stale_after=5., evidence_ttl=30.)
        callbacks_before = len(SimulationManager._callbacks)
        filtered = create_spatial_mapping(settings, stage=stage, frame_source=tap,
                                         robot_prim_path=config.robot.prim_path)
        baseline = create_spatial_mapping(replace(settings, automatic_robot_filter=False), frame_source=tap)
        truth = create_spatial_mapping(replace(settings, source='ground_truth'), stage=stage)
        probes = {}
        for name in PROBES:
            shape = UsdGeom.Cube.Define(stage, '/World/SelfFilterProbes/'+name)
            shape.CreateSizeAttr(.035 if name != 'foreground' else .05)
            shape.CreateDisplayColorAttr([Gf.Vec3f(.9, .15, .3)])
            probes[name] = UsdGeom.Xformable(shape).AddTranslateOp()

        def set_joints(q):
            # Fixture state/targets only; no change to the product controller or skills.
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros(len(q)))
            robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))

        def step(count=8):
            for _ in range(count):
                world.step(render=True)

        def place_probes():
            first = next(iter(rig.cameras.values()))
            transform = first.get_world_from_camera_transform()
            side, eye = transform[:3, 0], transform[:3, 3]
            centers = {}
            for name, link in (('forearm', 'panda_link5'), ('wrist', 'panda_link7'),
                               ('gripper', 'panda_hand'), ('foreground', 'panda_link4'),
                               ('occluded', 'panda_link3')):
                prim = stage.GetPrimAtPath(config.robot.prim_path+'/'+link+'/geometry/'+link)
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), float)
                pose = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                points = points @ pose[:3, :3].T+pose[:3, 3]
                direction = side if name not in ('foreground', 'occluded') else eye-points.mean(0)
                direction = direction/np.linalg.norm(direction)
                if name == 'occluded':
                    direction = -direction
                surface = points[np.argmax(points @ direction)]
                half = .025 if name == 'foreground' else .0175
                gap = .04 if name in ('foreground', 'occluded') else .01
                centers[name] = surface+direction*(half*np.abs(direction).sum()+gap)
                probes[name].Set(Gf.Vec3d(*centers[name].tolist()))
            return centers

        def evaluate(label, centers, fresh_batches=3, advance_steps=8, clear=True):
            if clear:
                baseline.map.clear()
                filtered.map.clear()
            timing = {'baseline_ms': [], 'filtered_ms': [], 'filter_ms': []}
            reference = None
            production_masks, production_pixels = {}, {}
            for _ in range(fresh_batches):
                step(advance_steps)
                frames = rig.capture()
                assert frames is tap.capture()
                before = time.perf_counter()
                assert baseline.update(), dict(baseline.map.health)
                timing['baseline_ms'].append((time.perf_counter()-before)*1000)
                original_masks = filtered.robot_filter.masks

                def record_masks(current, *, pixels_by_camera=None):
                    masks = original_masks(current, pixels_by_camera=pixels_by_camera)
                    production_masks.update(masks)
                    production_pixels.update(pixels_by_camera)
                    return masks

                before = time.perf_counter()
                with patch.object(filtered.robot_filter, 'masks', side_effect=record_masks):
                    updated = filtered.update()
                timing['filtered_ms'].append((time.perf_counter()-before)*1000)
                production_diag = filtered.robot_filter.diagnostics
                (output/'last_attempt.json').write_text(json.dumps(dict(configuration=label,
                    updated=updated, timings=timing, health=dict(filtered.map.health),
                    filter=production_diag, mapper=filtered.source.diagnostics), indent=2)+'\n')
                assert updated, dict(filtered.map.health)
                timing['filter_ms'].append(production_diag['elapsed_seconds']*1000)
                a, b = baseline.map.snapshot(), filtered.map.snapshot()
                # Read the model diagnostics/masks again only for independent evaluation.
                # This additional diagnostic call is excluded from the mapping timing.
                masks = filtered.robot_filter.masks(frames)
                diag = filtered.robot_filter.diagnostics
                reference = [(frame, *sampled_reference(frame, rig.cameras[frame.camera_id], settings.stride,
                                                       config.robot.prim_path))
                             for frame in frames]
            counts = {name: {'pixels': 0, 'masked': 0, 'unfiltered_occupied': 0,
                            'filtered_occupied': 0, 'unfiltered_free': 0, 'filtered_free': 0}
                      for name in ('robot', *PROBES)}
            selected_counts = {name: dict(value) for name, value in counts.items()}
            seen = {name: set() for name in counts}
            selected_seen = {name: set() for name in counts}
            occlusion_evidence = {}
            for frame, labels, points, valid in reference:
                mask = masks[frame.camera_id][::settings.stride, ::settings.stride].ravel()
                selected_image = np.zeros(frame.depth.shape, bool)
                selected_image[production_pixels[frame.camera_id]] = True
                selected_pixels = selected_image[::settings.stride, ::settings.stride].ravel()
                selected_mask = production_masks[frame.camera_id][::settings.stride, ::settings.stride].ravel()
                for name, selection in labels.items():
                    selection = selection.ravel() & valid
                    counts[name]['pixels'] += int(selection.sum())
                    counts[name]['masked'] += int((mask & selection).sum())
                    indices = a.spec.world_to_indices(points[selection])
                    seen[name].update(tuple(index) for index in indices if np.all(index >= 0))
                    production = selection & selected_pixels
                    selected_counts[name]['pixels'] += int(production.sum())
                    selected_counts[name]['masked'] += int((selected_mask & production).sum())
                    indices = a.spec.world_to_indices(points[production])
                    selected_seen[name].update(tuple(index) for index in indices if np.all(index >= 0))
                camera_point = np.linalg.inv(frame.world_from_camera) @ np.r_[centers['occluded'], 1.]
                u, v = np.rint(frame.intrinsics[:2, :2] @ (camera_point[:2]/camera_point[2])+
                               frame.intrinsics[:2, 2]-.5).astype(int)
                packet = rig.cameras[frame.camera_id].camera.get_current_frame()['instance_id_segmentation']
                inside = 0 <= v < frame.depth.shape[0] and 0 <= u < frame.depth.shape[1] and camera_point[2] > 0
                path = packet['info']['idToLabels'].get(str(int(np.asarray(packet['data']).squeeze()[v, u])), '') if inside else ''
                robot_in_front = (inside and path.startswith(config.robot.prim_path+'/') and
                                  frame.depth[v, u]+settings.robot_depth_tolerance < camera_point[2])
                occlusion_evidence[frame.camera_id] = dict(visible_probe_pixels=int(labels['occluded'].sum()),
                    projected_center_label=path, robot_in_front=bool(robot_in_front))
                frame_records.append(dict(configuration=label, camera_id=frame.camera_id,
                    token=frame.token, acquired_at=frame.acquired_at,
                    production_diagnostics=production_diag, dense_validation_diagnostics=diag))
            for inventory, destinations in ((seen, counts), (selected_seen, selected_counts)):
                for name, indices in inventory.items():
                    positions = a.spec.indices_to_centers(np.array(sorted(indices), int).reshape(-1, 3))
                    qa, qb = a.query_many(positions), b.query_many(positions)
                    destinations[name].update(reference_voxels=len(indices),
                        unfiltered_occupied=int((qa.states == OccupancyState.OCCUPIED).sum()),
                        filtered_occupied=int((qb.states == OccupancyState.OCCUPIED).sum()),
                        unfiltered_free=int((qa.states == OccupancyState.FREE).sum()),
                        filtered_free=int((qb.states == OccupancyState.FREE).sum()))
            if label.startswith('moving_'):
                moving_robot_indices.update(seen['robot'])
            assert counts['robot']['pixels'] > 0 and counts['robot']['masked'] > 0, counts
            assert counts['robot']['filtered_occupied'] < counts['robot']['unfiltered_occupied'], counts
            assert truth.update()
            actual = np.array(list(centers.values()))
            assert np.all(truth.map.query_many(actual).states == OccupancyState.OCCUPIED)
            center_queries = filtered.map.query_many(actual)
            false_free_centers = int((center_queries.states == OccupancyState.FREE).sum())
            assert false_free_centers == 0, (label, centers, center_queries)
            row = dict(configuration=label, actual_joints=robot.get_joint_positions().tolist(),
                counts=counts, production_selected_counts=selected_counts,
                baseline_counts=a.counts(), filtered_counts=b.counts(),
                false_free_probe_centers=false_free_centers, probe_centers={k: v.tolist() for k, v in centers.items()},
                center_states=[OccupancyState(int(state)).name for state in center_queries.states],
                timings=timing, health_at_integration=dict(b.health), occlusion_evidence=occlusion_evidence)
            rows.append(row)
            (output/'measurements.json').write_text(json.dumps(rows, indent=2)+'\n')
            if label == 'home_closed':
                np.savez_compressed(output/'home_maps.npz', unfiltered=a.states, filtered=b.states,
                    bounds_min=a.spec.bounds_min, bounds_max=a.spec.bounds_max, voxel_size=a.spec.voxel_size)
                for index, frame in enumerate(frames):
                    robot_ref = reference[index][1]['robot']
                    mask = masks[frame.camera_id][::settings.stride, ::settings.stride]
                    image = np.zeros((*mask.shape, 3), np.uint8)
                    image[:] = [90, 90, 90]
                    image[robot_ref] = [220, 55, 35]
                    image[mask & robot_ref] = [30, 170, 70]
                    image[mask & ~robot_ref] = [230, 190, 20]
                    save_ppm(output/f'home_mask_{index}.ppm', image)
                    np.savez_compressed(output/f'home_view_{index}.npz', depth=frame.depth,
                        rgb=frame.rgb, mask=masks[frame.camera_id], robot_reference=robot_ref,
                        intrinsics=frame.intrinsics, world_from_camera=frame.world_from_camera)
            print(label, json.dumps(dict(robot=counts['robot'], timings=timing)), flush=True)
            return frames, masks

        for q, label in targets:
            q = q.copy()
            if label == 'home_closed':
                q[-2:] = 0.
            set_joints(q)
            step(16)
            centers = place_probes()
            step(8)
            evaluate(label, centers)

        # Moving configurations exercise historical poses while frames lag live physics.
        q0, q1 = targets[-1][0], targets[1][0]
        baseline.map.clear()
        filtered.map.clear()
        for index in range(1, 13):
            q = q0+(q1-q0)*index/12
            set_joints(q)
            step(4)
            if index in (4, 8, 12):
                evaluate('moving_'+str(index), centers, fresh_batches=1, advance_steps=0, clear=False)

        assert truth.update()
        positions = filtered.map.spec.indices_to_centers(
            np.asarray(sorted(moving_robot_indices), int).reshape(-1, 3))
        environment_free = truth.map.query_many(positions).states == OccupancyState.FREE
        ghost_a = baseline.map.query_many(positions).states
        ghost_b = filtered.map.query_many(positions).states
        moving_robot_occupancy = dict(historical_reference_voxels_in_environment_free=int(environment_free.sum()),
            unfiltered_occupied=int(((ghost_a == OccupancyState.OCCUPIED) & environment_free).sum()),
            filtered_occupied=int(((ghost_b == OccupancyState.OCCUPIED) & environment_free).sum()))

        # Explicit delayed processing: old snapshots still use their original pose history.
        step()
        delayed = rig.capture()
        before_masks = filtered.robot_filter.masks(delayed)
        set_joints(targets[2][0])
        step(8)
        after_masks = filtered.robot_filter.masks(delayed)
        for key in before_masks:
            np.testing.assert_array_equal(before_masks[key], after_masks[key])
        assert filtered.update(delayed)
        previous = (filtered.map.timestamp, filtered.map.revision)
        assert not filtered.update(delayed)
        assert (filtered.map.timestamp, filtered.map.revision) == previous
        frozen = next(reversed(rig.cameras.values()))
        frozen.camera.pause()
        step()
        try:
            rig.capture()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Partially frozen camera batch accepted')
        try:
            filtered.update()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failed camera acquisition published a map')
        assert (filtered.map.timestamp, filtered.map.revision) == previous
        assert not filtered.map.health['fresh']
        frozen.camera.resume()
        step(12)
        rig.capture()
        assert filtered.update()
        filtered.close()
        filtered.close()
        assert len(SimulationManager._callbacks) == callbacks_before
        filtered = create_spatial_mapping(settings, stage=stage, frame_source=tap,
                                         robot_prim_path=config.robot.prim_path)
        step(12)
        rig.capture()
        assert filtered.update()
        totals = {name: {key: sum(row['counts'][name][key] for row in rows)
                        for key in ('pixels', 'masked', 'unfiltered_occupied', 'filtered_occupied', 'unfiltered_free', 'filtered_free')}
                  for name in ('robot', *PROBES)}
        selected_totals = {name: {key: sum(row['production_selected_counts'][name][key] for row in rows)
                        for key in ('pixels', 'masked', 'unfiltered_occupied', 'filtered_occupied', 'unfiltered_free', 'filtered_free')}
                          for name in ('robot', *PROBES)}
        for name in PROBES[:-1]:
            assert totals[name]['pixels'] > totals[name]['masked'], (name, totals[name])
            assert totals[name]['filtered_occupied'] > 0, (name, totals[name])
        assert any(view['robot_in_front'] for row in rows for view in row['occlusion_evidence'].values())
        assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SCENE, PROFILE)}
        (output/'frames.json').write_text(json.dumps(frame_records, indent=2)+'\n')
        metrics = dict(dimensions=filtered.map.spec.shape, voxel_size=settings.voxel_size,
            configurations=len(rows), measurements=rows, dense_stride_totals=totals,
            production_selected_totals=selected_totals, moving_robot_occupancy=moving_robot_occupancy,
            metric_scopes=dict(dense_stride='Independent instance-ID reference on every stride-8 valid depth pixel; pixel-frame counts sum over configurations.',
                production_selected='Exact fairly selected mapping pixels; voxel counts deduplicate within each configuration, then sum across configurations.',
                false_free='GT-occupied probe centers and instance-labeled surface voxel references only; not a complete volumetric certificate.',
                moving_robot_occupancy='Historical robot-reference cells in GT environment FREE; includes current robot volume, not only vacated cells.'),
            delayed_frames_use_historical_geometry=True, duplicate_tokens_rejected=True,
            source_failure_not_fresh=True, recovered=True, reinitialized=True,
            reference='instance_id_segmentation on existing render products, validation only',
            configured_robot_path=config.robot.prim_path, robot_commands='fixture joint states/PD targets only',
            saved_scenes_and_config_unchanged=True, maximum_depth_tolerance=settings.robot_depth_tolerance)
    finally:
        for resource in (filtered, baseline, truth, tap, rig):
            if resource is not None:
                resource.close()
        if rig is not None:
            assert all(camera._closed and camera.camera._render_product is None for camera in rig.cameras.values())
        if 'callbacks_before' in locals():
            assert len(SimulationManager._callbacks) == callbacks_before
    metrics['cleaned_up'] = True
    write_result(args.result, status='PASS', metrics=metrics)


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
