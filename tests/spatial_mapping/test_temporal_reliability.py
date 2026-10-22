"""Real PhysX entry, movement, removal and short/long occlusion at normal TTL5.

Exact integration evidence and exposure-aligned geometry are diagnostic outputs,
never mapping input. No product motion skills or planner are constructed.
"""
import numpy as np
if np.__version__ != '1.26.4':
    raise RuntimeError(f'Expected NumPy1.26.4, got {np.__version__}')

# Reuse the native startup and calibrated V3 fixture utilities: one Kit app only.
from tests.spatial_mapping.test_reliability import (
    simulation_app, argparse, hashlib, json, Path, time, patch, omni, World,
    SimulationManager, is_stage_loading, open_stage, ArticulationAction, Franka,
    Gf, UsdGeom, SpatialMappingConfig, create_spatial_mapping, load_scene_config,
    create_cameras, set_viewpoint, fail_result, write_result, ROOT, SCENE, PROFILE,
    FIXTURE, EYES, primitive, transform, projected, write_json)
from isaacsim.core.api.objects import DynamicCuboid
from pxr import PhysxSchema
from robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter import _pose_matrices
from robot_skill_stack.world.spatial_mapping.robot_filter import RobotSelfFilter
from tests.support.spatial_mapping_reliability import (
    ReferenceShape, neighborhood_indices, center_membership, score_voxel_centers,
    score_surface_events)


ACTOR = FIXTURE+'/temporal/PhysXMovingCube'
POSITION_A = np.array([.61, -.18, .065])
POSITION_B = np.array([.74, -.14, .065])
CONTROL = np.array([.36, .25, .10])


def references(payloads):
    return tuple(ReferenceShape.from_payload({key: value for key, value in payload.items()
        if key != 'authored_dimensions_m'}) for payload in payloads)


def robot_bounds(snapshots):
    bounds = []
    for snapshot in snapshots.values():
        for part in snapshot.parts:
            low, high = part.mesh.vertices.min(0), part.mesh.vertices.max(0)
            corners = np.array([[x, y, z] for x in (low[0], high[0])
                for y in (low[1], high[1]) for z in (low[2], high[2])])
            corners = corners @ part.world_from_local[:3, :3].T+part.world_from_local[:3, 3]
            bounds.append((corners.min(0), corners.max(0)))
    return bounds


def main(args):
    output = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.result).parent/'temporal_reliability'
    output.mkdir(parents=True, exist_ok=True)
    protected = (SCENE, PROFILE, ROOT/'scenes/multiview_playground.usda', ROOT/'scenes/playground.usd')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if not open_stage(str(SCENE)):
        raise RuntimeError('Could not open unchanged reference scene')
    for _ in range(600):
        if not is_stage_loading():
            break
        simulation_app.update()
    else:
        raise RuntimeError('Stage loading timed out')
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath('/World/FinalPerceptionObjects').SetActive(False)
    control = primitive(stage, 'temporal', 'HiddenMemoryControl', 'box', CONTROL, (.12, .12, .20))
    control_covers, actor_covers = [], []
    for index, eye in enumerate(map(np.asarray, EYES)):
        for target, label, destination in ((CONTROL, 'ControlCover', control_covers),
                (POSITION_A, 'ActorCover', actor_covers)):
            item = primitive(stage, 'temporal', label+str(index), 'box', (eye+target)/2,
                (.20, .20, .018), normal=eye-target)
            destination.append(item)
            stage.GetPrimAtPath(item['name']).SetActive(False)
    cfg = load_scene_config(PROFILE)
    world = World(stage_units_in_meters=1.)
    robot = world.scene.add(Franka(prim_path=cfg.robot.prim_path, name='temporal_reliability_robot'))
    actor = world.scene.add(DynamicCuboid(prim_path=ACTOR, name='temporal_reliability_cube',
        position=np.array([1.4, -1.1, .065]), size=.11, color=np.array([.12, .65, .25]), mass=.2))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(ACTOR)).CreateDisableGravityAttr(True)
    world.reset()
    world.play()
    home = robot.get_joint_positions().copy()
    actor.set_linear_velocity(np.zeros(3))
    actor.set_angular_velocity(np.zeros(3))
    view = SimulationManager.get_physics_sim_view().create_rigid_body_view(ACTOR)
    assert view.count == 1 and tuple(view.prim_paths) == (ACTOR,)
    rig = mapping = None
    rows, records, history = [], [], []
    state = dict(actor_active=True, control_covered=False, actor_covered=False)
    settings = SpatialMappingConfig(enabled=True, source='rgbd', automatic_robot_filter=True,
        bounds_min=(-.3, -.6, -.02), bounds_max=(.85, .6, 1.12), voxel_size=.01,
        camera_prim_paths=cfg.multi_view.camera_prim_paths, stride=8, max_rays=1024)
    manifest = dict(schema_version=1, status='RUNNING', records=records,
        settings={key: getattr(settings, key) for key in ('bounds_min', 'bounds_max', 'voxel_size',
            'stride', 'max_rays', 'max_voxel_visits', 'max_depth', 'max_skew',
            'max_frame_age', 'stale_after', 'evidence_ttl')},
        actor_path=ACTOR, control_path=control['name'],
        dynamic_fixture='Actual nonkinematic PhysX cuboid; gravity disabled, initial placement then linear velocity; moved outside mapped workspace for removal without changing rigid-body topology.',
        robot_commands='Fixture joint state/PD targets from existing V2 test; no backend or skill commands.',
        camera_eyes=EYES, camera_target=(.15, 0., .45),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scoring_scope='Complete fixture-local physical references above15mm, with explicitly counted possible-robot bounds omitted only from otherwise-free centers.')

    def record_pose():
        stamp = float(SimulationManager.get_simulation_time())
        matrix = None
        if state['actor_active']:
            raw = view.get_transforms()
            if hasattr(raw, 'cpu'):
                raw = raw.cpu()
            if hasattr(raw, 'numpy'):
                raw = raw.numpy()
            matrix = _pose_matrices(np.asarray(raw).copy(), 1.)[0]
        history.append((stamp, matrix))
        if len(history) > 4096:
            del history[0]

    def step(count=8):
        for _ in range(count):
            world.step(render=True)
            record_pose()

    def exposure(frame):
        stamp, matrix = min(history, key=lambda item: abs(item[0]-frame.token[1]))
        skew = abs(stamp-frame.token[1])
        assert skew <= settings.robot_pose_tolerance, (stamp, frame.token)
        active = [control]
        if state['control_covered']:
            active += control_covers
        if state['actor_covered']:
            active += actor_covers
        if matrix is not None:
            active.append(dict(name=ACTOR, kind='box', dimensions=[.11]*3,
                world_from_local=matrix.tolist(), reference_role='physical_and_visual'))
        latest = history[-1][1]
        displacement = None if matrix is None or latest is None else float(np.linalg.norm(latest[:3, 3]-matrix[:3, 3]))
        active = [{key: value for key, value in shape.items() if key != 'authored_dimensions_m'}
                  for shape in active]
        return active, dict(frame_simulation_time=frame.token[1], pose_simulation_time=stamp,
            pose_skew_seconds=skew, latest_simulation_time=history[-1][0],
            displacement_from_latest_m=displacement, actor_active_at_exposure=matrix is not None)

    try:
        rig = create_cameras(cfg.multi_view.camera_prim_paths)
        for camera, eye in zip(rig.cameras.values(), EYES):
            set_viewpoint(camera, eye, target=(.15, 0., .45))
            camera.camera.add_instance_id_segmentation_to_frame()
        callbacks_before = len(SimulationManager._callbacks)
        mapping = create_spatial_mapping(settings, stage=stage, robot_prim_path=cfg.robot.prim_path)
        record_pose()
        old_masks, old_generic, old_integrate = mapping.robot_filter.masks, RobotSelfFilter.masks, mapping.map.integrate
        retained = dict(masks={}, pixels={}, snapshots={}, evidence={})
        def generic(self, frames, snapshots_by_camera, *, pixels_by_camera=None):
            retained['snapshots'] = dict(snapshots_by_camera)
            return old_generic(self, frames, snapshots_by_camera, pixels_by_camera=pixels_by_camera)
        def masks(frames, *, pixels_by_camera=None):
            result = old_masks(frames, pixels_by_camera=pixels_by_camera)
            retained['masks'], retained['pixels'] = result, pixels_by_camera
            return result
        def integrate(free_indices, occupied_indices, *, timestamp, now=None):
            retained['evidence'] = dict(free_indices=free_indices, occupied_indices=occupied_indices,
                timestamp=timestamp, now=now)
            return old_integrate(free_indices, occupied_indices, timestamp=timestamp, now=now)

        def observe(phase, ordinal):
            step()
            if args.timing_only and rows:
                remaining = rows[-1]['original_acquired_at']+.5-time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            before = None if args.timing_only else mapping.map.snapshot()
            frames = rig.capture()
            started = time.perf_counter()
            with patch.object(RobotSelfFilter, 'masks', generic), \
                    patch.object(mapping.robot_filter, 'masks', side_effect=masks), \
                    patch.object(mapping.map, 'integrate', side_effect=integrate):
                updated = mapping.update(frames)
            completed = time.monotonic()
            elapsed = (time.perf_counter()-started)*1000
            health = dict(mapping.map.health)
            assert updated and health['fresh'], (phase, ordinal, health, mapping.source.diagnostics)
            assert mapping.map.timestamp == min(frame.acquired_at for frame in frames)
            if args.timing_only:
                row = dict(index=len(rows), phase=phase, ordinal=ordinal, update_ms=elapsed,
                    original_acquired_at=mapping.map.timestamp,
                    completion_age_ms=(completed-mapping.map.timestamp)*1000, health=health,
                    actual_joints=robot.get_joint_positions().tolist(),
                    actor_latest_center=history[-1][1][:3, 3].tolist(),
                    actual_capture_interval_seconds=None if not rows else
                        mapping.map.timestamp-rows[-1]['original_acquired_at'],
                    diagnostics=mapping.source.diagnostics)
                rows.append(row)
                print(f'Temporal timing {phase}/{ordinal}: {elapsed:.1f}ms age{row["completion_age_ms"]:.1f}ms', flush=True)
                return
            snapshot = mapping.map.snapshot(now=completed)
            index = len(records)
            record = dict(index=index, phase=phase, ordinal=ordinal, clear_map_before=False,
                cameras=[], exposure_shapes_by_camera={}, pose_match={}, snapshot_now=completed,
                evidence_file=f'evidence_{index:03d}.npz')
            row = dict(index=index, phase=phase, ordinal=ordinal, update_ms=elapsed,
                original_acquired_at=snapshot.timestamp, completion_age_ms=(completed-snapshot.timestamp)*1000,
                health=health, counts=snapshot.counts(), actual_joints=robot.get_joint_positions().tolist(),
                camera_visibility={}, evidence_counts={}, diagnostics=mapping.source.diagnostics)
            row['actor_latest_center'] = (None if history[-1][1] is None
                else history[-1][1][:3, 3].tolist())
            selected_points, selected_labels, selected_cameras = [], [], []
            retained_points, retained_labels, retained_cameras = [], [], []
            for camera_index, frame in enumerate(frames):
                payloads, match = exposure(frame)
                record['exposure_shapes_by_camera'][frame.camera_id] = payloads
                record['pose_match'][frame.camera_id] = match
                camera = rig.cameras[frame.camera_id]
                assert tuple(camera.get_frame_token()) == tuple(frame.token)
                packet = camera.camera.get_current_frame()['instance_id_segmentation']
                ids = np.asarray(packet['data']).squeeze().copy()
                paths = {str(key): value for key, value in packet['info']['idToLabels'].items()}
                assert tuple(camera.get_frame_token()) == tuple(frame.token)
                ys, xs = retained['pixels'][frame.camera_id]
                points = projected(frame, ys, xs)
                labels = np.array([paths.get(str(int(value)), '') for value in ids[ys, xs]])
                external = labels == ACTOR
                keep = ~retained['masks'][frame.camera_id][ys, xs]
                selected_points.extend(points[external]); selected_labels.extend(labels[external])
                selected_cameras.extend([frame.camera_id]*int(external.sum()))
                retained_points.extend(points[external & keep]); retained_labels.extend(labels[external & keep])
                retained_cameras.extend([frame.camera_id]*int((external & keep).sum()))
                row['camera_visibility'][frame.camera_id] = {name: int(np.isin(ids,
                    [int(key) for key, path in paths.items() if path == name]).sum())
                    for name in (ACTOR, control['name'])}
                filename = f'frame_{index:03d}_camera_{camera_index}.npz'
                np.savez(output/filename, depth=frame.depth, intrinsics=frame.intrinsics,
                    world_from_camera=frame.world_from_camera, token=frame.token,
                    acquired_at=frame.acquired_at, rgb=frame.rgb, selected_y=ys, selected_x=xs,
                    automatic_mask=retained['masks'][frame.camera_id], instance_ids=ids,
                    instance_paths_json=np.asarray(json.dumps(paths)))
                record['cameras'].append(dict(camera_id=frame.camera_id, file=filename,
                    token=frame.token, acquired_at=frame.acquired_at))
            groups = [references(payloads) for payloads in record['exposure_shapes_by_camera'].values()]
            domains = groups+[references([dict(name='scopeA', kind='box', dimensions=[.15]*3,
                world_from_local=transform(POSITION_A).tolist()), dict(name='scopeB', kind='box', dimensions=[.15]*3,
                world_from_local=transform(POSITION_B).tolist())])]
            indices = np.unique(np.concatenate([neighborhood_indices(snapshot.spec, group) for group in domains]), axis=0)
            centers = snapshot.spec.indices_to_centers(indices)
            valid = centers[:, 2] > .015
            indices, centers = indices[valid], centers[valid]
            occupied_any = np.logical_or.reduce([center_membership(centers, group) for group in groups])
            robot_possible = np.zeros(len(indices), bool)
            for low, high in robot_bounds(retained['snapshots']):
                robot_possible |= ((centers >= low) & (centers <= high)).all(1)
            excluded = indices[robot_possible & ~occupied_any]
            row['physical_center_metrics'] = score_voxel_centers(snapshot, groups[0], indices,
                other_exposures=groups[1:], excluded_indices=excluded)
            row['selected_actor_surface'] = score_surface_events(snapshot,
                np.asarray(selected_points).reshape(-1, 3), selected_labels, camera_ids=selected_cameras)
            row['retained_actor_surface'] = score_surface_events(snapshot,
                np.asarray(retained_points).reshape(-1, 3), retained_labels, camera_ids=retained_cameras)
            evidence = retained['evidence']
            hit_centers = snapshot.spec.indices_to_centers(evidence['occupied_indices'])
            previous, after = before.query_many(hit_centers), snapshot.query_many(hit_centers)
            payload = dict(evidence, integration_now=evidence['now'], snapshot_now=completed,
                evaluated_indices=indices, excluded_indices=excluded,
                map_timestamp=snapshot.timestamp, map_revision=snapshot.revision, map_generation=snapshot.generation)
            for label, queries in (('before', previous), ('after', after)):
                payload.update({label+'_'+field: getattr(queries, field) for field in
                    ('states', 'probabilities', 'observed', 'last_observed', 'expired')})
            np.savez(output/record['evidence_file'], **payload)
            checkpoint = (phase, ordinal) in (('seed_control', 5), ('precondition_free', 9),
                ('entered', 0), ('entered', 2), ('entered', 7), ('temporary_occlusion', 3),
                ('moving_with_robot', 13), ('removed', 13))
            if checkpoint:
                record['snapshot_file'] = f'snapshot_{index:03d}.npz'
                np.savez_compressed(output/record['snapshot_file'], states=snapshot.states,
                    probabilities=snapshot.probabilities, observed=snapshot.observed,
                    last_observed=snapshot.last_observed, expired=snapshot.expired,
                    bounds_min=snapshot.spec.bounds_min, bounds_max=snapshot.spec.bounds_max,
                    voxel_size=snapshot.spec.voxel_size, timestamp=snapshot.timestamp)
            row['evidence_counts'] = dict(free=len(evidence['free_indices']), occupied=len(evidence['occupied_indices']),
                integration_now=evidence['now'], snapshot_now=completed,
                directly_hit_previously_free=int((previous.states == 1).sum()),
                directly_hit_still_free=int(((previous.states == 1) & (after.states == 1)).sum()))
            row['actual_capture_interval_seconds'] = (None if not rows else
                snapshot.timestamp-rows[-1]['original_acquired_at'])
            rows.append(row); records.append(record)
            write_json(output/'samples.json', rows)
            write_json(output/'capture_manifest.json', manifest)
            print(f'Temporal {phase}/{ordinal}: {elapsed:.1f}ms, actorpixels{sum(v[ACTOR] for v in row["camera_visibility"].values())}, hitFREE{row["evidence_counts"]["directly_hit_still_free"]}', flush=True)

        step(24)
        for index in range(6):
            observe('seed_control', index)
        for cover in control_covers:
            stage.GetPrimAtPath(cover['name']).SetActive(True)
        state['control_covered'] = True
        for index in range(10):
            observe('precondition_free', index)
        actor.set_world_pose(position=POSITION_A)
        actor.set_linear_velocity(np.zeros(3))
        for index in range(8):
            observe('entered', index)
        for cover in actor_covers:
            stage.GetPrimAtPath(cover['name']).SetActive(True)
        state['actor_covered'] = True
        for index in range(4):
            observe('temporary_occlusion', index)
        for cover in actor_covers:
            stage.GetPrimAtPath(cover['name']).SetActive(False)
        state['actor_covered'] = False
        actor.set_linear_velocity(np.array([.07, .02, 0.]))
        target = np.array([.75, -.6, -.4, -2., .35, 1.55, .25, 0., 0.])
        for index in range(14):
            q = home+(target-home)*(index+1)/14
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros(len(q)))
            robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
            observe('moving_with_robot', index)
        actor.set_linear_velocity(np.zeros(3))
        # Removing a rigid prim invalidates the SDK's shared body views. This
        # fixture tests disappearance from the mapped workspace while preserving
        # topology; actual stage deletion requires explicit source reinitialization.
        actor.set_world_pose(position=np.array([1.4, -1.1, .065]))
        actor.set_angular_velocity(np.zeros(3))
        for index in range(14):
            observe('removed', index)
        seeded = [row for row in rows if row['phase'] == 'seed_control']
        hidden = [row for row in rows if row['phase'] == 'precondition_free']
        entered = [row for row in rows if row['phase'] == 'entered']
        occluded = [row for row in rows if row['phase'] == 'temporary_occlusion']
        moved = [row for row in rows if row['phase'] == 'moving_with_robot']
        if not args.timing_only:
            assert all(all(view[control['name']] >= 8 for view in row['camera_visibility'].values()) for row in seeded)
            assert all(all(view[control['name']] == 0 for view in row['camera_visibility'].values()) for row in hidden)
            assert all(all(view[ACTOR] >= 8 for view in row['camera_visibility'].values()) for row in entered)
            assert all(all(view[ACTOR] == 0 for view in row['camera_visibility'].values()) for row in occluded)
            assert any(row['retained_actor_surface']['overall']['unique_in_bounds_voxels'] > 0 for row in entered)
        assert np.linalg.norm(np.asarray(moved[-1]['actor_latest_center'])-moved[0]['actor_latest_center']) > .02
        assert np.linalg.norm(np.asarray(moved[-1]['actual_joints'])-home) > .1
        assert all(not all(np.asarray(settings.bounds_min) <= row['actor_latest_center']) or
                   not all(np.asarray(row['actor_latest_center']) < settings.bounds_max)
                   for row in rows if row['phase'] == 'removed')
        assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        manifest.update(status='PASS', saved_scenes_and_config_unchanged=True, batches=len(rows),
            timing_only=args.timing_only,
            fixture_contracts=dict(control_both_visible_then_fully_hidden=None if args.timing_only else True,
                actor_both_visible_then_fully_hidden=None if args.timing_only else True,
                in_bounds_selected_actor_hits=None if args.timing_only else True,
                actual_physx_motion=True, actual_robot_joint_change=True, workspace_removal=True))
        write_json(output/'samples.json', rows)
        write_json(output/'pose_history.json', [dict(simulation_time=stamp,
            world_from_local=None if matrix is None else matrix.tolist()) for stamp, matrix in history])
        write_json(output/'capture_manifest.json', manifest)
    finally:
        if mapping is not None:
            mapping.close()
        if rig is not None:
            rig.close()
        if 'callbacks_before' in locals():
            assert len(SimulationManager._callbacks) == callbacks_before
    write_result(args.result, status='PASS', metrics=dict(batches=len(rows), measurements=rows,
        settings=manifest['settings'], saved_scenes_and_config_unchanged=True, cleaned_up=True,
        timing_only=args.timing_only,
        timing_scope=('Clean update+freshness on validated fixture with >=.5s acquisition spacing; no oracle scoring/snapshot/NPZ per batch. Independent visibility quality is established by the companion full capture.'
                      if args.timing_only else 'Quality capture: update timer excludes scoring/I/O, but real capture cadence includes them.'),
        accuracy_gate='Completed measured fixtures/source contracts; reliability errors are retained metrics, not forced to zero.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--artifacts-dir')
    parser.add_argument('--timing-only', action='store_true')
    arguments = parser.parse_args()
    code = 0
    try:
        main(arguments)
    except Exception as error:
        fail_result(arguments.result, error)
        code = 1
    finally:
        simulation_app.close()
    raise SystemExit(code)
