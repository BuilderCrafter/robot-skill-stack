"""Independent immutable input, exact-cell, expiry and component audit checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec, OccupancyMap
from robot_skill_stack.world.spatial_mapping.obstacles import ObstacleBox, ObstacleExtractionSettings, ObstacleExtractor, ObstaclePacket
from scripts.evaluate_obstacle_representation_v3 import (
    ARRAYS, CountingNumericInterface, audit_packet, bounds_mask, fragmentation,
    gap_audit, load_snapshot, packet_bounds, radius2_components, snapshot_metadata,
    export_capture, _motion_reference, physical_reference, convex_depth, reference_matches)


class RepresentationEvaluatorTests(unittest.TestCase):
    def snapshot(self, *, partial=False):
        spec = GridSpec((0., 0., 0.), ((.025 if partial else .04), .04, .04), .01)
        world = OccupancyMap(spec, clock=lambda: 10.)
        for index in range(4):
            world.integrate([], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], timestamp=9.+index*.1, now=9.+index*.1)
        return world.snapshot(now=10.)

    def test_cumulative_hit_mask_is_not_canonical_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'wrong.npz'
            np.savez(path, unique_hit=np.ones((2, 2, 2), bool), unique_free=np.zeros((2, 2, 2), bool))
            record = dict(snapshot_npz=str(path), snapshot=dict(bounds_min=[0, 0, 0], bounds_max=[.02]*3, voxel_size=.01))
            with self.assertRaisesRegex(ValueError, 'Not a canonical snapshot'):
                load_snapshot(record)

    def test_loaded_snapshot_arrays_and_health_are_irreversibly_readonly(self):
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'canonical.npz'
            np.savez(path, **{name: getattr(snapshot, name) for name in ARRAYS})
            record = dict(snapshot_npz=str(path), snapshot=snapshot_metadata(snapshot, 10., 5.))
            loaded = load_snapshot(record)
            for name in ARRAYS:
                values = getattr(loaded, name)
                with self.assertRaises(ValueError):
                    values.setflags(write=True)
            with self.assertRaises(TypeError):
                loaded.health['status'] = 'invented'

    def test_exact_box_lower_bound_distinguishes_isolation_and_nonrectangular_groups(self):
        mask = np.zeros((8, 8, 8), bool)
        mask[0, 0, 0] = mask[3, 3, 3] = mask[6, 6, 6] = True
        value = fragmentation(mask)
        self.assertEqual(value['face6_components'], 3)
        self.assertEqual(value['face6_singletons'], 3)
        self.assertEqual(value['exact_box_count_lower_bound'], 3)
        self.assertTrue(all(n == 3 for n in value['axis_order_candidate_counts'].values()))
        mask = np.zeros((3, 3, 3), bool)
        mask[0, 0, 0] = mask[1, 0, 0] = mask[0, 1, 0] = True
        value = fragmentation(mask)
        self.assertEqual(value['face6_components'], 1)
        self.assertEqual(value['exact_box_count_lower_bound'], 2)

    def test_radius2_evidence_connectivity_does_not_fill_a_voxel_gap(self):
        cells = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        self.assertEqual(radius2_components(cells)['count'], 1)
        mask, overlap = bounds_mask(np.array([[0, 1, 0, 1, 0, 1], [2, 3, 0, 1, 0, 1], [4, 5, 0, 1, 0, 1]]), (5, 1, 1))
        self.assertEqual(int(mask.sum()), 3)
        self.assertFalse(mask[1, 0, 0])
        self.assertEqual(overlap, 0)

    def test_packet_audit_detects_unknown_geometry_and_wrong_expiry(self):
        snapshot = self.snapshot()
        supported = snapshot.states == 2
        bad = ObstacleBox('bad', (.015, .015, .015), (.03, .03, .03), 999., 27)
        packet = ObstaclePacket(snapshot.generation, snapshot.revision, snapshot.timestamp, 10., (bad,), {})
        _, _, errors = audit_packet(snapshot, packet, supported, supported, 10., 5.)
        self.assertGreater(errors['unknown_cells'], 0)
        self.assertEqual(errors['wrong_expiry_boxes'], 1)

    def test_partial_last_voxel_uses_grid_upper_face_without_padding(self):
        snapshot = self.snapshot(partial=True)
        box = ObstacleBox('2:3:0:1:0:1', (.0225, .005, .005), (.005, .01, .01), 20., 1)
        packet = ObstaclePacket(0, 0, 9., 10., (box,), {})
        np.testing.assert_array_equal(packet_bounds(packet, snapshot.spec), [[2, 3, 0, 1, 0, 1]])

    def test_whole_voxel_gap_audit_separates_unknown_from_known_free(self):
        spec = GridSpec((0., 0., 0.), (.10, .04, .04), .01)
        world = OccupancyMap(spec, clock=lambda: 10.)
        for index in range(3):
            world.integrate([[4, 1, 1]], [], timestamp=9.+index*.1, now=9.+index*.1)
        snapshot = world.snapshot(now=10.)
        record = dict(evaluation_objects=[dict(name='left', center=[.02, .02, .02], dimensions=[.02]*3),
            dict(name='right', center=[.08, .02, .02], dimensions=[.02]*3)], physical_attribution_status='STATIC_REFERENCE_ONLY')
        report = gap_audit(record, snapshot, np.zeros(spec.shape, bool))
        self.assertTrue(report['regions'])
        self.assertGreater(report['regions'][0]['known_free_cells'], 0)
        self.assertGreater(report['regions'][0]['unknown_cells'], 0)
        self.assertEqual(report['regions'][0]['known_free_cells_covered'], 0)

    def test_existing_registry_reuses_unchanged_geometry_and_expires_old_packet(self):
        from robot_skill_stack.integrations.isaac.manipulation.motion.obstacles import RmpflowObstacleRegistry
        snapshot = self.snapshot()
        packet = ObstacleExtractor(ObstacleExtractionSettings(64, 0.), temporal_probabilities=True).extract(snapshot, now=10., evidence_ttl=5.)
        registry = RmpflowObstacleRegistry(CountingNumericInterface(), max_obstacles=64, meters_per_unit=1.)
        registry.sync(packet, now=10.)
        before = registry.diagnostics
        registry.sync(packet, now=10.1)
        after = registry.diagnostics
        self.assertEqual(before['additions'], after['additions'])
        self.assertEqual(before['removals'], after['removals'])
        registry.sync(packet, now=20.)
        self.assertEqual(registry.diagnostics['registered'], 0)

    def test_export_preserves_distinct_snapshot_and_extraction_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'runtime_profile.toml').write_text('[spatial_mapping]\nevidence_ttl=5.0\n')
            (folder/'result.json').write_text(json.dumps(dict(research=dict(layout_id='A_current'),
                representation_v3=dict(scenario='empty'), metrics=dict(configuration=dict(objects=[])))))
            snapshot = snapshot_metadata(self.snapshot(), 10., 5.)
            snapshot['health']['age_seconds'] = 10.-snapshot['timestamp']
            (folder/'arrays.npz').write_bytes(b'existing raw array bytes only hashed by export')
            (folder/'meta.json').write_text('{}')
            item = dict(snapshot=snapshot, accepted=True, arrays_npz='arrays.npz', metadata_json='meta.json',
                input_timestamp=9.3, job=1, packet=dict(extracted_at=10.02),
                clock_reads=[dict(phase='processor', value=10.), dict(phase='processor', value=10.02)])
            (folder/'jobs.jsonl').write_text(json.dumps(item)+'\n')
            record = export_capture(folder, 1, 'all')[0]
            self.assertEqual(record['snapshot']['snapshot_evaluated_at'], 10.)
            self.assertEqual(record['snapshot']['evaluated_at'], 10.02)
            self.assertEqual(record['layout_id'], 'A_current')
            self.assertEqual(record['scenario'], 'empty')

    def test_motion_reference_uses_shared_all_camera_origins_without_vertex_reads(self):
        from scripts.obstacle_representation_v3 import public_motion_context
        def part(x):
            transform = np.eye(4)
            transform[0, 3] = x
            return SimpleNamespace(world_from_local=transform)
        snapshots = {'b': SimpleNamespace(parts=(part(2.), part(1.))), 'a': SimpleNamespace(parts=(part(1.),))}
        payload = SimpleNamespace(robot_snapshots=snapshots, public_goal=(1., 2., 3.), public_corridor=((0., 0., 0.), (1., 2., 3.)))
        result = _motion_reference(payload, dict(goal=[9., 9., 9.]))
        expected = public_motion_context(snapshots, goal=payload.public_goal, corridor=payload.public_corridor)
        self.assertEqual(result['robot_points'], expected.robot_points)
        self.assertEqual(result['goal'], expected.goal)
        self.assertEqual(result['corridor'], expected.corridor)

    def test_dynamic_pose_reference_requires_same_simulation_step(self):
        objects = [dict(name='Moving', dimensions=[.02]*3, center=[1., 0., 0.], moving=True)]
        metrics = dict(capture=dict(body_paths=['/World/Moving']), fixture_setup=dict(actual_actor_poses=[[1., 0., 0., 0., 0., 0., 1.]]))
        payload = SimpleNamespace(frames=[SimpleNamespace(camera_id='a', token=(1, 2.)), SimpleNamespace(camera_id='b', token=(1, 2.1))])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'trace.json').write_text(json.dumps(dict(samples=[dict(simulation_time=2., poses=[[1.2, 0., 0., 0., 0., 0., 1.]]) ])))
            _, frames, reference = physical_reference(folder, metrics, 'moving', dict(objects=objects), payload)
            self.assertEqual(reference['matched_frames'], 1)
            self.assertEqual(frames[0]['objects'][0]['center'], [1.2, 0., 0.])
            self.assertEqual(frames[1]['objects'], [])

    def test_actual_cooked_prism_depth_excludes_enclosing_box_corner(self):
        from scripts.obstacle_representation_v3_cylinder import unit_prism, _planes
        vertices, counts, indices = unit_prism(32)
        vertices = vertices*np.array([2., 2., 2.])
        # Both rays intersect the enclosing square; its corner is outside the
        # actual prism. A box fallback would incorrectly match the second ray.
        depths = convex_depth(np.array([[0., 0., 1.], [0., 0., 1.]]), np.array([0., 0., -3.]), [_planes((vertices, counts, indices))])
        self.assertEqual(depths[0], 2.)
        corner = convex_depth(np.array([[0., 0., 1.]]), np.array([.9, .9, -3.]), [_planes((vertices, counts, indices))])
        self.assertFalse(np.isfinite(corner[0]))

    def test_packet_audit_detects_changed_extraction_clock(self):
        snapshot = self.snapshot()
        supported = snapshot.states == 2
        packet = ObstaclePacket(snapshot.generation, snapshot.revision, snapshot.timestamp, 11., (), {})
        _, _, errors = audit_packet(snapshot, packet, supported, supported, 10., 5.)
        self.assertEqual(errors['wrong_extracted_at'], 1)

    def test_end_to_end_variants_use_same_canonical_cells_and_real_registry(self):
        from scripts.evaluate_obstacle_representation_v3 import evaluate, DEFAULT_VARIANTS, compact_report, write_json
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder/'snapshot.npz'
            np.savez(path, **{name: getattr(snapshot, name) for name in ARRAYS})
            record = dict(case_id='tiny/job1', sequence_id='tiny', scenario='unit_fixture', layout_id='unit',
                snapshot_npz=str(path), snapshot=snapshot_metadata(snapshot, 10., 5.),
                source_artifacts={}, evaluation_objects=[], support_plane_z=0., motion_context=dict(goal=[1., 1., 1.]))
            manifest = folder/'manifest.json'
            manifest.write_text(json.dumps(dict(records=[record])))
            report = evaluate([manifest], DEFAULT_VARIANTS, [64], folder/'output')
            self.assertTrue(report['all_geometry_audits_pass'])
            self.assertTrue(report['cases'][0]['snapshot_unchanged'])
            for value in report['cases'][0]['variants'].values():
                self.assertEqual(value['status'], 'PASS_EXACT_AUDIT')
                self.assertTrue(value['quiet_packet_expiry_removed_all'])
            bundle = json.loads((folder/'output/packet_bundle.json').read_text())
            self.assertEqual(len(bundle['cases']), 1)
            raw = folder/'raw.json'
            write_json(raw, report)
            compact = compact_report(report, raw)
            self.assertEqual(compact['exact_audit_count'], len(DEFAULT_VARIANTS))
            self.assertEqual(compact['baseline_at_proven_lower_bound_snapshots'], 1)
            self.assertFalse(any(compact['audit_error_totals'].values()))
            self.assertNotIn('variants', compact['cases'][0])
            self.assertNotIn('motion_context', json.dumps(compact))


if __name__ == '__main__':
    unittest.main()
