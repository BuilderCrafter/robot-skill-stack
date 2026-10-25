"""Actual sampling provenance, continuous distances and fixed-phase ambiguity."""
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from robot_skill_stack.world.spatial_mapping.occupancy import GridSpec
from scripts.diagnose_obstacle_representation_v3_sampling import (
    BASELINE_COMMIT, PARSER_SOURCES, ROOT, actual_selection, component_counts,
    diagnose_record, effective_mapping_settings, nearest_distances, phase_counts)
from scripts.evaluate_obstacle_representation_v3 import sha


class SamplingQuantizationDiagnosticTests(unittest.TestCase):
    def frame(self):
        return SimpleNamespace(camera_id='cam', depth=np.ones((2, 2)), intrinsics=np.eye(3),
            world_from_camera=np.eye(4), token=(1., 1.), acquired_at=10.)

    def test_independent_continuous_norms_and_duplicate_neighbors(self):
        points = np.array([[0., 0., 0.], [.009, .009, 0.], [.011, 0., 0.], [.011, 0., 0.]])
        delta = abs(points[:, None]-points[None])
        delta[np.arange(4), np.arange(4)] = np.inf
        euclidean = np.sqrt((delta*delta).sum(2)).min(1)
        linf = delta.max(2).min(1)
        result = nearest_distances(points)
        self.assertEqual(result['endpoints'], 4)
        self.assertEqual(result['exact_unique_coordinates'], 3)
        self.assertEqual(result['euclidean_m']['p50'], float(np.quantile(euclidean, .5)))
        self.assertEqual(result['linf_m']['p50'], float(np.quantile(linf, .5)))
        self.assertEqual(result['linf_m']['min'], 0.)
        self.assertEqual(result['fraction_nearest_linf_gt_010_m'], float(np.mean(linf > .01)))

    def test_sparse_continuous_points_stay_disconnected_under_all_fixed_phases(self):
        points = np.array([[.004, .004, .004], [.034, .004, .004]])
        original = points.copy()
        spec = GridSpec((0., 0., 0.), (.1, .1, .1), .01)
        values = phase_counts(points, spec)
        self.assertEqual(values['unique_cells'], [2]*8)
        self.assertEqual(values['face6'], [2]*8)
        self.assertEqual(values['face26'], [2]*8)
        self.assertEqual(nearest_distances(points)['fraction_nearest_linf_gt_020_m'], 1.)
        np.testing.assert_array_equal(points, original)

    def test_close_diagonal_points_show_origin_aliasing_without_new_points(self):
        points = np.array([[.009, .009, .005], [.011, .011, .005]])
        spec = GridSpec((0., 0., 0.), (.1, .1, .1), .01)
        values = phase_counts(points, spec)
        self.assertEqual(values['face6'][0], 2)
        self.assertEqual(values['face26'][0], 1)
        self.assertEqual(min(values['unique_cells']), 1)
        self.assertEqual(max(values['unique_cells']), 2)
        self.assertAlmostEqual(nearest_distances(points)['linf_m']['worst'], .002)

    def test_adjacency_graphs_never_fill_intermediate_cells(self):
        values = component_counts([[0, 0, 0], [1, 1, 0], [3, 1, 0]])
        self.assertEqual(values, dict(unique_cells=3, face6=3, face26=2, radius2_chebyshev=1))
        # Signed phase-lattice indices are valid evaluation coordinates.
        self.assertEqual(component_counts([[-1, 0, 0], [0, 0, 0]])['face6'], 1)

    def test_missing_recorded_masks_is_unmeasured_not_a_recomputed_sampler(self):
        frame = self.frame()
        with self.assertRaises(LookupError):
            actual_selection({}, 0, frame, None, 2., GridSpec((0., 0., 0.), (3., 3., 3.), 1.))

    def test_original_masks_and_depth_gate_are_enforced(self):
        frame = self.frame()
        automatic = np.zeros((2, 2), bool)
        automatic[0, 1] = True
        archive = dict(camera_0_selected_y=np.array([0, 0, 1]), camera_0_selected_x=np.array([0, 1, 0]),
            camera_0_automatic_mask=automatic)
        spec = GridSpec((0., 0., 0.), (3., 3., 3.), 1.)
        result = actual_selection(archive, 0, frame, None, 2., spec)
        self.assertEqual(result['recorded_selected_pixels'], 3)
        self.assertEqual(result['automatic_masked_selected'], 1)
        self.assertEqual(result['selected_in_grid_endpoint_rays'], 2)
        np.testing.assert_allclose(result['points'], [[.5, .5, 1.], [.5, 1.5, 1.]])
        with self.assertRaisesRegex(ValueError, 'pre-sampling'):
            actual_selection(archive, 0, frame, None, .5, spec)
        with self.assertRaisesRegex(ValueError, 'pre-sampling'):
            actual_selection(archive, 0, frame, np.ones((2, 2), bool), 2., spec)

    def captured_record(self, folder, *, wrong_integration=False):
        folder = Path(folder)
        audit = folder/'mapping-audit'
        audit.mkdir()
        (folder/'runtime_profile.toml').write_text('[spatial_mapping]\nmax_depth=0.1\nmax_rays=1\nstride=1\n')
        frame = self.frame()
        frame.depth = np.full((2, 2), .025)
        payload = SimpleNamespace(frames=(frame,), exclusion_masks=None)
        source = audit/'input.pkl'
        source.write_bytes(pickle.dumps(payload))
        states = np.zeros((10, 10, 10), np.uint8)
        states[1, 1, 2] = states[8, 8, 8] = 2  # one current and one historical endpoint
        arrays = audit/'arrays.npz'
        np.savez(arrays, states=states, probabilities=np.where(states == 2, .99, .5), observed=states > 0,
            last_observed=np.where(states == 2, 9., np.nan), expired=np.zeros_like(states, bool),
            camera_0_selected_y=np.array([0]), camera_0_selected_x=np.array([0]),
            camera_0_automatic_mask=np.zeros((2, 2), bool),
            integrate_occupied_indices=np.array([[8, 8, 8] if wrong_integration else [1, 1, 2]]))
        metadata = audit/'metadata.json'
        metadata.write_text(json.dumps(dict(camera_ids=['cam'])))
        return dict(case_id='unit/job1', sequence_id='unit', layout_id='unit', scenario='unit',
            snapshot_npz=str(arrays), snapshot=dict(bounds_min=[0.]*3, bounds_max=[.1]*3, voxel_size=.01,
                timestamp=9., revision=1, generation=0, health=dict(age_seconds=1.), evaluated_at=10., evidence_ttl=5.),
            source_artifacts=dict(input_pickle=str(source), input_sha256=sha(source), arrays_sha256=sha(arrays), metadata_json=str(metadata)),
            evaluation_objects=[], physical_attribution_status='NO_EXTERNAL_FIXTURE_OBJECTS')

    def test_current_ray_cloud_is_separate_from_cumulative_canonical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.captured_record(tmp)
            result = diagnose_record(record, phases=True)
            self.assertTrue(result['actual_integration_cell_set_matches'])
            self.assertEqual(result['current_input_cells']['unique_cells'], 1)
            self.assertEqual(result['canonical_eligible']['unique_cells'], 2)
            self.assertEqual(result['canonical_supported']['unique_cells'], 2)
            self.assertEqual(result['current_input_continuous_nn']['status'], 'UNMEASURED_FEWER_THAN_TWO_ENDPOINTS')

    def test_actual_integration_oracle_rejects_wrong_pixel_frame_mask_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.captured_record(tmp, wrong_integration=True)
            with self.assertRaisesRegex(ValueError, 'actual occupied integration'):
                diagnose_record(record)

    def omitted_default_proof(self, folder):
        folder = Path(folder)
        profile = folder/'runtime_profile.toml'
        profile.write_text('''[robot]
backend="franka"
prim_path="/Franka"
[world]
provider="ground_truth"
[scene]
support_plane_z=0.0
workspace_min=[0.0,0.0,0.0]
workspace_max=[1.0,1.0,1.0]
[ground_truth]
objects_root="/World/Objects"
[spatial_mapping]
max_rays=123
stride=4
''')
        result = folder/'result.json'
        result.write_text('{}')
        hashes = {name:sha(ROOT/name) for name in PARSER_SOURCES}
        (folder/'protected-baseline.json').write_text(json.dumps(dict(commit=BASELINE_COMMIT, sha256=hashes)))
        driver = dict(status='PASS', production_unchanged=True,
            production_hashes_before=hashes, production_hashes_after=hashes,
            jobs=[dict(status='PASS', result=str(result))])
        (folder/'pilot-driver.json').write_text(json.dumps(driver))
        return profile,result,driver

    def test_omitted_default_uses_real_parser_only_after_both_source_hashes_and_case_match(self):
        from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
        with tempfile.TemporaryDirectory() as tmp:
            profile,result,_ = self.omitted_default_proof(tmp)
            values,proof = effective_mapping_settings(profile, result, proof_directory=tmp)
            self.assertEqual(values['max_depth'], SpatialMappingConfig().max_depth)
            self.assertEqual(values['max_rays'], 123)
            self.assertEqual(values['stride'], 4)
            self.assertEqual(proof['inherited_fields'], ['max_depth'])
            self.assertEqual(set(proof['parser_and_leaf_sha256']), set(PARSER_SOURCES))
            self.assertEqual(proof['case_result'], str(result))

    def test_omitted_default_is_unmeasured_if_capture_leaf_hash_or_case_binding_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile,result,driver = self.omitted_default_proof(tmp)
            # The top-level parser is unchanged; the separate defaults leaf
            # must independently match the source that actually ran natively.
            driver['production_hashes_before'][PARSER_SOURCES[1]] = 'unproven-leaf'
            (Path(tmp)/'pilot-driver.json').write_text(json.dumps(driver))
            with self.assertRaisesRegex(LookupError, 'actual capture parser or leaf'):
                effective_mapping_settings(profile, result, proof_directory=tmp)
            _,_,driver = self.omitted_default_proof(tmp)
            driver['jobs'][0]['result'] = str(Path(tmp)/'different-case-result.json')
            (Path(tmp)/'pilot-driver.json').write_text(json.dumps(driver))
            with self.assertRaisesRegex(LookupError, 'uniquely bound'):
                effective_mapping_settings(profile, result, proof_directory=tmp)

    def test_omitted_default_does_not_guess_without_protected_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile,result,_ = self.omitted_default_proof(tmp)
            (Path(tmp)/'protected-baseline.json').unlink()
            with self.assertRaisesRegex(LookupError, 'protected baseline unavailable'):
                effective_mapping_settings(profile, result, proof_directory=tmp)


if __name__ == '__main__':
    unittest.main()
