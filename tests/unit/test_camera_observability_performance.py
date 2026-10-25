"""Research performance modes must preserve isolated camera workloads."""
import hashlib
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.camera_observability_performance import camera_selections, write_research_profile
from scripts.camera_observability_planner import THIRD, validate_sampling_policy
from scripts.camera_workload_config import ORIGINAL_CAMERAS, PROFILE


def layout(three=False):
    paths = [*ORIGINAL_CAMERAS, THIRD] if three else list(ORIGINAL_CAMERAS)
    return dict(layout_id='test', cameras=[dict(camera_id=path,
        resolution=[320, 240] if path == THIRD else [640, 480]) for path in paths])


class CameraObservabilityPerformanceTests(unittest.TestCase):
    def test_explicit_workloads_select_the_intended_semantic_and_mapping_views(self):
        two, three = layout(), layout(True)
        self.assertEqual(camera_selections(two, 'baseline'), (ORIGINAL_CAMERAS, ORIGINAL_CAMERAS))
        self.assertEqual(camera_selections(three, 'mapping'), (ORIGINAL_CAMERAS, (*ORIGINAL_CAMERAS, THIRD)))
        self.assertEqual(camera_selections(three, 'full'), ((*ORIGINAL_CAMERAS, THIRD), (*ORIGINAL_CAMERAS, THIRD)))
        for value, mode in ((two, 'mapping'), (two, 'full'), (three, 'baseline')):
            with self.assertRaises(ValueError):
                camera_selections(value, mode)

    def test_copied_profiles_hold_direct_motion_rates_bounds_and_original_resolution(self):
        before = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
        for mode, three in [('baseline', False), ('mapping', True), ('full', True)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                value = tomllib.loads(write_research_profile(temporary, layout(three), mode).read_text())
                self.assertEqual(value['motion']['strategy'], 'direct')
                self.assertEqual(value['world']['update_hz'], 5.)
                self.assertEqual(value['spatial_mapping']['update_hz'], 2.)
                self.assertEqual(value['spatial_mapping']['max_rays'], 1024)
                self.assertEqual(value['spatial_mapping']['bounds_min'], [-.3, -.6, -.02])
                self.assertEqual(value['spatial_mapping']['stale_after'], 1.)
                self.assertEqual(value['multi_view']['resolution'], [640, 480])
                expected = camera_selections(layout(three), mode)
                self.assertEqual(value['multi_view']['camera_prim_paths'], list(expected[0]))
                self.assertEqual(value['spatial_mapping']['camera_prim_paths'], list(expected[1]))
        self.assertEqual(hashlib.sha256(PROFILE.read_bytes()).hexdigest(), before)

    def test_higher_budgets_are_explicit_and_no_other_mapping_parameters_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = tomllib.loads(write_research_profile(temporary, layout(), 'baseline').read_text())
            for budget in (2048, 4096):
                modified = tomllib.loads(write_research_profile(temporary, layout(), 'baseline', max_rays=budget).read_text())
                self.assertEqual(modified['spatial_mapping'].pop('max_rays'), budget)
                expected = dict(original['spatial_mapping'])
                expected.pop('max_rays')
                self.assertEqual(modified['spatial_mapping'], expected)
            with self.assertRaises(ValueError):
                write_research_profile(temporary, layout(), 'baseline', max_rays=8192)

    def test_strategy_and_budget_experiments_cannot_be_silently_combined(self):
        for budget in (1024, 2048, 4096):
            self.assertEqual(validate_sampling_policy('fixed', budget), 'fixed')
        self.assertEqual(validate_sampling_policy('depth_edge', 1024), 'depth_edge')
        for budget in (2048, 4096):
            with self.assertRaises(ValueError):
                validate_sampling_policy('depth_edge', budget)


if __name__ == '__main__':
    unittest.main()
