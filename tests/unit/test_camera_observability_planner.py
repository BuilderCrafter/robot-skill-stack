"""Research camera routing must not change production fixtures or defaults."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tomllib
import unittest

from scripts.camera_observability_planner import (
    THIRD, copied_profile, enter_sampling_research, select_layout, validate_sampling_policy)
from scripts.camera_workload_config import ORIGINAL_CAMERAS, PROFILE


def layout(three=False):
    paths = (*ORIGINAL_CAMERAS, THIRD) if three else ORIGINAL_CAMERAS
    return dict(layout_id='test', family='H' if three else 'A', cameras=[dict(
        camera_id=path, world_from_camera_ros=[[1., 0., 0., i], [0., 1., 0., 0.],
            [0., 0., 1., 1.], [0., 0., 0., 1.]], resolution=[320, 240] if i == 2 else [640, 480])
        for i, path in enumerate(paths)])


class CameraObservabilityPlannerTests(unittest.TestCase):
    def test_layout_selection_rejects_ambiguous_or_invalid_calibration(self):
        chosen = layout()
        self.assertIs(select_layout({'layouts': [chosen]}, 'test'), chosen)
        with self.assertRaises(ValueError):
            select_layout({'layouts': [chosen, chosen]}, 'test')
        damaged = copy.deepcopy(chosen)
        damaged['cameras'][0]['world_from_camera_ros'][0][0] = 2.
        with self.assertRaises(ValueError):
            select_layout({'layouts': [damaged]}, 'test')
        damaged = copy.deepcopy(chosen)
        damaged['cameras'][0]['world_from_camera_ros'][0][0] = -1.
        with self.assertRaises(ValueError):
            select_layout({'layouts': [damaged]}, 'test')

    def test_unvalidated_wrist_attachment_is_not_silently_treated_as_fixed(self):
        chosen = layout(True)
        chosen['family'] = 'J'
        with self.assertRaisesRegex(ValueError, 'geometric-only'):
            select_layout({'layouts': [chosen]}, 'test')

    def test_mapping_third_camera_does_not_change_semantic_pair_or_saved_profile(self):
        text = PROFILE.read_text()+'\n[motion]\nstrategy="reactive"\n\n[spatial_mapping]\nenabled=true\nsource="rgbd"\n'
        before = PROFILE.read_bytes()
        result = tomllib.loads(copied_profile(text, layout(True), max_rays=2048))
        self.assertEqual(result['motion']['strategy'], 'reactive')
        self.assertEqual(result['multi_view']['camera_prim_paths'], list(ORIGINAL_CAMERAS))
        self.assertEqual(result['multi_view']['resolution'], [640, 480])
        self.assertEqual(result['spatial_mapping']['camera_prim_paths'], [*ORIGINAL_CAMERAS, THIRD])
        self.assertEqual(result['spatial_mapping']['max_rays'], 2048)
        self.assertEqual(PROFILE.read_bytes(), before)

    def test_unplanned_ray_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            copied_profile(PROFILE.read_text(), layout(), max_rays=8192)

    def test_nonfixed_sampling_is_explicit_and_cannot_replace_an_audit_processor(self):
        for policy in ('depth_edge', 'phase_cycle', 'cycle_stride'):
            self.assertEqual(validate_sampling_policy(policy, 1024), policy)
            with self.assertRaises(ValueError):
                validate_sampling_policy(policy, 2048)
            with self.assertRaisesRegex(ValueError, 'audit'):
                validate_sampling_policy(policy, 1024, audit_dir='diagnostic-captures')
        self.assertEqual(validate_sampling_policy('fixed', 4096, audit_dir='captures'), 'fixed')
        with self.assertRaises(ValueError):
            validate_sampling_policy('unknown', 1024)

    def test_research_processor_patch_preserves_clock_obstacles_and_production_factory(self):
        from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
        from robot_skill_stack.world.spatial_mapping import runtime
        original = runtime.MappingProcessor
        settings = SpatialMappingConfig(source='rgbd')
        clock = lambda: 123.4
        obstacles = object()
        with ExitStack() as stack:
            enter_sampling_research(stack, 'fixed', 1024)
            self.assertIs(runtime.MappingProcessor, original)
        for policy in ('depth_edge', 'phase_cycle', 'cycle_stride'):
            with self.subTest(policy=policy), ExitStack() as stack:
                enter_sampling_research(stack, policy, 1024)
                processor = runtime.MappingProcessor(settings, clock=clock, obstacle_settings=obstacles)
                self.assertIs(processor.settings, settings)
                self.assertIs(processor.clock, clock)
                self.assertIs(processor._obstacle_settings, obstacles)
                self.assertEqual(processor.allow_cycle_offsets, policy == 'cycle_stride')
            self.assertIs(runtime.MappingProcessor, original)


if __name__ == '__main__':
    unittest.main()
