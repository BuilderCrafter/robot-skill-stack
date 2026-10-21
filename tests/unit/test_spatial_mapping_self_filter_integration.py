"""Opt-in robot filtering, complete-frame transactions and resource ownership."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_skill_stack.integrations.isaac.config import load_scene_config
from robot_skill_stack.integrations.isaac.spatial_mapping.config import SpatialMappingConfig
from robot_skill_stack.integrations.isaac.spatial_mapping.factory import (
    MappingSession, create_spatial_mapping,
)
from robot_skill_stack.world.spatial_mapping import GridSpec, OccupancyMap, OccupancyState as State
from tests.unit.test_spatial_mapping_rgbd import ray


FACTORY = 'robot_skill_stack.integrations.isaac.spatial_mapping.factory'
ADAPTER = 'robot_skill_stack.integrations.isaac.spatial_mapping.robot_filter'
ROBOT_PATH = '/Lab/Station/Panda'


class SelfFilterIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.clock = lambda: self.now
        self.settings = SpatialMappingConfig(enabled=True, source='rgbd',
            bounds_min=(0., 0., 0.), bounds_max=(1., 1., 1.), voxel_size=.1,
            stride=1, automatic_robot_filter=True)
        self.robot_filter = Mock()
        self.robot_filter.masks.side_effect = lambda frames, **kwargs: {
            frame.camera_id: np.zeros(frame.depth.shape, bool) for frame in frames}

    def session(self, **kwargs):
        return create_spatial_mapping(kwargs.pop('settings', self.settings), clock=self.clock,
            robot_prim_path=ROBOT_PATH, robot_filter=self.robot_filter, **kwargs)

    def test_defaults_validation_and_toml_preserve_configured_robot_path(self):
        defaults = SpatialMappingConfig()
        self.assertFalse(defaults.automatic_robot_filter)
        self.assertEqual(defaults.robot_depth_tolerance, .005)
        self.assertEqual(defaults.robot_pose_tolerance, .01)
        for kwargs in (dict(automatic_robot_filter=1), dict(automatic_robot_filter='true'),
                dict(robot_depth_tolerance=0), dict(robot_depth_tolerance=True),
                dict(robot_depth_tolerance=np.inf), dict(robot_pose_tolerance=-1),
                dict(robot_pose_tolerance=np.nan), dict(robot_pose_tolerance=None)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpatialMappingConfig(**kwargs)
        root = Path(__file__).resolve().parents[2]
        text = (root/'config/scenes/playground.toml').read_text().replace(
            'prim_path = "/Franka"', f'prim_path = "{ROBOT_PATH}"')
        text += '\n[spatial_mapping]\nenabled = true\nsource = "rgbd"\n' \
                'automatic_robot_filter = true\nrobot_depth_tolerance = 0.003\nrobot_pose_tolerance = 0.008\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'setup.toml'
            path.write_text(text)
            config = load_scene_config(path)
        self.assertEqual(config.robot.prim_path, ROBOT_PATH)
        self.assertTrue(config.spatial_mapping.automatic_robot_filter)
        self.assertEqual(config.spatial_mapping.robot_depth_tolerance, .003)
        self.assertEqual(config.spatial_mapping.robot_pose_tolerance, .008)

    def test_disabled_and_unfiltered_paths_do_not_take_filter_or_camera_ownership(self):
        frames = Mock()
        with patch(FACTORY+'.OccupancyMap') as allocate:
            self.assertIsNone(create_spatial_mapping(replace(self.settings, enabled=False),
                robot_filter=self.robot_filter, frame_source=frames, own_frame_source=True))
        allocate.assert_not_called()
        session = self.session(settings=replace(self.settings, automatic_robot_filter=False))
        self.assertIsNone(session.robot_filter)
        self.assertTrue(session.update((ray(),), now=100.))
        session.close()
        self.robot_filter.masks.assert_not_called()
        self.robot_filter.close.assert_not_called()
        frames.capture.assert_not_called()
        frames.close.assert_not_called()

    def test_factory_requires_explicit_robot_path_even_with_injected_filter(self):
        for path in (None, '', 'relative', '/', '/Bad Name'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'robot_prim_path'):
                create_spatial_mapping(self.settings, robot_prim_path=path,
                    robot_filter=self.robot_filter, clock=self.clock)
        self.robot_filter.close.assert_not_called()

    def test_factory_constructs_adapter_with_exact_stage_path_and_tolerances(self):
        module = ModuleType(ADAPTER)
        module.IsaacRobotSelfFilter = Mock(return_value=self.robot_filter)
        stage = object()
        settings = replace(self.settings, robot_depth_tolerance=.003, robot_pose_tolerance=.008, stride=2)
        with patch.dict(sys.modules, {ADAPTER: module}):
            session = create_spatial_mapping(settings, stage=stage, robot_prim_path=ROBOT_PATH,
                                             clock=self.clock)
        module.IsaacRobotSelfFilter.assert_called_once_with(stage, ROBOT_PATH, stride=2,
            depth_tolerance=.003, pose_tolerance=.008)
        self.assertIs(session.robot_filter, self.robot_filter)
        session.close()
        self.robot_filter.close.assert_called_once()

    def test_ground_truth_does_not_construct_or_apply_rgbd_filter(self):
        session = create_spatial_mapping(replace(self.settings, source='ground_truth'),
            robot_filter=self.robot_filter, clock=self.clock)
        self.assertIsNone(session.robot_filter)
        session.close()
        self.robot_filter.masks.assert_not_called()
        self.robot_filter.close.assert_not_called()

    def test_manual_and_automatic_masks_union_without_mutation_or_carving_masked_rays(self):
        frame = replace(ray(origin=(.15, .15, .05), endpoint=(.15, .15, .65)),
            depth=np.full((1, 3), .6),
            intrinsics=np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]]))
        automatic, manual = np.array([[True, False, False]]), np.array([[False, True, False]])
        automatic.setflags(write=False)
        manual.setflags(write=False)
        self.robot_filter.masks.side_effect = None
        self.robot_filter.masks.return_value = {'A': automatic}
        session = self.session()
        frames = (frame,)
        self.assertTrue(session.update(frames, exclusion_masks={'A': manual}, now=100.))
        self.assertIs(self.robot_filter.masks.call_args.args[0][0], frame)
        self.assertEqual(session.source.diagnostics['rays'], 1)
        for position in ((.15, .15, .45), (.15, .15, .65), (.15, .15, .85), (.45, .15, .65)):
            self.assertFalse(session.map.query(position).observed, position)
        self.assertEqual(session.map.query((.75, .15, .65)).state, State.OCCUPIED)
        np.testing.assert_array_equal(automatic, [[True, False, False]])
        np.testing.assert_array_equal(manual, [[False, True, False]])
        self.assertFalse(automatic.flags.writeable)
        self.assertFalse(manual.flags.writeable)
        session.close()

    def test_all_automatic_camera_masks_required_for_three_and_five_views(self):
        for count in (3, 5):
            with self.subTest(count=count):
                frames = tuple(ray(camera=f'C{index}') for index in range(count))
                session = self.session()
                self.assertTrue(session.update(frames, now=100.))
                self.assertEqual(len(session.source.diagnostics['cameras']), count)
                self.assertEqual(session.source.diagnostics['rays'], count)
                session.close()

    def test_filter_queries_only_fair_selected_rays_and_does_not_backfill_a_masked_view(self):
        frames = tuple(replace(ray(camera=key, origin=(.05, y, .15), endpoint=(.65, y, .15)),
            depth=np.full((1, 20), .6), intrinsics=np.array([
                [1000., 0., .5], [0., 1000., .5], [0., 0., 1.]]))
            for key, y in (('A', .15), ('B', .65)))
        manual = np.zeros((1, 20), bool)
        manual[0, :2] = True
        queried = {}
        def masks(current, *, pixels_by_camera):
            queried.update({key: (y.copy(), x.copy()) for key, (y, x) in pixels_by_camera.items()})
            return {'A': np.ones((1, 20), bool), 'B': np.zeros((1, 20), bool)}
        self.robot_filter.masks.side_effect = masks
        session = self.session(settings=replace(self.settings, max_rays=4))
        self.assertTrue(session.update(frames, exclusion_masks={'A': manual}, now=100.))
        self.assertEqual(set(queried), {'A', 'B'})
        np.testing.assert_array_equal(queried['A'][1], [6, 15])
        np.testing.assert_array_equal(queried['B'][1], [5, 15])
        self.assertEqual(sum(len(value[0]) for value in queried.values()), 4)
        detail = session.source.diagnostics
        self.assertEqual((detail['allocated_rays'], detail['retained_rays'],
                          detail['automatic_masked_selected'], detail['rays']), (4, 2, 2, 2))
        self.assertEqual(detail['cameras']['A']['retained_rays'], 0)
        self.assertEqual(detail['cameras']['B']['allocated_rays'], 2)
        self.assertEqual(detail['cameras']['B']['retained_rays'], 2)
        self.assertFalse(session.map.query((.65, .15, .15)).observed)
        self.assertFalse(session.map.query((.85, .15, .15)).observed)
        self.assertEqual(session.map.query((.65, .65, .15)).state, State.OCCUPIED)
        session.close()

    def test_fully_automasked_selection_produces_no_evidence_or_replacement_queries(self):
        current = replace(ray(), depth=np.full((1, 20), .6))
        self.robot_filter.masks.side_effect = None
        self.robot_filter.masks.return_value = {'A': np.ones(current.depth.shape, bool)}
        session = self.session(settings=replace(self.settings, max_rays=2))
        self.assertFalse(session.update((current,), now=100.))
        selected = self.robot_filter.masks.call_args.kwargs['pixels_by_camera']['A']
        self.assertEqual(len(selected[0]), 2)
        self.assertEqual(session.source.diagnostics['allocated_rays'], 2)
        self.assertEqual(session.source.diagnostics['retained_rays'], 0)
        self.assertEqual(session.source.diagnostics['automatic_masked_selected'], 2)
        self.assertIsNone(session.map.timestamp)
        self.assertEqual(session.map.counts()['unknown'], 1000)
        self.assertFalse(session.update((current,), now=100.))
        self.robot_filter.masks.assert_called_once()
        session.close()

    def test_zero_allocation_skips_model_work_and_empty_camera_selection_is_explicit(self):
        first, second = ray(), ray(camera='B')
        session = self.session()
        self.assertFalse(session.update((first,), exclusion_masks={'A': np.ones((1, 1), bool)}, now=100.))
        self.robot_filter.masks.assert_not_called()
        self.assertEqual(session.source.diagnostics['allocated_rays'], 0)
        session.close()
        session = self.session()
        self.assertTrue(session.update((first, second),
            exclusion_masks={'A': np.ones((1, 1), bool)}, now=100.))
        selected = self.robot_filter.masks.call_args.kwargs['pixels_by_camera']
        self.assertEqual(set(selected), {'A', 'B'})
        self.assertEqual((selected['A'][0].size, selected['A'][1].size), (0, 0))
        self.assertEqual(selected['A'][0].dtype.kind, 'i')
        self.assertEqual(selected['B'][0].size, 1)
        session.close()

    def test_bad_manual_masks_are_rejected_before_filter_and_consume_attempted_tokens(self):
        for masks in ({'unknown': np.zeros((1, 1), bool)}, {'A': np.zeros((2, 1), bool)},
                      {'A': np.zeros((1, 1), int)}, {'A': None}):
            with self.subTest(masks=masks):
                session = self.session()
                self.robot_filter.masks.reset_mock()
                frames = (ray(),)
                self.assertFalse(session.update(frames, exclusion_masks=masks, now=100.))
                self.assertEqual(session.map.health['status'], 'error')
                self.assertIsNone(session.map.timestamp)
                self.robot_filter.masks.assert_not_called()
                self.assertFalse(session.update(frames, now=100.))
                self.robot_filter.masks.assert_not_called()
                session.close()

    def test_bad_or_incomplete_automatic_masks_cannot_fall_back_to_unfiltered_mapping(self):
        for masks in (None, {}, {'unexpected': np.zeros((1, 1), bool)},
                      {'A': np.zeros((2, 1), bool)}, {'A': np.zeros((1, 1), int)}):
            with self.subTest(masks=masks):
                self.robot_filter.masks.side_effect = None
                self.robot_filter.masks.return_value = masks
                self.robot_filter.masks.reset_mock()
                session = self.session()
                frames = (ray(),)
                self.assertFalse(session.update(frames, now=100.))
                self.assertEqual(session.map.health['status'], 'error')
                self.assertEqual(session.map.counts()['unknown'], 1000)
                self.robot_filter.masks.return_value = {'A': np.zeros((1, 1), bool)}
                self.assertFalse(session.update(frames, now=100.))
                self.robot_filter.masks.assert_called_once()
                session.close()

    def test_geometry_failure_keeps_previous_evidence_and_requires_new_frames_for_recovery(self):
        session = self.session()
        self.assertTrue(session.update((ray(),), now=100.))
        previous = session.map.snapshot()
        self.robot_filter.masks.side_effect = RuntimeError('No pose at render timestamp')
        self.now = 100.1
        frames = (ray(number=2, acquired=self.now),)
        self.assertFalse(session.update(frames, now=self.now))
        self.assertEqual(session.map.health['status'], 'error')
        self.assertIn('No pose', session.map.health['last_error'])
        self.assertEqual((session.map.timestamp, session.map.revision), (previous.timestamp, previous.revision))
        np.testing.assert_array_equal(session.map.snapshot().states, previous.states)
        self.robot_filter.masks.side_effect = lambda current, **kwargs: {'A': np.zeros((1, 1), bool)}
        calls = self.robot_filter.masks.call_count
        self.assertFalse(session.update(frames, now=self.now))
        self.assertEqual(self.robot_filter.masks.call_count, calls)
        self.now = 100.2
        self.assertTrue(session.update((ray(number=3, acquired=self.now),), now=self.now))
        self.assertTrue(session.map.health['fresh'])
        session.close()

    def test_all_frames_validated_before_filtering_and_original_render_tokens_preserved(self):
        session = self.session()
        invalid = replace(ray(camera='B'), depth=np.array([[np.nan]]))
        self.assertFalse(session.update((ray(), invalid), now=100.))
        self.robot_filter.masks.assert_not_called()
        session.close()
        session = self.session()
        original = replace(ray(), token=(37., 1.234))
        self.assertTrue(session.update((original,), now=100.))
        delivered = self.robot_filter.masks.call_args.args[0][0]
        self.assertIs(delivered, original)
        self.assertEqual(delivered.token, (37., 1.234))
        self.assertEqual(delivered.acquired_at, 100.)
        session.close()

    def test_default_completion_guard_includes_filter_cost_and_explicit_now_stays_deterministic(self):
        def slow_filter(frames, **kwargs):
            self.now += 1.1
            return {'A': np.zeros((1, 1), bool)}
        self.robot_filter.masks.side_effect = slow_filter
        session = self.session()
        self.assertFalse(session.update((ray(),)))
        self.assertIsNone(session.map.timestamp)
        self.assertIn('became stale', session.map.health['last_error'])
        session.close()
        self.now = 100.
        session = self.session()
        self.assertTrue(session.update((ray(),), now=100.))
        self.assertEqual(session.map.timestamp, 100.)
        session.close()

    def test_rate_limit_and_missing_capture_do_not_run_the_filter(self):
        capture = Mock()
        capture.capture.return_value = None
        session = self.session(settings=replace(self.settings, update_hz=2.), frame_source=capture)
        self.assertFalse(session.update())
        self.now = 100.1
        self.assertFalse(session.update())
        capture.capture.assert_called_once()
        self.robot_filter.masks.assert_not_called()
        session.close()
        capture.close.assert_not_called()

    def test_close_attempts_every_owned_resource_once_even_when_multiple_closes_fail(self):
        occupancy = OccupancyMap(GridSpec((0, 0, 0), (1, 1, 1), .1), clock=self.clock)
        source, capture = Mock(), Mock()
        source.close.side_effect = RuntimeError('source cleanup')
        self.robot_filter.close.side_effect = RuntimeError('filter cleanup')
        capture.close.side_effect = RuntimeError('capture cleanup')
        session = MappingSession(occupancy, source, robot_filter=self.robot_filter,
            frame_source=capture, own_frame_source=True, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'source cleanup') as caught:
            session.close()
        session.close()
        for resource in (source, self.robot_filter, capture):
            resource.close.assert_called_once()
        self.assertEqual(len(caught.exception.__notes__), 2)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            session.update((ray(),))

    def test_constructor_failure_cleans_filter_source_and_owned_capture_preserving_primary_error(self):
        source, capture = Mock(), Mock()
        self.robot_filter.close.side_effect = RuntimeError('filter cleanup')
        capture.close.side_effect = RuntimeError('capture cleanup')
        with patch('robot_skill_stack.world.spatial_mapping.rgbd.RgbdMapper', return_value=source), \
                patch(FACTORY+'.MappingSession', side_effect=ValueError('session initialization')):
            with self.assertRaisesRegex(ValueError, 'session initialization') as caught:
                self.session(frame_source=capture, own_frame_source=True)
        for resource in (source, self.robot_filter, capture):
            resource.close.assert_called_once()
        self.assertEqual(len(caught.exception.__notes__), 2)

    def test_adapter_initialization_failure_still_closes_source_and_owned_capture(self):
        source, capture = Mock(), Mock()
        capture.close.side_effect = RuntimeError('capture cleanup')
        module = ModuleType(ADAPTER)
        module.IsaacRobotSelfFilter = Mock(side_effect=RuntimeError('Robot geometry unavailable'))
        with patch.dict(sys.modules, {ADAPTER: module}), \
                patch('robot_skill_stack.world.spatial_mapping.rgbd.RgbdMapper', return_value=source):
            with self.assertRaisesRegex(RuntimeError, 'Robot geometry unavailable') as caught:
                create_spatial_mapping(self.settings, robot_prim_path=ROBOT_PATH,
                    frame_source=capture, own_frame_source=True, clock=self.clock)
        source.close.assert_called_once()
        capture.close.assert_called_once()
        self.assertIn('capture cleanup', caught.exception.__notes__[0])

    def test_owned_resource_aliases_are_not_closed_twice(self):
        occupancy = OccupancyMap(GridSpec((0, 0, 0), (1, 1, 1), .1), clock=self.clock)
        resource = Mock()
        session = MappingSession(occupancy, resource, frame_source=resource,
            own_frame_source=True, robot_filter=resource, clock=self.clock)
        session.close()
        session.close()
        resource.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
