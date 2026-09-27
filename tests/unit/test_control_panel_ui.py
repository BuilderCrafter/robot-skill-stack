from __future__ import annotations

import asyncio
import ast
from dataclasses import replace
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.presentation.world_model import WorldModelViewModel
from robot_skill_stack.runtime.skill import FailureCode, SkillResult, SkillStatus
from robot_skill_stack.world.model.entities import WorldObject
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.fake_omni_ui import EXT, ROOT, extension_environment


def object_(object_id='object_1', shape=PrimitiveShape.CUBE, visible=True):
    geometry = PrimitiveGeometry(shape=shape, confidence=.9, box_size=[.05]*3, yaw=.4) if shape == PrimitiveShape.CUBE else (
        PrimitiveGeometry(shape=shape, confidence=.9, radius=.025) if shape == PrimitiveShape.SPHERE else
        PrimitiveGeometry(shape=shape, confidence=.9, axis=[0, 0, 1], radius=.02, length=.1))
    return WorldObject(object_id, shape.value, Pose([.45, 0, .025]), [.05]*3, geometry=geometry,
                       visible=visible, source='rgbd_geometry_perception', confidence=.95)


def bundle_(provider='perception'):
    wm = WorldModel()
    wm.register(object_())
    backend = Mock()
    backend.robot.handles_initialized = True
    backend.get_end_effector_pose.return_value = Pose([.431, .128, .392], [0, 0, 1, 0])
    runtime = Mock()
    runtime.execute_async = AsyncMock(return_value=SkillResult(SkillStatus.SUCCESS, 'Done'))
    updater = Mock()
    updater.cleanup.return_value = ()
    updater.clear_lost.return_value = ()
    return SimpleNamespace(world_model=wm, backend=backend, runtime=runtime, world=Mock(),
                           world_updater=updater, state_provider=Mock(),
                           config=SimpleNamespace(world=SimpleNamespace(provider=provider)))


class ControlPanelTests(unittest.TestCase):
    def setUp(self):
        self.env = extension_environment()
        self.ui, self.c, self.module = self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)
        self.panel = self.c.panel

    def bind(self, bundle=None):
        self.c.bundle = bundle or bundle_()
        self.c.view_model = WorldModelViewModel(self.c.bundle.world_model)
        self.c._refresh(force=True)
        return self.c.bundle

    def hold(self, bundle=None):
        bundle = self.bind(bundle)
        bundle.world_model.set_held('object_1', object_offset_in_ee=[0, 0, .03])
        bundle.world_model.expect_object_at('object_1', [.45, 0, .2], reason='pick')
        self.c._refresh()
        return bundle

    def test_native_two_columns_and_independent_scrolling(self):
        self.assertEqual(self.c.window.args[0], 'Robot Skill Stack')
        self.assertEqual((self.c.window.width, self.c.window.height), (1180, 920))
        left = self.ui.find(name='controls_sidebar')[0]
        right = self.ui.find(name='world_model_panel')[0]
        self.assertIs(left.parent, right.parent)
        self.assertEqual((left.width, right.width), (.42, .58))
        self.assertIsNot(left, self.panel.object_scroll)
        self.assertEqual(self.panel.object_scroll.kind, 'ScrollingFrame')

    def test_expected_sections_and_live_controls_present(self):
        headers = {w.text for w in self.ui.find('Label', name='section')}
        self.assertEqual(headers, {'SCENE / RUNTIME', 'SELECTED OBJECT', 'TARGET XYZ', 'SKILLS',
                                  'TASKS', 'DEBUG / MAINTENANCE', 'STATUS'})
        self.assertEqual(set(self.panel.buttons), {'profile', 'initialize', 'move_to', 'pick', 'place',
                         'move', 'home', 'task', 'recovery', 'clear_lost', 'clear_held', 'capture'})
        for widget in self.panel.buttons.values():
            self.assertTrue(callable(widget.clicked_fn))

    def test_assets_bundled_and_no_font_or_network_dependency(self):
        images = [*self.ui.find('Image'), *self.ui.find('Button')]
        for widget in images:
            url = widget.args[0] if widget.kind == 'Image' else getattr(widget, 'image_url', None)
            if not url:
                continue
            data = Path(url).read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertTrue(str(Path(url)).startswith(str(EXT)))
        for name in ('cube', 'sphere', 'cylinder', 'unknown'):
            data = (EXT / 'icons' / f'{name}.png').read_bytes()
            self.assertEqual(struct.unpack('>II', data[16:24]), (104, 104))
        self.assertFalse(list((EXT / 'icons').glob('*.ttf')))

    def test_initial_state_does_not_allow_motion_or_maintenance(self):
        for key in ('pick', 'place', 'move', 'home', 'clear_held', 'clear_lost', 'capture'):
            self.assertFalse(self.panel.buttons[key].enabled, key)
        self.assertTrue(self.panel.buttons['initialize'].enabled)
        self.assertEqual(self.panel.count.text, '0 objects')
        self.assertIn('initialize', self.panel.status.text)

    def test_ready_state_uses_real_objects_and_pose(self):
        self.bind()
        self.assertTrue(self.panel.buttons['pick'].enabled)
        self.assertFalse(self.panel.buttons['place'].enabled)
        self.assertEqual(self.panel.count.text, '1 object')
        self.assertEqual(self.panel.runtime_info.text, 'Perception')
        self.assertIn('0.431', self.panel.ee.text)
        self.assertEqual(self.panel.selected_id, 'object_1')

    def test_cards_use_geometry_not_semantic_name(self):
        bundle = self.bind()
        cylinder = object_('object_2', PrimitiveShape.CYLINDER)
        cylinder.class_name = 'bottle'
        bundle.world_model.register(cylinder)
        self.c._refresh()
        self.assertTrue(any(str(w.args[0]).endswith('cylinder.png') for w in self.ui.find('Image')))
        self.assertTrue(self.ui.find('Label', text='bottle'))

    def test_dropdown_and_card_selection_stay_synchronized(self):
        bundle = self.bind()
        bundle.world_model.register(object_('object_2', PrimitiveShape.SPHERE))
        self.c._refresh()
        self.panel.combo.model.get_item_value_model().set_value(1)
        self.assertEqual(self.c.view_model.selected_id, 'object_2')
        self.ui.find('Button', text='object_1')[0].click()
        self.assertEqual(self.c.view_model.selected_id, 'object_1')
        self.assertEqual(self.panel.combo.model.get_item_value_model().as_int, 0)

    def test_geometry_change_refreshes_cards_but_preserves_inputs_and_scroll(self):
        bundle = self.bind()
        self.panel.target_fields[0].model.set_value(.63)
        self.panel.object_scroll.scroll_y = 150.
        old = self.panel.object_frame.rebuilds
        obj = bundle.world_model.require('object_1')
        obj.geometry = replace(obj.geometry, yaw=.7)
        self.c._refresh()
        self.assertGreater(self.panel.object_frame.rebuilds, old)
        self.assertEqual(self.panel.object_scroll.scroll_y, 150.)
        self.assertEqual(self.c._target()[0], .63)

    def test_no_rebuild_when_objects_unchanged(self):
        self.bind()
        before = (self.panel.object_frame.rebuilds, self.panel.selection_frame.rebuilds)
        self.c._refresh()
        self.assertEqual(before, (self.panel.object_frame.rebuilds, self.panel.selection_frame.rebuilds))

    def test_held_and_lost_badges_are_independent(self):
        bundle = self.hold()
        bundle.world_model.require('object_1').visible = False
        self.c._refresh()
        self.assertTrue(self.ui.find('Label', text='HELD'))
        self.assertTrue(self.ui.find('Label', text='LOST'))
        self.assertFalse(self.panel.buttons['pick'].enabled)
        self.assertTrue(self.panel.buttons['clear_held'].enabled)

    def test_confirm_does_not_clear_until_explicit_action(self):
        bundle = self.hold()
        self.panel.buttons['clear_held'].click()
        self.assertEqual(bundle.world_model.held_object_id, 'object_1')
        self.assertIsNotNone(self.panel.dialog)
        self.assertFalse(self.panel.buttons['home'].enabled)
        self.assertFalse(self.panel.buttons['clear_lost'].enabled)
        self.assertFalse(self.c._can_run())

    def test_cancel_and_window_close_preserve_state(self):
        bundle = self.hold()
        for cancel in (self.c._cancel_clear_held, lambda: setattr(self.panel.dialog, 'visible', False)):
            self.c._clear_held_clicked()
            cancel()
            self.assertEqual(bundle.world_model.held_object_id, 'object_1')
            self.assertIsNone(self.panel.dialog)
            self.assertIsNone(self.c._pending_clear)
            self.assertTrue(self.panel.buttons['clear_held'].enabled)

    def test_confirm_clears_attachment_and_hint_without_motion_or_open(self):
        bundle = self.hold()
        bundle.backend.reset_mock()
        self.c._clear_held_clicked()
        self.c._confirm_clear_held()
        self.assertIsNone(bundle.world_model.held_object_id)
        self.assertIsNone(bundle.world_model.held_object_offset_in_ee)
        self.assertEqual(bundle.world_model.association_hints(), ())
        self.assertTrue(bundle.world_model.exists('object_1'))
        self.assertTrue(self.panel.buttons['pick'].enabled)
        self.assertFalse(self.panel.buttons['place'].enabled)
        self.assertIn('None', self.panel.held.text)
        self.assertTrue(all(call[0] == 'get_end_effector_pose' for call in bundle.backend.method_calls))
        bundle.runtime.execute.assert_not_called()
        bundle.runtime.execute_async.assert_not_called()
        bundle.world.step.assert_not_called()

    def test_debug_reset_is_available_with_paused_uninitialized_articulation(self):
        bundle = self.hold()
        bundle.backend.robot.handles_initialized = False
        self.c._refresh()
        self.assertTrue(self.panel.buttons['clear_held'].enabled)
        self.c._clear_held_clicked(); self.c._confirm_clear_held()
        self.assertIsNone(bundle.world_model.held_object_id)
        self.assertFalse(self.panel.buttons['pick'].enabled)

    def test_busy_reset_refused_even_through_direct_callback(self):
        bundle = self.hold()
        self.c.busy = True
        self.c._clear_held_clicked()
        self.assertIsNone(self.c._pending_clear)
        self.assertEqual(bundle.world_model.held_object_id, 'object_1')

    def test_confirmation_rechecks_busy_and_object_identity(self):
        for change in ('busy', 'object', 'runtime'):
            with self.subTest(change=change):
                self.c.busy = False
                bundle = self.hold(bundle_())
                self.c._clear_held_clicked()
                if change == 'busy':
                    self.c.busy = True
                elif change == 'object':
                    bundle.world_model.register(object_('different'))
                    bundle.world_model.set_held('different')
                else:
                    self.c.bundle = bundle_()
                current = bundle.world_model.held_object_id
                self.c._confirm_clear_held()
                self.assertEqual(bundle.world_model.held_object_id, current)
                self.assertIn('Nothing was cleared', self.panel.status.text)

    def test_reset_without_runtime_or_held_object_is_noop(self):
        self.c._clear_held_clicked()
        self.assertIsNone(self.c._pending_clear)
        self.bind()
        self.c._clear_held_clicked(); self.c._confirm_clear_held()
        self.assertIsNone(self.c._pending_clear)
        self.assertIn('No held object', self.panel.status.text)

    def test_clear_lost_uses_updater_and_ground_truth_is_protected(self):
        for provider in ('perception', 'ground_truth'):
            bundle = self.bind(bundle_(provider))
            self.c._clear_lost_clicked()
            self.assertEqual(bundle.world_updater.clear_lost.call_count, int(provider == 'perception'))

    def test_capture_control_kept_and_uses_provider(self):
        bundle = self.bind()
        with patch.object(self.module, 'save_capture', return_value=ROOT / 'outputs/perception/test.npz') as save:
            self.panel.buttons['capture'].click()
            self.assertIs(save.call_args.args[0], bundle.state_provider)
            self.assertIn('test.npz', self.panel.status.text)

    def test_profile_picker_sets_existing_profile(self):
        self.panel.buttons['profile'].click()
        self.assertIsNotNone(self.panel.profile_dialog)
        self.ui.find('Button', text='config/scenes/playground.toml')[-1].click()
        self.assertIsNone(self.panel.profile_dialog)
        self.assertEqual(self.panel.profile.model.get_value_as_string(), 'config/scenes/playground.toml')

    def test_details_button_opens_actual_row_snapshot(self):
        self.bind()
        row = self.panel.rows[0]
        self.panel._show_details(row)
        self.assertIsNotNone(self.panel.detail_dialog)
        self.assertTrue(self.ui.find('Label', text='Object ID: object_1'))
        self.assertTrue(self.ui.find('Label', text='Source: rgbd_geometry_perception'))

    def test_bad_target_is_rejected_before_scheduling(self):
        bundle = self.bind()
        self.panel.target_fields[0].model.set_value(float('nan'))
        self.c._place_clicked()
        self.c._move_clicked()
        self.c._run_task(False)
        bundle.runtime.execute_async.assert_not_called()
        self.assertFalse(self.c.busy)
        self.assertFalse(self.c._tasks)

    def test_async_initialize_preserved(self):
        bundle = bundle_()
        async def run():
            with patch.object(self.module, 'build_runtime', AsyncMock(return_value=bundle)) as build:
                self.c._initialize_clicked()
                self.assertTrue(self.c.busy)
                await asyncio.gather(*self.c._tasks)
                build.assert_awaited_once()
            self.assertIs(self.c.bundle, bundle)
            self.assertFalse(self.c.busy)
            self.assertEqual(self.panel.status.text, 'Runtime ready')
        asyncio.run(run())

    def test_async_skill_path_and_double_click_guard(self):
        bundle = self.bind()
        async def run():
            self.c._home_clicked(); self.c._home_clicked()
            self.assertEqual(len(self.c._tasks), 1)
            await asyncio.gather(*self.c._tasks)
        asyncio.run(run())
        bundle.runtime.execute_async.assert_awaited_once_with('home')
        bundle.runtime.execute.assert_not_called()
        bundle.world.step.assert_not_called()
        self.assertFalse(self.c.busy)

    def test_all_motion_buttons_dispatch_async_with_expected_arguments(self):
        bundle = self.bind()
        async def run():
            for callback in (self.c._pick_clicked, self.c._place_clicked, self.c._move_clicked, self.c._home_clicked):
                callback()
                await asyncio.gather(*self.c._tasks)
        asyncio.run(run())
        names = [call.args[0] for call in bundle.runtime.execute_async.await_args_list]
        self.assertEqual(names, ['pick', 'place', 'move_to_pose', 'home'])
        self.assertEqual(bundle.runtime.execute_async.await_args_list[0].kwargs, {'object_id': 'object_1'})
        np.testing.assert_allclose(bundle.runtime.execute_async.await_args_list[1].kwargs['target'].position, [.45,.25,.025])
        bundle.runtime.execute.assert_not_called()

    def test_both_behavior_trees_remain_cooperative(self):
        bundle = self.bind()
        async def run():
            for recovery in (False, True):
                self.c._run_task(recovery)
                await asyncio.gather(*self.c._tasks)
                self.assertIn('SUCCESS', self.panel.status.text)
        asyncio.run(run())
        self.assertEqual([call.args[0] for call in bundle.runtime.execute_async.await_args_list],
                         ['pick', 'place', 'home', 'pick', 'place', 'home'])
        bundle.runtime.execute.assert_not_called()
        bundle.world.step.assert_not_called()

    def test_recovery_tree_retries_failed_pick_via_async_only(self):
        bundle = self.bind()
        failure = SkillResult(SkillStatus.FAILED, 'Pick failed', FailureCode.NO_VALID_GRASP)
        success = SkillResult(SkillStatus.SUCCESS, 'Done')
        bundle.runtime.execute_async.side_effect = [failure, success, success, success, success]
        async def run():
            self.c._run_task(True)
            await asyncio.gather(*self.c._tasks)
        asyncio.run(run())
        self.assertEqual([call.args[0] for call in bundle.runtime.execute_async.await_args_list],
                         ['pick', 'home', 'pick', 'place', 'home'])
        self.assertIn('SUCCESS', self.panel.status.text)

    def test_failed_result_is_displayed_and_reenables_controls(self):
        bundle = self.bind()
        bundle.runtime.execute_async.return_value = SkillResult(SkillStatus.FAILED, 'No valid grasp', FailureCode.NO_VALID_GRASP)
        async def run():
            self.c._pick_clicked()
            await asyncio.gather(*self.c._tasks)
        asyncio.run(run())
        self.assertIn('FAILED', self.panel.status.text)
        self.assertIn('No valid grasp', self.panel.status.tooltip)
        self.assertTrue(self.panel.buttons['home'].enabled)

    def test_shutdown_destroys_dialogs_and_cancels_task_safely(self):
        bundle = self.bind()
        async def forever(*args, **kwargs):
            await asyncio.Event().wait()
        bundle.runtime.execute_async.side_effect = forever
        window = self.c.window
        async def run():
            self.c._home_clicked()
            tasks = tuple(self.c._tasks)
            await asyncio.sleep(0)
            self.c.on_shutdown()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.assertTrue(all(task.cancelled() for task in tasks))
        asyncio.run(run())
        self.assertTrue(window.destroyed)
        self.assertIsNone(self.c._update_sub)
        bundle.world_updater.stop.assert_called_once_with(bundle.world)

    def test_ui_source_has_no_synchronous_stepping_or_dependency_rewrites(self):
        for path in (EXT / 'robot_skill_stack_ui' / 'ui').glob('*.py'):
            tree = ast.parse(path.read_text(), feature_version=(3,11))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            for call in calls:
                self.assertFalse(isinstance(call.func, ast.Attribute) and call.func.attr in ('step', 'execute'), path)
            self.assertNotIn('--no-ros-env', path.read_text())
            self.assertNotIn('pip install', path.read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
