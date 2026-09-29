"""Contract regressions for native 5.1 callback retention and async UI teardown."""
import asyncio
import builtins
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch, AsyncMock

from robot_skill_stack.integrations.isaac.runtime_lifecycle import PhysicsCallbackScope, construct_world_deferred
from robot_skill_stack.world.model.updater import WorldModelUpdater
from robot_skill_stack.world.model.world_model import WorldModel
from tests.support.fake_omni_ui import extension_environment
from tests.unit.test_control_panel_ui import bundle_


class StopPlayWorld:
    """Exact relevant SDK behavior: Stop drops subscriptions, not saved callables."""
    def __init__(self):
        self._physics_functions, self._physics_callback_functions = {}, {}
        self.errors = []
    def physics_callback_exists(self, name):
        return name in self._physics_callback_functions
    def add_physics_callback(self, name, callback_fn):
        if self.physics_callback_exists(name):
            self.errors.append('already exists')
            return
        self._physics_functions[name] = self._physics_callback_functions[name] = callback_fn
    def remove_physics_callback(self, name):
        if not self.physics_callback_exists(name):
            self.errors.append("doesn't exist")
            return
        del self._physics_functions[name]
        del self._physics_callback_functions[name]
    def stop(self):
        self._physics_callback_functions.clear()
    def play(self):
        self._physics_callback_functions.update(self._physics_functions)
    def tick(self):
        for fn in list(self._physics_callback_functions.values()): fn(.2)


def provider():
    return SimpleNamespace(observe=Mock(return_value=[]), close=Mock())


class CallbackLifecycleTests(unittest.TestCase):
    def test_stop_before_play_erases_retained_callable(self):
        w=StopPlayWorld(); p=provider(); u=WorldModelUpdater(WorldModel(),p)
        u.start(PhysicsCallbackScope(w)); w.stop(); u.stop(w); w.play()
        self.assertEqual(w._physics_functions,{})
        self.assertEqual(w._physics_callback_functions,{})
        self.assertEqual(w.errors,[]); p.close.assert_called_once()

    def test_reinit_after_stop_uses_only_new_provider(self):
        w=StopPlayWorld(); a,b=provider(),provider()
        x,y=WorldModelUpdater(WorldModel(),a),WorldModelUpdater(WorldModel(),b)
        x.start(PhysicsCallbackScope(w)); w.stop(); x.stop(w); w.play()
        y.start(PhysicsCallbackScope(w)); w.tick()
        a.observe.assert_not_called(); b.observe.assert_called_once()
        self.assertEqual(w.errors,[]); y.stop(w)

    def test_foreign_callback_survives_cleanup(self):
        w=StopPlayWorld(); foreign=Mock(); w.add_physics_callback('other_extension',foreign)
        u=WorldModelUpdater(WorldModel(),provider()); u.start(PhysicsCallbackScope(w))
        w.stop(); u.stop(w); w.play(); w.tick()
        foreign.assert_called_once(); self.assertEqual(set(w._physics_functions),{'other_extension'})

    def test_scope_rejects_unknown_retained_owner(self):
        w=StopPlayWorld(); w.add_physics_callback('world_model_updater',Mock()); w.stop()
        with self.assertRaisesRegex(RuntimeError,'already exists'):
            WorldModelUpdater(WorldModel(),provider()).start(PhysicsCallbackScope(w))

    def test_scope_does_not_remove_replaced_callable(self):
        w=StopPlayWorld(); scope=PhysicsCallbackScope(w); fn=Mock()
        scope.add_physics_callback('a',fn); w._physics_functions['a']=Mock()
        with self.assertRaisesRegex(RuntimeError,'foreign'): scope.remove_physics_callback('a')

    def test_stop_idempotent_and_late_invocation_inert(self):
        w=StopPlayWorld(); p=provider(); u=WorldModelUpdater(WorldModel(),p)
        u.start(PhysicsCallbackScope(w)); callback=w._physics_functions['world_model_updater']
        u.stop(w); u.stop(w); callback(.2); u.tick(.2); u.update(); u.cleanup()
        p.observe.assert_not_called(); p.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'closed'): u.start(PhysicsCallbackScope(w))

    def test_callback_exception_reported_then_can_recover(self):
        w=StopPlayWorld(); p=provider(); p.observe.side_effect=ValueError('camera bad')
        u=WorldModelUpdater(WorldModel(),p); u.start(PhysicsCallbackScope(w)); w.tick()
        self.assertEqual(u.last_error,'ValueError: camera bad')
        p.observe.side_effect=None; w.tick(); self.assertIsNone(u.last_error); self.assertEqual(u.update_count,1)
        u.stop(w)

    def test_start_failure_never_claims_a_callback(self):
        w=StopPlayWorld(); w.add_physics_callback=Mock(side_effect=ValueError('register failed'))
        u=WorldModelUpdater(WorldModel(),provider())
        with self.assertRaises(ValueError): u.start(PhysicsCallbackScope(w))
        self.assertIsNone(u._callback_name); u.stop(w)

    def test_twenty_switches_leave_one_updater(self):
        w=StopPlayWorld(); prev=None
        for i in range(20):
            if prev: w.stop(); prev.stop(w)
            w.play(); prev=WorldModelUpdater(WorldModel(),provider()); prev.start(PhysicsCallbackScope(w))
            w.tick(); self.assertEqual(len(w._physics_callback_functions),1)
        prev.stop(w); self.assertEqual(w.errors,[]); self.assertEqual(w._physics_functions,{})


class ConstructorTests(unittest.TestCase):
    def test_deferred_constructor_restores_flag_before_await(self):
        async def scenario():
            with patch.object(builtins,'ISAAC_LAUNCHED_FROM_TERMINAL',False,create=True):
                def ctor(**kw):
                    if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL: raise RuntimeError('nested update')
                    return kw
                self.assertEqual(construct_world_deferred(ctor,stage_units_in_meters=1),{'stage_units_in_meters':1})
                self.assertFalse(builtins.ISAAC_LAUNCHED_FROM_TERMINAL)
                await asyncio.sleep(0)
                self.assertFalse(builtins.ISAAC_LAUNCHED_FROM_TERMINAL)
        asyncio.run(scenario())

    def test_constructor_exception_restores_flag(self):
        with patch.object(builtins,'ISAAC_LAUNCHED_FROM_TERMINAL',False,create=True):
            with self.assertRaises(ValueError): construct_world_deferred(Mock(side_effect=ValueError()))
            self.assertFalse(builtins.ISAAC_LAUNCHED_FROM_TERMINAL)

    def test_missing_flag_is_not_left_installed(self):
        old=getattr(builtins,'ISAAC_LAUNCHED_FROM_TERMINAL',None); exists=hasattr(builtins,'ISAAC_LAUNCHED_FROM_TERMINAL')
        if exists: del builtins.ISAAC_LAUNCHED_FROM_TERMINAL
        try:
            construct_world_deferred(dict)
            self.assertFalse(hasattr(builtins,'ISAAC_LAUNCHED_FROM_TERMINAL'))
        finally:
            if exists: builtins.ISAAC_LAUNCHED_FROM_TERMINAL=old


class DeferredWindowTests(unittest.TestCase):
    def test_close_profile_is_hidden_then_destroyed_after_update(self):
        async def scenario():
            with extension_environment() as (ui,c,module):
                c.panel._show_profiles(); win=c.panel.profile_dialog
                ui.in_event=True
                c.panel._choose_profile('config/scenes/playground_v2.toml')
                ui.in_event=False
                self.assertFalse(win.visible); self.assertFalse(win.destroyed)
                self.assertIsNone(c.panel.profile_dialog)
                for _ in range(4): await asyncio.sleep(0)
                self.assertTrue(win.destroyed)
                c.on_shutdown()
                for _ in range(4): await asyncio.sleep(0)
        asyncio.run(scenario())

    def test_dialog_replacement_does_not_destroy_the_new_window(self):
        async def scenario():
            with extension_environment() as (_,c,module):
                c.panel._show_profiles(); old=c.panel.profile_dialog
                c.panel._show_profiles(); new=c.panel.profile_dialog
                for _ in range(4): await asyncio.sleep(0)
                self.assertTrue(old.destroyed); self.assertFalse(new.destroyed)
                c.on_shutdown()
                for _ in range(4): await asyncio.sleep(0)
        asyncio.run(scenario())

    def test_shutdown_defers_root_and_dialog(self):
        async def scenario():
            with extension_environment() as (ui,c,module):
                c.panel._show_profiles(); windows=[c.window,c.panel.profile_dialog]
                ui.in_event=True; c.on_shutdown(); ui.in_event=False
                self.assertTrue(all(not x.visible and not x.destroyed for x in windows))
                for _ in range(4): await asyncio.sleep(0)
                self.assertTrue(all(x.destroyed for x in windows))
        asyncio.run(scenario())

    def test_initialization_clears_old_bundle_while_waiting(self):
        async def scenario():
            with extension_environment() as (_,c,module):
                old=bundle_(); c.bundle=old
                async def build(path):
                    self.assertIsNone(c.bundle); self.assertIsNone(c.view_model)
                    await asyncio.sleep(0); return bundle_()
                with patch.object(module,'build_runtime',side_effect=build):
                    await c._initialize()
                old.world_updater.stop.assert_called_once_with(old.world)
                self.assertIsNot(c.bundle,old)
                c.on_shutdown()
                for _ in range(4): await asyncio.sleep(0)
        asyncio.run(scenario())


if __name__=='__main__': unittest.main(verbosity=2)
