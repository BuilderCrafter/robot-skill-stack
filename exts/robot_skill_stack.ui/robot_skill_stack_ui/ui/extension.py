from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / '.deps'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import carb
import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import AsyncSkillNode
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place import create_pick_and_place_tree
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place_recovery import create_pick_and_place_recovery_tree
from robot_skill_stack.presentation import WorldModelViewModel
from robot_skill_stack.world.perception.diagnostics import save_capture

from .panel import ControlPanel


class RobotSkillStackExtension(omni.ext.IExt):
    REFRESH_PERIOD = 0.2
    root = ROOT

    def on_startup(self, ext_id):
        self.bundle = self.view_model = None
        self.busy = self._shutting_down = False
        self._tasks = set()
        self._elapsed = 0.0
        self._object_signature = self._pending_clear = None
        self.window = ui.Window('Robot Skill Stack', width=1180, height=920)
        self.panel = ControlPanel(self)
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name='Robot Skill Stack UI refresh')
        self._refresh(force=True)
        carb.log_info(f'[robot_skill_stack.ui] started; NumPy {np.__version__} from {np.__file__}')

    def on_shutdown(self):
        self._shutting_down = True
        self._update_sub = None
        self._pending_clear = None
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()
        if self.bundle is not None:
            try:
                self.bundle.world_updater.stop(self.bundle.world)
            except Exception:
                pass
        self.panel.destroy()
        self.window.destroy()
        self.view_model = self.bundle = self.panel = self.window = None

    def _spawn(self, coroutine, label):
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)

        def done(completed):
            self._tasks.discard(completed)
            if not completed.cancelled() and completed.exception() is not None:
                carb.log_error(f'[robot_skill_stack.ui] {label}: {completed.exception()}')
        task.add_done_callback(done)
        return task

    def _set_status(self, text, level='info'):
        if not self._shutting_down:
            self.panel.set_status(text, level)

    def _robot_ready(self):
        return bool(self.bundle is not None and getattr(self.bundle.backend.robot, 'handles_initialized', False))

    def _on_update(self, event):
        if self._shutting_down:
            return
        try:
            self._elapsed += float(event.payload['dt'])
            if self._elapsed >= self.REFRESH_PERIOD:
                self._elapsed = 0.0
                self._refresh()
        except Exception as exc:
            self._elapsed = 0.0
            self._set_status(f'UI refresh failed: {exc}', 'error')

    def _update_controls(self):
        if self._shutting_down:
            return
        selected = self.view_model.selected_object() if self.view_model is not None else None
        self.panel.set_enabled(
            initialized=self.bundle is not None, ready=self._robot_ready(), busy=self.busy,
            held_id=None if self.bundle is None else self.bundle.world_model.held_object_id,
            selected=selected, perception=self.bundle is not None and self.bundle.config.world.provider == 'perception')

    def _initialize_clicked(self):
        if self.busy or self._pending_clear is not None:
            return
        if self.bundle is not None and self._robot_ready():
            self._set_status('Runtime already initialized.', 'success')
            return
        self.busy = True
        self._update_controls()
        self._spawn(self._initialize(), 'runtime initialization')

    async def _initialize(self):
        self._set_status('Initializing runtime...')
        try:
            profile = Path(self.panel.profile.model.get_value_as_string()).expanduser()
            if not profile.is_absolute():
                profile = ROOT / profile
            if self.bundle is not None:
                self.bundle.world_updater.stop(self.bundle.world)
            self.bundle = await build_runtime(profile)
            self.view_model = WorldModelViewModel(self.bundle.world_model)
            self.view_model.ensure_selection()
            self._object_signature = None
            self._set_status('Runtime ready', 'success')
        except Exception as exc:
            self.bundle = self.view_model = None
            self._set_status(f'Initialization failed: {type(exc).__name__}: {exc}', 'error')
            carb.log_error(f'[robot_skill_stack.ui] initialization: {exc}')
        finally:
            self.busy = False
            self._refresh(force=True)

    def _clear_lost_clicked(self):
        if self.bundle is None or self.bundle.config.world.provider != 'perception':
            self._set_status('Clear Lost is available for perception objects only.', 'warning')
            return
        if self.busy or self._pending_clear is not None:
            self._set_status('Wait for the current operation or confirmation to finish.', 'warning')
            return
        try:
            removed = self.bundle.world_updater.clear_lost()
            self._set_status(f'Cleared {len(removed)} lost object(s).', 'success')
            self._refresh(force=True)
        except Exception as exc:
            self._set_status(f'Clear Lost failed: {exc}', 'error')

    def _clear_held_clicked(self):
        if self.busy:
            self._set_status('Wait for the current operation to finish.', 'warning')
            return
        if self.bundle is None:
            self._set_status('Initialize the runtime first.', 'warning')
            return
        object_id = self.bundle.world_model.held_object_id
        if object_id is None:
            self._set_status('No held object to clear.')
            return
        self._pending_clear = (self.bundle, object_id)
        self.panel.show_clear_confirmation(object_id)
        self._update_controls()

    def _cancel_clear_held(self):
        self._pending_clear = None
        self.panel.hide_dialog('dialog')
        self._update_controls()

    def _confirm_clear_held(self):
        pending = self._pending_clear
        if pending is None:
            return
        bundle, object_id = pending
        self._cancel_clear_held()
        if self.busy or self.bundle is not bundle or bundle.world_model.held_object_id != object_id:
            self._set_status('Held state or runtime changed. Nothing was cleared.', 'warning')
            return
        bundle.world_model.clear_held_state()
        message = f'Cleared held state for {object_id}. No robot motion was commanded.'
        carb.log_warn('[robot_skill_stack.ui] manual reset: ' + message)
        self._set_status(message, 'success')
        self._refresh(force=True)

    def _capture_perception_clicked(self):
        if self.bundle is None or self.bundle.config.world.provider != 'perception':
            self._set_status('Capture is available for perception only.', 'warning')
            return
        try:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            path = save_capture(self.bundle.state_provider, ROOT / 'outputs' / 'perception' / f'capture_{stamp}.npz')
            self._set_status(f'Saved {path.name} in outputs/perception.', 'success')
            carb.log_info(f'[robot_skill_stack.ui] perception capture: {path}')
        except Exception as exc:
            self._set_status(f'Capture failed: {exc}', 'error')

    def _select_object(self, object_id):
        if self.view_model is not None:
            try:
                self.view_model.select(object_id)
                self._refresh(force=True)
            except Exception as exc:
                self._set_status(f'Selection failed: {exc}', 'warning')

    def _selected_object(self, *, require_visible=False):
        obj = None if self.view_model is None else self.view_model.selected_object()
        if obj is None:
            self._set_status('No WorldModel object is available. Initialize the runtime first.', 'warning')
        elif require_visible and not obj.visible:
            self._set_status(f"Cannot use '{obj.object_id}': object is not visible.", 'warning')
            return None
        return obj

    def _can_run(self):
        if self.bundle is None:
            message = 'Initialize the runtime first.'
        elif self.busy or self._pending_clear is not None:
            message = 'Another operation or confirmation is in progress.'
        elif not self._robot_ready():
            message = 'Robot articulation is not initialized. Press Initialize after Play/Reset.'
        else:
            return True
        self._set_status(message, 'warning')
        return False

    def _target(self):
        values = np.array([f.model.get_value_as_float() for f in self.panel.target_fields])
        if not np.isfinite(values).all():
            raise ValueError('Target XYZ must contain finite values in meters.')
        return values

    def _run(self, name, **kwargs):
        if self._can_run():
            # Reserve immediately, before scheduling: two clicks must not launch two skills.
            self.busy = True
            self._update_controls()
            self._spawn(self._execute(name, kwargs), f'skill {name}')

    async def _execute(self, name, kwargs):
        self._set_status(f'Running {name}...')
        try:
            result = await self.bundle.runtime.execute_async(name, **kwargs)
            code = '' if result.failure_code is None else f' [{result.failure_code.value}]'
            self._set_status(f'{"SUCCESS" if result.ok else "FAILED"}{code}: {result.message}',
                             'success' if result.ok else 'error')
        except Exception as exc:
            self._set_status(f'ERROR: {type(exc).__name__}: {exc}', 'error')
        finally:
            self.busy = False
            self._refresh(force=True)

    def _run_task(self, recovery):
        obj = self._selected_object(require_visible=not recovery)
        if obj is None or not self._can_run():
            return
        try:
            target = Pose(self._target())
        except ValueError as exc:
            self._set_status(str(exc), 'warning')
            return
        self.busy = True
        self._update_controls()
        self._spawn(self._execute_task(recovery, obj.object_id, target), 'behavior tree')

    async def _execute_task(self, recovery, object_id, target):
        name = 'Pick & Place + Recovery' if recovery else 'Pick & Place'
        self._set_status(f'Running {name}...')
        try:
            factory = create_pick_and_place_recovery_tree if recovery else create_pick_and_place_tree
            root = factory(self.bundle.runtime, object_id=object_id, target=target,
                           return_home=True, skill_node_cls=AsyncSkillNode)
            status = await BTExecutor(root).run_async(omni.kit.app.get_app().next_update_async, verbose=False)
            ok = status == py_trees.common.Status.SUCCESS
            self._set_status(f'{"SUCCESS" if ok else "FAILED"}: {name}', 'success' if ok else 'error')
        except Exception as exc:
            self._set_status(f'ERROR: {type(exc).__name__}: {exc}', 'error')
        finally:
            self.busy = False
            self._refresh(force=True)

    def _pick_clicked(self):
        obj = self._selected_object(require_visible=True)
        if obj is not None:
            self._run('pick', object_id=obj.object_id)

    def _place_clicked(self):
        try:
            self._run('place', target=Pose(self._target()), mode='stable')
        except ValueError as exc:
            self._set_status(str(exc), 'warning')

    def _move_clicked(self):
        if self._can_run():
            try:
                orientation = self.bundle.backend.get_end_effector_pose().orientation
                self._run('move_to_pose', target=Pose(self._target(), orientation))
            except Exception as exc:
                self._set_status(f'Could not prepare movement: {exc}', 'error')

    def _home_clicked(self):
        self._run('home')

    def _refresh(self, force=False):
        if self._shutting_down:
            return
        if self.bundle is None or self.view_model is None:
            self.panel.update_objects((), None, rebuild=force)
            self.panel.runtime_info.text = 'Not initialized'
            self.panel.held.text = 'Held object:   None'
            self.panel.ee.text = 'EE position:   unavailable'
            self._update_controls()
            return
        self.bundle.world_updater.cleanup()
        self.view_model.ensure_selection()
        signature = self.view_model.signature()
        self.panel.update_objects(self.view_model.rows(), self.view_model.selected_id,
                                  rebuild=force or signature != self._object_signature)
        self._object_signature = signature
        self.panel.runtime_info.text = 'Perception' if self.bundle.config.world.provider == 'perception' else 'Ground truth'
        self.panel.runtime_info.tooltip = f'NumPy {np.__version__} from {np.__file__}'
        self.panel.held.text = f'Held object:   {self.bundle.world_model.held_object_id or "None"}'
        self.panel.ee.text = 'EE position:   unavailable'
        if self._robot_ready():
            try:
                p = self.bundle.backend.get_end_effector_pose().position
                self.panel.ee.text = 'EE position:   [ ' + ', '.join(f'{v:.3f}' for v in p) + ' ] m'
            except Exception:
                pass
        self._update_controls()
