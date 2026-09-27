from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / ".deps"):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import py_trees
from omni.kit.async_engine import run_coroutine

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place import (
    create_pick_and_place_tree,
)
from robot_skill_stack.orchestration.behavior_trees.tasks.pick_and_place_recovery import (
    create_pick_and_place_recovery_tree,
)
from robot_skill_stack.presentation import WorldModelViewModel


class RobotSkillStackExtension(omni.ext.IExt):
    REFRESH_PERIOD = 0.2

    def on_startup(self, ext_id):
        self.bundle = None
        self.view_model = None
        self.busy = False
        self._elapsed = 0.0
        self._object_signature = None
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="Robot Skill Stack UI refresh",
            )
        )
        self.window = ui.Window("Robot Skill Runtime", width=470, height=760)
        self._build_ui()

    def on_shutdown(self):
        self._update_sub = None
        if self.bundle is not None:
            try:
                self.bundle.world_updater.stop(self.bundle.world)
            except Exception:
                pass
        self.view_model = None
        self.bundle = None
        self.window = None

    def _build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Robot Skill Stack", height=28, style={"font_size": 20})
                self.status = ui.Label(
                    "Open a compatible scene, choose a profile, then initialize.",
                    word_wrap=True,
                    height=52,
                )

                ui.Label("Scene profile")
                self.profile = ui.StringField()
                self.profile.model.set_value("config/scenes/playground.toml")
                ui.Button(
                    "Initialize Runtime",
                    clicked_fn=self._initialize_clicked,
                    height=32,
                )

                self.runtime_info = ui.Label("Runtime: not initialized", height=22)

                ui.Separator()
                ui.Label("World Model", style={"font_size": 16})
                self.selected_label = ui.Label("Selected: -", height=22)
                self.object_frame = ui.ScrollingFrame(
                    height=250,
                    build_fn=self._build_object_rows,
                )

                ui.Separator()
                ui.Label("Target position")
                self.x = self._float_field("X", 0.45)
                self.y = self._float_field("Y", 0.25)
                self.z = self._float_field("Z", 0.025)

                ui.Separator()
                ui.Label("Skills", style={"font_size": 16})
                with ui.HStack(spacing=5):
                    ui.Button("Pick Selected", clicked_fn=self._pick_clicked, height=30)
                    ui.Button("Place Held", clicked_fn=self._place_clicked, height=30)
                with ui.HStack(spacing=5):
                    ui.Button("Move To", clicked_fn=self._move_clicked, height=30)
                    ui.Button("Home", clicked_fn=self._home_clicked, height=30)

                ui.Separator()
                ui.Label("Tasks", style={"font_size": 16})
                ui.Button(
                    "Pick & Place Selected",
                    clicked_fn=lambda: self._run_task(recovery=False),
                    height=32,
                )
                ui.Button(
                    "Pick & Place + Recovery",
                    clicked_fn=lambda: self._run_task(recovery=True),
                    height=32,
                )

                ui.Separator()
                self.held = ui.Label("Held: -")
                self.ee = ui.Label("EE: -", word_wrap=True, height=36)
                ui.Button("Refresh Now", clicked_fn=lambda: self._refresh(force=True), height=28)

    def _build_object_rows(self):
        with ui.VStack(spacing=5):
            if self.view_model is None:
                ui.Label("Initialize the runtime to populate the WorldModel.")
                return

            rows = self.view_model.rows()
            if not rows:
                ui.Label("WorldModel currently contains no objects.")
                return

            selected = self.view_model.selected_id
            for row in rows:
                marker = ">" if row.object_id == selected else " "
                state = "VISIBLE" if row.visible else "LOST"
                held = " | HELD" if row.held else ""
                ui.Button(
                    f"{marker} {row.object_id}",
                    clicked_fn=lambda object_id=row.object_id: self._select_object(object_id),
                    height=26,
                )
                ui.Label(f"class: {row.class_name or 'unknown'} | {state}{held}", height=18)
                ui.Label(f"pos:  {self._fmt_vec(row.position)}", height=18)
                ui.Label(f"size: {self._fmt_vec(row.size)}", height=18)
                ui.Separator()

    @staticmethod
    def _fmt_vec(values):
        if values is None:
            return "[unknown]"
        return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"

    def _float_field(self, label, value):
        with ui.HStack(height=24):
            ui.Label(label, width=20)
            field = ui.FloatField()
            field.model.set_value(value)
        return field

    def _target(self):
        return np.array(
            [
                self.x.model.get_value_as_float(),
                self.y.model.get_value_as_float(),
                self.z.model.get_value_as_float(),
            ],
            dtype=float,
        )

    def _set_status(self, text):
        self.status.text = text

    def _on_update(self, event):
        if self.bundle is None or self.busy:
            return
        try:
            self._elapsed += float(event.payload["dt"])
            if self._elapsed >= self.REFRESH_PERIOD:
                self._elapsed = 0.0
                self._refresh()
        except Exception:
            self._elapsed = 0.0

    def _initialize_clicked(self):
        run_coroutine(self._initialize())

    async def _initialize(self):
        if self.bundle is not None:
            self._set_status("Runtime already initialized.")
            return

        self._set_status("Initializing...")
        await omni.kit.app.get_app().next_update_async()

        try:
            profile = Path(self.profile.model.get_value_as_string())
            if not profile.is_absolute():
                profile = ROOT / profile
            self.bundle = await build_runtime(profile)
            self.view_model = WorldModelViewModel(self.bundle.world_model)
            self.view_model.ensure_selection()
            self._object_signature = None
            self._set_status("READY")
            self._refresh(force=True)
        except Exception as exc:
            self._set_status(f"Initialization failed\n{type(exc).__name__}: {exc}")

    def _select_object(self, object_id):
        if self.view_model is None:
            return
        try:
            self.view_model.select(object_id)
            self._object_signature = None
            self._refresh(force=True)
        except Exception as exc:
            self._set_status(f"Selection failed: {exc}")

    def _selected_object(self, *, require_visible=False):
        if self.view_model is None:
            self._set_status("Initialize the runtime first.")
            return None
        obj = self.view_model.selected_object()
        if obj is None:
            self._set_status("No WorldModel object is available.")
            return None
        if require_visible and not obj.visible:
            self._set_status(
                f"Cannot use '{obj.object_id}': it is not currently visible."
            )
            return None
        return obj

    def _run(self, name, **kwargs):
        run_coroutine(self._execute(name, kwargs))

    async def _execute(self, name, kwargs):
        if not self._can_run():
            return
        self.busy = True
        self._set_status(f"Running skill: {name}...")
        await omni.kit.app.get_app().next_update_async()
        try:
            result = self.bundle.runtime.execute(name, **kwargs)
            prefix = "SUCCESS" if result.ok else "FAILED"
            code = "" if result.failure_code is None else f" [{result.failure_code.value}]"
            self._set_status(f"{prefix}{code}: {result.message}")
        except Exception as exc:
            self._set_status(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.busy = False
            self._refresh(force=True)

    def _run_task(self, recovery: bool):
        obj = self._selected_object(require_visible=not recovery)
        if obj is None:
            return
        run_coroutine(self._execute_task(recovery, obj.object_id))

    async def _execute_task(self, recovery: bool, object_id: str):
        if not self._can_run():
            return
        self.busy = True
        task_name = "Pick & Place + Recovery" if recovery else "Pick & Place"
        self._set_status(f"Running task: {task_name} ({object_id})...")
        await omni.kit.app.get_app().next_update_async()
        try:
            factory = (
                create_pick_and_place_recovery_tree
                if recovery
                else create_pick_and_place_tree
            )
            root = factory(
                self.bundle.runtime,
                object_id=object_id,
                target=Pose(self._target()),
                return_home=True,
            )
            status = BTExecutor(root).run(verbose=False)
            prefix = "SUCCESS" if status == py_trees.common.Status.SUCCESS else "FAILED"
            self._set_status(f"{prefix}: {task_name}")
        except Exception as exc:
            self._set_status(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.busy = False
            self._refresh(force=True)

    def _can_run(self):
        if self.bundle is None:
            self._set_status("Initialize the runtime first.")
            return False
        if self.busy:
            self._set_status("Another skill/task is running.")
            return False
        return True

    def _pick_clicked(self):
        obj = self._selected_object(require_visible=True)
        if obj is not None:
            self._run("pick", object_id=obj.object_id)

    def _place_clicked(self):
        self._run("place", target=Pose(self._target()), mode="stable")

    def _move_clicked(self):
        if self.bundle is None:
            self._set_status("Initialize the runtime first.")
            return
        orientation = self.bundle.backend.get_end_effector_pose().orientation
        self._run("move_to_pose", target=Pose(self._target(), orientation))

    def _home_clicked(self):
        self._run("home")

    def _refresh(self, force=False):
        if self.bundle is None or self.view_model is None:
            return

        self.view_model.ensure_selection()
        signature = self.view_model.signature()
        if force or signature != self._object_signature:
            self._object_signature = signature
            self.object_frame.rebuild()

        provider = self.bundle.config.world.provider
        source = "RGB-D perception" if provider == "perception" else "ground truth"
        rows = self.view_model.rows()
        self.runtime_info.text = (
            f"World source: {source} | Objects: {len(rows)} | NumPy: {np.__version__}"
        )
        self.selected_label.text = f"Selected: {self.view_model.selected_id or '-'}"
        self.held.text = f"Held: {self.bundle.world_model.held_object_id or '-'}"

        pose = self.bundle.backend.get_end_effector_pose()
        self.ee.text = (
            f"EE: [{pose.position[0]:.3f}, "
            f"{pose.position[1]:.3f}, "
            f"{pose.position[2]:.3f}]"
        )
