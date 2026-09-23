from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

for path in (ROOT / ".deps", ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import py_trees

from behavior_trees.executor import BTExecutor
from behavior_trees.tasks.pick_and_place import create_pick_and_place_tree
from behavior_trees.tasks.pick_and_place_recovery import (
    create_pick_and_place_recovery_tree,
)
from core.types import Pose
from runtime.runtime_builder import build_runtime


class RobotSkillStackExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self.bundle = None
        self.busy = False
        self.window = ui.Window("Robot Skill Runtime", width=380, height=560)
        self._build_ui()

    def on_shutdown(self):
        self.bundle = None
        self.window = None

    def _build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Robot Skill Stack", height=28, style={"font_size": 20})

                self.status = ui.Label(
                    "Open a compatible scene, choose a profile, then initialize.",
                    word_wrap=True,
                    height=50,
                )

                ui.Label("Scene profile")
                self.profile = ui.StringField()
                self.profile.model.set_value("config/scenes/playground.toml")

                self.init_btn = ui.Button(
                    "Initialize Runtime",
                    clicked_fn=self._initialize_clicked,
                    height=32,
                )

                ui.Separator()
                ui.Label("Object")

                self.object_field = ui.StringField()
                self.object_field.model.set_value("cube")

                ui.Label("Target position")
                self.x = self._float_field("X", 0.45)
                self.y = self._float_field("Y", 0.25)
                self.z = self._float_field("Z", 0.025)

                ui.Separator()
                ui.Label("Skills", style={"font_size": 16})

                with ui.HStack(spacing=5):
                    ui.Button("Pick", clicked_fn=self._pick_clicked, height=30)
                    ui.Button("Place", clicked_fn=self._place_clicked, height=30)

                with ui.HStack(spacing=5):
                    ui.Button("Move To", clicked_fn=self._move_clicked, height=30)
                    ui.Button("Home", clicked_fn=self._home_clicked, height=30)

                ui.Separator()
                ui.Label("Tasks", style={"font_size": 16})

                ui.Button(
                    "Pick & Place",
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
                self.ee = ui.Label("EE: -", word_wrap=True, height=40)

                ui.Button(
                    "Refresh Status",
                    clicked_fn=self._refresh_clicked,
                    height=28,
                )

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

    def _object_id(self):
        return self.object_field.model.get_value_as_string().strip()

    def _set_status(self, text):
        self.status.text = text

    def _initialize_clicked(self):
        asyncio.ensure_future(self._initialize())

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

            objects = list(self.bundle.config.objects)
            if objects:
                self.object_field.model.set_value(objects[0])

            self._set_status(f"READY\nObjects: {', '.join(objects)}")
            self._refresh()

        except Exception as exc:
            self._set_status(
                f"Initialization failed:\n{type(exc).__name__}: {exc}"
            )

    def _run(self, name, **kwargs):
        asyncio.ensure_future(self._execute(name, kwargs))

    async def _execute(self, name, kwargs):
        if not self._can_run():
            return

        self.busy = True
        self._set_status(f"Running skill: {name}...")
        await omni.kit.app.get_app().next_update_async()

        try:
            result = self.bundle.runtime.execute(name, **kwargs)
            prefix = "SUCCESS" if result.ok else "FAILED"
            self._set_status(f"{prefix}: {result.message}")
            self._refresh()
        except Exception as exc:
            self._set_status(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.busy = False

    def _run_task(self, recovery: bool):
        asyncio.ensure_future(self._execute_task(recovery))

    async def _execute_task(self, recovery: bool):
        if not self._can_run():
            return

        self.busy = True
        task_name = "Pick & Place + Recovery" if recovery else "Pick & Place"
        self._set_status(f"Running task: {task_name}...")
        await omni.kit.app.get_app().next_update_async()

        try:
            object_id = self._object_id()
            target = Pose(self._target())

            factory = (
                create_pick_and_place_recovery_tree
                if recovery
                else create_pick_and_place_tree
            )

            root = factory(
                self.bundle.runtime,
                object_id=object_id,
                target=target,
                return_home=True,
            )

            status = BTExecutor(root).run(verbose=False)

            if status == py_trees.common.Status.SUCCESS:
                self._set_status(f"SUCCESS: {task_name}")
            else:
                self._set_status(f"FAILED: {task_name}")

            self._refresh()

        except Exception as exc:
            self._set_status(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.busy = False

    def _can_run(self):
        if self.bundle is None:
            self._set_status("Initialize the runtime first.")
            return False

        if self.busy:
            self._set_status("Another skill/task is running.")
            return False

        return True

    def _pick_clicked(self):
        self._run("pick", object_id=self._object_id())

    def _place_clicked(self):
        self._run("place", target=Pose(self._target()), mode="stable")

    def _move_clicked(self):
        if self.bundle is None:
            self._set_status("Initialize the runtime first.")
            return

        orientation = self.bundle.backend.get_end_effector_pose().orientation
        self._run(
            "move_to_pose",
            target=Pose(self._target(), orientation),
        )

    def _home_clicked(self):
        self._run("home")

    def _refresh_clicked(self):
        if self.bundle is None:
            self._set_status("Initialize the runtime first.")
            return
        self._refresh()

    def _refresh(self):
        self.bundle.world_model.refresh_all()

        held = self.bundle.world_model.held_object_id
        pose = self.bundle.backend.get_end_effector_pose()

        self.held.text = f"Held: {held or '-'}"
        self.ee.text = (
            f"EE: [{pose.position[0]:.3f}, "
            f"{pose.position[1]:.3f}, "
            f"{pose.position[2]:.3f}]"
        )