from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / ".deps"):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

import carb
import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import py_trees

from robot_skill_stack.common.types import Pose
from robot_skill_stack.integrations.isaac.bootstrap import build_runtime
from robot_skill_stack.orchestration.behavior_trees.executor import BTExecutor
from robot_skill_stack.orchestration.behavior_trees.nodes.skill_node import (
    AsyncSkillNode,
)
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
        self._tasks = set()
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

        self.window = ui.Window(
            "Robot Skill Runtime",
            width=430,
            height=600,
        )
        self._build_ui()
        self.runtime_info.text = (
            f"Not initialized | NumPy {np.__version__}"
        )
        carb.log_info(
            "[robot_skill_stack.ui] started; "
            f"NumPy {np.__version__} from {np.__file__}"
        )

    def on_shutdown(self):
        self._update_sub = None
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()

        if self.bundle is not None:
            try:
                self.bundle.world_updater.stop(self.bundle.world)
            except Exception:
                pass

        self.view_model = None
        self.bundle = None
        self.window = None

    def _spawn(self, coroutine, label):
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)

        def done(completed):
            self._tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                exc = completed.exception()
            except BaseException:
                return
            if exc is not None:
                carb.log_error(
                    f"[robot_skill_stack.ui] {label} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        task.add_done_callback(done)
        return task

    def _build_ui(self):
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=5):
                    with ui.HStack(height=24):
                        ui.Label(
                            "Robot Skill Stack",
                            style={"font_size": 18},
                        )
                        self.runtime_info = ui.Label(
                            "Not initialized",
                            alignment=ui.Alignment.RIGHT_CENTER,
                            width=210,
                        )

                    self.status = ui.Label(
                        "Open the scene and initialize.",
                        word_wrap=True,
                        height=34,
                    )

                    with ui.HStack(height=28, spacing=5):
                        self.profile = ui.StringField()
                        self.profile.model.set_value(
                            "config/scenes/playground.toml"
                        )
                        ui.Button(
                            "Initialize",
                            clicked_fn=self._initialize_clicked,
                            width=95,
                        )

                    ui.Separator()
                    with ui.HStack(height=22):
                        ui.Label(
                            "World Model",
                            style={"font_size": 15},
                        )
                        self.selected_label = ui.Label(
                            "Selected: -",
                            alignment=ui.Alignment.RIGHT_CENTER,
                            width=180,
                        )

                    self.object_frame = ui.ScrollingFrame(
                        height=155,
                        build_fn=self._build_object_rows,
                    )
                    ui.Button("Clear Lost", clicked_fn=self._clear_lost_clicked, height=24)

                    ui.Separator()
                    with ui.HStack(height=26, spacing=5):
                        ui.Label("Target", width=46)
                        self.x = self._inline_float("X", 0.45)
                        self.y = self._inline_float("Y", 0.25)
                        self.z = self._inline_float("Z", 0.025)

                    with ui.HStack(height=28, spacing=5):
                        ui.Button(
                            "Pick",
                            clicked_fn=self._pick_clicked,
                        )
                        ui.Button(
                            "Place",
                            clicked_fn=self._place_clicked,
                        )
                        ui.Button(
                            "Move",
                            clicked_fn=self._move_clicked,
                        )
                        ui.Button(
                            "Home",
                            clicked_fn=self._home_clicked,
                        )

                    with ui.HStack(height=28, spacing=5):
                        ui.Button(
                            "Pick & Place",
                            clicked_fn=lambda: self._run_task(False),
                        )
                        ui.Button(
                            "+ Recovery",
                            clicked_fn=lambda: self._run_task(True),
                        )

                    ui.Separator()
                    with ui.HStack(height=22):
                        self.held = ui.Label("Held: -")
                        self.ee = ui.Label(
                            "EE: -",
                            alignment=ui.Alignment.RIGHT_CENTER,
                            width=230,
                        )

    def _build_object_rows(self):
        with ui.VStack(spacing=3):
            if self.view_model is None:
                ui.Label("Initialize to populate the WorldModel.")
                return

            rows = self.view_model.rows()
            if not rows:
                ui.Label("No objects.")
                return

            selected = self.view_model.selected_id
            for row in rows:
                state = "VISIBLE" if row.visible else "LOST"
                held = " | HELD" if row.held else ""
                prefix = ">" if row.object_id == selected else " "
                ui.Button(
                    f"{prefix} {row.object_id} | "
                    f"{row.class_name or 'unknown'} | {state}{held}",
                    clicked_fn=lambda object_id=row.object_id: (
                        self._select_object(object_id)
                    ),
                    height=23,
                )
                ui.Label(
                    f"p {self._fmt_vec(row.position)}   s {self._fmt_vec(row.size)}", height=17)
                obj = self.bundle.world_model.get(row.object_id) if self.bundle else None
                if obj is not None and obj.geometry is not None:
                    g=obj.geometry; detail=g.shape.value
                    if g.yaw is not None: detail += f" yaw {np.degrees(g.yaw):.0f}°"
                    elif g.shape.value == "sphere" and g.radius is not None: detail += f" r {g.radius:.3f}"
                    elif g.shape.value == "cylinder": detail += f" axis {self._fmt_vec(g.axis)} r {g.radius:.3f} L {g.length:.3f}"
                    ui.Label(detail, height=17)

    @staticmethod
    def _fmt_vec(values):
        if values is None:
            return "[?]"
        return "[" + " ".join(
            f"{value:.3f}" for value in values
        ) + "]"

    @staticmethod
    def _inline_float(label, value):
        with ui.HStack(spacing=2):
            ui.Label(label, width=13)
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

    def _robot_ready(self):
        return bool(
            self.bundle is not None
            and getattr(
                self.bundle.backend.robot,
                "handles_initialized",
                False,
            )
        )

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
        if self.busy:
            return
        if self.bundle is not None and self._robot_ready():
            self._set_status("Runtime already initialized.")
            return
        self._spawn(self._initialize(), "runtime initialization")

    async def _initialize(self):
        self.busy = True
        self._set_status("Initializing runtime...")
        carb.log_info("[robot_skill_stack.ui] initialization started")

        try:
            profile = Path(
                self.profile.model.get_value_as_string()
            )
            if not profile.is_absolute():
                profile = ROOT / profile

            if self.bundle is not None:
                try:
                    self.bundle.world_updater.stop(
                        self.bundle.world
                    )
                except Exception:
                    pass

            self.bundle = await build_runtime(profile)
            self.view_model = WorldModelViewModel(
                self.bundle.world_model
            )
            self.view_model.ensure_selection()
            self._object_signature = None
            self._set_status("READY")
            carb.log_info(
                "[robot_skill_stack.ui] runtime READY"
            )
        except Exception as exc:
            self.bundle = None
            self.view_model = None
            message = (
                f"Initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._set_status(message)
            carb.log_error(
                "[robot_skill_stack.ui] " + message
            )
        finally:
            self.busy = False
            self._refresh(force=True)

    def _clear_lost_clicked(self):
        if self.bundle is None or self.bundle.config.world.provider != "perception":
            self._set_status("Clear Lost is available for perception objects only."); return
        removed=self.bundle.world_model.clear_lost(getattr(self.bundle.state_provider,"forget",None))
        self._set_status(f"Cleared {len(removed)} lost object(s).")
        self._object_signature=None; self._refresh(force=True)

    def _select_object(self, object_id):
        if self.view_model is None:
            return
        try:
            self.view_model.select(object_id)
            self._object_signature = None
            self._refresh(force=True)
        except Exception as exc:
            self._set_status(
                f"Selection failed: {exc}"
            )

    def _selected_object(self, *, require_visible=False):
        if self.view_model is None:
            self._set_status(
                "Initialize the runtime first."
            )
            return None

        obj = self.view_model.selected_object()
        if obj is None:
            self._set_status(
                "No WorldModel object is available."
            )
            return None

        if require_visible and not obj.visible:
            self._set_status(
                f"Cannot use '{obj.object_id}': "
                "object is not visible."
            )
            return None
        return obj

    def _can_run(self):
        if self.bundle is None:
            self._set_status(
                "Initialize the runtime first."
            )
            return False
        if self.busy:
            self._set_status(
                "Another operation is running."
            )
            return False
        if not self._robot_ready():
            self._set_status(
                "Robot articulation is not initialized. "
                "Press Initialize again after Play/Reset."
            )
            return False
        return True

    def _run(self, name, **kwargs):
        if not self._can_run():
            return
        self._spawn(
            self._execute(name, kwargs),
            f"skill {name}",
        )

    async def _execute(self, name, kwargs):
        self.busy = True
        self._set_status(f"Running {name}...")
        try:
            result = await self.bundle.runtime.execute_async(
                name,
                **kwargs,
            )
            prefix = "SUCCESS" if result.ok else "FAILED"
            code = (
                ""
                if result.failure_code is None
                else f" [{result.failure_code.value}]"
            )
            self._set_status(
                f"{prefix}{code}: {result.message}"
            )
        except Exception as exc:
            self._set_status(
                f"ERROR: {type(exc).__name__}: {exc}"
            )
        finally:
            self.busy = False
            self._refresh(force=True)

    def _run_task(self, recovery):
        obj = self._selected_object(
            require_visible=not recovery
        )
        if obj is None or not self._can_run():
            return
        self._spawn(
            self._execute_task(recovery, obj.object_id),
            "behavior tree",
        )

    async def _execute_task(self, recovery, object_id):
        self.busy = True
        name = (
            "Pick & Place + Recovery"
            if recovery
            else "Pick & Place"
        )
        self._set_status(f"Running {name}...")

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
                skill_node_cls=AsyncSkillNode,
            )
            app = omni.kit.app.get_app()
            status = await BTExecutor(root).run_async(
                app.next_update_async,
                verbose=False,
            )
            prefix = (
                "SUCCESS"
                if status == py_trees.common.Status.SUCCESS
                else "FAILED"
            )
            self._set_status(f"{prefix}: {name}")
        except Exception as exc:
            self._set_status(
                f"ERROR: {type(exc).__name__}: {exc}"
            )
        finally:
            self.busy = False
            self._refresh(force=True)

    def _pick_clicked(self):
        obj = self._selected_object(
            require_visible=True
        )
        if obj is not None:
            self._run(
                "pick",
                object_id=obj.object_id,
            )

    def _place_clicked(self):
        self._run(
            "place",
            target=Pose(self._target()),
            mode="stable",
        )

    def _move_clicked(self):
        if not self._can_run():
            return
        try:
            orientation = (
                self.bundle.backend
                .get_end_effector_pose()
                .orientation
            )
        except Exception as exc:
            self._set_status(
                f"Could not read EE pose: {exc}"
            )
            return
        self._run(
            "move_to_pose",
            target=Pose(
                self._target(),
                orientation,
            ),
        )

    def _home_clicked(self):
        self._run("home")

    def _refresh(self, force=False):
        if self.bundle is None or self.view_model is None:
            self.runtime_info.text = (
                f"Not initialized | NumPy {np.__version__}"
            )
            self.selected_label.text = "Selected: -"
            self.held.text = "Held: -"
            self.ee.text = "EE: -"
            return

        self.view_model.ensure_selection()
        signature = self.view_model.signature()
        if force or signature != self._object_signature:
            self._object_signature = signature
            self.object_frame.rebuild()

        provider = self.bundle.config.world.provider
        source = (
            "Perception"
            if provider == "perception"
            else "Ground truth"
        )
        rows = self.view_model.rows()
        self.runtime_info.text = (
            f"{source} | {len(rows)} objects"
        )
        self.selected_label.text = (
            f"Selected: "
            f"{self.view_model.selected_id or '-'}"
        )
        self.held.text = (
            f"Held: "
            f"{self.bundle.world_model.held_object_id or '-'}"
        )

        if not self._robot_ready():
            self.ee.text = "EE: unavailable"
            return

        try:
            pose = self.bundle.backend.get_end_effector_pose()
            self.ee.text = (
                f"EE [{pose.position[0]:.3f} "
                f"{pose.position[1]:.3f} "
                f"{pose.position[2]:.3f}]"
            )
        except Exception:
            self.ee.text = "EE: unavailable"
