from __future__ import annotations

from datetime import datetime
from functools import partial

import omni.ui as ui

from .style import AMBER, DIM, GREEN, RED, STYLE
from .widgets import badge, button, image, padded, section, vector


class ControlPanel:
    """Native OmniUI view. All robot operations are delegated to the controller."""

    CARD_HEIGHT, CARD_SPACING, EMPTY_HEIGHT = 128, 10, 94

    def __init__(self, controller):
        self.c = controller
        self.buttons = {}
        self.rows, self.selected_id = (), None
        self.combo = None
        self.dialog = self.profile_dialog = self.detail_dialog = None
        self._selection_signature = None
        controller.window.frame.style = STYLE
        with controller.window.frame:
            with ui.ZStack():
                ui.Rectangle(name='background')
                with padded(10):
                    with ui.HStack(spacing=14, name='two_column_layout'):
                        with ui.ScrollingFrame(width=ui.Fraction(0.42), name='controls_sidebar',
                                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                            self._build_controls()
                        with ui.ZStack(width=ui.Fraction(0.58), name='world_model_panel'):
                            ui.Rectangle(name='panel')
                            with padded(10):
                                with ui.VStack(spacing=8):
                                    with ui.HStack(height=42, spacing=12):
                                        image('cube_outline', 26)
                                        ui.Label('World Model', name='world_title')
                                        self.count = ui.Label('0 objects', name='muted', width=86,
                                                             alignment=ui.Alignment.RIGHT_CENTER)
                                    ui.Separator(height=1)
                                    self.object_scroll = ui.ScrollingFrame(
                                        height=ui.Fraction(1), name='world_model_scroll',
                                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED)
                                    with self.object_scroll:
                                        self.object_frame = ui.Frame(
                                            height=self._cards_height(), name='world_model_cards',
                                            build_fn=self._build_cards)
        self.set_status('Open the scene and initialize.', 'idle')

    def _action(self, key, text, callback, **kwargs):
        result = button(text, callback, **kwargs)
        self.buttons[key] = result
        return result

    def _build_controls(self):
        c = self.c
        with ui.VStack(height=0, spacing=8):
            with ui.HStack(height=52, spacing=6):
                ui.Label('Robot Skill Stack', name='title')
                with ui.VStack(width=155, spacing=3):
                    ui.Spacer(height=6)
                    ui.Label('Robot Control Panel', name='muted', height=16, alignment=ui.Alignment.RIGHT_CENTER)
                    self.runtime_info = ui.Label('Not initialized', name='dim', height=14,
                                                 alignment=ui.Alignment.RIGHT_CENTER)
                    ui.Spacer(height=7)
            with section('Scene / Runtime', 'database'):
                with ui.HStack(height=28, spacing=7):
                    ui.Label('Scene Profile', name='muted', width=84)
                    self.profile = ui.StringField(tooltip='TOML profile path, relative to the repository or absolute.')
                    self.profile.model.set_value('config/scenes/playground.toml')
                    self._action('profile', '', self._show_profiles, icon='folder', width=30)
                self._action('initialize', 'Initialize Runtime', c._initialize_clicked, icon='play', name='primary')
            with section('Selected Object', 'cube_outline'):
                with ui.HStack(height=28, spacing=7):
                    ui.Label('Object ID', name='muted', width=84)
                    self.selection_frame = ui.Frame(build_fn=self._build_selection)
                with ui.ZStack(height=48):
                    ui.Rectangle(name='inset')
                    with padded(6):
                        with ui.VStack(spacing=3):
                            self.held = ui.Label('Held object:   None', name='muted', height=16)
                            self.ee = ui.Label('EE position:   unavailable', name='muted', height=16)
            with section('Target XYZ', 'crosshair'):
                with ui.HStack(height=48, spacing=10):
                    self.target_fields = []
                    for axis, value in zip('XYZ', (.45, .25, .025)):
                        with ui.VStack(spacing=4):
                            ui.Label(f'{axis} (m)', name='muted', height=16)
                            field = ui.FloatField(height=28, tooltip=f'Target {axis} in world-frame meters.')
                            field.model.set_value(value)
                            self.target_fields.append(field)
                self._action('move_to', 'Move To', c._move_clicked, icon='move', name='move',
                             tooltip='Move the end effector to Target XYZ, preserving its orientation.')
            with section('Skills', 'gear'):
                with ui.HStack(height=30, spacing=9):
                    self._action('pick', 'Pick', c._pick_clicked, name='primary')
                    self._action('place', 'Place', c._place_clicked,
                                 tooltip='Target XYZ is the placed object center, not the wrist position.')
                    self._action('move', 'Move', c._move_clicked)
                    self._action('home', 'Home', c._home_clicked)
            with section('Tasks', 'list'):
                with ui.HStack(height=30, spacing=9):
                    self._action('task', 'Pick & Place', partial(c._run_task, False))
                    self._action('recovery', 'Recovery', partial(c._run_task, True))
            with section('Debug / Maintenance', 'wrench'):
                with ui.HStack(height=30, spacing=7):
                    self._action('clear_lost', 'Clear Lost', c._clear_lost_clicked,
                                 tooltip='Forget unprotected lost perception objects and their tracker records.')
                    self._action('clear_held', 'Clear Held State', c._clear_held_clicked,
                                 width=150, tooltip='After a dropped grasp: clear software state only. No robot motion.')
                    self._action('capture', 'Capture', c._capture_perception_clicked, width=76,
                                 tooltip='Save an RGB-D diagnostic capture in outputs/perception.')
            with ui.ZStack(height=94):
                ui.Rectangle(name='panel')
                with padded():
                    with ui.VStack(spacing=5):
                        with ui.HStack(height=16, spacing=10):
                            self.status_dot = ui.Circle(width=10, height=10, style={'background_color': DIM})
                            ui.Label('STATUS', name='section')
                            self.status_time = ui.Label('', name='dim', width=64,
                                                        alignment=ui.Alignment.RIGHT_CENTER)
                        self.status = ui.Label('', height=32, word_wrap=True)
                        self.status_detail = ui.Label('No active operation', name='dim', height=16)

    def _build_selection(self):
        self._ids = tuple(row.object_id for row in self.rows)
        labels = self._ids or ('No objects',)
        index = self._ids.index(self.selected_id) if self.selected_id in self._ids else 0
        self.combo = ui.ComboBox(index, *labels, height=28, enabled=bool(self._ids))
        self.combo.model.add_item_changed_fn(self._selection_changed)

    def _selection_changed(self, model, item):
        index = model.get_item_value_model().as_int
        if 0 <= index < len(self._ids) and self._ids[index] != self.selected_id:
            self.c._select_object(self._ids[index])

    def _cards_height(self):
        count = len(self.rows)
        return self.EMPTY_HEIGHT if not count else count * self.CARD_HEIGHT + (count - 1) * self.CARD_SPACING

    def _build_cards(self):
        with ui.VStack(height=self._cards_height(), spacing=self.CARD_SPACING):
            if not self.rows:
                with padded(18):
                    ui.Label('No objects yet. Initialize the runtime, then let perception observe the workspace.',
                             name='muted', word_wrap=True, height=58)
            for row in self.rows:
                self._card(row)

    def _card(self, row):
        choose = partial(self.c._select_object, row.object_id)
        with ui.ZStack(height=self.CARD_HEIGHT, name=f'object_card_{row.object_id}'):
            ui.Rectangle(name='card', selected=row.object_id == self.selected_id)
            with padded(10):
                with ui.HStack(spacing=14):
                    with ui.VStack(width=84):
                        ui.Spacer()
                        thumb = row.primitive if row.primitive in ('cube', 'sphere', 'cylinder') else 'unknown'
                        tile = image(thumb, 84, tooltip='Primitive symbol; not a camera image. Click to select.')
                        tile.set_mouse_released_fn(lambda x, y, b, m: choose() if b == 0 else None)
                        ui.Spacer()
                    with ui.VStack(spacing=4):
                        with ui.HStack(height=25, spacing=8):
                            button(row.object_id, choose, name='object_name', width=0, height=24)
                            ui.Label('|', name='dim', width=5)
                            ui.Label(row.class_name or 'unknown', name='muted', width=0)
                            badge('VISIBLE' if row.visible else 'LOST')
                            if row.held:
                                badge('HELD')
                            ui.Spacer()
                            button('', partial(self._show_details, row), icon='dots', name='icon', width=22, height=24)
                        for title, value in (('Position', row.position), ('Size', row.size)):
                            with ui.HStack(height=18, spacing=8):
                                ui.Label(title, name='muted', width=60)
                                ui.Label(vector(value) + (' m' if value is not None else ''), name='detail')
                        ui.Label(row.primitive_detail or 'No additional geometry', name='detail',
                                 height=28, word_wrap=True, tooltip=row.primitive_detail or '')

    def update_objects(self, rows, selected_id, *, rebuild=False):
        self.rows, self.selected_id = rows, selected_id
        selection = (tuple(row.object_id for row in rows), selected_id)
        if selection != self._selection_signature:
            self._selection_signature = selection
            self.selection_frame.rebuild()
        if rebuild:
            # Lazy Frames build only when visible: reserve content height BEFORE rebuilding.
            self.object_frame.height = self._cards_height()
            self.object_frame.rebuild()
            if not rows:
                self.object_scroll.scroll_y = 0.0
        self.count.text = f'{len(rows)} object' + ('' if len(rows) == 1 else 's')

    def set_status(self, message, level='info'):
        tint = {'success': GREEN, 'error': RED, 'warning': AMBER, 'idle': DIM}.get(level, DIM)
        self.status.text = message
        self.status.tooltip = message
        self.status.style = {'color': tint, 'font_size': 13}
        self.status_dot.style = {'background_color': tint}
        self.status_time.text = datetime.now().strftime('%H:%M:%S')

    def set_enabled(self, *, initialized, ready, busy, held_id, selected, perception):
        idle = not busy and self.dialog is None
        motion = initialized and ready and idle
        values = {'initialize': idle, 'profile': idle and not (initialized and ready),
                  'pick': motion and not held_id and selected is not None and selected.visible,
                  'place': motion and bool(held_id), 'move': motion, 'move_to': motion, 'home': motion,
                  'task': motion and not held_id and selected is not None and selected.visible,
                  'recovery': motion and not held_id and selected is not None,
                  'clear_lost': initialized and perception and idle,
                  'clear_held': initialized and bool(held_id) and idle,
                  'capture': initialized and perception and not busy}
        for key, widget in self.buttons.items():
            widget.enabled = bool(values.get(key, idle))
        self.profile.enabled = values['profile']
        for field in self.target_fields:
            field.enabled = idle
        self.status_detail.text = 'Operation in progress' if busy else (
            'Waiting for confirmation' if self.dialog else 'No active operation')

    def show_clear_confirmation(self, object_id):
        self.hide_dialog('dialog')
        self.dialog = ui.Window('Clear Held State', width=470, height=220)
        self.dialog.frame.style = STYLE
        self.dialog.set_visibility_changed_fn(lambda visible: None if visible else self.c._cancel_clear_held())
        with self.dialog.frame:
            with padded(16):
                with ui.VStack(spacing=12):
                    ui.Label(f'Clear software attachment: {object_id}?', height=24)
                    ui.Label('Only confirm after visually checking that the gripper is empty.\n'
                             'This clears the held ID, attachment offset and its association hint.\n'
                             'It does NOT open the gripper or move the robot.',
                             name='muted', word_wrap=True, height=82)
                    with ui.HStack(height=32, spacing=10):
                        button('Cancel', self.c._cancel_clear_held)
                        button('Clear Held State', self.c._confirm_clear_held, name='primary')

    def _show_profiles(self):
        if self.c.busy or self.dialog is not None:
            return
        self.hide_dialog('profile_dialog')
        self.profile_dialog = ui.Window('Scene Profiles', width=500, height=240)
        self.profile_dialog.frame.style = STYLE
        with self.profile_dialog.frame:
            with padded(12):
                with ui.VStack(spacing=8):
                    ui.Label('Select a profile, or type a path in Scene Profile.', name='muted', height=22)
                    with ui.ScrollingFrame():
                        with ui.VStack(height=0, spacing=6):
                            for path in sorted((self.c.root / 'config' / 'scenes').glob('*.toml')):
                                value = path.relative_to(self.c.root).as_posix()
                                button(value, partial(self._choose_profile, value))

    def _choose_profile(self, value):
        if not self.c.busy and not self.c._robot_ready():
            self.profile.model.set_value(value)
        self.hide_dialog('profile_dialog')

    def _show_details(self, row):
        self.hide_dialog('detail_dialog')
        self.detail_dialog = ui.Window(f'Object Details: {row.object_id}', width=510, height=280)
        self.detail_dialog.frame.style = STYLE
        details = [f'Object ID: {row.object_id}', f'Class: {row.class_name or "unknown"}',
                   f'Primitive: {row.primitive or "unknown"}', f'Position: {vector(row.position)} m',
                   f'Size: {vector(row.size)} m', row.primitive_detail or 'Geometry: unavailable',
                   f'Source: {row.source or "unknown"}',
                   f'Confidence: {"unknown" if row.confidence is None else f"{row.confidence:.3f}"}',
                   'Snapshot at ' + datetime.now().strftime('%H:%M:%S')]
        with self.detail_dialog.frame:
            with padded(14):
                with ui.ScrollingFrame():
                    with ui.VStack(height=0, spacing=7):
                        for text in details:
                            ui.Label(text, name='muted', height=20, word_wrap=True)

    def hide_dialog(self, name):
        window = getattr(self, name)
        setattr(self, name, None)
        if window is not None:
            window.set_visibility_changed_fn(lambda visible: None)
            window.destroy()

    def destroy(self):
        for name in ('dialog', 'profile_dialog', 'detail_dialog'):
            self.hide_dialog(name)
